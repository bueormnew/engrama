# ENGRAMA V5.5 — Diseño, pilares y resultados

> V5.5 = V5 estabilizado + **Recall Tap asimétrico** (eje léxico aislado + eje de
> sentido contextual). Recuperación exacta en texto real desambiguando palabras
> polisémicas, manteniendo intacta la filosofía inviolable de ENGRAMA y siendo
> **lineal en entrenamiento e inferencia**.

## 0. Filosofía inviolable (NO se toca)

1. **Huella aislada** `T0[j] = f(x_j)`: depende solo del token `j`. Cero mezcla
   temporal en Fase 1.
2. **Traza FIFO explícita, sin compresión**: almacena `T0` pristino.
3. **Cero `QK^T` `N×N`, cero softmax sobre el eje temporal**.
4. **Consolidación por offsets causales fijos** `D_l = {0, 1, 2^{l-1}, 2^l}`.
5. **Invarianza causal exacta**: forward paralelo == generación incremental.

V5.5 cumple todo esto y suma los 6 pilares de abajo.

## 1. Diagnóstico forense de V5 — por qué no llega al 100 % en texto real

V5 resolvió lo que mató a V4 (superposición aditiva ~10⁻⁴). Quedan 3 cuellos:

- **a) Simetría léxica**: `q = P_q(T0[i])` y `K = P_k(T0[j])` son los dos
  aislados. En KV sintético funciona (tokens iguales → mismo `K`). En texto real
  *«banco»* (de río / de dinero) tiene el mismo `K` en dos sentidos: **no hay
  desambiguación**.
- **b) Recall duro sin desempate contextual**: el argmax top-1 elige la
  ocurrencia más reciente del mismo token, sin mirar el contexto.
- **c) LSH con constante alta en CPU**: 181 candidatos con gathers aleatorios.

V5.5 ataca **a)** con un segundo eje (sentido) y **b)** con una selección
lexicográfica que desempata por contexto; **c)** se hereda de V5.1.

## 2. Los 6 pilares

### Pilar 1 — Isolated Encoder V2 (estable)

`x → RMSNorm → mezcla C×C factorizada → SwiGLU (zero-init) → RMSNorm`.

- RMSNorm en todas las entradas (preserva signo para la ruta de identidad).
- **Zero-identity init**: `beta=1`, `s=0`, bases `U,V` pequeñas; la celula
  SwiGLU tiene su down-proyección a cero → el encoder es identidad en el paso 0.
- Cero NaN en fp16 desde el paso 0. Sigue siendo `B×N×d` totalmente paralelo.

### Pilar 2 — Traza Dual Paginada

Cada posición guarda una tupla **sin comprimir**:

```
Trace[j] = ( T0[j] pristino, T_shallow[j], K_lex[j], K_sense[j], token_id[j], t )
```

`T_shallow` = salida de la capa 0 de consolidación (offsets `{0,1}` → contexto
local de 2 tokens): es la entrada del código de **sentido**.

- **Paginación** estilo PagedAttention: páginas de 256 tokens preasignadas.
  Append `O(1)` (escribir el slot), cero `torch.cat`. Memoria lineal
  (~1.2 KB/token en fp16, ~2.4 KB en fp32; 16k ≈ 20–40 MB).
- **Almacenamiento batched** `(page, B, d)`: soporta inferencia/generación por
  lotes, preservando la invarianza causal.
- **Fast path de identidad** `O(1)`: `{(batch, token_id) → última posición}`.

### Pilar 3 — Consolidation Stack V5.5

Offsets resonantes `D_l` (la mejor pieza de V4) + 3 estabilizaciones:

1. **Mezcla normalizada por conteo** (V5): `T_pos = Σ w_p·y_p / (Σ w_p + ε)` →
   promedio acotado, no suma creciente.
2. **Compuerta dual acotada con RMSNorm**:
   `q_tgt = RMSNorm(Q_tgt)`, `k_src = RMSNorm(K_src)`;
   `α = σ( ⟨q,k⟩/√d_g · scale + qW + kW )` con `scale = C·tanh(b/C)`, `b` init 0.
3. **Residual zero-init**: `T_l = T_pos + tanh(γ)·FFN(RMSNorm(T_pos))`, `γ` init 0.

Sin control flow dinámico → 100 % `torch.compile`.

### Pilar 4 — Recall Tap Asimétrico V2 (el corazón)

El eje léxico se **parte en dos**:

```
K_lex[j]   = P_k_lex(T0[j])            # aislado, induce (mismo token → mismo código)
K_sense[j] = P_k_sense(T_shallow[j])   # contexto local de 2 tokens
q_lex[i]   = P_q_lex(T0[i])            # aislado
q_ctx[i]   = P_q_ctx(RMSNorm(T_L[i]))  # contextual, ve el presente
```

**Selección lexicográfica** (clave de la robustez — ver §3):

```
1. eje LÉXICO domina: solo candidatos con score_lex ≈ máx (los de mismo token,
   que tienen K_lex IDENTICO → score_lex idéntico). El sentido NUNCA sobreescribe
   la señal léxica.
2. entre esos, gana el de mayor score_sense (desambigua "banco").
3. empate de sentido → ocurrencia MÁS RECIENTE.
lectura = T0[j*+1]   (huella completa del siguiente, como V5)
```

Gradiente por **straight-through** (softmax solo en el backward, sobre el
composite `score_lex·(1+β·score_sense)`).

**Recuperación 100 % garantizada**: el eje léxico aislado hace que mismo token
tenga siempre el score máximo → la ocurrencia previa siempre queda en el
conjunto ganador (inducción intacta). En generación incremental la ruta es
**siempre densa** (matvec `O(N·d_k)`), así que el 100 % es estructural.

### Pilar 5 — LSH V2 cuantizado + kernel fusionado

`K_lex` binarizado (`sign` → código de signos); bucket por código determinista
(mismo token → mismo signo → mismo bucket, recall 1.0 para identidad).
Multi-tabla de proyecciones aleatorias = multi-probe LSH (forma eficiente y
lineal del "hamming < umbral"). Candidatos: `1 identidad + 4 recientes +
t·cap LSH + n_neg`. Coste `O(N·C·d_k)` — **lineal**.

Kernel `v55/kernels.py`: fusión `score_lex + score_sense + argmax + gather` en
un solo paso (memoria `O(filas)`). Referencia torch exacta (paridad con
`RecallTapV2.forward_parallel_dense`) + kernel **Triton** para GPU con fallback
automático y `validate_kernel()`. En CPU se usa denso; en GPU LSH.

### Pilar 6 — Evocador con softcap

```
c_bar  = Σ softmax(W_fusion h*) · c_m
logits = softcap( c_bar @ E^T / √d , C=30 )      # C·tanh(·/C)
```

El softcap acota los logits en `[-C, C]`, evitando la explosión que clavaba la
loss en el marginal de V4 a LR 4e-3. Estable en fp16.

## 3. Por qué la selección es lexicográfica (no composite ingenuo)

El spec inicial proponía `score = score_lex·(1+β·score_sense)` con `β≈0.3` y
argmax directo. **Eso rompe la garantía de inducción al inicio**: con
`score_sense` aleatorio (sin entrenar), el término `β·score_sense` puede hacer
que un token *distinto* con `score_lex` cercano a 1 supere al token correcto
(rango del composite para mismo token `[1−β, 1+β]` vs. otro token
`[0.95·(1−β), 0.95·(1+β)]` — se solapan). Resultado medido: 0 % de recuperación.

La **selección lexicográfica** lo arregla y es más fiel a la filosofía:

- el eje léxico **domina** (solo candidatos con `score_lex` ≈ máx, i.e. mismo
  token con `K_lex` idéntico);
- el sentido **solo desempata** entre tokens idénticos (exactamente el caso
  polisémico);
- así `β` puede ser grande (gradiente fuerte al sentido) **sin** romper la
  inducción.

Esto preserva la propiedad de V5 (mismo token siempre gana) y añade
desambiguación contextual donde hace falta. Es la corrección más importante de
V5.5 respecto al borrador.

## 4. Complejidad — entrenamiento E inferencia lineales

| operación | entrenamiento (LSH) | generación (incremental) |
|---|---|---|
| Encoder | `O(N·d²)` | `O(d²)` |
| Consolidación (`L` capas) | `O(N·L·k·d·r)` | `O(L·k·d·r)` (offsets fijos) |
| Recall Tap | `O(N·C·d_k)` candidatos | `O(N·d_k)` matvec |
| Memoria | Traza `O(N·d)` (sin compresión) | igual |

`C = 1 + recent + t·cap + n_neg` ≈ 181 constante. Todo lineal en `N`.
`torch.compile` + `autocast` compatibles, sin control flow dinámico en el núcleo.

## 5. Entrenamiento V5.5

```
Loss = CE_LM + λ(step) · CE_retrieval
```

- **CE_LM**: softcap linear-CE troceado (sin materializar logits).
- **CE_retrieval** (auto-supervisado, Sección 6 del spec): en una fracción de
  posiciones, empuja el tap a apuntar a la posición candidata cuyo **siguiente
  token** coincide con el objetivo (señal de inducción/copia, sin etiquetas
  externas). `λ=0.2`, annealed a 0 tras 50k steps.
- **Currículum** (recomendado): 0–10k denso + `q_lex` solo; 10k–30k LSH `n_neg=0`;
  30k+ LSH + `q_ctx` + `n_neg=48` + `β` entrenable.

> Nota práctica (medida): con `rt_gate_init=1.0` (receta V5 probada, estable por
> el softcap + mezcla normalizada) y CE ponderado, el tap aprende la
> recuperación KV **sin** necesitar `CE_retrieval` — la lectura léxica es
> correcta desde el paso 0 y la loss de LM entrena `W_r`. **`CE_retrieval` es
> INTERNA y obligatoria** para activar el eje de sentido en polisemia: da
> gradiente directo a las proyecciones de sentido (el argmax duro lo bloquea vía
> STE débil). El **peso es la palanca** — `rw=0.3 → 52 %`, **`rw=1.0 (default) →
> 99 %`**, `rw=2.0 → 100 %**` (léxico solo → 41 %, azar 50 %). Versión **densa y
> vectorizada** (matvec BLAS, reutiliza huellas, sin índices LSH), restringida a
> candidatos del mismo token con temperatura aguda.

## 6. Resultados

### R7 — Recuperación KV exacta e invariante a la distancia ✓

Entrenamiento denso puro (`CE_LM`, `λ=0`, `rt_gate_init=1.0`) a **2048** tokens
(900 pasos, ~27 min, LM 4.29 → 0.085); evaluación **sin reentrenar** a 8192 y
16384. Azar 6.2 %:

| contexto | precisión | todas las distancias |
|---|---|---|
| 2048 (entrenado) | **100 %** | 100 % |
| 8192 (4×) | **100 %** | 100 % |
| 16384 (8×) | **100 %** | 100 % |

> **OBJETIVO ≥ 95 % (KV enorme): CUMPLE.** La precisión a 16k es idéntica a 2k:
> el argmax sobre `K_lex` aislado no decae con la distancia. JSON:
> `benchmarks/analysis_lab/results/v55_kv_longcontext.json`.

### R6 — Lineal en memoria y generación ✓

| métrica | medición |
|---|---|
| Pendiente log-log forward LSH (256→16k) | **1.04** (`1.0` = `O(N)`) |
| Generación incremental | ~9–10 ms/token (~105 tok/s), cache paginada |
| Aceleración vs recomputar | ×15 / ×35 / ×80 a ctx 1k / 2k / 4k |
| Memoria de traza dual | **1288 B/token constante** (lineal); 16k ≈ 21 MB |

### Resto de requisitos

| requisito | medición | veredicto |
|---|---|---|
| R3 sin atención | argmax + gather; sin softmax temporal | ✓ |
| R4 sin compresión | traza `T0+T_shallow+K` completa; bytes/token constantes | ✓ |
| R5 paralelización | invarianza causal paralelo == incremental exacta (44 tests V5.5, 162 totales) | ✓ |
| R8 sin NaN | fp16, extremos, LR 10×, traza vacía: finito (tests) | ✓ |
| Polisemia | **99 %** (`rw=1.0` default) / 100 % (`rw=2.0`); 41 % léxico solo (azar 50 %) | ✓ |

## 7. Uso

```python
from engrama import EngraModelV55, V55Config

model = EngraModelV55(V55Config.from_preset("base", vocab_size=50257))
loss  = model.forward_loss(x[:, :-1], x[:, 1:])           # CE_LM + CE_retrieval
ids   = model.generate(prompt_ids, max_new_tokens=200)    # cache nativa paginada
model.save("ckpt"); model2 = EngraModelV55.load("ckpt")
```

Benchmarks: `benchmarks/analysis_lab/v55_kv_longcontext.py`,
`v55_speed_memory.py`, `v55_polyseme.py`.

> **Nota sobre polisemia y el eje de sentido**: `CE_retrieval` es **interna,
> obligatoria, densa y vectorizada** (matvec BLAS, reutiliza huellas, sin índices
> LSH), restringida a candidatos del **mismo token** con temperatura aguda. El
> **peso es la palanca**: `rw=0.3 → 52 %`, **`rw=1.0 (default) → 99 %`**,
> `rw=2.0 → 100 %` (léxico solo → 41 %, azar 50 %). Con LM pura el gradiente al
> sentido a través del argmax duro (STE) es demasiado débil y nada supera el azar.
> La selección lexicográfica garantiza que el sentido **nunca rompe la inducción**:
> el KV al 100 % se obtiene *con el sentido activo*. Escalar la desambiguación a
> polisemia natural de lenguaje es trabajo en curso.
