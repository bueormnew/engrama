# ENGRAMA 🧠⚡
## Arquitectura Neuronal Autorregresiva **sin Atención** — Recuperación Exacta + Recordación Asociativa por Significado

**ENGRAMA** es una arquitectura autorregresiva de memoria **explícita** que **no
usa atención**: cero productos $QK^\top$, cero matrices de afinidad $N\times N$,
cero *softmax* sobre el eje temporal y cero compresión de memoria. Implementada
en **PyTorch puro**.

> **V5.5** — la novedad: sin atención ni compresión, recuperación **exacta a
> cualquier distancia** (copia léxica), desambiguación por **sentido** en texto
> real, y **recordación asociativa por significado** nativa — la arquitectura
> resuelve la *aguja semántica* (lookup asociativo tipo diccionario) **al 100 %**,
> igual que un transformador en tareas semánticas, pero sin atención.

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](#-instalación)
[![PyTorch ≥ 2.0](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-165%20passing-brightgreen)](#-tests-y-verificación)
[![Sin atención](https://img.shields.io/badge/attention-zero-important)](#-filosofía-inviolable)

- **Autor**: Gerson Fabian Buenahora Ormaza (BUEORM) · **Año**: 2026
- **Licencia**: AGPL-3.0 · **Versión**: 0.7.0 (Arquitectura V5.5)

---

## 📑 Tabla de contenidos

- [🎯 Qué es ENGRAMA, en una frase](#-qué-es-engrama-en-una-frase)
- [✅ Lo bueno / ⚠️ Lo malo (honesto)](#-lo-bueno--lo-malo-honesto)
- [🎯 Resultados](#-resultados)
- [🧠 Filosofía inviolable](#-filosofía-inviolable)
- [🧩 Los 6 pilares de V5.5](#-los-6-pilares-de-v55)
- [⚡ El tap semántico (recordación asociativa)](#-el-tap-semántico-recordación-asociativa)
- [🔍 El Recall Tap asimétrico (léxico + semántico)](#-el-recall-tap-asimétrico-léxico--semántico)
- [📊 CE_retrieval: interna, obligatoria, auto-supervisada](#-ce_retrieval-interna-obligatoria-auto-supervisada)
- [⚡ Linealidad: entrenamiento e inferencia](#-linealidad-entrenamiento-e-inferencia)
- [🛡️ Estabilidad numérica (sin NaN)](#️-estabilidad-numérica-sin-nan)
- [🔧 Kernel unificado](#-kernel-unificado)
- [🚀 Quickstart](#-quickstart)
- [🏋️ Receta de entrenamiento](#️-receta-de-entrenamiento)
- [🛠️ Configuración experta](#️-configuración-experta)
- [🧱 Arquitectura módulo a módulo](#-arquitectura-módulo-a-módulo)
- [🧪 Invarianza causal y traza paginada](#-invarianza-causal-y-traza-paginada)
- [📈 Evolución V1 → V5.5](#-evolución-v1--v55)
- [📦 Instalación](#-instalación)
- [📂 Estructura del repositorio](#-estructura-del-repositorio)
- [❓ Preguntas frecuentes](#-preguntas-frecuentes)
- [📄 Cómo citar y licencia](#-cómo-citar-y-licencia)

---

## 🎯 Qué es ENGRAMA, en una frase

Un modelo de lenguaje autorregresivo que **almacena cada huella de memoria de
forma aislada e incorruptible** y la **recupera por búsqueda exacta/semántica**
(argmax + lectura), en vez de mezclar el pasado con atención ($O(N^2)$) o
comprimirlo en un estado fijo (SSM/RNN). El resultado: **recuperación exacta a
cualquier distancia**, lineal en memoria, y ahora también **recordación
asociativa por significado**.

---

## ✅ Lo bueno / ⚠️ Lo malo (honesto)

### ✅ Lo bueno

- **Sin atención, sin compresión.** Recuperación 100 % exacta a cualquier
  distancia (copia léxica vía identidad $O(1)$). Un mismo token ya no colisiona
  (desambiguación por sentido). Y ahora resuelve **agujas semánticas** (lookup
  asociativo) al 100 %.
- **Lineal en el núcleo.** Consolidación $O(N)$; Recall Tap léxico $O(N{\cdot}C)$
  vía LSH + identidad $O(1)$. Inferencia incremental $O(N)$ por token. Vence al
  transformador en memoria y cómputo del núcleo.
- **Estable por construcción.** Zero-init en todos los residuales, RMSNorm antes
  de cada producto punto, *softcap* en el evocador. Sin NaN en fp16 ni a LR alto.
- **Invarianza causal exacta.** `forward_paralelo == generación_incremental`
  bit a bit (error $<10^{-6}$), verificada por tests.
- **Auto-supervisado.** La pérdida `CE_retrieval` enseña a los taps a apuntar al
  sitio correcto **sin etiquetas humanas** (la señal es "el sitio cuyo siguiente
  token es el objetivo").
- **1250 B/token** de memoria, traza FIFO paginada, append $O(1)$.

### ⚠️ Lo malo (limitaciones reales)

- **El tap semántico denso es $O(N^2)$.** Es el camino **exacto** validado al
  100 %. Existe un camino LSH $O(N{\cdot}C)$ pero **aproximado** (~20 % de
  accuracy en la aguja por pérdida de *recall*). La linealidad estricta del tap
  semántico sin perder calidad es **trabajo abierto**. El núcleo (léxico +
  consolidación) sí es lineal.
- **El `CE_retrieval` es denso** $O(\text{frac}\cdot N^2)$. Es la señal de
  entrenamiento de los taps. Linealizarlo sin perder la convergencia es **trabajo
  abierto** (la restricción *same-token* ya lo hace disperso en la práctica).
- **Sin GPU en este entorno.** Los kernels **Triton** están escritos y su
  **referencia torch está validada** (diff $0.0$), pero la **ruta Triton misma
  no se ejecutó** aquí (requiere GPU).
- **No es un LLM preentrenado.** Es una arquitectura validada en tareas
  controladas (KV, polisemia, aguja semántica). Escalar a corpus reales es
  trabajo futuro.
- **Empate por recencia en claves idénticas** (mismo token con claves $T_0$): se
  resuelve con un redondeo `SEM_TIE` para mantener la invarianza ante ruido de
  coma flotante. Es un parche correcto pero específico.

---

## 🎯 Resultados

| Benchmark | Métrica | Resultado | Azar |
|---|---|---|---|
| **Aguja semántica asociativa** (K=8 alias, valor aleatorio) | exactitud | **tap semántico 100 %** · léxico-solo ~15 % | 6.25 % |
| **KV contexto largo** (copia léxica, 64–1024 tokens) | exactitud | **100 %** a toda distancia | ≈1/V |
| **Polisemia** (misma palabra, contextos distintos) | exactitud | **100 %** (rw=1.0, 600 pasos) · léxico ~33–41 % | — |

**La aguja semántica es la prueba reina de V5.5**: la consulta usa un *alias*
`b_i` (token que **nunca apareció**) y debe predecir el valor `v_i` del hecho
`a_i v_i`. Como `v_i` es aleatorio por ejemplo, el modelo **no puede memorizar**
`b_i → v_i`: debe retener el hecho `a_i` y **puentear `b_i ~ a_i` por
significado**. El tap léxico falla (~azar); el semántico, al 100 %.

Resultados en [`benchmarks/results/`](benchmarks/results); scripts reproducibles
en [`benchmarks/analysis_lab/`](benchmarks/analysis_lab).

---

## 🧠 Filosofía inviolable

La mayoría de modelos usan **atención** ($QK^\top$, $O(N^2)$) o **memoria
recurrente comprimida** (SSM/RNN) donde el pasado se diluye en un estado fijo.
**ENGRAMA** sigue una teoría alternativa inspirada en el engrama biológico: *la
experiencia deja una huella aislada e incorruptible, se almacena explícitamente,
y el contexto se consolida con sinapsis causales relativas*.

Cinco reglas **no se tocan** en ninguna versión:

1. **Huella aislada** $T_0[j] = f(x_j)$: depende solo del token $j$. Cero mezcla
   temporal en la Fase 1.
2. **Traza FIFO explícita, sin compresión**: almacena $T_0$ pristino por posición.
3. **Cero $QK^\top$ $N\times N$, cero *softmax* sobre el eje temporal.**
4. **Consolidación por *offsets* causales fijos** $D_l = \{0, 1, 2^{l-1}, 2^l\}$.
5. **Invarianza causal exacta**: `forward_paralelo == generación_incremental`.

El **tap semántico** respeta las cinco: es un *segundo* tap aditivo que recupera
por significado con el **mismo mecanismo** que el léxico — producto punto +
**argmax duro** + **lectura única** de $T_0[j^*{+}1]$ — **no** es atención.

---

## 🧩 Los 6 pilares de V5.5

| # | Pilar | Qué hace |
|---|---|---|
| 1 | **Encoder aislado V2** | `RMSNorm → SwiGLU → RMSNorm`, Zero-Identity. $T_0[j]$ solo del token $j$. |
| 2 | **Traza dual paginada** | $T_0 / T_{\text{shallow}}$ en páginas; append $O(1)$; sin compresión. |
| 3 | **Consolidación count-normalizada** | Mezcla por $D_l = \{0,1,2^{l-1},2^l\}$, residual zero-init. |
| 4 | **Recall Tap asimétrico V2** | $K_{\text{lex}}$ aislado + $K_{\text{sense}}$ desempata + $q_{\text{ctx}}$ + identidad $O(1)$ **y tap semántico**. |
| 5 | **LSH V2 cuantizado** | `sign → 64-bit`, distancia de Hamming, $O(N{\cdot}C)$. |
| 6 | **Evoker con softcap** | `cap·tanh(logits/cap)`, acota los logits, estable en fp16. |

---

## ⚡ El tap semántico (recordación asociativa)

El tap léxico recupera por **token idéntico** (copia exacta). Muchas tareas
requieren recuperar por **significado** — un alias, sinónimo o hecho asociado.
V5.5 añade un segundo tap para esto.

**Diseño** (aditivo, convive con el léxico):

- **Claves/consultas desde la huella aislada $T_0$** (¡Pilar 1!):
  $K_{\text{sem}}[j] = P_k^{\text{sem}}(\text{RMSNorm}(T_0[j]))$,
  $q_{\text{sem}}[i] = P_q^{\text{sem}}(\text{RMSNorm}(T_0[i]))$.
- **Score** = coseno sobre **todos** los candidatos causales $j < i$.
- **Selección** = `argmax` duro + desempate por **recencia**.
- **Lectura** única de $T_0[j^*{+}1]$, inyectada como residual
  `state + g_sem · W_r^sem(lectura)` con **straight-through** en backward.
- **Entrenamiento** vía `CE_retrieval` semántica.

**¿Por qué $T_0$ y no la consolidación $T_{\text{last}}$?** $T_0$ retiene la
**identidad de token**: cada $a_i$ es un token distinto → $K_{\text{sem}}$
distinto → la asociación alias ($b_i \sim a_i$) se vuelve **separable**. La
consolidación lava la identidad (todos los hechos $a_j$ quedaban como
*hard-negatives* idénticos y el argmax elegía el par equivocado). Usar $T_0$ es,
además, **más fiel al aislamiento**.

**No es atención.** Atención = *softmax* sobre $N$ claves (mezcla suave,
$O(N^2)$). El tap semántico = `argmax` duro + lectura **única** de un solo valor
(lookup de diccionario, sin mezcla temporal).

---

## 🔍 El Recall Tap asimétrico (léxico + semántico)

El estado en cada posición es el consolidado **más** las lecturas de dos taps:

$$\text{estado}_i = T_L[i] + g_{\text{rt}}\, W_r(\text{lectura léxica}_i)
                              + g_{\text{sem}}\, W_r^{\text{sem}}(\text{lectura semántica}_i)$$

- **Tap léxico** (copia exacta): empareja por **token idéntico**. El fast path de
  identidad ($O(1)$) garantiza el candidato de inducción; el LSH rescata vecinos.
- **Tap semántico** (asociativo): empareja por **significado aprendido** sobre
  todos los candidatos.

Ambos son **causales**, con `argmax` duro + recencia + lectura única y
**straight-through estimator** (STE) para el gradiente.

---

## 📊 CE_retrieval: interna, obligatoria, auto-supervisada

El `argmax` duro bloquea el gradiente. Para que los taps **aprendan** sin
etiquetas humanas, ENGRAMA usa **`CE_retrieval`** — una pérdida auto-supervisada
**siempre activa** (parte del sistema, no opcional):

- Para cada posición supervisada $i$, puntúa contra las posiciones previas $j$ y
  empuja el score hacia los $j$ cuyo **token siguiente** ($\text{token}_{j+1}$)
  coincide con el objetivo $y_i$.
- **Término léxico**: restringido a candidatos del **mismo token** (no diluye el
  gradiente con copia trivial); el sentido desempata.
- **Término semántico**: sobre **todos** los candidatos causales — entrena
  $q_{\text{sem}}/k_{\text{sem}}$ directamente (la señal que aprende la asociación).

El **peso es la palanca**: `retrieval_weight` (default `1.0`). Sin `CE_retrieval`
el sistema degrada (el léxico solo no resuelve sentido ni asociación).

---

## ⚡ Linealidad: entrenamiento e inferencia

| Componente | Coste por paso | Notas |
|---|---|---|
| Encoder + consolidación | $O(N)$ | sin $N\times N$ |
| Tap léxico (lectura) | $O(N\cdot C)$ | LSH por defecto; identidad $O(1)$ |
| Tap semántico (lectura) | $O(N^2)$ denso · $O(N\cdot C)$ LSH opt-in | `dense` (exacto) \| `"lsh"` |
| `CE_retrieval` | $O(\text{frac}\cdot N\cdot C)$ | léxico same-token (disperso) + semántico |
| Inferencia incremental | $O(N)$ por token | traza paginada, lectura exacta |

- El **núcleo** (consolidación + tap léxico) es **lineal** en $N$.
- El **tap semántico denso** es $O(N^2)$ pero *exacto* (100 %). Como las claves
  $T_0$ son **token-determinadas**, es **deduplicable a $O(N\cdot V)$** exacto
  cuando $N \gg V$. El modo **LSH** lo vuelve $O(N\cdot C)$ siempre, a costa de
  *recall*.
- Objetivo declarado: **batir al transformador** en velocidad y memoria, en
  entrenamiento **e** inferencia, con recuperación exacta sin compresión. El
  núcleo ya lo cumple; la linealidad estricta del tap semántico exacto es trabajo
  abierto.

---

## 🛡️ Estabilidad numérica (sin NaN)

El modelo está diseñado para **no producir NaN**, incluso en fp16 y a LR alto:

- **Zero-init** en todos los residuales (SwiGLU, GatedFFN, mezcla): en el paso 0
  la red es identidad exacta.
- **RMSNorm** antes de cada producto punto de las compuertas (acota el módulo).
- **Softcap** (`cap·tanh(x/cap)`) en el evocador: los logits nunca explotan.
- **`_NEG = -1e30`** enmascara candidatos inválidos: `exp()` nunca desborda.
- **`nan_to_num`** en los pesos del STE como red de seguridad.
- **Sigmoid en fp32** (`_sigmoid_fp32`) y normalización en fp32 para coma flotante
  de 16 bits.

Verificado: secuencias de longitud 2+, tokens todos-iguales, *targets*
todos-ignorados, forward en fp16 y 50 pasos a `lr=1e-2` — **todo finito**.

---

## 🔧 Kernel unificado

[`src/engrama/v55/kernels.py`](src/engrama/v55/kernels.py) fusiona la lectura de
**ambos taps en un solo despachador**:

- `unified_argmax_read(...)` — produce la lectura léxica **y** la semántica en
  una pasada (cubre entrenamiento e inferencia).
- `asymmetric_argmax_read` (léxico) y `semantic_argmax_read` (semántico), cada
  uno con **referencia torch exacta** + **kernel Triton** (GPU).
- Fusión: solo `(best_score, best_j)` en registros — cero escritura de la matriz
  de scores, memoria $O(\text{filas})$ en vez de $O(\text{filas}\cdot N)$.
- **Despacho automático**: Triton en GPU; si no, referencia torch (diff $0.0$ vs
  `forward_semantic_dense`).
- `validate_unified_kernel()` compara kernel vs referencia (**requiere GPU**).

> ⚠️ La ruta Triton requiere GPU para compilar/validar. Aquí solo se validó la
> referencia torch.

---

## 🚀 Quickstart

```python
from engrama.v55 import EngraModelV55

m = EngraModelV55.from_preset("base")          # tap léxico + semántico activos
logits = m(tokens[:, :-1])                     # forward paralelo

# entrenamiento: CE_LM + CE_retrieval (interna, auto-supervisada)
loss = m.forward_loss(tokens[:, :-1], tokens[:, 1:], retrieval_weight=1.0)
loss.backward()

# generación incremental O(N)/token, traza paginada, invariante al forward
out = m.generate(prompt_ids, max_new_tokens=64, temperature=0.8, top_k=40)
```

Presets: `tiny` (depuración), `small`, `base`, `large`. El tap semántico se
activa/desactiva con `semantic_recall_enabled`.

---

## 🏋️ Receta de entrenamiento

```python
import torch
from engrama.v55 import EngraModelV55

model = EngraModelV55.from_preset("base").train()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)

for step in range(num_steps):
    x = next(dataloader)                      # (B, N) long
    loss = model.forward_loss(x[:, :-1], x[:, 1:], retrieval_weight=1.0)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # recomendado
    opt.step()
```

**Recomendaciones**:

- **Optimizador**: AdamW, `betas=(0.9, 0.95)`, `weight_decay=0.01`.
- **LR**: `1e-3` (tiny/small) a `3e-4` (base/large), con *warmup* ~5 % y decaimiento coseno.
- **Grad clip**: `1.0` (el STE del Recall Tap puede dar gradientes punteagudos).
- **`retrieval_weight`**: `1.0` por defecto (sentido + asociación). `0.2` para
  tareas de copia pura donde no hace falta.
- **`rt_train_mode="lsh"`** (default) para entrenamiento lineal del tap léxico.
- Ejemplo completo y ejecutable: [`examples/train_demo.py`](examples/train_demo.py).

---

## 🛠️ Configuración experta

```python
from engrama.v55 import V55Config

cfg = V55Config(
    vocab_size=256, d_model=128, num_consolidation_layers=8,
    context_length=2048, page_size=256,
    rt_train_mode="lsh",            # tap léxico lineal (default)
    semantic_recall_enabled=True,   # tap semántico (default)
    rt_sem_key_source="t0",         # claves semánticas desde T0 (aislado, default)
    rt_sem_query_source="t0",
    rt_sem_recall_mode="dense",     # denso exacto | "lsh" lineal
    retrieval_weight=1.0,           # peso de CE_retrieval (palanca del sentido)
    retrieval_positions_frac=0.25,
)
```

| Parámetro | Default | Para qué |
|---|---|---|
| `rt_train_mode` | `"lsh"` | `"lsh"` (lineal) \| `"dense"` (exacto) para el tap léxico |
| `semantic_recall_enabled` | `True` | activa el tap semántico asociativo |
| `rt_sem_key_source` / `rt_sem_query_source` | `"t0"` | fuente de claves/consultas semánticas (`"t0"` recomendado) |
| `rt_sem_recall_mode` | `"dense"` | `"dense"` (exacto) \| `"lsh"` (lineal) para el tap semántico |
| `retrieval_weight` | `1.0` | peso de la pérdida auto-supervisada CE_retrieval |
| `retrieval_positions_frac` | `0.25` | fracción de posiciones supervisadas por CE_retrieval |
| `retrieval_same_token_only` | `True` | restringe el CE léxico a candidatos mismo-token |
| `logit_cap` | `30.0` | cota del softcap del evocador (0 desactiva) |
| `rt_gap` | `1` | distancia mínima entre la posición actual y el *match* |

---

## 🧱 Arquitectura módulo a módulo

```
tokens → embeddings → IsolatedEncoderV2 → T0 (huella aislada, Pilar 1)
                                       │
                V55ConsolidationStack ←─┘  (Pilar 3: mezcla D_l normalizada)
                    ├── capa 0 → T_shallow (contexto local, eje de sentido)
                    └── capa L → T_L (consolidación final)
                                       │
            RecallTapV2 (Pilar 4) ←────┘
              ├── tap LÉXICO: K_lex(T0) + K_sense(T_shallow) + q_ctx(T_L)
              │     score = cos_lex · (1 + β·cos_sense), argmax duro, lee T0[j*+1]
              └── tap SEMÁNTICO: K_sem(T0) + q_sem(T0)
                    score = cos(q_sem, K_sem), argmax duro, lee T0[j*+1]
                                       │
   estado = T_L + g_rt·W_r(lectura_léx) + g_sem·W_r_sem(lectura_sem)   (inyección)
                                       │
              MultiCandidateEvoker ←───┘  (fusión latente + proyección a vocab)
                                       │
              logits = softcap(·, cap)     (Pilar 6, anti-NaN)
```

Módulos ([`src/engrama/v55/`](src/engrama/v55/)):

- `config.py` — `V55Config` (dataclass + presets + receptivo).
- `encoder.py` — `IsolatedEncoderV2` (RMSNorm + sinapsis + SwiGLU, zero-identity).
- `primitives.py` — `softcap`, `SwiGLU`, `GatedFFN`, `SynapseMixV55`, RMSNorm.
- `trace.py` — `PagedDualTrace` (traza FIFO paginada, $T_0/T_{\text{shallow}}/K$).
- `consolidation.py` — `V55Mix` / `V55Layer` / `V55ConsolidationStack`.
- `recall.py` — `RecallTapV2` (tap léxico + semántico, denso/LSH/incremental).
- `lsh.py` — `LSHIndexV2` (sign → 64-bit, candidatos en $O(N\cdot C)$).
- `kernels.py` — kernel unificado léxico+semántico (Triton + ref torch).
- `losses.py` — `softcap_linear_cross_entropy`, `retrieval_cross_entropy[_dense]`.
- `model.py` — `EngraModelV55` (forward, forward_loss, step_forward, generate, save/load).

---

## 🧪 Invarianza causal y traza paginada

`forward_paralelo == generación_incremental` **bit a bit** (error $<10^{-6}$),
validado en `tests/test_v55_architecture.py` (6 configuraciones). La **traza dual
paginada** (`PagedDualTrace`) almacena $T_0$, $T_{\text{shallow}}$, $K_{\text{lex}}$,
$K_{\text{sense}}$ y $K_{\text{sem}}$ por páginas; append $O(1)$, memoria lineal.

---

## 📈 Evolución V1 → V5.5

| Versión | Idea clave | Métrica clave |
|---|---|---|
| **V1–V3** | Traza FIFO, consolidación por *offsets* causales | recupera secuencias simples |
| **V4** | Recall Tap asimétrico (copia exacta), *softcap* | KV ~100 % corto alcance |
| **V5** | LSH cuantizado + `CE_retrieval` (eje de sentido) | polisemia 99 % |
| **V5.5** | **Tap semántico asociativo** (claves $T_0$) + fixes de invarianza + kernel unificado | **aguja semántica 100 %** |

**Cambios V5 → V5.5**:

1. **Tap semántico** (Pilar 4 extendido): segundo tap que recupera por
   **significado** ($K_{\text{sem}}$ desde $T_0$). Resuelve la aguja semántica.
2. **Claves semánticas desde $T_0$** (no $T_{\text{last}}$): distingue pares,
   más fiel al aislamiento.
3. **Fix de invarianza #1**: la consolidación filtraba *offsets* `p<N` en paralelo
   pero no en incremental → divergencia cuando $\text{max\_offset}\ge N$. Corregido.
4. **Fix de invarianza #2**: empates exactos por mismo token + ruido de FP volcaba
   el argmax de recencia. Corregido con redondeo `SEM_TIE`.
5. **Kernel unificado** (`unified_argmax_read`): ambos taps en un despachador.
6. **`CE_retrieval` semántica**: entrena el tap asociativo sobre todos los candidatos.

---

## 📦 Instalación

```bash
git clone <repo> && cd engrama
python -m venv .venv && source .venv/bin/activate
pip install -e .            # PyTorch ≥ 2.0, numpy
pytest tests/ -q            # 165 tests
python examples/train_demo.py
```

Para los kernels Triton: GPU NVIDIA + `pip install triton`. Sin GPU, todo corre
con la referencia torch exacta.

---

## 📂 Estructura del repositorio

```
src/engrama/
  v55/        # ENGRAMA V5.5 (actual)
  v5/         # V5 (referencia, sin cambios)
  primitives.py encoder.py evoker.py losses.py config.py
tests/        # 165 tests (V4/V5/V5.5)
benchmarks/analysis_lab/   # scripts de benchmark reproducibles
benchmarks/results/        # resultados JSON
examples/train_demo.py     # receta de entrenamiento minimal
docs/         # ENGRAMA-V55-Teorica.md
```

---

## ❓ Preguntas frecuentes

- **¿Es atención el tap semántico?** No. Atención = *softmax* sobre $N$ claves
  (mezcla suave, $O(N^2)$). El tap = `argmax` duro + lectura **única** de un
  valor (lookup de diccionario, sin mezcla temporal, linealizable por LSH).
- **¿Por qué no comprimir la traza?** La filosofía: una huella comprimida se
  degrada con el contexto. Almacenar $T_0$ pristino garantiza recuperación
  **exacta** a cualquier distancia.
- **¿Cómo entrena si el argmax es no diferenciable?** Con un **straight-through
  estimator**: forward = lectura dura del ganador; backward = gradiente suave
  (*softmax* del score). Más la `CE_retrieval` que entrena las proyecciones
  directamente.
- **¿NaN en entrenamiento?** No debería. Zero-init + RMSNorm + softcap +
  `_NEG` + `nan_to_num`. Si aparece, baja `lr` o `logit_cap`.
- **¿Puedo desactivar el tap semántico?** Sí: `semantic_recall_enabled=False`
  (vuelve al comportamiento V5, solo léxico+sentido).

---

## 📄 Cómo citar y licencia

Gerson Fabian Buenahora Ormaza (BUEORM), *ENGRAMA: arquitectura neuronal
autorregresiva sin atención con recuperación exacta y recordación asociativa*,
2026. Licencia **AGPL-3.0**.
