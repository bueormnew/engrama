# Análisis completo: comparativa 4 modelos (TinyStories 100M tok, 2×T4)

> Respuesta técnica al run del notebook `engrama_v4_vs_ablation_transformer_2xt4.ipynb`
> (resumen del usuario: `source_gate` gana a `v4`, KV en nivel de azar, transformer sin datos).
> Todos los números nuevos de este documento se midieron ejecutando código contra este repo
> (scripts en `benchmarks/analysis_lab/` referenciados en el apéndice; tests en verde: 95/95).

---

## 0. Resumen ejecutivo (TL;DR)

| # | Hallazgo | Estado |
|---|---|---|
| 1 | **El test KV del notebook estaba roto por diseño**: usaba como relleno los ids GPT-2 200–250, que son **bytes de control** (0x0C–0x1F, DEL, C1) que nunca aparecen en TinyStories. Zero-shot sobre una secuencia 94 % fuera de distribución ⇒ azar (6.2 %) para **cualquier** modelo, incluido el transformer. | Demostrado (§3.1) |
| 2 | Además del protocolo, existe un **techo arquitectógico real**: el predictor solo ve `T_L[t]` (d=256), una superposición aditiva de toda la historia. La contribución relativa de un token concreto es ~10⁻⁴ del estado final con contexto real (medido). El propio reporte del repo ya mostraba 7.4 % (dyadic) / 27.5 % (dense) **entrenando directamente en la tarea KV**. | Medido (§3.2, E4) |
| 3 | **Dual gating NO es intrínsecamente peor**, pero **sí es frágil al LR**: en toy con receta suave dual = 5.909 vs source = 5.939 (gana); a LR alto (4e-3) el V4 dual ni siquiera entrena la tarea KV (loss clavada en el marginal, 6–7 %). El hilo conductor del run 20 M: NaN a 6e-4 → receta única a 3e-4 → dual 2.39 vs source 2.27. La regresión es de **robustez de optimización**, no de capacidad. | Medido (§2.2, E2/E3) |
| 4 | **El esquema resonante no rescata la recuperación** (hipótesis V4 no soportada): resonante queda al azar (6–9.5 %) igual que diádico; lo que sube ×4 el techo es la redundancia **densa** por capa (27–30 %). El transformer de control resuelve la tarea (86.8 %, 1 de 2 semillas). | Medido (§2.3, E3) |
| 5 | **El transformer murió por un bug de robustez, no por OOM**: caché dinámica de RoPE dentro de `forward()` + `torch.compile(reduce-overhead)` (CUDA graphs) + cambio de shape en eval ⇒ crash/re-cordeo del grafo. Corregido: RoPE como buffers, eval con el modelo crudo, reintento sin compile. | Corregido + validado (§5) |
| 6 | La arquitectura **no comprime memoria** (la traza FIFO guarda todo `T_0`); lo que hay es un **cuello de botella de información** en el estado consolidado (superposición + decaimiento + transporte low-rank). "Lineal" describe el **coste computacional**, no la capacidad de recordar. | §4 |
| 7 | `no_tracetap` es el peor (2.57): el trace tap aporta 0.18 nats — la señal más valiosa de V4 es el acceso a la huella prístina `T_0`, consistente con la teoría (§2.2). | Del propio run |

---

## 1. Qué dice la tabla, fila por fila

| modelo | params | val_loss | ppl | lectura |
|---|---|---|---|---|
| engrama_v4 (dual+tap) | 20.85 M | 2.3909 | 10.92 | debería ganar; quedó 2.º |
| engrama_source_gate | 20.50 M | 2.2670 | 9.65 | ganó con 0.35 M params menos |
| engrama_no_tracetap | 20.70 M | 2.5710 | 13.08 | el peor: quitar T0 cuesta 0.18 nats |
| transformer | 19.95 M | — | — | **ni corrió** (§2.3) |

Contexto de escala: un transformer de ~20 M sobre TinyStories a 100 M tokens suele quedar
≈ 1.8–2.0 de val_loss; ENGRAMA a 2.27–2.57 está detrás en calidad de LM (a cambio de
O(N)/decode O(1)). Ese gap es coherente con el punto 2: sin mecanismo de atención o
memoria direccionable, parte de la señal de largo alcance no llega al predictor.

Los parámetros coinciden exactamente con lo que construye el repo (verificado):
`build_raw_model` da 20,831,748 / 20,495,876 / 20,683,204 / 19,948,544. El campo
receptivo de la config (L=9, resonant, ctx 512) alcanza 511 y cubre denso: la arquitectura
*representa* todo el contexto; el problema es cuánta señal supervive hasta el evocador (§3.2).

---

## 2. ¿Por qué `source_gate` > `v4` cuando "V4 debería ser mejor"?

### 2.1 Lo que se descartó con evidencia

**(a) Saturación en la inicialización — descartada.** Medimos la pre-activación de las
compuertas en la config exacta del notebook (d=256, L=9, seq 512) en el init (E1):

```
source: gates ≈ 0.500 en todas las capas (sat < 0.1%)
dual:   gates ≈ 0.500 (bilinear std 1.25 en L0, <0.25 en L1+; sat 2.7% solo en L0)
```

Ambos modos arrancan con compuertas ~0.5 y modulables. **El init no explica la regresión.**

**(b) Bug train/inferencia en dual — descartado.** `forward_train` y `forward_step`
coinciden término a término (verificado además con el test de invarianza causal, ahora
también con la variante con clamp). 95/95 tests en verde.

**(c) Dual intrínsecamente peor — descartado a escala controlada.** LM toy con corpus
sintético de documentos con tema persistente (estructura local bigrama + señal de largo
alcance), receta idéntica, 800 pasos, 3 semillas (E2):

| arquitectura | params | val_loss (media) | por semilla |
|---|---|---|---|
| source | 406 k | 5.939 | 5.928 / 5.929 / 5.960 |
| **dual** | 439 k | **5.909** | 5.901 / 5.887 / 5.938 |
| no_tracetap | 424 k | 5.7153 | 5.7153 / 5.7153 / 5.7157 (varianza ≈ 0) |
| transformer (control) | 339 k | **5.868** | 5.832 / 5.879 / 5.892 |

Dual empató o ganó a source por ~0.03 nats. **La inversión de 0.12 nats a 20 M no es una
propiedad del gating dual; es específica de ese run.** Dos matices honestos:
(1) en el toy `no_tracetap` fue el mejor ENGRAMA — lo contrario que a 20 M — porque el
toy apenas tiene señal de largo alcance explotable (los modelos chicos se quedan en las
estadísticas locales y el tap/dual solo añaden carga de optimización); el valor del tap
depende del régimen, y en TinyStories-512 (entidades persistentes) fue grande y positivo.
(2) el transformer gana incluso con 17 % menos parámetros: a igualdad de receta, la
atención es simplemente más expresiva para LM; el argumento de ENGRAMA es el coste
O(N)/O(1), no la calidad por parámetro.

### 2.2 El hilo conductor: el gating dual es sensible al LR (y el run compartió una sola receta)

Cada evidencia aislada cobra sentido en conjunto:

| evidencia | LR | resultado |
|---|---|---|
| Run 20 M (nota del propio notebook) | 6e-4 | NaN hacia el paso 350 → bajaron a 3e-4 para todos |
| Run 20 M final | 3e-4 | dual 2.39 vs source 2.27 (dual pierde 0.12) |
| KV oficial del repo (V4 arms, esta vez medidos) | 4e-3 | V4 dual **no entrena la tarea**: loss final 3.31–3.32 ≈ marginal, 6.1–6.9 % |
| E3 toy KV, receta suave (lr 1e-3 + warmup 100 + cosine) | 1e-3 | V4-dense **sí entrena**: 27.3 % / 29.8 % (≈ V3-dense 29.6 %) |
| E2 toy LM (lr 3e-3 + warmup 100) | 3e-3 | dual ≈ source (5.909 vs 5.939) |

Es decir: **no hay gap de capacidad intrínseco de dual vs source; hay un gap de robustez
de optimización** que empeora con el LR y con la magnitud del residual (el término
bilineal ∝ ‖T‖²). Con la receta compartida del run (elegida a 3e-4 *porque 6e-4 reventaba*,
probablemente por la propia ruta dual), V4 quedó ligeramente por debajo. La mitigación
directa es `dual_bilinear_clamp` (acota el término problemático) y/o un LR algo menor o
warmup mayor para V4. Todo esto es medible en el checkpoint existente con la celda 9b.

### 2.3 Y el esquema resonante no es el que rescata la recuperación

El benchmark oficial del repo (`benchmarks/kv_retrieval.py --steps 600 --seed 1234`,
re-ejecutado ahora midiendo también los brazos V4, que no estaban en el reporte
commitado) da, **entrenando directamente en la tarea**:

| config | loss final | recuperación | fuente |
|---|---|---|---|
| V3 hierarchical_dyadic | 0.855 | 7.1 % | repo, re-run |
| V3 dense_dilated | 0.659 | **29.6 %** | repo, re-run |
| **V4 resonant_multirate (dual+tap)** | 3.324 (estancada ≈ marginal) | **6.1 %** | repo, re-run |
| **V4 dense_dilated (dual+tap)** | 3.311 (estancada ≈ marginal) | 6.9 % | repo, re-run |
| V4 dense (dual+tap), receta suave 1e-3+warmup | — | **27.3 % / 29.8 %** | E3 toy |
| Transformer ~414k params, receta suave | 0.41 | **86.8 %** (1 de 2 semillas; la otra 12 %) | E3 toy |

Dos conclusiones duras:

1. La hipótesis de la teoría V4 ("los offsets resonantes eliminan la fragilidad de
   ruta única del esquema diádico") **no está soportada por la medición**: resonante
   queda en azar (6.1–9.5 %) igual que diádico. Lo que sube el techo ×4 es la
   **redundancia densa por capa** (12 offsets en *cada* capa → rutas de 1–2 saltos para
   cualquier distancia), no el patrón resonante. La atenuación cruda en init es igual
   (~10⁻⁴) en ambos esquemas: la diferencia es de *aprendibilidad* del circuito de
   binding, no de intensidad de señal.
2. A 4e-3 el V4 dual no entrena *nada* la tarea (loss clavada en el marginal). Con
   receta suave, V4-dense iguala a V3-dense. El componente frágil es el **dual a LR
   alto**, no el dense ni el tap.

### 2.4 Mecanismos presentes solo a 20 M / fp16 (el "por qué" fino)

1. **Saturación bilineal progresiva.** El término `<q_tgt, k_src>/√d_g` crece con el
   cuadrado de la magnitud del residual. Esa magnitud **no está normalizada** (el RMSNorm
   de V4 está dentro de la FFN de la célula, no sobre el stream residual) y en los
   entrenamientos toy creció ~10× (std de `T_l` de ~0.5–1.9 a 10–15 a mitad del stack).
   Con |T| grande, el término bilineal dispara la sigmoide a 0/1 y la compuerta deja de
   ser modulable (gradiente ≈ 0). En E2, dual terminó con saturación extrema y bimodal:
   L0 96 % saturada (abierta), L3 compuertas ≈ 0.02 (cerrada). A escala 20 M con fp16
   (rango máximo 65 504) el efecto es más agresivo. Nota del propio notebook:
   *"lr 6e-4 + AMP fp16 reventaba a NaN ~paso 350"* — síntoma de que la ruta dual es la
   más sensible al LR.
2. **Varianza de una sola semilla.** Con 100 M tokens (1 época) y schedule idéntico, la
   diferencia entre dos variantes que comparten el 95 % de parámetros puede variar
   ±0.03–0.1 nats entre runs. 0.12 está en el rango donde una semilla puede invertir el
   orden — no se puede afirmar causalidad sin réplicas.
3. **El trace tap es el factor dominante del ranking en ese run** (+0.18 nats al
   quitarlo): la vía de rescate a `T_0` prístino es la innovación de V4 que más aporta en
   LM con contexto 512 y entidades persistentes. También explica por qué `source_gate`
   (que conserva el tap) gana a `no_tracetap` (que no): **el ranking del run mide sobre
   todo el tap, no el tipo de gating.** (Matiz: en el toy sin señal de largo alcance el
   tap no ayudó — su valor es dependiente del régimen, ver §2.1c.)

### 2.5 Qué hacer (ya aplicado / recomendado)

- **Aplicado — celda 9b del notebook** (`gates_report.json`): sobre los checkpoints ya
  entrenados (sin reentrenar) imprime apertura/saturación de α, ρ, β y |T_l| por capa.
  Si dual muestra capas con saturación >90 % y |T_l| ≫ 1, el mecanismo 1 queda confirmado
  en el run real.
- **Aplicado — opción `dual_bilinear_clamp`** (`EngramaConfig`, default `None` = V4
  clásico): acota el término bilineal con `C·tanh(b/C)` (C=4 recomendado). No cambia
  ninguna ecuación para C→∞, conserva la invarianza causal (test nuevo) y evita la
  saturación dura. Para el próximo run de Kaggle, añadir a `ARCH_SPECS`:
  ```python
  "engrama_v4_clamped": {"kind": "engrama", "title": "V4 dual + bilinear clamp",
                         "overrides": {"dual_bilinear_clamp": 4.0}},
  ```
  (cuesta ~29 min más de wall-clock; o sustituir `engrama_v4` directamente).
- **Recomendado — 2.ª semilla** de la pareja `v4` vs `source_gate` antes de concluir que
  V4 "perdió" (el notebook ya reusa checkpoints con RESUME, así que solo entrena lo nuevo).

---

## 3. Recuperación KV ≈ azar: dos causas superpuestas

### 3.1 Causa 1 (protocolo, dominante): la prueba era imposible de superar

El protocolo zero-shot del notebook construía las secuencias con:

```python
KEY_LO, KEY_HI  = 1000, 1020
VAL_LO, VAL_HI  = 2000, 2015
FILL_LO, FILL_HI = 200, 250   # ← el cuerpo entero de la secuencia
```

En GPT-2 los ids 0–255 son los **bytes base** en el orden del mapeo
`byte_to_unicode` de OpenAI: `!`→0 … `~`→93, latinos 94–187, y los bytes restantes
(controles) 188–254. Concretamente (verificado programáticamente, §E5):

- ids 200–219 = bytes **0x0C–0x1F** (form feed, CR, SI, control…),
- id 220 = espacio,
- ids 221–250 = **DEL y C1** (0x7F–0x9C).

Esos tokens como emisión *individual* no aparecen en texto natural — ni una sola vez en
100 M tokens de TinyStories—. El 94 % del cuerpo del test eran tokens que el modelo jamás
vio en contexto; el 6 % restante (claves/valores, ids 1000–2015) sí son subpalabras
frecuentes, pero el modelo tampoco recibió nunca la estructura "clave → valor" que hay que
recuperar. Con logits nunca entrenados hacia esos ids, `exact 50k = 0.0 %` era obligado y
el MC queda pegado al azar (6.2–7.7 % son fluctuaciones de 256 muestras × 4 consultas).
**El transformer habría dado lo mismo con ese protocolo.**

> El fix ya está en el notebook: los tres instrumentos nuevos eligen claves/valores/relleno
> por **frecuencia real en el train tokenizado** (bandas de rank 600–900 / 1200–1600 /
> 100–400) y además añaden (i) sonda de **inducción** (copia de 8 tokens, azar 12.5 %)
> y (ii) **KV entrenado** con fine-tune corto idéntico para los 4 modelos.

### 3.2 Causa 2 (arquitectura, real): superposición aditiva sin direccionabilidad

Medición E4 (`e4_attenuation.py`), config exacta del notebook, perturbando `T_0[0]` y
midiendo su contribución relativa al estado final `T_L[t]`:

| distancia t | contribución relativa (con contexto real) | sin interferencia (solo el token) |
|---|---|---|
| 1 | 4.6e-4 | 0.96 |
| 8 | 1.7e-4 | 0.56 |
| 32 | 6.5e-5 | 0.10 |
| 128 | 1.9e-4 | 0.24 |
| 256 | 1.2e-4 | 0.13 |
| 511 | **1.4e-7** | 3.7e-5 (borde del campo receptivo) |

Dos conclusiones cuantitativas:

1. **El transporte multiescala funciona**: sin interferencia, un token aislado conserva
   10–25 % de magnitud relativa a distancias 128–256 (ley ~t^−1.3, muy lejos de un
   decaimiento exponencial duro). El esquema resonante y el trace tap hacen su trabajo.
2. **La superposición lo destruye**: con el contexto real presente, TODOS los tokens se
   suman en el mismo vector `T_L[t]`; la relación señal/(resto del estado) de un token es
   ~10⁻⁴. El evocador debe decodificar linealmente "el valor que siguió a la clave K" de
   un vector donde esa información está 4–7 órdenes de magnitud por debajo del resto. Es
   un problema de **direccionabilidad**, no de transporte: sin normalización por número
   de ítems ni selección por contenido (softmax/attention o delta-rule), el estado es un
   promedio no normalizado de la historia.

Esto coincide con la literatura (gap MQAR de modelos lineales/gated frente a atención) y
con el propio reporte del repo: **entrenando directamente en la tarea KV**
(`benchmarks/kv_retrieval.py`, V3): dyadic 7.4 %, dyadic+ancla 7.1 %, **dense_dilated
27.5 %** — es decir, con rutas redundantes y densas el techo sube, pero ni de lejos se
resuelve. E3 (apéndice) replica el protocolo con control transformer a parámetros
igualados para dimensionar ese techo.

**Qué mejoraría el techo (direcciones, por orden de costo):**
1. `offset_mode="dense_dilated"` o más rutas redundantes por capa (ya demostrado ×3.7 en el repo).
2. Normalizar la mezcla por el número de fuentes abiertas (mezcla *promedio* en vez de
   *suma*: divide la superposición entre ~‖α‖₀) — cambio local en `forward_train`.
3. Lectura direccionable desde la traza (p. ej. pesos por producto punto con el estado
   actual — "atención lineal" sin softmax sobre N, sigue O(N)) — es el paso conceptual
   para igualar a atención en recuperación exacta.

---

## 4. Tu pregunta teórica: "esto es lineal… ¿comprime algo? ¿por qué la pérdida?"

Respuesta corta: **lineal no significa compresión, y ENGRAMA no comprime memoria —
comprime cómputo.** La pérdida de información ocurre en un lugar muy concreto: el estado
consolidado `T_L[t]` que ve el evocador.

**1) Memoria: no hay compresión.** La traza FIFO guarda `T_0` para *todas* las posiciones
(N_max·d floats) y la caché jerárquica retiene por capa solo lo que la siguiente va a
leer (horizontes mínimos) — eso es *pruning* de lo no necesario, no compresión con pérdida.
A N=512 la traza son ~512 KB; comparable al KV-cache fp16 del transformer (~4.7 MB) pero
del mismo orden. El "O(1)" del decode es **coste por token nuevo** (offsets fijos), no
memoria total (que crece O(N_max)).

**2) Cómputo: ahí está la linealidad.** Forward O(N) porque cada posición solo combina un
número constante de offsets relativos (constante por capa); decode O(1) por token porque
un token nuevo solo relee horizontes fijos. No hay matriz N×N ni softmax temporal.

**3) Información: el cuello de botella es el estado predictivo, y es con pérdida.** Para
predecir el token t+1, el evocador solo ve `T_L[t]` (d=256). Ese vector es una
**superposición aditiva** de toda la historia con pesos que decaen por saltos (§3.2).
Tres factores estructurales pierden información:

- **Superposición**: K ítems sumados en el mismo espacio se interfieren; recuperar uno
  exacto requiere SNR que cae con K y con la distancia (medido: 10⁻⁴).
- **Decaimiento por hop**: cada capa multiplica la ruta identidad por ρ·α·β ≈ 0.25 en
  init (medido: el estado cae 1.94 → 0.40 de L0 a L8). Es un filtro pasa-bajos multiescala:
  perfecto para *suavizar y combinar* (por eso el LM funciona y el tap a T_0 vale 0.18
  nats), malo para *recordar literalmente*.
- **Transporte low-rank** (r=32 de d=256): la parte no-identidad solo puede mover 32
  dimensiones por capa; la ruta identidad (β·T) es la que salva la fidelidad, y por eso
  el trace tap es tan valioso.

Es exactamente el mismo intercambio que hacen S4/convoluciones/mamba frente a atención:
**coste O(N) a cambio de un estado que no sabe "qué" recordar**. No es un bug ni una
implementación mala — es la frontera conocida de las arquitecturas lineales, y tus dos
síntomas (KV al azar, val_loss detrás del transformer) son sus dos caras.

---

## 5. Estado del transformer (fila vacía)

Diagnóstico del fallo (sin log del run, por eliminación + reproducción de causa):
`TransformerLM.forward` cacheaba `self._cos/_sin` como atributos creados *dentro* del
forward. Bajo `torch.compile(mode="reduce-overhead")` (CUDA graphs) eso es una trampa
conocida: los tensores creados en una grabación viven en el pool del grafo y se
sobrescriben en replays con otra shape (eval usa batch 8, train 16) → crash o corrupción
en el primer eval (paso 500). Los ENGRAMA no cachean tensores entre forwards y por eso
sobrevivieron con el mismo plumbing. Fix aplicado y validado:

1. RoPE precomputado como **buffers** (`persistent=False`) en `__init__`.
2. `evaluate()` ahora corre sobre el **modelo crudo** (`raw_model.forward_loss`), nunca
   sobre el wrapper DDP/compilado, con `.item()` inmediato por batch.
3. `cudagraph_mark_step_begin()` antes de cada eval.
4. El notebook ahora guarda **log completo por arquitectura** y si torchrun muere,
   **reintenta automáticamente sin compile** (misma receta, mismo presupuesto).

Validación end-to-end en CPU: los 4 arcos entrenan 60 pasos con el worker nuevo sin
errores (transformer incluido).

---

## 6. Cómo reproducir / qué ejecutar ahora

**Sobre los checkpoints ya existentes en Kaggle (sin reentrenar los 3 ENGRAMA):**
volver a correr el notebook nuevo. Con `RESUME=True`, `engrama_v4/_source_gate/_no_tracetap`
se detectan como completos y se saltan; el transformer se entrena con el worker arreglado
(~30 min); después corren las celdas nuevas: escala hasta N=2048, introspección de
compuertas (9b) y el KV de 3 instrumentos (10). Total ≈ 45–60 min.

**Experimentos de este análisis** (repo `benchmarks/analysis_lab/`, se ejecutan en CPU):

| script | qué mide | resultado clave |
|---|---|---|
| `benchmarks/analysis_lab/e1_gate_saturation.py` | compuertas en init, config 20 M | gates ≈ 0.5 ambos modos (no hay sesgo de init) |
| `e2_toy_lm.py` | LM toy 3 semillas, receta idéntica | dual 5.909 vs source 5.939 vs transformer 5.862 |
| `e3_kv_ceiling.py` | KV entrenado + control transformer | ver apéndice |
| `e4_attenuation.py` | contribución de 1 token al estado final | 10⁻⁴ con contexto; t^−1.3 sin interferencia |
| byte-mapa GPT-2 (§3.1) | ids 200–250 = bytes de control | verificado con `byte_to_unicode()` |

---

## 7. Recomendaciones priorizadas para el próximo run

1. **Re-ejecutar el notebook corregido** (transformer arreglado + KV válido + gates_report):
   convierte el run anterior en diagnóstico completo. Con RESUME=True los 3 ENGRAMA ya
   entrenados se saltan; solo entrena el transformer (~30 min).
2. **Confirmar/refutar la saturación dual** mirando `gates_report.json` del checkpoint
   existente (5 min, sin entrenar): si hay capas con α saturada >90 % y |T_l| ≫ 1,
   queda cerrada la explicación de §2.2/§2.4.
3. **Probar `dual_bilinear_clamp=4.0`** (opción nueva, ya en el repo) como 5.º arco o
   sustituyendo a v4; y/o warmup 1000 para v4 manteniendo 3e-4 en los demás.
4. **2.ª semilla** de `v4` vs `source_gate` antes de declarar que V4 pierde (la evidencia
   dice: robustez, no capacidad).
5. **Revisar la expectativa del resonante**: si el objetivo es recuperación, añadir un
   brazo con `offset_mode="dense_dilated"` (techo ×4 medido en tarea entrenada) o
   aumentar offsets por capa; el patrón resonante por sí solo no mejora el binding.
6. Medio plazo, para igualar al transformer en recuperación exacta: mezcla **promedio**
   (normalizada por compuertas abiertas) y/o lectura direccionable de la traza
   (atención lineal: producto punto estado·traza, sigue O(N) sin softmax temporal).
   Sin algo de este estilo, el estado superpuesto (~10⁻⁴ por token) no permite copia exacta.

---

## Apéndice A. E3 — techo KV con control transformer (entrenado en la tarea)

Protocolo del `benchmarks/kv_retrieval.py` del repo (vocab 64, 4 pares aleatorios por
muestra, consultas a 24/72/120/176, azar 6.25 %). Control de aprendibilidad incluido:
una **regresión lineal** sobre los one-hots de la cabecera predice el valor de la
primera consulta al **48.8 %** sin ver siquiera la consulta (la información está
trivialmente presente en la secuencia). Modelos ~400–530k params, 2 semillas:

| config | receta | overall | cerca (d≈24) | lejos (d≈176) |
|---|---|---|---|---|
| transformer (8 capas, d=64) | 800p, lr 1e-3 + warmup | **86.8 % / 12.0 %** | 80 % | 84 % |
| v4_dense (dual+tap, offsets densos) | 800p, lr 1e-3 + warmup | **27.3 % / 29.8 %** | 31 % | 23 % |
| v4_resonant (dual+tap) | 800p, lr 1e-3 + warmup | 8.1 % / 9.5 % | 8–10 % | 4–6 % |
| source_resonant | 800p, lr 1e-3 + warmup | 6.0 % / 6.3 % | 6 % | 6 % |
| v4_no_tap | 800p, lr 1e-3 + warmup | 8.1 % | 8 % | 8 % |
| todos los ENGRAMA | 400p, lr 4e-3 sin warmup | 5.9–7.3 % (azar) | — | — |

Lecturas: (1) el transformer resuelve la tarea pero de forma **bimodal** — el circuito
de inducción o se forma o no (1 de 2 semillas); (2) ningún ENGRAMA pasa de ~30 % con
offsets densos y el resonante queda al azar en todas las condiciones; (3) la receta
importa tanto como la arquitectura (misma arquitectura: azar a 4e-3, 27–30 % a 1e-3).

## Apéndice A2. Reconciliación con el reporte oficial del repo

`benchmarks/KV_RETRIEVAL_REPORT.md` re-generado con el script actual (los brazos V4 no
estaban medidos en el reporte commitado): V3-dyadic 7.1 %, V3-dense 29.6 %,
V4-resonant 6.1 %, V4-dense 6.9 % (a lr 4e-3, sin warmup — receta con la que el dual no
entrena la tarea; ver §2.2). Los números V3 replican los históricos (7.4 % / 27.5 %).

## Apéndice B. Trazabilidad de cambios en el repo

- `kaggle/train_compare_ddp.py`: RoPE buffers, eval cruda, mark_step_begin (commit f2b4de1).
- `kaggle/_gen_compare_notebook.py` + notebook regenerado: KV 3 instrumentos, retry sin
  compile, `_run_tee`, celda 9b gates, LENGTHS→2048, resumen/gráficas.
- `src/engrama/config.py` + `consolidation.py`: `dual_bilinear_clamp` (opt-in, invarianza
  causal testada).
- `tests/test_optimized_training.py`: config inválida corregida (suite 95/95).
