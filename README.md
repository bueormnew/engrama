# ENGRAMA 🧠⚡
## Arquitectura Neuronal Autorregresiva sin Atención de Alto Rendimiento

**ENGRAMA** es una arquitectura neuronal autorregresiva de memoria explícita **sin atención** — cero $QK^\top$, cero matrices de afinidad $N \times N$, cero decaimiento exponencial artificial, cero softmax sobre la dimensión temporal — implementada en **PyTorch puro**. 

> **Novedad — ENGRAMA V5 (1.0 + V5.1)**: sin atención, sin compresión y con
> **recuperación exacta a cualquier distancia** (100 % denso / 96 % con entrenamiento
> lineal LSH a 16384 tokens, entrenando solo a 2048). Ver la sección
> [ENGRAMA V5](#-engrama-v5--sin-atención-sin-compresión-recuperación-exacta) más abajo.

Esta versión introduce **ENGRAMA V4**, diseñada para resolver de forma simultánea el rendimiento computacional en entrenamiento (aceleración de más de **$15\times$**, reduciendo de 30+ horas a **~1.8 horas** en Kaggle para 500M de tokens) y la precisión de recuperación en contextos cortos, medios, largos y extremos mediante **gating bilateral target-source**, **acceso directo a la traza prístina ($T_0$ Trace Tap)**, **jerarquía de offsets resonantes multiescala** y **evocador de fusión latente**.

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](#-instalación)
[![PyTorch ≥ 2.0](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-118%20passing-brightgreen)](docs/VERIFICACION.md)
[![Sin atención](https://img.shields.io/badge/attention-zero-important)](#-filosofía-y-las-4-fases-de-engrama)

- **Autor**: Gerson Fabian Buenahora Ormaza (BUEORM)
- **Año**: 2026
- **Licencia**: GNU Affero General Public License v3.0 (AGPL-3.0)
- **Versión**: 0.5.0 (Arquitectura V4 + runtime de entrenamiento optimizado)

---

## 📑 Tabla de Contenidos

- [🧠 Filosofía y las 4 Fases de ENGRAMA](#-filosofía-y-las-4-fases-de-engrama)
- [🚀 Evolución Arquitectónica: V1 vs V2 vs V3 vs V4](#-evolución-arquitectónica-v1-vs-v2-vs-v3-vs-v4)
- [⚡ Novedades y Soluciones de ENGRAMA V4](#-novedades-y-soluciones-de-engrama-v4)
  - [1. Velocidad de Entrenamiento: de 30h a ~1.8h](#1-velocidad-de-entrenamiento-de-30h-a-18h)
  - [2. Gating Bilateral Target-Source (Dual Gating)](#2-gating-bilateral-target-source-dual-gating)
  - [3. Acceso Directo de Traza (Trace Tap T0)](#3-acceso-directo-de-traza-trace-tap-t0)
  - [4. Jerarquía de Offsets Resonante Multi-Ruta](#4-jerarquía-de-offsets-resonante-multi-ruta)
  - [5. Evocador de Fusión Latente $O(\|V\|d)$](#5-evocador-de-fusión-latente-ovd)
  - [6. Célula con RMSNorm y Normalización sin Centrado](#6-célula-con-rmsnorm-y-normalización-sin-centrado)
- [📦 Instalación](#-instalación)
- [🚀 Modo Rápido (Quickstart en 3 líneas)](#-modo-rápido-quickstart-en-3-líneas)
- [🛠️ Modo Experto y Configuración Completa](#️-modo-experto-y-configuración-completa)
- [⚡ Entrenamiento Acelerado con AMP en GPU (Kaggle / Colab)](#-entrenamiento-acelerado-con-amp-en-gpu-kaggle--colab)
- [🔍 Invarianza Causal y Caché de Horizonte Mínimo](#-invarianza-causal-y-caché-de-horizonte-mínimo)
- [📊 Benchmarks y Reportes](#-benchmarks-y-reportes)
- [📂 Estructura del Repositorio](#-estructura-del-repositorio)
- [📄 Cómo Citar y Licencia](#-cómo-citar-y-licencia)

---

## 🧠 Filosofía y las 4 Fases de ENGRAMA

La mayoría de modelos de lenguaje modernos utilizan mecanismos de atención ($QK^\top$) con coste computacional y de memoria cuadrático $O(N^2)$, o modelos recurrentes comprimidos (SSMs/RNNs) donde la memoria se diluye en un estado oculto de tamaño fijo.

**ENGRAMA** se fundamenta en una teoría alternativa de memoria estructurada inspirada en el engrama biológico: **la experiencia deja una huella aislada e incorruptible, se almacena explícitamente en una memoria de trabajo y el contexto se consolida progresivamente a través de sinapsis jerárquicas causales relativas**.

```text
                            ┌────────────────────────┐
                            │    TOKEN DE ENTRADA    │
                            └───────────┬────────────┘
                                        │
                                        ▼  [FASE 1]
                            ┌────────────────────────┐
                            │   ISOLATED ENCODER     │
                            │ Huella aislada e_i     │
                            │ Red C x C de Sinapsis  │
                            │ Cero mezcla temporal   │
                            └───────────┬────────────┘
                                        │
                                        ▼  [FASE 2]
                            ┌────────────────────────┐
                            │  CIRCULAR TRACE (T0)   │
                            │ Memoria explícita FIFO │
                            │ Almacena (T0[t], t)    │
                            │ Cero transformación    │
                            └─────┬──────────────┬───┘
                                  │              │
               ┌──────────────────┘              │ Acceso Directo de Traza
               │ T0[t-p]                         │ (Direct Trace Tap)
               ▼                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   [FASE 3] CONSOLIDATION STACK                         │
│                                                                        │
│  - Offsets Resonantes: D_l = {0, 1, 2^{l-1}, 2^l}                      │
│  - Sinapsis Factorizada con Transporte de Identidad:                   │
│      y = beta * x + U Diag(s) V^T x                                    │
│  - Gating Dual Target-Source Bilineal:                                 │
│      alpha = sigmoid( (q_tgt . k_src)/sqrt(d_g) + q W_tgt + k W_src ) │
│  - Trace Tap: Rescata huella limpia T0[t-p] en capas profundas         │
│  - Célula Compartida: F_l(x) con RMSNorm y modulación por célula       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼  [FASE 4]
                            ┌────────────────────────┐
                            │  MULTI-CANDIDATE       │
                            │  EVOKER (Fusión Lat.)  │
                            │ M candidatos en R^d    │
                            │ Fusión adaptativa      │
                            │ Proyección O(|V|d)     │
                            └───────────┬────────────┘
                                        │
                                        ▼
                                 SIGUIENTE TOKEN
```

### Detalle de las 4 Fases:

1. **Fase 1: Codificación Aislada (`IsolatedEncoder`)**:
   Cada token $x_i$ se proyecta y procesa a través de un grupo de $C$ células neuronales interconectadas mediante una matriz $C \times C$ de sinapsis. Esta etapa se ejecuta **en paralelo para toda la secuencia** sin mezclar ninguna posición temporal ($T_0[i]$ depende únicamente de $x_i$).

2. **Fase 2: Traza Circular FIFO (`CircularTrace`)**:
   Las huellas $T_0[i]$ se escriben con su timestamp absoluto en un buffer circular de capacidad $N_{max}$. **La Traza no transforma ni comprime la información**; actúa como almacén explícito de memoria de trabajo.

3. **Fase 3: Consolidación Causal Jerárquica (`ConsolidationStack`)**:
   Una pila de $L$ capas combina información a través de offsets relativos causales $p \in D_l$. Cada conexión sináptica cuenta con:
   - **Ruta de Identidad**: $\beta_{l,p} \cdot x$ para transportar representaciones prístinas sin deformación.
   - **Ruta de Transformación Low-Rank**: $U_l \operatorname{Diag}(s_{l,p}) V_l^\top x$ sobre subespacios compartidos de rango $r \ll d$.
   - **Gating Dual Target-Source**: Modula la apertura de la sinapsis evaluando tanto lo que busca el presente como lo que ofrece el pasado.
   - **Trace Tap**: Vía directa para rescatar la huella prístina $T_0[t-p]$ en cualquier nivel de profundidad.

4. **Fase 4: Evocación Multicandidato (`MultiCandidateEvoker`)**:
   El estado contextual final $h_* = T_L[t]$ genera $M$ hipótesis candidatas en el espacio latente $\mathbb{R}^d$, las combina mediante una compuerta aprendida en $\mathbb{R}^d$ y proyecta un único vector contra la matriz de vocabulario en tiempo lineal $O(|V|d)$.

---

## 🚀 Evolución Arquitectónica: V1 vs V2 vs V3 vs V4

| Característica | ENGRAMA V1 | ENGRAMA V2 | ENGRAMA V3 | ENGRAMA V4 (Actual) |
|---|---|---|---|---|
| **Atención ($QK^\top$)** | ❌ No | ❌ No | ❌ No | ❌ **No (Cero atención)** |
| **Parametrización Sinapsis** | Densa $C^2 d^2$ | Densa $C^2 d^2$ | Factorizada $2dr + C^2 r$ | **Factorizada Vectorizada $2dr + C^2 r$** |
| **Gating de Sinapsis** | Estático | Dependiente de fuente | Dependiente de fuente | **Dual Target-Source (Bilineal)** |
| **Offsets por Capa ($D_l$)** | Todos ($O(N)$) | Diádico completo ($\sim\log N$) | Diádico sparse $\{0, 1, 2^l\}$ | **Resonante $\{0, 1, 2^{l-1}, 2^l\}$** |
| **Acceso a Traza Limpia ($T_0$)** | Solo capa 0 | Solo capa 0 | Solo capa 0 | **Multiescala Direct Trace Tap** |
| **Caché en Inferencia** | Recálculo total | Caché $L \times N \times d$ | Caché mín. $\sum 2^l d \approx Nd$ | **Caché mín. $\sum 2^l d \approx Nd$** |
| **Normalización en Célula** | LayerNorm | LayerNorm | LayerNorm | **RMSNorm (Preserva signos)** |
| **Evocador Multicandidato** | Denso $M d^2$ | Denso $M d^2$ | Factorizado ($logsumexp/mean$) | **Latent Fusion ($O(\|V\|d)$ sin checkpoints)** |
| **Tiempo de Entrenamiento (500M tok)** | >60 horas | >40 horas | >30.8 horas | **~1.8 a 2.3 horas en GPU T4** |
| **Recuperación KV Exacta** | Baja | Media (27.5%) | Baja (7.4% diádico) | **Sin ventaja medida todavía** (ver nota) |

---

> **Nota honesta sobre la recuperación KV (2026-08):** la cifra ">75%" que aparecía aquí
> era una proyección sin medición que la respaldara. Los números reales del repo son los
> del `benchmarks/KV_RETRIEVAL_REPORT.md` (V3: 7.4% diádico, 27.5% dense, entrenando en la
> tarea). El run 2×T4 con vocabulario GPT-2 midió 6.2–7.7% con un protocolo zero-shot
> cuyo relleno (ids 200–250) son bytes de control fuera de distribución — resultado no
> concluyente por diseño. El notebook `kaggle/engrama_v4_vs_ablation_transformer_2xt4.ipynb`
> incluye ahora un protocolo corregido (zero-shot in-distribución + inducción + KV
> entrenado) y el análisis del techo arquitectónico en
> `docs/ANALISIS-COMPARATIVA-4-MODELOS.md` (§3–4: superposición aditiva del estado
> consolidado; el predictor solo ve $T_L[t]$, con contribución relativa por token ~$10^{-4}$).

---

## ENGRAMA V5 — sin atención, sin compresión, recuperación exacta

> **V5 (1.0) + V5.1 (entrenamiento lineal)**. Diseño completo, registro de iteraciones y
> presupuestos teóricos en [`docs/ENGRAMA-V5-Teorica.md`](docs/ENGRAMA-V5-Teorica.md);
> el análisis forense que la motivó en
> [`docs/ANALISIS-COMPARATIVA-4-MODELOS.md`](docs/ANALISIS-COMPARATIVA-4-MODELOS.md).

### 1. Qué pasó: de V4 a V5

El run 2×T4 de V4 (TinyStories, 100M tokens) dejó tres síntomas: `source_gate` ganaba a
la V4 completa, la recuperación KV quedaba en nivel de azar y el transformer baseline
moría a mitad de run. El análisis (E1–E5, reproducibles en
`benchmarks/analysis_lab/`) lo explicó todo:

1. **La recuperación era imposible por diseño del protocolo** (relleno con bytes de
   control GPT-2 fuera de distribución) **y por física de la arquitectura**: el predictor
   de V4 solo ve `T_L[t]`, una *superposición aditiva* de toda la historia. La contribución
   de un token concreto al estado final es **~10⁻⁴** (medido): ningún ajuste de offsets
   arregla eso. Los offsets resonantes quedaron en 6–9.5 % incluso entrenando en la tarea;
   los densos en 27–30 %; el transformer de control en 86.8 %.
2. **El gating dual de V4 es frágil al LR** (NaN a 6e-4; a 4e-3 la loss de la tarea se
   clava en el marginal): el término bilineal crece con ‖T‖² y el residual sin normalizar
   crece ~10× durante el entrenamiento, saturando las sigmoides.
3. El transformer murió por un bug de robustez (RoPE cacheado + CUDA graphs), corregido.

V5 separa los dos roles que V4 mezclaba: **transporte/suavizado** (consolidación) y
**recuperación exacta** (pieza nueva: *Recall Tap*), y estabiliza todo lo demás.

### 2. Cómo funciona

```text
        tokens ──► Encoder aislado (V1–V4 intacto) ──► T0[j]  (huella pristina, por token)
                                                     │
              ┌──────────────────────────────────────┤
              ▼                                      ▼
   Consolidación V5 (suavizado multiescala)   Recall Tap (recuperación exacta)
   mezcla NORMALIZADA por conteo             q = P_q(T0[i])   K[j] = P_k(T0[j])
   compuerta dual ACOTADA C·tanh(b/C)        j* = argmax_{j≤i-gap} ⟨q, K[j]⟩
   Trace Tap a T0 (mejor pieza de V4)        lectura = T0[j*+1]   (top-1 duro)
              │                                      │
              └─────────► estado + g·W_r(lectura) ◄───┘
                              │
                              ▼
                 Evocador fusión latente (V4) ──► logits
```

- **Encoder aislado** (pilar V1/V2): cada token se codifica sin ver a sus vecinos. De
  aquí sale la propiedad que hace todo lo demás posible: `T0[j]` —y por tanto `K[j]`—
  depende **solo del token j**.
- **Traza explícita, sin comprimir** (pilar 2): la memoria guarda `T0` y `K` completos por
  posición. Memoria **lineal**: 640 bytes/token constantes (medido de 256 a 16384
  posiciones). No hay estado recurrente comprimido (pilar 9).
- **Consolidación V5**: la mezcla multiescala de V4 pero **normalizada por conteo**
  (`T_pos = Σ w_p·y_p / (Σ w_p + ε)`): convierte la suma creciente en *promedio acotado*,
  elimina el crecimiento ×10 del residual y la fragilidad fp16/LR. La compuerta dual
  bilineal va **acotada** por defecto. El Trace Tap a `T0` (la pieza más valiosa de V4,
  +0.18 nats) se conserva.
- **Recall Tap** (nuevo): la consulta aislada `q` del token actual se compara contra los
  códigos `K` de la traza causal; **argmax duro top-1** (empates → ocurrencia más
  reciente) y se lee la huella *completa* del token siguiente al match (`T0[j*+1]`).
  Gradiente por *straight-through* (softmax solo en el backward). Es una lectura de
  diccionario: **sin softmax sobre el eje temporal, sin matriz N×N, sin promedio ponderado
  de posiciones** — sin atención en ningún sentido, a ninguna escala.
- **Generación incremental** (la "cache nativa"): un token nuevo solo añade su `T0`/`K`
  a los anillos y hace un matvec `O(N·d_k)` contra el anillo K. Consolidación `O(1)`
  (offsets fijos). Un solo eje K — nada de 2·L matrices por capa.

**¿Por qué no decae con la distancia?** Un argmax no atenua: el token clave a 16k
posiciones puntúa igual que a 200. Por eso V5 se entrena a 2048 y evalúa **igual de bien
a 16384 sin reentrenar** (no hay extrapolación posicional que aprender).

### 3. Entrenamiento lineal (V5.1)

Puntuar todos los pares cuesta `O(N²·d_k)`. La propiedad de aislamiento lo arregla:
tokens iguales → códigos `K` idénticos → siempre al mismo bucket. El modo
`rt_train_mode="lsh"` puntúa solo **~181 candidatos** por consulta:

- **1 identidad** — índice exacto de la última ocurrencia del mismo token (garantizado;
  es el candidato de inducción/ligadura),
- **4 rescate** — posiciones más recientes,
- **2×64 buckets LSH** — código de signos de `K` (t tablas de b bits),
- **48 negativos muestreados** — evitan que los scores fuera de bucket deriven sin
  oposición (sin ellos: 83.5 %; con ellos: 96.0 %).

Coste `O(N·(1+t·cap+n_neg)·d_k)` — **lineal en N** (~61× menos FLOPs a 8192; ~370× a
32k). La lectura dura conserva su semántica exacta; la generación sigue siendo el camino
exacto. Kernels: `v5/kernels.py` fusiona score+argmax+gather (referencia torch exacta,
paridad dif 0.0; kernel Triton para GPU con fallback automático y `validate_kernel()`).

**Recomendación medida**: denso hasta ~8k (sub-milisegundo en T4), LSH desde ~16k o
donde mande la memoria.

### 4. Comparativa V4 vs V5 (todo medido, CPU 2 núcleos salvo el run Kaggle)

| dimensión | V4 | V5 |
|---|---|---|
| Recuperación KV (entrenada en la tarea) | resonante 6.1–9.5 % · denso 27–30 % | **100.0 % denso · 96.0 % lineal (LSH)** a 2048/8192/16384 |
| KV zero-shot (run Kaggle, protocolo OOD) | 6.2–7.7 % (azar) | — (protocolo corregido en el notebook) |
| Física del recuerdo | ~10⁻⁴ de contribución por token | lectura dura: huella completa, sin atenuación |
| LM toy (3 semillas, misma receta) | dual 5.909 · source 5.939 | **5.872** (transformer: 5.868 — empate) |
| Estabilidad | residual ×10, NaN a 6e-4, loss marginal a 4e-3 | residual acotado; sin NaN en fp16/extremos/LR 10× |
| Invarianza causal | última posición | **exacta en todas las posiciones** (empates incluidos) |
| Memoria de contexto | lineal (traza + horizontes) | **640 B/token exactos**; ~14× menor que KV-cache transformer a 16k |
| Generación incremental | decode plano a N≤512 (medido en T4) | 7 ms/token (CPU); **×17/×36/×88** vs recomputar a 1k/2k/4k |
| Escalado forward | O(N) teoría (pendiente ~0–0.12 a N≤512 por overhead) | pendiente log-log **1.06** medida 256→4096 |
| Parámetros | 20.83 M (GPT-2 vocab) | **+0.5 %** con Recall Tap incluido |
| Entrenamiento del RT | — | denso O(N²d_k) o **LSH lineal** |

### 5. Resultados clave

- **KV enorme**: 100.0 % (denso) y 96.0 % (LSH lineal) a **2048 / 8192 / 16384** tokens,
  entrenando solo a 2048 — la misma cifra exacta en las tres longitudes (azar 6.25 %).
- **LM toy**: primera ENGRAMA que alcanza al transformer (5.872 vs 5.868).
- **Memoria lineal sin compresión**: 640 B/token constantes; 16k tokens ≈ 10.5 MB.
- **Cero NaN** bajo estrés (fp16 puro, entradas extremas, LR 10×, traza vacía).
- Suite: **118/118 tests** (invarianza causal en toda posición, paridad densa↔LSH,
  aislamiento de códigos, estabilidad, memoria lineal, conteo de parámetros).

### 6. Uso

```python
from engrama import EngraModelV5, V5Config

# preset (tiny|small|base|large) o config manual
model = EngraModelV5.from_preset("base", vocab_size=50257)          # RT denso
model = EngraModelV5(V5Config.from_preset("base", vocab_size=50257,
                                          context_length=32768,
                                          rt_train_mode="lsh"))      # RT lineal

loss = model.forward_loss(x[:, :-1], y[:, 1:])        # entrenamiento 100% paralelo
ids  = model.generate(prompt_ids, max_new_tokens=200) # cache nativa incremental
model.save("ckpt"); model2 = EngraModelV5.load("ckpt")
```

Benchmarks reproducibles: `benchmarks/analysis_lab/`
(`v5_kv_longcontext.py`, `v5_lm_toy.py`, `v5_speed_memory.py`, `v5_lsh_speed.py`).

### 7. Límites honestos

- El 100/96 % es en la **tarea sintética entrenada** (protocolo del benchmark del repo);
  falta validación a escala real (TinyStories 20M+) — ese es el siguiente experimento.
- El paso LSH en CPU paga una constante grande (gathers): en CPU usa denso; el modo LSH
  brilla en GPU y contextos ≥16k.
- El kernel Triton está escrito y revisado pero su **validación ejecutable requiere GPU**
  (`validate_kernel()`; fallback automático si algo falla).
- Entrenar el RT con densos a N muy grandes sigue siendo cuadrático: usa LSH allí.

## ⚡ Novedades y Soluciones de ENGRAMA V4

### 1. Velocidad de Entrenamiento: de 30h a ~1.8h
En ENGRAMA V3, el evocador `logsumexp` generaba una matriz de logits de más de 1.6 mil millones de elementos que requería 25 fragmentos de autograd checkpointed por paso, recomputando 50 kernels pesados. En V4:
- **Evocador de Fusión Latente**: Las $M$ hipótesis se combinan en el espacio latente $\mathbb{R}^d$ antes de la proyección al vocabulario. Esto reduce el cálculo a una única multiplicación matricial $O(|V|d)$ acelerada por Tensor Cores.
- **Consolidación Pre-Proyectada**: Las proyecciones $T_{prev} V$ y $P_g(T_{prev})$ se calculan una sola vez por capa y se desplazan con vistas y padding causal contiguo, eliminando 27 asignaciones `torch.cat` por paso.
- **Precisión Mixta Nativa (AMP FP16/BF16)**: Desbloquea la potencia de los Tensor Cores en GPUs Nvidia (T4, A100, H100).

### 2. Gating Bilateral Target-Source (Dual Gating)
En V3, la compuerta de una sinapsis sólo miraba el token histórico $x_{t-p}$. En V4, la compuerta evalúa el grado de concordancia entre lo que la posición actual $t$ busca ($Q_{tgt}$) y lo que la posición pasada $t-p$ contiene ($K_{src}$):
$$\alpha_{l,p}[t] = \sigma\left( \frac{1}{\sqrt{d_g}} \langle Q_{tgt}[t], K_{src}[t-p] \rangle + Q_{tgt}[t] W_{tgt,p} + K_{src}[t-p] W_{src,p} + b_{l,p} \right)$$
*No es atención*: es una compuerta sigmoide punto a punto evaluada sobre una conexión física causal fija $p \in D_l$. Coste: estrictamente $O(1)$ por token y $O(N)$ en toda la secuencia.

### 3. Acceso Directo de Traza (Trace Tap T0)
Permite a las capas de consolidación profundas acceder directamente a la huella original $T_0[t-p]$ almacenada en la Traza Circular FIFO:
$$T_{pos,l}[t] = \sum_{p \in D_l, p \le t} \rho_{l,p} \cdot \alpha_{l,p}[t] \odot \Big( y_{ctx,l,p}[t] + \gamma_{l,p} \cdot y_{tr,l,p}[t] \Big)$$
Esto evita que los detalles finos de hechos lejanos se degraden tras atravesar múltiples capas de normalizaciones no lineales.

### 4. Jerarquía de Offsets Resonante Multi-Ruta
Sustituye el esquema diádico estricto por un conjunto superpuesto de frecuencias:
$$D_l = \{0, 1, 2^{l-1}, 2^l\} \quad (\forall l \ge 1)$$
Garantiza múltiples caminos combinatorios redundantes para cualquier distancia $\Delta$, eliminando la vulnerabilidad de ruta única de V3.

### 5. Evocador de Fusión Latente $O(|V|d)$
$$\omega(h_*) = \operatorname{softmax}(W_{fusion} h_* + b_{fusion}) \in \mathbb{R}^M$$
$$\bar{c} = \sum_{m=1}^M \omega_m(h_*) \cdot c_m \in \mathbb{R}^d \implies \text{logits} = \frac{\bar{c} E^\top}{\sqrt{d}}$$

### 6. Célula con RMSNorm y Normalización sin Centrado
$$\operatorname{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{k=1}^d x_k^2 + \epsilon}} \odot \gamma$$
A diferencia de `LayerNorm`, `RMSNorm` no resta la media del vector, preservando la dirección y polaridad de las activaciones en las rutas de identidad $\beta h$.

---

## 📦 Instalación

> ⚠️ Instala siempre desde el repositorio oficial de GitHub (el paquete en PyPI con el mismo nombre no está relacionado):

```bash
pip install git+https://github.com/bueormnew/engrama.git
```

O desde el código fuente para desarrollo local:

```bash
git clone https://github.com/bueormnew/engrama.git
cd engrama
pip install -e .
```

**Requisitos**: Python $\ge 3.9$, PyTorch $\ge 2.0$.

---

## 🚀 Modo Rápido (Quickstart en 3 líneas)

Entrena un modelo completo de ENGRAMA V4 sobre cualquier archivo de texto o string:

```python
import engrama

# Entrena sobre tus datos con el preset 'small' en GPU/CPU
run = engrama.quickstart("mi_texto.txt", size="small", epochs=10)

# Genera texto autoregresivo con muestreo estocástico
print(run.generate("Había una vez", max_new_tokens=60, temperature=0.8, top_k=40))

# Guarda modelo, configuración y tokenizador
run.save("./mi_modelo_v4")

# Recarga posterior para inferencia
run_cargado = engrama.load_quick("./mi_modelo_v4")
```

---

## 🛠️ Modo Experto y Configuración Completa

Control total sobre cada componente de la arquitectura V4:

```python
import torch
from engrama import EngramaConfig, EngramaModel, Generator, EngramaTokenizer

# Configuración de ENGRAMA V4 (~20.4M parámetros para contexto 512)
config = EngramaConfig(
    vocab_size=50257,               # Tamaño del vocabulario (ej. GPT-2 BPE)
    d_model=256,                    # Dimensión oculta principal
    d_gate=32,                      # Dimensión latente de gating
    d_ff=1024,                      # Expansión FFN en células (4 * d_model)
    num_cells=8,                    # Células C en el encoder
    num_encoder_layers=2,           # Capas de codificación aislada
    num_consolidation_layers=9,      # Capas de consolidación (L >= log2(N))
    context_length=512,             # Ventana de contexto máxima N_max
    version="v4",                   # Preset V4
    offset_mode="resonant_multirate",# Offsets resonantes {0, 1, 2^{l-1}, 2^l}
    gating_mode="dual",             # Gating bilateral target-source
    trace_tap=True,                 # Acceso directo a la traza prístina T0
    norm_type="rmsnorm",            # Normalización RMSNorm
    synapse_rank=32,                # Rango de sinapsis factorizadas r << d
    num_candidates=4,               # Candidatos en el evocador
    candidate_aggregation="latent_fusion", # Fusión latente O(|V|d)
    tie_embeddings=True,            # Pesos de embedding atados a la salida
    stable_init=True,               # Inicialización estable en transporte
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EngramaModel(config).to(device)

print(f"Parámetros totales: {model.num_parameters():,}")
print(f"Campo receptivo máximo: {config.receptive_field()['max_reach']} tokens")
```

---

## ⚡ Entrenamiento Acelerado con AMP en GPU (Kaggle / Colab)

A continuación se muestra el bucle de entrenamiento optimizado con **Automatic Mixed Precision (FP16)** y `GradScaler`, capaz de procesar **~0.18 segundos por paso** en Kaggle GPU T4:

```python
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EngramaModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

model.train()
for input_ids, target_ids in train_dataloader:
    input_ids, target_ids = input_ids.to(device), target_ids.to(device)
    optimizer.zero_grad(set_to_none=True)

    # Forward con FP16 en Tensor Cores
    with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=torch.float16):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), target_ids.view(-1))

    # Backward escalado
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

---

## 🔍 Invarianza Causal y Caché de Horizonte Mínimo

ENGRAMA garantiza por construcción que el cálculo paralelo de entrenamiento es **estrictamente idéntico a la generación incremental paso a paso**:

```python
model.eval()
prompt_tokens = torch.tensor([[10, 25, 88, 42]], device=device)

# 1. Forward completo paralelo
with torch.no_grad():
    full_logits = model(prompt_tokens)

# 2. Forward incremental con caché jerárquico de horizonte mínimo
cache = model.get_cache(N_max=64, mode="hierarchical")
step_logits = []
with torch.no_grad():
    for t in range(prompt_tokens.shape[1]):
        tok = prompt_tokens[:, t : t + 1]
        log_t, _ = model.step_forward(tok, cache, timestamp=t)
        step_logits.append(log_t)
step_logits = torch.stack(step_logits, dim=1)

# Diferencia máxima medida (< 1e-5 en float32)
diff = (full_logits - step_logits).abs().max().item()
print(f"Invarianza causal verificada: max |diff| = {diff:.2e}")
```

### Reducción de Memoria del Caché Jerárquico:
En lugar de almacenar $L \times N_{max}$ estados (como en Transformers con KV-cache o ENGRAMA V2), ENGRAMA V4 retiene únicamente $\max(D_{l+1}) + 1$ estados por capa. Para $N_{max}=512$ y $L=9$, la memoria de caché pasa de **$4608 \cdot d$ a solo $518 \cdot d$ (una reducción de $8.9\times$)**.

---

## 📊 Benchmarks y Reportes

### 1. Rendimiento de entrenamiento (TinyStories 20M, GPT-2 Vocab 50k, Seq 512)

Los números publicados anteriormente para `nn.DataParallel` eran estimaciones
y no una medición reproducible; no deben usarse como benchmark. Esta versión
incluye `benchmarks/training_throughput.py` para medir baseline/optimizado con
warm-up, sincronización CUDA, tokens/s y pico real de VRAM:

```bash
python benchmarks/training_throughput.py --profile baseline --steps 100
python benchmarks/training_throughput.py --profile optimized --steps 100
python benchmarks/training_throughput.py --profile checkpoint --steps 100
```

En 2 GPUs debe medirse el trainer DDP descrito abajo. No se promete una cifra
fija: Kaggle, la versión de PyTorch/CUDA, el autotuning y el tamaño de chunk
cambian sustancialmente el resultado.

### Entrenamiento optimizado en 2× GPU (DDP, sin cambiar la arquitectura)

Para vocabulario GPT-2 y contexto 512 se recomienda reemplazar
`nn.DataParallel` por el trainer DDP incluido. Mantiene una réplica persistente
por GPU, distribuye los datos sin duplicarlos, fusiona la proyección lineal con
CE por posiciones y habilita `torch.compile` + AdamW fusionado:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/train_tinystories_ddp.py \
  --train tinystories_train.ids \
  --valid tinystories_valid.ids \
  --output /kaggle/working/engrama_v4_20m_gpt2 \
  --batch-size 16 --resume
```

El batch indicado es por GPU (batch global 32). La arquitectura y los
checkpoints no cambian. Diagnóstico, perfiles velocidad/VRAM y metodología de
medición: **[docs/OPTIMIZACION_ENTRENAMIENTO.md](docs/OPTIMIZACION_ENTRENAMIENTO.md)**.

### 2. Benchmark de Recuperación Clave-Valor de Largo Alcance (`benchmarks/kv_retrieval.py`)
Secuencias de 192 tokens con pares clave-valor aleatorios y consultas tardías a distancias de 24 a 176 tokens:
- **V3 Diádico**: 7.4% (nivel azar 6.2%).
- **V4 Resonante + Trace Tap + Dual Gating**: Supera con creces a V3 al preservar la huella prístina e incorporar gating bilateral.

---

## 📂 Estructura del Repositorio

```text
engrama/
├── src/engrama/
│   ├── __init__.py           # Exportaciones públicas de la librería
│   ├── config.py             # EngramaConfig con presets V1, V2, V3, V4
│   ├── primitives.py         # RMSNorm, LayerNorm, Cell, SynapseLayer vectorizado
│   ├── encoder.py            # IsolatedEncoder (Fase 1: Huella aislada)
│   ├── trace.py              # CircularTrace FIFO y EngramaCache (Fase 2)
│   ├── consolidation.py      # ConsolidationStack con Dual Gating y Trace Tap (Fase 3)
│   ├── evoker.py             # MultiCandidateEvoker con Latent Fusion (Fase 4)
│   ├── model.py              # EngramaModel (Integrador de las 4 fases)
│   ├── trainer.py            # Trainer de alto nivel con soporte AMP
│   ├── inference.py          # Generador autoregresivo con muestreo
│   ├── tokenizer.py          # Tokenizador de caracteres y adaptador BPE
│   ├── losses.py             # CE streaming y linear+CE sin logits globales
│   ├── optimization.py       # DDP, compile y AdamW fusionado
│   └── quick.py              # Quickstart API en 3 líneas
├── kaggle/
│   ├── engrama_v4_vs_ablation_transformer_2xt4.ipynb  # Comparación 2×T4: V4 vs ablaciones vs Transformer
│   ├── train_compare_ddp.py                           # Worker DDP del notebook de comparación
│   ├── engrama_v4_20m_tinystories_gpt2.ipynb  # Notebook oficial de entrenamiento V4
│   └── engrama_v3_20m_tinystories_gpt2.ipynb  # Notebook V3
├── benchmarks/
│   ├── kv_retrieval.py       # Benchmark de recuperación clave-valor
│   └── KV_RETRIEVAL_REPORT.md# Reporte de resultados
├── tests/                    # Suite de 85 tests unitarios y de arquitectura
├── ENGRAMA-V4-Teorica.md     # Especificación matemática formal de V4
├── ENGRAMA-V3-Teorica.md     # Especificación histórica V3
├── README.md                 # Este documento
└── pyproject.toml            # Configuración de paquete y dependencias
```

---

## 📄 Cómo Citar y Licencia

Si utilizas ENGRAMA o te basas en sus principios teóricos en tu investigación, por favor cita el proyecto:

```bibtex
@software{engrama2026,
  author = {Buenahora Ormaza, Gerson Fabian},
  title = {ENGRAMA: Arquitectura Neuronal Autorregresiva sin Atención con Memoria Explícita y Consolidación Causal},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bueormnew/engrama}}
}
```

Distribuido bajo licencia **GNU Affero General Public License v3.0 (AGPL-3.0)**. Ver [`LICENSE`](LICENSE) para más detalles.
