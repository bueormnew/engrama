# ENGRAMA 🧠⚡

**ENGRAMA** es una arquitectura neuronal autorregresiva **sin atención** — sin $QK^T$, sin matrices de afinidad $N \times N$, sin softmax sobre la dimensión de secuencia — implementada como librería en **PyTorch puro**. Esta versión implementa fielmente la especificación **ENGRAMA V3** ([`ENGRAMA-V3-Teorica.md`](ENGRAMA-V3-Teorica.md)).

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](#instalación)
[![PyTorch ≥ 2.0](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-78%20passing-brightgreen)](docs/VERIFICACION.md)
[![CI](https://github.com/bueormnew/engrama/actions/workflows/ci.yml/badge.svg)](https://github.com/bueormnew/engrama/actions/workflows/ci.yml)
[![Sin atención](https://img.shields.io/badge/attention-zero-important)](#qué-es-engrama-v3)

- **Autor**: Gerson Fabian Buenahora Ormaza (BUEORM)
- **Año**: 2026
- **Licencia**: GNU Affero General Public License v3.0 (AGPL-3.0)
- **Versión**: 0.3.0 (arquitectura V3)

---

## 📑 Índice

- [Qué es ENGRAMA V3](#-qué-es-engrama-v3)
- [Documentación](#-documentación)
- [Instalación](#-instalación)
- [Modo rápido — de texto a modelo entrenado en minutos](#-modo-rápido--de-texto-a-modelo-entrenado-en-minutos)
- [Modo experto — control total y ablaciones](#-modo-experto--control-total-y-ablaciones)
- [Interfaces de datos: inputs y outputs](#-interfaces-de-datos-inputs-y-outputs)
- [CLI](#-cli)
- [Verificación](#-verificación)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Limitaciones conocidas](#-limitaciones-conocidas)
- [Roadmap](#-roadmap)
- [Contribución](#-contribución)
- [Cómo citar](#-cómo-citar)
- [Licencia](#-licencia)

---

## 🌟 Qué es ENGRAMA V3

ENGRAMA reemplaza la atención por un pipeline de 4 fases con coste $O(N)$ en entrenamiento e incremental en inferencia:

| Fase | Componente | Qué hace |
|---|---|---|
| 1 | `IsolatedEncoder` | Codifica cada token de forma **aislada y paralela** (células + sinapsis de enrutamiento). Cero fuga de contexto en esta etapa. |
| 2 | `CircularTrace` | Traza circular FIFO que **solo almacena** las huellas $T_0$ con timestamp exacto; nunca las transforma. |
| 3 | `ConsolidationStack` | Mezcla causal relativa por offsets diádicos jerárquicos $D_l = \{0, 1, 2^l\}$ con **sinapsis factorizadas compartidas** $W_{a \to b} = \beta_{a \to b} \cdot I + U_l \, \mathrm{Diag}(s_{a \to b}) \, V_l^\top$ (transporte identidad) y compuerta escalar por escala $\rho_{l,p}$. |
| 4 | `MultiCandidateEvoker` | Hasta 8 candidatos factorizados sobre un núcleo $W_{shared}$ común, agregados por `logsumexp` / `max` / `mean`. |

Propiedades clave:

1. **Cero atención dinámica**: la interacción temporal se realiza con pesos estáticos por offset relativo. El único softmax del modelo es el de la distribución final sobre el vocabulario.
2. **Invarianza causal garantizada y testeada**: el forward paralelo de entrenamiento coincide con la inferencia incremental token a token con error máximo medido de **5.96e-07** (float32), en una matriz de 17 combinaciones de modos × 2 modos de caché.
3. **Caché jerárquica de horizonte mínimo** (teorema §24): en inferencia solo se retienen `max(D_{l+1}) + 1` estados por capa. Ejemplo real con $N_{max}=256$: `[3, 5, 9, 17, 33, 65, 129, 1]` estados por capa (262 en total) frente a 256 × 8 = 2048 del caché completo → **7.8× menos estado**.
4. **Menos parámetros que V2 con los mismos hiperparámetros**: 6.89M (V3) frente a 29.55M (V2) en la configuración de referencia `d_model=256, C=8, L=8`, gracias a las sinapsis factorizadas compartidas por capa.
5. **Dos modos de uso**: *modo rápido* (un preset de tamaño y tus datos → modelo entrenado) y *modo experto* (control total de la arquitectura y ablaciones V2/V3 desde `EngramaConfig`).
6. **Ecosistema completo**: entrenador con schedulers, generación con top-k/top-p/streaming, serialización, inspección de sinapsis, suite de benchmarks, CLI y suite de tests.

> **Compatibilidad V1/V2**: los presets `version="v1"` y `version="v2"` comparten la misma parametrización densa (la librería implementa el algoritmo cacheado de V2 para ambos; V1 se conserva por proveniencia). `version="v3"` resuelve todos los modos a su forma factorizada jerárquica. Cualquier modo explícito sobrescribe al preset.

---

## 📚 Documentación

| Documento | Contenido |
|---|---|
| [`docs/guia_uso_completa.md`](docs/guia_uso_completa.md) | Guía completa de uso: cada parámetro, flujos de entrenamiento, inferencia, serialización e inspección. |
| [`docs/VERIFICACION.md`](docs/VERIFICACION.md) | Qué está verificado, cómo y con qué resultados medidos. |
| [`ENGRAMA-V3-Teorica.md`](ENGRAMA-V3-Teorica.md) | Especificación teórica V3 implementada (56 secciones). |
| [`ENGRAMA-Paper-Final-Verificado.md`](ENGRAMA-Paper-Final-Verificado.md) | Paper V1+V2 (referencia histórica). |
| [`benchmarks/KV_RETRIEVAL_REPORT.md`](benchmarks/KV_RETRIEVAL_REPORT.md) | Benchmark de recuperación clave–valor de largo alcance (números reales). |
| [`CHANGELOG.md`](CHANGELOG.md) | Historial de versiones. |

---

## 📦 Instalación

> ⚠️ El nombre `engrama` en PyPI pertenece a un paquete **no relacionado**. Instala siempre desde este repositorio:

```bash
pip install git+https://github.com/bueormnew/engrama.git
```

O desde el código fuente (desarrollo):

```bash
git clone https://github.com/bueormnew/engrama.git
cd engrama
pip install -e .
```

**Requisitos**: Python ≥ 3.9. Única dependencia: `torch >= 2.0`.

---

## 🚀 Modo rápido — de texto a modelo entrenado en minutos

```python
import engrama

# Entrena sobre un archivo de texto (o un string) con un preset de tamaño
run = engrama.quickstart("corpus.txt", size="small", epochs=10)

print(run.summary())
print(run.generate("Once upon a time", max_new_tokens=50, temperature=0.8, top_k=40))

run.save("./mi_modelo")                 # modelo + config + tokenizador
run2 = engrama.load_quick("./mi_modelo")  # carga posterior
```

Salida real (corpus de juguete de 1 KB, preset `tiny`, CPU, 30 épocas):

```
[ENGRAMA quickstart] size=tiny params=277,586 device=cpu samples=18 seq_len=64 lr=0.005 epochs=30
[ENGRAMA quickstart] done: loss 3.2778 -> 2.6491
ENGRAMA V3 | params=277,586 | d_model=64 C=2 L=6 N_max=64 | final_loss=2.6491
```

> Con 120 épocas sobre ese mismo corpus el modelo converge a loss 0.051 y reproduce el texto entrenado verbatim — es un corpus de juguete; con datos reales (p. ej. TinyStories, ver `kaggle/`) el modelo aprende estructura lingüística real.

### Crear solo el modelo (sin entrenar)

```python
import engrama

model = engrama.create_model(size="small", vocab_size=256)
print(engrama.list_sizes())   # presets disponibles
```

| Preset | d_model | Células | L_enc | L | N_max | rango r |
|---|---|---|---|---|---|---|
| `tiny`  | 64  | 2  | 1 | 6  | 64   | 8  |
| `small` | 128 | 4  | 1 | 8  | 256  | 16 |
| `base`  | 256 | 8  | 2 | 8  | 256  | 32 |
| `large` | 512 | 16 | 2 | 11 | 2048 | 32 |

---

## 🛠️ Modo experto — control total y ablaciones

`EngramaConfig` expone cada modo de la arquitectura. El campo `version` es un **preset arquitectónico real**: `v1`/`v2` resuelven todos los modos a su forma densa y `v3` a su forma factorizada jerárquica; cualquier modo explícito **sobrescribe** al preset, lo que habilita las suites de ablación de la especificación V3 (secciones 43–44) directamente:

```python
import torch
from engrama import EngramaConfig, EngramaModel

cfg = EngramaConfig(
    vocab_size=128,
    d_model=256,
    num_cells=8,
    context_length=256,
    version="v3",                      # preset base
    offset_mode="hierarchical_dyadic", # D_l = {0, 1, 2^l}
    cache_mode="hierarchical",         # caché de horizonte mínimo
    num_candidates=4,
    candidate_aggregation="logsumexp",
    global_anchor=False,               # ancla global g(N) (§11)
    stable_init=True,                  # init estable s≈0, β=1 (§32)
)
model = EngramaModel(cfg)
print(f"{model.num_parameters():,}")   # 6,889,102

# Inspección de la conectividad por capa
print(cfg.get_layer_offsets(3))        # [0, 1, 8]
print(cfg.cache_horizons())            # [3, 5, 9, 17, 33, 65, 129, 1]
print(cfg.receptive_field()["max_reach"])  # 255

# Ablación: misma escala, arquitectura V2 densa
cfg_v2 = EngramaConfig(vocab_size=128, d_model=256, num_cells=8,
                       context_length=256, version="v2", num_candidates=4)
print(f"{EngramaModel(cfg_v2).num_parameters():,}")  # 29,549,952
```

### Parámetros principales de `EngramaConfig`

| Parámetro | Default | Descripción |
|---|---|---|
| `vocab_size` | 256 | Tamaño del vocabulario |
| `d_model` | 256 | Dimensión oculta $d$ |
| `d_gate` | 32 | Dimensión latente de compuertas $d_g \ll d$ |
| `num_cells` | 8 | Células $C$ por capa del encoder |
| `num_encoder_layers` | 2 | Capas de sinapsis del encoder |
| `num_consolidation_layers` | 8 | Capas de consolidación $L$ (regla: $L \ge \lceil \log_2 N \rceil$) |
| `context_length` | 256 | Ventana de la traza $N_{max}$ |
| `num_candidates` | 4 | Candidatos del evocador $M \in [1, 8]$ |
| `synapse_rank` | 32 | Rango $r$ de las sinapsis factorizadas |
| `version` | `"v3"` | Preset: `"v1"` \| `"v2"` \| `"v3"` |
| `synapse_mode` | preset | `"dense"` (V2) \| `"factorized"` (V3 §6) |
| `cell_mode` | preset | `"independent"` \| `"shared_core"` (V3 §5) |
| `offset_mode` | preset | `"dense_dilated"` \| `"hierarchical_dyadic"` \| `"binary_minimal"` |
| `cache_mode` | preset | `"full"` \| `"hierarchical"` (V3 §12) |
| `evoker_mode` | preset | `"dense"` \| `"factorized"` (V3 §14) |
| `identity_transport` | preset | Ruta identidad $\beta \cdot h$ (V3 §6.4) |
| `hierarchical_gate` | preset | Compuerta escalar $\rho_{l,p}$ (V3 §17) |
| `global_anchor` | `False` | Ancla determinista $g(N)$ en la última capa (V3 §11) |
| `tie_embeddings` | `True` | Cabeza de salida atada a los embeddings |
| `dtype` | `"float32"` | Precisión del modelo (`float32/64/16`, `bfloat16`) |

### Entrenamiento explícito (modo experto)

```python
from engrama import EngramaTokenizer, TextDataset, Trainer

with open("corpus.txt", encoding="utf-8") as f:
    texto = f.read()

tokenizer = EngramaTokenizer().fit_on_text(texto)
dataset = TextDataset(texto, tokenizer, sequence_length=128)

trainer = Trainer(model, lr=1e-3, scheduler="cosine", warmup_steps=200,
                  device="cuda" if torch.cuda.is_available() else "cpu")
history = trainer.fit(dataset, batch_size=16, epochs=10)
```

### Generación

```python
from engrama import Generator

gen = Generator(model, tokenizer)
texto = gen.generate("El gato", max_new_tokens=100,
                     temperature=0.8, top_k=40, top_p=0.95)
for chunk in gen.generate_stream("El gato", max_new_tokens=100):
    print(chunk, end="", flush=True)
```

### Invarianza causal verificable en 5 líneas

```python
x = torch.randint(0, cfg.vocab_size, (2, 32))
cache = model.get_cache(N_max=32)
steps = [model.step_forward(x[:, t:t+1], cache, t)[0] for t in range(32)]
inc = torch.stack(steps, dim=1)
diff = (model(x) - inc).abs().max().item()   # float32: ~5.96e-07
```

---

## 🔄 Interfaces de datos: inputs y outputs

Contratos estables de la librería (útiles para integrar ENGRAMA en pipelines propios):

| API | Input | Output |
|---|---|---|
| `model.forward(input_ids)` | `(B, N)` int64, ids ∈ `[0, vocab_size)` | Logits `(B, N, vocab_size)` |
| `model.step_forward(token_id, cache, t)` | `token_id` `(B,1)`/`(B,)`/escalar; `cache`; `t` absoluto | `(logits_t (B,V), hidden_t (B,d))` |
| `model.generate(prompt_ids, ...)` | `List[int]` (vacío ⇒ `DEFAULT_BOS_TOKEN_ID`) | `List[int]` prompt + generados |
| `model.generate_stream(...)` | ídem + `use_cache` | generador de ids uno a uno |
| `Generator.generate(prompt, ...)` | `str` | `str` (prompt + completado, especiales omitidos) |
| `TextDataset[índice]` | — | `{"input_ids": (S,), "target_ids": (S,)}` |
| `Trainer.fit(dataset, batch_size, epochs)` | `Dataset` con pares input/target | `List[float]` pérdida por época |
| `chunked_cross_entropy(logits, targets)` | `(..., V)` / `(...)` | CE por trozos de vocabulario (memoria O(N·chunk), equivalente a `F.cross_entropy`) |

Detalles de contrato:

- **Padding y pérdida**: `TextDataset` marca las posiciones de padding con `IGNORE_INDEX = -100` en `target_ids`; `F.cross_entropy` las ignora por defecto, así el padding nunca contamina el loss.
- **Tokens fuera de rango**: `generate`/`generate_stream` lanzan `ValueError` si el prompt contiene ids fuera de `[0, vocab_size)`.
- **Modo del modelo**: `Trainer.evaluate` y `BenchmarkSuite` restauran el modo train/eval previo del modelo tras ejecutarse.
- **Config estricta-avisadora**: `EngramaConfig.from_dict` ignora claves desconocidas (compatibilidad hacia adelante) pero emite un warning descriptivo.

---

## 💻 CLI

```bash
engrama sizes      # presets de tamaño
engrama info       # versión, autor, torch, dispositivo
engrama train --text-file corpus.txt --size small --epochs 10 \
              --scheduler cosine --output-dir checkpoints/mi_modelo
engrama generate --model-dir checkpoints/mi_modelo --prompt "Hola" --stream
engrama generate --model-dir checkpoints/mi_modelo --prompt "Hola" --no-cache
engrama evaluate --model-dir checkpoints/mi_modelo --text-file valid.txt
engrama inspect --model-dir checkpoints/mi_modelo --sample-text "Hola"
engrama benchmark --size small --seq-len 256 --runs 10
```

---

## ✅ Verificación

Todo número de este repositorio proviene de ejecuciones reales (política de honestidad §56 de la especificación). Detalle completo en [`docs/VERIFICACION.md`](docs/VERIFICACION.md):

- **78 tests** (`python -m unittest discover` dentro de `tests/`, ~14 s en CPU): presets de versión, validación de config, primitivas (sinapsis factorizadas, compuerta desde la fuente, transporte identidad exacto), encoder/traza, matriz de invarianza causal de 17 modos × 2 cachés, evocador, tokenizador/dataset y ecosistema de entrenamiento.
- **Invarianza causal**: error máximo 5.96e-07 (float32) entre forward paralelo e incremental, incluido el régimen de desborde FIFO cuando el cono de dependencias cabe en la ventana retenida.
- **Transporte identidad (§31)**: con $s = 0, \beta = 1$ la sinapsis transporta el vector *exactamente* (error 0.0).
- **Benchmark de recuperación clave–valor de largo alcance**: ver [`benchmarks/KV_RETRIEVAL_REPORT.md`](benchmarks/KV_RETRIEVAL_REPORT.md) (script: `benchmarks/kv_retrieval.py`). Tarea sintética con bindings aleatorios por muestra (imposible memorizar); compara `hierarchical_dyadic` vs `dense_dilated` a distancias de hasta ~176 tokens. Los resultados son honestos: confirman el riesgo principal señalado por la propia especificación (§42) y motivan la ablación con ancla global.

```bash
cd tests && python -m unittest discover -q   # 78 tests
python benchmarks/kv_retrieval.py --steps 600 --seed 1234 --force
```

> El benchmark escribe su reporte en `benchmarks/KV_RETRIEVAL_REPORT.md` solo con `--force`; sin él genera un reporte con sufijo de ejecución y nunca sobrescribe los números publicados.

---

## 📁 Estructura del repositorio

```
engrama/
├── src/engrama/            # la librería
│   ├── config.py           # EngramaConfig + presets v1/v2/v3 + presets de tamaño
│   ├── primitives.py       # células, sinapsis (densa/factorizada), shared-core
│   ├── encoder.py          # Fase 1: codificación aislada
│   ├── trace.py            # Fase 2: traza circular + caché jerárquico
│   ├── consolidation.py    # Fase 3: mezcla diádica relativa + sinapsis factorizadas
│   ├── evoker.py           # Fase 4: evocador multi-candidato
│   ├── model.py            # EngramaModel (forward, step_forward, generate)
│   ├── trainer.py          # Trainer (AdamW, clipping, warmup/cosine)
│   ├── inference.py        # Generator (top-k/top-p/streaming)
│   ├── tokenizer.py        # tokenizador de caracteres
│   ├── datasets.py         # TextDataset autorregresivo (padding ignorado en el loss)
│   ├── serialization.py    # save/load (modelo + config + tokenizador)
│   ├── inspection.py       # EngramaInspector (fidelidad de sinapsis §50)
│   ├── benchmarks.py       # BenchmarkSuite de latencia/estado de caché
│   ├── quick.py            # modo rápido: quickstart / create_model / QuickRun
│   └── cli.py              # CLI `engrama`
├── tests/                  # 78 tests (unittest)
├── benchmarks/             # benchmark KV de largo alcance + reporte real
├── examples/               # ejemplos ejecutables (01, 02, 03)
├── docs/                   # guía completa + verificación
├── kaggle/                 # notebooks TinyStories (char y ~20M con GPT-2, GPU recomendada;
│                           #  descarga verificada, tokenización streaming, CE por trozos)
├── .github/workflows/      # CI (Python 3.9–3.12, torch CPU)
├── ENGRAMA-V3-Teorica.md   # especificación V3 implementada
├── ENGRAMA-Paper-Final-Verificado.md  # paper V1+V2 (referencia)
├── CHANGELOG.md            # historial de versiones
├── CONTRIBUTING.md         # guía de contribución
└── README.md
```

---

## ⚠️ Limitaciones conocidas

Transparencia sobre lo que ENGRAMA V3 **no** promete (spec §41–42):

- **Recuperación de largo alcance limitada**: en la tarea clave–valor sintética, `hierarchical_dyadic` obtiene ~7–9% de acierto (frente a ~27% de `dense_dilated` y 6.2% del azar). La jerarquía diádica es barata pero diluye señales lejanas; el transporte de identidad mitiga, no elimina, el problema.
- **No hay evidencia de equivalencia con atención en lenguaje natural**: el benchmark sintético no demuestra que ENGRAMA iguale a transformers en tareas de contexto largo reales (spec §41).
- **Tokenizador de caracteres**: el tokenizador incluido es char-level; para vocabularios grandes (BPE/WordPiece) hay que aportar uno externo.
- **Sin atención, también sin sus ventajas**: no hay selección dinámica de posiciones; la capacidad de "foco" depende de los pesos estáticos aprendidos.
- **Entrenamiento en GPU**: los notebooks de `kaggle/` con ~20M de parámetros requieren GPU; en CPU use presets `tiny`/`small`.

---

## 🗺️ Roadmap

Ideas priorizadas (contribuciones bienvenidas, ver [CONTRIBUTING.md](CONTRIBUTING.md)):

1. **Ablación V3 completa (§43–44)**: suite automatizada que reporte parámetros, FLOPs, memoria, tokens/s, perplexity y recuperación exacta para las 11 ablaciones obligatorias.
2. **Ancla global adaptativa**: explorar variantes de $g(N)$ que reduzcan la brecha de recuperación de largo alcance medida en el benchmark KV.
3. **Tokenizadores subpalabra**: BPE/WordPiece opcionales (sin dependencias nuevas).
4. **Entrenamiento distribuido**: DDP y precisiones `bfloat16`/`float16` validadas a escala (el soporte de dtype ya existe en `EngramaConfig`).
5. **Modelos preentrenados**: publicar checkpoints TinyStories (char y ~20M) entrenados con los notebooks de `kaggle/`.
6. **Benchmarks en lenguaje natural**: perplexity y long-context accuracy en TinyStories/PG-19 frente a un baseline transformer del mismo tamaño.

---

## 🤝 Contribución

Lee [`CONTRIBUTING.md`](CONTRIBUTING.md) antes de abrir un issue o PR. En resumen:

- Los 78 tests deben quedar en verde (`cd tests && python -m unittest discover -q`).
- Todo número publicado debe venir de una **ejecución real** (política de honestidad §56).
- El código debe seguir siendo PyTorch puro, sin atención y sin dependencias nuevas.

Issues y PRs: <https://github.com/bueormnew/engrama/issues>

---

## 📖 Cómo citar

Si usas ENGRAMA en tu investigación:

```bibtex
@misc{buenahora2026engrama,
  author       = {Buenahora Ormaza, Gerson Fabian},
  title        = {{ENGRAMA V3}: A Non-Attention Autoregressive Neural Architecture with Isolated Encoding, Circular Trace, Hierarchical Dyadic Consolidation and Factorized Synapses},
  year         = {2026},
  howpublished = {\url{https://github.com/bueormnew/engrama}},
  note         = {Licencia AGPL-3.0; especificaci\'on te\'orica en ENGRAMA-V3-Teorica.md}
}
```

---

## 📜 Licencia

GNU Affero General Public License v3.0 — ver [`LICENSE`](LICENSE).

**Autor**: Gerson Fabian Buenahora Ormaza (BUEORM) — 2026.
