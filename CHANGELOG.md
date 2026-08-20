# Changelog

Todos los cambios notables de ENGRAMA se documentan aquí.
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.1.0/),
versionado semántico (semver).

## [No publicado] — 2026-08-20

### Corregido (NaN en TinyStories V4 + GPT-2 bajo AMP)

- `kaggle/engrama_v4_20m_tinystories_gpt2.ipynb`: el loss se iba a `nan`
  ~paso 350 (justo al terminar el warmup a `lr=6e-4`) y la muestra
  degeneraba a `'Once upon a time!!!!…'` (token GPT-2 id 0 = `!`).
  Causa del notebook, no de la arquitectura:
  - `F.cross_entropy` **dentro** de `autocast(fp16)` sobre vocabulario
    50,257: `log_softmax` fp16 desborda a Inf/NaN.
  - `clip_grad_norm_` sobre grads Inf (`inf * 0 = nan`) envenenaba Adam
    aunque GradScaler intentara saltar el paso.
  - GEMM fp16 con reducción fp16 + warmup de 200 pasos + `beta2=0.999`.
  Receta (arquitectura intacta): CE por trozos en fp32 (upcast **por
  chunk de 2048, sin clonar** el `(B,N,V)` — pico extra ~64 MiB frente
  a ~1.6 GiB de `logits.float()` o del `log_softmax` de `F.cross_entropy`),
  pasos no-finitos se saltan, GEMM con acumulador fp32, AdamW
  `betas=(0.9, 0.95)`, `lr=3e-4`, warmup 500, GradScaler `init_scale=2**12`.
  El forward AMP fp16 no cambia: el paso sigue en ~0.4 s, la CE es <10 ms.
- `losses.chunked_cross_entropy` y `Trainer`: mismos acumuladores fp32
  por trozo (el overflow no puede colarse por el Trainer, y no se duplica
  el vocabulario en VRAM).

### Corregido (memoria y notebooks Kaggle)

- `evoker.py`: la agregación `logsumexp`/`max` con vocabularios grandes
  materializaba logits `(B, N, M, V)` completos en fp32 (con GPT-2 a batch
  16×512×4 candidatos ≈ 6.6 GB por GPU, +20 GB con los temporales del
  softmax → CUDA OOM en T4 de 16 GB). Ahora el vocabulario se procesa por
  trozos con gradient checkpointing: pico medido 1.92 GiB (batch 4, seq
  512, vocab 50,257) frente a 3.10 GiB del path antiguo a la mitad de
  batch; resultados matemáticamente idénticos (verificado ≤1e-5).
- Nueva `engrama.losses.chunked_cross_entropy`: misma cross-entropy que
  `F.cross_entropy` (valores, `ignore_index`, reducciones y gradientes
  verificados ≤1e-5) sin materializar la segunda copia de logits del
  `log_softmax`. `Trainer` la usa automáticamente con vocabularios > 16k.
- `kaggle/engrama_v3_20m_tinystories_gpt2.ipynb` reescrito:
  - `FAST_MODE = False` por defecto (antes `True` construía en silencio un
    modelo de 7.9M en lugar del anunciado de ~20.3M); el notebook ahora
    aborta si el modelo no tiene los ~20M esperados o si el tokenizer
    GPT-2 no carga en modo FULL.
  - Descarga robusta de TinyStories: reintentos con backoff, reanudación
    con `Range`, verificación de tamaño exacto (train 2,227,753,162 B;
    valid 22,502,601 B) y sin fallbacks silenciosos: si el dataset no
    puede obtenerse completo, el notebook aborta con instrucciones.
  - Tokenización en streaming a `np.memmap` (antes acumulaba ~190M ids en
    listas de Python, ~5 GB de RAM); el dataset son vistas sobre el
    memmap, sin copias.
  - Entrenamiento con pérdida por GPU (DataParallel solo reúne escalares)
    y evaluación con CE por trozos; `RESUME=True` reanuda desde el último
    paso con el mismo orden de datos.
- `kaggle/engrama_v3_tinystories.ipynb`: corregidas las celdas rotas
  (variables `size`/`n_epochs` sin definir, `text` podía quedar `None`
  offline) y descarga con verificación de tamaño.

### Tests

- Suite ampliada de 68 a **78 tests** (nuevos `test_memory_safe_paths.py`:
  equivalencia CE por trozos, path chunked vs plain del evocador,
  gradientes, invarianza causal bajo el path chunked, Trainer con
  vocabulario grande).

## [0.3.0] - 2026-08-20

### Arquitectura V3 (núcleo)

- Implementación completa de la especificación `ENGRAMA-V3-Teorica.md`:
  codificación aislada, traza circular FIFO, consolidación jerárquica
  diádica con sinapsis factorizadas y transporte de identidad, caché de
  horizonte mínimo, y evocador multi-candidato factorizado.
- Presets de versión reales `v1` / `v2` / `v3` con overrides por modo
  (habilita las suites de ablación de la spec, secciones 43–44).
- Invarianza causal verificada: `forward` paralelo ≡ `step_forward`
  incremental (error máximo ~5.96e-07 en float32, 17 modos × 2 cachés).
- Modo rápido (`quickstart`) y modo experto (`EngramaConfig`), CLI,
  serialización, inspección, benchmarks y 68 tests.

### Corregido en esta versión

- `LICENSE`: fecha del texto AGPL-3.0 corregida (2007; era 2027).
- README y `docs/VERIFICACION.md`: total de estados de caché corregido
  (262; era 235) para los horizontes `[3, 5, 9, 17, 33, 65, 129, 1]`.
- `model.py`: el prompt vacío usa la constante `DEFAULT_BOS_TOKEN_ID` en
  lugar del id hardcodeado `2`; `generate`/`generate_stream` validan que los
  token ids del prompt estén dentro del vocabulario.
- `model.generate_stream` / `Generator.generate_stream` / CLI `--stream`:
  ahora respetan `use_cache` / `--no-cache` (antes el streaming siempre
  usaba caché).
- `datasets.py`: las posiciones de padding usan `IGNORE_INDEX = -100` en
  `target_ids`, de modo que el padding nunca contamina la pérdida de
  entrenamiento ni la de evaluación.
- `trainer.py` y `benchmarks.py`: `evaluate` y los benchmarks restauran el
  modo train/eval del modelo tras ejecutarse.
- `config.py`: `EngramaConfig.from_dict` emite un warning descriptivo ante
  claves desconocidas en lugar de ignorarlas en silencio.
- `benchmarks/kv_retrieval.py`: el reporte versionado ya no se sobrescribe
  por accidente; se escribe un reporte con sufijo de ejecución salvo
  `--force`.
- `pyproject.toml`: `requires-python >= 3.9` y autor con nombre completo.

### Documentación

- README reescrito: badges, índice, secciones de limitaciones, roadmap,
  contribución y cómo citar; enlaces a `docs/` y a la especificación.
- Nuevos `CONTRIBUTING.md` y `CHANGELOG.md`.
- CI: workflow de GitHub Actions (Python 3.9–3.12, torch CPU, tests y
  empaquetado).

## [0.1.0] / [0.2.0] — Implementaciones V1 / V2

- Implementaciones previas de las arquitecturas V1 y V2 (paper de
  referencia: `ENGRAMA-Paper-Final-Verificado.md`). La versión 0.3.0 las
  conserva como presets de compatibilidad (`version="v1" | "v2"`).
- Sin changelog individual publicado.
