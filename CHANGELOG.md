# Changelog

Todos los cambios notables de ENGRAMA se documentan aquí.
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.1.0/),
versionado semántico (semver).

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
