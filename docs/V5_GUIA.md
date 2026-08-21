# ENGRAMA V5 — Guía de uso

ENGRAMA V5 es una arquitectura autoregresiva **sin atención** y **sin
compresión** que recupera los tokens exactos guardados en su Traza mediante
**resonancia sináptica** (una compuerta sigmoide punto a punto sobre la memoria
explícita). Ver la teoría completa en `ENGRAMA-V5-Teorica.md` y el diagnóstico
que la motivó en `docs/DIAGNOSTICO_V4.md`.

## Instalación

```bash
pip install -e .
```

## Uso mínimo

```python
import torch
from engrama import EngramaV5, EngramaV5Config

cfg = EngramaV5Config(
    vocab_size=32000,
    d_model=256,
    num_layers=6,
    num_heads=8,
    context_length=8192,
)
model = EngramaV5(cfg)

ids = torch.randint(0, 32000, (2, 128))
logits = model(ids)                        # (2, 128, 32000)
loss = model.loss(ids[:, :-1], ids[:, 1:]) # cross-entropy escalar
```

## Generación (caché nativa de traza explícita)

```python
out = model.generate(
    prompt_ids=[1, 2, 3],
    max_new_tokens=50,
    temperature=0.8,
    top_k=40,
    use_cache=True,   # caché nativa: O(N) memoria, sin recomputar tokens previos
)
```

## Entrenamiento

```python
from engrama.v5 import V5Trainer
from engrama.v5.trainer import V5TrainConfig

trainer = V5Trainer(model, V5TrainConfig(lr=3e-3, max_steps=1000, warmup_steps=100))

def batch_fn(step):
    ids = torch.randint(0, cfg.vocab_size, (16, 256))
    return ids[:, :-1], ids[:, 1:]

trainer.fit(batch_fn, steps=1000)
```

## Presets

```python
EngramaV5Config.preset("tiny")    # d=128, L=4,  H=4,  ctx=1024
EngramaV5Config.preset("small")   # d=256, L=6,  H=8,  ctx=2048
EngramaV5Config.preset("base")    # d=512, L=8,  H=8,  ctx=4096
EngramaV5Config.preset("large")   # d=768, L=12, H=12, ctx=8192
```

## Parámetros clave

| Parámetro | Qué controla | Recomendación |
|---|---|---|
| `num_heads` | Cabezas de resonancia | `d_model % num_heads == 0`; 4–12 |
| `tau_init` | Agudeza inicial de la sinapsis | **4.0** (medido óptimo; 8.0 satura y baja el recall) |
| `read_norm` | `None` (superposición Hebbiana pura) o `"softcount"` | `None` da mejor recall exacto |
| `num_encoder_layers` | Células por token antes de la resonancia | **0** (el embedding ya es la huella aislada; añadir células difumina el contenido) |
| `chunk_size` | Tiling causal para contextos enormes | `256`–`512` para N grande; `0` para N corto |

## Cómputo sub-cuadrático (block-sparse) + kernels Triton

La lectura densa es O(N²). Para contextos grandes, activa la **resonancia
block-sparse**, que reduce el cómputo drásticamente **sin comprimir nada** (los N
tokens siguen explícitos; solo se podan bloques de claves que no resuenan):

```python
cfg = EngramaV5Config(
    vocab_size=32000, d_model=512, num_layers=8, num_heads=8,
    context_length=8192,
    resonance_mode="block_sparse",   # <-- sub-cuadrático
    block_size=128,                  # tokens por bloque
    top_k=8,                         # bloques de claves visitados por bloque de queries
)
model = EngramaV5(cfg)
```

- Con `top_k` fijo el cómputo total es **O(N)**; con `top_k` creciendo lento, O(N·√N).
- Medido: **98.8% recall** (igual o mejor que denso) mirando ~25% del cómputo;
  speedup 6.8× a N=4096 en CPU (mucho mayor en GPU con Triton).
- Con `top_k ≥ nº de bloques`, el resultado es **idéntico** al denso (|Δ|≈1e-6).
- La **generación** usa siempre la lectura exacta densa sobre la traza (la caché
  no cambia), así que puedes entrenar block-sparse y generar sin diferencias.

**Kernels Triton** (`engrama.v5.triton_kernels`): `resonance_dense` y
`resonance_blocksparse`, fusionados y sin softmax. Se usan automáticamente en
CUDA; en CPU hay un fallback PyTorch numéricamente idéntico. No requieren
configuración: el modelo los invoca solo cuando el tensor está en GPU.

## Contextos enormes (8000+ tokens)

Activa el tiling causal para acotar memoria de activación sin cambiar el
resultado (verificado idéntico al forward completo):

```python
cfg = EngramaV5Config(..., context_length=8192, chunk_size=512)
```

El tiling **no usa softmax**, así que no necesita el truco de resta del máximo:
la lectura es una suma directa, numéricamente trivial y sin estado global.

## Garantías (verificadas por `tests/test_v5.py`)

- **Invarianza causal**: `forward` paralelo ≡ generación incremental (|Δ| < 1e-4).
- **Chunked ≡ full**: la ruta de contexto enorme es idéntica al forward completo.
- **Sin NaN**: forward/backward finitos en fp32 y fp16.
- **Memoria lineal**: la caché de generación crece exactamente O(N) (bytes/token
  constante), sin compresión.
- **Sin atención**: la suma de compuertas por fila NO es 1 (no hay softmax sobre
  posiciones); cada sinapsis resuena de forma independiente.
