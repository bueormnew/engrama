# Entrenamiento de alto rendimiento (ENGRAMA V4)

Esta guía optimiza **la ejecución**, no la arquitectura. Los parámetros, las
ecuaciones, el campo receptivo y los checkpoints siguen siendo compatibles.

## Diagnóstico del paso de ~0.92 s en 2× T4

El 98 % de utilización no significa por sí mismo que el flujo sea eficiente:
solo indica que siempre hay trabajo CUDA. En el notebook anterior coexistían
varios costes evitables:

1. `nn.DataParallel` replicaba el modelo desde GPU 0 en cada forward, hacía
   scatter/gather en un único proceso y reducía los gradientes hacia GPU 0.
2. La CE recorría 25 trozos del vocabulario. `if hit.any()` sincronizaba CPU y
   GPU en cada trozo; autograd retenía todos los `exp`, máscaras y upcasts.
3. `_grads_finite()` recorría parámetro por parámetro y cada condición booleana
   obligaba a sincronizar con CUDA. `GradScaler` ya realiza este chequeo con
   operaciones `foreach`.
4. Cada offset de cada capa de consolidación ejecutaba su propio conjunto de
   pads, matmuls, sumas y sigmoides pequeños.
5. El loader copiaba de forma bloqueante y no mantenía workers persistentes.
6. No se usaban `torch.compile` ni AdamW fusionado.

La VRAM al 50 % no es un problema: DDP mantiene deliberadamente una réplica en
cada GPU. Lo importante es que ambas procesen lotes distintos y solo comuniquen
buckets de gradientes.

Un 20M a batch 16×512 en T4 **no está limitado por FLOPs**: está limitado por
lanzamientos de kernel y por sincronizaciones CPU. En el comparador 4-modelos
eso se veía como ~0.27 s/paso y ~60k tok/s con la GPU “poco ocupada”. La receta
correcta (sin cambiar ecuaciones ni batch, para no romper un run ya entrenado):

1. No llamar `.item()` / `torch.isfinite(loss)` en cada step (GradScaler ya
   descarta updates no finitos).
2. `linear_chunk_size >= local_batch * seq_len` para un solo GEMM de CE.
3. `torch.compile(..., mode="reduce-overhead")` (CUDA graphs) en formas
   estáticas 16×512. `max-autotune` avisa `Not enough SMs` en T4 (40 SMs).

## Ruta recomendada: DDP

Tras crear los ficheros `tinystories_train.ids` y `tinystories_valid.ids` del
notebook (arrays planos `int32`), ejecutar desde la raíz del repositorio:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/train_tinystories_ddp.py \
  --train tinystories_train.ids \
  --valid tinystories_valid.ids \
  --output /kaggle/working/engrama_v4_20m_gpt2 \
  --batch-size 16 \
  --resume
```

Si ENGRAMA fue instalado con `pip` y no está disponible el checkout del repo
(común en Kaggle), el mismo trainer se lanza como módulo instalado:

```bash
torchrun --standalone --nproc_per_node=2 -m engrama.ddp_train \
  --train tinystories_train.ids --valid tinystories_valid.ids \
  --output /kaggle/working/engrama_v4_20m_gpt2 --batch-size 16 --resume
```

`--batch-size` es **por GPU**, por lo que 16 en 2 GPUs conserva el batch global
32 del notebook. Cada proceso recibe un shard diferente mediante
`DistributedSampler`; no hay duplicación de muestras.

La primera iteración con `torch.compile --compile-mode max-autotune` tarda más
por compilación. Para comparar rendimiento se debe descartar el warm-up y medir
al menos 100 pasos. Para una prueba corta puede usarse `--compile-mode default`.

### Perfiles de memoria/velocidad de la salida de 50,257 tokens

```bash
# Prioriza rendimiento. No crea logits B×N×V completos, pero autograd conserva
# los chunks de CE necesarios para backward.
... --linear-chunk-size 2048

# Prioriza VRAM mínima. Recalcula el vocab projection durante backward.
... --linear-chunk-size 2048 --checkpoint-loss
```

Aumentar `--linear-chunk-size` reduce lanzamientos y suele acelerar si cabe en
VRAM. Probar 2048 y 4096 en T4. No se debe elegir por intuición: medir
`max_memory_allocated` y tokens/s.

## API reutilizable

```python
from engrama import (
    EngramaModel, LanguageModelLoss, adamw, compile_model,
    init_distributed, wrap_ddp,
)

ctx = init_distributed()                    # lee RANK/LOCAL_RANK de torchrun
model = EngramaModel(config).cuda(ctx.local_rank)
optimizer = adamw(model.parameters(), lr=3e-4, fused=True)
step_model = LanguageModelLoss(
    model,
    linear_chunk_size=2048,
    checkpoint_chunks=False,
)
step_model = compile_model(step_model, mode="max-autotune")
step_model = wrap_ddp(step_model, ctx)

with torch.autocast("cuda", dtype=torch.float16):
    loss = step_model(input_ids, target_ids)  # DDP comunica escalares/grads, no logits
```

La pérdida está dentro del módulo DDP. Esto es importante: generar logits en
cada GPU y reunirlos en GPU 0 reintroduciría el cuello de botella.

## Qué cambió en los kernels

- Consolidación: offsets resonantes apilados y contracciones batched; un solo
  padding por tensor/capa. Conserva exactamente la máscara causal.
- Sinapsis: contracción low-rank directa con `einsum`, sin dos temporales 5-D.
- CE por vocabulario existente: `autograd.Function` con backward analítico;
  guarda logits + normalizador, no un grafo Python de cientos de nodos.
- CE lineal: `EngramaModel.forward_loss()` fusiona lógicamente evocador lineal
  y CE por posiciones, evitando retener el tensor global `(B,N,V)`.
- Optimizador: AdamW fusionado en CUDA.
- Distribución: DDP/NCCL con réplica persistente, `static_graph` y
  `gradient_as_bucket_view`.
- Entrada: transferencias `non_blocking`, memoria pinned, prefetch y workers
  persistentes.

## Medición correcta

Usar tokens por segundo global, no solo segundos por paso:

```python
tokens_s = world_size * local_batch * sequence_length / seconds_per_step
```

Para perfilar 10–20 pasos estabilizados:

```bash
torchrun --standalone --nproc_per_node=2 -m torch.distributed.run ... # lanzamiento normal
nsys profile -t cuda,nvtx,osrt -o engrama_report <comando de un proceso>
```

También son útiles `torch.profiler` y estas métricas por proceso:

```python
torch.cuda.reset_peak_memory_stats()
# pasos medidos
print(torch.cuda.max_memory_allocated() / 2**30, "GiB")
```

No se publica una promesa de aceleración fija: el resultado depende de versión
de PyTorch/CUDA, throttling de Kaggle, tamaño de chunk y coste inicial de
compilación. La comparación válida mantiene batch global, secuencia, AMP y
configuración del modelo idénticos.

## Estabilidad numérica

- La CE acumula en FP32.
- `GradScaler.unscale_()` se ejecuta antes del clipping.
- No se hace un segundo escaneo Python de gradientes; `GradScaler.step()` salta
  automáticamente actualizaciones no finitas.
- Para reproducibilidad estricta, desactivar compile/autotune y activar modos
  deterministas; esto sacrifica rendimiento y puede cambiar el orden de sumas.
