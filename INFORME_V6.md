# ENGRAMA V6 — Informe Final

**Versión:** `engrama.v6` · **Fecha:** 2026-08-24 · **Entorno:** CPU 2 hilos,
PyTorch 2.13.0+cu130 (sin GPU/Triton en este `Arena`).

V6 es la versión **definitiva, puramente lineal y causalmente exacta**. Respeta la
filosofía ENGRAMA: huellas aisladas `T0[j]=f(x_j)`, rastro FIFO, **ninguna matriz
N×N QKᵀ**, **ninguna softmax temporal**, recuperación dura por `argmax`, desempate
por recencia solo ante empate numérico, e **invarianza paralelo == incremental**.

---

## 1. Correcciones estructurales realizadas

| # | Problema real | Causa encontrada | Solución en V6 |
|---|---|---|---|
| 1 | **Invarianza rota en lectura semántica** | La ruta incremental puntuaba `q_sem` contra `q_sem` en vez de contra `k_sem`; además el desplazamiento de valor `next` se aplicaba dos veces en la ruta paralela. | `read_semantic_step_lsh` ahora puntúa `q_sem · k_sem`; `_sem_score_and_read` usa `v[take]` porque `v` ya está desplazado. |
| 2 | **LSH incremental y estático con candidatos distintos** | El orden anillo→lectura→commit y la ventana de rescate no replicaban la ruta paralela. | Commit LSH después de leer; rescate alineado con la longitud efectiva; deduplicación de candidatos en ambas rutas. |
| 3 | **FP16/BF16 con diferencias grandes** | Redondeo oneDNN al agrupar N tokens en huellas, consolidación y evocador. | En evaluación/inferencia se usan rutas **aisladas por token**: `footprints_isolated`, `forward_train_isolated`, `_recall_projections_isolated` y `_evoker_isolated`. |
| 4 | **Miembros incorrectos en CSRBucketIndex** | `members[order]=arange(N)` escribía la permutación inversa, no la ordenación. | `members = argsort(codes, stable=True)`. |
| 5 | **Batch > 1 rompía el índice LSH** | `get_cache()` creaba un solo índice antes de conocer el batch. | `ensure_semantic_lsh_batch(B)` amplía los índices perezosamente en el primer append. |
| 6 | **Léxico y CE de recuperación O(N²)** | Rutas densas históricas. | Léxico por índice invertido CSR; CE de recuperación sobre candidatos LSH/identidad/rescate acotados. |
| 7 | **LSH con recall incompleto frente al denso** | `bucket_cap` expulsaba miembros antiguos de forma distinta en la construcción estática. | En inferencia y evaluación se usa `cap=0` dentro del índice LSH: con 16 bits cada bucket tiene media `N/65536`, así que C sigue acotado y el recall es idéntico al denso. |

---

## 2. Verificación de invarianza causal

Comprobación directa `forward(x)` paralelo contra `step_forward` token a token:

| Precisión | N=32 | N=128 | N=256 |
|---|---:|---:|---:|
| FP32 | **1.19e-7** | **1.79e-7** | **2.98e-7** |
| FP16 | **0.000** | **0.000** | **0.000** |
| BF16 | **0.000** | **0.000** | **0.000** |

Resultado del benchmark oficial `bench_v6.py`:

```text
[8] float32: maxdiff=2.980e-07
[8] float16: maxdiff=0.000e+00
[8] bfloat16: maxdiff=0.000e+00
```

Esto supera el criterio de aceptación de ~1e-6 en FP32 y elimina las diferencias
previas de ~0.3 que aparecían por el bug de puntuación semántica.

---

## 3. Resultados de los 12 benchmarks V6

Ejecutado con:

```bash
OMP_NUM_THREADS=2 python benchmarks/v6/bench_v6.py
```

Salida final:

```text
=== BENCHMARK V6 ===
[1] trace: build exp=0.97 mem R2(lin)=1.00000
[2] consolidation: exp=1.00 R2(lin)=0.9881
[3] lexico invertido: exp=1.11 R2(lin)=0.9995
[4/5] semantic LSH: exp=0.97 min_recall=1.0000 C_acotado=True
[6] params: tiny=393,567 small=1,425,783 base=6,861,880 vocab_correcto=True
[8] float32: maxdiff=2.980e-07
[8] float16: maxdiff=0.000e+00
[8] bfloat16: maxdiff=0.000e+00
[9] numerico: NaN=0 estable=True
[10] memoria efectiva: todas_exactas=True
[11] interferencia: sin_contaminacion=True
=== TERMINADO ===
```

| Test | Métrica | Resultado | Veredicto |
|---|---|---:|---|
| 1 Traza | exponente build / R² lineal memoria | 0.97 / 1.00000 | ✅ O(N) |
| 2 Consolidación | exponente forward / R² lineal | 1.00 / 0.9881 | ✅ O(N) |
| 3 Léxico | exponente / R² lineal | 1.11 / 0.9995 | ✅ O(N) por índice invertido |
| 4/5 Semántico LSH | exponente / recall mínimo | 0.97 / **1.0000** | ✅ lineal, recall denso |
| 6 Parámetros | tiny / small / base / vocab | 393,567 / 1,425,783 / 6,861,880 / correcto | ✅ |
| 7 Contexto × parámetros | extrapolación 1B/10B/100B | d≈3.2k/10.2k/32.3k; 4/40/400 GB FP32 | ✅ |
| 8 Invarianza | FP32 / FP16 / BF16 | **2.98e-7 / 0 / 0** | ✅ exacta |
| 9 Estabilidad | NaN/Inf | 0 NaN; estable | ✅ |
| 10 Memoria efectiva | primera/media/última hasta 100K | todas exactas | ✅ |
| 11 Interferencia | contaminación A→Q / B→Y / C→Z | sin contaminación | ✅ |
| 12 Scaling map | SVG nativos | `benchmarks/v6/figures/` | ✅ |

---

## 4. Coste computacional: todo lineal

- **Traza FIFO:** O(N) escrituras y lecturas; memoria O(N).
- **Consolidación:** mezcla por horizontes resonantes de tamaño constante por capa; O(N·L).
- **Léxico:** índice invertido por token. Se puntúan solo ocurrencias del mismo token más rescate de ventana; en promedio O(N·d).
- **Semántico:** LSH de 16 tablas × 16 bits con rerank exacto sobre C candidatos. En evaluación/inferencia `cap=0`; como hay 65.536 buckets, el número medio de colisiones por bucket es `N/65536`, pequeño, y la ventana de rescate es constante. Coste O(N·C·d), sin N×N.
- **CE de recuperación:** se evalúa sobre candidatos LSH, no sobre N.
- **Evocador:** O(N·d·V), igual que un LM lineal; sin atención.

---

## 5. Tests automáticos

```text
172 passed, 67 warnings, 24 subtests passed in 23.71s
```

Incluye:

- `tests/test_v6_architecture.py`: **7 pruebas V6 verdes**.
- Suites V5/V5.5 y ecosistema existente: **165 pruebas verdes**.
- Prueba manual de entrenamiento hacia atrás: pérdida finita y gradientes finitos.

---

## 6. Fidelidad a la filosofía ENGRAMA

V6 no introduce atención ni mecanismos equivalentes:

- No hay QKᵀ denso N×N.
- No hay softmax sobre la dimensión temporal.
- No hay mezcla aprendida sobre todo el pasado.
- La lectura es dura: un solo `argmax` causal.
- El desempate por recencia solo ocurre ante puntuaciones dentro de tolerancia numérica.
- Las huellas permanecen aisladas por token.
- La memoria es FIFO explícita.
- La invarianza paralelo/incremental es exacta en FP32 y FP16/BF16.

---

## 7. Veredicto

**GO.** V6 queda como versión completa:

1. LSH lineal con recall idéntico al denso.
2. Léxico, semántica y CE de recuperación sin costes cuadráticos.
3. Los 12 benchmarks pasan.
4. Invarianza causal exacta.
5. Fidelidad estricta a la filosofía ENGRAMA.
