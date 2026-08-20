# Verificación de la Arquitectura ENGRAMA V3

Este documento resume **qué está verificado, cómo y con qué resultados**.
Todos los números provienen de ejecuciones reales en este repositorio
(política de honestidad de la especificación V3, sección 56: ningún número
proyectado se presenta como medido). Fecha de la ejecución: 2026-08-19,
CPU, torch 2.13, float32 salvo indicación.

## 1. Cumplimiento de la especificación V3

| Requisito (spec) | Implementación | Verificación |
|---|---|---|
| Sinapsis factorizada $W_{a \to b} = \beta I + U_l \mathrm{Diag}(s) V_l^\top$, $U_l,V_l$ **compartidas por capa** (§6–7) | `primitives.SynapseLayer` / `consolidation.PositionalDilatedMix` | `test_factorized_shapes_3d_4d`, `test_stable_init_gives_zero_s_and_unit_beta` |
| Compuerta desde la **fuente** $\alpha_{a \to b} = \sigma(w \cdot P_g h_a + b)$ (§6) | `SynapseLayer` | `test_perturbing_source_changes_only_its_outgoing_gates` (perturbar la fuente solo altera sus aristas salientes) |
| Transporte identidad (§31): $s{=}0, \beta{=}1 \Rightarrow W h = h$ | init estable por defecto (`stable_init=True`) | `test_single_synapse_is_identity` (error **0.0**), `test_zero_scale_is_residual_identity` |
| Offsets $D_l = \{0,1,2^l\}$ (§8), `binary_minimal`, `dense_dilated` | `EngramaConfig.get_layer_offsets` | `test_hierarchical_dyadic_offsets`, `test_binary_minimal_offsets`, `test_dense_dilated_uses_explicit_offsets` |
| Compuerta escalar por escala $\rho_{l,p}$ (§17) | `PositionalDilatedMix.rho` | `test_all_modes` (variante `v3_no_hierarchical_gate`), inspección confirma $\sigma(\rho){=}0.5$ al init |
| Caché jerárquico de horizonte mínimo $\max(D_{l+1})+1$ (§12, §24) | `trace.HierarchicalStateCache` | `test_cache_horizons`, `test_hierarchical_horizons_enforced`, `test_state_reduction` |
| Regla de profundidad $L \ge \lceil \log_2 N \rceil$ (§26) | warning descriptivo en `EngramaConfig` | `test_depth_rule_warning`, `test_no_warning_when_coverage_ok` |
| Evocador factorizado con $W_{shared}$ compartido y agregación (§14, §37) | `evoker.MultiCandidateEvoker` | `test_m1_collapses_to_linear_classifier`, `test_mean_opt_equals_naive`, `test_lse_matches_reference`, `test_lse_large_values_no_overflow` |
| Presets de versión v1/v2/v3 como ablaciones reales (§54) | `VERSION_PRESETS` + overrides explícitos | `test_v1_maps_to_dense_parameterization`, `test_v2_preset_resolution`, `test_v3_preset_resolution`, `test_explicit_override_wins_over_preset` |
| Cero atención: sin $QK^T$, sin softmax sobre la secuencia | toda la librería | búsqueda exhaustiva del código: sin `softmax`/`einsum` de afinidad temporal (el único softmax es el de vocabulario) |
| Ancla global $g(N)$ opcional en la última capa (§11) | `global_anchor=True` | `test_global_anchor_only_on_last_layer` |

## 2. Invarianza causal (garantía central)

- **Matriz de 17 combinaciones de modos** (`test_causal_invariance.py`):
  v3 default, binary_minimal, dense_dilated, sin compuerta jerárquica, sin
  transporte identidad, con ancla global, init inestable, agregaciones
  `mean`/`max`, `M=1`, presets `v1`/`v2`, sinapsis densas, offsets densos
  con células independientes, embeddings no atados, activaciones `relu`/`silu`.
- Cada combinación se verifica bajo **ambos modos de caché** (`full` y
  `hierarchical`): `forward(seq)` vs `step_forward` token a token.
- **Resultado medido: error máximo 5.96e-07** (float32). Umbral del test: 1e-4.
- Régimen de desborde FIFO (`test_overflow_equivalence`): igualdad exacta
  (1.19e-07) cuando el cono de dependencias de las posiciones comparadas
  cabe dentro de la ventana retenida.
- Causalidad estricta (§4.5): modificar tokens futuros no altera logits
  pasados (`test_future_tokens_do_not_change_past`, diff < 1e-5).
- Generación con y sin caché produce idéntica secuencia
  (`test_cached_and_uncached_generation_match`).

## 3. Caché jerárquico — reducción de estado medida

Con $N_{max} = 256$ y $L = 8$ (preset `small`/`base`):

- Capacidades por capa: `[3, 5, 9, 17, 33, 65, 129, 1]` (total 262 estados).
- Caché completo V2: 256 × 8 = 2048 estados.
- **Reducción de estado: 7.82×** (`test_state_reduction`).

La traza circular usa `collections.deque(maxlen=N)` — inserción O(1) real
(`test_fifo_circular_overwrite`).

## 4. Suite de tests

**68 tests, todos en verde** (~14 s, CPU):

```bash
cd tests && python -m unittest discover -q
# Ran 68 tests in 13.5s — OK
```

Cobertura: config (presets, validación, campos receptivos, horizontes),
primitivas (células, sinapsis densa/factorizada, compuerta-desde-fuente,
transporte identidad, shared-core), encoder + traza (aislamiento de tokens,
FIFO, horizontes, vistas/device), invarianza causal (matriz 17×2 +
desborde + estricta), evocador (shapes, LSE estable, M=1,
optimización `mean`), tokenizador/dataset (round-trip, stride, archivos),
ecosistema (entrenamiento con schedulers, quickstart end-to-end,
serialización en todos los modos, benchmarks hooks, generación).

## 5. Benchmark de largo alcance

`benchmarks/kv_retrieval.py` implementa una tarea de recuperación
clave→valor científicamente válida (bindings aleatorios por muestra —
imposible memorizarlos globalmente —, consultas a distancias de hasta
~176 tokens, 200 muestras de evaluación con semilla independiente).

Los resultados completos y su interpretación honesta están en
`benchmarks/KV_RETRIEVAL_REPORT.md`. Adelanto (600 pasos, seed 1234, CPU):
`dense_dilated` alcanza **27.5%** de precisión sostenida hasta ~176 tokens
(azar: 6.2%), mientras `hierarchical_dyadic` puro queda en 7.4% y con
ancla global en 7.1% — **confirmando empíricamente el riesgo principal
identificado por la propia especificación (§42)**: la conectividad diádica
mínima {0, 1, 2^l} por sí sola no garantiza rutas de transporte para
todas las distancias en esta tarea, y el benchmark queda incluido como
suite de ablación D/E reproducible.

## 6. Números de parámetros (presets, vocab=256)

| Preset | Parámetros |
|---|---|
| tiny | 277,330 |
| small | 1,451,646 |
| base | 6,862,990 |
| large | 38,034,624 |

Referencia de escala idéntica (d_model=256, C=8, L=8, N=256, vocab=128):
**V3 = 6,889,102** vs **V2 = 29,549,952** parámetros (4.29× menos).

## 7. Cómo reproducir

```bash
pip install -e .
cd tests && python -m unittest discover -q
python benchmarks/kv_retrieval.py --steps 600 --seed 1234
engrama benchmark --size small --seq-len 256 --runs 10   # incluye chequeo de invarianza causal
```
