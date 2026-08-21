"""ENGRAMA V5 configuration.

V5 = esqueleto V4 estabilizado (mezcla normalizada por conteo + bilineal acotada
+ trace tap) mas el **Recall Tap**: lectura dura (top-1) direccionada por
contenido desde la Traza explicita, que hace la recuperacion exacta a cualquier
distancia sin atencion y sin compresion.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

_OFFSET_MODES = ("resonant_multirate", "dense_dilated")
_RT_QUERY = ("t0", "state")
_RT_VALUE = ("next", "self")


@dataclass
class V5Config:
    """Configuracion de ENGRAMA V5.

    Args:
        vocab_size: Tamano del vocabulario.
        d_model: Dimension oculta ``d``.
        d_gate: Dimension de las proyecciones de compuerta ``d_g << d``.
        d_ff: Expansion FFN de la celula.
        num_cells: Celulas ``C`` por capa del encoder.
        num_encoder_layers: Capas del encoder aislado.
        num_consolidation_layers: Capas de consolidacion ``L`` (alcance ~2^L-1).
        context_length: Ventana maxima ``N_max`` de la traza explicita.
        synapse_rank: Rango ``r`` de las sinapsis factorizadas.
        num_candidates: Candidatos del evocador ``M``.
        dropout: Dropout (0 en inferencia).
        activation: ``gelu`` | ``relu`` | ``silu``.
        tie_embeddings: Evocador ligado al embedding de entrada.
        offset_mode: ``resonant_multirate`` (suavizado multiescala) o ``dense_dilated``.
        offsets: Explicito solo para ``dense_dilated``.
        dual_bilinear_clamp: Cota ``C`` del termino bilineal ``C*tanh(b/C)``.
            >=0 activa la version acotada (default 4.0; V5 siempre acotada).
        count_normalize: Normalizar la mezcla por la suma de compuertas abiertas.
        trace_tap: Acceso directo a la huella pristina T0 (mejor pieza de V4).
        recall_enabled: Activar el Recall Tap.
        rt_layers: Capas de consolidacion donde se inyecta la lectura (p.ej. ``[4]``).
        d_recall: Dimension ``d_k`` de los codigos K de la traza.
        rt_query: ``t0`` (huella del token actual; aislada) o ``state`` (estado consolidado).
        rt_value: ``next`` (huella del token siguiente al match; induccion) o ``self``.
        rt_gap: Distancia minima (en tokens) entre la posicion actual y el match.
        rt_temperature: Temperatura del gradiente straight-through (solo backward).
        rt_score_chunk: Filas de puntuacion por trozo (control de VRAM O(chunk*N)).
        rt_gate_init: Valor inicial de la compuerta de inyeccion ``g = 2*sigmoid(.)``.
        dtype: Precision del modelo.
    """

    vocab_size: int = 256
    d_model: int = 256
    d_gate: int = 32
    d_ff: int = 1024
    num_cells: int = 8
    num_encoder_layers: int = 2
    num_consolidation_layers: int = 9
    context_length: int = 512
    synapse_rank: int = 32
    num_candidates: int = 4
    dropout: float = 0.0
    activation: str = "gelu"
    tie_embeddings: bool = True
    offset_mode: str = "resonant_multirate"
    offsets: Optional[Sequence[int]] = None
    dual_bilinear_clamp: float = 4.0
    count_normalize: bool = True
    trace_tap: bool = True
    recall_enabled: bool = True
    rt_layers: Sequence[int] = (4,)
    d_recall: int = 64
    rt_query: str = "t0"
    rt_value: str = "next"
    rt_gap: int = 1
    rt_temperature: float = 0.5
    rt_score_chunk: int = 1024
    rt_gate_init: float = 1.0
    rt_init_std: float = 0.1
    rt_shared_init: bool = True
    # Entrenamiento del Recall Tap: "dense" (exacto O(N^2 d_k), pequenas N) o
    # "lsh" (lineal O(N (1+t*cap) d_k) con indice exacto de misma-token +
    # buckets LSH; la generacion incremental es siempre exacta y lineal).
    rt_train_mode: str = "dense"
    rt_lsh_tables: int = 2
    rt_lsh_bits: int = 8
    rt_lsh_cap: int = 64
    # Negativos muestreados por consulta en modo LSH: mantienen la metrica
    # global alineada con el argmax denso (evita que scores fuera de bucket
    # deriven hacia arriba sin oposicion). Coste O(N * n_neg * d_k).
    rt_lsh_neg: int = 32
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.offset_mode not in _OFFSET_MODES:
            raise ValueError(f"offset_mode debe ser { _OFFSET_MODES }, no {self.offset_mode!r}")
        if self.rt_query != "t0":
            # 'state' queda reservado para V5.1: la consulta aislada (t0) es la
            # unica que preserva aislamiento total y equivalencia incremental.
            raise ValueError("rt_query debe ser 't0' en V5")
        if self.rt_value not in _RT_VALUE:
            raise ValueError(f"rt_value debe ser { _RT_VALUE }, no {self.rt_value!r}")
        if self.d_gate >= self.d_model:
            raise ValueError("d_gate debe ser < d_model")
        if not 1 <= self.synapse_rank <= self.d_model:
            raise ValueError("synapse_rank debe estar en [1, d_model]")
        if self.recall_enabled:
            layers = list(self.rt_layers)
            if not layers or any(not (0 <= l < self.num_consolidation_layers) for l in layers):
                raise ValueError(
                    f"rt_layers {layers} fuera de [0, {self.num_consolidation_layers})"
                )
        if self.rt_gap < 1:
            raise ValueError("rt_gap >= 1 (el valor leido debe ser causal)")
        if self.rt_train_mode not in ("dense", "lsh"):
            raise ValueError("rt_train_mode debe ser 'dense' o 'lsh'")
        if self.context_length < 2:
            raise ValueError("context_length >= 2")

    # ------------------------------------------------------------------
    # Offsets por capa (suavizado multiescala; la recuperacion exacta es del RT)
    # ------------------------------------------------------------------
    def layer_offsets(self, layer_idx: int) -> List[int]:
        if not 0 <= layer_idx < self.num_consolidation_layers:
            raise IndexError("layer_idx fuera de rango")
        if self.offset_mode == "dense_dilated":
            base = list(self.offsets or [0, 1, 2, 4, 8, 16, 32, 64, 128])
        else:  # resonant_multirate
            if layer_idx == 0:
                base = [0, 1]
            else:
                base = [0, 1, 2 ** (layer_idx - 1), 2 ** layer_idx]
        # Offsets mas alla del alcance util no aportan y cuestan ancho de cache.
        cap = 2 ** max(1, self.num_consolidation_layers - 1)
        return sorted({p for p in base if p <= cap})

    def layer_offsets_all(self) -> List[List[int]]:
        return [self.layer_offsets(l) for l in range(self.num_consolidation_layers)]

    def receptive_field(self) -> Dict[str, object]:
        reachable = {0}
        for offs in self.layer_offsets_all():
            reachable = {r + p for r in reachable for p in offs}
        reach = max(reachable)
        return {
            "max_reach": reach,
            "dense_coverage": all(i in reachable for i in range(reach + 1)),
            "layers": self.num_consolidation_layers,
            "layer_offsets": self.layer_offsets_all(),
        }

    def cache_horizons(self) -> List[int]:
        """Estados minimos por capa para la lectura incremental (invariante V3 §24)."""
        offs = self.layer_offsets_all()
        return [max(offs[l + 1]) + 1 if l + 1 < len(offs) else 1
                for l in range(len(offs))]

    def torch_dtype(self):
        import torch
        return getattr(torch, self.dtype)

    def describe(self) -> str:
        rf = self.receptive_field()
        return (
            f"ENGRAMA V5  d={self.d_model} dg={self.d_gate} r={self.synapse_rank} "
            f"L={self.num_consolidation_layers} (alcance {rf['max_reach']}) N_max={self.context_length}\n"
            f"  mezcla: count_normalize={self.count_normalize} "
            f"bilinear_clamp={self.dual_bilinear_clamp} trace_tap={self.trace_tap}\n"
            f"  recall: enabled={self.recall_enabled} layers={list(self.rt_layers)} "
            f"d_k={self.d_recall} query={self.rt_query} value={self.rt_value} "
            f"gap={self.rt_gap} tau={self.rt_temperature}"
        )

    def to_dict(self) -> Dict[str, object]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "V5Config":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    # ------------------------------------------------------------------
    # Presets listos para usar (API comoda)
    # ------------------------------------------------------------------
    @classmethod
    def from_preset(cls, size: str, **overrides) -> "V5Config":
        presets = {
            "tiny": dict(d_model=64, d_gate=16, d_ff=256, num_cells=2,
                         num_encoder_layers=1, num_consolidation_layers=6,
                         context_length=256, synapse_rank=16, d_recall=32,
                         rt_layers=(3,)),
            "small": dict(d_model=128, d_gate=16, d_ff=512, num_cells=4,
                          num_encoder_layers=1, num_consolidation_layers=8,
                          context_length=1024, synapse_rank=16, d_recall=64,
                          rt_layers=(4,)),
            "base": dict(d_model=256, d_gate=32, d_ff=1024, num_cells=8,
                         num_encoder_layers=2, num_consolidation_layers=9,
                         context_length=8192, synapse_rank=32, d_recall=64,
                         rt_layers=(4,)),
            "large": dict(d_model=512, d_gate=64, d_ff=2048, num_cells=16,
                          num_encoder_layers=2, num_consolidation_layers=11,
                          context_length=32768, synapse_rank=32, d_recall=96,
                          rt_layers=(5,)),
        }
        if size not in presets:
            raise ValueError(f"preset {size!r} desconocido: {tuple(presets)}")
        kw = dict(presets[size])
        kw.update(overrides)
        return cls(**kw)
