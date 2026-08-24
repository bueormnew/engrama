"""ENGRAMA V6 — configuracion (version DEFINITIVA).

V6 parte de V5.5 y ELIMINA todo coste cuadratico respetando la filosofia:

* Huella aislada ``T0[j] = f(x_j)`` (cero mezcla temporal en Fase 1).
* Traza FIFO explicita, sin compresion, paginada.
* Cero ``QK^T`` ``N x N``, cero softmax sobre el eje temporal.
* Consolidacion por offsets causales fijos ``D_l`` (O(N)).
* Invarianza causal exacta: forward paralelo == generacion incremental.

Las tres correcciones estructurales que convierten a V6 en puramente lineal:

1. **Recall lexico por INDICE INVERTIDO de token** (no denso O(N^2)):
   ``K_lex[j]`` depende solo del token, asi que las posiciones con el mismo
   token se agrupan por un ``inverted_index[token]`` (lista de posiciones).
   La puntuacion se hace SOLO sobre el grupo del token de la consulta:
   coste ``O(m_i d_k)`` con ``m_i`` la frecuencia del token. Caso promedio
   O(N * (N/V) * d_k) = O(N d_k) para ``V`` razonable; identidad O(1).
2. **LSH de ALTA RECUPERACION**: 16 tablas de 16 bits (65536 buckets) SIN cap
   de expulsion. Con 16 proyecciones ortogonales-escaladas, el vecino mas
   cercano cae en el mismo bucket en >= 1 tabla con probabilidad ~1 para
   vectores coseno-cercanos (los verdaderos matches). Buckets diminutos
   (~N/65536 entradas) => ``O(N * C * d)`` con C medio pequeno y LINEAL.
3. **Claves L2 en FP32 siempre** aunque el modelo vaya en FP16/BF16: la
   comparacion de scores y el argmax se computan en FP32 => invarianza causal
   exacta (<1e-6) en TODAS las precisiones.

Ademas se corrige el bug de ``vocab_size`` del evocador (V5.5 usaba 4096 fijo
en ``_inner``). En V6 el evocador conoce el ``vocab_size`` real.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

_OFFSET_MODES = ("resonant_multirate", "dense_dilated")
_RT_TRAIN_MODE = ("dense", "lsh")
_ACTIVATIONS = ("gelu", "relu", "silu")


@dataclass
class V6Config:
    """Configuracion de ENGRAMA V6 (puramente lineal, recall exacto)."""

    vocab_size: int = 256
    d_model: int = 256
    d_gate: int = 32
    d_ff: int = 1024
    num_cells: int = 8
    num_encoder_layers: int = 2
    num_consolidation_layers: int = 9
    context_length: int = 8192
    synapse_rank: int = 32
    num_candidates: int = 4
    dropout: float = 0.0
    activation: str = "silu"
    tie_embeddings: bool = True
    offset_mode: str = "resonant_multirate"
    offsets: Optional[Sequence[int]] = None
    dual_bilinear_clamp: float = 4.0
    count_normalize: bool = True
    trace_tap: bool = True
    dtype: str = "float32"

    # --- Recall (lexico + sentido) -----------------------------------------
    recall_enabled: bool = True
    d_recall: int = 64
    d_sense: Optional[int] = None
    rt_value: str = "next"
    rt_gap: int = 1
    rt_temperature: float = 0.5
    rt_score_chunk: int = 1024
    rt_gate_init: float = 1.0
    rt_init_std: float = 0.1
    rt_shared_lex_init: bool = True
    rt_sense_beta_init: float = 0.3
    rt_sense_beta_trainable: bool = True
    rt_score_threshold: float = 0.5
    rt_fallback_window: int = 2048
    rt_use_identity_fast_path: bool = True
    rt_ctx_query: str = "final"

    # V6: ruta lexico-paralela por INDICE INVERTIDO (nunca denso O(N^2)).
    # "inverted" = exacto y lineal; "dense" se conserva solo para tests.
    rt_lex_parallel_mode: str = "inverted"

    # --- Tap SEMANTICO (asociativo) ----------------------------------------
    semantic_recall_enabled: bool = True
    d_semantic: int = 64
    rt_sem_temperature: float = 0.1
    rt_sem_gate_init: float = 1.0
    rt_sem_key_source: str = "t0"
    rt_sem_query_source: str = "t0"
    # "lsh" (lineal, alta recuperacion) es el default V6; "dense" O(N^2) solo
    # para validacion/benchmarks.
    rt_sem_recall_mode: str = "lsh"

    # --- LSH V6 (alta recuperacion, sin cap) -------------------------------
    rt_train_mode: str = "lsh"
    rt_lsh_tables: int = 16          # V6: 16 tablas
    rt_lsh_bits: int = 16            # V6: 16 bits => 65536 buckets
    rt_lsh_cap: int = 4              # V6: 4 posiciones mas recientes por bucket
    rt_lsh_neg: int = 48
    rt_lsh_hamming: int = 0          # 0 = solo colision exacta en >=1 tabla
    # Forzar claves FP32 en el LSH aun con el modelo en FP16/BF16 (invarianza).
    rt_lsh_fp32_keys: bool = True
    # Ventana de rescate denso (barrido exacto de los ultimos W tokens) que se
    # suma a los candidatos LSH. Garantiza que un match NUNCA se pierde por mala
    # suerte del hashing, sin volver a O(N^2): W es fijo (p.ej. 256).
    rt_lsh_rescue_window: int = 256

    # --- Evocador ----------------------------------------------------------
    logit_cap: float = 30.0

    # --- Traza paginada ----------------------------------------------------
    page_size: int = 256
    # Precision a la que la traza ALMACENA claves (K_lex, K_sense, K_sem).
    # "fp32" garantiza invarianza paralelo==incremental en cualquier precision
    # de computo. Es el default V6.
    trace_store_dtype: str = "fp32"

    # --- Entrenamiento: CE_retrieval auto-supervisado ----------------------
    retrieval_weight: float = 1.0
    retrieval_anneal_steps: int = 0
    retrieval_positions_frac: float = 0.25
    retrieval_same_token_only: bool = True
    retrieval_temperature: float = 0.1

    def __post_init__(self) -> None:
        if self.offset_mode not in _OFFSET_MODES:
            raise ValueError(f"offset_mode debe ser {_OFFSET_MODES}, no {self.offset_mode!r}")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"activation debe ser {_ACTIVATIONS}, no {self.activation!r}")
        if self.rt_value not in ("next", "self"):
            raise ValueError("rt_value debe ser 'next' o 'self'")
        if self.rt_ctx_query not in ("final", "shallow"):
            raise ValueError("rt_ctx_query debe ser 'final' o 'shallow'")
        if self.rt_train_mode not in _RT_TRAIN_MODE:
            raise ValueError(f"rt_train_mode debe ser {_RT_TRAIN_MODE}, no {self.rt_train_mode!r}")
        if self.rt_sem_recall_mode not in _RT_TRAIN_MODE:
            raise ValueError(f"rt_sem_recall_mode debe ser {_RT_TRAIN_MODE}")
        if self.rt_sem_key_source not in ("t0", "t_last"):
            raise ValueError("rt_sem_key_source debe ser 't0' o 't_last'")
        if self.rt_sem_query_source not in ("t0", "t_last"):
            raise ValueError("rt_sem_query_source debe ser 't0' o 't_last'")
        if self.rt_lex_parallel_mode not in ("inverted", "dense"):
            raise ValueError("rt_lex_parallel_mode debe ser 'inverted' o 'dense'")
        if self.d_gate >= self.d_model:
            raise ValueError("d_gate debe ser < d_model")
        if not 1 <= self.synapse_rank <= self.d_model:
            raise ValueError("synapse_rank debe estar en [1, d_model]")
        if self.d_sense is None:
            object.__setattr__(self, "d_sense", self.d_recall)
        if self.rt_gap < 1:
            raise ValueError("rt_gap >= 1 (el valor leido debe ser causal)")
        if not 1 <= self.d_semantic <= 4096:
            raise ValueError("d_semantic debe estar en [1, 4096]")
        if not 0.0 <= self.retrieval_positions_frac <= 1.0:
            raise ValueError("retrieval_positions_frac debe estar en [0, 1]")
        if not 0.0 < self.retrieval_temperature <= 1.0:
            raise ValueError("retrieval_temperature debe estar en (0, 1]")
        if self.context_length < 2:
            raise ValueError("context_length >= 2")
        if self.page_size < 8:
            raise ValueError("page_size >= 8")
        if not 1 <= self.rt_lsh_bits <= 24:
            raise ValueError("rt_lsh_bits debe estar en [1, 24]")
        if self.rt_lsh_tables < 1:
            raise ValueError("rt_lsh_tables >= 1")

    def layer_offsets(self, layer_idx: int) -> List[int]:
        if not 0 <= layer_idx < self.num_consolidation_layers:
            raise IndexError("layer_idx fuera de rango")
        if self.offset_mode == "dense_dilated":
            base = list(self.offsets or [0, 1, 2, 4, 8, 16, 32, 64, 128])
        else:  # resonant_multirate  D_l = {0, 1, 2^{l-1}, 2^l}
            if layer_idx == 0:
                base = [0, 1]
            else:
                base = [0, 1, 2 ** (layer_idx - 1), 2 ** layer_idx]
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
        offs = self.layer_offsets_all()
        return [max(offs[l + 1]) + 1 if l + 1 < len(offs) else 1
                for l in range(len(offs))]

    def torch_dtype(self):
        import torch
        return getattr(torch, self.dtype)

    def trace_store_torch_dtype(self):
        import torch
        return torch.float32 if self.trace_store_dtype == "fp32" else self.torch_dtype()

    def describe(self) -> str:
        rf = self.receptive_field()
        return (
            f"ENGRAMA V6  d={self.d_model} dg={self.d_gate} r={self.synapse_rank} "
            f"L={self.num_consolidation_layers} (alcance {rf['max_reach']}) N_max={self.context_length}\n"
            f"  mezcla: count_normalize={self.count_normalize} "
            f"clamp={self.dual_bilinear_clamp} trace_tap={self.trace_tap}\n"
            f"  recall asim: d_k={self.d_recall} d_sense={self.d_sense} "
            f"lex_mode={self.rt_lex_parallel_mode}\n"
            f"  semantic: d={self.d_semantic} mode={self.rt_sem_recall_mode} "
            f"key_src={self.rt_sem_key_source}\n"
            f"  LSH: t={self.rt_lsh_tables} b={self.rt_lsh_bits} "
            f"cap={self.rt_lsh_cap} rescue_w={self.rt_lsh_rescue_window} "
            f"fp32_keys={self.rt_lsh_fp32_keys}\n"
            f"  evoker: logit_cap={self.logit_cap}  trace: page={self.page_size} "
            f"store={self.trace_store_dtype}"
        )

    def to_dict(self) -> Dict[str, object]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "V6Config":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_preset(cls, size: str, **overrides) -> "V6Config":
        presets = {
            "tiny": dict(d_model=64, d_gate=16, d_ff=256, num_cells=2,
                         num_encoder_layers=1, num_consolidation_layers=6,
                         context_length=256, synapse_rank=16, d_recall=32,
                         d_sense=32, d_semantic=32),
            "small": dict(d_model=128, d_gate=16, d_ff=512, num_cells=4,
                          num_encoder_layers=1, num_consolidation_layers=8,
                          context_length=4096, synapse_rank=16, d_recall=64,
                          d_sense=64, d_semantic=64),
            "base": dict(d_model=256, d_gate=32, d_ff=1024, num_cells=8,
                         num_encoder_layers=2, num_consolidation_layers=9,
                         context_length=16384, synapse_rank=32, d_recall=64,
                         d_sense=64, d_semantic=64),
            "large": dict(d_model=512, d_gate=64, d_ff=2048, num_cells=16,
                          num_encoder_layers=2, num_consolidation_layers=11,
                          context_length=131072, synapse_rank=32, d_recall=96,
                          d_sense=96, d_semantic=96),
        }
        if size not in presets:
            raise ValueError(f"preset {size!r} desconocido: {tuple(presets)}")
        kw = dict(presets[size])
        kw.update(overrides)
        return cls(**kw)
