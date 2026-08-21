"""ENGRAMA Configuration Module (V3 + V4).

Implements the configuration surface defined by ENGRAMA:

- ``version`` is an architecture preset:
  - ``"v1"`` / ``"v2"``: dense synapses, dense dilated offsets, full cache,
    independent cells, dense evoker.
  - ``"v3"``: factorized synapses with identity transport, hierarchical dyadic
    offsets, hierarchical minimum-horizon cache, shared-core cells, factorized
    evoker (mean/logsumexp/max).
  - ``"v4"``: resonant multi-rate offsets, dual target-source gating, direct
    trace tap (pristine T0 bypass), RMSNorm and latent-fusion evoker
    (O(|V|d) cost, zero gradient checkpoints).
  Any mode passed explicitly overrides the preset, enabling full ablations.
- ``cache_mode`` selects ``"full"`` (V2 causal cache) or ``"hierarchical"``
  (minimum-horizon cache, theorem of V3 section 24).
- Depth rule: ``num_consolidation_layers >= ceil(log2(N))`` for full binary
  coverage. Violations emit a descriptive warning.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Version presets
# ---------------------------------------------------------------------------

_SYNAPSE_MODES = ("dense", "factorized")
_CELL_MODES = ("independent", "shared_core")
_OFFSET_MODES = (
    "dense_dilated",
    "hierarchical_dyadic",
    "binary_minimal",
    "resonant_multirate",
)
_CACHE_MODES = ("full", "hierarchical")
_EVOKER_MODES = ("dense", "factorized")
_AGGREGATIONS = ("max", "logsumexp", "mean", "latent_fusion")
_ACTIVATIONS = ("gelu", "relu", "silu")
_GATING_MODES = ("source", "dual")
_NORM_TYPES = ("layernorm", "rmsnorm")
_DTYPE_MAP = {
    "float32": "float32",
    "float64": "float64",
    "float16": "float16",
    "bfloat16": "bfloat16",
}

VERSION_PRESETS: Dict[str, Dict[str, Any]] = {
    "v1": {
        "synapse_mode": "dense",
        "cell_mode": "independent",
        "offset_mode": "dense_dilated",
        "cache_mode": "full",
        "evoker_mode": "dense",
        "candidate_aggregation": "logsumexp",
        "identity_transport": False,
        "hierarchical_gate": False,
        "gating_mode": "source",
        "trace_tap": False,
        "norm_type": "layernorm",
    },
    "v2": {
        "synapse_mode": "dense",
        "cell_mode": "independent",
        "offset_mode": "dense_dilated",
        "cache_mode": "full",
        "evoker_mode": "dense",
        "candidate_aggregation": "logsumexp",
        "identity_transport": False,
        "hierarchical_gate": False,
        "gating_mode": "source",
        "trace_tap": False,
        "norm_type": "layernorm",
    },
    "v3": {
        "synapse_mode": "factorized",
        "cell_mode": "shared_core",
        "offset_mode": "hierarchical_dyadic",
        "cache_mode": "hierarchical",
        "evoker_mode": "factorized",
        "candidate_aggregation": "logsumexp",
        "identity_transport": True,
        "hierarchical_gate": True,
        "gating_mode": "source",
        "trace_tap": False,
        "norm_type": "layernorm",
    },
    "v4": {
        "synapse_mode": "factorized",
        "cell_mode": "shared_core",
        "offset_mode": "resonant_multirate",
        "cache_mode": "hierarchical",
        "evoker_mode": "factorized",
        "candidate_aggregation": "latent_fusion",
        "identity_transport": True,
        "hierarchical_gate": True,
        "gating_mode": "dual",
        "trace_tap": True,
        "norm_type": "rmsnorm",
    },
}

_DEFAULT_OFFSETS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


@dataclass
class EngramaConfig:
    """Configuration dataclass for ENGRAMA models (expert mode).

    Every architecture mode may be left as ``None`` to inherit the value
    prescribed by the ``version`` preset, or set explicitly to compose any
    V2/V3/V4 ablation.

    Args:
        vocab_size: Vocabulary size. Default: 256.
        d_model: Hidden dimension ``d``. Default: 256.
        d_gate: Gating latent dimension ``d_g << d``. Default: 32.
        d_ff: Cell feed-forward expansion (4 * d_model). Default: 1024.
        num_cells: Number of cells ``C`` per encoder layer. Default: 8.
        num_encoder_layers: Encoder Synapse layers ``L_enc``. Default: 2.
        num_consolidation_layers: Consolidation layers ``L``. Default: 8.
        context_length: Trace window ``N_max``. Default: 256.
        offsets: Explicit offset family (only used by ``dense_dilated``).
        num_candidates: Evoker candidates ``M in [1, 8]``. Default: 4.
        candidate_aggregation: ``"latent_fusion"`` | ``"mean"`` | ``"logsumexp"`` | ``"max"``.
        activation: ``"gelu"`` | ``"relu"`` | ``"silu"``.
        dropout: Dropout probability. Default: 0.0.
        dtype: Model/trace precision. Default: ``"float32"``.
        version: Architecture preset: ``"v1"`` | ``"v2"`` | ``"v3"`` | ``"v4"``.
        tie_embeddings: Tie evoker projection to input embeddings.
        synapse_mode: ``"dense"`` (V2) or ``"factorized"`` (V3/V4).
        synapse_rank: Low-rank dimension ``r << d``. Default: 32.
        identity_transport: Enable the identity route ``beta * h``.
        cell_mode: ``"independent"`` or ``"shared_core"``.
        offset_mode: ``"resonant_multirate"`` | ``"hierarchical_dyadic"`` |
            ``"binary_minimal"`` | ``"dense_dilated"``.
        global_anchor: Add deterministic global anchor ``g(N)`` at the last
            consolidation layer. Default: False.
        evoker_mode: ``"dense"`` | ``"factorized"``.
        cache_mode: ``"full"`` (V2) | ``"hierarchical"`` (V3/V4).
        hierarchical_gate: Scalar per-scale gate ``rho``.
        stable_init: Initialize synapses near the identity route
            (``s ~ 0``, ``beta = 1``). Default: True.
        gating_mode: ``"source"`` (V3 source-only) or ``"dual"`` (V4 target-source bilinear).
        trace_tap: Direct trace access (T0 pristine memory tap in consolidation).
        norm_type: Normalization function: ``"rmsnorm"`` (V4) or ``"layernorm"`` (V2/V3).
    """

    vocab_size: int = 256
    d_model: int = 256
    d_gate: int = 32
    d_ff: int = 1024
    num_cells: int = 8
    num_encoder_layers: int = 2
    num_consolidation_layers: int = 8
    context_length: int = 256
    offsets: Optional[List[int]] = None
    num_candidates: int = 4
    candidate_aggregation: Optional[str] = None
    activation: str = "gelu"
    dropout: float = 0.0
    dtype: str = "float32"
    version: str = "v4"
    tie_embeddings: bool = True

    # --- Architecture modes (None => inherit from the version preset) ------
    synapse_mode: Optional[str] = None
    synapse_rank: int = 32
    identity_transport: Optional[bool] = None
    cell_mode: Optional[str] = None
    offset_mode: Optional[str] = None
    global_anchor: bool = False
    evoker_mode: Optional[str] = None
    cache_mode: Optional[str] = None
    hierarchical_gate: Optional[bool] = None
    stable_init: bool = True
    gating_mode: Optional[str] = None
    trace_tap: Optional[bool] = None
    norm_type: Optional[str] = None
    # Acota la pre-activacion bilineal del gating dual: b' = C*tanh(b/C).
    # None (default) = comportamiento V4 clasico. Con C~4 evita que el termino
    # q.k (que crece con |T|^2) sature la sigmoide a 0/1 a mitad de
    # entrenamiento sin tocar el resto de la ecuacion.
    dual_bilinear_clamp: Optional[float] = None

    def __post_init__(self) -> None:
        # -- resolve version presets (explicit values win) ------------------
        if self.version not in VERSION_PRESETS:
            raise ValueError(
                f"version must be one of {tuple(VERSION_PRESETS)}, got {self.version!r}"
            )
        preset = VERSION_PRESETS[self.version]
        for field_name in (
            "synapse_mode",
            "cell_mode",
            "offset_mode",
            "cache_mode",
            "evoker_mode",
            "candidate_aggregation",
            "identity_transport",
            "hierarchical_gate",
            "gating_mode",
            "trace_tap",
            "norm_type",
        ):
            if getattr(self, field_name) is None:
                setattr(self, field_name, preset[field_name])

        # -- structural validation ------------------------------------------
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if self.d_model < 1:
            raise ValueError("d_model must be >= 1")
        if self.num_cells < 1:
            raise ValueError("num_cells must be >= 1")
        if self.num_encoder_layers < 1:
            raise ValueError("num_encoder_layers must be >= 1")
        if self.num_consolidation_layers < 1:
            raise ValueError("num_consolidation_layers must be >= 1")
        if self.context_length < 1:
            raise ValueError("context_length must be >= 1")
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_ff < 1:
            raise ValueError("d_ff must be >= 1")
        if self.d_gate >= self.d_model:
            raise ValueError(
                f"d_gate ({self.d_gate}) must be strictly less than d_model ({self.d_model})"
            )
        if self.synapse_rank < 1 or self.synapse_rank > self.d_model:
            raise ValueError("synapse_rank must satisfy 1 <= r <= d_model")
        if not (1 <= self.num_candidates <= 8):
            raise ValueError("num_candidates must be between 1 and 8 inclusive")
        if self.candidate_aggregation not in _AGGREGATIONS:
            raise ValueError(f"candidate_aggregation must be one of {_AGGREGATIONS}")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {_ACTIVATIONS}")
        if self.synapse_mode not in _SYNAPSE_MODES:
            raise ValueError(f"synapse_mode must be one of {_SYNAPSE_MODES}")
        if self.cell_mode not in _CELL_MODES:
            raise ValueError(f"cell_mode must be one of {_CELL_MODES}")
        if self.offset_mode not in _OFFSET_MODES:
            raise ValueError(f"offset_mode must be one of {_OFFSET_MODES}")
        if self.cache_mode not in _CACHE_MODES:
            raise ValueError(f"cache_mode must be one of {_CACHE_MODES}")
        if self.evoker_mode not in _EVOKER_MODES:
            raise ValueError(f"evoker_mode must be one of {_EVOKER_MODES}")
        if self.gating_mode not in _GATING_MODES:
            raise ValueError(f"gating_mode must be one of {_GATING_MODES}")
        if self.norm_type not in _NORM_TYPES:
            raise ValueError(f"norm_type must be one of {_NORM_TYPES}")
        if self.dtype not in _DTYPE_MAP:
            raise ValueError(f"dtype must be one of {tuple(_DTYPE_MAP)}")
        if self.offsets is None:
            self.offsets = list(_DEFAULT_OFFSETS)
        if any(o < 0 for o in self.offsets):
            raise ValueError("All positional offsets must be non-negative (>= 0)")

        # -- depth rule: L >= ceil(log2(N)) ---------------------------------
        if self.offset_mode in (
            "hierarchical_dyadic",
            "binary_minimal",
            "resonant_multirate",
        ):
            required_layers = max(1, math.ceil(math.log2(max(2, self.context_length))))
            if self.num_consolidation_layers < required_layers and not self.global_anchor:
                warnings.warn(
                    f"[ENGRAMA] Depth rule: with "
                    f"num_consolidation_layers={self.num_consolidation_layers} and "
                    f"offset_mode='{self.offset_mode}' the binary receptive field "
                    f"covers ~{self.receptive_field()['max_reach']} positions, below "
                    f"context_length={self.context_length} (recommended "
                    f"L >= {required_layers}, or enable global_anchor=True).",
                    stacklevel=3,
                )

    # ------------------------------------------------------------------
    # Offset families and receptive field
    # ------------------------------------------------------------------
    def get_layer_offsets(
        self, layer_idx: int, total_layers: Optional[int] = None
    ) -> List[int]:
        """Return the relative offset family ``D_l`` for consolidation layer l."""
        if not 0 <= layer_idx < self.num_consolidation_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range for "
                f"{self.num_consolidation_layers} consolidation layers"
            )
        if total_layers is None:
            total_layers = self.num_consolidation_layers

        if self.offset_mode == "dense_dilated":
            res = list(self.offsets)
        elif self.offset_mode == "hierarchical_dyadic":
            dyadic = 2 ** layer_idx
            if dyadic == 1:
                res = [0, 1]
            elif dyadic < self.context_length:
                res = [0, 1, dyadic]
            else:
                res = [0, 1]
        elif self.offset_mode == "binary_minimal":
            dyadic = 2 ** layer_idx
            if dyadic == 1:
                res = [0, 1]
            elif dyadic < self.context_length:
                res = [0, dyadic]
            else:
                res = [0, 1]
        elif self.offset_mode == "resonant_multirate":
            if layer_idx == 0:
                res = [0, 1]
            else:
                res = [0, 1]
                p_prev = 2 ** (layer_idx - 1)
                p_curr = 2 ** layer_idx
                if p_prev < self.context_length:
                    res.append(p_prev)
                if p_curr < self.context_length:
                    res.append(p_curr)
        else:  # pragma: no cover - guarded by __post_init__
            res = list(self.offsets)

        # Global anchor only on the last layer (V3/V4 spec).
        if self.global_anchor and layer_idx == total_layers - 1:
            anchor = self.context_length - 1
            if anchor > 0 and anchor not in res:
                res.append(anchor)

        return sorted(dict.fromkeys(res))

    def layer_offsets(self) -> List[List[int]]:
        """Return ``D_l`` for every consolidation layer."""
        return [self.get_layer_offsets(l) for l in range(self.num_consolidation_layers)]

    def receptive_field(self) -> Dict[str, Any]:
        """Compute the exact reachable offset set of the consolidation stack."""
        reachable = {0}
        for layer_idx in range(self.num_consolidation_layers):
            offsets = self.get_layer_offsets(layer_idx)
            reachable = {r + d for r in reachable for d in offsets}
        max_reach = max(reachable) if reachable else 0
        dense = all(i in reachable for i in range(max_reach + 1))
        required = max(1, math.ceil(math.log2(max(2, self.context_length))))
        return {
            "max_reach": max_reach,
            "dense_coverage": dense,
            "covers_context": max_reach >= self.context_length - 1,
            "context_length": self.context_length,
            "num_consolidation_layers": self.num_consolidation_layers,
            "required_layers_for_full_coverage": required,
            "layer_offsets": self.layer_offsets(),
        }

    def cache_horizons(self) -> List[int]:
        """Minimum retained states per consolidation layer.

        ``horizons[l]`` is the number of entries the buffer of layer output
        ``T_l`` must retain so that layer ``l + 1`` can still read all its
        offsets. The final layer only feeds the evoker (horizon 1).
        """
        horizons: List[int] = []
        for l in range(self.num_consolidation_layers):
            if l < self.num_consolidation_layers - 1:
                next_offsets = self.get_layer_offsets(l + 1)
                horizons.append(max(next_offsets) + 1)
            else:
                horizons.append(1)
        return horizons

    # ------------------------------------------------------------------
    # Torch dtype
    # ------------------------------------------------------------------
    def torch_dtype(self):
        """Return the configured ``torch.dtype`` object."""
        import torch

        return getattr(torch, _DTYPE_MAP[self.dtype])

    # ------------------------------------------------------------------
    # (De)serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a plain dictionary (resolved values)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngramaConfig":
        """Construct configuration from a dictionary."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = sorted(set(d) - known)
        if unknown:
            warnings.warn(
                f"[ENGRAMA] EngramaConfig.from_dict ignoring unknown key(s): "
                f"{unknown}. Known keys: {sorted(known)}",
                stacklevel=2,
            )
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def save(self, filepath: str) -> None:
        """Save configuration to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "EngramaConfig":
        """Load configuration from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Size presets (quick mode)
    # ------------------------------------------------------------------
    @classmethod
    def preset(cls, size: str, **overrides: Any) -> "EngramaConfig":
        """Return a ready-to-train preset configuration."""
        size = size.lower()
        presets: Dict[str, Dict[str, Any]] = {
            "tiny": dict(
                d_model=64, d_gate=8, d_ff=256, num_cells=2,
                num_encoder_layers=1, num_consolidation_layers=6,
                context_length=64, synapse_rank=8, version="v4",
            ),
            "small": dict(
                d_model=128, d_gate=16, d_ff=512, num_cells=4,
                num_encoder_layers=1, num_consolidation_layers=8,
                context_length=256, synapse_rank=16, version="v4",
            ),
            "base": dict(
                d_model=256, d_gate=32, d_ff=1024, num_cells=8,
                num_encoder_layers=2, num_consolidation_layers=8,
                context_length=256, synapse_rank=32, version="v4",
            ),
            "large": dict(
                d_model=512, d_gate=64, d_ff=2048, num_cells=16,
                num_encoder_layers=2, num_consolidation_layers=11,
                context_length=2048, synapse_rank=32, version="v4",
            ),
        }
        if size not in presets:
            raise ValueError(f"Unknown size preset {size!r}; choose from {tuple(presets)}")
        kwargs = presets[size]
        kwargs.update(overrides)
        return cls(**kwargs)

    def describe(self) -> str:
        """Human-readable summary of the resolved architecture."""
        rf = self.receptive_field()
        lines = [
            f"ENGRAMA {self.version.upper()} resolved configuration",
            f"  vocab={self.vocab_size} d_model={self.d_model} d_gate={self.d_gate} "
            f"d_ff={self.d_ff} C={self.num_cells}",
            f"  L_enc={self.num_encoder_layers} L={self.num_consolidation_layers} "
            f"N_max={self.context_length} M={self.num_candidates}",
            f"  synapse_mode={self.synapse_mode} (r={self.synapse_rank}, "
            f"identity_transport={self.identity_transport})",
            f"  cell_mode={self.cell_mode} offset_mode={self.offset_mode} "
            f"cache_mode={self.cache_mode} evoker_mode={self.evoker_mode} "
            f"candidate_aggregation={self.candidate_aggregation}",
            f"  gating_mode={self.gating_mode} trace_tap={self.trace_tap} "
            f"norm_type={self.norm_type}",
            f"  hierarchical_gate={self.hierarchical_gate} "
            f"global_anchor={self.global_anchor} stable_init={self.stable_init}",
            f"  receptive_field: max_reach={rf['max_reach']} "
            f"dense={rf['dense_coverage']} covers_context={rf['covers_context']}",
        ]
        return "\n".join(lines)
