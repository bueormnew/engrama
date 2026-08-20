"""ENGRAMA Configuration Module (V3).

Implements the configuration surface defined by ENGRAMA V3
(``ENGRAMA-V3-Teorica.md``, sections 26, 27, 38, 54 and 55):

- ``version`` is a *real* architecture preset: ``"v2"`` resolves every
  architecture mode to its V2 form (dense synapses, dense dilated offsets,
  full cache, independent cells, dense evoker) and ``"v3"`` to the V3 form
  (factorized synapses with identity transport, hierarchical dyadic offsets,
  hierarchical minimum-horizon cache, shared-core cells, factorized evoker).
  Any mode passed explicitly overrides the preset, which enables the ablation
  suites of V3 sections 43-44 directly from the config object.
- ``cache_mode`` selects ``"full"`` (V2 causal cache) or ``"hierarchical"``
  (V3 minimum-horizon cache, theorem of V3 section 24).
- Depth rule (V3 section 26): ``num_consolidation_layers >= ceil(log2(N))``
  for full binary coverage. Violations emit a descriptive warning.

Author: BUEORM
License: AGPL-3.0
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Version presets (V3 spec, section 54)
# ---------------------------------------------------------------------------

_SYNAPSE_MODES = ("dense", "factorized")
_CELL_MODES = ("independent", "shared_core")
_OFFSET_MODES = ("dense_dilated", "hierarchical_dyadic", "binary_minimal")
_CACHE_MODES = ("full", "hierarchical")
_EVOKER_MODES = ("dense", "factorized")
_AGGREGATIONS = ("max", "logsumexp", "mean")
_ACTIVATIONS = ("gelu", "relu", "silu")
_DTYPE_MAP = {
    "float32": "float32",
    "float64": "float64",
    "float16": "float16",
    "bfloat16": "bfloat16",
}

VERSION_PRESETS: Dict[str, Dict[str, Any]] = {
    # V1 shares the V2 dense parameterization (the library implements the
    # cached algorithm of V2 for both); it is kept for provenance.
    "v1": {
        "synapse_mode": "dense",
        "cell_mode": "independent",
        "offset_mode": "dense_dilated",
        "cache_mode": "full",
        "evoker_mode": "dense",
        "identity_transport": False,
        "hierarchical_gate": False,
    },
    "v2": {
        "synapse_mode": "dense",
        "cell_mode": "independent",
        "offset_mode": "dense_dilated",
        "cache_mode": "full",
        "evoker_mode": "dense",
        "identity_transport": False,
        "hierarchical_gate": False,
    },
    "v3": {
        "synapse_mode": "factorized",
        "cell_mode": "shared_core",
        "offset_mode": "hierarchical_dyadic",
        "cache_mode": "hierarchical",
        "evoker_mode": "factorized",
        "identity_transport": True,
        "hierarchical_gate": True,
    },
}

_DEFAULT_OFFSETS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


@dataclass
class EngramaConfig:
    """Configuration dataclass for ENGRAMA models (expert mode).

    Every architecture mode may be left as ``None`` to inherit the value
    prescribed by the ``version`` preset, or set explicitly to compose any
    V2/V3 ablation (V3 spec, sections 43-44 and 54).

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
        candidate_aggregation: ``"max"`` | ``"logsumexp"`` | ``"mean"``.
        activation: ``"gelu"`` | ``"relu"`` | ``"silu"``.
        dropout: Dropout probability. Default: 0.0.
        dtype: Model/trace precision. Default: ``"float32"``.
        version: Architecture preset: ``"v1"`` | ``"v2"`` | ``"v3"``.
        tie_embeddings: Tie evoker projection to input embeddings.
        synapse_mode: ``"dense"`` (V2) or ``"factorized"`` (V3 section 6).
        synapse_rank: Low-rank dimension ``r << d``. Default: 32.
        identity_transport: Enable the identity route ``beta * h`` (V3 §6.4).
        cell_mode: ``"independent"`` or ``"shared_core"`` (V3 section 5).
        offset_mode: ``"dense_dilated"`` | ``"hierarchical_dyadic"`` |
            ``"binary_minimal"`` (V3 section 27).
        global_anchor: Add deterministic global anchor ``g(N)`` at the last
            consolidation layer (V3 section 11). Default: False.
        evoker_mode: ``"dense"`` | ``"factorized"`` (V3 section 14).
        cache_mode: ``"full"`` (V2) | ``"hierarchical"`` (V3 section 12).
        hierarchical_gate: Scalar per-scale gate ``rho`` (V3 section 17).
        stable_init: Initialize synapses near the identity route
            (``s ~ 0``, ``beta = 1``), per V3 section 32. Default: True.
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
    candidate_aggregation: str = "logsumexp"
    activation: str = "gelu"
    dropout: float = 0.0
    dtype: str = "float32"
    version: str = "v3"
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
            "identity_transport",
            "hierarchical_gate",
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
        if self.dtype not in _DTYPE_MAP:
            raise ValueError(f"dtype must be one of {tuple(_DTYPE_MAP)}")
        if self.offsets is None:
            self.offsets = list(_DEFAULT_OFFSETS)
        if any(o < 0 for o in self.offsets):
            raise ValueError("All positional offsets must be non-negative (>= 0)")

        # -- depth rule (V3 spec, section 26): L >= ceil(log2(N)) -----------
        if self.offset_mode in ("hierarchical_dyadic", "binary_minimal"):
            required_layers = max(1, math.ceil(math.log2(max(2, self.context_length))))
            if self.num_consolidation_layers < required_layers and not self.global_anchor:
                warnings.warn(
                    f"[ENGRAMA] Depth rule (V3 spec 26): with "
                    f"num_consolidation_layers={self.num_consolidation_layers} and "
                    f"offset_mode='{self.offset_mode}' the binary receptive field "
                    f"covers ~{self.receptive_field()['max_reach']} positions, below "
                    f"context_length={self.context_length} (recommended "
                    f"L >= {required_layers}, or enable global_anchor=True).",
                    stacklevel=3,
                )

    # ------------------------------------------------------------------
    # Offset families and receptive field (V3 spec, sections 8, 25, 26, 27)
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
        else:  # pragma: no cover - guarded by __post_init__
            res = list(self.offsets)

        # Global anchor only on the last layer (V3 spec, section 11).
        if self.global_anchor and layer_idx == total_layers - 1:
            anchor = self.context_length - 1
            if anchor > 0 and anchor not in res:
                res.append(anchor)

        return sorted(dict.fromkeys(res))

    def layer_offsets(self) -> List[List[int]]:
        """Return ``D_l`` for every consolidation layer."""
        return [self.get_layer_offsets(l) for l in range(self.num_consolidation_layers)]

    def receptive_field(self) -> Dict[str, Any]:
        """Compute the exact reachable offset set of the consolidation stack.

        Returns a dict with the maximum reachable distance, whether coverage
        is dense over ``[0, max]``, the required layers for full binary
        coverage of ``context_length`` and the per-layer offsets.
        """
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
        """Minimum retained states per consolidation layer (V3 §12 teorema 2).

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
        """Construct configuration from a dictionary, ignoring unknown keys."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
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
    # Size presets (quick mode) -- V3 spec, section 38 spirit
    # ------------------------------------------------------------------
    @classmethod
    def preset(cls, size: str, **overrides: Any) -> "EngramaConfig":
        """Return a ready-to-train preset configuration.

        Sizes follow the V3 recommended-profile spirit (section 38): dyadic
        offsets, factorized synapses with identity transport and a depth that
        satisfies the ``L >= ceil(log2(N))`` coverage rule for the preset
        context length unless overridden.
        """
        size = size.lower()
        presets: Dict[str, Dict[str, Any]] = {
            "tiny": dict(
                d_model=64, d_gate=8, d_ff=256, num_cells=2,
                num_encoder_layers=1, num_consolidation_layers=6,
                context_length=64, synapse_rank=8,
            ),
            "small": dict(
                d_model=128, d_gate=16, d_ff=512, num_cells=4,
                num_encoder_layers=1, num_consolidation_layers=8,
                context_length=256, synapse_rank=16,
            ),
            "base": dict(
                d_model=256, d_gate=32, d_ff=1024, num_cells=8,
                num_encoder_layers=2, num_consolidation_layers=8,
                context_length=256, synapse_rank=32,
            ),
            "large": dict(
                d_model=512, d_gate=64, d_ff=2048, num_cells=16,
                num_encoder_layers=2, num_consolidation_layers=11,
                context_length=2048, synapse_rank=32,
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
            f"cache_mode={self.cache_mode} evoker_mode={self.evoker_mode}",
            f"  hierarchical_gate={self.hierarchical_gate} "
            f"global_anchor={self.global_anchor} stable_init={self.stable_init}",
            f"  receptive_field: max_reach={rf['max_reach']} "
            f"dense={rf['dense_coverage']} covers_context={rf['covers_context']}",
        ]
        return "\n".join(lines)
