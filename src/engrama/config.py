"""
ENGRAMA Configuration Module
Author: BUEORM
License: AGPL-3.0
"""

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EngramaConfig:
    """Configuration dataclass for ENGRAMA models.

    Args:
        vocab_size (int): Vocabulary size. Default: 256.
        d_model (int): Hidden dimension size. Default: 512.
        d_gate (int): Gating latent projection size (d_gate < d_model). Default: 64.
        d_ff (int): Cell feed-forward expansion dimension. Default: 2048.
        num_cells (int): Number of parallel cellular representations in Synapse layers. Default: 16.
        num_encoder_layers (int): Number of IsolatedEncoder Synapse layers. Default: 2.
        num_consolidation_layers (int): Number of Consolidation Stack layers. Default: 6.
        context_length (int): Maximum sequence context length (N_max). Default: 2048.
        offsets (Optional[List[int]]): Dilated positional offset powers of 2. Default: [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].
        num_candidates (int): Number of recall candidates in MultiCandidateEvoker (M in [1, 8]). Default: 4.
        candidate_aggregation (str): Aggregation method ('logsumexp', 'max', 'mean'). Default: 'logsumexp'.
        activation (str): Activation function for Cells ('gelu', 'relu', 'silu'). Default: 'gelu'.
        dropout (float): Dropout probability. Default: 0.0.
        dtype (str): Tensor precision format. Default: 'float32'.
        version (str): Architecture version specification ('v1', 'v2', 'v3'). Default: 'v3'.
        tie_embeddings (bool): Whether Evoker weights tie with input embeddings. Default: True.
        synapse_mode (str): Synapse routing mode ('dense', 'factorized'). Default: 'factorized'.
        synapse_rank (int): Low-rank dimension for factorized synapses (r << d_model). Default: 32.
        identity_transport (bool): Enables explicit identity transport path in synapses. Default: True.
        cell_mode (str): Cellular non-linear transformation mode ('independent', 'shared_core'). Default: 'shared_core'.
        offset_mode (str): Consolidation positional offset scheme ('dense_dilated', 'hierarchical_dyadic', 'binary_minimal'). Default: 'hierarchical_dyadic'.
        global_anchor (bool): Whether to include a deterministic global anchor offset g(N) at final layer. Default: False.
        evoker_mode (str): Evoker candidate projection mode ('dense', 'factorized'). Default: 'factorized'.
        hierarchical_cache (bool): Minimum horizon trace cache pruning for V3. Default: True.
    """

    vocab_size: int = 256
    d_model: int = 512
    d_gate: int = 64
    d_ff: int = 2048
    num_cells: int = 16
    num_encoder_layers: int = 2
    num_consolidation_layers: int = 6
    context_length: int = 2048
    offsets: Optional[List[int]] = None
    num_candidates: int = 4
    candidate_aggregation: str = "logsumexp"
    activation: str = "gelu"
    dropout: float = 0.0
    dtype: str = "float32"
    version: str = "v3"
    tie_embeddings: bool = True
    synapse_mode: str = "factorized"
    synapse_rank: int = 32
    identity_transport: bool = True
    cell_mode: str = "shared_core"
    offset_mode: str = "hierarchical_dyadic"
    global_anchor: bool = False
    evoker_mode: str = "factorized"
    hierarchical_cache: bool = True

    def __post_init__(self) -> None:
        if self.offsets is None:
            self.offsets = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_gate >= self.d_model:
            raise ValueError(f"d_gate ({self.d_gate}) must be strictly less than d_model ({self.d_model})")
        if not (1 <= self.num_candidates <= 8):
            raise ValueError("num_candidates must be between 1 and 8 inclusive")
        if self.candidate_aggregation not in ("max", "logsumexp", "mean"):
            raise ValueError("candidate_aggregation must be 'max', 'logsumexp', or 'mean'")
        if self.activation not in ("gelu", "relu", "silu"):
            raise ValueError("activation must be 'gelu', 'relu', or 'silu'")
        if self.version not in ("v1", "v2", "v3"):
            raise ValueError("version must be 'v1', 'v2', or 'v3'")
        if self.synapse_mode not in ("dense", "factorized"):
            raise ValueError("synapse_mode must be 'dense' or 'factorized'")
        if self.cell_mode not in ("independent", "shared_core"):
            raise ValueError("cell_mode must be 'independent' or 'shared_core'")
        if self.offset_mode not in ("dense_dilated", "hierarchical_dyadic", "binary_minimal"):
            raise ValueError("offset_mode must be 'dense_dilated', 'hierarchical_dyadic', or 'binary_minimal'")
        if self.evoker_mode not in ("dense", "factorized"):
            raise ValueError("evoker_mode must be 'dense' or 'factorized'")
        if any(o < 0 for o in self.offsets):
            raise ValueError("All positional offsets must be non-negative (>= 0)")
        
        # When version is v1 or v2, adjust defaults if not explicitly set to v3 modes
        if self.version in ("v1", "v2"):
            # V1/V2 backward compatibility adjustments if legacy defaults are intended
            pass

    def get_layer_offsets(
        self, layer_idx: int, total_layers: Optional[int] = None
    ) -> List[int]:
        """Get positional relative offsets D_l for a specific consolidation layer l."""
        if total_layers is None:
            total_layers = self.num_consolidation_layers

        if self.offset_mode == "dense_dilated":
            res = list(self.offsets)
        elif self.offset_mode == "hierarchical_dyadic":
            if layer_idx == 0:
                res = [0, 1]
            else:
                dyadic = 2 ** layer_idx
                if dyadic < self.context_length:
                    res = [0, 1, dyadic]
                else:
                    res = [0, 1]
        elif self.offset_mode == "binary_minimal":
            dyadic = 2 ** layer_idx
            if dyadic < self.context_length:
                res = [0, dyadic]
            else:
                res = [0, 1]
        else:
            res = list(self.offsets)

        if self.global_anchor and layer_idx == total_layers - 1:
            anchor = self.context_length - 1
            if anchor > 0 and anchor not in res:
                res.append(anchor)

        # Deduplicate and sort
        unique_offsets = sorted(list(dict.fromkeys(res)))
        return unique_offsets

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngramaConfig":
        """Construct configuration from dictionary."""
        return cls(**d)

    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "EngramaConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
