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
        version (str): Architecture version specification ('v1', 'v2'). Default: 'v2'.
        tie_embeddings (bool): Whether Evoker weights tie with input embeddings. Default: True.
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
    version: str = "v2"
    tie_embeddings: bool = True

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
        if self.version not in ("v1", "v2"):
            raise ValueError("version must be 'v1' or 'v2'")
        if any(o < 0 for o in self.offsets):
            raise ValueError("All positional offsets must be non-negative (>= 0)")

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
