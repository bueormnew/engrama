"""ENGRAMA V5 configuration.

A single, small dataclass — no version presets, no per-offset knobs. The V5
architecture is intentionally simple: isolated encoder, stack of synaptic
resonance blocks, latent-fusion evoker.

Author: ENGRAMA contributors.
License: AGPL-3.0
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

_DTYPES = {"float32", "float16", "bfloat16"}
_NORMS = ("rmsnorm", "layernorm")
_READ_NORMS = (None, "softcount")
_ACTS = ("gelu", "silu", "relu")


@dataclass
class EngramaV5Config:
    """Configuration for :class:`~engrama.v5.model.EngramaV5`.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension ``d``.
        num_layers: Number of synaptic-resonance blocks.
        num_heads: Number of resonance heads ``H`` (``d_model % num_heads == 0``).
        d_ff: Feed-forward expansion inside each Cell (default ``4 * d_model``).
        context_length: Maximum trace length ``N_max`` used by the cache.
        num_candidates: Evoker candidates ``M`` (latent fusion).
        num_encoder_layers: Isolated-encoder depth (per-token MLP mixing).
            ``0`` (default) uses the embedding itself as the isolated footprint,
            which keeps the raw content signal intact for synaptic resonance.
            Extra per-token Cells can blur the key/query content, so enable only
            if you specifically need a deeper per-token encoder.
        read_norm: ``None`` (pure Hebbian superposition) or ``"softcount"``
            (divide the read by the summed gate mass — NOT a softmax; it is a
            per-token scalar count, so positions never compete).
        tau_init: Initial per-head resonance sharpness (inverse temperature).
        activation: Cell activation function.
        norm_type: ``"rmsnorm"`` (default) or ``"layernorm"``.
        dropout: Dropout probability inside the Cell.
        tie_embeddings: Tie the output projection to the input embedding.
        chunk_size: Causal tiling block size for long-context training/eval.
            ``0`` disables tiling (full ``N×N`` score in one matmul).
        dtype: Parameter/compute dtype.
    """

    vocab_size: int = 32000
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    d_ff: Optional[int] = None
    context_length: int = 8192
    num_candidates: int = 4
    num_encoder_layers: int = 0

    read_norm: Optional[str] = None
    tau_init: float = 4.0
    activation: str = "gelu"
    norm_type: str = "rmsnorm"
    dropout: float = 0.0
    tie_embeddings: bool = True
    chunk_size: int = 0
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if self.d_model < 1:
            raise ValueError("d_model must be >= 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.num_encoder_layers < 0:
            raise ValueError("num_encoder_layers must be >= 0")
        if self.context_length < 1:
            raise ValueError("context_length must be >= 1")
        if not (1 <= self.num_candidates <= 8):
            raise ValueError("num_candidates must be in [1, 8]")
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_ff < 1:
            raise ValueError("d_ff must be >= 1")
        if self.read_norm not in _READ_NORMS:
            raise ValueError(f"read_norm must be one of {_READ_NORMS}")
        if self.activation not in _ACTS:
            raise ValueError(f"activation must be one of {_ACTS}")
        if self.norm_type not in _NORMS:
            raise ValueError(f"norm_type must be one of {_NORMS}")
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
        if self.chunk_size < 0:
            raise ValueError("chunk_size must be >= 0")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def torch_dtype(self):
        import torch

        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype]

    # -- (de)serialization ------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngramaV5Config":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EngramaV5Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def preset(cls, size: str, **overrides: Any) -> "EngramaV5Config":
        presets = {
            "tiny": dict(d_model=128, num_layers=4, num_heads=4, context_length=1024),
            "small": dict(d_model=256, num_layers=6, num_heads=8, context_length=2048),
            "base": dict(d_model=512, num_layers=8, num_heads=8, context_length=4096),
            "large": dict(d_model=768, num_layers=12, num_heads=12, context_length=8192),
        }
        size = size.lower()
        if size not in presets:
            raise ValueError(f"unknown preset {size!r}; choose {tuple(presets)}")
        kw = dict(presets[size])
        kw.update(overrides)
        return cls(**kw)

    def describe(self) -> str:
        return (
            f"ENGRAMA V5  vocab={self.vocab_size} d_model={self.d_model} "
            f"L={self.num_layers} H={self.num_heads} d_ff={self.d_ff}\n"
            f"  N_max={self.context_length} M={self.num_candidates} "
            f"read_norm={self.read_norm} norm={self.norm_type} dtype={self.dtype}"
        )
