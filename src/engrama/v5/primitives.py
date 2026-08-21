"""ENGRAMA V5 primitive blocks: per-vector norms, isolated encoder, cell.

All primitives operate strictly per token (never across positions), preserving
the isolated-encoding pillar. Norm statistics are computed in fp32 under AMP to
guarantee NaN-free training.

Author: ENGRAMA contributors.
License: AGPL-3.0
"""

from __future__ import annotations

import torch
from torch import nn


def activation(name: str) -> nn.Module:
    return {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}[name]()


class RMSNorm(nn.Module):
    """RMSNorm over the last dimension (fp32 stats under AMP)."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x32 = x.float() if dt in (torch.float16, torch.bfloat16) else x
        rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * rms).to(dt) * self.gamma.to(dt)


class LayerNorm(nn.Module):
    """LayerNorm over the last dimension (fp32 stats under AMP)."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x32 = x.float() if dt in (torch.float16, torch.bfloat16) else x
        mean = x32.mean(-1, keepdim=True)
        var = x32.var(-1, keepdim=True, unbiased=False)
        y = (x32 - mean) * torch.rsqrt(var + self.eps)
        return (y.to(dt)) * self.gamma.to(dt) + self.beta.to(dt)


def make_norm(name: str, d: int) -> nn.Module:
    return RMSNorm(d) if name == "rmsnorm" else LayerNorm(d)


class Cell(nn.Module):
    """ENGRAMA Cell: ``x + W2 * act(W1 * Norm(x))`` (pre-norm residual FFN)."""

    def __init__(self, d: int, d_ff: int, dropout: float, act: str, norm: str):
        super().__init__()
        self.norm = make_norm(norm, d)
        self.w1 = nn.Linear(d, d_ff)
        self.w2 = nn.Linear(d_ff, d)
        self.act = activation(act)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.w2(self.drop(self.act(self.w1(self.norm(x)))))


class IsolatedEncoder(nn.Module):
    """Per-token isolated footprint encoder.

    Each token is encoded independently of its neighbours (isolated-encoding
    pillar): a small stack of per-token Cells. Because no operation mixes
    positions, ``T0[i]`` depends only on ``x_i``.

    With ``layers == 0`` this is an identity: the embedding *is* the isolated
    footprint, which keeps the raw content signal maximally intact for the
    synaptic-resonance read (empirically the best default for retrieval).
    """

    def __init__(self, d: int, d_ff: int, layers: int, dropout: float,
                 act: str, norm: str):
        super().__init__()
        self.cells = nn.ModuleList(
            [Cell(d, d_ff, dropout, act, norm) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for cell in self.cells:
            x = cell(x)
        return x
