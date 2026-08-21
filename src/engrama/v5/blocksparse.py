"""ENGRAMA V5 — Block-Sparse Synaptic Resonance (sub-quadratic, no compression).

The dense resonance read is O(N^2): every query gates against every past key.
This module reduces that cost **natively** by exploiting a property attention
does *not* have: the ENGRAMA gate ``sigma(tau*<q,k>+b)`` is **not normalized**
over positions, so a whole block of keys that does not resonate contributes ~0
and can be skipped without changing the result meaningfully.

Mechanism (content routing + block pruning, NOT compression):

1. Split the explicit trace into blocks of ``block_size`` tokens.
2. Summarize each key block by a **landmark** = normalized mean of its keys
   (a routing index only; the individual keys/values stay fully explicit).
3. For each query block, route to the ``top_k`` most-resonant causal key blocks
   (always including its own block for locality).
4. Compute exact per-pair sigmoid gates **only inside the selected blocks** and
   read their explicit values (fused Triton kernel on GPU, identical PyTorch
   reference on CPU).

Cost: with fixed ``top_k`` each query examines ``top_k * block_size`` keys — a
**constant**, so the total is **O(N)**; with ``top_k`` growing slowly it is
O(N*sqrt(N)). Either way it is drastically below O(N^2), and nothing is
compressed: all N key/value vectors remain explicit in the trace.

Setting ``top_k >= number of causal blocks`` recovers the dense result exactly
(verified in tests).

Author: ENGRAMA contributors.
License: AGPL-3.0
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from engrama.v5.primitives import make_norm
from engrama.v5.triton_kernels import resonance_blocksparse


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y)) if y < 20 else y


class BlockSparseResonance(nn.Module):
    """Sub-quadratic synaptic-resonance read via content routing + block pruning."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int = 128,
        top_k: int = 8,
        read_norm: Optional[str] = None,
        tau_init: float = 4.0,
        norm_type: str = "rmsnorm",
        eps: float = 1e-4,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.h = num_heads
        self.dh = d_model // num_heads
        self.block_size = block_size
        self.top_k = top_k
        self.read_norm = read_norm
        self.eps = eps

        self.norm = make_norm(norm_type, d_model)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.tau_raw = nn.Parameter(torch.full((num_heads,), _inv_softplus(tau_init)))
        self.gate_bias = nn.Parameter(torch.zeros(num_heads))

    def tau(self) -> torch.Tensor:
        return F.softplus(self.tau_raw)

    def _project(self, x: torch.Tensor):
        B, N, _ = x.shape
        xn = self.norm(x)
        q = self.wq(xn).view(B, N, self.h, self.dh).transpose(1, 2)
        k = self.wk(xn).view(B, N, self.h, self.dh).transpose(1, 2)
        v = self.wv(xn).view(B, N, self.h, self.dh).transpose(1, 2)
        return F.normalize(q, dim=-1), F.normalize(k, dim=-1), v

    def _route(self, q, k):
        """Return routing indices (B,H,nb,top_k), long, -1 for empty slots.

        Uses per-block query/key landmarks (normalized means). Always includes
        the query's own block (locality) and enforces causality.
        """
        Bsz, H, N, dh = q.shape
        Bk = self.block_size
        nb = (N + Bk - 1) // Bk
        pad = nb * Bk - N
        qb = F.pad(q, (0, 0, 0, pad)).view(Bsz, H, nb, Bk, dh)
        kb = F.pad(k, (0, 0, 0, pad)).view(Bsz, H, nb, Bk, dh)
        qland = F.normalize(qb.mean(dim=3), dim=-1)      # (B,H,nb,dh)
        kland = F.normalize(kb.mean(dim=3), dim=-1)      # (B,H,nb,dh)
        route = torch.einsum("bhqd,bhkd->bhqk", qland, kland)  # (B,H,nb,nb)
        # causal block mask: query block iq can see key blocks <= iq
        qi = torch.arange(nb, device=q.device).view(1, 1, nb, 1)
        ki = torch.arange(nb, device=q.device).view(1, 1, 1, nb)
        route = route.masked_fill(ki > qi, -1e9)
        route = route + torch.where(ki == qi, torch.full_like(route, 1e9), torch.zeros_like(route))
        tk = min(self.top_k, nb)
        idx = route.topk(tk, dim=-1).indices  # (B,H,nb,tk)
        # mark invalid (future) picks as -1
        picked = torch.gather(route, -1, idx)
        idx = torch.where(picked <= -1e8, torch.full_like(idx, -1), idx)
        return idx, nb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        q, k, v = self._project(x)
        idx, nb = self._route(q, k)
        r = resonance_blocksparse(
            q, k, v, self.tau(), self.gate_bias, idx,
            block=self.block_size, read_norm=self.read_norm,
        )
        r = r.transpose(1, 2).reshape(B, N, d)
        return x + self.wo(r)
