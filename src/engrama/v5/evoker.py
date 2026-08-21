"""ENGRAMA V5 Latent-Fusion Evoker (Phase 4, carried over from V4).

Generates ``M`` low-rank candidate vectors from the final state, fuses them
adaptively in latent space (softmax over the *M candidates*, never over
positions), and projects once to the vocabulary — O(|V| d), zero checkpoints.

Author: ENGRAMA contributors.
License: AGPL-3.0
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LatentFusionEvoker(nn.Module):
    def __init__(self, d_model: int, num_candidates: int, rank: int = 32):
        super().__init__()
        self.d_model = d_model
        self.m = num_candidates
        self.rank = min(rank, d_model)
        self.w_shared = nn.Linear(d_model, d_model, bias=False)
        self.u_e = nn.Parameter(torch.randn(d_model, self.rank) * 0.01)
        self.v_e = nn.Parameter(torch.randn(d_model, self.rank) * 0.01)
        self.s_m = nn.Parameter(torch.zeros(num_candidates, self.rank))
        self.b_m = nn.Parameter(torch.zeros(num_candidates, d_model))
        self.gate = nn.Linear(d_model, num_candidates)

    def fused_latent(self, h: torch.Tensor) -> torch.Tensor:
        base = self.w_shared(h)
        z = h @ self.v_e
        low_rank = (z.unsqueeze(-2) * self.s_m) @ self.u_e.T  # (..., M, d)
        cands = base.unsqueeze(-2) + low_rank + self.b_m
        logits = self.gate(h)
        if logits.dtype in (torch.float16, torch.bfloat16):
            w = F.softmax(logits.float(), dim=-1).to(cands.dtype)
        else:
            w = F.softmax(logits, dim=-1)
        return (cands * w.unsqueeze(-1)).sum(dim=-2)

    def forward(self, h: torch.Tensor, embedding_weight: torch.Tensor) -> torch.Tensor:
        latent = self.fused_latent(h)
        return F.linear(latent, embedding_weight) * (1.0 / math.sqrt(self.d_model))
