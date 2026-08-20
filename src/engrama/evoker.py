"""ENGRAMA Multi-Candidate Evoker (Phase 4).

Implements the evoker (V3/V4):

::

    c_m = W_shared h_* + U_e Diag(s_m) V_e^T h_* + b_m      (factorized)
    c_m = W_m h_* + b_m                                    (dense)

    l_{m,v} = <c_m, E_v> / sqrt(d)

Candidate aggregation modes:
- ``latent_fusion`` (V4 default): Candidates are adaptively fused in latent space
  via a learned gating distribution before the vocabulary projection, achieving
  O(|V|d) cost with zero gradient checkpoints.
- ``mean`` (V3): Arithmetic mean in latent space before vocabulary projection.
- ``logsumexp`` / ``max``: Multi-candidate logit aggregation with chunked
  memory guards for very large vocabularies.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from engrama.config import EngramaConfig

# Memory guard for very large vocabularies in logsumexp/max paths.
_MAX_AGGREGATE_ELEMENTS = 2 ** 26  # 67M elements == 256 MB in fp32
_EVOKER_CHUNK = 2048


def _logsumexp_chunk(cands: torch.Tensor, weight: torch.Tensor, scale: float) -> torch.Tensor:
    """One vocabulary chunk of the candidate logsumexp aggregation."""
    logits = F.linear(cands, weight) * scale  # (P, M, c)
    m = logits.max(dim=-2, keepdim=True).values  # (P, 1, c)
    return m.squeeze(-2) + torch.log(torch.sum(torch.exp(logits - m), dim=-2))


def _max_chunk(cands: torch.Tensor, weight: torch.Tensor, scale: float) -> torch.Tensor:
    """One vocabulary chunk of the candidate max aggregation."""
    logits = F.linear(cands, weight) * scale  # (P, M, c)
    return logits.max(dim=-2).values  # (P, c)


class MultiCandidateEvoker(nn.Module):
    """Multi-candidate recall from the last consolidated state ``h_*``."""

    def __init__(self, config: EngramaConfig):
        super().__init__()
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.num_candidates = config.num_candidates
        self.aggregation = config.candidate_aggregation
        self.evoker_mode = config.evoker_mode
        self.synapse_rank = min(config.synapse_rank, config.d_model)

        if self.evoker_mode == "factorized":
            # Shared trunk
            self.W_shared = nn.Linear(self.d_model, self.d_model, bias=False)
            self.U_e = nn.Parameter(torch.randn(self.d_model, self.synapse_rank) * 0.01)
            self.V_e = nn.Parameter(torch.randn(self.d_model, self.synapse_rank) * 0.01)
            self.s_m = nn.Parameter(torch.zeros(self.num_candidates, self.synapse_rank))
            self.b_m = nn.Parameter(torch.zeros(self.num_candidates, self.d_model))
        elif self.evoker_mode == "dense":
            self.candidates = nn.ModuleList(
                [
                    nn.Linear(self.d_model, self.d_model)
                    for _ in range(self.num_candidates)
                ]
            )
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"Unknown evoker_mode: {self.evoker_mode!r}")

        if self.aggregation == "latent_fusion":
            self.gate_fusion = nn.Linear(self.d_model, self.num_candidates)
        else:
            self.gate_fusion = None

    # ------------------------------------------------------------------
    def candidates_forward(self, h_star: torch.Tensor) -> torch.Tensor:
        """Return all candidate vectors, shape ``(..., M, d)``."""
        if self.evoker_mode == "factorized":
            base = self.W_shared(h_star)  # (..., d)
            z = h_star @ self.V_e  # (..., r)
            low_rank = (z.unsqueeze(-2) * self.s_m) @ self.U_e.T  # (..., M, d)
            return base.unsqueeze(-2) + low_rank + self.b_m
        return torch.stack([cand(h_star) for cand in self.candidates], dim=-2)

    def forward(
        self, h_star: torch.Tensor, embedding_weights: torch.Tensor
    ) -> torch.Tensor:
        """Map ``h_*`` to vocabulary logits of shape ``(..., vocab_size)``."""
        scale = 1.0 / math.sqrt(self.d_model)
        cands = self.candidates_forward(h_star)  # (..., M, d)

        if self.aggregation == "latent_fusion":
            # V4 latent adaptive candidate fusion: aggregate in R^d before vocab projection
            # Fusion softmax in fp32 under AMP (M is tiny; vocab GEMM stays fp16).
            fusion_logits = self.gate_fusion(h_star)
            if fusion_logits.dtype in (torch.float16, torch.bfloat16):
                w = F.softmax(fusion_logits.float(), dim=-1).to(cands.dtype).unsqueeze(-1)
            else:
                w = F.softmax(fusion_logits, dim=-1).unsqueeze(-1)  # (..., M, 1)
            c_fused = (cands * w).sum(dim=-2)  # (..., d)
            return F.linear(c_fused, embedding_weights) * scale

        if self.aggregation == "mean":
            # V3 section 14.2 / 37: arithmetic mean in R^d
            c_mean = cands.mean(dim=-2)  # (..., d)
            return F.linear(c_mean, embedding_weights) * scale

        flat = cands.reshape(-1, self.num_candidates, self.d_model)  # (P, M, d)
        total = flat.shape[0] * self.num_candidates * self.vocab_size
        if total <= _MAX_AGGREGATE_ELEMENTS:
            logits = F.linear(cands, embedding_weights) * scale  # (..., M, V)
            if self.aggregation == "logsumexp":
                m = logits.max(dim=-2, keepdim=True).values
                return m.squeeze(-2) + torch.log(
                    torch.sum(torch.exp(logits - m), dim=-2)
                )
            if self.aggregation == "max":
                return logits.max(dim=-2).values
            raise ValueError(f"Unknown aggregation: {self.aggregation!r}")

        return self._aggregate_chunked(
            flat, embedding_weights, scale, cands.shape, self.aggregation
        )

    # ------------------------------------------------------------------
    def _aggregate_chunked(
        self,
        flat_cands: torch.Tensor,
        embedding_weights: torch.Tensor,
        scale: float,
        output_shape: torch.Size,
        aggregation: str,
    ) -> torch.Tensor:
        """Chunked candidate aggregation over the vocabulary axis."""
        if aggregation == "logsumexp":
            fn = _logsumexp_chunk
        elif aggregation == "max":
            fn = _max_chunk
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"Unknown aggregation: {aggregation!r}")

        pieces: List[torch.Tensor] = []
        scale_t = flat_cands.new_tensor(scale)
        for v0 in range(0, self.vocab_size, _EVOKER_CHUNK):
            v1 = min(self.vocab_size, v0 + _EVOKER_CHUNK)
            weight_chunk = embedding_weights[v0:v1]
            piece = checkpoint(
                fn, flat_cands, weight_chunk, scale_t, use_reentrant=True
            )
            pieces.append(piece)
        out = torch.cat(pieces, dim=-1)
        return out.reshape(output_shape[:-2] + (self.vocab_size,))

    # ------------------------------------------------------------------
    def candidate_diversity(self, h_star: torch.Tensor) -> torch.Tensor:
        """Pairwise cosine distance between candidates (inspection helper)."""
        c = self.candidates_forward(h_star).reshape(-1, self.num_candidates, self.d_model)
        c_n = F.normalize(c, dim=-1)
        sim = torch.einsum("bmd,bnd->bmn", c_n, c_n)  # (B, M, M)
        return 1.0 - sim
