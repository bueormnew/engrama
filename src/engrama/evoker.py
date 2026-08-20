"""ENGRAMA V3 Multi-Candidate Evoker (Phase 4).

Implements the evoker of V3 spec section 14:

::

    c_m = W_shared h_* + U_e Diag(s_m) V_e^T h_* + b_m      (factorized, V3)
    c_m = W_m h_* + b_m                                    (dense, V2 ablation)

    l_{m,v} = <c_m, E_v> / sqrt(d)

with aggregation over candidates by ``max``, ``logsumexp`` (numerically
stable) or ``mean``. The ``mean`` mode applies the V3 optimization of
sections 14.2 / 37: candidates are aggregated **before** the vocabulary
projection, reducing the dominant cost from ``O(M |V| d)`` to ``O(|V| d)``
with mathematically identical results (linearity of the mean).

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

# Memory guard for very large vocabularies (e.g. GPT-2's 50,257 tokens).
#
# The plain logsumexp/max path materializes a (..., M, V) logits tensor.
# At batch 16 x 512 positions x 4 candidates x 50,257 vocab that is 6.6 GB
# in fp32 -- plus softmax temporaries -- which exhausts a 16 GB T4 even
# before the backward pass. Above this many elements the evoker switches to
# a chunked-over-vocabulary path whose per-chunk intermediates are wrapped
# in a gradient checkpoint: peak memory stays O(P * chunk) instead of
# O(P * M * V), at the cost of recomputing each chunk once during backward.
_MAX_AGGREGATE_ELEMENTS = 2 ** 26  # 67M elements == 256 MB in fp32
_EVOKER_CHUNK = 2048


def _logsumexp_chunk(cands: torch.Tensor, weight: torch.Tensor, scale: float) -> torch.Tensor:
    """One vocabulary chunk of the candidate logsumexp aggregation.

    Inputs: ``cands`` (P, M, d), ``weight`` (c, d). Output: ``(P, c)``.
    Pure tensor function so it can run inside a gradient checkpoint.
    """
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
            # Truly shared trunk (V3 spec section 14).
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
        """Map ``h_*`` to vocabulary logits of shape ``(..., vocab_size)``.

        For ``logsumexp``/``max`` with a large candidate x vocab product this
        switches to a checkpointed chunked path (see module docstring) that
        keeps peak memory bounded; results are identical to the plain path.
        """
        scale = 1.0 / math.sqrt(self.d_model)
        cands = self.candidates_forward(h_star)  # (..., M, d)

        if self.aggregation == "mean":
            # V3 sections 14.2/37: aggregate first -- one vocabulary matmul.
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
        """Chunked candidate aggregation over the vocabulary axis.

        Each chunk is evaluated through a reentrant gradient checkpoint:
        its ``(P, M, chunk)`` intermediates are freed right after the chunk
        and recomputed once during backward, so peak memory is bounded by
        ``O(P * chunk)`` while the graph stays intact. The per-position
        results are mathematically identical to the plain path.
        """
        if aggregation == "logsumexp":
            fn = _logsumexp_chunk
        elif aggregation == "max":
            fn = _max_chunk
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"Unknown aggregation: {aggregation!r}")

        pieces: List[torch.Tensor] = []
        scale_t = flat_cands.new_tensor(scale)  # 0-dim tensor (checkpoint arg)
        for v0 in range(0, self.vocab_size, _EVOKER_CHUNK):
            v1 = min(self.vocab_size, v0 + _EVOKER_CHUNK)
            weight_chunk = embedding_weights[v0:v1]  # (c, d)
            piece = checkpoint(
                fn, flat_cands, weight_chunk, scale_t, use_reentrant=True
            )
            pieces.append(piece)  # (P, c)
        out = torch.cat(pieces, dim=-1)  # (P, V)
        return out.reshape(output_shape[:-2] + (self.vocab_size,))

    # ------------------------------------------------------------------
    def candidate_diversity(self, h_star: torch.Tensor) -> torch.Tensor:
        """Pairwise cosine distance between candidates (inspection helper)."""
        c = self.candidates_forward(h_star).reshape(-1, self.num_candidates, self.d_model)
        c_n = F.normalize(c, dim=-1)
        sim = torch.einsum("bmd,bnd->bmn", c_n, c_n)  # (B, M, M)
        return 1.0 - sim
