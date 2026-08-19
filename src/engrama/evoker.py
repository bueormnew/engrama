"""
ENGRAMA Multi-Candidate Evoker Module (Phase 4)
Author: BUEORM
License: AGPL-3.0
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class MultiCandidateEvoker(nn.Module):
    """ENGRAMA Multi-Candidate Memory Evoker (Phase 4).

    Generates M distinct candidate projections from final consolidated representations
    h* and aggregates token logits using LogSumExp, Max, or Mean over candidate similarity scores.

    Args:
        d_model (int): Hidden dimension size.
        vocab_size (int): Output vocabulary size.
        num_candidates (int): Number of recall candidates M (1 <= M <= 8).
        aggregation (str): Aggregation function ('logsumexp', 'max', 'mean').
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        num_candidates: int = 4,
        aggregation: str = "logsumexp",
    ):
        super().__init__()
        if not (1 <= num_candidates <= 8):
            raise ValueError("num_candidates must be between 1 and 8 inclusive")
        if aggregation.lower() not in ("max", "logsumexp", "mean"):
            raise ValueError("aggregation must be 'max', 'logsumexp', or 'mean'")

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_candidates = num_candidates
        self.aggregation = aggregation.lower()
        self.candidates = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_candidates)]
        )

    def forward(
        self, h_star: torch.Tensor, embedding_weights: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass to compute aggregated vocabulary logits.

        Args:
            h_star (Tensor): Consolidated hidden representation (B, N, D) or (B, D).
            embedding_weights (Tensor): Vocabulary embedding weight matrix (vocab_size, D).

        Returns:
            Tensor: Unnormalized token logits of shape (B, N, vocab_size) or (B, vocab_size).
        """
        is_3d = h_star.dim() == 3
        if is_3d:
            c = torch.stack([cand(h_star) for cand in self.candidates], dim=2)
        else:
            c = torch.stack([cand(h_star) for cand in self.candidates], dim=1)

        sim = F.linear(c, embedding_weights) / math.sqrt(self.d_model)

        if self.aggregation == "logsumexp":
            logits = torch.logsumexp(sim, dim=-2)
        elif self.aggregation == "max":
            logits = torch.max(sim, dim=-2).values
        elif self.aggregation == "mean":
            logits = torch.mean(sim, dim=-2)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        return logits
