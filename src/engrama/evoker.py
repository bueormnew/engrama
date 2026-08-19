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
        scale = 1.0 / math.sqrt(self.d_model)

        if self.aggregation == "logsumexp":
            logits_list = [
                F.linear(cand(h_star), embedding_weights) * scale
                for cand in self.candidates
            ]
            max_logits = logits_list[0]
            for l_c in logits_list[1:]:
                max_logits = torch.maximum(max_logits, l_c)
            sum_exp = torch.zeros_like(max_logits)
            for l_c in logits_list:
                sum_exp = sum_exp + torch.exp(l_c - max_logits)
            return max_logits + torch.log(sum_exp)
        elif self.aggregation == "max":
            max_logits = None
            for cand in self.candidates:
                l_c = F.linear(cand(h_star), embedding_weights) * scale
                max_logits = l_c if max_logits is None else torch.maximum(max_logits, l_c)
            return max_logits
        elif self.aggregation == "mean":
            sum_logits = torch.zeros_like(F.linear(self.candidates[0](h_star), embedding_weights))
            for cand in self.candidates:
                sum_logits = sum_logits + F.linear(cand(h_star), embedding_weights) * scale
            return sum_logits / self.num_candidates
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
