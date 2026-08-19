import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from engrama.primitives import FactorizedSynapse


class FactorizedEvokerCandidate(nn.Module):
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.d_model = d_model
        self.rank = rank

        self.W_shared = nn.Linear(d_model, d_model, bias=False)
        self.U_e = nn.Parameter(torch.randn(d_model, rank) * 0.01)
        self.V_e = nn.Parameter(torch.randn(rank, d_model) * 0.01)
        self.s_m = nn.Parameter(torch.randn(rank) * 0.01)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        lr_part = (h @ self.V_e.T) @ torch.diag(self.s_m) @ self.U_e.T
        return self.W_shared(h) + lr_part


class MultiCandidateEvoker(nn.Module):
    def __init__(
        self,
        config: EngramaConfig,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.num_candidates = config.num_candidates
        self.aggregation = config.candidate_aggregation
        self.evoker_mode = config.evoker_mode
        self.synapse_rank = config.synapse_rank

        if config.evoker_mode == "factorized":
            self.candidates = nn.ModuleList(
                [
                    FactorizedEvokerCandidate(config.d_model, config.synapse_rank)
                    for _ in range(config.num_candidates)
                ]
            )
        else:
            self.candidates = nn.ModuleList(
                [nn.Linear(config.d_model, config.d_model) for _ in range(config.num_candidates)]
            )

    def forward(
        self, h_star: torch.Tensor, embedding_weights: torch.Tensor
    ) -> torch.Tensor:
        is_3d = h_star.dim() == 3
        scale = 1.0 / math.sqrt(self.d_model)
        candidate_logits = []

        for cand in self.candidates:
            if self.evoker_mode == "factorized":
                c_m = cand(h_star)
            else:
                c_m = cand(h_star)
            logits = F.linear(c_m, embedding_weights) * scale
            candidate_logits.append(logits)

        if self.aggregation == "logsumexp":
            stacked = torch.stack(candidate_logits, dim=-1)
            max_logits = stacked.max(dim=-1).values
            sum_exp = torch.sum(torch.exp(stacked - max_logits.unsqueeze(-1)), dim=-1)
            return max_logits + torch.log(sum_exp)
        elif self.aggregation == "max":
            return torch.stack(candidate_logits, dim=-1).max(dim=-1).values
        elif self.aggregation == "mean":
            summed = torch.sum(torch.stack(candidate_logits, dim=-1), dim=-1)
            return summed / self.num_candidates
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")}}
