"""
ENGRAMA Consolidation Module (Phase 3)
Author: BUEORM
License: AGPL-3.0
"""

from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn

from engrama.primitives import Cell


class PositionalDilatedMix(nn.Module):
    """Positional Dilated Gated Mixing layer.

    Combines representations across exponential positional offsets P_l(i) = [0, 1, 2, 4, 8, ...]
    using relative dynamic gating projections without dynamic QK^T matrix operations.

    Args:
        d_model (int): Hidden dimension size.
        d_gate (int): Gate projection size (d_gate < d_model).
        offsets (List[int]): Powers of 2 relative offsets.
    """

    def __init__(self, d_model: int, d_gate: int, offsets: List[int]):
        super().__init__()
        self.d_model = d_model
        self.d_gate = d_gate
        self.offsets = list(offsets)

        self.p_g = nn.Linear(d_model, d_gate, bias=False)
        self.w_offsets = nn.ModuleDict(
            {str(p): nn.Linear(d_model, d_model, bias=False) for p in self.offsets}
        )
        self.gate_w = nn.ParameterDict(
            {
                str(p): nn.Parameter(torch.randn(d_gate, d_model) * 0.01)
                for p in self.offsets
            }
        )
        self.gate_b = nn.ParameterDict(
            {str(p): nn.Parameter(torch.zeros(d_model)) for p in self.offsets}
        )

    def forward_train(self, T_prev: torch.Tensor) -> torch.Tensor:
        """Parallel sequence forward pass for training."""
        b, n, d = T_prev.shape
        t_pos = torch.zeros_like(T_prev)
        for p in self.offsets:
            str_p = str(p)
            if p == 0:
                t_shifted = T_prev
            elif p < n:
                t_shifted = torch.cat(
                    [
                        torch.zeros(b, p, d, device=T_prev.device, dtype=T_prev.dtype),
                        T_prev[:, :-p, :],
                    ],
                    dim=1,
                )
            else:
                continue
            h_g = self.p_g(t_shifted)
            gate_logits = torch.matmul(h_g, self.gate_w[str_p]) + self.gate_b[str_p]
            g = torch.sigmoid(gate_logits)
            transformed = self.w_offsets[str_p](t_shifted)
            t_pos = t_pos + g * transformed
        return t_pos

    def forward_step(
        self,
        T_prev_history: Union[torch.Tensor, List[torch.Tensor]],
        current_pos: int,
    ) -> torch.Tensor:
        """Single-step incremental lookup pass for cached inference."""
        if isinstance(T_prev_history, torch.Tensor):
            b, n_hist, d = T_prev_history.shape
            device = T_prev_history.device
            dtype = T_prev_history.dtype
        else:
            first = T_prev_history[0]
            b = first.shape[0]
            d = first.shape[-1]
            device = first.device
            dtype = first.dtype

        t_pos = torch.zeros(b, d, device=device, dtype=dtype)
        for p in self.offsets:
            idx = current_pos - p
            if idx < 0:
                continue

            if isinstance(T_prev_history, torch.Tensor):
                if idx >= T_prev_history.shape[1]:
                    continue
                t_p = T_prev_history[:, idx, :]
            else:
                if idx >= len(T_prev_history):
                    continue
                t_p = T_prev_history[idx]
                if t_p.dim() == 3:
                    t_p = t_p.squeeze(1)

            str_p = str(p)
            h_g = self.p_g(t_p)
            gate_logits = torch.matmul(h_g, self.gate_w[str_p]) + self.gate_b[str_p]
            g = torch.sigmoid(gate_logits)
            transformed = self.w_offsets[str_p](t_p)
            t_pos = t_pos + g * transformed
        return t_pos


class ConsolidationLayer(nn.Module):
    """Single Consolidation Stack Layer: PositionalDilatedMix followed by Cell processing."""

    def __init__(
        self,
        d_model: int,
        d_gate: int,
        offsets: List[int],
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.mix = PositionalDilatedMix(d_model, d_gate, offsets)
        self.cell = Cell(d_model, d_ff, dropout, activation)

    def forward_train(self, T_prev: torch.Tensor) -> torch.Tensor:
        t_pos = self.mix.forward_train(T_prev)
        return self.cell(t_pos)

    def forward_step(
        self,
        T_prev_history: Union[torch.Tensor, List[torch.Tensor]],
        current_pos: int,
    ) -> torch.Tensor:
        t_pos = self.mix.forward_step(T_prev_history, current_pos)
        return self.cell(t_pos)

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        current_pos: Optional[int] = None,
    ) -> torch.Tensor:
        if current_pos is not None:
            return self.forward_step(x, current_pos)
        return self.forward_train(x)


class ConsolidationStack(nn.Module):
    """Deep Consolidation Stack consisting of multiple stacked ConsolidationLayers."""

    def __init__(
        self,
        d_model: int,
        d_gate: int,
        offsets: List[int],
        num_consolidation_layers: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConsolidationLayer(
                    d_model=d_model,
                    d_gate=d_gate,
                    offsets=offsets,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_consolidation_layers)
            ]
        )

    def forward_train(self, T0: torch.Tensor) -> torch.Tensor:
        t = T0
        for layer in self.layers:
            t = layer.forward_train(t)
        return t

    def forward(self, T0: torch.Tensor) -> torch.Tensor:
        return self.forward_train(T0)

    def step_forward(
        self,
        cache: Any,
        T0_current: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Incremental cached step pass through the entire consolidation stack."""
        if T0_current is not None:
            history_t0 = cache.T0 + [T0_current]
            current_pos = len(cache.T0)
        else:
            history_t0 = cache.T0
            current_pos = len(cache.T0) - 1

        layer_outputs: List[torch.Tensor] = []
        for l, layer in enumerate(self.layers):
            if l == 0:
                prev_hist = history_t0
            else:
                prev_hist = cache.Tl[l - 1] + [layer_outputs[l - 1]]
            t_out = layer.forward_step(prev_hist, current_pos)
            layer_outputs.append(t_out)

        t_l = layer_outputs[-1]
        if return_all_layers:
            return t_l, layer_outputs
        return t_l
