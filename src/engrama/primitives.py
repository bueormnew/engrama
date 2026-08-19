"""
ENGRAMA Primitives Module: LayerNorm, Cell, and SynapseLayer
Author: BUEORM
License: AGPL-3.0
"""

from typing import Optional

import torch
from torch import nn


class EngramaLayerNorm(nn.Module):
    """Custom Layer Normalization with learnable scale (gamma) and shift (beta)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        return (x - mean) / std * self.gamma + self.beta


class Cell(nn.Module):
    """ENGRAMA Cellular Non-Linear Transformation Unit with Residual Connection.

    Equation:
        c'(x) = x + W_2 * Dropout(Activation(W_1 * LayerNorm(x)))
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.ln = EngramaLayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        
        act = activation.lower()
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.w2(self.dropout(self.act(self.w1(self.ln(x)))))


class SynapseLayer(nn.Module):
    """ENGRAMA Synapse Layer: Multi-Cell Channel Routing without Dynamic Attention.

    Routes representation vectors across `C` distinct internal cells using static
    channel projection and latent gating vectors without sequence-length affinity matrices.

    Args:
        d_model (int): Model dimension size.
        d_gate (int): Gate projection latent dimension (d_gate < d_model).
        num_cells (int): Number of parallel cellular pathways (C).
        d_ff (Optional[int]): Feed-forward hidden dimension size for cells.
        dropout (float): Dropout probability.
        activation (str): Activation function for internal cells.
    """

    def __init__(
        self,
        d_model: int,
        d_gate: int,
        num_cells: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.d_model = d_model
        self.d_gate = d_gate
        self.num_cells = num_cells
        self.d_ff = d_ff

        self.p_g = nn.Linear(d_model, d_gate, bias=False)
        self.w_transform = nn.Linear(d_model, d_model, bias=False)
        self.gate_w = nn.Parameter(torch.randn(num_cells, num_cells, d_gate) * 0.01)
        self.gate_b = nn.Parameter(torch.zeros(num_cells, num_cells))
        self.w_channel = nn.Parameter(
            torch.ones(num_cells, num_cells, d_model) * (1.0 / num_cells)
        )
        self.cells = nn.ModuleList(
            [Cell(d_model, d_ff, dropout, activation) for _ in range(num_cells)]
        )
        self.cell_layers = self.cells

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Forward pass for Synapse routing layer.

        Args:
            H (Tensor): Input tensor of shape (B, N, C, D) or (B, C, D).

        Returns:
            Tensor: Output tensor of matching shape (B, N, C, D) or (B, C, D).
        """
        is_3d = H.dim() == 3
        if is_3d:
            H = H.unsqueeze(1)  # (B, 1, C, D)

        h_gate = self.p_g(H)  # (B, N, C, d_gate)
        alpha = torch.sigmoid(
            torch.einsum("bnad,acd->bnac", h_gate, self.gate_w) + self.gate_b
        )
        z = self.w_transform(H)  # (B, N, C, d_model)
        u = torch.einsum("bnac,acd,bnad->bncd", alpha, self.w_channel, z)

        out_list = [self.cells[c](u[:, :, c, :]) for c in range(self.num_cells)]
        out = torch.stack(out_list, dim=2)

        if is_3d:
            out = out.squeeze(1)
        return out
