"""
ENGRAMA Isolated Encoder Module (Phase 1)
Author: BUEORM
License: AGPL-3.0
"""

import torch
from torch import nn

from engrama.primitives import SynapseLayer


class IsolatedEncoder(nn.Module):
    """Isolated Encoder (Phase 1): Processes input tokens without temporal context leakage.

    Each token position t in x is encoded independently into an initial state T_0[t] using
    isolated Synapse layers.

    Args:
        d_model (int): Hidden dimension size.
        d_gate (int): Gate projection size.
        num_cells (int): Number of cellular channels.
        num_encoder_layers (int): Number of Synapse layers in the encoder.
        d_ff (int): Feedforward expansion dimension.
        dropout (float): Dropout rate.
        activation (str): Activation function name.
    """

    def __init__(
        self,
        d_model: int,
        d_gate: int,
        num_cells: int,
        num_encoder_layers: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_cells = num_cells
        self.init_proj = nn.Linear(d_model, num_cells * d_model)
        self.layers = nn.ModuleList(
            [
                SynapseLayer(
                    d_model=d_model,
                    d_gate=d_gate,
                    num_cells=num_cells,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.w_pool = nn.Linear(num_cells * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for token-isolated encoding.

        Args:
            x (Tensor): Input embeddings of shape (B, N, D) or (B, D).

        Returns:
            Tensor: Isolated representations T_0 of shape (B, N, D) or (B, D).
        """
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        b, n, _ = x.shape
        h = self.init_proj(x).view(b, n, self.num_cells, self.d_model)
        for layer in self.layers:
            h = layer(h)
        h = h.view(b, n, self.num_cells * self.d_model)
        out = self.w_pool(h)

        if is_2d:
            out = out.squeeze(1)
        return out
