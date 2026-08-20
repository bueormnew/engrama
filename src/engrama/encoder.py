"""ENGRAMA Isolated Encoder (Phase 1).

Each token is encoded independently of its neighbors:

    H_i^(0) = init_proj(e_i) reshaped to (C, d)
    H^(k+1) = SynapseLayer(H^(k))        (C x C routing, token-wise)
    T_0[i]  = w_pool(Flatten(H_i^(L_enc)))

Because no operation mixes positions, ``T_0[i]`` depends exclusively on
``x_i`` (isolated encoding theorem, V1/V2 paper section 5.1; untouched by
V3/V4) and the whole sequence encodes in parallel.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""

from __future__ import annotations

import torch
from torch import nn

from engrama.config import EngramaConfig
from engrama.primitives import SynapseLayer


class IsolatedEncoder(nn.Module):
    """Stack of C x C SynapseLayers applied independently to every token."""

    def __init__(self, config: EngramaConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_cells = config.num_cells
        self.init_proj = nn.Linear(config.d_model, config.num_cells * config.d_model)
        self.layers = nn.ModuleList(
            [
                SynapseLayer(
                    d_model=config.d_model,
                    d_gate=config.d_gate,
                    num_cells=config.num_cells,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                    synapse_mode=config.synapse_mode,
                    synapse_rank=config.synapse_rank,
                    identity_transport=config.identity_transport,
                    cell_mode=config.cell_mode,
                    stable_init=config.stable_init,
                    norm_type=config.norm_type or "layernorm",
                )
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.w_pool = nn.Linear(config.num_cells * config.d_model, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (B, N, d) embeddings to isolated footprints (B, N, d)."""
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)

        b, n, _ = x.shape
        h = self.init_proj(x).view(b, n, self.num_cells, self.d_model)
        for layer in self.layers:
            h = layer(h)
        out = self.w_pool(h.reshape(b, n, self.num_cells * self.d_model))

        if squeeze:
            out = out.squeeze(1)
        return out
