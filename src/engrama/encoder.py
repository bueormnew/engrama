from torch import nn
import torch

from engrama.primitives import SynapseLayer, SharedCoreCellGroup, FactorizedSynapse
from engrama.config import EngramaConfig

class IsolatedEncoder(nn.Module):
    def __init__(
        self,
        config: EngramaConfig,
    ):
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
                )
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.w_pool = nn.Linear(config.num_cells * config.d_model, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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