import torch
from torch import nn

class EngramaLayerNorm(nn.Module):
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


class SharedCoreCellGroup(nn.Module):
    def __init__(
        self,
        num_cells: int,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.num_cells = num_cells
        self.d_model = d_model

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

        self.s_scale = nn.Parameter(torch.ones(num_cells, d_model))
        self.n_mod = nn.Parameter(torch.ones(num_cells, d_model))
        self.q_bias = nn.Parameter(torch.zeros(num_cells, d_model))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        is_3d = u.dim() == 3
        if is_3d:
            u = u.unsqueeze(1)  # (B, 1, C, D)

        x_mod = self.n_mod.unsqueeze(0).unsqueeze(0) * u + self.q_bias.unsqueeze(0).unsqueeze(0)
        f_out = self.w2(self.dropout(self.act(self.w1(self.ln(x_mod)))))
        out = u + self.s_scale.unsqueeze(0).unsqueeze(0) * f_out

        if is_3d:
            out = out.squeeze(1)
        return out


class FactorizedSynapse(nn.Module):
    def __init__(self, d_model: int, rank: int, identity_transport: bool = True):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.identity_transport = identity_transport

        if identity_transport:
            self.beta = nn.Parameter(torch.ones(1))
        else:
            self.beta = None

        self.U = nn.Parameter(torch.randn(d_model, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, d_model) * 0.01)
        self.s = nn.Parameter(torch.randn(rank) * 0.01)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        transformed = (h @ self.V) @ torch.diag(self.s) @ self.U.T
        if self.identity_transport:
            return self.beta * h + transformed
        return transformed


class SynapseLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_gate: int,
        num_cells: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        synapse_mode: str = "factorized",
        synapse_rank: int = 32,
        identity_transport: bool = True,
        cell_mode: str = "shared_core",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_gate = d_gate
        self.num_cells = num_cells
        self.synapse_mode = synapse_mode
        self.synapse_rank = synapse_rank
        self.identity_transport = identity_transport
        self.cell_mode = cell_mode

        self.p_g = nn.Linear(d_model, d_gate, bias=False)
        self.gate_w = nn.Parameter(torch.randn(num_cells, num_cells, d_gate) * 0.01)
        self.gate_b = nn.Parameter(torch.zeros(num_cells, num_cells))

        if synapse_mode == "factorized":
            self.synapses = nn.ModuleDict(
                {
                    f"{a}_{b}": FactorizedSynapse(
                        d_model, synapse_rank, identity_transport
                    )
                    for a in range(num_cells)
                    for b in range(num_cells)
                }
            )
        else:
            self.w_transform = nn.Linear(d_model, d_model, bias=False)
            self.w_channel = nn.Parameter(
                torch.ones(num_cells, num_cells, d_model) * (1.0 / num_cells)
            )

        if cell_mode == "shared_core":
            self.cell_group = SharedCoreCellGroup(
                num_cells=num_cells,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
            )
            self.cells = None
        else:
            self.cell_group = None
            self.cells = nn.ModuleList(
                [Cell(d_model, d_ff, dropout, activation) for _ in range(num_cells)]
            )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        is_3d = H.dim() == 3
        if is_3d:
            H = H.unsqueeze(1)  # (B, 1, C, D)

        h_gate = self.p_g(H)  # (B, N, C, d_gate)
        alpha = torch.sigmoid(
            torch.einsum("bnad,acd->bnac", h_gate, self.gate_w) + self.gate_b
        )

        if self.synapse_mode == "factorized":
            trans = torch.zeros_like(H)
            for a in range(self.num_cells):
                for b in range(self.num_cells):
                    synapse = self.synapses[f"{a}_{b}"]
                    trans[:, :, a, :] += alpha[:, :, a, b, None] * synapse(H[:, :, b, :])
        else:
            z = self.w_transform(H)  # (B, N, C, d_model)
            trans = torch.einsum(
                "bnac,acd,bnad->bncd", alpha, self.w_channel, z
            )

        u = trans

        if self.cell_mode == "shared_core" and self.cell_group is not None:
            out = self.cell_group(u)
        else:
            out_list = [self.cells[c](u[:, :, c, :]) for c in range(self.num_cells)]
            out = torch.stack(out_list, dim=2)

        if is_3d:
            out = out.squeeze(1)
        return out