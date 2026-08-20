"""ENGRAMA V3 Primitive Blocks.

Implements the primitive blocks of the ENGRAMA V3 specification
(``ENGRAMA-V3-Teorica.md``, sections 5, 6, 7 and 32):

- :class:`EngramaLayerNorm` -- per-vector LayerNorm (no cross-position ops).
- :class:`Cell` -- V2/V3 cell: ``x + W2 * act(W1 * LN(x))``.
- :class:`SharedCoreCellGroup` -- V3 section 5 shared-core cells with
  per-cell diagonal modulation ``x + s_b (.) F_l(n_b (.) x + q_b)``.
- :class:`SynapseLayer` -- the C x C synaptic routing layer of the encoder.

V3 factorized synapse (spec sections 6-7), shared bases per layer::

    q_a       = P_g h_a                        (once per source cell)
    alpha_ab  = sigmoid(w_ab . q_a + b_ab)     (gate from the SOURCE)
    W_ab h    = beta_ab * h + U Diag(s_ab) V^T h
    o_ab      = alpha_ab (.) W_ab h_a
    u_b       = sum_a o_ab

with ``U, V in R^{d x r}`` shared by all synapses of the layer and
``s_ab in R^r``, ``beta_ab in R`` exclusive per synapse. ``r << d``.

Everything is fully vectorized: no per-synapse Python loops.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    act = name.lower()
    if act == "gelu":
        return nn.GELU()
    if act == "relu":
        return nn.ReLU()
    if act == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name!r}")


class EngramaLayerNorm(nn.Module):
    """LayerNorm over the last dimension only (never across positions)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta


class Cell(nn.Module):
    """ENGRAMA Cell: ``Cell(x) = x + W2 * act(W1 * LN(x))`` (paper section 4.1)."""

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
        self.act = _activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.w2(self.dropout(self.act(self.w1(self.ln(x)))))


class SharedCoreCellGroup(nn.Module):
    """V3 shared-core cell group (spec section 5.1).

    A single feed-forward core ``F_l`` is shared by the ``C`` cells of the
    layer; each cell keeps its identity through a diagonal modulation::

        Cell_{l,b}(x) = x + s_b (.) F_l(n_b (.) x + q_b)

    Accepts tensors shaped ``(B, N, C, d)`` or ``(B, C, d)``.
    """

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
        self.act = _activation(activation)
        self.dropout = nn.Dropout(dropout)

        self.s_scale = nn.Parameter(torch.ones(num_cells, d_model))
        self.n_mod = nn.Parameter(torch.ones(num_cells, d_model))
        self.q_bias = nn.Parameter(torch.zeros(num_cells, d_model))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x_mod = self.n_mod * u + self.q_bias
        f_out = self.w2(self.dropout(self.act(self.w1(self.ln(x_mod)))))
        return u + self.s_scale * f_out


class SynapseLayer(nn.Module):
    """C x C synaptic routing layer of the isolated encoder.

    Modes (V3 spec sections 6-7, ablations of section 44):

    - ``factorized`` (V3): shared bases ``U, V`` per layer, per-synapse scale
      ``s_ab`` and identity coefficient ``beta_ab``.
    - ``dense`` (V2): one dense matrix ``W_ab in R^{d x d}`` per synapse.

    In both modes the gate is computed from the **source** cell state, once
    per source (spec section 7)::

        alpha_{a->b} = sigmoid(w_ab . (P_g h_a) + b_ab)

    Input/output shape: ``(B, N, C, d)`` or ``(B, C, d)``.
    """

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
        stable_init: bool = True,
    ):
        super().__init__()
        if synapse_mode not in ("dense", "factorized"):
            raise ValueError(f"synapse_mode must be 'dense' or 'factorized'")
        if cell_mode not in ("independent", "shared_core"):
            raise ValueError(f"cell_mode must be 'independent' or 'shared_core'")

        self.d_model = d_model
        self.d_gate = d_gate
        self.num_cells = num_cells
        self.synapse_mode = synapse_mode
        self.synapse_rank = min(synapse_rank, d_model)
        self.identity_transport = identity_transport
        self.cell_mode = cell_mode

        # -- gating: one projection per source, one vector per synapse ------
        self.p_g = nn.Linear(d_model, d_gate, bias=False)
        self.gate_w = nn.Parameter(torch.randn(num_cells, num_cells, d_gate) * 0.02)
        self.gate_b = nn.Parameter(torch.zeros(num_cells, num_cells))

        if synapse_mode == "factorized":
            # Shared transformation bases for the layer (V3 spec 6.2).
            self.U = nn.Parameter(torch.randn(d_model, self.synapse_rank) * 0.01)
            self.V = nn.Parameter(torch.randn(d_model, self.synapse_rank) * 0.01)
            # Per-synapse specialization in the shared basis.
            if stable_init:
                self.s_scale = nn.Parameter(torch.zeros(num_cells, num_cells, self.synapse_rank))
            else:
                self.s_scale = nn.Parameter(
                    torch.randn(num_cells, num_cells, self.synapse_rank) * 0.02
                )
            if identity_transport:
                # Identity route (V3 spec 6.4 / 16): start as transport.
                self.beta = nn.Parameter(torch.ones(num_cells, num_cells))
            else:
                self.register_parameter("beta", None)
        else:
            # V2 dense synapses: one W_ab matrix per (a, b) pair.
            scale = d_model ** -0.5
            self.w_dense = nn.Parameter(
                torch.randn(num_cells, num_cells, d_model, d_model) * scale
            )

        if cell_mode == "shared_core":
            self.cell_group: Optional[SharedCoreCellGroup] = SharedCoreCellGroup(
                num_cells=num_cells,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
            )
            self.cells: Optional[nn.ModuleList] = None
        else:
            self.cell_group = None
            self.cells = nn.ModuleList(
                [Cell(d_model, d_ff, dropout, activation) for _ in range(num_cells)]
            )

    # ------------------------------------------------------------------
    def gates(self, H: torch.Tensor) -> torch.Tensor:
        """Gate tensor ``alpha[b, n, a, c]`` from the source cells (spec 7)."""
        q = self.p_g(H)  # (B, N, C, d_g) -- one projection per source cell
        return torch.sigmoid(
            torch.einsum("bnas,acs->bnac", q, self.gate_w) + self.gate_b
        )

    def _mix_factorized(self, H: torch.Tensor) -> torch.Tensor:
        """u_b = sum_a alpha_ab (.) (beta_ab h_a + U Diag(s_ab) V^T h_a)."""
        alpha = self.gates(H)  # (B, N, Ca, Cb)
        z = H @ self.V  # (B, N, Ca, r)

        # Identity route: u_id[c] = sum_a (alpha_ab * beta_ab) * h_a.
        if self.identity_transport and self.beta is not None:
            alpha_beta = alpha * self.beta  # (B, N, Ca, Cb)
        else:
            alpha_beta = alpha
        u = torch.matmul(alpha_beta.transpose(-1, -2), H)  # (B, N, Cb, d)

        # Low-rank route: u_lr[c] = sum_a alpha_ab * U (s_ab (.) z_a).
        for a in range(self.num_cells):
            spread = z[:, :, a, :].unsqueeze(2) * self.s_scale[a]  # (B, N, Cb, r)
            gated = alpha[:, :, a, :].unsqueeze(-1) * spread  # (B, N, Cb, r)
            u = u + gated @ self.U.T  # (B, N, Cb, d)
        return u

    def _mix_dense(self, H: torch.Tensor) -> torch.Tensor:
        """u_b = sum_a alpha_ab (.) (W_ab h_a) with dense per-synapse matrices."""
        alpha = self.gates(H)  # (B, N, Ca, Cb)
        u = torch.zeros_like(H)
        for b in range(self.num_cells):
            z_ab = torch.einsum("ade,bnae->bnad", self.w_dense[:, b], H)
            u[:, :, b, :] = torch.einsum("bna,bnad->bnd", alpha[:, :, :, b], z_ab)
        return u

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        squeeze = H.dim() == 3
        if squeeze:
            H = H.unsqueeze(1)  # (B, 1, C, d)

        if self.synapse_mode == "factorized":
            u = self._mix_factorized(H)
        else:
            u = self._mix_dense(H)

        if self.cell_group is not None:
            out = self.cell_group(u)
        else:
            out = torch.stack(
                [self.cells[c](u[:, :, c, :]) for c in range(self.num_cells)], dim=2
            )

        if squeeze:
            out = out.squeeze(1)
        return out

    def identity_fidelity(self) -> float:
        """Mean |beta| of the identity route (inspection helper, spec 31/50)."""
        if self.beta is None:
            return 0.0
        return float(self.beta.detach().abs().mean().item())
