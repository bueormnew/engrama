"""ENGRAMA V5.5 — Encoder aislado V2 (Pilar 1).

Misma filosofia que V1-V5: ``T0[j] = f(x_j)`` depende SOLO del token ``j``,
cero mezcla temporal. Estabilidad mejorada:

* ``RMSNorm`` en todas las entradas (preserva signo para la ruta de identidad).
* Celula ``SwiGLU`` con down-proyeccion zero-init -> residual identidad en paso 0.
* Mezcla sinaptica ``C x C`` factorizada con inicializacion **zero-identity**
  (``beta=1``, ``s=0``).

Resultado: cero NaN en fp16 desde el paso 0; sigue siendo ``B x N x d``
totalmente paralelo por token.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""
from __future__ import annotations

import torch
from torch import nn

from engrama.v55.primitives import EngramaRMSNorm, SwiGLU, SynapseMixV55


class EncoderLayerV2(nn.Module):
    """Una capa del encoder aislado: RMSNorm -> mezcla CxC -> SwiGLU -> residual."""

    def __init__(self, d_model: int, d_gate: int, num_cells: int,
                 d_ff: int, synapse_rank: int, dropout: float = 0.0):
        super().__init__()
        self.norm = EngramaRMSNorm(d_model)
        self.mix = SynapseMixV55(d_model, d_gate, num_cells, synapse_rank)
        # gamma zero-init: celula identidad al inicio (estabilidad absoluta).
        self.gamma = nn.Parameter(torch.zeros(1))
        self.swiglu = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (..., C, d)
        m = self.mix(self.norm(h))
        return h + torch.tanh(self.gamma) * self.dropout(self.swiglu(m))


class IsolatedEncoderV2(nn.Module):
    """Codificador aislado por token (Pilar 1).

    Flujo (paralelo en toda la secuencia, sin mezcla entre posiciones)::

        h = reshape(init_proj(x), (C, d))
        for layer in layers: h = layer(h)
        T0 = w_pool(flatten(h))

    ``T0[i]`` depende exclusivamente de ``x_i``.
    """

    def __init__(self, d_model: int, d_gate: int, num_cells: int,
                 d_ff: int, num_encoder_layers: int, synapse_rank: int,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_cells = num_cells
        self.init_proj = nn.Linear(d_model, num_cells * d_model)
        self.layers = nn.ModuleList([
            EncoderLayerV2(d_model, d_gate, num_cells, d_ff, synapse_rank, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.pool_norm = EngramaRMSNorm(d_model)
        self.w_pool = nn.Linear(num_cells * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, N, d)`` o ``(B, d)`` -> huellas aisladas de igual shape."""
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)
        b, n, _ = x.shape
        h = self.init_proj(x).view(b, n, self.num_cells, self.d_model)
        for layer in self.layers:
            h = layer(h)
        flat = h.reshape(b, n, self.num_cells * self.d_model)
        out = self.pool_norm(self.w_pool(flat))
        if squeeze:
            out = out.squeeze(1)
        return out
