"""ENGRAMA V5.5 — bloques primitivos.

* :class:`SwiGLU` — GLU con activacion SiLU, down-proyeccion zero-init.
* :class:`SynapseMixV55` — enrutado sinaptico ``C x C`` factorizado con
  inicializacion **zero-identity** (``beta=1``, ``s=0``): al inicio del
  entrenamiento es identidad estable.
* :class:`EngramaRMSNorm` se reutiliza de :mod:`engrama.primitives`.
* :func:`softcap` — cota tangencial de logits (Pilar 6), evita la explosion
  que saturaba el entrenamiento de V4 a LR alto.

Todo es por-token y totalmente paralelo (cero mezcla entre posiciones).

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from engrama.primitives import EngramaRMSNorm  # noqa: F401  (reexport)


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """Cota tangencial simetrica: ``cap * tanh(x / cap)``.

    ``cap <= 0`` desactiva (devuelve ``x``). Estable en fp16 (``tanh`` esta
    acotada en ``(-1, 1)``) y derivable: ``d/dx = 1 - tanh^2``. Usada en el
    evocador (Pilar 6) para impedir la explosion de logits.
    """
    if cap is None or cap <= 0:
        return x
    return cap * torch.tanh(x.float() / cap).to(x.dtype)


class SwiGLU(nn.Module):
    """Unidad GLU con activacion SiLU: ``down(silu(gate(x)) * up(x))``.

    La down-proyeccion se inicializa a CERO: al inicio ``SwiGLU(x) == 0`` y el
    residual al que se suma es identidad exacta. Es la pieza que hace que el
    encoder V2 y la celula de consolidacion sean identidad en el paso 0 (Pilar 1
    y Pilar 3), eliminando los NaN en fp16 desde el primer paso.
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=bias)
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)
        nn.init.zeros_(self.down.weight)
        if self.down.bias is not None:
            nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GatedFFN(nn.Module):
    """FFN de 2 matrices con residual ``tanh(gamma)`` zero-init (Pilar 3).

    ``out = x + tanh(gamma) * w2(act(w1(norm(x))))`` con ``gamma`` init 0 ->
    la celula es identidad exacta en el paso 0 (estabilidad absoluta). Usa 2
    matrices (como la celula V4) en vez de las 3 de SwiGLU para mantener el
    conteo de parametros cercano a V4; SwiGLU se reserva para el encoder (Pilar 1).
    """

    def __init__(self, d_model: int, d_ff: int, activation: str = "silu",
                 dropout: float = 0.0):
        super().__init__()
        self.norm = EngramaRMSNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        nn.init.zeros_(self.w2.weight)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.act = _activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.gamma) * self.dropout(
            self.w2(self.act(self.w1(self.norm(x)))))


def _activation(name: str) -> nn.Module:
    act = name.lower()
    if act == "gelu":
        return nn.GELU()
    if act == "relu":
        return nn.ReLU()
    if act == "silu":
        return nn.SiLU()
    raise ValueError(f"activation no soportada: {name!r}")


class SynapseMixV55(nn.Module):
    """Enrutado sinaptico ``C x C`` factorizado, **zero-identity**.

    Para ``C`` celulas por token:

    ``u_b = sum_a alpha_ab * (beta_ab * h_a + U Diag(s_ab) V^T h_a)``

    con ``alpha_ab = sigmoid(<p_g(h_a), w_ab> + b_ab)``.

    Inicializacion zero-identity: ``beta = 1``, ``s = 0`` y bases ``U, V``
    pequenas. Al inicio el termino low-rank es nulo y ``u_b = sum_a alpha_ab h_a``
    (mezcla acotada por compuerta), nunca inestable. Es totalmente paralelo y
    sin mezcla entre posiciones (pilar 1 de aislamiento).

    Entrada/salida: ``(..., C, d)``.
    """

    def __init__(self, d_model: int, d_gate: int, num_cells: int, synapse_rank: int):
        super().__init__()
        self.num_cells = num_cells
        self.d_model = d_model
        self.d_gate = d_gate
        self.synapse_rank = min(synapse_rank, d_model)
        # compuerta (una proyeccion por fuente, un vector por sinapsis)
        self.p_g = nn.Linear(d_model, d_gate, bias=False)
        self.gate_w = nn.Parameter(torch.randn(num_cells, num_cells, d_gate) * 0.02)
        self.gate_b = nn.Parameter(torch.zeros(num_cells, num_cells))
        # transporte factorizado
        self.U = nn.Parameter(torch.randn(d_model, self.synapse_rank) * 0.01)
        self.V = nn.Parameter(torch.randn(d_model, self.synapse_rank) * 0.01)
        self.s_scale = nn.Parameter(torch.zeros(num_cells, num_cells, self.synapse_rank))
        self.beta = nn.Parameter(torch.ones(num_cells, num_cells))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (..., C, d)
        leading = h.shape[:-2]
        c, d = h.shape[-2], h.shape[-1]
        h_flat = h.reshape(-1, c, d)                      # (P, C, d)
        q = self.p_g(h_flat)                              # (P, C, dg)
        alpha = torch.sigmoid(
            torch.einsum("pac,abc->pab", q, self.gate_w) + self.gate_b
        )                                                 # (P, Ca, Cb)
        z = h_flat @ self.V                               # (P, C, r)
        # ruta identidad: u_id[b] = sum_a (alpha_ab beta_ab) h_a
        ab = alpha * self.beta                            # (P, Ca, Cb)
        u_id = torch.einsum("pab,pad->pbd", ab, h_flat)   # (P, Cb, d)
        # ruta low-rank: sum_a alpha_ab s_ab (V^T h_a), luego x U^T
        gated = torch.einsum("pab,par,abr->pbr", alpha, z, self.s_scale)  # (P, Cb, r)
        u_lr = gated @ self.U.T                            # (P, Cb, d)
        out = u_id + u_lr
        return out.reshape(*leading, c, d)
