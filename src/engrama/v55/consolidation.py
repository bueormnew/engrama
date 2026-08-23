"""ENGRAMA V5.5 — Consolidacion (Pilar 3).

Mezcla multiescala por offsets resonantes ``D_l = {0, 1, 2^{l-1}, 2^l}`` con
tres estabilizaciones respecto a V5:

1. **Mezcla NORMALIZADA por conteo** (default V5, se conserva):
   ``T_pos = sum_p w_p y_p / (sum_p w_p + eps)`` -> promedio acotado, no suma
   creciente. Elimina el crecimiento ~10x del residual que saturaba V4.
2. **Compuerta dual acotada con RMSNorm** (nuevo):
   ``q_tgt = RMSNorm(Q_tgt[t])``, ``k_src = RMSNorm(K_src[t-p])``;
   ``alpha = sigmoid( dot(q_tgt,k_src)/sqrt(d_g) * scale + qW + kW )`` con
   ``scale = C*tanh(b/C)`` y ``b`` init 0. La normalizacion acota el producto
   punto en ``[-1, 1]*d_g`` (vectores unitarios), no crece con ||T||^2.
3. **Residual zero-init**: ``T_l = T_pos + tanh(gamma) * FFN(RMSNorm(T_pos))``,
   ``gamma`` init 0. Los primeros pasos la celula es identidad exacta.

Sin control flow dinamico -> 100 % torch.compile. El camino paralelo y el
incremental son identicos termino a termino (invarianza causal).

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from engrama.v55.primitives import EngramaRMSNorm, GatedFFN

_EPS_NORM = 1e-4


def _sigmoid_fp32(x: torch.Tensor) -> torch.Tensor:
    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.sigmoid(x.float()).to(x.dtype)
    return torch.sigmoid(x)


def _rmsnorm_no_affine(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm sin parametros (para entradas de compuerta): acota el modulo."""
    out_dtype = x.dtype
    x32 = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x32 * rms).to(out_dtype)


def _clamp_bilinear(b: torch.Tensor, c: float) -> torch.Tensor:
    if c is None or c <= 0:
        return b
    return c * torch.tanh(b.float() / c).to(b.dtype)


class V55Mix(nn.Module):
    """Mezcla posicional dilatada NORMALIZADA + compuerta dual con RMSNorm."""

    def __init__(
        self,
        d_model: int,
        d_gate: int,
        offsets: Sequence[int],
        synapse_rank: int,
        *,
        bilinear_clamp: float = 4.0,
        count_normalize: bool = True,
        trace_tap: bool = True,
    ):
        super().__init__()
        if not offsets or any(p < 0 for p in offsets):
            raise ValueError("offsets debe ser una lista no vacia de enteros >= 0")
        self.d_model = d_model
        self.d_gate = d_gate
        self.offsets = sorted(dict.fromkeys(int(p) for p in offsets))
        self.num_offsets = len(self.offsets)
        self.synapse_rank = min(synapse_rank, d_model)
        self.bilinear_clamp = bilinear_clamp
        self.count_normalize = count_normalize
        self.trace_tap = trace_tap
        self.max_offset = max(self.offsets)

        # proyecciones de compuerta (target=fuente, en offset)
        self.p_g_src = nn.Linear(d_model, d_gate, bias=False)
        self.p_g_tgt = nn.Linear(d_model, d_gate, bias=False)
        self.gate_w_src = nn.ParameterDict(
            {str(p): nn.Parameter(torch.randn(d_gate, d_model) * 0.02) for p in self.offsets}
        )
        self.gate_w_tgt = nn.ParameterDict(
            {str(p): nn.Parameter(torch.randn(d_gate, d_model) * 0.02) for p in self.offsets}
        )
        self.gate_b = nn.ParameterDict(
            {str(p): nn.Parameter(torch.zeros(d_model)) for p in self.offsets}
        )
        # escala acotada del bilineal: scale = C*tanh(b/C), b init 0
        self.bil_scale_b = nn.ParameterDict(
            {str(p): nn.Parameter(torch.zeros(1)) for p in self.offsets}
        )
        self.rho = nn.ParameterDict(
            {str(p): nn.Parameter(torch.zeros(1)) for p in self.offsets}
        )
        # transporte factorizado
        self.U = nn.Parameter(torch.randn(d_model, self.synapse_rank) * 0.01)
        self.V = nn.Parameter(torch.randn(d_model, self.synapse_rank) * 0.01)
        self.s_scale = nn.ParameterDict(
            {str(p): nn.Parameter(torch.zeros(self.synapse_rank)) for p in self.offsets}
        )
        self.beta = nn.ParameterDict(
            {str(p): nn.Parameter(torch.ones(1)) for p in self.offsets}
        )
        # trace tap a T0 (mejor pieza de V4)
        if trace_tap:
            self.U_tr = nn.Parameter(torch.randn(d_model, self.synapse_rank) * 0.01)
            self.V_tr = nn.Parameter(torch.randn(d_model, self.synapse_rank) * 0.01)
            self.s_scale_tr = nn.ParameterDict(
                {str(p): nn.Parameter(torch.zeros(self.synapse_rank)) for p in self.offsets}
            )
            self.beta_tr = nn.ParameterDict(
                {str(p): nn.Parameter(torch.ones(1) * 0.5) for p in self.offsets}
            )
            self.gamma_tr = nn.ParameterDict(
                {str(p): nn.Parameter(torch.zeros(1)) for p in self.offsets}
            )
        else:
            self.U_tr = self.V_tr = None
            self.s_scale_tr = self.beta_tr = self.gamma_tr = None

    # ------------------------------------------------------------------
    @staticmethod
    def _causal_views(x: torch.Tensor, offsets: Sequence[int]) -> torch.Tensor:
        n, m = x.size(1), max(offsets)
        padded = F.pad(x, (0, 0, m, 0))
        return torch.stack([padded[:, m - p: m - p + n] for p in offsets], dim=2)

    def _scale(self, key: str) -> torch.Tensor:
        c = self.bilinear_clamp
        return _clamp_bilinear(self.bil_scale_b[key], c)

    # ------------------------------------------------------------------
    def forward_train(
        self, t_prev: torch.Tensor, t0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # No filtrar offsets por N: forward_step los incluye todos con zero-padding
        # (out-of-bounds -> contribuye 0 al numerador, w al denominador). Filtrar
        # aqui romperia la invarianza causal paralelo == incremental cuando
        # max_offset >= N (stacks profundos con secuencias cortas).
        offsets = list(self.offsets)
        keys = [str(p) for p in offsets]
        scale_dg = 1.0 / math.sqrt(self.d_gate)

        srcs = self._causal_views(t_prev, offsets)              # (B,N,P,d)
        # RMSNorm de las proyecciones de compuerta (acota el producto punto)
        k_src = self._causal_views(_rmsnorm_no_affine(self.p_g_src(t_prev)), offsets)
        q_tgt = _rmsnorm_no_affine(self.p_g_tgt(t_prev))
        gate_w_src = torch.stack([self.gate_w_src[k] for k in keys])
        gate_w_tgt = torch.stack([self.gate_w_tgt[k] for k in keys])
        gate_b = torch.stack([self.gate_b[k] for k in keys])
        g_src = torch.einsum("bnpq,pqd->bnpd", k_src, gate_w_src)
        g_tgt = torch.einsum("bnq,pqd->bnpd", q_tgt, gate_w_tgt)
        scales = torch.stack([self._scale(k) for k in keys]).view(1, 1, -1, 1)
        bil = (q_tgt.unsqueeze(2) * k_src).sum(dim=-1, keepdim=True) * scale_dg * scales
        rho = _sigmoid_fp32(torch.stack([self.rho[k] for k in keys])).view(1, 1, -1, 1)
        w = rho * _sigmoid_fp32(g_src + g_tgt + bil + gate_b)   # (B,N,P,d)

        z = self._causal_views(t_prev @ self.V, offsets)
        s = torch.stack([self.s_scale[k] for k in keys])
        beta = torch.stack([self.beta[k] for k in keys]).view(1, 1, -1, 1)
        y = (z * s.view(1, 1, len(offsets), -1)) @ self.U.T + beta * srcs
        if self.trace_tap and t0 is not None and self.V_tr is not None:
            tsrc = self._causal_views(t0, offsets)
            ztr = self._causal_views(t0 @ self.V_tr, offsets)
            str_ = torch.stack([self.s_scale_tr[k] for k in keys])
            btr = torch.stack([self.beta_tr[k] for k in keys]).view(1, 1, -1, 1)
            gtr = torch.stack([self.gamma_tr[k] for k in keys]).view(1, 1, -1, 1)
            y = y + gtr * (btr * tsrc + (ztr * str_.view(1, 1, len(offsets), -1)) @ self.U_tr.T)

        num = (w * y).sum(dim=2)
        if self.count_normalize:
            den = w.sum(dim=2) + _EPS_NORM
            return num / den
        return num

    # ------------------------------------------------------------------
    def forward_step(
        self,
        history: Sequence[torch.Tensor],
        trace_history: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        cur = history[-1]
        q_tgt = _rmsnorm_no_affine(self.p_g_tgt(cur))
        scale_dg = 1.0 / math.sqrt(self.d_gate)
        num = torch.zeros_like(cur)
        den = torch.zeros_like(cur)
        zero = torch.zeros_like(cur)
        for p in self.offsets:
            kp = str(p)
            if p + 1 > len(history):
                src = zero
                k_src = _rmsnorm_no_affine(self.p_g_src(zero))
            else:
                src = history[-(p + 1)]
                k_src = _rmsnorm_no_affine(self.p_g_src(src))
            scale = self._scale(kp)
            bil = (q_tgt * k_src).sum(dim=-1, keepdim=True) * scale_dg * scale
            g = _sigmoid_fp32(
                q_tgt @ self.gate_w_tgt[kp] + k_src @ self.gate_w_src[kp] + bil + self.gate_b[kp]
            )
            rho = _sigmoid_fp32(self.rho[kp])
            w = rho * g
            y = self.beta[kp] * src + (src @ self.V * self.s_scale[kp]) @ self.U.T
            if self.trace_tap and trace_history is not None and self.V_tr is not None:
                t0s = trace_history[-(p + 1)] if p + 1 <= len(trace_history) else zero
                y_tr = self.beta_tr[kp] * t0s + (t0s @ self.V_tr * self.s_scale_tr[kp]) @ self.U_tr.T
                y = y + self.gamma_tr[kp] * y_tr
            num = num + w * y
            den = den + w
        if self.count_normalize:
            return num / (den + _EPS_NORM)
        return num


class V55Layer(nn.Module):
    """Capa V5.5: mezcla normalizada + celula GatedFFN con residual zero-init."""

    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.cfg = cfg
        self.mix = V55Mix(
            cfg.d_model, cfg.d_gate, cfg.layer_offsets(layer_idx),
            cfg.synapse_rank, bilinear_clamp=cfg.dual_bilinear_clamp,
            count_normalize=cfg.count_normalize, trace_tap=cfg.trace_tap,
        )
        # GatedFFN integra RMSNorm + residual tanh(gamma) zero-init (2 matrices)
        self.cell = GatedFFN(cfg.d_model, cfg.d_ff, cfg.activation, cfg.dropout)

    def forward_train(self, t_prev: torch.Tensor,
                      t0: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_pos = self.mix.forward_train(t_prev, t0=t0)
        return self.cell(t_pos)

    def forward_step(self, history, trace_history=None) -> torch.Tensor:
        t_pos = self.mix.forward_step(history, trace_history=trace_history)
        return self.cell(t_pos)


class V55ConsolidationStack(nn.Module):
    """Pila de consolidacion V5.5. Produce ``T_L`` y expone ``T_shallow``
    (salida de la capa 0, contexto local de 2 tokens) para el codigo de
    sentido del Recall Tap asimetrico."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [V55Layer(cfg, i) for i in range(cfg.num_consolidation_layers)]
        )

    def forward_train(self, t0: torch.Tensor) -> torch.Tensor:
        """Devuelve ``(T_L, T_shallow)``; ``T_shallow`` = salida de la capa 0."""
        t_shallow = self.layers[0].forward_train(t0, t0=t0 if self.cfg.trace_tap else None)
        t = t_shallow
        for layer in self.layers[1:]:
            t = layer.forward_train(t, t0=t0 if self.cfg.trace_tap else None)
        return t, t_shallow

    def step_forward(self, cache):
        """Un token (T0 ya escrito en la traza): lee horizontes, calcula todas
        las capas y escribe sus salidas en los buffers por capa.

        Devuelve ``(T_L, T_shallow)``.
        """
        outputs = []
        t_shallow = None
        for l, layer in enumerate(self.layers):
            need = layer.mix.max_offset + 1
            hist = cache.t0_history(need) if l == 0 else cache.layer_history(l - 1, need)
            trace_hist = cache.t0_history(need) if self.cfg.trace_tap else None
            out = layer.forward_step(hist, trace_history=trace_hist)
            cache.append_layer(l, out)
            if l == 0:
                t_shallow = out
            outputs.append(out)
        return outputs[-1], t_shallow
