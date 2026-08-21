"""ENGRAMA V5 — Consolidacion: mezcla multiescala NORMALIZADA por conteo.

Ecuacion V5 (por capa l, offset p en D_l):

    w_p    = rho_p * alpha_p                        (por canal; alpha dual acotada)
    y_p    = beta_p * T_{l-1}[i-p] + U Diag(s_p) V^T T_{l-1}[i-p]
             + gamma_p * (beta_tr * T_0[i-p] + U_tr Diag(s_tr) V_tr^T T_0[i-p])
    T_pos  = ( SUM_p w_p (.) y_p ) / ( SUM_p w_p + eps )      <- promedio, no suma
    T_l    = T_pos + FFN(RMSNorm(T_pos))                       (celula)

Si la capa recibe lecturas del Recall Tap:

    T_pos  <-  T_pos + g_rt * W_r(r_i)              (g_rt = 2*sigmoid(escalar))

La normalizacion por conteo (nuevo en V5):
* acota la magnitud del estado (elimina el crecimiento ~10x del residual que
  saturaba las compuertas duales de V4 y rompia fp16);
* convierte el estado en PROMEDIO de fuentes abiertas: la contribucion relativa
  de una fuente es su peso de compuerta (no 1/N);
* exactamente reproducible en incremental (misma suma, mismo denominador).

El paralelo y el incremental son identicos termino a termino (invarianza causal).

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from engrama.primitives import Cell

_EPS_NORM = 1e-4


def _sigmoid_fp32(x: torch.Tensor) -> torch.Tensor:
    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.sigmoid(x.float()).to(x.dtype)
    return torch.sigmoid(x)


def _clamp_bilinear(b: torch.Tensor, c: float) -> torch.Tensor:
    if c is None or c <= 0:
        return b
    return c * torch.tanh(b.float() / c).to(b.dtype)


class V5Mix(nn.Module):
    """Mezcla posicional dilatada normalizada con compuerta dual acotada."""

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

        # compuerta
        self.p_g_src = nn.Linear(d_model, d_gate, bias=False)
        self.gate_w_src = nn.ParameterDict(
            {str(p): nn.Parameter(torch.randn(d_gate, d_model) * 0.02) for p in self.offsets}
        )
        self.gate_b = nn.ParameterDict(
            {str(p): nn.Parameter(torch.zeros(d_model)) for p in self.offsets}
        )
        self.p_g_tgt = nn.Linear(d_model, d_gate, bias=False)
        self.gate_w_tgt = nn.ParameterDict(
            {str(p): nn.Parameter(torch.randn(d_gate, d_model) * 0.02) for p in self.offsets}
        )
        self.rho = nn.ParameterDict(
            {str(p): nn.Parameter(torch.zeros(1)) for p in self.offsets}
        )
        # transporte factorizado (V3+)
        self.U = nn.Parameter(torch.randn(d_model, self.synapse_rank) * 0.01)
        self.V = nn.Parameter(torch.randn(d_model, self.synapse_rank) * 0.01)
        self.s_scale = nn.ParameterDict(
            {str(p): nn.Parameter(torch.zeros(self.synapse_rank)) for p in self.offsets}
        )
        self.beta = nn.ParameterDict(
            {str(p): nn.Parameter(torch.ones(1)) for p in self.offsets}
        )
        # trace tap (V4)
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
                {str(p): nn.Parameter(torch.ones(1) * 0.1) for p in self.offsets}
            )
        else:
            self.U_tr = self.V_tr = None
            self.s_scale_tr = self.beta_tr = self.gamma_tr = None

    # ------------------------------------------------------------------
    @staticmethod
    def _causal_views(x: torch.Tensor, offsets: Sequence[int]) -> torch.Tensor:
        n, m = x.size(1), max(offsets)
        padded = F.pad(x, (0, 0, m, 0))
        return torch.stack([padded[:, m - p : m - p + n] for p in offsets], dim=2)

    # ------------------------------------------------------------------
    def forward_train(
        self, t_prev: torch.Tensor, t0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        offsets = [p for p in self.offsets if p < t_prev.size(1)]
        keys = [str(p) for p in offsets]
        scale_dg = 1.0 / math.sqrt(self.d_gate)

        srcs = self._causal_views(t_prev, offsets)              # (B,N,P,d)
        k_src = self._causal_views(self.p_g_src(t_prev), offsets)
        q_tgt = self.p_g_tgt(t_prev)
        gate_w_src = torch.stack([self.gate_w_src[k] for k in keys])
        gate_w_tgt = torch.stack([self.gate_w_tgt[k] for k in keys])
        gate_b = torch.stack([self.gate_b[k] for k in keys])
        g_src = torch.einsum("bnpq,pqd->bnpd", k_src, gate_w_src)
        g_tgt = torch.einsum("bnq,pqd->bnpd", q_tgt, gate_w_tgt)
        bil = (q_tgt.unsqueeze(2) * k_src).sum(dim=-1, keepdim=True) * scale_dg
        bil = _clamp_bilinear(bil, self.bilinear_clamp)
        rho = _sigmoid_fp32(torch.stack([self.rho[k] for k in keys])).view(1, 1, -1, 1)
        w = rho * _sigmoid_fp32(g_src + g_tgt + bil + gate_b)   # (B,N,P,d) pesos

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
        q_tgt = self.p_g_tgt(cur)
        scale_dg = 1.0 / math.sqrt(self.d_gate)
        num = torch.zeros_like(cur)
        den = torch.zeros_like(cur)
        zero = torch.zeros_like(cur)
        for p in self.offsets:
            kp = str(p)
            if p + 1 > len(history):
                # El camino paralelo rellena con CEROS las fuentes ausentes:
                # replicarlo aqui hace la invarianza causal EXACTA en toda posicion.
                src = zero
                k_src = self.p_g_src(zero)
            else:
                src = history[-(p + 1)]
                k_src = self.p_g_src(src)
            bil = _clamp_bilinear(
                (q_tgt * k_src).sum(dim=-1, keepdim=True) * scale_dg,
                self.bilinear_clamp,
            )
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


class V5Layer(nn.Module):
    """Una capa V5: mezcla normalizada + (lectura RT si toca) + celula."""

    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        from engrama.v5.config import V5Config  # local para evitar ciclo

        assert isinstance(cfg, V5Config)
        self.layer_idx = layer_idx
        self.cfg = cfg
        self.mix = V5Mix(
            cfg.d_model,
            cfg.d_gate,
            cfg.layer_offsets(layer_idx),
            cfg.synapse_rank,
            bilinear_clamp=cfg.dual_bilinear_clamp,
            count_normalize=cfg.count_normalize,
            trace_tap=cfg.trace_tap,
        )
        self.cell = Cell(cfg.d_model, cfg.d_ff, cfg.dropout, cfg.activation,
                         norm_type="rmsnorm")
        self.with_recall = cfg.recall_enabled and layer_idx in set(cfg.rt_layers)
        if self.with_recall:
            self.rt_gate_param = nn.Parameter(
                torch.tensor(float(_inv2sigmoid(cfg.rt_gate_init)))
            )
            self.rt_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            nn.init.normal_(self.rt_proj.weight, std=0.02)

    def inject(self, t_pos: torch.Tensor, reads: Optional[torch.Tensor]) -> torch.Tensor:
        if self.with_recall and reads is not None:
            g = 2.0 * torch.sigmoid(self.rt_gate_param)
            return t_pos + g * self.rt_proj(reads)
        return t_pos

    def forward_train(
        self, t_prev: torch.Tensor, t0: Optional[torch.Tensor], reads=None
    ) -> torch.Tensor:
        t_pos = self.mix.forward_train(t_prev, t0=t0)
        return self.cell(self.inject(t_pos, reads))

    def forward_step(self, history, trace_history=None, read=None) -> torch.Tensor:
        t_pos = self.mix.forward_step(history, trace_history=trace_history)
        return self.cell(self.inject(t_pos, read))


def _inv2sigmoid(y: float) -> float:
    y = min(max(y, 1e-3), 2.0 - 1e-3)
    p = y / 2.0
    return float(math.log(p / (1.0 - p)))


class V5ConsolidationStack(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [V5Layer(cfg, i) for i in range(cfg.num_consolidation_layers)]
        )
        self.rt_layers = {l for l in cfg.rt_layers} if cfg.recall_enabled else set()

    def forward_train(self, t0: torch.Tensor, reads: Optional[torch.Tensor]) -> torch.Tensor:
        t = t0
        for layer in self.layers:
            read = reads if layer.layer_idx in self.rt_layers else None
            t = layer.forward_train(t, t0=t0 if self.cfg.trace_tap else None, reads=read)
        return t

    def step_forward(self, cache, read_now=None):
        """Un token (T0 ya escrito en la traza por el modelo): lee horizontes,
        calcula todas las capas y escribe sus salidas en los buffers por capa."""
        outputs = []
        for l, layer in enumerate(self.layers):
            need = layer.mix.max_offset + 1
            hist = cache.t0_history(need) if l == 0 else cache.layer_history(l - 1, need)
            trace_hist = cache.t0_history(need) if self.cfg.trace_tap else None
            read = read_now if layer.layer_idx in self.rt_layers else None
            out = layer.forward_step(hist, trace_history=trace_hist, read=read)
            cache.append_layer(l, out)
            outputs.append(out)
        return outputs[-1], outputs
