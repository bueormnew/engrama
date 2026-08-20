"""ENGRAMA Consolidation Stack (Phase 3).

Implements the hierarchical causal consolidation of ENGRAMA (V3 + V4):

::

    T_pos_l[i] = sum_{p in D_l, p <= i} rho_{l,p} * G_{l,p}(T_{l-1}[i-p], T_0[i-p])

    G_{l,p}(x, t0) = alpha_{l,p}(.) (y_context + gamma_{l,p} * y_trace)

    y_context      = beta_{l,p} x + U_l Diag(s_{l,p}) V_l^T x
    y_trace        = beta_tr_{l,p} t0 + U_tr Diag(s_tr_{l,p}) V_tr^T t0

    T_l[i]         = Cell_l(T_pos_l[i])

Gating Modes:
- ``dual`` (V4 default): Bilinear target-source point-to-point gating:
  ``alpha = sigmoid(q_tgt . k_src / sqrt(d_g) + q_tgt W_tgt + k_src W_src + b)``.
  No attention matrix, no softmax over sequence, strictly O(N) linear time.
- ``source`` (V3): Source-only gate ``alpha = sigmoid(k_src W_src + b)``.

Both parallel training (``forward_train``) and incremental inference with
minimum-horizon windows (``forward_step``) are fully vectorized and exactly
equivalent by causal invariance.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from engrama.config import EngramaConfig
from engrama.primitives import Cell


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid in fp32 under AMP so dual-gating pre-activations cannot saturate to 0/1 inf."""
    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.sigmoid(x.float()).to(x.dtype)
    return torch.sigmoid(x)


class PositionalDilatedMix(nn.Module):
    """Positional dilated mix with factorized fidelity transport and dual gating."""

    def __init__(
        self,
        d_model: int,
        d_gate: int,
        offsets: List[int],
        synapse_mode: str = "factorized",
        synapse_rank: int = 32,
        identity_transport: bool = True,
        hierarchical_gate: bool = True,
        stable_init: bool = True,
        gating_mode: str = "dual",
        trace_tap: bool = True,
    ):
        super().__init__()
        if synapse_mode not in ("dense", "factorized"):
            raise ValueError("synapse_mode must be 'dense' or 'factorized'")
        if not offsets:
            raise ValueError("offsets must be a non-empty list")
        if any(o < 0 for o in offsets):
            raise ValueError("offsets must be non-negative and causal (>= 0)")
        if gating_mode not in ("source", "dual"):
            raise ValueError(f"gating_mode must be 'source' or 'dual', got {gating_mode!r}")

        self.d_model = d_model
        self.d_gate = d_gate
        self.offsets = sorted(dict.fromkeys(int(p) for p in offsets))
        self.num_offsets = len(self.offsets)
        self.synapse_mode = synapse_mode
        self.synapse_rank = min(synapse_rank, d_model)
        self.identity_transport = identity_transport
        self.hierarchical_gate = hierarchical_gate
        self.gating_mode = gating_mode
        self.trace_tap = trace_tap
        self.max_offset = max(self.offsets)

        # Source gating projection (shared per source state)
        self.p_g_src = nn.Linear(d_model, d_gate, bias=False)
        self.gate_w_src = nn.ParameterDict(
            {
                str(p): nn.Parameter(torch.randn(d_gate, d_model) * 0.02)
                for p in self.offsets
            }
        )
        self.gate_b = nn.ParameterDict(
            {str(p): nn.Parameter(torch.zeros(d_model)) for p in self.offsets}
        )

        # Target gating projection for Dual Target-Source Gating (V4)
        if gating_mode == "dual":
            self.p_g_tgt = nn.Linear(d_model, d_gate, bias=False)
            self.gate_w_tgt = nn.ParameterDict(
                {
                    str(p): nn.Parameter(torch.randn(d_gate, d_model) * 0.02)
                    for p in self.offsets
                }
            )
        else:
            self.p_g_tgt = None
            self.gate_w_tgt = None

        if synapse_mode == "factorized":
            self.U = nn.Parameter(torch.randn(d_model, self.synapse_rank) * 0.01)
            self.V = nn.Parameter(torch.randn(d_model, self.synapse_rank) * 0.01)
            if stable_init:
                init_s = torch.zeros(len(self.offsets), self.synapse_rank)
            else:
                init_s = torch.randn(len(self.offsets), self.synapse_rank) * 0.02
            self.s_scale = nn.ParameterDict(
                {
                    str(p): nn.Parameter(init_s[i].clone())
                    for i, p in enumerate(self.offsets)
                }
            )
            if identity_transport:
                self.beta = nn.ParameterDict(
                    {str(p): nn.Parameter(torch.ones(1)) for p in self.offsets}
                )
            else:
                self.beta = None  # type: ignore[assignment]
        else:
            scale = d_model ** -0.5
            self.w_dense = nn.ParameterDict(
                {
                    str(p): nn.Parameter(torch.randn(d_model, d_model) * scale)
                    for p in self.offsets
                }
            )

        # Direct Trace Tap (V4): pristine memory bypass parameters
        if trace_tap and synapse_mode == "factorized":
            self.U_tr = nn.Parameter(torch.randn(d_model, self.synapse_rank) * 0.01)
            self.V_tr = nn.Parameter(torch.randn(d_model, self.synapse_rank) * 0.01)
            self.s_scale_tr = nn.ParameterDict(
                {
                    str(p): nn.Parameter(torch.zeros(self.synapse_rank))
                    for p in self.offsets
                }
            )
            self.beta_tr = nn.ParameterDict(
                {str(p): nn.Parameter(torch.ones(1) * 0.5) for p in self.offsets}
            )
            self.gamma_tr = nn.ParameterDict(
                {str(p): nn.Parameter(torch.ones(1) * 0.1) for p in self.offsets}
            )
        else:
            self.U_tr = None
            self.V_tr = None
            self.s_scale_tr = None
            self.beta_tr = None
            self.gamma_tr = None

        if hierarchical_gate:
            self.rho = nn.ParameterDict(
                {str(p): nn.Parameter(torch.zeros(1)) for p in self.offsets}
            )
        else:
            self.rho = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Parallel training path: T_prev (B, N, d), T_0 (B, N, d) -> T_pos (B, N, d)
    # ------------------------------------------------------------------
    @staticmethod
    def _causal_views(x: torch.Tensor, offsets: List[int]) -> torch.Tensor:
        """Stack causal shifts with one padding allocation instead of P pads."""
        n, max_offset = x.size(1), max(offsets)
        padded = F.pad(x, (0, 0, max_offset, 0))
        return torch.stack(
            [padded[:, max_offset - p : max_offset - p + n] for p in offsets],
            dim=2,
        )

    def forward_train(
        self, T_prev: torch.Tensor, T_0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Vectorized parallel path over every offset in the layer.

        The previous implementation launched the full gate/transport pipeline
        separately for each offset.  Stacking the (at most four) resonant views
        turns those dozens of small kernels into batched contractions that
        ``torch.compile`` can fuse, without changing the equations.
        """
        # Offsets beyond this concrete sequence have no causally available
        # source (the eager reference path skipped them as well).
        offsets = [p for p in self.offsets if p < T_prev.size(1)]
        keys = [str(p) for p in offsets]
        scale_dg = 1.0 / math.sqrt(self.d_gate)
        sources = self._causal_views(T_prev, offsets)  # (B,N,P,d)

        k_all = self.p_g_src(T_prev)
        k_src = self._causal_views(k_all, offsets)  # (B,N,P,dg)
        gate_w_src = torch.stack([self.gate_w_src[k] for k in keys])
        gate_b = torch.stack([self.gate_b[k] for k in keys])
        g_src = torch.einsum("bnpq,pqd->bnpd", k_src, gate_w_src)

        if self.gating_mode == "dual" and self.p_g_tgt is not None:
            q_tgt = self.p_g_tgt(T_prev)
            gate_w_tgt = torch.stack([self.gate_w_tgt[k] for k in keys])
            g_tgt = torch.einsum("bnq,pqd->bnpd", q_tgt, gate_w_tgt)
            bilinear = (q_tgt.unsqueeze(2) * k_src).sum(dim=-1, keepdim=True)
            gates = _sigmoid(g_src + g_tgt + bilinear * scale_dg + gate_b)
        else:
            gates = _sigmoid(g_src + gate_b)

        if self.synapse_mode == "factorized":
            z_prev = T_prev @ self.V
            z_src = self._causal_views(z_prev, offsets)
            scales = torch.stack([self.s_scale[k] for k in keys])
            y_ctx = (z_src * scales.view(1, 1, len(offsets), -1)) @ self.U.T
            if self.identity_transport and self.beta is not None:
                beta = torch.stack([self.beta[k] for k in keys]).view(1, 1, -1, 1)
                y_ctx = y_ctx + beta * sources
        else:
            dense = torch.stack([self.w_dense[k] for k in keys])
            y_ctx = torch.einsum("bnpd,pde->bnpe", sources, dense)

        y_total = y_ctx
        if self.trace_tap and T_0 is not None and self.V_tr is not None:
            trace_sources = self._causal_views(T_0, offsets)
            z_trace = self._causal_views(T_0 @ self.V_tr, offsets)
            trace_scales = torch.stack([self.s_scale_tr[k] for k in keys])
            y_trace = (z_trace * trace_scales.view(1, 1, len(offsets), -1)) @ self.U_tr.T
            beta_tr = torch.stack([self.beta_tr[k] for k in keys]).view(1, 1, -1, 1)
            gamma_tr = torch.stack([self.gamma_tr[k] for k in keys]).view(1, 1, -1, 1)
            y_total = y_total + gamma_tr * (y_trace + beta_tr * trace_sources)

        if self.hierarchical_gate and self.rho is not None:
            rho = _sigmoid(torch.stack([self.rho[k] for k in keys])).view(1, 1, -1, 1)
            gates = gates * rho
        return (gates * y_total).sum(dim=2)

    # ------------------------------------------------------------------
    # Incremental path: end-relative reads from retained windows.
    # ------------------------------------------------------------------
    def forward_step(
        self,
        history: Sequence[torch.Tensor],
        trace_history: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if not history:
            raise ValueError("forward_step requires a non-empty history window")

        current_state = history[-1]
        scale_dg = 1.0 / math.sqrt(self.d_gate)

        if self.gating_mode == "dual" and self.p_g_tgt is not None:
            q_tgt = self.p_g_tgt(current_state)
        else:
            q_tgt = None

        t_pos = torch.zeros_like(current_state)
        for p in self.offsets:
            if p + 1 > len(history):
                continue
            str_p = str(p)
            src_state = history[-(p + 1)]
            k_src = self.p_g_src(src_state)

            if self.gating_mode == "dual" and q_tgt is not None and self.gate_w_tgt is not None:
                bilinear = (q_tgt * k_src).sum(dim=-1, keepdim=True) * scale_dg
                g_tgt = q_tgt @ self.gate_w_tgt[str_p]
                g_src = k_src @ self.gate_w_src[str_p]
                g = _sigmoid(bilinear + g_tgt + g_src + self.gate_b[str_p])
            else:
                g = _sigmoid(k_src @ self.gate_w_src[str_p] + self.gate_b[str_p])

            if self.synapse_mode == "factorized":
                z_s = src_state @ self.V
                low_rank = (z_s * self.s_scale[str_p]) @ self.U.T
                if self.identity_transport and self.beta is not None:
                    y_ctx = self.beta[str_p] * src_state + low_rank
                else:
                    y_ctx = low_rank
            else:
                y_ctx = src_state @ self.w_dense[str_p]

            if (
                self.trace_tap
                and trace_history is not None
                and (p + 1 <= len(trace_history))
                and self.U_tr is not None
            ):
                t0_s = trace_history[-(p + 1)]
                z0_s = t0_s @ self.V_tr
                low_rank_tr = (z0_s * self.s_scale_tr[str_p]) @ self.U_tr.T
                y_tr = self.beta_tr[str_p] * t0_s + low_rank_tr
                y_total = y_ctx + self.gamma_tr[str_p] * y_tr
            else:
                y_total = y_ctx

            if self.hierarchical_gate and self.rho is not None:
                rho_p = _sigmoid(self.rho[str_p])
            else:
                rho_p = 1.0

            t_pos = t_pos + rho_p * (g * y_total)

        return t_pos

    # ------------------------------------------------------------------
    def identity_fidelity(self) -> Optional[float]:
        """Mean |beta| of identity routes (inspection helper)."""
        if self.synapse_mode != "factorized" or self.beta is None:
            return None
        return float(
            torch.stack([self.beta[s].detach().abs() for s in self.gate_w_src])
            .mean()
            .item()
        )


class ConsolidationLayer(nn.Module):
    """One consolidation layer: positional dilated mix + per-channel Cell."""

    def __init__(self, config: EngramaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        offsets = config.get_layer_offsets(layer_idx, config.num_consolidation_layers)
        self.mix = PositionalDilatedMix(
            d_model=config.d_model,
            d_gate=config.d_gate,
            offsets=offsets,
            synapse_mode=config.synapse_mode,
            synapse_rank=config.synapse_rank,
            identity_transport=config.identity_transport,
            hierarchical_gate=config.hierarchical_gate,
            stable_init=config.stable_init,
            gating_mode=config.gating_mode or "source",
            trace_tap=bool(config.trace_tap),
        )
        self.cell = Cell(
            config.d_model,
            config.d_ff,
            config.dropout,
            config.activation,
            norm_type=config.norm_type or "layernorm",
        )
        self.max_offset = self.mix.max_offset

    def forward_train(
        self, T_prev: torch.Tensor, T_0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.cell(self.mix.forward_train(T_prev, T_0=T_0))

    def forward_step(
        self,
        history: Sequence[torch.Tensor],
        trace_history: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        return self.cell(self.mix.forward_step(history, trace_history=trace_history))

    def forward(
        self,
        x: Union[torch.Tensor, Sequence[torch.Tensor]],
        history: bool = False,
        T_0: Optional[torch.Tensor] = None,
        trace_history: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if history:
            return self.forward_step(x, trace_history=trace_history)  # type: ignore[arg-type]
        return self.forward_train(x, T_0=T_0)  # type: ignore[arg-type]


class ConsolidationStack(nn.Module):
    """Stack of L consolidation layers with train/step execution paths."""

    def __init__(self, config: EngramaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                ConsolidationLayer(config, i)
                for i in range(config.num_consolidation_layers)
            ]
        )

    # ------------------------- training --------------------------------
    def forward_train(
        self, T0: torch.Tensor, T0_pristine: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pristine = T0 if T0_pristine is None else T0_pristine
        t = T0
        for layer in self.layers:
            t = layer.forward_train(t, T_0=pristine if self.config.trace_tap else None)
        return t

    def forward(
        self, T0: torch.Tensor, T0_pristine: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.forward_train(T0, T0_pristine=T0_pristine)

    # ------------------------- incremental -----------------------------
    def step_forward(
        self,
        cache: Any,
        T0_current: torch.Tensor,
        timestamp: int,
        return_all_layers: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Compute T_1..T_L for current token and write them to cache."""
        cache.trace.append(T0_current, timestamp)

        layer_outputs: List[torch.Tensor] = []
        for l, layer in enumerate(self.layers):
            maxoff = layer.max_offset
            if l == 0:
                history = cache.trace_history(maxoff + 1)
            else:
                history = cache.layer_history(l - 1, maxoff + 1)

            trace_history = (
                cache.trace_history(maxoff + 1) if self.config.trace_tap else None
            )
            t_out = layer.forward_step(history, trace_history=trace_history)
            cache.states.append(l, t_out)
            layer_outputs.append(t_out)

        cache.commit_step()
        t_l = layer_outputs[-1]
        if return_all_layers:
            return t_l, layer_outputs
        return t_l
