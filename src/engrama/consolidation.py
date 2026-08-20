"""ENGRAMA V3 Consolidation Stack (Phase 3).

Implements the hierarchical dilated causal consolidation of the V3 spec
(sections 8, 10, 17):

::

    T_pos_l[i] = sum_{p in D_l, p <= i} rho_{l,p} * G_{l,p}(T_{l-1}[i-p])

    G_{l,p}(x)   = alpha_{l,p}(x) (.) (beta_{l,p} x + U_l Diag(s_{l,p}) V_l^T x)

    T_l[i]       = Cell_l(T_pos_l[i])

where ``alpha_{l,p}`` is a per-channel gate computed from the **source**
state (spec section 7), ``rho_{l,p} = sigmoid(a_{l,p})`` is the scalar
per-scale gate (spec section 17), and ``U_l, V_l in R^{d x r}`` are the
shared bases of the layer (spec section 8.2, 35).

Both paths -- parallel training (``forward_train``) and token-by-token
inference with minimum-horizon windows (``forward_step``) -- are fully
vectorized over offsets and exactly equivalent (causal invariance, V3
spec section 23 / theorem 1; minimum-horizon correctness, section 24 /
theorem 2).

Author: BUEORM
License: AGPL-3.0
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from engrama.config import EngramaConfig
from engrama.primitives import Cell


class PositionalDilatedMix(nn.Module):
    """V3 positional dilated mix with factorized fidelity transport.

    Params per layer (offsets ``D_l``):

    - ``p_g``: gate projection ``d -> d_g`` (shared, per source).
    - ``gate_w[p] in R^{d_g x d}``, ``gate_b[p] in R^d``: per-channel gate.
    - ``U_l, V_l in R^{d x r}``: shared low-rank bases (spec section 8.2).
    - ``s_scale[p] in R^r`` and ``beta[p] in R``: per-offset specialization
      and identity route (spec sections 6, 10, 16).
    - ``rho[p] in R``: scalar per-scale gate (spec section 17), only when
      ``hierarchical_gate=True``.
    - ``w_dense[p] in R^{d x d}``: dense ablation (``synapse_mode="dense"``,
      spec ablation B/D of section 44).
    """

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
    ):
        super().__init__()
        if synapse_mode not in ("dense", "factorized"):
            raise ValueError("synapse_mode must be 'dense' or 'factorized'")
        if not offsets:
            raise ValueError("offsets must be a non-empty list")
        if any(o < 0 for o in offsets):
            raise ValueError("offsets must be non-negative and causal (>= 0)")

        self.d_model = d_model
        self.d_gate = d_gate
        self.offsets = sorted(dict.fromkeys(int(p) for p in offsets))
        self.synapse_mode = synapse_mode
        self.synapse_rank = min(synapse_rank, d_model)
        self.identity_transport = identity_transport
        self.hierarchical_gate = hierarchical_gate
        self.max_offset = max(self.offsets)

        self.p_g = nn.Linear(d_model, d_gate, bias=False)
        self.gate_w = nn.ParameterDict(
            {
                str(p): nn.Parameter(torch.randn(d_gate, d_model) * 0.02)
                for p in self.offsets
            }
        )
        self.gate_b = nn.ParameterDict(
            {str(p): nn.Parameter(torch.zeros(d_model)) for p in self.offsets}
        )

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

        if hierarchical_gate:
            self.rho = nn.ParameterDict(
                {str(p): nn.Parameter(torch.zeros(1)) for p in self.offsets}
            )
        else:
            self.rho = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Shared math
    # ------------------------------------------------------------------
    def _transform(self, x: torch.Tensor, str_p: str) -> torch.Tensor:
        """beta_p * x + U Diag(s_p) V^T x (factorized) or W_p x (dense)."""
        if self.synapse_mode == "factorized":
            low_rank = ((x @ self.V) * self.s_scale[str_p]) @ self.U.T
            if self.identity_transport and self.beta is not None:
                return self.beta[str_p] * x + low_rank
            return low_rank
        return x @ self.w_dense[str_p]

    def _gate(self, x: torch.Tensor, str_p: str) -> torch.Tensor:
        """Per-channel sigmoid gate computed from the source state."""
        q = self.p_g(x)
        return torch.sigmoid(q @ self.gate_w[str_p] + self.gate_b[str_p])

    def _scale_gate(self, str_p: str) -> Union[torch.Tensor, float]:
        """Scalar per-scale gate rho (spec section 17); 1.0 when disabled."""
        if self.hierarchical_gate and self.rho is not None:
            return torch.sigmoid(self.rho[str_p])
        return 1.0

    # ------------------------------------------------------------------
    # Parallel training path: T_prev (B, N, d) -> T_pos (B, N, d)
    # ------------------------------------------------------------------
    def forward_train(self, T_prev: torch.Tensor) -> torch.Tensor:
        b, n, d = T_prev.shape
        t_pos = torch.zeros_like(T_prev)
        for p in self.offsets:
            str_p = str(p)
            if p == 0:
                t_shifted = T_prev
            elif p < n:
                t_shifted = torch.cat(
                    [
                        torch.zeros(b, p, d, device=T_prev.device, dtype=T_prev.dtype),
                        T_prev[:, :-p, :],
                    ],
                    dim=1,
                )
            else:
                continue  # causal mask: p > i never contributes
            g = self._gate(t_shifted, str_p)
            y = self._transform(t_shifted, str_p)
            t_pos = t_pos + self._scale_gate(str_p) * g * y
        return t_pos

    # ------------------------------------------------------------------
    # Incremental path: end-relative reads from a retained window.
    # ``history``: newest-last list of (B, d) states of the PREVIOUS level.
    # ------------------------------------------------------------------
    def forward_step(self, history: Sequence[torch.Tensor]) -> torch.Tensor:
        if not history:
            raise ValueError("forward_step requires a non-empty history window")

        sources: List[torch.Tensor] = []
        valid_params: List[str] = []
        for p in self.offsets:
            if p + 1 <= len(history):  # causal availability mask
                sources.append(history[-(p + 1)])
                valid_params.append(str(p))
        if not sources:
            raise ValueError("No causally available offsets in history window")

        S = torch.stack(sources, dim=1)  # (B, K, d)
        q = self.p_g(S)  # (B, K, d_g)

        gate_w = torch.stack([self.gate_w[s] for s in valid_params])  # (K, d_g, d)
        gate_b = torch.stack([self.gate_b[s] for s in valid_params])  # (K, d)
        g = torch.sigmoid(torch.einsum("bkg,kgd->bkd", q, gate_w) + gate_b)

        if self.synapse_mode == "factorized":
            z = S @ self.V  # (B, K, r)
            s_vec = torch.stack([self.s_scale[s] for s in valid_params])  # (K, r)
            y = (z * s_vec) @ self.U.T  # (B, K, d)
            if self.identity_transport and self.beta is not None:
                beta_vec = torch.stack([self.beta[s] for s in valid_params])  # (K, 1)
                y = beta_vec.unsqueeze(0) * S + y
        else:
            w = torch.stack([self.w_dense[s] for s in valid_params])  # (K, d_in, d_out)
            y = torch.einsum("kio,bki->bko", w, S)

        if self.hierarchical_gate and self.rho is not None:
            rho_vec = torch.stack([self.rho[s] for s in valid_params]).squeeze(-1)
            rho_vec = torch.sigmoid(rho_vec)  # (K,)
            return torch.einsum("k,bkd->bd", rho_vec, g * y)
        return (g * y).sum(dim=1)

    # ------------------------------------------------------------------
    def identity_fidelity(self) -> Optional[float]:
        """Mean |beta| of identity routes (inspection helper, spec 31/50)."""
        if self.synapse_mode != "factorized" or self.beta is None:
            return None
        return float(
            torch.stack([self.beta[s].detach().abs() for s in self.gate_w])
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
        )
        self.cell = Cell(
            config.d_model, config.d_ff, config.dropout, config.activation
        )
        self.max_offset = self.mix.max_offset

    def forward_train(self, T_prev: torch.Tensor) -> torch.Tensor:
        return self.cell(self.mix.forward_train(T_prev))

    def forward_step(self, history: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.cell(self.mix.forward_step(history))

    def forward(
        self,
        x: Union[torch.Tensor, Sequence[torch.Tensor]],
        history: bool = False,
    ) -> torch.Tensor:
        if history:
            return self.forward_step(x)  # type: ignore[arg-type]
        return self.forward_train(x)  # type: ignore[arg-type]


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
    def forward_train(self, T0: torch.Tensor) -> torch.Tensor:
        t = T0
        for layer in self.layers:
            t = layer.forward_train(t)
        return t

    def forward(self, T0: torch.Tensor) -> torch.Tensor:
        return self.forward_train(T0)

    # ------------------------- incremental -----------------------------
    def step_forward(
        self,
        cache: Any,
        T0_current: torch.Tensor,
        timestamp: int,
        return_all_layers: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Compute T_1..T_L for the current token and write them to the cache.

        The trace and every per-layer buffer are written progressively, so
        each layer reads causally available states only -- exactly matching
        the parallel training path (V3 spec sections 13, 23, 24).
        """
        cache.trace.append(T0_current, timestamp)

        layer_outputs: List[torch.Tensor] = []
        for l, layer in enumerate(self.layers):
            maxoff = layer.max_offset
            if l == 0:
                history = cache.trace_history(maxoff + 1)
            else:
                history = cache.layer_history(l - 1, maxoff + 1)
            t_out = layer.forward_step(history)
            cache.states.append(l, t_out)
            layer_outputs.append(t_out)

        cache.commit_step()
        t_l = layer_outputs[-1]
        if return_all_layers:
            return t_l, layer_outputs
        return t_l
