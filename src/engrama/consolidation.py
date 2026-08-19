from typing import Any, List, Optional, Tuple, Union
import torch
from torch import nn

from engrama.primitives import Cell


class PositionalDilatedMix(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_gate: int,
        offsets: List[int],
        synapse_mode: str = "factorized",
        synapse_rank: int = 32,
        identity_transport: bool = True,
        hierarchical_gate: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_gate = d_gate
        self.offsets = list(offsets)
        self.synapse_mode = synapse_mode
        self.synapse_rank = synapse_rank
        self.identity_transport = identity_transport
        self.hierarchical_gate = hierarchical_gate

        self.p_g = nn.Linear(d_model, d_gate, bias=False)
        self.gate_w = nn.ParameterDict(
            {
                str(p): nn.Parameter(torch.randn(d_gate, d_model) * 0.01)
                for p in self.offsets
            }
        )
        self.gate_b = nn.ParameterDict(
            {str(p): nn.Parameter(torch.zeros(d_model)) for p in self.offsets}
        )

        if synapse_mode == "factorized":
            self.v_proj = nn.ModuleDict(
                {
                    str(p): nn.Linear(d_model, synapse_rank, bias=False)
                    for p in self.offsets
                }
            )
            self.u_proj = nn.ModuleDict(
                {
                    str(p): nn.Linear(synapse_rank, d_model, bias=False)
                    for p in self.offsets
                }
            )
            self.s_scale = nn.ParameterDict(
                {
                    str(p): nn.Parameter(torch.randn(synapse_rank) * 0.01)
                    for p in self.offsets
                }
            )
            if identity_transport:
                self.beta_id = nn.ParameterDict(
                    {str(p): nn.Parameter(torch.ones(1)) for p in self.offsets}
                )
        else:
            self.w_offsets = nn.ModuleDict(
                {str(p): nn.Linear(d_model, d_model, bias=False) for p in self.offsets}
            )

        if hierarchical_gate:
            self.rho = nn.ParameterDict(
                {str(p): nn.Parameter(torch.zeros(1)) for p in self.offsets}
            )

    def _transform(self, x: torch.Tensor, str_p: str) -> torch.Tensor:
        if self.synapse_mode == "factorized":
            v = self.v_proj[str_p](x)
            v_scaled = v * self.s_scale[str_p]
            low_rank = self.u_proj[str_p](v_scaled)
            if self.identity_transport and self.beta_id is not None:
                return self.beta_id[str_p] * x + low_rank
            return low_rank
        return self.w_offsets[str_p](x)

    def _scale_gate(self, str_p: str) -> float:
        if self.rho and str_p in self.rho:
            return torch.sigmoid(self.rho[str_p])
        return 1.0

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
                continue
            h_g = self.p_g(t_shifted)
            gate_logits = torch.matmul(h_g, self.gate_w[str_p]) + self.gate_b[str_p]
            g = torch.sigmoid(gate_logits)
            transformed = self._transform(t_shifted, str_p)
            t_pos = t_pos + self._scale_gate(str_p) * g * transformed
        return t_pos

    def forward_step(
        self,
        T_prev_history: Union[torch.Tensor, List[torch.Tensor]],
        current_pos: int,
    ) -> torch.Tensor:
        if isinstance(T_prev_history, torch.Tensor):
            b, n_hist, d = T_prev_history.shape
            device = T_prev_history.device
            dtype = T_prev_history.dtype
        else:
            first = T_prev_history[0]
            b = first.shape[0]
            d = first.shape[-1]
            device = first.device
            dtype = first.dtype

        t_pos = torch.zeros(b, d, device=device, dtype=dtype)
        for p in self.offsets:
            idx = current_pos - p
            if idx < 0:
                continue

            if isinstance(T_prev_history, torch.Tensor):
                if idx >= T_prev_history.shape[1]:
                    continue
                t_p = T_prev_history[:, idx, :]
            else:
                if idx >= len(T_prev_history):
                    continue
                t_p = T_prev_history[idx]
                if t_p.dim() == 3:
                    t_p = t_p.squeeze(1)

            str_p = str(p)
            h_g = self.p_g(t_p)
            gate_logits = torch.matmul(h_g, self.gate_w[str_p]) + self.gate_b[str_p]
            g = torch.sigmoid(gate_logits)
            transformed = self._transform(t_p, str_p)
            t_pos = t_pos + self._scale_gate(str_p) * g * transformed
        return t_pos


class ConsolidationLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_gate: int,
        offsets: List[int],
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        synapse_mode: str = "factorized",
        synapse_rank: int = 32,
        identity_transport: bool = True,
        hierarchical_gate: bool = True,
    ):
        super().__init__()
        self.mix = PositionalDilatedMix(
            d_model, d_gate, offsets, synapse_mode, synapse_rank, identity_transport, hierarchical_gate
        )
        self.cell = Cell(d_model, d_ff, dropout, activation)
        self.max_offset = max(offsets) if offsets else 0

    def forward_train(self, T_prev: torch.Tensor) -> torch.Tensor:
        t_pos = self.mix.forward_train(T_prev)
        return self.cell(t_pos)

    def forward_step(
        self,
        T_prev_history: Union[torch.Tensor, List[torch.Tensor]],
        current_pos: int,
    ) -> torch.Tensor:
        t_pos = self.mix.forward_step(T_prev_history, current_pos)
        return self.cell(t_pos)

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        current_pos: Optional[int] = None,
    ) -> torch.Tensor:
        if current_pos is not None:
            return self.forward_step(x, current_pos)
        return self.forward_train(x)


class ConsolidationStack(nn.Module):
    def __init__(
        self,
        config: EngramaConfig,
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                ConsolidationLayer(
                    d_model=config.d_model,
                    d_gate=config.d_gate,
                    offsets=config.get_layer_offsets(i, config.num_consolidation_layers),
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                    synapse_mode=config.synapse_mode,
                    synapse_rank=config.synapse_rank,
                    identity_transport=config.identity_transport,
                    hierarchical_gate=config.hierarchical_gate,
                )
                for i in range(config.num_consolidation_layers)
            ]
        )

    def forward_train(self, T0: torch.Tensor) -> torch.Tensor:
        t = T0
        for layer in self.layers:
            t = layer.forward_train(t)
        return t

    def forward(self, T0: torch.Tensor) -> torch.Tensor:
        return self.forward_train(T0)

    def step_forward(
        self,
        cache: Any,
        T0_current: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if T0_current is not None:
            history_t0 = cache.T0 + [T0_current]
            current_pos = cache.absolute_index()
        else:
            history_t0 = cache.T0
            current_pos = cache.absolute_index() - 1

        layer_outputs: List[torch.Tensor] = []
        for l, layer in enumerate(self.layers):
            if l == 0:
                prev_hist = history_t0
            else:
                prev_hist = cache.Tl[l - 1] + [layer_outputs[l - 1]]
            t_out = layer.forward_step(prev_hist, current_pos)
            layer_outputs.append(t_out)

        t_l = layer_outputs[-1]
        if return_all_layers:
            return t_l, layer_outputs
        return t_l