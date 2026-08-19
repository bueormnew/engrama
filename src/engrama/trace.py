"""
ENGRAMA Cache and Circular Trace Module (Phase 2)
Author: BUEORM
License: AGPL-3.0
"""

from typing import Any, Dict, List, Optional, Union
import torch


class EngramaCache:
    """Incremental Logarithmic Circular Cache (Trace) for ENGRAMA inference.

    Stores historical token states T_0 and layer representations T_l up to context size `N_max`
    to guarantee strictly equivalent incremental inference without dynamic key-value attention matrices.

    Args:
        N_max (int): Maximum capacity (context length) of the trace.
        num_layers (int): Number of consolidation layers in the model stack.
        d_model (int): Model hidden dimension size.
    """

    def __init__(self, N_max: int, num_layers: int, d_model: int):
        self.N_max = N_max
        self.num_layers = num_layers
        self.d_model = d_model
        self.T0: List[torch.Tensor] = []
        self.Tl: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
        self.timestamps: List[int] = []

    def append(
        self,
        T0_v: torch.Tensor,
        Tl_v: Union[List[torch.Tensor], Dict[int, torch.Tensor]],
        timestamp: int,
    ) -> None:
        """Append a step state into trace buffers."""
        self.T0.append(T0_v)
        if isinstance(Tl_v, dict):
            for i in range(self.num_layers):
                self.Tl[i].append(Tl_v[i])
        else:
            for i in range(self.num_layers):
                self.Tl[i].append(Tl_v[i])
        self.timestamps.append(timestamp)

        if len(self.T0) > self.N_max:
            self.T0.pop(0)
            for i in range(self.num_layers):
                self.Tl[i].pop(0)
            self.timestamps.pop(0)

    def clear(self) -> None:
        """Clear all historical states in cache."""
        self.T0.clear()
        for layer_list in self.Tl:
            layer_list.clear()
        self.timestamps.clear()

    def to(self, device: Union[str, torch.device]) -> "EngramaCache":
        """Move all cached tensors to specified device."""
        self.T0 = [t.to(device) for t in self.T0]
        self.Tl = [[t.to(device) for t in layer] for layer in self.Tl]
        return self

    def get_memory_footprint(self) -> int:
        """Estimate peak memory usage of cache in bytes."""
        total_bytes = sum(t.numel() * t.element_size() for t in self.T0)
        for layer in self.Tl:
            total_bytes += sum(t.numel() * t.element_size() for t in layer)
        return total_bytes

    def __len__(self) -> int:
        return len(self.T0)

    def absolute_index(self) -> int:
        """Return the current absolute sequence position (cache length)."""
        return len(self.T0)

    def __repr__(self) -> str:
        return (
            f"EngramaCache(len={len(self)}, N_max={self.N_max}, "
            f"num_layers={self.num_layers}, d_model={self.d_model})"
        )
