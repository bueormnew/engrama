"""ENGRAMA V3 Trace and Hierarchical Cache (Phase 2).

Implements the explicit separation of V3 spec section 22:

- :class:`CircularTrace` -- the *semantic* working memory: a FIFO circular
  buffer of ``(T_0 vector, absolute timestamp)`` pairs with capacity
  ``N_max``. It only stores; it never transforms. All operations are O(1)
  (``collections.deque(maxlen=...)`` provides true circular overwrite, no
  physical shifting).

- :class:`HierarchicalStateCache` -- the *computational* memory: per-layer
  circular buffers of consolidated states ``T_l`` whose capacities follow
  the minimum-horizon theorem (V3 spec sections 12 and 24)::

      capacity(T_l) = max(D_{l+1}) + 1     for l < L - 1
      capacity(T_{L-1}) = 1

  The last layer only feeds the evoker, so a single state suffices.
  Buffer ``l - 1`` therefore always covers every offset of layer ``l``.

- :class:`EngramaCache` -- public facade combining both, with the legacy
  read-only views (``T0``, ``Tl``, ``timestamps``) kept for inspection.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Union

import torch


class CircularTrace:
    """FIFO circular trace of ``(vector, timestamp)`` pairs (capacity N_max).

    This is the V2/V3 "Traza": explicit, inspectable, and transform-free.
    ``deque(maxlen=N)`` yields O(1) append/evict semantics with logical
    circular overwrite (no physical shift), per V3 spec section 4.3.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("Trace capacity must be >= 1")
        self.capacity = capacity
        self._values: Deque[torch.Tensor] = deque(maxlen=capacity)
        self._timestamps: Deque[int] = deque(maxlen=capacity)

    def append(self, vector: torch.Tensor, timestamp: int) -> None:
        self._values.append(vector)
        self._timestamps.append(timestamp)

    def latest(self) -> torch.Tensor:
        if not self._values:
            raise IndexError("Trace is empty")
        return self._values[-1]

    def history(self, count: int) -> List[torch.Tensor]:
        """Return up to the last ``count`` vectors, oldest first."""
        if count <= 0:
            return []
        values = list(self._values)
        return values[-count:]

    @property
    def values(self) -> List[torch.Tensor]:
        return list(self._values)

    @property
    def timestamps(self) -> List[int]:
        return list(self._timestamps)

    def clear(self) -> None:
        self._values.clear()
        self._timestamps.clear()

    def to(self, device: Union[str, torch.device]) -> "CircularTrace":
        self._values = deque((t.to(device) for t in self._values), maxlen=self.capacity)
        return self

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"CircularTrace(len={len(self)}, capacity={self.capacity})"


class HierarchicalStateCache:
    """Per-layer circular buffers with minimum-horizon capacities (V3 §12)."""

    def __init__(self, capacities: Sequence[int], mode: str = "hierarchical"):
        if mode not in ("full", "hierarchical"):
            raise ValueError(f"cache mode must be 'full' or 'hierarchical', got {mode!r}")
        if not capacities:
            raise ValueError("capacities must be a non-empty sequence")
        if any(c < 1 for c in capacities):
            raise ValueError("All cache capacities must be >= 1")
        self.mode = mode
        self.capacities = [int(c) for c in capacities]
        self.buffers: List[Deque[torch.Tensor]] = [
            deque(maxlen=c) for c in self.capacities
        ]

    def append(self, layer: int, vector: torch.Tensor) -> None:
        self.buffers[layer].append(vector)

    def latest(self, layer: int) -> torch.Tensor:
        if not self.buffers[layer]:
            raise IndexError(f"Cache buffer for layer {layer} is empty")
        return self.buffers[layer][-1]

    def history(self, layer: int, count: int) -> List[torch.Tensor]:
        """Return up to the last ``count`` states of a layer, oldest first."""
        if count <= 0:
            return []
        values = list(self.buffers[layer])
        return values[-count:]

    def clear(self) -> None:
        for buf in self.buffers:
            buf.clear()

    def to(self, device: Union[str, torch.device]) -> "HierarchicalStateCache":
        self.buffers = [
            deque((t.to(device) for t in buf), maxlen=buf.maxlen) for buf in self.buffers
        ]
        return self

    def __len__(self) -> int:
        return len(self.buffers)

    def __repr__(self) -> str:
        lens = [len(b) for b in self.buffers]
        return (
            f"HierarchicalStateCache(mode={self.mode!r}, "
            f"capacities={self.capacities}, lengths={lens})"
        )


class EngramaCache:
    """Combined trace + consolidated-state cache used by ``step_forward``.

    Args:
        N_max: Capacity of the circular trace (context window).
        num_layers: Number of consolidation layers.
        d_model: Hidden dimension (metadata for inspection).
        mode: ``"full"`` (V2: every layer keeps up to ``N_max`` states) or
            ``"hierarchical"`` (V3: per-layer minimum horizons).
        horizons: Required when ``mode="hierarchical"``: number of states
            retained per consolidation layer (see ``config.cache_horizons()``).
    """

    def __init__(
        self,
        N_max: int,
        num_layers: int,
        d_model: int,
        mode: str = "hierarchical",
        horizons: Optional[Sequence[int]] = None,
    ):
        if mode == "hierarchical":
            if horizons is None:
                raise ValueError("horizons must be provided in hierarchical cache mode")
            capacities: List[int] = [int(h) for h in horizons]
            if len(capacities) != num_layers:
                raise ValueError(
                    f"horizons length {len(capacities)} != num_layers {num_layers}"
                )
        else:
            capacities = [N_max] * num_layers

        self.N_max = N_max
        self.num_layers = num_layers
        self.d_model = d_model
        self.mode = mode
        self.trace = CircularTrace(N_max)
        self.states = HierarchicalStateCache(capacities, mode=mode)
        self._tokens_written = 0

    # ------------------------------------------------------------------
    # Public write API (kept stable for EngramaModel.step_forward)
    # ------------------------------------------------------------------
    def append(
        self,
        T0_v: torch.Tensor,
        Tl_v: Union[List[torch.Tensor], Dict[int, torch.Tensor]],
        timestamp: int,
    ) -> None:
        """Append one full step: trace vector + one state per layer."""
        self.trace.append(T0_v, timestamp)
        for i in range(self.num_layers):
            self.states.append(i, Tl_v[i])
        self.commit_step()

    def commit_step(self) -> None:
        """Mark one token as fully written (used after progressive writes)."""
        self._tokens_written += 1

    # ------------------------------------------------------------------
    # Read helpers consumed by ConsolidationStack.step_forward
    # ------------------------------------------------------------------
    def trace_history(self, count: int) -> List[torch.Tensor]:
        return self.trace.history(count)

    def layer_history(self, layer: int, count: int) -> List[torch.Tensor]:
        return self.states.history(layer, count)

    def latest(self, layer: Optional[int] = None) -> torch.Tensor:
        """Latest trace vector (``layer=None``) or latest state of a layer."""
        if layer is None:
            return self.trace.latest()
        return self.states.latest(layer)

    @property
    def tokens_written(self) -> int:
        """Absolute number of tokens appended since the last reset."""
        return self._tokens_written

    # ------------------------------------------------------------------
    # Lifecycle / devices
    # ------------------------------------------------------------------
    def clear(self) -> None:
        self.trace.clear()
        self.states.clear()
        self._tokens_written = 0

    def to(self, device: Union[str, torch.device]) -> "EngramaCache":
        self.trace.to(device)
        self.states.to(device)
        return self

    # ------------------------------------------------------------------
    # Inspection (legacy-compatible views)
    # ------------------------------------------------------------------
    @property
    def T0(self) -> List[torch.Tensor]:
        return self.trace.values

    @property
    def Tl(self) -> List[List[torch.Tensor]]:
        return [list(buf) for buf in self.states.buffers]

    @property
    def timestamps(self) -> List[int]:
        return self.trace.timestamps

    def absolute_index(self) -> int:
        """Current absolute sequence position (tokens written - 1)."""
        return max(0, self._tokens_written - 1)

    def get_memory_footprint(self) -> int:
        """Footprint in bytes of every retained tensor."""
        total = sum(t.numel() * t.element_size() for t in self.trace.values)
        for buf in self.states.buffers:
            total += sum(t.numel() * t.element_size() for t in buf)
        return total

    def describe(self) -> Dict[str, Any]:
        """Structured description of cache state and memory savings."""
        per_layer = [
            {"layer": l, "capacity": self.states.capacities[l], "length": len(buf)}
            for l, buf in enumerate(self.states.buffers)
        ]
        full_states = self.N_max * self.num_layers
        actual_states = sum(self.states.capacities)
        return {
            "mode": self.mode,
            "N_max": self.N_max,
            "trace_length": len(self.trace),
            "trace_capacity": self.trace.capacity,
            "tokens_written": self._tokens_written,
            "layers": per_layer,
            "total_state_capacity": actual_states,
            "full_cache_equivalent_capacity": full_states,
            "state_reduction_ratio": (
                round(full_states / actual_states, 2) if actual_states else None
            ),
            "memory_bytes": self.get_memory_footprint(),
        }

    def __len__(self) -> int:
        return len(self.trace)

    def __repr__(self) -> str:
        return (
            f"EngramaCache(mode={self.mode!r}, trace={len(self.trace)}/{self.N_max}, "
            f"capacities={self.states.capacities})"
        )
