"""ENGRAMA V5 native generation cache — the explicit trace of (key, value) pairs.

This is the ENGRAMA-native equivalent of a Transformer KV-cache, but it stores
the *explicit trace* (one key/value per token, no compression) and is read with
synaptic resonance instead of softmax attention. On each step the new key/value
is appended (O(1)) and the read scans the retained trace (O(N)); previous tokens
are never recomputed.

The cache grows up to ``N_max``; beyond that it keeps the most recent ``N_max``
pairs (FIFO circular trace). Set ``N_max`` large enough for your context.

Author: ENGRAMA contributors.
License: AGPL-3.0
"""

from __future__ import annotations

from typing import List, Optional

import torch


class LayerCache:
    """Explicit per-layer trace of keys/values, shaped (B, H, T, dh)."""

    def __init__(self, n_max: int):
        self.n_max = n_max
        self._k: Optional[torch.Tensor] = None
        self._v: Optional[torch.Tensor] = None

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append one step. ``k``/``v`` are (B, H, dh)."""
        k = k.unsqueeze(2)
        v = v.unsqueeze(2)
        if self._k is None:
            self._k, self._v = k, v
        else:
            self._k = torch.cat([self._k, k], dim=2)
            self._v = torch.cat([self._v, v], dim=2)
        if self._k.size(2) > self.n_max:
            self._k = self._k[:, :, -self.n_max:]
            self._v = self._v[:, :, -self.n_max:]

    def keys(self) -> torch.Tensor:
        return self._k

    def values(self) -> torch.Tensor:
        return self._v

    def __len__(self) -> int:
        return 0 if self._k is None else self._k.size(2)


class ResonanceCache:
    """One :class:`LayerCache` per resonance layer."""

    def __init__(self, num_layers: int, n_max: int):
        self.n_max = n_max
        self.layers: List[LayerCache] = [LayerCache(n_max) for _ in range(num_layers)]

    def layer(self, i: int) -> LayerCache:
        return self.layers[i]

    def clear(self) -> None:
        for lc in self.layers:
            lc._k = None
            lc._v = None

    def __len__(self) -> int:
        return len(self.layers[0]) if self.layers else 0

    def memory_bytes(self) -> int:
        total = 0
        for lc in self.layers:
            if lc._k is not None:
                total += lc._k.numel() * lc._k.element_size()
                total += lc._v.numel() * lc._v.element_size()
        return total
