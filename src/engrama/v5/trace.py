"""ENGRAMA V5 — Traza explicita y cache incremental.

La Traza V5 guarda, SIN comprimir y SIN transformar:
* ``T0_ring``: la huella pristina de cada token (N_max x d)  — pilar 2,
* ``K_ring``:  el codigo de lectura ``P_k(T0)`` de cada token (N_max x d_k),
* horizontes minimos por capa (invariante V3 §24): solo los estados que la
  capa siguiente va a leer.

La generacion incremental es O(1) en consolidacion (offsets fijos) y O(N d_k)
en la lectura (un matvec contra ``K_ring``): lineal en el contexto, como un
KV-cache pero con UN solo eje y sin matrices K/V por capa.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional, Sequence, Tuple

import torch


class V5Trace:
    """Memoria de trabajo explicita de ENGRAMA V5 (nunca comprime)."""

    def __init__(
        self,
        n_max: int,
        d_model: int,
        d_recall: int,
        horizons: Sequence[int],
        *,
        capacity: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        if n_max < 2:
            raise ValueError("n_max >= 2")
        self.n_max = int(n_max)
        self.d_model = int(d_model)
        self.d_recall = int(d_recall)
        self.horizons = list(int(h) for h in horizons)
        # Anillos preasignados: append O(1), lectura de traza O(1) por indice.
        # Se aplanan a (cap, d) para B=1 y (cap, B, d) para lotes (lazy).
        cap = int(capacity or n_max)
        self._cap = cap
        self._device = device
        self._dtype = dtype
        self.t0_ring = torch.zeros(cap, self.d_model, device=device, dtype=dtype)
        self.k_ring = torch.zeros(cap, self.d_recall, device=device, dtype=dtype)
        self.length = 0                     # tokens escritos
        self._start = 0                     # inicio logico (para FIFO circular)
        self.layer_buffers: List[deque] = [
            deque(maxlen=h) for h in self.horizons
        ]

    # ------------------------------------------------------------------
    def _ensure_shape(self, t0: torch.Tensor) -> None:
        want = (self._cap,) + tuple(t0.shape)
        if tuple(self.t0_ring.shape) != want:
            self.t0_ring = torch.zeros(want, device=self._device, dtype=self._dtype)
            dk = self.k_ring.size(-1)
            self.k_ring = torch.zeros(want[:-1] + (dk,), device=self._device, dtype=self._dtype)

    def append_t0(self, t0: torch.Tensor, k: Optional[torch.Tensor] = None) -> None:
        """Escribe la huella (y su codigo K) del token actual (shape (B, d))."""
        self._ensure_shape(t0)
        if self.length >= self.n_max:
            # FIFO circular: se descarta el mas antiguo (sliding window).
            self._start = (self._start + 1) % self.n_max
            self.length = self.n_max - 1
        idx = (self._start + self.length) % self.n_max
        self.t0_ring[idx] = t0
        if k is not None:
            self.k_ring[idx] = k
        self.length += 1

    def append_layer(self, layer: int, state: torch.Tensor) -> None:
        self.layer_buffers[layer].append(state)

    # ------------------------------------------------------------------
    def t0_history(self, count: int) -> List[torch.Tensor]:
        """Ultimas ``count`` huellas, de la mas antigua a la mas reciente."""
        count = max(0, min(count, self.length))
        base = self._start + self.length
        return [self.t0_ring[(base - i - 1) % self.n_max] for i in range(count)][::-1]

    def layer_history(self, layer: int, count: int) -> List[torch.Tensor]:
        buf = self.layer_buffers[layer]
        return list(buf)[-max(0, count):]

    # ------------------------------------------------------------------
    def linear_views(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(T0, K)`` lineales (copia solo si el anillo dio la vuelta)."""
        if self._start == 0:
            return self.t0_ring[: self.length], self.k_ring[: self.length]
        idx = (self._start + torch.arange(self.length, device=self.t0_ring.device)) % self.n_max
        return self.t0_ring[idx], self.k_ring[idx]

    # ------------------------------------------------------------------
    def memory_bytes(self) -> int:
        rings = self.t0_ring.numel() * self.t0_ring.element_size()
        rings += self.k_ring.numel() * self.k_ring.element_size()
        layers = sum(
            b[0].numel() * b[0].element_size() for b in self.layer_buffers if b
        )
        return int(rings + layers)

    def describe(self) -> dict:
        return {
            "n_max": self.n_max,
            "length": self.length,
            "t0_ring_bytes": self.t0_ring.numel() * self.t0_ring.element_size(),
            "k_ring_bytes": self.k_ring.numel() * self.k_ring.element_size(),
            "horizons": self.horizons,
            "bytes_per_token": (self.d_model + self.d_recall) * self.t0_ring.element_size(),
            "total_bytes": self.memory_bytes(),
        }

    def __len__(self) -> int:
        return self.length
