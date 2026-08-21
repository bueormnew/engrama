"""ENGRAMA V5 Synaptic Resonance — content-addressable recall, no attention.

For each head ``h`` and each pair (target ``t``, source ``j <= t``)::

    g[t,j] = sigmoid(tau_h * <qhat_t, khat_j> + b_h)      (point-to-point gate)
    r_t    = sum_{j<=t} g[t,j] * v_j        (Hebbian superposition read)

Key properties (see ``ENGRAMA-V5-Teorica.md``):

* **No softmax over positions.** The gate for pair (t, j) is decided
  independently; positions never compete for a normalized probability mass.
  This is the ENGRAMA synapse generalized to the full temporal range, NOT
  attention — and it is exactly what makes long-range retrieval robust.
* **No compressed state.** The parallel path reads the explicit key/value of
  every token; the incremental path reads the explicit trace in the cache.
* **Exact causal invariance.** The triangular mask in the parallel path equals
  the causal accumulation in :meth:`step`.

Three execution paths, all mathematically identical:

* :meth:`forward` — full ``N×N`` masked score in one matmul (fastest for short N).
* :meth:`forward_chunked` — causal tiling for huge contexts (bounded memory,
  no softmax means no max-subtraction trick is needed).
* :meth:`step` — single-token incremental read over the explicit trace cache.

Author: ENGRAMA contributors.
License: AGPL-3.0
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from engrama.v5.primitives import make_norm


class SynapticResonance(nn.Module):
    """Multi-head synaptic-resonance recall over an explicit trace."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        read_norm: Optional[str] = "softcount",
        tau_init: float = 8.0,
        norm_type: str = "rmsnorm",
        eps: float = 1e-4,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.h = num_heads
        self.dh = d_model // num_heads
        self.read_norm = read_norm
        self.eps = eps

        self.norm = make_norm(norm_type, d_model)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        # Per-head resonance sharpness (softplus keeps it positive & NaN-free).
        self.tau_raw = nn.Parameter(torch.full((num_heads,), _inv_softplus(tau_init)))
        self.gate_bias = nn.Parameter(torch.zeros(num_heads))

    # ------------------------------------------------------------------
    def tau(self) -> torch.Tensor:
        return F.softplus(self.tau_raw)

    def _project(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (q_hat, k_hat, v) each shaped (B, H, N, dh)."""
        B, N, _ = x.shape
        xn = self.norm(x)
        q = self.wq(xn).view(B, N, self.h, self.dh).transpose(1, 2)
        k = self.wk(xn).view(B, N, self.h, self.dh).transpose(1, 2)
        v = self.wv(xn).view(B, N, self.h, self.dh).transpose(1, 2)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        return q, k, v

    def _gate_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Sigmoid resonance gates ``g[b,h,t,j]`` (no softmax)."""
        tau = self.tau().view(1, self.h, 1, 1)
        b = self.gate_bias.view(1, self.h, 1, 1)
        s = torch.matmul(q, k.transpose(-1, -2))  # cosine sims in [-1, 1]
        # Sigmoid in fp32 under AMP to avoid saturation artefacts.
        if s.dtype in (torch.float16, torch.bfloat16):
            return torch.sigmoid((tau.float() * s.float() + b.float())).to(s.dtype)
        return torch.sigmoid(tau * s + b)

    # ------------------------------------------------------------------
    # Parallel path (training / full-context eval)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        q, k, v = self._project(x)
        g = self._gate_scores(q, k)  # (B, H, N, N)
        mask = torch.ones(N, N, device=x.device, dtype=torch.bool).tril_()
        g = g.masked_fill(~mask, 0.0)
        r = torch.matmul(g, v)  # (B, H, N, dh)
        if self.read_norm == "softcount":
            denom = g.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            r = r / denom
        r = r.transpose(1, 2).reshape(B, N, self.d_model)
        return x + self.wo(r)

    # ------------------------------------------------------------------
    # Chunked parallel path for huge contexts (bounded activation memory)
    # ------------------------------------------------------------------
    def forward_chunked(self, x: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
        if chunk_size <= 0:
            return self.forward(x)
        B, N, _ = x.shape
        q, k, v = self._project(x)
        out = x.new_zeros(B, self.h, N, self.dh)
        tau = self.tau().view(1, self.h, 1, 1)
        bias = self.gate_bias.view(1, self.h, 1, 1)
        for t0 in range(0, N, chunk_size):
            t1 = min(N, t0 + chunk_size)
            qc = q[:, :, t0:t1]  # (B,H,C,dh)
            acc = x.new_zeros(B, self.h, t1 - t0, self.dh)
            den = x.new_zeros(B, self.h, t1 - t0, 1)
            # Only keys up to t1 can be causally attended.
            for j0 in range(0, t1, chunk_size):
                j1 = min(t1, j0 + chunk_size)
                kc = k[:, :, j0:j1]
                vc = v[:, :, j0:j1]
                s = torch.matmul(qc, kc.transpose(-1, -2))
                if s.dtype in (torch.float16, torch.bfloat16):
                    g = torch.sigmoid(tau.float() * s.float() + bias.float()).to(s.dtype)
                else:
                    g = torch.sigmoid(tau * s + bias)
                # Causal mask only needed on the diagonal tile.
                if j1 > t0:
                    rows = torch.arange(t0, t1, device=x.device).view(-1, 1)
                    cols = torch.arange(j0, j1, device=x.device).view(1, -1)
                    g = g.masked_fill((cols > rows).unsqueeze(0).unsqueeze(0), 0.0)
                acc = acc + torch.matmul(g, vc)
                if self.read_norm == "softcount":
                    den = den + g.sum(dim=-1, keepdim=True)
            if self.read_norm == "softcount":
                acc = acc / den.clamp_min(self.eps)
            out[:, :, t0:t1] = acc
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return x + self.wo(out)

    # ------------------------------------------------------------------
    # Incremental path (native ENGRAMA cache, O(N) per token, no recompute)
    # ------------------------------------------------------------------
    def step(self, x_t: torch.Tensor, layer_cache) -> torch.Tensor:
        """Single-token read. ``x_t`` is (B, d_model); returns (B, d_model).

        Appends the current key/value to ``layer_cache`` and reads over the
        whole retained trace. Exactly equivalent to :meth:`forward` at position
        t by causal invariance.
        """
        B, d = x_t.shape
        xn = self.norm(x_t)
        q = self.wq(xn).view(B, self.h, self.dh)
        k = self.wk(xn).view(B, self.h, self.dh)
        v = self.wv(xn).view(B, self.h, self.dh)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        layer_cache.append(k, v)  # explicit trace, no compression
        K = layer_cache.keys()    # (B, H, T, dh)
        V = layer_cache.values()  # (B, H, T, dh)
        tau = self.tau().view(1, self.h, 1)
        bias = self.gate_bias.view(1, self.h, 1)
        s = torch.einsum("bhd,bhtd->bht", q, K)
        if s.dtype in (torch.float16, torch.bfloat16):
            g = torch.sigmoid(tau.float() * s.float() + bias.float()).to(s.dtype)
        else:
            g = torch.sigmoid(tau * s + bias)
        r = torch.einsum("bht,bhtd->bhd", g, V)
        if self.read_norm == "softcount":
            denom = g.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            r = r / denom
        r = r.reshape(B, self.d_model)
        return x_t + self.wo(r)


def _inv_softplus(y: float) -> float:
    """Return x such that softplus(x) == y (for initializing tau_raw)."""
    return math.log(math.expm1(y)) if y < 20 else y
