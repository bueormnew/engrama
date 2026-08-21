"""ENGRAMA V5 — Triton kernels for fused synaptic-resonance reads.

Two kernels, both **without softmax** (so no online max/renormalization is
needed — the read is a plain masked sum, which is exactly what makes ENGRAMA
kernels simpler and faster than attention):

* :func:`resonance_dense` — fused causal dense read, tiled FlashAttention-style
  but with a sigmoid gate instead of softmax. O(N^2) compute, O(1) extra memory
  (the N x N score is never materialized).
* :func:`resonance_blocksparse` — fused block-sparse read that only visits the
  ``top_k`` routed key-blocks per query-block. Sub-quadratic compute.

Both fall back to a numerically-identical pure-PyTorch implementation when
Triton or CUDA is unavailable (e.g. CPU), so the same code runs everywhere and
tests validate equivalence.

Author: ENGRAMA contributors.
License: AGPL-3.0
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton optional
    _HAS_TRITON = False


# =====================================================================
# Pure-PyTorch reference (always correct, runs on CPU and GPU)
# =====================================================================
def _dense_ref(q, k, v, tau, bias, read_norm, eps=1e-4):
    """q,k,v: (B,H,N,dh) normalized q,k. tau,bias: (H,). -> (B,H,N,dh)."""
    B, H, N, dh = q.shape
    s = torch.matmul(q, k.transpose(-1, -2))
    g = torch.sigmoid(tau.view(1, H, 1, 1) * s + bias.view(1, H, 1, 1))
    mask = torch.ones(N, N, device=q.device, dtype=torch.bool).tril_()
    g = g.masked_fill(~mask, 0.0)
    r = torch.matmul(g, v)
    if read_norm == "softcount":
        r = r / g.sum(-1, keepdim=True).clamp_min(eps)
    return r


# =====================================================================
# Triton dense kernel (sigmoid-gated, causal, no softmax)
# =====================================================================
if _HAS_TRITON:

    @triton.jit
    def _resonance_dense_kernel(
        Q, K, V, Out,
        tau_ptr, bias_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        N, H,
        READ_NORM: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DH: tl.constexpr,
    ):
        pid_m = tl.program_id(0)      # query block
        pid_bh = tl.program_id(1)     # batch*head
        b = pid_bh // H
        h = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, DH)

        q_ptrs = (Q + b * stride_qb + h * stride_qh
                  + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd)
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)

        tau = tl.load(tau_ptr + h)
        bias = tl.load(bias_ptr + h)

        acc = tl.zeros([BLOCK_M, DH], dtype=tl.float32)
        denom = tl.zeros([BLOCK_M], dtype=tl.float32)

        end_n = (pid_m + 1) * BLOCK_M  # causal: only keys up to this query block
        for start_n in range(0, end_n, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = (K + b * stride_kb + h * stride_kh
                      + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
            k = tl.load(k_ptrs, mask=offs_n[:, None] < N, other=0.0)
            v_ptrs = (V + b * stride_vb + h * stride_vh
                      + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)

            s = tl.dot(q, tl.trans(k))                      # (BLOCK_M, BLOCK_N)
            g = tl.sigmoid(tau * s + bias)
            # causal mask (strict lower-triangular by absolute index)
            causal = offs_m[:, None] >= offs_n[None, :]
            valid = offs_n[None, :] < N
            g = tl.where(causal & valid, g, 0.0)
            acc += tl.dot(g.to(v.dtype), v).to(tl.float32)
            if READ_NORM:
                denom += tl.sum(g, axis=1)

        if READ_NORM:
            acc = acc / tl.maximum(denom[:, None], 1e-4)

        o_ptrs = (Out + b * stride_ob + h * stride_oh
                  + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od)
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)


def resonance_dense(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    tau: torch.Tensor, bias: torch.Tensor,
    read_norm: Optional[str] = None,
    block_m: int = 64, block_n: int = 64,
) -> torch.Tensor:
    """Fused causal sigmoid-gated read. Triton on CUDA, PyTorch fallback else."""
    if not (_HAS_TRITON and q.is_cuda):
        return _dense_ref(q, k, v, tau, bias, read_norm)

    B, H, N, dh = q.shape
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    out = torch.empty_like(q)
    grid = (triton.cdiv(N, block_m), B * H)
    _resonance_dense_kernel[grid](
        q, k, v, out, tau.contiguous(), bias.contiguous(),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N, H,
        READ_NORM=(read_norm == "softcount"),
        BLOCK_M=block_m, BLOCK_N=block_n, DH=dh,
    )
    return out


# =====================================================================
# Block-sparse fused read (Triton on CUDA, PyTorch fallback else)
# =====================================================================
if _HAS_TRITON:

    @triton.jit
    def _resonance_blocksparse_kernel(
        Q, K, V, Out, Idx,
        tau_ptr, bias_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        stride_ib, stride_ih, stride_iq, stride_ik,
        N, H, TOPK,
        READ_NORM: tl.constexpr,
        BLOCK: tl.constexpr, DH: tl.constexpr,
    ):
        pid_q = tl.program_id(0)      # query block index
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        offs_m = pid_q * BLOCK + tl.arange(0, BLOCK)
        offs_d = tl.arange(0, DH)
        q_ptrs = (Q + b * stride_qb + h * stride_qh
                  + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd)
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)

        tau = tl.load(tau_ptr + h)
        bias = tl.load(bias_ptr + h)
        acc = tl.zeros([BLOCK, DH], dtype=tl.float32)
        denom = tl.zeros([BLOCK], dtype=tl.float32)

        for slot in range(0, TOPK):
            kb_idx = tl.load(Idx + b * stride_ib + h * stride_ih
                             + pid_q * stride_iq + slot * stride_ik)
            if kb_idx >= 0:
                offs_n = kb_idx * BLOCK + tl.arange(0, BLOCK)
                k_ptrs = (K + b * stride_kb + h * stride_kh
                          + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
                k = tl.load(k_ptrs, mask=offs_n[:, None] < N, other=0.0)
                v_ptrs = (V + b * stride_vb + h * stride_vh
                          + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
                v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)
                s = tl.dot(q, tl.trans(k))
                g = tl.sigmoid(tau * s + bias)
                # causal only matters when the selected block is the diagonal one
                is_diag = kb_idx == pid_q
                causal = offs_m[:, None] >= offs_n[None, :]
                keep = (offs_n[None, :] < N) & ((not is_diag) | causal)
                g = tl.where(keep, g, 0.0)
                acc += tl.dot(g.to(v.dtype), v).to(tl.float32)
                if READ_NORM:
                    denom += tl.sum(g, axis=1)

        if READ_NORM:
            acc = acc / tl.maximum(denom[:, None], 1e-4)
        o_ptrs = (Out + b * stride_ob + h * stride_oh
                  + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od)
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)


def _blocksparse_ref(q, k, v, tau, bias, idx, block, read_norm, eps=1e-4):
    """Reference block-sparse read using a precomputed routing index ``idx``.

    idx: (B,H,nb,topk) long, -1 for empty slots.
    """
    Bsz, H, N, dh = q.shape
    nb = idx.shape[2]
    topk = idx.shape[3]
    pad = nb * block - N
    if pad:
        q = F.pad(q, (0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, pad)); v = F.pad(v, (0, 0, 0, pad))
    qb = q.view(Bsz, H, nb, block, dh)
    kb = k.view(Bsz, H, nb, block, dh)
    vb = v.view(Bsz, H, nb, block, dh)
    rows = torch.arange(block, device=q.device).view(1, 1, block, 1)
    cols = torch.arange(block, device=q.device).view(1, 1, 1, block)
    causal_diag = cols > rows
    out = torch.zeros(Bsz, H, nb, block, dh, device=q.device, dtype=q.dtype)
    for iq in range(nb):
        qcur = qb[:, :, iq]
        acc = torch.zeros(Bsz, H, block, dh, device=q.device, dtype=q.dtype)
        dacc = torch.zeros(Bsz, H, block, 1, device=q.device, dtype=q.dtype)
        for slot in range(topk):
            sel = idx[:, :, iq, slot]  # (B,H)
            valid = sel >= 0
            sel_clamped = sel.clamp_min(0)
            gi = sel_clamped.view(Bsz, H, 1, 1, 1).expand(Bsz, H, 1, block, dh)
            ksel = torch.gather(kb, 2, gi).squeeze(2)
            vsel = torch.gather(vb, 2, gi).squeeze(2)
            s = torch.einsum("bhid,bhjd->bhij", qcur, ksel)
            g = torch.sigmoid(tau.view(1, H, 1, 1) * s + bias.view(1, H, 1, 1))
            same = (sel == iq).view(Bsz, H, 1, 1)
            g = torch.where(same & causal_diag, torch.zeros_like(g), g)
            g = g * valid.view(Bsz, H, 1, 1)
            acc = acc + torch.einsum("bhij,bhjd->bhid", g, vsel)
            if read_norm == "softcount":
                dacc = dacc + g.sum(-1, keepdim=True)
        if read_norm == "softcount":
            acc = acc / dacc.clamp_min(eps)
        out[:, :, iq] = acc
    return out.view(Bsz, H, nb * block, dh)[:, :, :N]


def resonance_blocksparse(
    q, k, v, tau, bias, idx, block: int,
    read_norm: Optional[str] = None,
) -> torch.Tensor:
    """Fused block-sparse read given routing ``idx`` (B,H,nb,topk)."""
    if not (_HAS_TRITON and q.is_cuda):
        return _blocksparse_ref(q, k, v, tau, bias, idx, block, read_norm)

    Bsz, H, N, dh = q.shape
    nb = idx.shape[2]
    topk = idx.shape[3]
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous(); idx = idx.contiguous().int()
    out = torch.empty_like(q)
    grid = (nb, Bsz * H)
    _resonance_blocksparse_kernel[grid](
        q, k, v, out, idx, tau.contiguous(), bias.contiguous(),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        idx.stride(0), idx.stride(1), idx.stride(2), idx.stride(3),
        N, H, topk,
        READ_NORM=(read_norm == "softcount"),
        BLOCK=block, DH=dh,
    )
    return out


def has_triton() -> bool:
    return _HAS_TRITON
