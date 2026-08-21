"""ENGRAMA V5 — kernels especializados para el Recall Tap.

Kernel principal: **lectura causal argmax+gather fusionada** (inferencia).
El camino estandar materializa scores por trozos ``(chunk, N)`` (matmul +
argmax + gather = 3 pasadas sobre datos). El kernel fusionado mantiene SOLO
el ``(max, argmax)`` en registros: cero escritura de scores, memoria O(filas)
en vez de O(filas·N), una unica lectura de K.

Empate -> ocurrencia MAS RECIENTE (misma semantica que
:meth:`engrama.v5.recall.RecallTap.forward_parallel`): dentro de cada bloque
se toma el indice MAYOR entre los maximos, y entre bloques se actualiza con
``>=`` recorriendo j en orden ascendente.

El kernel requiere GPU (Triton). Sin GPU, o si Triton no esta disponible, el
despachador cae automaticamente a la referencia vectorizada en torch
(identica semantica, validada por :func:`validate_kernel`).

Uso en GPU::

    from engrama.v5.kernels import validate_kernel, causal_argmax_read
    print(validate_kernel())          # paridad kernel vs referencia
    reads, j_star = causal_argmax_read(q, k, v_shifted, gap=1)

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""
from __future__ import annotations

from typing import Tuple

import torch

try:  # pragma: no cover - depende del entorno
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _HAS_TRITON = False


# ----------------------------------------------------------------------
# Referencia exacta (torch, CPU/GPU, misma semantica que recall.forward_parallel)
# ----------------------------------------------------------------------
@torch.no_grad()
def causal_argmax_read_torch(
    q: torch.Tensor,          # (R, d)  consultas
    k: torch.Tensor,          # (N, d)  codigos de la traza
    v: torch.Tensor,          # (N, d)  matriz de valores YA desplazada
    gap: int = 1,
    row_index: torch.Tensor = None,   # (R,) posicion absoluta de cada consulta
    chunk: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Referencia: lectura dura causal con empate a la mas reciente."""
    n = k.size(0)
    device = q.device
    rows = row_index if row_index is not None else torch.arange(q.size(0), device=device)
    out = torch.zeros(q.size(0), v.size(-1), device=device, dtype=v.dtype)
    j_out = torch.full((q.size(0),), -1, dtype=torch.long, device=device)
    kf = k.float()
    for s in range(0, q.size(0), chunk):
        e = min(q.size(0), s + chunk)
        sc = q[s:e].float() @ kf.T                        # (c, N)
        limit = rows[s:e].view(-1, 1) - gap
        colj = torch.arange(n, device=device).view(1, -1)
        valid = colj <= limit
        sc = torch.where(valid, sc, torch.full_like(sc, -float("inf")))
        row_ok = valid.any(dim=-1)
        j_star = (n - 1) - sc.flip(-1).argmax(-1)          # empate -> mas reciente
        j_star = torch.where(row_ok, j_star, torch.full_like(j_star, -1))
        take = j_star.clamp(min=0)
        val = v[take]
        out[s:e] = torch.where(row_ok.unsqueeze(-1), val, torch.zeros_like(val)).to(v.dtype)
        j_out[s:e] = j_star
    return out, j_out


# ----------------------------------------------------------------------
# Kernel Triton (GPU): fusion score+argmax+gather, memoria O(R)
# ----------------------------------------------------------------------
if _HAS_TRITON:  # pragma: no cover - solo compilable con GPU

    @triton.jit
    def _causal_argmax_read_kernel(
        Q, K, V, OUT, JSTAR,
        rows_ptr,
        stride_qm, stride_qd, stride_kn, stride_kd,
        stride_vn, stride_vd, stride_om, stride_od, stride_r,
        DK, GAP: tl.constexpr,
        BLOCK_D: tl.constexpr, BLOCK_J: tl.constexpr,
    ):
        pid = tl.program_id(0)
        m = tl.load(rows_ptr + pid * stride_r).to(tl.int32)   # posicion absoluta
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < DK
        q = tl.load(Q + pid * stride_qm + offs_d * stride_qd,
                    mask=mask_d, other=0.0).to(tl.float32)
        limit = m - GAP
        best = -1e30
        best_j = -1
        for j0 in range(0, 0 + limit + 1, BLOCK_J):
            offs_j = j0 + tl.arange(0, BLOCK_J)
            mask_j = offs_j <= limit
            kj = tl.load(K + offs_j[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                         mask=mask_j[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            s = tl.sum(kj * q[None, :], axis=1)               # (BJ,)
            s = tl.where(mask_j, s, -1e30)
            blk_max = tl.max(s, axis=0)
            # entre los maximos del bloque, el indice MAYOR (empate -> reciente)
            eq = (s == blk_max) & mask_j
            blk_j = tl.max(tl.where(eq, offs_j, -1), axis=0)
            take = blk_max >= best                            # bloques ascendentes
            best = tl.where(take, blk_max, best)
            best_j = tl.where(take, blk_j, best_j)
        jv = tl.maximum(best_j + 1, 0)
        vrow = tl.load(V + jv * stride_vn + offs_d * stride_vd,
                       mask=mask_d, other=0.0)
        valid = best_j >= 0
        out = tl.where(valid, vrow, 0.0)
        tl.store(OUT + pid * stride_om + offs_d * stride_od, out, mask=mask_d)
        tl.store(JSTAR + pid, best_j)

    def causal_argmax_read_triton(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        gap: int = 1, row_index: torch.Tensor = None,
        block_j: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lectura dura fusionada (GPU). Semantica identica a la referencia."""
        assert q.is_cuda, "el kernel Triton requiere CUDA"
        r, dk = q.shape
        n = k.size(0)
        rows = (row_index if row_index is not None
                else torch.arange(r, device=q.device))
        rows = rows.to(torch.int32).contiguous()
        out = torch.zeros(r, v.size(-1), device=q.device, dtype=v.dtype)
        jstar = torch.full((r,), -1, dtype=torch.int32, device=q.device)
        block_d = triton.next_power_of_2(max(dk, 16))
        grid = (r,)
        _causal_argmax_read_kernel[grid](
            q, k, v, out, jstar, rows,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1),
            v.stride(0), v.stride(1), out.stride(0), out.stride(1), rows.stride(0),
            dk, gap=gap, BLOCK_D=block_d, BLOCK_J=block_j,
            num_warps=4,
        )
        return out, jstar.long()


def causal_argmax_read(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    gap: int = 1, row_index: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Despachador automatico: Triton en GPU si esta disponible; si no, la
    referencia exacta en torch."""
    if _HAS_TRITON and q.is_cuda:
        try:
            return causal_argmax_read_triton(q, k, v, gap=gap, row_index=row_index)
        except Exception:  # pragma: no cover - degradacion segura
            pass
    return causal_argmax_read_torch(q, k, v, gap=gap, row_index=row_index)


def validate_kernel(trials: int = 8, n: int = 4096, d: int = 64, seed: int = 0,
                    verbose: bool = True) -> dict:
    """Compara kernel Triton vs referencia torch en datos aleatorios con
    repeticiones (empates garantizados). Ejecutar EN GPU antes de activar."""
    if not _HAS_TRITON:
        return {"ok": False, "reason": "triton no disponible (sin GPU)"}
    g = torch.Generator(device="cpu").manual_seed(seed)
    results = {"ok": True, "trials": trials, "max_diff": 0.0, "j_mismatch": 0}
    for t in range(trials):
        torch.manual_seed(seed + t)
        q = torch.randn(n, d, device="cuda", dtype=torch.float32)
        k = torch.randn(n, d, device="cuda", dtype=torch.float32)
        # copiar bloques para forzar empates exactos
        dup = torch.randint(0, n, (n // 8,), device="cuda", generator=None)
        k[dup] = k[torch.randint(0, n, (n // 8,), device="cuda")]
        v = torch.randn(n, d, device="cuda", dtype=torch.float32)
        gap = 1 + (t % 3)
        ref_out, ref_j = causal_argmax_read_torch(q, k, v, gap=gap)
        ker_out, ker_j = causal_argmax_read_triton(q, k, v, gap=gap)
        diff = (ref_out - ker_out).abs().max().item()
        mism = int((ref_j != ker_j).sum().item())
        results["max_diff"] = max(results["max_diff"], diff)
        results["j_mismatch"] += mism
        if diff > 1e-4 or mism > 0:
            results["ok"] = False
    if verbose:
        print(results)
    return results
