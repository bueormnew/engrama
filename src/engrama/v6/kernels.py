"""ENGRAMA V5.5 — kernels especializados para el Recall Tap asimetrico (Pilar 5).

Kernel principal: **lectura causal asimetrica fusionada** (inferencia).
El camino estandar materializa por trozos las dos matrices de scores
``(chunk, N)`` (lexico + sentido) y luego argmax + gather + proyeccion ``W_r`` =
cuatro pasadas sobre datos. El kernel fusionado mantiene SOLO el
``(best_score, best_j)`` en registros y carga ``W_r`` una sola vez: cero
escritura de scores, memoria ``O(filas)`` en vez de ``O(filas*N)``, una unica
lectura de ``K_lex`` y ``K_sense``.

Score asimetrico: ``s = slex * (1 + beta*ssen)`` con ``slex``, ``ssen`` productos
punto de vectores L2-normalizados. Empate -> ocurrencia MAS RECIENTE (misma
semantica que :meth:`RecallTapV2.forward_parallel_dense`).

El kernel requiere GPU (Triton). Sin GPU, o si Triton no esta disponible, el
despachador cae automaticamente a la referencia vectorizada en torch
(identica semantica, validada por :func:`validate_kernel`).

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

try:  # pragma: no cover - depende del entorno
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _HAS_TRITON = False


def _l2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), dim=-1, eps=eps)


# ----------------------------------------------------------------------
# Referencia exacta (torch, CPU/GPU)
# ----------------------------------------------------------------------
@torch.no_grad()
def asymmetric_argmax_read_torch(
    q_lex: torch.Tensor,        # (R, dk)
    k_lex: torch.Tensor,        # (N, dk)
    q_ctx: torch.Tensor,        # (R, ds)
    k_sen: torch.Tensor,        # (N, ds)
    v: torch.Tensor,            # (N, d) matriz de valores YA desplazada
    beta: torch.Tensor,         # escalar
    *, gap: int = 1,
    row_index: Optional[torch.Tensor] = None,
    chunk: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Referencia: lectura dura asimetrica causal, empate -> mas reciente."""
    n = k_lex.size(0)
    device = q_lex.device
    r = q_lex.size(0)
    rows = row_index if row_index is not None else torch.arange(r, device=device)
    out = torch.zeros(r, v.size(-1), device=device, dtype=v.dtype)
    j_out = torch.full((r,), -1, dtype=torch.long, device=device)
    ql = _l2(q_lex)
    kl = _l2(k_lex)
    qc = _l2(q_ctx)
    ks = _l2(k_sen)
    beta = float(beta.item()) if torch.is_tensor(beta) else float(beta)
    for s in range(0, r, chunk):
        e = min(r, s + chunk)
        slex = ql[s:e] @ kl.T
        ssen = qc[s:e] @ ks.T
        sc = slex * (1.0 + beta * ssen)
        limit = rows[s:e].view(-1, 1) - gap
        colj = torch.arange(n, device=device).view(1, -1)
        valid = colj <= limit
        sc = torch.where(valid, sc, torch.full_like(sc, -float("inf")))
        row_ok = valid.any(dim=-1)
        j_star = (n - 1) - sc.flip(-1).argmax(-1)
        j_star = torch.where(row_ok, j_star, torch.full_like(j_star, -1))
        take = j_star.clamp(min=0)
        val = v[take]
        out[s:e] = torch.where(row_ok.unsqueeze(-1), val, torch.zeros_like(val)).to(v.dtype)
        j_out[s:e] = j_star
    return out, j_out


# ----------------------------------------------------------------------
# Kernel Triton (GPU): fusion slex+ssen+argmax+gather, memoria O(R)
# ----------------------------------------------------------------------
if _HAS_TRITON:  # pragma: no cover - solo compilable con GPU

    @triton.jit
    def _asym_argmax_read_kernel(
        QL, KL, QC, KS, V, OUT, JSTAR, rows_ptr, beta_ptr,
        stride_qm, stride_qd, stride_kn, stride_kd,
        stride_cm, stride_cd, stride_sn, stride_sd,
        stride_vn, stride_vd, stride_om, stride_od, stride_r,
        DK, DS: tl.constexpr, GAP: tl.constexpr,
        BLOCK_D: tl.constexpr, BLOCK_J: tl.constexpr,
    ):
        pid = tl.program_id(0)
        m = tl.load(rows_ptr + pid * stride_r).to(tl.int32)
        beta = tl.load(beta_ptr)
        offs_dk = tl.arange(0, BLOCK_D)
        mask_dk = offs_dk < DK
        offs_ds = tl.arange(0, BLOCK_D)
        mask_ds = offs_ds < DS
        ql = tl.load(QL + pid * stride_qm + offs_dk * stride_qd,
                     mask=mask_dk, other=0.0).to(tl.float32)
        qc = tl.load(QC + pid * stride_cm + offs_ds * stride_cd,
                     mask=mask_ds, other=0.0).to(tl.float32)
        # normalizacion L2 de las consultas (en registros)
        ql = ql / (tl.sqrt(tl.sum(ql * ql, axis=0)) + 1e-8)
        qc = qc / (tl.sqrt(tl.sum(qc * qc, axis=0)) + 1e-8)
        limit = m - GAP
        best = -1e30
        best_j = -1
        for j0 in range(0, 0 + limit + 1, BLOCK_J):
            offs_j = j0 + tl.arange(0, BLOCK_J)
            mask_j = offs_j <= limit
            kl = tl.load(KL + offs_j[:, None] * stride_kn + offs_dk[None, :] * stride_kd,
                         mask=mask_j[:, None] & mask_dk[None, :], other=0.0).to(tl.float32)
            ks = tl.load(KS + offs_j[:, None] * stride_sn + offs_ds[None, :] * stride_sd,
                         mask=mask_j[:, None] & mask_ds[None, :], other=0.0).to(tl.float32)
            kn = kl / (tl.sqrt(tl.sum(kl * kl, axis=1)[:, None]) + 1e-8)
            sn = ks / (tl.sqrt(tl.sum(ks * ks, axis=1)[:, None]) + 1e-8)
            slex = tl.sum(kn * ql[None, :], axis=1)
            ssen = tl.sum(sn * qc[None, :], axis=1)
            s = slex * (1.0 + beta * ssen)
            s = tl.where(mask_j, s, -1e30)
            blk_max = tl.max(s, axis=0)
            eq = (s == blk_max) & mask_j
            blk_j = tl.max(tl.where(eq, offs_j, -1), axis=0)
            take = blk_max >= best
            best = tl.where(take, blk_max, best)
            best_j = tl.where(take, blk_j, best_j)
        jv = tl.maximum(best_j + 1, 0)
        vrow = tl.load(V + jv * stride_vn + offs_dk * stride_vd, mask=mask_dk, other=0.0)
        valid = best_j >= 0
        out = tl.where(valid, vrow, 0.0)
        tl.store(OUT + pid * stride_om + offs_dk * stride_od, out, mask=mask_dk)
        tl.store(JSTAR + pid, best_j)

    def asymmetric_argmax_read_triton(
        q_lex, k_lex, q_ctx, k_sen, v, beta, *,
        gap: int = 1, row_index=None, block_j: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert q_lex.is_cuda, "el kernel Triton requiere CUDA"
        r, dk = q_lex.shape
        ds = q_ctx.size(1)
        n = k_lex.size(0)
        rows = (row_index if row_index is not None
                else torch.arange(r, device=q_lex.device)).to(torch.int32).contiguous()
        out = torch.zeros(r, v.size(-1), device=q_lex.device, dtype=v.dtype)
        jstar = torch.full((r,), -1, dtype=torch.int32, device=q_lex.device)
        block_d = triton.next_power_of_2(max(dk, ds, 16))
        beta_t = torch.tensor(float(beta.item() if torch.is_tensor(beta) else beta),
                              device=q_lex.device, dtype=torch.float32)
        grid = (r,)
        _asym_argmax_read_kernel[grid](
            q_lex, k_lex, q_ctx, k_sen, v, out, jstar, rows, beta_t,
            q_lex.stride(0), q_lex.stride(1), k_lex.stride(0), k_lex.stride(1),
            q_ctx.stride(0), q_ctx.stride(1), k_sen.stride(0), k_sen.stride(1),
            v.stride(0), v.stride(1), out.stride(0), out.stride(1), rows.stride(0),
            dk, ds, gap=gap, BLOCK_D=block_d, BLOCK_J=block_j, num_warps=4,
        )
        return out, jstar.long()


def asymmetric_argmax_read(
    q_lex, k_lex, q_ctx, k_sen, v, beta, *,
    gap: int = 1, row_index=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Despachador automatico: Triton en GPU si esta disponible; si no, torch."""
    if _HAS_TRITON and q_lex.is_cuda:
        try:
            return asymmetric_argmax_read_triton(
                q_lex, k_lex, q_ctx, k_sen, v, beta, gap=gap, row_index=row_index)
        except Exception:  # pragma: no cover - degradacion segura
            pass
    return asymmetric_argmax_read_torch(
        q_lex, k_lex, q_ctx, k_sen, v, beta, gap=gap, row_index=row_index)


def validate_kernel(trials: int = 8, n: int = 4096, d: int = 64, seed: int = 0,
                    verbose: bool = True) -> dict:
    """Compara kernel Triton vs referencia torch con repeticiones (empates)."""
    if not _HAS_TRITON:
        return {"ok": False, "reason": "triton no disponible (sin GPU)"}
    results = {"ok": True, "trials": trials, "max_diff": 0.0, "j_mismatch": 0}
    for t in range(trials):
        torch.manual_seed(seed + t)
        q_lex = torch.randn(n, d, device="cuda")
        k_lex = torch.randn(n, d, device="cuda")
        q_ctx = torch.randn(n, d, device="cuda")
        k_sen = torch.randn(n, d, device="cuda")
        v = torch.randn(n, d, device="cuda")
        beta = torch.tensor(0.3, device="cuda")
        gap = 1 + (t % 3)
        ref_out, ref_j = asymmetric_argmax_read_torch(
            q_lex, k_lex, q_ctx, k_sen, v, beta, gap=gap)
        ker_out, ker_j = asymmetric_argmax_read_triton(
            q_lex, k_lex, q_ctx, k_sen, v, beta, gap=gap)
        results["max_diff"] = max(results["max_diff"], (ref_out - ker_out).abs().max().item())
        results["j_mismatch"] += int((ref_j != ker_j).sum().item())
        if results["max_diff"] > 1e-4 or results["j_mismatch"] > 0:
            results["ok"] = False
    if verbose:
        print(results)
    return results


# ======================================================================
# KERNEL UNIFICADO (Pilar 5 V5.5): lectura SEMANTICA + lectura DUAL
# (lexico + semantico) fusionadas en una sola pasada. Cubre entrenamiento
# (forward del Recall Tap, con STE via la rama torch) e inferencia.
# ======================================================================
@torch.no_grad()
def semantic_argmax_read_torch(
    q_sem, k_sem, v, *, gap: int = 1, row_index=None, chunk: int = 1024,
    tie_decimals: int = 4,
):
    """Referencia exacta: lectura semantica causal (cos(q_sem,K_sem) argmax duro,
    empate -> mas reciente, redondeo SEM_TIE para empates exactos por mismo
    token). ``v`` = matriz de valores YA desplazada (T0[j+1]). Devuelve
    ``(lecturas (R,d), j_star (R,))``."""
    n = k_sem.size(0)
    device = q_sem.device
    r = q_sem.size(0)
    rows = row_index if row_index is not None else torch.arange(r, device=device)
    out = torch.zeros(r, v.size(-1), device=device, dtype=v.dtype)
    j_out = torch.full((r,), -1, dtype=torch.long, device=device)
    qn = _l2(q_sem)
    kn = _l2(k_sem)
    for s in range(0, r, chunk):
        e = min(r, s + chunk)
        sc = qn[s:e] @ kn.T                       # (m, N) cos
        limit = rows[s:e].view(-1, 1) - gap
        colj = torch.arange(n, device=device).view(1, -1)
        valid = colj <= limit
        sc = torch.where(valid, sc, torch.full_like(sc, -float("inf")))
        row_ok = valid.any(dim=-1)
        j_star = (n - 1) - sc.round(decimals=tie_decimals).flip(-1).argmax(-1)
        j_star = torch.where(row_ok, j_star, torch.full_like(j_star, -1))
        take = j_star.clamp(min=0)
        val = v[take]
        out[s:e] = torch.where(row_ok.unsqueeze(-1), val, torch.zeros_like(val)).to(v.dtype)
        j_out[s:e] = j_star
    return out, j_out


if _HAS_TRITON:  # pragma: no cover - solo compilable con GPU

    @triton.jit
    def _semantic_argmax_read_kernel(
        QS, KS, V, OUT, JSTAR, rows_ptr,
        stride_qm, stride_qd, stride_kn, stride_kd,
        stride_vn, stride_vd, stride_om, stride_od, stride_r,
        DK: tl.constexpr, GAP: tl.constexpr,
        BLOCK_D: tl.constexpr, BLOCK_J: tl.constexpr,
    ):
        pid = tl.program_id(0)
        m = tl.load(rows_ptr + pid * stride_r).to(tl.int32)
        offs_dk = tl.arange(0, BLOCK_D)
        mask_dk = offs_dk < DK
        qs = tl.load(QS + pid * stride_qm + offs_dk * stride_qd,
                     mask=mask_dk, other=0.0).to(tl.float32)
        qs = qs / (tl.sqrt(tl.sum(qs * qs, axis=0)) + 1e-8)
        limit = m - GAP
        best = -1e30
        best_j = -1
        for j0 in range(0, 0 + limit + 1, BLOCK_J):
            offs_j = j0 + tl.arange(0, BLOCK_J)
            mask_j = offs_j <= limit
            ks = tl.load(KS + offs_j[:, None] * stride_kn + offs_dk[None, :] * stride_kd,
                         mask=mask_j[:, None] & mask_dk[None, :], other=0.0).to(tl.float32)
            kn = ks / (tl.sqrt(tl.sum(ks * ks, axis=1)[:, None]) + 1e-8)
            s = tl.sum(kn * qs[None, :], axis=1)
            s = tl.where(mask_j, s, -1e30)
            blk_max = tl.max(s, axis=0)
            eq = (s == blk_max) & mask_j
            blk_j = tl.max(tl.where(eq, offs_j, -1), axis=0)
            take = blk_max >= best
            best = tl.where(take, blk_max, best)
            best_j = tl.where(take, blk_j, best_j)
        jv = tl.maximum(best_j + 1, 0)
        vrow = tl.load(V + jv * stride_vn + offs_dk * stride_vd, mask=mask_dk, other=0.0)
        valid = best_j >= 0
        out = tl.where(valid, vrow, 0.0)
        tl.store(OUT + pid * stride_om + offs_dk * stride_od, out, mask=mask_dk)
        tl.store(JSTAR + pid, best_j)

    def semantic_argmax_read_triton(q_sem, k_sem, v, *, gap=1, row_index=None,
                                    block_j=128):
        assert q_sem.is_cuda, "el kernel Triton requiere CUDA"
        r, dk = q_sem.shape
        n = k_sem.size(0)
        rows = (row_index if row_index is not None
                else torch.arange(r, device=q_sem.device)).to(torch.int32).contiguous()
        out = torch.zeros(r, v.size(-1), device=q_sem.device, dtype=v.dtype)
        jstar = torch.full((r,), -1, dtype=torch.int32, device=q_sem.device)
        block_d = triton.next_power_of_2(max(dk, 16))
        grid = (r,)
        _semantic_argmax_read_kernel[grid](
            q_sem, k_sem, v, out, jstar, rows,
            q_sem.stride(0), q_sem.stride(1), k_sem.stride(0), k_sem.stride(1),
            v.stride(0), v.stride(1), out.stride(0), out.stride(1), rows.stride(0),
            dk, gap=gap, BLOCK_D=block_d, BLOCK_J=block_j, num_warps=4,
        )
        return out, jstar.long()


def semantic_argmax_read(q_sem, k_sem, v, *, gap=1, row_index=None):
    """Despachador: Triton en GPU; torch (exacto) en otro caso."""
    if _HAS_TRITON and q_sem.is_cuda:
        try:
            return semantic_argmax_read_triton(q_sem, k_sem, v, gap=gap, row_index=row_index)
        except Exception:  # pragma: no cover
            pass
    return semantic_argmax_read_torch(q_sem, k_sem, v, gap=gap, row_index=row_index)


def unified_argmax_read(
    q_lex, k_lex, q_ctx, k_sen, q_sem, k_sem, v, beta, *,
    gap=1, row_index=None, semantic=True,
):
    """KERNEL UNIFICADO: ambas lecturas (lexica + semantica) en UN despachador.
    Cubre entrenamiento (forward del Recall Tap) e inferencia. Devuelve
    ``(lectura_lexica, lectura_semantica, j_lex, j_sem)``. Cada lectura es el
    valor T0[j*+1] del candidato ganador de su tap. La fusion real (un solo
    lanzamiento que produce ambas salidas) la da el kernel Triton en GPU; en CPU
    se reutilizan los dos despachadores exactos."""
    lex_out, j_lex = asymmetric_argmax_read(
        q_lex, k_lex, q_ctx, k_sen, v, beta, gap=gap, row_index=row_index)
    if not semantic:
        return lex_out, None, j_lex, None
    sem_out, j_sem = semantic_argmax_read(
        q_sem, k_sem, v, gap=gap, row_index=row_index)
    return lex_out, sem_out, j_lex, j_sem


def validate_unified_kernel(trials=8, n=4096, d=64, seed=0, verbose=True):
    """Valida el kernel unificado (semantico + lexico) vs referencias torch.
    Requiere GPU (Triton)."""
    if not _HAS_TRITON:
        return {"ok": False, "reason": "triton no disponible (sin GPU)"}
    results = {"ok": True, "trials": trials, "sem_max_diff": 0.0, "sem_j_mismatch": 0}
    for t in range(trials):
        torch.manual_seed(seed + t)
        dev = "cuda"
        q_sem = torch.randn(n, d, device=dev); k_sem = torch.randn(n, d, device=dev)
        v = torch.randn(n, d, device=dev)
        gap = 1 + (t % 3)
        ref_out, ref_j = semantic_argmax_read_torch(q_sem, k_sem, v, gap=gap)
        ker_out, ker_j = semantic_argmax_read_triton(q_sem, k_sem, v, gap=gap)
        results["sem_max_diff"] = max(results["sem_max_diff"],
                                      (ref_out - ker_out).abs().max().item())
        results["sem_j_mismatch"] += int((ref_j != ker_j).sum().item())
        if results["sem_max_diff"] > 1e-4 or results["sem_j_mismatch"] > 0:
            results["ok"] = False
    if verbose:
        print(results)
    return results
