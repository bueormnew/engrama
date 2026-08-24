"""Prueba 3 — Recall lexico: identidad O(1) vs denso O(N) vs LSH O(N*C).

Construimos una secuencia sintetica donde el eje lexico es la identidad del
token (K_lex = vector unitario por token) y el eje de sentido desempata entre
apariciones del mismo token. Medimos:
  - Recall@1 del denso (debe ser ~1.0: el sentido apunta al objetivo)
  - Recall@1 del fast-path de identidad (ultima ocurrencia)
  - Recall@1 del LSH
  - latencia y su crecimiento (exponente empirico)
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fit_scaling, infer_exponent, save_json

from engrama.v55 import lsh as lsh_mod
from engrama.v55.recall import RecallTapV2, _l2, _lex_dominant_argmax


def make_seq(n: int, V: int, dk: int, ds: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, V, (n,), generator=g)
    # K_lex: vector ortogonal por token id (usamos dk dimensiones; reparto por mod)
    k_lex = torch.zeros(n, dk)
    for i in range(n):
        k_lex[i, int(tokens[i]) % dk] = 1.0
    q_lex = k_lex.clone()
    # K_sen aleatorio normalizado
    k_sen = torch.randn(n, ds, generator=g)
    k_sen = torch.nn.functional.normalize(k_sen, dim=-1)
    # objetivo: ultima ocurrencia previa del mismo token (lo que el fast-path de
    # identidad devuelve). Hacemos que q_ctx[i] sea EXACTAMENTE k_sen[target]
    # para que el denso acierte ese objetivo.
    last = {}
    target = torch.full((n,), -1, dtype=torch.long)
    q_ctx = torch.zeros(n, ds)
    for i in range(n):
        t = int(tokens[i])
        if t in last:
            j = last[t]
            target[i] = j
            q_ctx[i] = k_sen[j]
        last[t] = i
    return tokens, k_lex, q_lex, k_sen, q_ctx, target


def dense_jstar(k_lex, k_sen, q_lex, q_ctx, gap=1):
    n = k_lex.size(0)
    ql = _l2(q_lex); kl = _l2(k_lex)
    qc = _l2(q_ctx); ks = _l2(k_sen)
    slex = ql @ kl.T
    ssen = qc @ ks.T
    col = torch.arange(n).view(1, n)
    row = torch.arange(n).view(n, 1)
    valid = col <= (row - gap)
    return _lex_dominant_argmax(slex, ssen, valid)


def lsh_jstar(recall, k_lex, k_sen, q_lex, q_ctx, tokens,
              n_tables=2, n_bits=8, cap=64, n_neg=48, hamming=3, gap=1):
    """Extrae j* del LSH sustituyendo el value por el INDICE de posicion."""
    n = k_lex.size(0)
    index = lsh_mod.LSHIndexV2.build(
        k_lex, tokens, gap=gap, n_tables=n_tables, n_bits=n_bits,
        cap=cap, seed=7, hamming=hamming)
    cand, valid = index.candidates()
    if n_neg > 0:
        gen = torch.Generator(device="cpu").manual_seed(7 + 1234)
        lim = torch.arange(n) - gap
        r = torch.rand(n, n_neg, generator=gen)
        neg = (r * (lim.clamp(min=0) + 1).unsqueeze(1)).long()
        ok = (lim.unsqueeze(1) >= 0).expand(n, n_neg)
        cand = torch.cat([cand, neg], dim=1)
        valid = torch.cat([valid, ok], dim=1)
    c = cand.size(1)
    ck = k_lex.index_select(0, cand.clamp(min=0).reshape(-1)).view(n, c, -1)
    cs = k_sen.index_select(0, cand.clamp(min=0).reshape(-1)).view(n, c, -1)
    beta = float(recall.beta.detach())
    ql = _l2(q_lex); qc = _l2(q_ctx)
    out = torch.full((n,), -1, dtype=torch.long)
    chunk = 2048
    valid = valid.bool()
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        m = e - s
        qlm = _l2(ql[s:e])
        qcm = _l2(qc[s:e])
        ckl = _l2(ck[s:e].float())
        cks = _l2(cs[s:e].float())
        slex = (qlm.unsqueeze(1) * ckl).sum(-1)
        ssen = (qcm.unsqueeze(1) * cks).sum(-1)
        _ = slex * (1.0 + beta * ssen)
        ok = valid[s:e]
        j = _lex_dominant_argmax(slex, ssen, ok, positions=cand[s:e])
        j = torch.where(ok.any(dim=-1), j, torch.full_like(j, -1))
        out[s:e] = j
    return out


def identity_jstar(tokens, gap=1):
    n = tokens.numel()
    last = {}
    out = torch.full((n,), -1, dtype=torch.long)
    for i, t in enumerate(tokens.tolist()):
        if t in last:
            out[i] = last[t]
        last[t] = i
    return out


def recall_of(jstar, target):
    mask = target >= 0
    if mask.sum() == 0:
        return float("nan")
    return ((jstar[mask] == target[mask]).sum().item() / mask.sum().item())


def run():
    torch.set_num_threads(2)
    d, dk, ds, V = 256, 64, 64, 4096
    # denso es O(N^2) en la construccion de la matriz de scores; en CPU y 3.8GB
    # aguanta hasta ~8K-16K. LSH es O(N*C).
    Ns = [256, 512, 1024, 2048, 4096, 8192]
    Ns_lsh = Ns + [16384, 32768]
    recall_mod = RecallTapV2(d, dk, ds, semantic_enabled=False).eval()

    rows = []
    print(f"{'N':>7} {'ident_us':>9} {'dense_ms':>10} {'lsh_ms':>9} "
          f"{'R_dense':>8} {'R_ident':>8} {'R_lsh':>8}")
    for n in Ns:
        tokens, k_lex, q_lex, k_sen, q_ctx, target = make_seq(n, V, dk, ds, 42)

        t = time.perf_counter()
        for _ in range(5):
            jid = identity_jstar(tokens)
        t_id = (time.perf_counter() - t) / 5

        for _ in range(1):
            jd = dense_jstar(k_lex, k_sen, q_lex, q_ctx)
        best = 1e9
        reps = 3 if n <= 2048 else 1
        for _ in range(reps):
            t = time.perf_counter()
            jd = dense_jstar(k_lex, k_sen, q_lex, q_ctx)
            best = min(best, time.perf_counter() - t)
        t_dense = best

        for _ in range(1):
            jl = lsh_jstar(recall_mod, k_lex, k_sen, q_lex, q_ctx, tokens)
        best = 1e9
        for _ in range(reps):
            t = time.perf_counter()
            jl = lsh_jstar(recall_mod, k_lex, k_sen, q_lex, q_ctx, tokens)
            best = min(best, time.perf_counter() - t)
        t_lsh = best

        rows.append({
            "N": n, "identity_s": t_id, "dense_s": t_dense, "lsh_s": t_lsh,
            "recall_dense": recall_of(jd, target),
            "recall_identity": recall_of(jid, target),
            "recall_lsh": recall_of(jl, target),
            "thr_dense": n / t_dense, "thr_lsh": n / t_lsh,
        })
        print(f"{n:>7} {t_id*1e6:>9.1f} {t_dense*1000:>10.2f} {t_lsh*1000:>9.2f} "
              f"{rows[-1]['recall_dense']:>8.3f} "
              f"{rows[-1]['recall_identity']:>8.3f} "
              f"{rows[-1]['recall_lsh']:>8.3f}")
        del k_lex, q_lex, k_sen, q_ctx, jd, jl, jid; gc.collect()

    # LSH-only a N grande (el denso O(N^2) no cabe/tiempo)
    print("\n-- LSH a N grande --")
    for n in [16384, 32768]:
        tokens, k_lex, q_lex, k_sen, q_ctx, target = make_seq(n, V, dk, ds, 42)
        t = time.perf_counter()
        jl = lsh_jstar(recall_mod, k_lex, k_sen, q_lex, q_ctx, tokens)
        t_lsh = time.perf_counter() - t
        r = recall_of(jl, target)
        rows.append({"N": n, "lsh_s": t_lsh, "recall_lsh": r, "thr_lsh": n / t_lsh})
        print(f"{n:>7} {'--':>9} {'--':>10} {t_lsh*1000:>9.2f} "
              f"{'--':>8} {'--':>8} {r:>8.3f}")
        del k_lex, q_lex, k_sen, q_ctx, jl; gc.collect()

    dense_rows = [r for r in rows if "dense_s" in r]
    fit_dense = fit_scaling([r["N"] for r in dense_rows],
                            [r["dense_s"] for r in dense_rows])
    fit_lsh = fit_scaling([r["N"] for r in rows], [r["lsh_s"] for r in rows])
    p_dense = infer_exponent([r["N"] for r in dense_rows],
                             [r["dense_s"] for r in dense_rows])
    p_lsh = infer_exponent([r["N"] for r in rows], [r["lsh_s"] for r in rows])

    result = {
        "test": "3_lexical_recall", "d": d, "dk": dk, "ds": ds, "V": V,
        "device": "cpu", "rows": rows,
        "fit": {"dense": fit_dense, "lsh": fit_lsh},
        "exponent": {"dense": p_dense, "lsh": p_lsh},
    }
    path = save_json("3_lexical_recall.json", result)
    print(f"\nDense exponente ~{p_dense:.2f} (lin R2 {fit_dense['linear']['r2']:.4f})")
    print(f"LSH   exponente ~{p_lsh:.2f} (lin R2 {fit_lsh['linear']['r2']:.4f})")
    print("Guardado:", path)


if __name__ == "__main__":
    run()
