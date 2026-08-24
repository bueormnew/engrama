"""Prueba 4 — Semantic Tap (la prueba critica).

Sin entrenar. Construimos embeddings sinteticos con una estructura semantica
CONOCIDA:

  concepto A_i  -> vector base a_i
  alias    A_i  -> a_i + ruido (cercano al concepto)
  concepto B_i  -> vector base b_i
  alias    B_i  -> b_i + ruido

Se inyectan en la secuencia de modo que la consulta (alias) debe recuperar el
concepto correspondiente. La clave/consulta semantica se fija para que sea
EXACTAMENTE el embedding sintetico (sin entrenar la Linear): llamamos
directamente al nucleo de score/argmax de forward_semantic_dense / lsh.

Medimos accuracy (Recall@1) en una superficie N x d_sem:
  N = 1K,2K,4K,8K,16K,32K,64K,128K  (solo LSH a N grande; dense es O(N^2))
  d = 64,128,256,512,1024
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fit_scaling, hr_bytes, infer_exponent, save_json

from engrama.v55 import lsh as lsh_mod
from engrama.v55.recall import _l2


def build_semantic_seq(n: int, d: int, n_concepts: int, noise: float, seed: int):
    """Crea n posiciones. Cada concepto tiene un alias que lo sigue.

    Diseno:
      - se generan n_concepts vectores base aleatorios (ortogonales-ish)
      - la secuencia alterna concepto y alias: [C0, A0, C1, A1, ...]
      - el alias A_i = C_i + ruido pequeno
      - consulta en la posicion del alias debe recuperar la posicion del concepto
        (que esta 1 atras, causal valido)
    Devuelve (K, Q, target_pos, valid_query_mask).
    """
    g = torch.Generator().manual_seed(seed)
    K = torch.zeros(n, d)
    Q = torch.zeros(n, d)
    target = torch.full((n,), -1, dtype=torch.long)
    valid = torch.zeros(n, dtype=torch.bool)
    # bases: vectores aleatorios normalizados (casi ortogonales si d grande)
    bases = torch.randn(n_concepts, d, generator=g)
    bases = torch.nn.functional.normalize(bases, dim=-1)
    i = 0
    c = 0
    while i + 1 < n:
        base = bases[c % n_concepts]
        # concepto en i
        K[i] = base
        Q[i] = 0  # no consultamos el concepto
        # alias en i+1
        a = base + noise * torch.randn(d, generator=g)
        K[i + 1] = a
        Q[i + 1] = base  # la consulta es el concepto puro -> debe elegir K[i]=base
        target[i + 1] = i
        valid[i + 1] = True
        i += 2
        c += 1
    return K, Q, target, valid


def semantic_dense(K, Q, valid, gap=1, chunk=1024):
    """Replica forward_semantic_dense (argmax coseno causal). Devuelve j*."""
    n = K.size(0)
    Kn = _l2(K); Qn = _l2(Q)
    out = torch.full((n,), -1, dtype=torch.long)
    rows = valid.nonzero().flatten()
    for s in range(0, rows.numel(), chunk):
        idx = rows[s:s + chunk]
        m = idx.numel()
        scores = Qn[idx] @ Kn.T                      # (m,N)
        pos = idx.view(m, 1)
        col = torch.arange(n).view(1, n)
        mask = col <= (pos - gap)
        scores = torch.where(mask, scores, torch.full_like(scores, -1e30))
        # recencia en empates (igual que el codigo)
        j = (n - 1) - scores.round(decimals=4).flip(-1).argmax(dim=-1)
        out[idx] = j
    return out


def semantic_lsh(K, Q, valid, tokens, n_tables=2, n_bits=8, cap=64, gap=1):
    """LSH de planos compartidos (igual que forward_semantic_lsh)."""
    n, d = K.shape
    gen = torch.Generator(device="cpu").manual_seed(7)
    planes = (torch.randint(0, 2, (n_tables, d, n_bits), generator=gen,
                            dtype=torch.float32) * 2 - 1)
    weights = (1 << torch.arange(n_bits)).long()
    n_codes = 1 << min(n_bits, 16)
    Kn = _l2(K).float(); Qn = _l2(Q).float()
    k_bits = (Kn.unsqueeze(0) @ planes) > 0
    k_codes = (k_bits.long() @ weights).T % n_codes
    q_bits = (Qn.unsqueeze(0) @ planes) > 0
    q_codes = (q_bits.long() @ weights).T % n_codes
    from engrama.v55.lsh import _bucket_matrix, previous_same_occurrence
    cols = [previous_same_occurrence(tokens.long(), gap=gap).unsqueeze(1)]
    recent_k = 4
    off = torch.arange(gap, gap + recent_k).unsqueeze(0)
    cols.append(torch.arange(n).unsqueeze(1) - off)
    for t in range(n_tables):
        bucket = _bucket_matrix(k_codes[:, t], n, n_codes, cap)
        cols.append(bucket[q_codes[:, t]])
    cand = torch.cat(cols, dim=1)
    ar = torch.arange(n).unsqueeze(1)
    ok = (cand >= 0) & (cand <= ar - gap)
    c = cand.size(1)
    out = torch.full((n,), -1, dtype=torch.long)
    rows = valid.nonzero().flatten()
    chunk = 1024  # recolecta candidatos por trozo para no materializar (N,C,d)
    for s in range(0, rows.numel(), chunk):
        idx = rows[s:s + chunk]
        m = idx.numel()
        local_cand = cand[idx]                       # (m, C)
        ck = K.index_select(0, local_cand.clamp(min=0).reshape(-1)).view(m, c, -1)
        qm = _l2(Qn[idx])
        ckn = _l2(ck.float())
        scores = (qm.unsqueeze(1) * ckn).sum(-1)
        mok = ok[idx]
        scores = torch.where(mok, scores, torch.full_like(scores, -1e30))
        j = (c - 1) - scores.round(decimals=4).flip(-1).argmax(dim=-1)
        jj = local_cand.gather(1, j.view(m, 1)).squeeze(1)
        jj = torch.where(mok.any(dim=-1), jj, torch.full_like(jj, -1))
        out[idx] = jj
    return out


def acc(jstar, target, valid):
    return ((jstar[valid] == target[valid]).sum().item() / valid.sum().item())


def run():
    torch.set_num_threads(2)
    Ns_dense = [1024, 2048, 4096, 8192]
    Ns_lsh = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    ds_list = [64, 128, 256, 512, 1024]
    # para denso, limita N segun d (O(N^2*d) en CPU/RAM)
    dense_cap = {64: 8192, 128: 8192, 256: 4096, 512: 4096, 1024: 2048}
    noise = 0.1
    surface = []  # N, d, mode, acc, ms

    for d in ds_list:
        n_concepts = 256
        print(f"\n=== d_sem = {d} ===")
        print(f"{'N':>7} {'dense_acc':>10} {'dense_ms':>9} "
              f"{'lsh_acc':>8} {'lsh_ms':>9}")
        # dense hasta el cap por d (O(N^2 d))
        for n in Ns_dense:
            if n > dense_cap[d]:
                continue
            K, Q, tgt, valid = build_semantic_seq(n, d, n_concepts, noise, 42)
            tokens = torch.arange(n) % n_concepts  # tokens sinteticos p/ LSH
            # warmup + medir
            for _ in range(1):
                jd = semantic_dense(K, Q, valid)
            t = time.perf_counter()
            reps = 3 if n <= 2048 else 1
            for _ in range(reps):
                jd = semantic_dense(K, Q, valid)
            td = (time.perf_counter() - t) / reps
            ad = acc(jd, tgt, valid)
            surface.append({"N": n, "d": d, "mode": "dense",
                            "acc": ad, "ms": td * 1000})
            del jd; gc.collect()
            # lsh en el mismo N
            for _ in range(1):
                jl = semantic_lsh(K, Q, valid, tokens)
            t = time.perf_counter()
            for _ in range(reps):
                jl = semantic_lsh(K, Q, valid, tokens)
            tl = (time.perf_counter() - t) / reps
            al = acc(jl, tgt, valid)
            surface.append({"N": n, "d": d, "mode": "lsh",
                            "acc": al, "ms": tl * 1000})
            print(f"{n:>7} {ad:>10.3f} {td*1000:>9.1f} {al:>8.3f} {tl*1000:>9.1f}")
            del K, Q, tgt, valid, jl; gc.collect()
        # LSH solo a N grande (en CPU con 3.8GB, d grande no cabe a 128K)
        max_big = 131072 if d <= 128 else (65536 if d <= 512 else 32768)
        for n in [16384, 32768, 65536, 131072]:
            if n > max_big:
                continue
            try:
                K, Q, tgt, valid = build_semantic_seq(n, d, n_concepts, noise, 42)
                tokens = torch.arange(n) % n_concepts
                t = time.perf_counter()
                jl = semantic_lsh(K, Q, valid, tokens)
                tl = time.perf_counter() - t
                al = acc(jl, tgt, valid)
                surface.append({"N": n, "d": d, "mode": "lsh",
                                "acc": al, "ms": tl * 1000})
                print(f"{n:>7} {'--':>10} {'--':>9} {al:>8.3f} {tl*1000:>9.1f}")
                del K, Q, tgt, valid, jl; gc.collect()
            except (RuntimeError, MemoryError) as e:
                print(f"{n:>7}  OOM/error: {str(e)[:60]}")
                gc.collect()
                break
        # guardado incremental
        save_json("4_semantic_tap_partial.json", {"surface": surface})

    # ajustes de tiempo por modo y dimension (fija d=256)
    timing = {"dense": {}, "lsh": {}}
    for d in ds_list:
        for mode in ("dense", "lsh"):
            pts = [(p["N"], p["ms"]) for p in surface
                   if p["d"] == d and p["mode"] == mode]
            if len(pts) >= 3:
                Ns = [p[0] for p in pts]; T = [p[1] for p in pts]
                fit = fit_scaling(Ns, T)
                p = infer_exponent(Ns, T)
                timing[mode][d] = {"fit": fit, "exponent": p}

    result = {
        "test": "4_semantic_tap",
        "noise": noise, "n_concepts": n_concepts,
        "surface": surface, "timing": timing,
    }
    path = save_json("4_semantic_tap.json", result)
    print("\nGuardado:", path)
    # resumen
    print("\nResumen accuracy d=256:")
    for p in surface:
        if p["d"] == 256:
            print(f"  {p['mode']:5s} N={p['N']:>7}  acc={p['acc']:.3f}  {p['ms']:.1f} ms")
    return result


if __name__ == "__main__":
    run()
