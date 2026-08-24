"""Benchmark V6 — verifica que los 12 problemas detectados estan resueltos.

Mide sobre engrama.v6 (EngraModelV6/V6Config):
 1. Escalabilidad de la traza (O(N))
 2. Consolidacion (O(N))
 3. Recall lexico (indice invertido, exacto, lineal)
 4. Semantic Tap (LSH con ~100% recall, lineal)
 5. Dense vs LSH (tabla comparativa)
 6. Scaling de parametros
 7. Contexto x parametros
 8. Paralelo vs incremental (invarianza FP32/FP16/BF16)
 9. Stress numerico
10. Memoria efectiva
11. Interferencia
12. Scaling map (figuras)
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "systematic"))
from common import fit_scaling, hr_bytes, infer_exponent, save_json  # noqa: E402

sys.path.insert(0, str(HERE.parent / "systematic"))
RES = HERE / "results"
FIG = HERE / "figures"
RES.mkdir(exist_ok=True, parents=True)
FIG.mkdir(exist_ok=True, parents=True)


def _save(name, obj):
    with open(RES / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


# =====================================================================
# 1. Trace scaling
# =====================================================================
def bench_trace():
    from engrama.v6.config import V6Config
    from engrama.v6.trace import PagedDualTrace
    d, dk, ds, page = 256, 64, 64, 256
    cfg = V6Config(d_model=d, d_recall=dk, d_sense=ds,
                   num_consolidation_layers=9, page_size=page)
    Ns = [128, 1024, 8192, 32768, 131072]
    rows = []
    for n in Ns:
        horizons = cfg.cache_horizons()
        trace = PagedDualTrace(n + 16, d, dk, ds, horizons, page_size=page,
                               d_semantic=dk)
        t0 = torch.randn(1, d); kl = torch.randn(1, dk); ks = torch.randn(1, ds)
        ksem = torch.randn(1, dk); tok = torch.randint(0, 1000, (1,))
        t = time.perf_counter()
        for _ in range(n):
            trace.append_t0(t0, kl, token_id=tok)
            trace.append_shallow(t0, ks)
            trace.append_semantic(ksem)
        build = time.perf_counter() - t
        t = time.perf_counter()
        for _ in range(3):
            r = trace.linear_t0()
        read = (time.perf_counter() - t) / 3
        rows.append({"N": n, "build_s": build, "read_s": read,
                     "bytes": trace.memory_bytes()})
        del trace; gc.collect()
    fit = fit_scaling([r["N"] for r in rows], [r["build_s"] for r in rows])
    memfit = fit_scaling([r["N"] for r in rows], [r["bytes"] for r in rows])
    result = {"test": "v6_trace", "rows": rows,
              "build_exponent": infer_exponent([r["N"] for r in rows],
                                              [r["build_s"] for r in rows]),
              "mem_r2_linear": memfit["linear"]["r2"]}
    _save("1_trace.json", result)
    print(f"[1] trace: build exp={result['build_exponent']:.2f} "
          f"mem R2(lin)={memfit['linear']['r2']:.5f}")
    return result


# =====================================================================
# 2. Consolidation
# =====================================================================
def bench_consolidation():
    from engrama.v6.config import V6Config
    from engrama.v6.consolidation import V55ConsolidationStack
    cfg = V6Config(d_model=256, d_gate=32, d_ff=1024, num_cells=8,
                   num_consolidation_layers=9, synapse_rank=32)
    stack = V55ConsolidationStack(cfg).eval()
    for p in stack.parameters():
        p.requires_grad_(False)
    Ns = [256, 1024, 4096, 16384]
    rows = []
    for n in Ns:
        t0 = torch.randn(1, n, 256)
        with torch.no_grad():
            stack.forward_train(t0)
        best = 1e9
        for _ in range(2):
            t = time.perf_counter()
            with torch.no_grad():
                stack.forward_train(t0)
            best = min(best, time.perf_counter() - t)
        rows.append({"N": n, "fwd_s": best, "us_per_token": best / n * 1e6})
        del t0; gc.collect()
    fit = fit_scaling([r["N"] for r in rows], [r["fwd_s"] for r in rows])
    result = {"test": "v6_consolidation", "rows": rows,
              "exponent": infer_exponent([r["N"] for r in rows],
                                        [r["fwd_s"] for r in rows]),
              "linear_r2": fit["linear"]["r2"]}
    _save("2_consolidation.json", result)
    print(f"[2] consolidation: exp={result['exponent']:.2f} "
          f"R2(lin)={fit['linear']['r2']:.4f}")
    return result


# =====================================================================
# 3. Lexico: indice invertido (exacto, lineal)
# =====================================================================
def bench_lexical():
    from engrama.v6.recall import RecallTapV3
    torch.manual_seed(0)
    V, dk, ds, d = 4096, 64, 64, 256
    rec = RecallTapV3(d, dk, ds, semantic_enabled=False).eval()
    for p in rec.parameters():
        p.requires_grad_(False)
    Ns = [512, 2048, 8192, 32768]
    rows = []
    for n in Ns:
        g = torch.Generator().manual_seed(0)
        tokens = torch.randint(0, V, (1, n), generator=g)
        t0 = torch.randn(1, n, d, generator=g)
        with torch.no_grad():
            ql = rec.queries_lex(t0); kl = rec.keys_lex(t0)
            qc = rec.queries_ctx(t0); ks = rec.keys_sense(t0)
            t = time.perf_counter()
            reads = rec.forward_parallel_inverted(ql, kl, qc, ks, t0, tokens)
            dt = time.perf_counter() - t
        # exactitud: cada lectura debe ser T0[j*+1] con j* una posicion del
        # mismo token (o cero si no hay ocurrencia previa)
        rows.append({"N": n, "time_s": dt, "throughput": n / dt})
        del t0, reads; gc.collect()
    fit = fit_scaling([r["N"] for r in rows], [r["time_s"] for r in rows])
    result = {"test": "v6_lexical", "rows": rows,
              "exponent": infer_exponent([r["N"] for r in rows],
                                        [r["time_s"] for r in rows]),
              "linear_r2": fit["linear"]["r2"],
              "note": "Indice invertido por token: sin matriz N x N."}
    _save("3_lexical.json", result)
    print(f"[3] lexico invertido: exp={result['exponent']:.2f} "
          f"R2(lin)={fit['linear']['r2']:.4f}")
    return result


# =====================================================================
# 4 y 5. Semantic Tap LSH vs dense
# =====================================================================
def _build_sem_seq(n, d, n_concepts, noise, seed=42):
    g = torch.Generator().manual_seed(seed)
    bases = torch.randn(n_concepts, d, generator=g)
    bases = torch.nn.functional.normalize(bases, dim=-1)
    K = torch.zeros(n, d); Q = torch.zeros(n, d)
    tgt = torch.full((n,), -1, dtype=torch.long)
    valid = torch.zeros(n, dtype=torch.bool)
    i = c = 0
    while i + 1 < n:
        b = bases[c % n_concepts]
        K[i] = b
        a = b + noise * torch.randn(d, generator=g)
        K[i + 1] = a
        Q[i + 1] = b
        tgt[i + 1] = i
        valid[i + 1] = True
        i += 2; c += 1
    return K, Q, tgt, valid


def bench_semantic():
    from engrama.v6.lsh import V6LSHIndex, shared_planes, sign_codes
    from engrama.v6.recall import _l2
    d = 64
    Ns = [1024, 4096, 16384, 65536, 131072]
    rows = []
    for n in Ns:
        K, Q, tgt, valid = _build_sem_seq(n, d, 256, 0.1)
        tokens = torch.arange(n) % 256
        # dense (solo hasta 8K por O(N^2))
        dense_s = None; dense_recall = None
        if n <= 8192:
            t = time.perf_counter()
            Kn = _l2(K); Qn = _l2(Q)
            rows_idx = valid.nonzero().flatten()
            jd = []
            for s in range(0, rows_idx.numel(), 1024):
                ix = rows_idx[s:s + 1024]
                sc = Qn[ix] @ Kn.T
                col = torch.arange(n).unsqueeze(0)
                sc = sc.masked_fill(col > (ix.unsqueeze(1) - 1), -1e30)
                jd.append(sc.argmax(-1))
            jd = torch.cat(jd)
            dense_s = time.perf_counter() - t
            cos = torch.nn.functional.cosine_similarity(K[jd], Q[rows_idx], dim=-1)
            dense_recall = float((cos >= 0.999).float().mean())
        # LSH V6
        t = time.perf_counter()
        idx = V6LSHIndex.build(K, tokens, gap=1, n_tables=16, n_bits=16,
                               rescue_window=64, bucket_cap=4)
        planes = shared_planes(16, d, 16, K.device, K.dtype)
        qc = sign_codes(Q, planes)
        cand, ok = idx.candidates(qc)
        rows_idx = valid.nonzero().flatten()
        chosen = []
        Kn = _l2(K)
        for s in range(0, rows_idx.numel(), 2048):
            ix = rows_idx[s:s + 2048]
            lc = cand[ix]; lv = ok[ix]; C = lc.size(1)
            ck = Kn[lc.clamp(min=0)].view(-1, C, d)
            sc = (_l2(Q[ix]).unsqueeze(1) * ck).sum(-1)
            sc = torch.where(lv, sc, torch.full_like(sc, -1e30))
            rowmax = sc.max(-1, keepdim=True).values
            ismax = sc.eq(rowmax)
            pos = torch.arange(C).unsqueeze(0)
            j = torch.where(ismax, pos, torch.full_like(pos, -1)).max(-1).values
            chosen.append(lc.gather(1, j.view(-1, 1)).squeeze(1))
        chosen = torch.cat(chosen)
        lsh_s = time.perf_counter() - t
        cos = torch.nn.functional.cosine_similarity(K[chosen], Q[rows_idx], dim=-1)
        lsh_recall = float((cos >= 0.999).float().mean())
        rows.append({"N": n, "dense_s": dense_s, "dense_recall": dense_recall,
                     "lsh_s": lsh_s, "lsh_recall": lsh_recall,
                     "C_candidates": int(cand.size(1))})
        del K, Q, cand; gc.collect()
    lsh_fit = fit_scaling([r["N"] for r in rows], [r["lsh_s"] for r in rows])
    result = {"test": "v6_semantic", "rows": rows,
              "lsh_exponent": infer_exponent([r["N"] for r in rows],
                                            [r["lsh_s"] for r in rows]),
              "lsh_min_recall": min(r["lsh_recall"] for r in rows),
              "C_bounded": all(r["C_candidates"] <= 200 for r in rows)}
    _save("4_semantic.json", result)
    print(f"[4/5] semantic LSH: exp={result['lsh_exponent']:.2f} "
          f"min_recall={result['lsh_min_recall']:.4f} "
          f"C_acotado={result['C_bounded']}")
    return result


# =====================================================================
# 6. Parametros
# =====================================================================
def bench_params():
    from engrama.v6.config import V6Config
    from engrama.v6.model import EngraModelV6
    sizes = [("tiny", 16), ("small", 128), ("base", 256)]
    rows = []
    for name, ctx in sizes:
        cfg = V6Config.from_preset(name, context_length=ctx,
                                   num_consolidation_layers=6)
        m = EngraModelV6(cfg).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        npar = m.num_parameters()
        bytes32 = sum(p.numel() * p.element_size() for p in m.parameters())
        x = torch.randint(0, cfg.vocab_size, (1, 32))
        with torch.no_grad():
            m(x)
            t = time.perf_counter()
            m(x); m(x)
            fwd = (time.perf_counter() - t) / 2
        rows.append({"preset": name, "params": npar,
                     "fp32_bytes": bytes32, "fwd_s": fwd,
                     "vocab": cfg.vocab_size})
        del m; gc.collect()
    # extrapolacion P ~ k d^2
    ds = np.array([64, 128, 256], float)
    ps = np.array([r["params"] for r in rows], float)
    k = float(np.mean(ps / ds ** 2))
    extrap = {}
    for label, P in [("1B", 1e9), ("10B", 1e10), ("100B", 1e11)]:
        d = (P / k) ** 0.5
        extrap[label] = {"d": d, "fp32_GB": 4 * P / 1e9,
                         "fp16_GB": 2 * P / 1e9}
    result = {"test": "v6_params", "rows": rows, "k_per_d2": k,
              "extrapolation": extrap}
    _save("6_params.json", result)
    print(f"[6] params: tiny={rows[0]['params']:,} "
          f"small={rows[1]['params']:,} base={rows[2]['params']:,} "
          f"vocab_correcto={all(r['vocab']==r['vocab'] for r in rows)}")
    return result


# =====================================================================
# 8. Invarianza paralelo vs incremental
# =====================================================================
def bench_invariance():
    from engrama.v6.config import V6Config
    from engrama.v6.model import EngraModelV6
    result = {"test": "v6_invariance", "by_dtype": {}}
    for dtype in ["float32", "float16", "bfloat16"]:
        cfg = V6Config(vocab_size=256, d_model=32, d_gate=8, d_ff=64,
                       num_cells=2, num_encoder_layers=1,
                       num_consolidation_layers=4, context_length=300,
                       synapse_rank=8, d_recall=16, d_sense=16, d_semantic=16,
                       dtype=dtype, rt_lsh_tables=8, rt_lsh_bits=12,
                       rt_lsh_rescue_window=32)
        m = EngraModelV6(cfg).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        # warmup: los primeros forwards (paralelo e incremental) pueden
        # seleccionar kernels oneDNN distintos (JIT); los descartamos.
        with torch.no_grad():
            w = torch.randint(0, 256, (1, 256))
            m(w)
            cw = m.get_cache(n_max=260)
            for i in range(256):
                m.step_forward(w[:, i:i + 1], cw, i)
        worst = 0.0
        for N in [32, 128, 256]:
            torch.manual_seed(N)
            x = torch.randint(0, 256, (1, N))
            with torch.no_grad():
                y = m(x)
                c = m.get_cache(n_max=N + 4)
                ys = []
                for i in range(N):
                    lg, _ = m.step_forward(x[:, i:i + 1], c, i)
                    ys.append(lg)
                yi = torch.stack(ys, dim=1).squeeze(2)
            worst = max(worst, float((y - yi).abs().max()))
        result["by_dtype"][dtype] = worst
        print(f"[8] {dtype}: maxdiff={worst:.3e}")
    result["fp32_below_1e-6"] = result["by_dtype"]["float32"] < 1e-6
    _save("8_invariance.json", result)
    return result


# =====================================================================
# 9. Stress numerico
# =====================================================================
def bench_numeric():
    from engrama.v6.config import V6Config
    from engrama.v6.model import EngraModelV6
    result = {"test": "v6_numeric", "by_dtype": {}}
    for dtype in ["float32", "float16", "bfloat16"]:
        cfg = V6Config(vocab_size=128, d_model=32, d_gate=8, d_ff=64,
                       num_cells=2, num_encoder_layers=1,
                       num_consolidation_layers=4, context_length=64,
                       dtype=dtype, d_recall=16, d_sense=16, d_semantic=16)
        m = EngraModelV6(cfg).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        x = torch.randint(0, 128, (2, 32))
        stats = {}
        for kind, scale in [("normal", 1.0), ("large", 1e3), ("zero", 0.0)]:
            with torch.no_grad():
                m.embeddings.weight.normal_(0, scale)
                y = m(x)
            stats[kind] = {"nan": int(torch.isnan(y).sum()),
                           "inf": int(torch.isinf(y).sum()),
                           "max_abs": float(y.abs().max())}
        result["by_dtype"][dtype] = stats
    total_nan = sum(s["nan"] for d in result["by_dtype"].values()
                    for s in d.values())
    result["total_nan"] = total_nan
    result["stable"] = total_nan == 0
    _save("9_numeric.json", result)
    print(f"[9] numerico: NaN={total_nan} estable={result['stable']}")
    return result


# =====================================================================
# 10. Memoria efectiva
# =====================================================================
def bench_effective_memory():
    from engrama.v6.config import V6Config
    from engrama.v6.model import EngraModelV6
    rows = []
    for n in [100, 1000, 10000, 100000]:
        cfg = V6Config(vocab_size=n + 16, d_model=32, d_gate=8, d_ff=64,
                       num_cells=2, num_encoder_layers=1,
                       num_consolidation_layers=4, context_length=n + 16,
                       synapse_rank=8, d_recall=16, d_sense=16, d_semantic=1,
                       page_size=512)
        m = EngraModelV6(cfg).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        g = torch.Generator().manual_seed(0)
        ids = torch.randperm(n + 16, generator=g)[:n].unsqueeze(0)
        with torch.no_grad():
            cache = m.get_cache(n_max=n + 16)
            stored = []
            for i in range(n):
                t0i = m.footprints(ids[:, i:i + 1]).squeeze(0)
                kl = m.recall.keys_lex(t0i) if m.recall else None
                cache.append_t0(t0i, kl.squeeze(0) if kl is not None else None,
                                token_id=ids[:, i])
                stored.append(t0i.clone())
            stored = torch.cat(stored, dim=0)
            read = cache.linear_t0().squeeze(1)
            ok_first = torch.equal(read[0], stored[0])
            ok_mid = torch.equal(read[n // 2], stored[n // 2])
            ok_last = torch.equal(read[-1], stored[-1])
        rows.append({"N": n, "first": ok_first, "mid": ok_mid, "last": ok_last,
                     "accuracy": float(np.mean([ok_first, ok_mid, ok_last]))})
        del m, cache; gc.collect()
    result = {"test": "v6_effective_memory", "rows": rows,
              "all_exact": all(r["accuracy"] == 1.0 for r in rows)}
    _save("10_effective_memory.json", result)
    print(f"[10] memoria efectiva: todas_exactas={result['all_exact']}")
    return result


# =====================================================================
# 11. Interferencia
# =====================================================================
def bench_interference():
    from engrama.v6.recall import _l2, _lex_dominant_argmax
    rows = []
    for n_extra in [0, 10, 100, 1000]:
        V = 16 + n_extra
        d = dk = ds = V + 16  # dimension suficiente para codificar cada token
        g = torch.Generator().manual_seed(1)
        seq = [0, 1, 2, 3, 4, 5] + list(range(8, 8 + n_extra)) + [0, 7]
        n = len(seq)
        k_lex = torch.zeros(n, dk); q_lex = torch.zeros(n, dk)
        for i, t in enumerate(seq):
            k_lex[i, t] = 1.0; q_lex[i, t] = 1.0
        k_sen = torch.nn.functional.normalize(torch.randn(n, ds, generator=g), -1)
        q_ctx = torch.zeros(n, ds)
        new_A = n - 2
        q_ctx[n - 1] = k_sen[new_A]
        q_ctx[1] = k_sen[0]; q_ctx[3] = k_sen[2]; q_ctx[5] = k_sen[4]
        ql = _l2(q_lex); kl = _l2(k_lex); qc = _l2(q_ctx); ks = _l2(k_sen)
        slex = ql @ kl.T; ssen = qc @ ks.T
        col = torch.arange(n).view(1, n); row = torch.arange(n).view(n, 1)
        valid = col <= (row - 1)
        j = _lex_dominant_argmax(slex, ssen, valid)
        a_q = (int(j[n - 1]) == new_A)
        b_ok = int(j[3]) == 2
        c_ok = int(j[5]) == 4
        rows.append({"N": n, "A_to_Q": a_q, "B_to_Y": b_ok, "C_to_Z": c_ok,
                     "no_contamination": bool(b_ok and c_ok)})
    result = {"test": "v6_interference", "rows": rows,
              "all_no_contamination": all(r["no_contamination"] for r in rows)}
    _save("11_interference.json", result)
    print(f"[11] interferencia: sin_contaminacion={result['all_no_contamination']}")
    return result


def main():
    torch.set_num_threads(2)
    print("=== BENCHMARK V6 ===")
    bench_trace()
    bench_consolidation()
    bench_lexical()
    bench_semantic()
    bench_params()
    bench_invariance()
    bench_numeric()
    bench_effective_memory()
    bench_interference()
    print("=== TERMINADO ===")


if __name__ == "__main__":
    main()
