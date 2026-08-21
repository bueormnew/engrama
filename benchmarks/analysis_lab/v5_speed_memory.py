"""ENGRAMA V5 — benchmark de velocidad y memoria.

Mide (CPU como referencia; en GPU los mismos numeros escalan ~100x):
1. Tiempo por paso de entrenamiento (seq 512/1024) — objetivo GPU ~0.2 s/iter.
2. Generacion incremental: ms/token y tokens/s a contextos 1k/2k/4k, y la
   aceleracion vs recomputar el forward completo por token.
3. Memoria de la cache vs N: lineal (bytes/token constantes, sin compresion).
4. Escalado del forward: pendiente log-log (debe ser ~1, lineal en N).
"""
from __future__ import annotations

import argparse
import json
import time

import torch

torch.set_num_threads(max(1, (torch.get_num_threads() + 1) // 2))

from engrama.v5 import EngraModel, V5Config, V5Trace


def make_model(seq, **over):
    torch.manual_seed(0)
    cfg = V5Config(
        vocab_size=1000, d_model=128, d_gate=16, d_ff=512, num_cells=4,
        num_encoder_layers=1, num_consolidation_layers=8, context_length=seq,
        synapse_rank=16, num_candidates=2, d_recall=32, rt_layers=(4,),
        rt_score_chunk=512, **over,
    )
    return EngraModel(cfg)


def timeit(fn, warmup=1, reps=3):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="v5_speed_results.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results = {"device": dev, "threads": torch.get_num_threads()}

    # ---------------- 1) paso de entrenamiento ----------------
    print("== Paso de entrenamiento ==")
    for seq, bs in ((512, 4), (1024, 2)):
        model = make_model(seq).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, 1000, (bs, seq), device=dev)
        y = torch.randint(0, 1000, (bs, seq), device=dev)

        def step():
            loss = model.forward_loss(x, y, linear_chunk_size=8192)
            opt.zero_grad()
            loss.backward()
            opt.step()

        dt = timeit(step, warmup=1, reps=2)
        tps = bs * seq / dt
        results[f"train_step_seq{seq}"] = {
            "seconds": dt, "batch": bs, "seq": seq, "tokens_per_sec": tps,
        }
        print(f"  seq={seq} bs={bs}: {dt:.3f} s/paso  ({tps:,.0f} tok/s)")
        del model, opt, x, y

    # ---------------- 2) generacion incremental ----------------
    print("== Generacion incremental (cache nativa) ==")
    for seq in (1024, 2048, 4096):
        model = make_model(seq).to(dev).eval()
        cache = model.get_cache(seq)
        x = torch.randint(0, 1000, (seq,), device=dev)

        with torch.no_grad():
            def feed_all():
                c = model.get_cache(seq)
                for t in range(seq):
                    model.step_forward(x[t : t + 1].view(1, 1), c, timestamp=t)

            feed_all()  # dejar la cache llena
            tok = x[-1].view(1, 1)
            dt_step = timeit(
                lambda: model.step_forward(tok, cache, timestamp=seq), warmup=2, reps=10
            )
            dt_full = timeit(
                lambda: model.forward(x.view(1, seq)), warmup=1, reps=2
            )
        speedup = dt_full / dt_step
        results[f"gen_ctx{seq}"] = {
            "ms_per_token": dt_step * 1e3,
            "tokens_per_sec": 1 / dt_step,
            "full_forward_s": dt_full,
            "speedup_vs_recompute": speedup,
        }
        print(f"  ctx={seq}: {dt_step*1e3:.2f} ms/token ({1/dt_step:,.0f} tok/s) | "
              f"recompute={dt_full*1e3:.0f} ms | aceleracion x{speedup:.0f}")
        del model, cache, x

    # ---------------- 3) memoria de cache vs N ----------------
    print("== Memoria de cache (debe crecer LINEAL, sin compresion) ==")
    per_token = []
    for n in (256, 1024, 4096, 16384):
        tr = V5Trace(n, 128, 32, horizons=[1] * 8)
        per_token.append(tr.memory_bytes() / n)
        del tr
    slope = (per_token[-1] - per_token[0]) / (16384 - 256)
    results["cache_bytes_per_token"] = per_token
    results["cache_slope_bytes_per_token2"] = slope
    print(f"  bytes/token: {[round(b, 1) for b in per_token]}  (pendiente {slope:.2e} -> lineal)")

    # ---------------- 4) escalado del forward ----------------
    print("== Forward: tiempo vs N (pendiente log-log ~1 = lineal) ==")
    import math
    model = make_model(8192).to(dev).eval()
    ns, ts = [], []
    for n in (256, 512, 1024, 2048, 4096):
        x = torch.randint(0, 1000, (1, n), device=dev)
        with torch.no_grad():
            dt = timeit(lambda: model.forward(x), warmup=1, reps=2)
        ns.append(n); ts.append(dt)
        print(f"  N={n:5d}: {dt*1e3:7.1f} ms")
    # pendiente robusta via polyfit
    import numpy as np
    slope_fwd = float(np.polyfit(np.log(ns), np.log(ts), 1)[0])
    results["forward_scaling"] = {"ns": ns, "seconds": ts, "loglog_slope": slope_fwd}
    print(f"  pendiente log-log = {slope_fwd:.2f} (1.0 = O(N))")

    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print("resultados ->", a.out)


if __name__ == "__main__":
    main()
