"""ENGRAMA V5.5 — benchmark de velocidad y memoria.

Mide (CPU como referencia; en GPU los mismos numeros escalan ~100x):
1. Paso de entrenamiento denso vs LSH (lineal): seq 512/1024/2048.
2. Generacion incremental: ms/token y aceleracion vs recomputar (cache nativa).
3. Memoria de la traza dual paginada vs N: lineal (bytes/token constantes).
4. Escalado del forward LSH: pendiente log-log ~1 (lineal en N) — la mision de
   V5.5: entrenamiento E inferencia igual de lineales, sin compresion.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

torch.set_num_threads(max(1, (torch.get_num_threads() + 1) // 2))

from engrama.v55 import EngraModelV55, V55Config, PagedDualTrace


def make_model(seq, mode="dense", **over):
    torch.manual_seed(0)
    cfg = V55Config(
        vocab_size=1000, d_model=128, d_gate=16, d_ff=512, num_cells=4,
        num_encoder_layers=1, num_consolidation_layers=8, context_length=seq,
        synapse_rank=16, num_candidates=2, d_recall=32, d_sense=32,
        rt_score_chunk=512, rt_train_mode=mode, page_size=256, **over,
    )
    return EngraModelV55(cfg)


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
    ap.add_argument("--out", default="v55_speed_results.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results = {"device": dev, "threads": torch.get_num_threads()}

    # ---------------- 1) paso de entrenamiento: denso vs LSH ----------------
    print("== Paso de entrenamiento: denso (O(N^2)) vs LSH (O(N)) ==")
    for seq, bs in ((512, 4), (1024, 2), (2048, 2)):
        for mode in ("dense", "lsh"):
            model = make_model(seq, mode=mode).to(dev)
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            x = torch.randint(0, 1000, (bs, seq), device=dev)
            y = torch.randint(0, 1000, (bs, seq), device=dev)

            def step():
                loss = model.forward_loss(x, y, linear_chunk_size=8192,
                                          retrieval_weight=0.0)
                opt.zero_grad(); loss.backward(); opt.step()
            try:
                dt = timeit(step, warmup=1, reps=2)
            except Exception as e:  # pragma: no cover
                dt = float("nan"); print("  ", mode, "err", e)
            tps = bs * seq / dt if dt == dt else 0
            results[f"train_{mode}_seq{seq}"] = {
                "seconds": dt, "tokens_per_sec": tps}
            print(f"  {mode:5s} seq={seq} bs={bs}: {dt:.3f} s/paso ({tps:,.0f} tok/s)")
            del model, opt, x, y

    # ---------------- 2) generacion incremental (cache nativa) -------------
    print("== Generacion incremental (cache nativa paginada) ==")
    for seq in (1024, 2048, 4096):
        model = make_model(seq).to(dev).eval()
        cache = model.get_cache(seq)
        x = torch.randint(0, 1000, (seq,), device=dev)
        with torch.no_grad():
            c = model.get_cache(seq)
            for t in range(seq):
                model.step_forward(x[t:t + 1].view(1, 1), c, timestamp=t)
            tok = x[-1].view(1, 1)
            dt_step = timeit(lambda: model.step_forward(tok, cache, timestamp=seq),
                             warmup=2, reps=8)
            dt_full = timeit(lambda: model.forward(x.view(1, seq)),
                             warmup=1, reps=2)
        speedup = dt_full / dt_step
        results[f"gen_ctx{seq}"] = {"ms_per_token": dt_step * 1e3,
                                    "tokens_per_sec": 1 / dt_step,
                                    "speedup_vs_recompute": speedup}
        print(f"  ctx={seq}: {dt_step*1e3:.2f} ms/token ({1/dt_step:,.0f} tok/s) | "
              f"recompute={dt_full*1e3:.0f} ms | x{speedup:.0f}")
        del model, cache, x

    # ---------------- 3) memoria de traza vs N ----------------
    print("== Memoria de traza dual paginada (LINEAL, sin compresion) ==")
    per_token = []
    for n in (256, 1024, 4096, 16384):
        tr = PagedDualTrace(n, 128, 32, 32, horizons=[1] * 8, page_size=256)
        # poblar la traza para medir memoria real por token escrito
        for t in range(n):
            tr.append_t0(torch.zeros(1, 128), torch.zeros(1, 32),
                         token_id=torch.tensor([t % 1000]))
            tr.append_shallow(torch.zeros(1, 128), torch.zeros(1, 32))
        per_token.append(tr.memory_bytes() / n)
        del tr
    slope = (per_token[-1] - per_token[0]) / (16384 - 256)
    results["trace_bytes_per_token"] = per_token
    results["trace_slope"] = slope
    print(f"  bytes/token: {[round(b, 1) for b in per_token]}  (pendiente {slope:.2e})")

    # ---------------- 4) escalado forward LSH (lineal) ----------------
    print("== Forward LSH: tiempo vs N (pendiente log-log ~1 = O(N)) ==")
    import numpy as np
    model = make_model(8192, mode="lsh").to(dev).eval()
    ns, ts = [], []
    for n in (256, 512, 1024, 2048, 4096):
        x = torch.randint(0, 1000, (1, n), device=dev)
        with torch.no_grad():
            dt = timeit(lambda: model.forward(x), warmup=1, reps=2)
        ns.append(n); ts.append(dt)
        print(f"  N={n:5d}: {dt*1e3:7.1f} ms")
    slope_fwd = float(np.polyfit(np.log(ns), np.log(ts), 1)[0])
    results["forward_lsh_scaling"] = {"ns": ns, "seconds": ts, "loglog_slope": slope_fwd}
    print(f"  pendiente log-log = {slope_fwd:.2f} (1.0 = O(N))")

    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print("resultados ->", a.out)


if __name__ == "__main__":
    main()
