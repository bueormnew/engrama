"""ENGRAMA V5 — entrenamiento lineal: denso O(N^2 d_k) vs LSH O(N (1+t*cap) d_k).

Mide tiempo de forward+backward (modo train, STE activo) y memoria pico de
activaciones para ambos modos del Recall Tap a N creciente. En CPU sirve de
referencia; el mismo script en GPU da los numeros de produccion.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

torch.set_num_threads(max(1, (torch.get_num_threads() + 1) // 2))

from engrama.v5 import RecallTap


def bench(rt, q, k, t0, tokens, reps=2):
    # calentar
    _ = rt.forward_parallel(q, k, t0)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0t = time.perf_counter()
    for _ in range(reps):
        _ = rt.forward_parallel(q, k, t0)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dense = (time.perf_counter() - t0t) / reps

    _ = rt.forward_parallel_lsh(q, k, t0, tokens)
    t0t = time.perf_counter()
    for _ in range(reps):
        _ = rt.forward_parallel_lsh(q, k, t0, tokens)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    lsh = (time.perf_counter() - t0t) / reps
    return dense, lsh


def train_step_time(rt, q, k, t0, tokens, mode, reps=2):
    def step():
        if mode == "dense":
            reads = rt.forward_parallel(q, k, t0)
        else:
            reads = rt.forward_parallel_lsh(q, k, t0, tokens)
        loss = reads.square().mean()
        loss.backward()
    for _ in range(1):
        step()
        rt.zero_grad()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0t = time.perf_counter()
    for _ in range(reps):
        step()
        rt.zero_grad()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.perf_counter() - t0t) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="v5_lsh_speed.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dispositivo: {dev} | hilos: {torch.get_num_threads()}")
    results = {}
    print(f"{'N':>6} | {'fwd denso':>10} {'fwd LSH':>9} {'x':>5} | "
          f"{'paso denso':>10} {'paso LSH':>9} {'x':>5}")
    for n in (1024, 2048, 4096, 8192):
        torch.manual_seed(0)
        d, dk, vocab = 128, 64, 1000
        rt = RecallTap(d, dk, value="next", gap=1, score_chunk=1024).to(dev)
        q = torch.randn(1, n, dk, device=dev, requires_grad=True)
        k = torch.randn(1, n, dk, device=dev, requires_grad=True)
        t0 = torch.randn(1, n, d, device=dev, requires_grad=True)
        tokens = torch.randint(0, vocab, (1, n), device=dev)
        dense_f, lsh_f = bench(rt, q, k, t0, tokens)
        dense_t = train_step_time(rt, q, k, t0, tokens, "dense")
        lsh_t = train_step_time(rt, q, k, t0, tokens, "lsh")
        results[n] = {"dense_fwd": dense_f, "lsh_fwd": lsh_f,
                      "dense_step": dense_t, "lsh_step": lsh_t}
        print(f"{n:>6} | {dense_f*1e3:>8.1f}ms {lsh_f*1e3:>7.1f}ms {dense_f/lsh_f:>4.1f}x | "
              f"{dense_t*1e3:>8.1f}ms {lsh_t*1e3:>7.1f}ms {dense_t/lsh_t:>4.1f}x", flush=True)
        del rt, q, k, t0, tokens
    # pendiente log-log del paso LSH (debe ser ~1: lineal en N)
    import math
    ns = sorted(results)
    xs = [math.log(n) for n in ns]
    ys = [math.log(results[n]["lsh_step"]) for n in ns]
    b = (len(xs) * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys)) / \
        (len(xs) * sum(x * x for x in xs) - sum(xs) ** 2)
    print(f"pendiente log-log paso LSH: {b:.2f} (1.0 = lineal en N)")
    results["lsh_step_slope"] = b
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
