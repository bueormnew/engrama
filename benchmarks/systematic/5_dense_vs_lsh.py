"""Prueba 5 — Dense vs LSH (tabla comparativa directa de Semantic Recall).

Mide para cada N:
  - tiempo dense (O(N^2))
  - tiempo LSH   (O(N*C))
  - memoria (pico de tensores)
  - recall (accuracy) de cada uno
Usa el mismo setup sintetico de la prueba 4 (concepto/alias) con d=256.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fit_scaling, infer_exponent, save_json, torch_tensor_bytes

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "t4", str(Path(__file__).resolve().parent / "4_semantic_tap.py"))
t4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(t4)


def run():
    torch.set_num_threads(2)
    d = 256
    n_concepts, noise = 256, 0.1
    # dense hasta donde la RAM/tiempo lo permitan; LSH hasta 128K
    rows = []
    print(f"{'N':>7} {'dense_ms':>10} {'lsh_ms':>9} {'dense_KB':>9} "
          f"{'lsh_KB':>9} {'R_dense':>8} {'R_lsh':>8} {'speedup_LSH':>11}")
    Ns = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    for n in Ns:
        K, Q, tgt, valid = t4.build_semantic_seq(n, d, n_concepts, noise, 42)
        tokens = torch.arange(n) % n_concepts
        row = {"N": n}
        # dense (solo hasta 8K en CPU)
        if n <= 8192:
            for _ in range(1):
                t4.semantic_dense(K, Q, valid)
            gc.collect()
            b0 = torch_tensor_bytes()
            best = 1e9
            reps = 3 if n <= 2048 else 1
            for _ in range(reps):
                t = time.perf_counter()
                jd = t4.semantic_dense(K, Q, valid)
                best = min(best, time.perf_counter() - t)
            b1 = torch_tensor_bytes()
            row["dense_s"] = best
            row["dense_peak_bytes"] = max(0, b1 - b0)
            row["recall_dense"] = t4.acc(jd, tgt, valid)
            del jd
        # LSH
        for _ in range(1):
            t4.semantic_lsh(K, Q, valid, tokens)
        gc.collect()
        b0 = torch_tensor_bytes()
        best = 1e9
        reps = 2 if n <= 4096 else 1
        for _ in range(reps):
            t = time.perf_counter()
            jl = t4.semantic_lsh(K, Q, valid, tokens)
            best = min(best, time.perf_counter() - t)
        b1 = torch_tensor_bytes()
        row["lsh_s"] = best
        row["lsh_peak_bytes"] = max(0, b1 - b0)
        row["recall_lsh"] = t4.acc(jl, tgt, valid)
        if "dense_s" in row:
            row["speedup_lsh_vs_dense"] = row["dense_s"] / best
        rows.append(row)
        print(f"{n:>7} {row.get('dense_s',float('nan'))*1000:>10.1f} "
              f"{best*1000:>9.1f} "
              f"{row.get('dense_peak_bytes',0)/1024:>9.1f} "
              f"{row['lsh_peak_bytes']/1024:>9.1f} "
              f"{row.get('recall_dense',float('nan')):>8.3f} "
              f"{row['recall_lsh']:>8.3f} "
              f"{row.get('speedup_lsh_vs_dense',float('nan')):>11.2f}")
        del K, Q, tgt, valid, jl; gc.collect()

    dense_pts = [(r["N"], r["dense_s"]) for r in rows if "dense_s" in r]
    lsh_pts = [(r["N"], r["lsh_s"]) for r in rows]
    fit_dense = fit_scaling([p[0] for p in dense_pts], [p[1] for p in dense_pts])
    fit_lsh = fit_scaling([p[0] for p in lsh_pts], [p[1] for p in lsh_pts])
    result = {
        "test": "5_dense_vs_lsh", "d": d,
        "rows": rows,
        "fit": {"dense": fit_dense, "lsh": fit_lsh},
        "exponent": {
            "dense": infer_exponent([p[0] for p in dense_pts], [p[1] for p in dense_pts]),
            "lsh": infer_exponent([p[0] for p in lsh_pts], [p[1] for p in lsh_pts]),
        },
    }
    path = save_json("5_dense_vs_lsh.json", result)
    print("Guardado:", path)
    return result


if __name__ == "__main__":
    run()
