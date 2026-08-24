"""Prueba 2 — Escalabilidad de la Consolidacion (T0 -> Consolidation -> T).

Mide tiempo de forward_train de la pila de consolidacion en funcion de N, y
ajusta:
  - modelo lineal      aN + b
  - modelo cuadratico  aN^2 + bN + c

Objetivo: evidencia experimental de O(N) (offsets fijos por capa, sin matriz
N x N). Se mide tambien memoria de activaciones y throughput.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fit_scaling, infer_exponent, save_json, timeit

from engrama.v55.config import V55Config
from engrama.v55.consolidation import V55ConsolidationStack


def run():
    torch.set_num_threads(2)
    d = 256
    # 9 capas -> receptive field ~256; para N grandes los offsets quedan fijos
    cfg = V55Config(d_model=d, d_gate=32, d_ff=1024, num_cells=8,
                    num_consolidation_layers=9, synapse_rank=32)
    stack = V55ConsolidationStack(cfg).eval()
    for p in stack.parameters():
        p.requires_grad_(False)

    # hasta 16K; el coste es O(L*P*N*d) donde P=4 offsets -> lineal. 16K seguro.
    Ns = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    rows = []
    print(f"{'N':>8} {'fwd_ms':>10} {'thr_tok_s':>11} {'rss_mb':>8} {'act_KB':>9}")
    for n in Ns:
        t0 = torch.randn(1, n, d)
        def _fwd(t0=t0):
            with torch.no_grad():
                return stack.forward_train(t0)
        tr = timeit(_fwd, warmup=1, repeats=3)
        tl, tsh = tr.value
        # memoria de salida
        act_bytes = tl.numel() * tl.element_size() + tsh.numel() * tsh.element_size()
        rows.append({
            "N": n, "fwd_s": tr.seconds,
            "throughput_tok_s": n / tr.seconds,
            "rss_mb": tr.rss_mb, "output_bytes": act_bytes,
        })
        print(f"{n:>8} {tr.seconds*1000:>10.2f} {n/tr.seconds:>11.0f} "
              f"{tr.rss_mb:>8.1f} {act_bytes/1024:>9.1f}")
        del t0, tl, tsh; gc.collect()

    fit = fit_scaling([r["N"] for r in rows], [r["fwd_s"] for r in rows])
    p = infer_exponent([r["N"] for r in rows], [r["fwd_s"] for r in rows])

    # Estimacion del coste teorico: cada capa hace ~ (varias einsums) sobre
    # (B,N,P,d) con P<=4 y L capas => O(L*P*N*d^2-ish por gate) pero lineal en N.
    result = {
        "test": "2_consolidation_scaling",
        "d_model": d, "num_layers": cfg.num_consolidation_layers,
        "device": "cpu",
        "rows": rows,
        "fit": fit,
        "empirical_exponent": p,
        "conclusion": {
            "linear_r2": fit["linear"]["r2"],
            "quadratic_r2": fit["quadratic"]["r2"],
            "nlogn_r2": fit["nlogn"]["r2"],
            "best_model": fit["best"],
            "is_linear": fit["linear"]["r2"] > 0.98
                         and fit["linear"]["r2"] >= fit["quadratic"]["r2"] - 0.005,
        },
    }
    path = save_json("2_consolidation_scaling.json", result)
    print(f"\nExponente empirico ~ {p:.3f}")
    print(f"Lineal R2={fit['linear']['r2']:.6f}  "
          f"NlogN R2={fit['nlogn']['r2']:.6f}  "
          f"Cuadratico R2={fit['quadratic']['r2']:.6f}  -> mejor: {fit['best']}")
    print("Guardado:", path)
    return result


if __name__ == "__main__":
    run()
