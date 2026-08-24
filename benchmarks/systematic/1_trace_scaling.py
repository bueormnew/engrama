"""Prueba 1 — Escalabilidad de la traza (T0 -> trace -> append -> read).

Mide:
  - tiempo de CONSTRUCCION (append secuencial de N huellas)
  - tiempo de APPEND individual (amortizado)
  - tiempo de LECTURA lineal (linear_t0)
  - bytes almacenados reales vs teoricos
  - throughput de tokens (tokens/seg) en append y lectura
  - crecimiento empirico de memoria (ajuste O(N))

No requiere entrenamiento: tensores aleatorios.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (fit_scaling, hr_bytes, infer_exponent, save_json, timeit)

from engrama.v55.config import V55Config
from engrama.v55.trace import PagedDualTrace


def build_trace(n: int, d: int, dk: int, ds: int, page: int, B: int = 1):
    cfg = V55Config(d_model=d, d_recall=dk, d_sense=ds,
                    num_consolidation_layers=9, page_size=page)
    horizons = cfg.cache_horizons()
    trace = PagedDualTrace(n_max=n + 16, d_model=d, d_recall=dk, d_sense=ds,
                           horizons=horizons, page_size=page,
                           dtype=torch.float32, d_semantic=dk)
    g = torch.Generator().manual_seed(0)
    # construccion: append secuencial de N huellas (batch B)
    t0 = torch.randn(B, d, generator=g)
    klex = torch.randn(B, dk, generator=g)
    ksen = torch.randn(B, ds, generator=g)
    ksem = torch.randn(B, dk, generator=g)
    tok = torch.randint(0, 1000, (B,), generator=g)
    for _ in range(n):
        trace.append_t0(t0, klex, token_id=tok)
        trace.append_shallow(t0, ksen)
        trace.append_semantic(ksem)
    return trace, horizons


def run():
    torch.set_num_threads(2)
    d, dk, ds, page = 256, 64, 64, 256
    # El append en CPU es O(1) pero con sobrecarga Python; cubrimos un amplio
    # rango. 128K huellas * (256+64+64+64)*4 bytes ~ 220 MB -> seguro en 3.8GB.
    Ns = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    rows = []
    print(f"{'N':>8} {'build_ms':>10} {'append_us':>10} {'read_ms':>10} "
          f"{'thr_app':>9} {'thr_rd':>9} {'bytes_real':>12} {'bytes_theo':>12}")
    for n in Ns:
        # tiempo de construccion
        def _build():
            return build_trace(n, d, dk, ds, page)
        tr = timeit(_build, warmup=0, repeats=2 if n <= 8192 else 1)
        trace, horizons = tr.value
        # append individual (marginal): medimos 200 appends fresco
        def _append():
            t0 = torch.randn(1, d)
            trace.append_t0(t0, torch.randn(1, dk), token_id=torch.tensor([1]))
            trace.append_shallow(t0, torch.randn(1, ds))
            trace.append_semantic(torch.randn(1, dk))
            return None
        ap = timeit(_append, warmup=3, repeats=20)
        # lectura lineal
        def _read():
            return trace.linear_t0()
        rd = timeit(_read, warmup=1, repeats=3)
        # memoria
        real_bytes = trace.memory_bytes()
        # teorico: por token = (d + d + dk + ds + dk)*4  (t0, ts, klex, ksen, ksem)
        per_tok = (2 * d + dk + ds + dk) * 4
        theo_bytes = n * per_tok
        # sobrecarga de paginas (preasignadas)
        n_pages = (n + page - 1) // page
        page_overhead = n_pages * page * (2 * d + dk + ds + dk + 1) * 4
        rows.append({
            "N": n,
            "build_s": tr.seconds,
            "append_s": ap.seconds,
            "read_s": rd.seconds,
            "rss_mb_build": tr.rss_mb,
            "bytes_real": real_bytes,
            "bytes_theoretical_active": theo_bytes,
            "bytes_allocated_pages": page_overhead,
            "n_pages": n_pages,
            "throughput_append_tok_s": n / tr.seconds,
            "throughput_read_tok_s": n / rd.seconds,
            "append_us": ap.seconds * 1e6,
        })
        print(f"{n:>8} {tr.seconds*1000:>10.1f} {ap.seconds*1e6:>10.1f} "
              f"{rd.seconds*1000:>10.2f} {n/tr.seconds:>9.0f} "
              f"{n/rd.seconds:>9.0f} {hr_bytes(real_bytes):>12} "
              f"{hr_bytes(theo_bytes):>12}")
        del trace
        import gc; gc.collect()

    # ajuste de crecimiento
    build_fit = fit_scaling([r["N"] for r in rows], [r["build_s"] for r in rows])
    read_fit = fit_scaling([r["N"] for r in rows], [r["read_s"] for r in rows])
    mem_fit = fit_scaling([r["N"] for r in rows], [r["bytes_real"] for r in rows])
    p_build = infer_exponent([r["N"] for r in rows], [r["build_s"] for r in rows])
    p_read = infer_exponent([r["N"] for r in rows], [r["read_s"] for r in rows])
    p_mem = infer_exponent([r["N"] for r in rows], [r["bytes_real"] for r in rows])

    result = {
        "test": "1_trace_scaling",
        "d_model": d, "d_recall": dk, "d_sense": ds, "page_size": page,
        "device": "cpu",
        "rows": rows,
        "fit": {
            "build_time": build_fit,
            "read_time": read_fit,
            "memory_bytes": mem_fit,
        },
        "empirical_exponent": {
            "build": p_build, "read": p_read, "memory": p_mem,
        },
        "conclusion": {
            "build_O_N_r2": build_fit["linear"]["r2"],
            "read_O_N_r2": read_fit["linear"]["r2"],
            "memory_O_N_r2": mem_fit["linear"]["r2"],
            "build_best_model": build_fit["best"],
            "read_best_model": read_fit["best"],
            "memory_best_model": mem_fit["best"],
        },
    }
    path = save_json("1_trace_scaling.json", result)
    print("\nAjuste build:", build_fit["best"],
          f"lineal R2={build_fit['linear']['r2']:.5f}",
          f"cuad R2={build_fit['quadratic']['r2']:.5f}",
          f"exponente~{p_build:.3f}")
    print("Ajuste read :", read_fit["best"],
          f"lineal R2={read_fit['linear']['r2']:.5f}",
          f"cuad R2={read_fit['quadratic']['r2']:.5f}",
          f"exponente~{p_read:.3f}")
    print("Ajuste mem  :", mem_fit["best"],
          f"lineal R2={mem_fit['linear']['r2']:.6f}",
          f"exponente~{p_mem:.3f}")
    print("Guardado:", path)
    return result


if __name__ == "__main__":
    run()
