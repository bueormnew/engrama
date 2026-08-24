"""Prueba 8 — Forward paralelo vs incremental (invarianza causal).

El repositorio afirma que ambas rutas difieren en < 1e-6. Medimos
    max |Y_paralelo - Y_incremental|
para N = 16..16K y FP32/FP16/BF16. Tambien tiempo, memoria, throughput.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_json

from engrama.v55.config import V55Config
from engrama.v55.model import EngraModelV55


def run_for_dtype(dtype_name: str, Ns, seq=512):
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[dtype_name]
    cfg = V55Config(
        vocab_size=4096, d_model=64, d_gate=16, d_ff=256, num_cells=4,
        num_encoder_layers=1, num_consolidation_layers=8,
        context_length=max(Ns) + 16, synapse_rank=16, d_recall=32, d_sense=32,
        d_semantic=32, recall_enabled=True, semantic_recall_enabled=True,
        dtype=dtype_name, page_size=256, tie_embeddings=True,
        rt_sem_recall_mode="dense")
    model = EngraModelV55(cfg).eval().to(dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    rows = []
    print(f"\n--- {dtype_name} ---")
    print(f"{'N':>7} {'max_abs_err':>12} {'mean_abs':>11} "
          f"{'par_ms':>9} {'inc_ms/tok':>11}")
    g = torch.Generator().manual_seed(123)
    for n in Ns:
        x = torch.randint(0, cfg.vocab_size, (1, n), generator=g)
        # paralelo
        with torch.no_grad():
            y_par = model(x)  # (1,N,V)
        t = time.perf_counter()
        with torch.no_grad():
            y_par = model(x)
        t_par = time.perf_counter() - t
        # incremental
        cache = model.get_cache(n_max=n + 4)
        with torch.no_grad():
            logits_last = None
            for i in range(n):
                tok = x[:, i:i + 1]
                logits_last, _ = model.step_forward(tok, cache, i)
        # y_inc en posicion i es el logits tras procesar el token i (estado en i)
        # Reconstruimos todos los logits incrementales para comparar con y_par.
        cache2 = model.get_cache(n_max=n + 4)
        y_inc = []
        with torch.no_grad():
            for i in range(n):
                tok = x[:, i:i + 1]
                lg, _ = model.step_forward(tok, cache2, i)
                y_inc.append(lg)
        y_inc = torch.stack(y_inc, dim=1).squeeze(2)  # (1,N,V)
        t_inc_start = time.perf_counter()
        cache3 = model.get_cache(n_max=n + 4)
        with torch.no_grad():
            for i in range(n):
                model.step_forward(x[:, i:i + 1], cache3, i)
        t_inc = time.perf_counter() - t_inc_start
        diff = (y_par.float() - y_inc.float()).abs()
        max_err = float(diff.max().item())
        mean_err = float(diff.mean().item())
        rows.append({
            "N": n, "max_abs_err": max_err, "mean_abs_err": mean_err,
            "parallel_s": t_par, "incremental_s": t_inc,
            "incremental_s_per_token": t_inc / n,
            "throughput_parallel": n / t_par,
            "throughput_incremental": n / t_inc,
        })
        print(f"{n:>7} {max_err:>12.3e} {mean_err:>11.3e} "
              f"{t_par*1000:>9.1f} {t_inc/n*1000:>11.2f}")
        del x, y_par, y_inc, cache, cache2, cache3; gc.collect()
    del model; gc.collect()
    return rows


def run():
    torch.set_num_threads(2)
    # N pequenas para todas las precisiones; FP32 tambien a N grande.
    Ns_small = [16, 32, 64, 128, 256, 512, 1024]
    # En CPU el incremental es lento; cubrimos hasta 4K para FP32.
    Ns_fp32 = Ns_small + [2048, 4096]
    result = {"test": "8_parallel_vs_incremental", "device": "cpu", "by_dtype": {}}
    result["by_dtype"]["float32"] = run_for_dtype("float32", Ns_fp32)
    for dt in ("float16", "bfloat16"):
        try:
            result["by_dtype"][dt] = run_for_dtype(dt, Ns_small)
        except Exception as e:
            result["by_dtype"][dt] = {"error": str(e)}
            print(f"{dt} fallo: {e}")
    # veredicto
    all_err = [r["max_abs_err"] for rows in result["by_dtype"].values()
               if isinstance(rows, list) for r in rows]
    result["max_error_overall"] = max(all_err) if all_err else None
    result["invariant_below_1e-6"] = all(e < 1e-6 for e in all_err) if all_err else None
    path = save_json("8_parallel_vs_incremental.json", result)
    print(f"\nError max global: {result['max_error_overall']:.3e}")
    print(f"Inv <1e-6: {result['invariant_below_1e-6']}")
    print("Guardado:", path)


if __name__ == "__main__":
    run()
