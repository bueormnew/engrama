"""Prueba 9 — Stress test numerico (sin entrenar).

Generamos entradas con distribuciones extremas:
  normal, large magnitude (1e3), small (1e-3), repeated, near-zero, random noise
y ejecutamos el modelo en FP32, FP16, BF16. Registramos:
  NaN, Inf, overflow, underflow, max|abs|, mean|abs|, varianza.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_json

from engrama.v55.config import V55Config
from engrama.v55.model import EngraModelV55


def make_input(kind: str, n: int, V: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    # input_ids son enteros; la "distribucion extrema" se aplica a los embeddings
    # interviniendo el peso de embedding. Generamos ids y luego escalamos el peso.
    ids = torch.randint(0, V, (1, n), generator=g)
    return ids, kind


def perturb_embeddings(model, kind: str):
    with torch.no_grad():
        w = model.embeddings.weight
        if kind == "normal":
            w.normal_(0, 1.0)
        elif kind == "large":
            w.normal_(0, 1e3)
        elif kind == "small":
            w.normal_(0, 1e-3)
        elif kind == "repeated":
            w.fill_(0.7)
            w.add_(torch.randn_like(w) * 1e-4)
        elif kind == "nearzero":
            w.zero_()
            w.add_(torch.randn_like(w) * 1e-6)
        elif kind == "noise":
            w.uniform_(-10, 10)


def run_dtype(dtype_name: str, n: int = 512):
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[dtype_name]
    cfg = V55Config.from_preset(
        "tiny", context_length=n + 16, vocab_size=256,
        recall_enabled=True, dtype=dtype_name, num_consolidation_layers=8,
        d_recall=32, d_sense=32, d_semantic=32)
    model = EngraModelV55(cfg).eval().to(dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    kinds = ["normal", "large", "small", "repeated", "nearzero", "noise"]
    rows = []
    for k in kinds:
        perturb_embeddings(model, k)
        ids, _ = make_input(k, n, cfg.vocab_size, 42)
        try:
            with torch.no_grad():
                y = model(ids)
            yf = y.float()
            nan = int(torch.isnan(yf).sum().item())
            inf = int(torch.isinf(yf).sum().item())
            mx = float(yf.abs().max().item())
            mean = float(yf.abs().mean().item())
            var = float(yf.var().item())
            # underflow: valores que se volvieron exactamente cero
            underflow = int((yf == 0).sum().item())
            overflow = int((yf.abs() > torch.finfo(dtype).max / 2 if dtype != torch.float32
                            else yf.new_tensor(False)).sum().item()) if dtype != torch.float32 else 0
            rows.append({
                "kind": k, "nan": nan, "inf": inf, "overflow": overflow,
                "underflow": underflow, "max_abs": mx, "mean_abs": mean,
                "variance": var,
            })
        except RuntimeError as e:
            rows.append({"kind": k, "error": str(e)[:120]})
        gc.collect()
    del model; gc.collect()
    return rows


def run():
    torch.set_num_threads(2)
    result = {"test": "9_numeric_stress", "device": "cpu", "by_dtype": {}}
    for dt in ("float32", "float16", "bfloat16"):
        try:
            result["by_dtype"][dt] = run_dtype(dt)
            print(f"\n=== {dt} ===")
            for r in result["by_dtype"][dt]:
                if "error" in r:
                    print(f"  {r['kind']:>10}: ERROR {r['error'][:60]}")
                else:
                    print(f"  {r['kind']:>10}: NaN={r['nan']} Inf={r['inf']} "
                          f"max|.|={r['max_abs']:.3e} mean|.|={r['mean_abs']:.3e} "
                          f"var={r['variance']:.3e}")
        except Exception as e:
            result["by_dtype"][dt] = {"error": str(e)}
            print(f"{dt} fallo: {e}")
    # veredicto: cuantos NaN/Inf en total
    total_nan = sum(r.get("nan", 0) for rows in result["by_dtype"].values()
                    if isinstance(rows, list) for r in rows)
    total_inf = sum(r.get("inf", 0) for rows in result["by_dtype"].values()
                    if isinstance(rows, list) for r in rows)
    result["total_nan"] = total_nan
    result["total_inf"] = total_inf
    result["stable"] = (total_nan == 0 and total_inf == 0)
    path = save_json("9_numeric_stress.json", result)
    print(f"\nTotal NaN={total_nan}, Inf={total_inf}; estable={result['stable']}")
    print("Guardado:", path)


if __name__ == "__main__":
    run()
