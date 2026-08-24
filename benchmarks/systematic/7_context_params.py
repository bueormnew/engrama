"""Prueba 7 — Matriz Contexto x Parametros (analitica + puntos medidos).

Para cada celda (P, N) estimamos:
  - memoria de pesos (FP32)
  - memoria de traza (N * bytes/token)
  - memoria de activaciones en forward paralelo (dominada por el recall denso:
    score matrix ~ N*N*d_k y por la consolidacion ~ L*N*d)
  - FLOPs teoricas por secuencia
  - throughput esperado (basado en los puntos medidos en las pruebas 1,2,5)

Las celdas 1B se calculan solo analiticamente (no se instancian).
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import hr_bytes, save_json

from engrama.v55.config import V55Config
from engrama.v55.model import EngraModelV55


def model_stats(cfg: V55Config, seq_len: int):
    """Mide pesos y estima activaciones/traza para un forward de seq_len."""
    model = EngraModelV55(cfg).eval()
    params = model.num_parameters()
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # traza: por token = (2d + dk + ds + dsem) * 4 bytes (fp32)
    bpt = (2 * cfg.d_model + cfg.d_recall + cfg.d_sense + cfg.d_semantic) * 4
    trace_bytes = seq_len * bpt
    # activaciones (forward paralelo, eval):
    #  - T0, T_last, T_shallow: 3 * N * d
    #  - proyecciones recall: q_lex,k_lex,q_ctx,k_sen,q_sem,k_sem: 6*N*(dk/dsem)
    #  - score dense lexical: N*N*dk (una matriz) -> O(N^2)
    #  - score dense semantic: N*N*dsem
    #  - consolidation views: L * (B,N,P,d) con P=4
    P_offsets = 4
    activations_bytes = (
        3 * seq_len * cfg.d_model
        + 4 * seq_len * cfg.d_recall
        + 2 * seq_len * cfg.d_semantic
        + seq_len * seq_len * (cfg.d_recall + cfg.d_semantic)
        + cfg.num_consolidation_layers * P_offsets * seq_len * cfg.d_model
    ) * 4
    # FLOPs (orden de magnitud):
    #  - encoder/consolidation por token: O(L * (d^2 + d*dff)) ~ L*(d^2+4d^2)
    per_token = cfg.num_consolidation_layers * (cfg.d_model**2 + 4 * cfg.d_model**2)
    #  - recall denso: N * N * (dk+dsem) productos punto
    recall_flops = seq_len * seq_len * (cfg.d_recall + cfg.d_semantic)
    flops = seq_len * per_token + recall_flops
    res = {
        "params": params, "weight_bytes": weight_bytes,
        "trace_bytes": trace_bytes,
        "activations_bytes": activations_bytes,
        "total_bytes": weight_bytes + trace_bytes + activations_bytes,
        "flops": flops,
        "d_model": cfg.d_model,
    }
    del model; gc.collect()
    return res


def run():
    torch.set_num_threads(2)
    # modelos medibles (reutilizamos tamanos de la prueba 6)
    meas = [
        ("10M",  dict(d_model=256, d_ff=1024, d_gate=32, num_cells=8,
                      num_encoder_layers=2, num_consolidation_layers=9,
                      synapse_rank=32, d_recall=64, d_sense=64, d_semantic=64)),
        ("30M",  dict(d_model=384, d_ff=1536, d_gate=48, num_cells=12,
                      num_encoder_layers=2, num_consolidation_layers=9,
                      synapse_rank=32, d_recall=64, d_sense=64, d_semantic=64)),
        ("100M", dict(d_model=640, d_ff=2560, d_gate=64, num_cells=16,
                      num_encoder_layers=2, num_consolidation_layers=9,
                      synapse_rank=32, d_recall=96, d_sense=96, d_semantic=96)),
    ]
    contexts = [1024, 4096, 16384, 65536]
    matrix = {}
    for name, kw in meas:
        matrix[name] = {}
        for N in contexts:
            cfg = V55Config(vocab_size=256, context_length=max(N, 256),
                            page_size=256, **kw)
            try:
                s = model_stats(cfg, N)
                matrix[name][str(N)] = {
                    "params": s["params"],
                    "weights_MB": s["weight_bytes"] / 1e6,
                    "trace_MB": s["trace_bytes"] / 1e6,
                    "activations_MB": s["activations_bytes"] / 1e6,
                    "total_MB": s["total_bytes"] / 1e6,
                    "flops_G": s["flops"] / 1e9,
                    "dominant": ("semantic_recall_N2"
                                 if s["activations_bytes"] > s["weight_bytes"] * 2
                                 else "weights"),
                }
                print(f"{name:>5} N={N:>6}: total={s['total_bytes']/1e6:8.1f}MB "
                      f"(w={s['weight_bytes']/1e6:6.1f} tr={s['trace_bytes']/1e6:7.1f} "
                      f"act={s['activations_bytes']/1e6:9.1f}) "
                      f"FLOPs={s['flops']/1e9:.2f}G")
            except (RuntimeError, MemoryError) as e:
                matrix[name][str(N)] = {"error": str(e)[:80]}
                print(f"{name:>5} N={N:>6}: OOM")
                gc.collect()

    # 1B y 300M*: extrapolacion matematica (param ~ d^2). Usamos el ajuste de la
    # prueba 6 si esta disponible; si no, una regla d^2.
    # Parametros empiricos para V5.5 base: P ~ k * d^2. Estimamos k con los puntos.
    ds = np.array([256, 384, 640], float)
    ps = []
    for kw in [m[1] for m in meas]:
        cfg = V55Config(vocab_size=256, context_length=512, **kw)
        mdl = EngraModelV55(cfg); ps.append(mdl.num_parameters()); del mdl; gc.collect()
    ps = np.array(ps, float)
    k = float(np.mean(ps / ds ** 2))
    extrap = {}
    for label, P in [("300M", 3e8), ("1B", 1e9), ("10B", 1e10), ("100B", 1e11)]:
        d = (P / k) ** 0.5
        d_ff, d_gate = 4 * d, max(8, d // 8)
        dk = min(128, max(64, d // 8))
        bpt = (2 * d + 2 * dk + dk) * 4
        for N in contexts:
            w = 4 * P
            tr = N * bpt
            act = (N * N * (2 * dk) + 9 * 4 * N * d) * 4
            extrap[f"{label}_N{N}"] = {
                "params": P, "d_model_est": d,
                "weights_GB": w / 1e9,
                "trace_GB": tr / 1e9,
                "activations_GB": act / 1e9,
                "total_GB": (w + tr + act) / 1e9,
                "note": "estimacion matematica (no ejecutado)",
            }

    result = {
        "test": "7_context_x_params",
        "k_param_per_d2": k,
        "measured_matrix": matrix,
        "extrapolated": extrap,
    }
    path = save_json("7_context_x_params.json", result)
    print("\nExtrapolacion (GB):")
    for key, v in list(extrap.items())[:8]:
        print(f"  {key:>14}: total={v['total_GB']:8.2f}GB "
              f"(w={v['weights_GB']:6.2f} tr={v['trace_GB']:6.2f} "
              f"act={v['activations_GB']:8.2f}) d~{v['d_model_est']:.0f}")
    print("Guardado:", path)


if __name__ == "__main__":
    run()
