"""Prueba 6 — Scaling de parametros sin entrenamiento.

Instanciamos modelos ENGRAMA V5.5 de tamanos crecientes (sin entrenar) y medimos:
  - numero real de parametros
  - memoria de parametros (FP32)
  - memoria de buffers
  - memoria de la traza (a contexto fijo)
  - tiempo de forward (lote 1, secuencia corta)
  - tiempo por token incremental
  - extrapolacion analitica de memoria a 1B/10B/100B parametros.

En CPU/3.8GB no instanciamos mas alla de ~100M; el resto es extrapolacion.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import hr_bytes, save_json

from engrama.v55.config import V55Config
from engrama.v55.model import EngraModelV55


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes32 = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return total, trainable, bytes32, buffers


def make_config(target_params_M: float, ctx: int = 512) -> V55Config:
    """Mapea un tamano de parametros objetivo a una config V5.5.

    Los parametros dominantes en V5.5 son:
      - embeddings: V*d
      - encoder: ~ L_enc * (4*d*dg + 2*d*dff/rank etc)
      - consolidation: L capas, cada una ~ 2*d*dg + P*(2*dg*d + 2*d*rank) + 2*d*dff
      - recall: ~4*d*dk + 2*d*dsem + 2*d*d (w_read)
      - evoker/tied embeddings
    Ajustamos d (y d_ff=4d, dg=d/8, rank=min(32,d/4)) para acercarnos al objetivo.
    """
    P = target_params_M * 1e6
    V = 256
    # busqueda ingenua de d
    best = None
    for d in [32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 960, 1280]:
        d_ff = 4 * d
        d_gate = max(8, d // 8)
        rank = min(32, max(8, d // 4))
        n_enc = 2
        n_con = 9
        cfg = V55Config(
            vocab_size=V, d_model=d, d_gate=d_gate, d_ff=d_ff,
            num_cells=max(2, d // 32), num_encoder_layers=n_enc,
            num_consolidation_layers=n_con, context_length=ctx,
            synapse_rank=rank, recall_enabled=True, d_recall=min(64, d),
            d_sense=min(64, d), d_semantic=min(64, d), page_size=256,
        )
        m = EngraModelV55(cfg)
        n = m.num_parameters()
        del m; gc.collect()
        if best is None or abs(n - P) < abs(best[0] - P):
            best = (n, cfg)
        if n > P * 1.5:
            break
    return best[1]


def measure(cfg: V55Config, seq_len: int = 64):
    torch.set_num_threads(2)
    model = EngraModelV55(cfg).eval()
    total, trainable, bytes32, buffers = count_params(model)
    # forward paralelo
    x = torch.randint(0, cfg.vocab_size, (1, seq_len))
    with torch.no_grad():
        model(x)  # warmup
    best = 1e9
    for _ in range(2):
        t = time.perf_counter()
        with torch.no_grad():
            y = model(x)
        best = min(best, time.perf_counter() - t)
    fwd_s = best
    # incremental (un token tras precargar cache con seq_len//2)
    cache = model.get_cache(n_max=cfg.context_length)
    with torch.no_grad():
        for t in range(seq_len // 2):
            tok = torch.tensor([[x[0, t]]])
            model.step_forward(tok, cache, t)
    n_inc = 16
    t = time.perf_counter()
    with torch.no_grad():
        for k in range(n_inc):
            tok = torch.tensor([[int(x[0, (seq_len // 2 + k) % seq_len])]])
            model.step_forward(tok, cache, seq_len // 2 + k)
    inc_s = (time.perf_counter() - t) / n_inc
    # memoria de traza
    trace_bytes = cache.memory_bytes()
    res = {
        "params": total, "trainable": trainable,
        "param_bytes_fp32": bytes32, "buffer_bytes": buffers,
        "trace_bytes": trace_bytes,
        "fwd_s_seq64": fwd_s, "fwd_ms_seq64": fwd_s * 1000,
        "inc_s_per_token": inc_s, "inc_ms_per_token": inc_s * 1000,
        "d_model": cfg.d_model, "d_ff": cfg.d_ff, "d_gate": cfg.d_gate,
        "num_con_layers": cfg.num_consolidation_layers,
        "synapse_rank": cfg.synapse_rank,
    }
    del model, cache, x, y; gc.collect()
    return res


def run():
    torch.set_num_threads(2)
    targets_M = [1, 5, 10, 20, 50, 100]
    rows = []
    print(f"{'target_M':>9} {'real_params':>12} {'param_MB':>10} {'trace_KB':>9} "
          f"{'fwd_ms':>8} {'inc_ms/tok':>10} {'d':>5}")
    for tm in targets_M:
        cfg = make_config(tm, ctx=512)
        r = measure(cfg)
        rows.append(r)
        print(f"{tm:>9} {r['params']:>12,} {r['param_bytes_fp32']/1e6:>10.1f} "
              f"{r['trace_bytes']/1024:>9.1f} {r['fwd_ms_seq64']:>8.1f} "
              f"{r['inc_ms_per_token']:>10.2f} {r['d_model']:>5}")

    # extrapolacion: parametros escala ~ d^2 (consolidation/encoder dominan con
    # d_ff=4d, P=const, L=const). Embeddings ~ V*d (lineal, despreciable a gran d).
    # Ajustamos P(d) = a*d^2 + b*d + c con los puntos medidos.
    import numpy as np
    ds = np.array([r["d_model"] for r in rows], float)
    ps = np.array([r["params"] for r in rows], float)
    c2 = np.polyfit(ds, ps, 2)  # a d^2 + b d + c
    # a bytes FP32 = 4 * params. A 1B,10B,100B
    def d_for_params(P):
        # resuelve a d^2 + b d + (c-P)=0
        a, b, c = c2
        disc = b**2 - 4*a*(c - P)
        return (-b + disc**0.5) / (2*a)
    extrap = {}
    for label, P in [("1B", 1e9), ("10B", 1e10), ("100B", 1e11)]:
        d_est = d_for_params(P)
        bytes32 = 4 * P
        # memoria fp16/bf16 = 2*P; + optimizador Adam (8 bytes/param entrenando)
        extrap[label] = {
            "params": P,
            "estimated_d_model": float(d_est),
            "weights_fp32_GB": bytes32 / 1e9,
            "weights_fp16_GB": 2 * P / 1e9,
            "train_adam_fp32_GB": (4 + 4 + 4 + 8) * P / 1e9,  # grad+mom+var+master
            "note": "Sin activadores/traza. Entrenar requiere ademas activaciones y optimizer states.",
        }
    result = {
        "test": "6_param_scaling", "device": "cpu",
        "rows": rows,
        "param_fit_d2": {"a": float(c2[0]), "b": float(c2[1]), "c": float(c2[2])},
        "extrapolation": extrap,
    }
    path = save_json("6_param_scaling.json", result)
    print("\nExtrapolacion:")
    for k, v in extrap.items():
        print(f"  {k:>5}: d~{v['estimated_d_model']:.0f}, "
              f"pesos fp32={v['weights_fp32_GB']:.1f}GB, "
              f"fp16={v['weights_fp16_GB']:.1f}GB, "
              f"entrenar(Adam fp32)={v['train_adam_fp32_GB']:.1f}GB")
    print("Guardado:", path)


if __name__ == "__main__":
    run()
