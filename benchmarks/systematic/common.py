"""Utilidades compartidas para el benchmark sistematico de ENGRAMA V5.5.

Entorno: CPU, 2 nucleos, ~3.8 GB RAM. Todo se mide en CPU con torch.
Incluye:
  * cronometro de alta resolicion
  * medicion de memoria RSS (psutil) y de tensores
  * contabilidad de bytes
  * ajuste de modelos O(N), O(N log N), O(N^2) con R^2
"""
from __future__ import annotations

import gc
import json
import os
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------- entorno ----------
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def rss_mb() -> float:
    """RSS del proceso en MB (lectura de /proc/self/status, portable Linux)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return float("nan")


def torch_tensor_bytes() -> int:
    """Bytes que ocupan todos los tensors vivos de torch (approx)."""
    total = 0
    seen = set()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if id(obj) in seen:
                    continue
                seen.add(id(obj))
                total += obj.numel() * obj.element_size()
        except Exception:
            pass
    return total


@dataclass
class TimedResult:
    value: object
    seconds: float
    rss_mb: float
    tensor_bytes: int


def timeit(fn: Callable, *, warmup: int = 1, repeats: int = 3) -> TimedResult:
    """Ejecuta fn(), descarta warmups, mide el mejor tiempo (menos ruido en CPU)."""
    last = None
    for _ in range(warmup):
        last = fn()
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    times = []
    rss = []
    tb = []
    for _ in range(repeats):
        rss_before = rss_mb()
        t0 = time.perf_counter()
        last = fn()
        dt = time.perf_counter() - t0
        times.append(dt)
        rss.append(max(0.0, rss_mb() - rss_before))
        tb.append(torch_tensor_bytes())
    return TimedResult(
        value=last,
        seconds=float(min(times)),
        rss_mb=float(np.median(rss)),
        tensor_bytes=int(np.median(tb)),
    )


def tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


# ---------- ajuste de curvas ----------
def _poly_r2(x: np.ndarray, y: np.ndarray, deg: int) -> Tuple[np.ndarray, float]:
    c = np.polyfit(x, y, deg)
    yhat = np.polyval(c, x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return c, r2


def fit_scaling(N: Sequence[int], T: Sequence[float]) -> Dict[str, object]:
    """Ajusta lineal aN+b, n log N, y cuadratico aN^2+bN+c; reporta R^2.

    Para el cuadratico, los coeficientes se devuelven en escala del N tal cual,
    lo que puede ser numericamente dificil para N grande; se normaliza por
    N_max para el ajuste y se desnormaliza.
    """
    x = np.asarray(N, dtype=np.float64)
    y = np.asarray(T, dtype=np.float64)
    out: Dict[str, object] = {}

    # lineal
    c1, r1 = _poly_r2(x, y, 1)
    out["linear"] = {"a": float(c1[0]), "b": float(c1[1]), "r2": float(r1)}

    # n log n: y = a * N log2(N) + b
    xn = x * np.log2(np.maximum(x, 2))
    A = np.vstack([xn, np.ones_like(xn)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = sol[0] * xn + sol[1]
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_nlogn = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    out["nlogn"] = {"a": float(sol[0]), "b": float(sol[1]), "r2": float(r_nlogn)}

    # cuadratico (con escalado para estabilidad numerica)
    scale = float(np.max(x))
    xs = x / scale
    c2, r2 = _poly_r2(xs, y, 2)
    # desnormalizar: y = c2[0]*(N/s)^2 + c2[1]*(N/s) + c2[2]
    out["quadratic"] = {
        "a": float(c2[0] / scale ** 2),
        "b": float(c2[1] / scale),
        "c": float(c2[2]),
        "r2": float(r2),
    }
    # cual explica mejor?
    best = max(("linear", "nlogn", "quadratic"), key=lambda k: out[k]["r2"])
    out["best"] = best
    return out


def infer_exponent(N: Sequence[int], T: Sequence[float]) -> float:
    """Estima el exponente p asumiendo T ~ N^p via regresion log-log.
    Usa solo N>=128 para evitar ruido de pequena escala."""
    x = np.asarray(N, dtype=np.float64)
    y = np.asarray(T, dtype=np.float64)
    m = (x >= 128) & (y > 0)
    if m.sum() < 3:
        m = y > 0
    lx, ly = np.log2(x[m]), np.log2(y[m])
    A = np.vstack([lx, np.ones_like(lx)]).T
    sol, *_ = np.linalg.lstsq(A, ly, rcond=None)
    return float(sol[0])


def save_json(name: str, obj) -> str:
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (torch.dtype,)):
        return str(o)
    return str(o)


def hr_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"
