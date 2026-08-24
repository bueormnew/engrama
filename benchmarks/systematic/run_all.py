"""Orquestador: ejecuta las pruebas 4..11 en un solo proceso (larga duracion).

Guarda resultados en results/ y registra el progreso en results/progress.log.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LOG = HERE / "results" / "progress.log"


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    open(LOG, "w").close()
    tests = [
        ("4_semantic_tap", "4_semantic_tap"),
        ("5_dense_vs_lsh", "5_dense_vs_lsh"),
        ("6_param_scaling", "6_param_scaling"),
        ("7_context_params", "7_context_params"),
        ("8_parallel_vs_incremental", "8_parallel_vs_incremental"),
        ("9_numeric_stress", "9_numeric_stress"),
        ("10_effective_memory", "10_effective_memory"),
        ("11_interference", "11_interference"),
    ]
    for name, mod in tests:
        log(f"=== INICIO {name} ===")
        t0 = time.perf_counter()
        try:
            m = __import__(mod)
            m.run()
            log(f"=== OK {name} ({time.perf_counter()-t0:.1f}s) ===")
        except Exception as e:
            log(f"!!! ERROR {name}: {e}")
            log(traceback.format_exc())
    log("TODAS LAS PRUEBAS TERMINADAS")


if __name__ == "__main__":
    main()
