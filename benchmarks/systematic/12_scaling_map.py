"""Prueba 12 — Scaling map (graficos SVG nativos, sin matplotlib).

Genera:
  - fig_trace.png-equivalent.svg: tiempo build/read de la traza vs N
  - fig_consolidation.svg
  - fig_semantic_recall.svg: tiempo dense vs LSH y accuracy vs N
  - fig_surface.svg: superficie N x d_sem -> accuracy LSH
  - fig_params.svg: memoria de parametros vs N de params
  - fig_context_params.svg: memoria total por (P,N)
  - summary.json: metricas clave consolidadas
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True, parents=True)


def load(name):
    return json.load(open(RES / f"{name}.json"))


# ---------- mini motor SVG de graficos de lineas ----------
def line_chart(filename, title, xlabel, ylabel, series, xlog=True, ylog=True,
              width=760, height=420):
    """series: lista de dict(name, xs, ys, color)."""
    ml, mr, mt, mb = 90, 30, 50, 70
    pw, ph = width - ml - mr, height - mt - mb
    allx = [v for s in series for v in s["xs"]]
    ally = [v for s in series for v in s["ys"] if v > 0]
    xmin, xmax = min(allx), max(allx)
    ymin = min(ally) * 0.8
    ymax = max(ally) * 1.2
    def x2px(x):
        import math
        if xlog:
            l0, l1 = math.log10(xmin), math.log10(xmax)
            return ml + pw * (math.log10(x) - l0) / (l1 - l0)
        return ml + pw * (x - xmin) / (xmax - xmin)
    def y2px(y):
        import math
        if ylog and y > 0:
            l0, l1 = math.log10(ymin), math.log10(ymax)
            return mt + ph * (1 - (math.log10(y) - l0) / (l1 - l0))
        return mt + ph * (1 - (y - ymin) / (ymax - ymin))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'font-family="Helvetica,Arial,sans-serif" style="background:#fff">']
    svg.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18" font-weight="bold">{title}</text>')
    # ejes
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#333"/>')
    svg.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333"/>')
    # ticks X
    import math
    xticks = _log_ticks(xmin, xmax) if xlog else _lin_ticks(xmin, xmax, 6)
    for tv in xticks:
        if tv < xmin or tv > xmax:
            continue
        px = x2px(tv)
        svg.append(f'<line x1="{px:.1f}" y1="{mt+ph}" x2="{px:.1f}" y2="{mt+ph+5}" stroke="#333"/>')
        svg.append(f'<text x="{px:.1f}" y="{mt+ph+20}" text-anchor="middle" font-size="11">{_fmt(tv)}</text>')
    yticks = _log_ticks(ymin, ymax) if ylog else _lin_ticks(ymin, ymax, 6)
    for tv in yticks:
        if tv < ymin or tv > ymax:
            continue
        py = y2px(tv)
        svg.append(f'<line x1="{ml-5}" y1="{py:.1f}" x2="{ml}" y2="{py:.1f}" stroke="#333"/>')
        svg.append(f'<text x="{ml-8}" y="{py+4:.1f}" text-anchor="end" font-size="11">{_fmt(tv)}</text>')
    svg.append(f'<text x="{ml+pw/2}" y="{height-25}" text-anchor="middle" font-size="13">{xlabel}</text>')
    svg.append(f'<text x="20" y="{mt+ph/2}" text-anchor="middle" font-size="13" '
               f'transform="rotate(-90 20,{mt+ph/2})">{ylabel}</text>')
    # lineas
    for s in series:
        pts = []
        for x, y in zip(s["xs"], s["ys"]):
            if y is None or y <= 0:
                continue
            pts.append(f"{x2px(x):.1f},{y2px(y):.1f}")
        if pts:
            svg.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                       f'stroke="{s["color"]}" stroke-width="2.2"/>')
            for x, y in zip(s["xs"], s["ys"]):
                if y is None or y <= 0:
                    continue
                svg.append(f'<circle cx="{x2px(x):.1f}" cy="{y2px(y):.1f}" r="3" fill="{s["color"]}"/>')
    # leyenda
    ly = mt + 10
    for i, s in enumerate(series):
        y = ly + i * 20
        svg.append(f'<rect x="{ml+pw-150}" y="{y-10}" width="14" height="14" fill="{s["color"]}"/>')
        svg.append(f'<text x="{ml+pw-130}" y="{y+2}" font-size="12">{s["name"]}</text>')
    svg.append("</svg>")
    (FIG / filename).write_text("\n".join(svg), encoding="utf-8")


def _log_ticks(lo, hi):
    import math
    out = []
    a = math.floor(math.log10(lo))
    b = math.ceil(math.log10(hi))
    for e in range(a, b + 1):
        for m in (1, 2, 5):
            out.append(m * 10 ** e)
    return out


def _lin_ticks(lo, hi, k):
    import math
    step = (hi - lo) / k
    return [lo + i * step for i in range(k + 1)]


def _fmt(v):
    if v >= 1e9:
        return f"{v/1e9:.0f}G"
    if v >= 1e6:
        return f"{v/1e6:.0f}M"
    if v >= 1e3:
        return f"{v/1e3:.0f}K"
    if v >= 1:
        return f"{v:.0f}"
    return f"{v:.2g}"


def heatmap_svg(filename, title, xvals, yvals, matrix, xlabel, ylabel,
                fmt=lambda v: f"{v:.2f}", width=720, height=420, vmax=None):
    """matrix: dict (x,y) -> valor."""
    ml, mr, mt, mb = 100, 30, 60, 80
    cw = (width - ml - mr) / len(xvals)
    ch = (height - mt - mb) / len(yvals)
    vmax = vmax or max(abs(v) for v in matrix.values())
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'font-family="Helvetica,Arial,sans-serif" style="background:#fff">']
    svg.append(f'<text x="{width/2}" y="30" text-anchor="middle" font-size="17" font-weight="bold">{title}</text>')
    for iy, yv in enumerate(yvals):
        for ix, xv in enumerate(xvals):
            v = matrix.get((xv, yv))
            if v is None:
                continue
            t = max(0, min(1, v / vmax))
            # rojo (bajo) -> amarillo -> verde (alto)
            r = int(220 * (1 - t) + 30 * t)
            g = int(60 * (1 - t) + 180 * t)
            b = int(60 * (1 - t) + 60 * t)
            x = ml + ix * cw
            y = mt + (len(yvals) - 1 - iy) * ch
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" height="{ch:.1f}" '
                       f'fill="rgb({r},{g},{b})" stroke="#fff" stroke-width="1"/>')
            svg.append(f'<text x="{x+cw/2:.1f}" y="{y+ch/2+4:.1f}" text-anchor="middle" '
                       f'font-size="11" fill="#000">{fmt(v)}</text>')
    for ix, xv in enumerate(xvals):
        svg.append(f'<text x="{ml+ix*cw+cw/2:.1f}" y="{mt+len(yvals)*ch+20}" text-anchor="middle" font-size="11">{_fmt(xv)}</text>')
    for iy, yv in enumerate(yvals):
        label = _fmt(yv) if isinstance(yv, (int, float)) else str(yv)
        svg.append(f'<text x="{ml-8}" y="{mt+(len(yvals)-1-iy)*ch+ch/2+4:.1f}" text-anchor="end" font-size="11">{label}</text>')
    svg.append(f'<text x="{ml+len(xvals)*cw/2}" y="{height-25}" text-anchor="middle" font-size="13">{xlabel}</text>')
    svg.append(f'<text x="25" y="{mt+len(yvals)*ch/2}" text-anchor="middle" font-size="13" '
               f'transform="rotate(-90 25,{mt+len(yvals)*ch/2})">{ylabel}</text>')
    svg.append("</svg>")
    (FIG / filename).write_text("\n".join(svg), encoding="utf-8")


def run():
    # 1. trace
    t1 = load("1_trace_scaling")
    line_chart("trace_time.svg", "Prueba 1 — Tiempo de traza vs N (CPU)",
               "N (tokens)", "segundos",
               [{"name": "construccion (append N)", "xs": [r["N"] for r in t1["rows"]],
                 "ys": [r["build_s"] for r in t1["rows"]], "color": "#1f77b4"},
                {"name": "lectura lineal", "xs": [r["N"] for r in t1["rows"]],
                 "ys": [r["read_s"] for r in t1["rows"]], "color": "#d62728"}])
    line_chart("trace_memory.svg", "Prueba 1 — Memoria de traza vs N",
               "N (tokens)", "bytes",
               [{"name": "bytes reales", "xs": [r["N"] for r in t1["rows"]],
                 "ys": [r["bytes_real"] for r in t1["rows"]], "color": "#2ca02c"},
                {"name": "bytes teoricos activos", "xs": [r["N"] for r in t1["rows"]],
                 "ys": [r["bytes_theoretical_active"] for r in t1["rows"]], "color": "#9467bd"}],
               ylog=True)

    # 2. consolidation
    t2 = load("2_consolidation_scaling")
    line_chart("consolidation.svg", "Prueba 2 — Tiempo de consolidacion vs N",
               "N (tokens)", "segundos",
               [{"name": "forward_train (medio)", "xs": [r["N"] for r in t2["rows"]],
                 "ys": [r["fwd_s"] for r in t2["rows"]], "color": "#ff7f0e"}])

    # 3 & 4. semantic recall dense vs lsh
    t5 = load("5_dense_vs_lsh")
    dense = [r for r in t5["rows"] if "dense_s" in r]
    line_chart("dense_vs_lsh_time.svg", "Prueba 5 — Dense vs LSH: tiempo",
               "N (tokens)", "segundos",
               [{"name": "dense O(N^2)", "xs": [r["N"] for r in dense],
                 "ys": [r["dense_s"] for r in dense], "color": "#d62728"},
                {"name": "LSH O(N*C)", "xs": [r["N"] for r in t5["rows"]],
                 "ys": [r["lsh_s"] for r in t5["rows"]], "color": "#1f77b4"}])
    line_chart("dense_vs_lsh_recall.svg", "Prueba 5 — Recall semantico",
               "N (tokens)", "Recall@1",
               [{"name": "dense", "xs": [r["N"] for r in dense],
                 "ys": [r["recall_dense"] for r in dense], "color": "#d62728"},
                {"name": "LSH", "xs": [r["N"] for r in t5["rows"]],
                 "ys": [r["recall_lsh"] for r in t5["rows"]], "color": "#1f77b4"}],
               ylog=False)

    # surface N x d -> LSH accuracy
    t4 = load("4_semantic_tap")
    Nvals = sorted(set(p["N"] for p in t4["surface"]))
    dvals = sorted(set(p["d"] for p in t4["surface"]))
    lsh_acc = {(p["N"], p["d"]): p["acc"] for p in t4["surface"] if p["mode"] == "lsh"}
    heatmap_svg("surface_lsh_acc.svg", "Prueba 4 — Accuracy LSH (superficie N x d)",
                Nvals, dvals, lsh_acc, "N (tokens)", "d_sem",
                fmt=lambda v: f"{v:.2f}", vmax=1.0)
    lsh_ms = {(p["N"], p["d"]): p["ms"] for p in t4["surface"] if p["mode"] == "lsh"}
    heatmap_svg("surface_lsh_ms.svg", "Prueba 4 — Latencia LSH ms (superficie N x d)",
                Nvals, dvals, lsh_ms, "N (tokens)", "d_sem",
                fmt=lambda v: f"{v:.0f}")

    # 6. params
    t6 = load("6_param_scaling")
    line_chart("params_memory.svg", "Prueba 6 — Memoria de pesos vs parametros",
               "parametros", "bytes FP32",
               [{"name": "pesos medidos", "xs": [r["params"] for r in t6["rows"]],
                 "ys": [r["param_bytes_fp32"] for r in t6["rows"]], "color": "#1f77b4"},
                {"name": "4 * params (teorico)", "xs": [r["params"] for r in t6["rows"]],
                 "ys": [4 * r["params"] for r in t6["rows"]], "color": "#2ca02c"}],
               xlog=True, ylog=True)

    # 7. context x params heatmap total_MB
    t7 = load("7_context_x_params")
    Pvals = list(t7["measured_matrix"].keys())
    Cvals = sorted(int(c) for c in next(iter(t7["measured_matrix"].values())).keys())
    cell = {(int(C), P): t7["measured_matrix"][P][str(C)]["total_MB"]
            for P in Pvals for C in Cvals if "total_MB" in t7["measured_matrix"][P][str(C)]}
    heatmap_svg("context_x_params_MB.svg", "Prueba 7 — Memoria total MB (paralelo, recall denso)",
                Cvals, Pvals, cell, "N contexto", "modelo",
                fmt=lambda v: f"{v:.0f}")

    print("Graficos generados en", FIG)
    for f in sorted(os.listdir(FIG)):
        print("  ", f)


if __name__ == "__main__":
    run()
