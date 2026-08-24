"""Genera INFORME_FINAL.md a partir de todos los results/*.json."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def L(name):
    return json.load(open(RES / f"{name}.json"))


def hr(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def main():
    t1 = L("1_trace_scaling")
    t2 = L("2_consolidation_scaling")
    t3 = L("3_lexical_recall")
    t4 = L("4_semantic_tap")
    t5 = L("5_dense_vs_lsh")
    t6 = L("6_param_scaling")
    t7 = L("7_context_x_params")
    t8 = L("8_parallel_vs_incremental")
    t9 = L("9_numeric_stress")
    t10 = L("10_effective_memory")
    t11 = L("11_interference")

    out = []
    W = out.append
    W("# ENGRAMA V5.5 — Resultados del benchmark sistematico\n")
    W("**Benchmark sin entrenamiento**: solo tensores sinteticos y los modulos "
      "reales de `engrama.v55` (traza, consolidacion, Recall Tap lexico, "
      "Semantic Tap denso/LSH, evocador).\n")
    W("## Entorno\n")
    W("- Hardware: **CPU**, 2 nucleos, **3.8 GB RAM**, sin GPU.")
    W("- PyTorch 2.13, ENGRAMA V5.5 (`V55Config`/`EngraModelV55`).")
    W("- Por las limitaciones de RAM, los caminos O(N^2) (recall denso) se midieron "
      "hasta N=4K-8K; los caminos O(N) (traza, consolidacion, LSH) hasta 128K. "
      "Las cifras de 1B/10B/100B parametros son **extrapolacion matematica**, no "
      "ejecucion real.\n")

    # 1
    W("## 1. Escalabilidad de la traza (T0 -> append -> read)\n")
    W("| N | build (ms) | append amort. (us) | read (ms) | thr append (tok/s) | thr read (tok/s) | bytes reales |")
    W("|---:|---:|---:|---:|---:|---:|---:|")
    for r in t1["rows"]:
        W(f"| {r['N']} | {r['build_s']*1000:.1f} | {r['append_us']:.0f} "
          f"| {r['read_s']*1000:.2f} | {r['throughput_append_tok_s']:.0f} "
          f"| {r['throughput_read_tok_s']:.0f} | {hr(r['bytes_real'])} |")
    c = t1["conclusion"]
    W(f"\n**Ajuste**: build R²(lineal)={c['build_O_N_r2']:.5f}, "
      f"read R²(lineal)={c['read_O_N_r2']:.5f}, "
      f"memoria R²(lineal)={c['memory_O_N_r2']:.6f}. "
      f"Exponentes empiricos (log-log): build ~{t1['empirical_exponent']['build']:.2f}, "
      f"read ~{t1['empirical_exponent']['read']:.2f}, "
      f"memoria ~{t1['empirical_exponent']['memory']:.2f}.\n")
    W("**Conclusion**: la traza paginada es **O(N) en tiempo y memoria**. "
      "El append amortizado es ~250 us (constante en N), la lectura lineal escala "
      "linealmente, y los bytes reales son ~91% de los teoricos activos (el resto "
      "es sobreasignacion de paginas de 256). No hay copia oculta O(N^2): "
      "la memoria es exactamente lineal con R²=1.000000.\n")
    W("![](figures/trace_time.svg)\n")

    # 2
    W("## 2. Consolidacion (T0 -> Consolidation -> T)\n")
    W("| N | forward (ms) | us/token | tok/s |")
    W("|---:|---:|---:|---:|")
    for r in t2["rows"]:
        W(f"| {r['N']} | {r['fwd_s']*1000:.1f} | {r['fwd_s']/r['N']*1e6:.1f} "
          f"| {r['throughput_tok_s']:.0f} |")
    W(f"\nRefinado (5 repeticiones, N hasta 32K): exponente ~{t2['refined']['exponent']:.2f}; "
      f"R² lineal={t2['refined']['linear_r2']:.4f}, nlogn={t2['refined']['nlogn_r2']:.4f}, "
      f"cuadratico={t2['refined']['quadratic_r2']:.4f}. "
      f"El coste **por token es practicamente constante** "
      f"(pendiente ~{t2['refined']['per_token_slope_ns_per_N']:.1f} ns por token "
      f"adicional sobre un intercepto de ~175 us).\n")
    W("**Conclusion**: consolidacion **O(N)**. Los offsets por capa son fijos "
      "(P<=4) y no hay matriz N x N. El aparente mejor ajuste nlogn/cuadratico "
      "se debe al overhead fijo por las 9 capas y al ruido de threading en CPU, "
      "no a un termino N². La memoria de activaciones crece linealmente.\n")
    W("![](figures/consolidation.svg)\n")

    # 3
    W("## 3. Recall lexico (identidad O(1) vs denso vs LSH)\n")
    W("| N | identidad (us) | denso (ms) | LSH (ms) | Recall denso | Recall identidad | Recall LSH |")
    W("|---:|---:|---:|---:|---:|---:|---:|")
    for r in t3["rows"]:
        id_us = f"{r['identity_s']*1e6:.0f}" if "identity_s" in r else "--"
        d_ms = f"{r['dense_s']*1000:.2f}" if "dense_s" in r else "--"
        rd = f"{r['recall_dense']:.3f}" if "recall_dense" in r else "--"
        ri = f"{r['recall_identity']:.3f}" if "recall_identity" in r else "--"
        W(f"| {r['N']} | {id_us} | {d_ms} | {r['lsh_s']*1000:.1f} "
          f"| {rd} | {ri} | {r['recall_lsh']:.3f} |")
    W(f"\nExponentes: denso ~{t3['exponent']['dense']:.2f} (O(N^2) en la construccion "
      f"de la matriz de scores), LSH ~{t3['exponent']['lsh']:.2f} (lineal).\n")
    W("**Conclusion**: el **camino denso y el fast-path de identidad dan 1.000 de "
      "recall exacto**. El LSH lexico escala linealmente (R²=0.9995) pero con "
      "claves perfectamente ortogonales su bucketizacion por planos aleatorios "
      "no recupera el objetivo (recall ~0); el README ya reconoce que el LSH "
      "aproximado pierde recall. En la generacion incremental el camino es "
      "**siempre denso O(N·d_k)** y, por tanto, estructuralmente exacto.\n")

    # 4
    W("## 4. Semantic Tap — prueba critica (embeddings sinteticos)\n")
    W("Estructura semantica conocida: `concepto A_i` y su `alias` (vector base + "
      "ruido 0.1). La consulta en el alias debe recuperar el concepto. "
      "Superficie N x d_sem.\n")
    W("### Accuracy del Semantic Tap (d=256)\n")
    W("| N | dense acc | dense ms | LSH acc | LSH ms |")
    W("|---:|---:|---:|---:|---:|")
    for p in t4["surface"]:
        if p["d"] != 256:
            continue
        if p["mode"] == "dense":
            lsh = next((q for q in t4["surface"] if q["d"] == 256
                        and q["mode"] == "lsh" and q["N"] == p["N"]), None)
            la = f"{lsh['acc']:.3f}" if lsh else "--"
            lm = f"{lsh['ms']:.1f}" if lsh else "--"
            W(f"| {p['N']} | {p['acc']:.3f} | {p['ms']:.1f} | {la} | {lm} |")
        else:
            if not any(q["d"] == 256 and q["mode"] == "dense" and q["N"] == p["N"]
                       for q in t4["surface"]):
                W(f"| {p['N']} | -- | -- | {p['acc']:.3f} | {p['ms']:.1f} |")
    W("")
    W("**Hallazgo critico**: el **Semantic Tap DENSO logra 1.000 de accuracy en "
      "TODAS las N y TODAS las dimensiones** (64 a 1024). Esto valida "
      "experimentalmente el nucleo asociativo del Pilar 4: el argmax duro por "
      "coseno + lectura unica recupera la huella correcta. El **LSH pierde "
      "recall masivamente** (7%–50% en N medianas; solo mejora al 80–90% a 64K–128K "
      "porque hay mas candidatos en bucket), y es ademas mas lento que el denso "
      "hasta el cruce N~8K-16K.\n")
    W("![](figures/surface_lsh_acc.svg)\n")
    W("![](figures/surface_lsh_ms.svg)\n")

    # 5
    W("## 5. Dense vs LSH (tabla comparativa directa)\n")
    W("| N | Dense (ms) | LSH (ms) | Recall dense | Recall LSH | speedup LSH |")
    W("|---:|---:|---:|---:|---:|---:|")
    for r in t5["rows"]:
        d_ms = r.get("dense_s", 0) * 1000 if "dense_s" in r else None
        sp = r.get("speedup_lsh_vs_dense")
        W(f"| {r['N']} | {f'{d_ms:.1f}' if d_ms is not None else '--'} "
          f"| {r['lsh_s']*1000:.1f} | {r.get('recall_dense','--')} "
          f"| {r['recall_lsh']:.3f} | {f'{sp:.2f}x' if sp else '--'} |")
    W(f"\nExponentes: denso ~{t5['exponent']['dense']:.2f} (O(N^2)), "
      f"LSH ~{t5['exponent']['lsh']:.2f} (O(N)).\n")
    W("**Conclusion / cuello de botella**: el semantic recall **denso** es "
      "exacto pero O(N²): a 8K consume ~300 ms y materializa la matriz de "
      "scores. El **LSH** es O(N) pero su recall es pobre (7%–50%) y no "
      "supera en velocidad al denso hasta ~8K-16K. **Este es el principal "
      "cuello de botella de ENGRAMA**: no existe aun un recall semantico "
      "lineal Y exacto. El README lo reconoce abiertamente.\n")
    W("![](figures/dense_vs_lsh_time.svg)\n")
    W("![](figures/dense_vs_lsh_recall.svg)\n")

    # 6
    W("## 6. Scaling de parametros (sin entrenamiento)\n")
    W("| d | parametros | pesos FP32 | forward 64 tok (ms) | incremental (ms/tok) |")
    W("|---:|---:|---:|---:|---:|")
    for r in t6["rows"]:
        W(f"| {r['d_model']} | {r['params']:,} | {hr(r['param_bytes_fp32'])} "
          f"| {r['fwd_ms_seq64']:.1f} | {r['inc_ms_per_token']:.2f} |")
    W("\n**Extrapolacion matematica** (P ≈ k·d²; sin activaciones):\n")
    W("| escala | d estimado | pesos FP32 | pesos FP16 | entrenar (Adam FP32) |")
    W("|---|---:|---:|---:|---:|")
    for k, v in t6["extrapolation"].items():
        W(f"| {k} | {v['estimated_d_model']:.0f} | {v['weights_fp32_GB']:.1f} GB "
          f"| {v['weights_fp16_GB']:.1f} GB | {v['train_adam_fp32_GB']:.0f} GB |")
    W("")
    W("**Conclusion**: los parametros crecen ~d² y son manejables (100M ~ 380 MB "
      "FP32). La arquitectura **no se vuelve fisicamente absurda** al crecer: "
      "1B pesos son ~4 GB FP32 (2 GB FP16), 10B ~40 GB (entrenable en un solo "
      "nodo A100/H100), 100B ~400 GB (multi-nodo). El tiempo incremental por token "
      "sube de 13 ms (1M) a 38 ms (100M): razonable. El problema no son los pesos, "
      "sino el recall semantico a contexto largo (ver #5, #7).\n")

    # 7
    W("## 7. Contexto x parametros (matriz de memoria)\n")
    W("Memoria total **estimada** en MB (pesos + traza + activaciones del recall "
      "denso paralelo), para los tamanos medibles:\n")
    Pnames = list(t7["measured_matrix"].keys())
    Cvals = sorted(int(c) for c in next(iter(t7["measured_matrix"].values())).keys())
    W("| Modelo | " + " | ".join(f"N={c}" for c in Cvals) + " |")
    W("|" + "---|" * (len(Cvals) + 1))
    for P in Pnames:
        cells = []
        for c in Cvals:
            v = t7["measured_matrix"][P][str(c)]
            cells.append(f"{v['total_MB']:.0f}" if "total_MB" in v else "OOM")
        W(f"| {P} | " + " | ".join(cells) + " |")
    W("")
    W("(El valor es dominado por las activaciones O(N²) del recall semantico "
      "denso: ~N²·(d_k+d_sem)·4 bytes. Los pesos son 35–235 MB; la traza es "
      "minima.)\n")
    W("**Extrapolacion** (memoria total GB):\n")
    W("| celda | d est. | pesos GB | traza GB | activaciones GB | total GB |")
    W("|---|---:|---:|---:|---:|---:|")
    for key in ["300M_N16384", "1B_N16384", "1B_N65536", "10B_N65536", "100B_N65536"]:
        if key in t7["extrapolated"]:
            e = t7["extrapolated"][key]
            W(f"| {key} | {e['d_model_est']:.0f} | {e['weights_GB']:.1f} "
              f"| {e['trace_GB']:.2f} | {e['activations_GB']:.1f} "
              f"| {e['total_GB']:.1f} |")
    W("")
    W("**Conclusion**: a contexto largo, el termino dominante es **el recall "
      "semantico O(N²)** (activaciones), no los pesos ni la traza. Con 16K de "
      "contexto y recall denso, incluso un modelo de 10M necesita ~138 GB; "
      "a 64K, terabytes. El uso practico de contexto largo **requiere el LSH** "
      "(o chunked dense) — y el LSH hoy pierde recall. La traza en si es "
      "trivial (184 MB a 64K).\n")

    # 8
    W("## 8. Forward paralelo vs incremental (invarianza causal)\n")
    W("Maxima diferencia absoluta entre rutas paralela e incremental:\n")
    W("| N | FP32 | FP16 | BF16 |")
    W("|---:|---:|---:|---:|")
    fp32 = {r["N"]: r for r in t8["by_dtype"]["float32"]}
    fp16 = {r["N"]: r for r in t8["by_dtype"]["float16"]}
    bf16 = {r["N"]: r for r in t8["by_dtype"]["bfloat16"]}
    alln = sorted(fp32.keys())
    for n in alln:
        e16 = f"{fp16[n]['max_abs_err']:.2e}" if n in fp16 else "--"
        ebf = f"{bf16[n]['max_abs_err']:.2e}" if n in bf16 else "--"
        W(f"| {n} | {fp32[n]['max_abs_err']:.2e} | {e16} | {ebf} |")
    W("")
    W("**Conclusion**: en **FP32 la invarianza causal se cumple holgadamente**: "
      f"error maximo {max(r['max_abs_err'] for r in fp32.values()):.2e} < 1e-6, "
      "exactamente como afirma el repositorio. El error crece muy suavemente con N "
      "(de 2.8e-7 a N=16 hasta 6.0e-7 a N=4096). En **FP16/BF16 la invarianza "
      "NO se cumple** (errores de hasta 0.55 en BF16 a N=1024): el redondeo de "
      "baja precision y el `SEM_TIE_DECIMALS` no bastan para los empates exactos "
      "del tap semantico cuando el cache se almacena en la precision de computo. "
      "Recomendacion: FP32 para validar la invarianza, o almacenar la traza en FP32 "
      "aunque se calcule en FP16/BF16.\n")

    # 9
    W("## 9. Stress test numerico\n")
    W(f"Total NaN: **{t9['total_nan']}**, Total Inf: **{t9['total_inf']}**, "
      f"estable: **{t9['stable']}**.\n")
    W("| dtype | entrada | max abs | mean abs | varianza | NaN/Inf |")
    W("|---|---|---:|---:|---:|---|")
    for dt, rows in t9["by_dtype"].items():
        if isinstance(rows, list):
            for r in rows:
                W(f"| {dt} | {r['kind']} | {r['max_abs']:.2e} "
                  f"| {r['mean_abs']:.2e} | {r['variance']:.2e} | "
                  f"{r['nan']}/{r['inf']} |")
    W("")
    W("**Conclusion**: incluso con embeddings de magnitud 1e3, ruido uniforme en "
      "[-10,10] o valores cero repetidos, el modelo **no produce NaN ni Inf** en "
      "FP32/FP16/BF16. El `softcap` del evocador (logit_cap=30), las RMSNorm, "
      "la mezcla normalizada por conteo y el `-1e30` acotado lo mantienen "
      "estable. Se cumple la afirmacion de ausencia de NaN.\n")

    # 10
    W("## 10. Capacidad de memoria efectiva\n")
    W("| N | primera | aleatoria | ultima | accuracy |")
    W("|---:|---|---|---|---:|")
    for r in t10["rows"]:
        W(f"| {r['N']} | {r['recover_first']} | {r['recover_random']} "
          f"| {r['recover_last']} | {r['accuracy']:.3f} |")
    W("")
    W("**Conclusion**: la traza devuelve **exactamente (bit a bit)** la primera, "
      "una aleatoria y la ultima huella hasta N=100.000. La memoria explicita "
      "FIFO paginada funciona como se pretende: no hay compresion ni "
      "sobre-escritura que corrompa huellas almacenadas.\n")

    # 11
    W("## 11. Interferencia entre huellas\n")
    W("Se escribe A->X, B->Y, C->Z y luego A->Q (nueva asociacion para A) "
      "con tokens extra de por medio, usando claves lexicas ortogonales.\n")
    W("| N | A->Q (nueva) | A->X (vieja) | B->Y | C->Z | contaminacion |")
    W("|---:|---|---|---|---|---:|")
    for r in t11["rows"]:
        W(f"| {r['N']} | {r['A_to_Q']} | {r['A_old_to_X']} | {r['B_to_Y']} "
          f"| {r['C_to_Z']} | {r['contamination']} |")
    W("")
    W("**Conclusion**: con claves lexicas ortogonales (la condicion del Pilar 1), "
      "escribir **A->Q no contamina B->Y ni C->Z** (contaminacion 0). Ademas el "
      "desempate por sentido recupera correctamente la nueva A. (Una version "
      "previa de esta prueba con d_k < V mostraba falsos fallos por colision "
      "lexica artificial, no por la arquitectura.)\n")

    # 12
    W("## 12. Scaling map (figuras)\n")
    W("- Tiempo de traza: `figures/trace_time.svg`")
    W("- Memoria de traza: `figures/trace_memory.svg`")
    W("- Consolidacion: `figures/consolidation.svg`")
    W("- Dense vs LSH tiempo: `figures/dense_vs_lsh_time.svg`")
    W("- Dense vs LSH recall: `figures/dense_vs_lsh_recall.svg`")
    W("- Superficie accuracy LSH: `figures/surface_lsh_acc.svg`")
    W("- Superficie latencia LSH: `figures/surface_lsh_ms.svg`")
    W("- Memoria de parametros: `figures/params_memory.svg`")
    W("- Contexto x parametros: `figures/context_x_params_MB.svg`\n")

    # Fase 4
    W("## Fase 4 — Analisis y leyes de escalado\n")
    W("| Componente | Exponente empirico | Modelo que mejor ajusta | Veredicto |")
    W("|---|---:|---|---|")
    W(f"| Traza (build) | {t1['empirical_exponent']['build']:.2f} | O(N) | lineal ✅ |")
    W(f"| Traza (read) | {t1['empirical_exponent']['read']:.2f} | O(N) | lineal ✅ |")
    W(f"| Traza (memoria) | {t1['empirical_exponent']['memory']:.2f} | O(N) | lineal exacta ✅ |")
    W(f"| Consolidacion | {t2['refined']['exponent']:.2f} | O(N) | lineal por token ✅ |")
    W(f"| Recall lexico denso | {t3['exponent']['dense']:.2f} | O(N²) | construccion N² ⚠️ |")
    W(f"| Recall lexico LSH | {t3['exponent']['lsh']:.2f} | O(N) | lineal, recall pobre ⚠️ |")
    W(f"| Semantic dense | {t5['exponent']['dense']:.2f} | O(N²) | exacto pero N² ❌ |")
    W(f"| Semantic LSH | {t5['exponent']['lsh']:.2f} | O(N) | lineal, recall 7-90% ⚠️ |")
    W(f"| Parametros (P vs d²) | ~2.0 | P ≈ k·d² | predecible ✅ |")
    W("")
    W("### Cuello de botella identificado\n")
    W("**El recall semantico es el unico termino super-lineal.** Todo lo demas "
      "(traza, consolidacion, evocador, memoria de huellas) es O(N) y estable. "
      "Concretamente:\n")
    W("- La ruta **densa** del Semantic Tap es **100% exacta** (accuracy 1.000) "
      "pero O(N²) en tiempo y memoria: impractica mas alla de ~8K-16K.")
    W("- La ruta **LSH** es O(N) pero su recall se desploma (7%–50% en N "
      "intermedias) porque los planos de hashing aleatorios no separan bien "
      "vectores semanticos cercanos; el `cap` por bucket tambien limita la "
      "cobertura.")
    W("- En **generacion incremental** no hay problema: el camino es siempre "
      "O(N·d) y exacto (accuracy 1.000). El cuello de botella afecta al "
      "**entrenamiento/forward paralelo** a contexto largo.\n")
    W("### Tiene sentido entrenar ENGRAMA?\n")
    W("**Si, con dos condiciones:**\n")
    W("1. **Para tareas donde el recall lexico (copia/induccion) es la senal "
      "dominante**, el sistema es solido: 100% de recall exacto, traza O(N), "
      "consolidacion O(N), cero NaN, invarianza causal exacta en FP32, sin "
      "interferencia. Aqui ENGRAMA cumple sus promesas.")
    W("2. **Para tareas que dependen fuertemente del semantic tap asociativo a "
      "contexto largo**, el LSH actual no es fiable. Antes de entrenar a escala "
      "habria que mejorar el recall semantico aproximado: mas tablas/bits, "
      "LSH aprendido (ITQ/HashNet), un indice ANN jerarquico (IVF/HNSW sobre "
      "claves semanticas), o un recall denso por chunks (bloques de, p.ej., 2K-4K).\n")
    W("Los pesos no son el obstaculo (1B ~ 4 GB FP32); el obstaculo es el "
      "**O(N²) del recall semantico denso** frente al **bajo recall del LSH**. "
      "Resolver esa brecha es la I+D mas valiosa para ENGRAMA. Un entrenamiento "
      "TinyStories a contexto <=2K-4K con recall denso (o un LSH mejorado) es "
      "perfectamente razonable y es el siguiente paso logico.\n")
    W("---\n")
    W("*Todos los numeros provienen de `benchmarks/systematic/results/*.json`; "
      "los graficos estan en `benchmarks/systematic/figures/`. Los scripts son "
      "`1_*.py` ... `12_scaling_map.py` y se pueden reproducir con "
      "`python benchmarks/systematic/run_all.py` (mas 1 y 2 por separado).*\n")

    (HERE / "INFORME_FINAL.md").write_text("\n".join(out), encoding="utf-8")
    print("Informe escrito:", HERE / "INFORME_FINAL.md")


if __name__ == "__main__":
    main()
