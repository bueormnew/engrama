"""ENGRAMA V3 benchmark: long-range key-value retrieval.

Scientifically valid implementation of the retrieval tasks demanded by the
V3 spec (sections 30.3, 43, 47):

- Every sample binds each key to a **fresh random value**, so the mapping
  cannot be memorized globally: the model must retrieve it from context.
- Bindings live in a header; queries land at increasing distances, so
  accuracy-vs-distance measures the identity-transport hypothesis of V3.
- Two offset policies are compared head to head (spec ablation D/E):
  ``dense_dilated`` (V2 connectivity) vs ``hierarchical_dyadic`` (V3).

The report is generated from REAL runs only (spec section 56).

Usage::

    python benchmarks/kv_retrieval.py --steps 250 --out benchmarks/KV_RETRIEVAL_REPORT.md
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from engrama.config import EngramaConfig
from engrama.model import EngramaModel

VOCAB_SIZE = 64
FILLER_LO, FILLER_HI = 1, 15          # repeating body pattern tokens
KEY_LO, KEY_HI = 20, 29               # 10 possible key tokens
VAL_LO, VAL_HI = 40, 55               # 16 possible value tokens
BOS = 0

N_KEYS = 4
SEQ_LEN = 192
HEADER_SLOTS = [0, 2, 4, 6]           # (key_i, value_i) pairs at 0..7
QUERY_POSITIONS = [32, 80, 128, 184]  # increasing retrieval distance
ANSWER_WEIGHT = 10.0                  # extra loss weight on answer tokens


def make_sample(rng: random.Random) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """One sample: random header bindings + periodic body + queries."""
    keys = rng.sample(range(KEY_LO, KEY_HI + 1), N_KEYS)
    values = [rng.randint(VAL_LO, VAL_HI) for _ in range(N_KEYS)]
    bindings = list(zip(keys, values))
    value_of: Dict[int, int] = dict(bindings)

    seq = [BOS] + [0] * (SEQ_LEN - 1)
    for slot, (k, v) in zip(HEADER_SLOTS, bindings):
        seq[1 + slot] = k
        seq[1 + slot + 1] = v

    body = [rng.randint(FILLER_LO, FILLER_HI) for _ in range(8)]
    for pos in range(9, SEQ_LEN):
        seq[pos] = body[(pos - 9) % 8]

    query_order = keys[:]
    rng.shuffle(query_order)
    used = set(HEADER_SLOTS) | {s + 1 for s in HEADER_SLOTS} | {0}
    for pos, key in zip(QUERY_POSITIONS, query_order):
        used.add(pos)
        used.add(pos + 1)
        seq[pos] = key
        seq[pos + 1] = value_of[key]
    for pos in range(9, SEQ_LEN):  # keep body tokens off query cells
        if pos not in used:
            seq[pos] = body[(pos - 9) % 8]

    targets = [(pos, value_of[key]) for pos, key in zip(QUERY_POSITIONS, query_order)]
    return torch.tensor(seq, dtype=torch.long), targets


def make_batch(batch_size: int, rng: random.Random):
    samples = [make_sample(rng) for _ in range(batch_size)]
    seqs = torch.stack([s for s, _ in samples])
    answers = [t for _, t in samples]
    return seqs, answers


def evaluate(model: EngramaModel, num_samples: int, rng: random.Random,
             device: str) -> Dict[str, float]:
    """Accuracy at query answers, split by retrieval distance."""
    model.eval()
    per_query_correct = [0] * N_KEYS
    total = 0
    with torch.no_grad():
        remaining = num_samples
        while remaining > 0:
            b = min(32, remaining)
            remaining -= b
            seqs, answers = make_batch(b, rng)
            seqs = seqs.to(device)
            logits = model(seqs[:, :-1])
            preds = logits.argmax(dim=-1)  # preds[t] predicts token t+1
            for row, ans in enumerate(answers):
                for qi, (pos, value) in enumerate(ans):
                    # prediction at index pos-1 of seqs[:, :-1] predicts seq[pos]
                    if preds[row, pos].item() == value:
                        per_query_correct[qi] += 1
                    total += 1
    overall = sum(per_query_correct) / max(1, total)
    result = {"overall": overall}
    for qi in range(N_KEYS):
        dist = QUERY_POSITIONS[qi] - max(HEADER_SLOTS) - 2
        result[f"query_{qi}_distance_{dist}"] = per_query_correct[qi] / max(
            1, num_samples
        )
    return result


def train_model(offset_mode: str, steps: int, seed: int, device: str, log_every: int,
                global_anchor: bool = False):
    cfg = EngramaConfig(
        vocab_size=VOCAB_SIZE, d_model=64, d_gate=8, d_ff=256, num_cells=2,
        num_encoder_layers=1, num_consolidation_layers=8, context_length=SEQ_LEN,
        num_candidates=1, candidate_aggregation="mean", offset_mode=offset_mode,
        global_anchor=global_anchor,
        offsets=[0, 1, 2, 4, 8, 16, 32, 64, 128] if offset_mode == "dense_dilated" else None,
    )
    torch.manual_seed(seed)
    model = EngramaModel(cfg).to(device)
    rng = random.Random(seed + 7)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    t0 = time.time()
    model.train()
    log: List[Tuple[int, float]] = []
    for step in range(1, steps + 1):
        seqs, answers = make_batch(16, rng)
        seqs = seqs.to(device)
        logits = model(seqs[:, :-1])
        targets = seqs[:, 1:]
        raw = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), reduction="none"
        ).view(16, -1)
        weights = torch.ones_like(raw)
        for row, ans in enumerate(answers):
            for pos, _ in ans:
                weights[row, pos] = ANSWER_WEIGHT
        loss = (raw * weights).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % log_every == 0:
            log.append((step, float(loss.item())))

    elapsed = time.time() - t0
    eval_rng = random.Random(seed + 999)
    metrics = evaluate(model, num_samples=200, rng=eval_rng, device=device)
    return model, log, metrics, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "KV_RETRIEVAL_REPORT.md"))
    ap.add_argument(
        "--force", action="store_true",
        help="Overwrite the target report file if it already exists "
             "(by default an existing file is never overwritten; a run "
             "suffixed report is written instead)",
    )
    args = ap.parse_args()

    # Never clobber an existing report silently: the versioned
    # KV_RETRIEVAL_REPORT.md holds the canonical published numbers.
    out_path = args.out
    if os.path.exists(out_path) and not args.force:
        base, ext = os.path.splitext(out_path)
        out_path = f"{base}_{args.steps}steps_seed{args.seed}{ext}"
        print(
            f"[benchmark] {args.out} already exists; writing run-specific "
            f"report to {out_path} (pass --force to overwrite)"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chance = 1.0 / (VAL_HI - VAL_LO + 1)

    lines = [
        "# ENGRAMA V3 — Benchmark de recuperacion clave-valor de largo alcance",
        "",
        "Benchmark de las tareas 30.3/43/47 de la especificacion V3.",
        "**Todos los numeros de este reporte se generaron ejecutando "
        "`benchmarks/kv_retrieval.py`; ningun valor es proyectado.**",
        "",
        "## Protocolo",
        "",
        f"- Secuencia de {SEQ_LEN} tokens; cabecera con {N_KEYS} pares "
        "clave-valor **aleatorios por muestra** (imposible memorizarlos).",
        f"- Consultas en posiciones {QUERY_POSITIONS} (distancias "
        f"~{QUERY_POSITIONS[0] - 8}..{QUERY_POSITIONS[-1] - 8} tokens).",
        "- 200 muestras de evaluacion con semilla independiente del "
        "entrenamiento.",
        f"- Nivel azar: {chance:.1%} (16 valores posibles).",
        f"- Dispositivo: {device}.",
        "",
        "| Configuracion | Pasos | Loss inicial | Loss final | "
        "Precision recuperacion | Tiempo |",
        "|---|---|---|---|---|---|",
    ]

    summary: Dict[str, Dict[str, float]] = {}
    configs = [
        ("hierarchical_dyadic", "hierarchical_dyadic", False),
        ("hierarchical_dyadic", "hierarchical_dyadic + ancla", True),
        ("dense_dilated", "dense_dilated", False),
    ]
    for mode, tag, anchor in configs:
        model, log, metrics, elapsed = train_model(
            mode, args.steps, args.seed,
            device, args.log_every, global_anchor=anchor,
        )
        summary[tag] = metrics
        params = model.num_parameters()
        lines.append(
            f"| V3 `{tag}` ({params:,} params) | {args.steps} | "
            f"{log[0][1]:.4f} | {log[-1][1]:.4f} | "
            f"**{metrics['overall']:.1%}** | {elapsed:.1f}s |"
        )

    lines += ["", "## Precision por distancia de recuperacion", ""]
    header = "| Distancia |"
    sep = "|---|"
    rows: Dict[str, str] = {m: f"| V3 `{m}` |" for m in summary}
    for qi in range(N_KEYS):
        dist = QUERY_POSITIONS[qi] - max(HEADER_SLOTS) - 2
        header += f" ~{dist} tok |"
        sep += "---|"
        for m, metrics in summary.items():
            rows[m] += f" {metrics[f'query_{qi}_distance_{dist}']:.1%} |"
    lines += [header, sep, *rows.values()]
    lines += [
        "",
        "## Interpretacion honesta",
        "",
        "- La precision por encima del azar indica que la informacion del "
        "encabezado sobrevive el transporte hasta la consulta (hipotesis V3, "
        "seccion 29); la comparacion entre politicas de offsets mide el "
        "riesgo principal de V3 (seccion 42).",
        "- Este benchmark mide una tarea sintetica; no demuestra equivalencia "
        "con atencion en lenguaje natural (V3 seccion 41).",
        "- Ejecutado con: `python benchmarks/kv_retrieval.py "
        f"--steps {args.steps} --seed {args.seed}` en este entorno (CPU).",
        "",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[benchmark] report written to {out_path}")


if __name__ == "__main__":
    main()
