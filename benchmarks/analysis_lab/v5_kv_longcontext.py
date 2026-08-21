"""ENGRAMA V5 — benchmark de recuperacion KV en contextos ENORMES (v2, limpio).

Protocolo (extension del benchmarks/kv_retrieval.py del repo):
- N pares clave-valor aleatorios por muestra (imposibles de memorizar).
- Consultas a distancias crecientes, hasta ~N de la secuencia.
- ENTRENA en seq=2048 y EVALUA en 8192 y 16384: la lectura dura no extrapola,
  el argmax no depende de la distancia. Objetivo >=85% global (azar 6.25%).

Semantica LM estandar en todo el archivo:
    logits = model(x[:, :-1]);  logits[i] predice x[i+1].
    La respuesta en la consulta `pos` es x[pos+1] (el valor), target index pos
    en el espacio desplazado.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch
import torch.nn.functional as F

torch.set_num_threads(max(1, (torch.get_num_threads() + 1) // 2))

from engrama.v5 import EngraModel, V5Config

VOCAB = 64
FILL_LO, FILL_HI = 1, 15
KEY_LO, KEY_HI = 20, 29          # 10 claves
VAL_LO, VAL_HI = 40, 55          # 16 valores
BOS = 0
CHANCE = 1 / 16
ANSWER_WEIGHT = 10.0  # redefinible por CLI


def make_sample(rng: random.Random, seq_len: int, n_keys: int, query_pos):
    keys = rng.sample(range(KEY_LO, KEY_HI + 1), n_keys)
    values = [rng.randint(VAL_LO, VAL_HI) for _ in range(n_keys)]
    value_of = dict(zip(keys, values))
    seq = [BOS] + [FILL_LO] * (seq_len - 1)
    used = {0}
    for i, (k, v) in enumerate(zip(keys, values)):
        s = 2 * i
        seq[1 + s], seq[1 + s + 1] = k, v
        used.update({1 + s, 1 + s + 1})
    order = keys[:]
    rng.shuffle(order)
    answers = []
    for pos, key in zip(query_pos, order):
        seq[pos] = key
        seq[pos + 1] = value_of[key]
        used.update({pos, pos + 1})
        answers.append((pos, value_of[key]))
    body = [rng.randint(FILL_LO, FILL_HI) for _ in range(8)]
    for i in range(1, seq_len):
        if i not in used:
            seq[i] = body[i % 8]
    return seq, answers


def train_positions(rng, seq_len, n_keys):
    lo = 2 * n_keys + 4
    return sorted(rng.sample(range(lo, seq_len - 4), n_keys))


def query_grid(seq_len, n_keys):
    lo = 2 * n_keys + 4
    hi = seq_len - 4
    return sorted({int(lo + (hi - lo) * f) for f in (0.05, 0.2, 0.4, 0.6, 0.8, 0.93, 0.985)})


def build_model(seed=0, **over):
    torch.manual_seed(seed)
    kw = dict(
        vocab_size=VOCAB, d_model=64, d_gate=16, d_ff=256, num_cells=2,
        num_encoder_layers=1, num_consolidation_layers=7, context_length=2048,
        synapse_rank=16, num_candidates=1, d_recall=32, rt_layers=(3,),
        rt_score_chunk=512, rt_temperature=0.5,
    )
    kw.update(over)
    return EngraModel(V5Config(**kw))


def batch_of(rng, bs, seq_len, n_keys, query_pos):
    seqs, answers = [], []
    for _ in range(bs):
        s, a = make_sample(rng, seq_len, n_keys, query_pos)
        seqs.append(s)
        answers.append(a)
    return torch.tensor(seqs, dtype=torch.long), answers


def train(model, steps, lr, bs, seq_len, n_keys, seed, log_every=50, answer_weight=ANSWER_WEIGHT):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    t0 = time.time()
    hist = []
    for step in range(1, steps + 1):
        lr_t = (lr * min(1.0, step / 50)
                * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = lr_t
        qp = train_positions(rng, seq_len, n_keys)
        x, answers = batch_of(rng, bs, seq_len, n_keys, qp)
        xin, y = x[:, :-1], x[:, 1:]                      # objetivo desplazado
        w = torch.ones_like(y, dtype=torch.float32)
        for r, ans in enumerate(answers):
            for pos, _val in ans:
                w[r, pos] = answer_weight                # y[pos] = x[pos+1] = valor
        logits = model(xin)
        raw = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1),
                              reduction="none").view(bs, -1)
        loss = (raw * w).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            hist.append((step, float(loss.item())))
            print(f"    paso {step:4d} loss {loss.item():.4f}", flush=True)
    return hist, time.time() - t0


@torch.no_grad()
def evaluate(model, seq_len, n_keys, n_samples, seed, bs=2, fast=True):
    """`fast=True`: puntuacion RT solo en las filas de consulta (O(N*Q)).
    `fast=False`: forward completo — validacion honesta final."""
    rng = random.Random(seed)
    model.eval()
    qs = query_grid(seq_len, n_keys)
    qi_of = {q: i for i, q in enumerate(qs)}
    per_q = [0] * len(qs)
    total = exact = 0
    done = 0
    while done < n_samples:
        b = min(bs, n_samples - done)
        done += b
        x, answers = batch_of(rng, b, seq_len, n_keys, qs)
        xin = x[:, :-1]
        if fast:
            feats = model.forward_features(xin, score_rows=torch.tensor(qs))
            logits = model.evoker(feats, model.output_embeddings)  # (B, N-1, V)
        else:
            logits = model(xin)
        pred = logits.argmax(-1)
        for r, ans in enumerate(answers):
            for qi, (pos, val) in enumerate(ans):
                hit = int(pred[r, pos].item() == val)      # logits[pos] -> x[pos+1]
                per_q[qi] += hit
                exact += hit
                total += 1
    out = {"seq_len": seq_len, "overall": exact / max(1, total), "chance": CHANCE}
    for qi, q in enumerate(qs):
        out[f"d{q}"] = per_q[qi] / max(1, n_samples)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--answer-weight", type=float, default=10.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--train-seq", type=int, default=2048)
    ap.add_argument("--n-keys", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="v5_kv_results.json")
    ap.add_argument("--over", default="", help="overrides k=v,k=v para V5Config")
    ap.add_argument("--full-eval", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="cargar /tmp/v5_kv_model.pt y solo evaluar")
    ap.add_argument("--dense-steps", type=int, default=0,
                    help="arranque hibrido: primeros N pasos en modo denso "
                         "(bootstrap de la metrica), resto en modo LSH")
    a = ap.parse_args()

    over = {}
    for kv in filter(None, a.over.split(",")):
        k, v = kv.split("=")
        try:
            over[k] = int(v)
        except ValueError:
            try:
                over[k] = float(v)
            except ValueError:
                over[k] = v
    model = build_model(a.seed, **over)
    print(f"V5 params={model.num_parameters():,}", flush=True)
    print(model.config.describe(), flush=True)
    secs, hist = 0.0, []
    if not a.eval_only:
        if a.dense_steps > 0:
            # arranque hibrido: denso (exacto, O(N^2 d_k)) mientras la metrica
            # P_q/P_k aprende la identidad; luego LSH lineal para siempre.
            model.config.rt_train_mode = "dense"
            hist1, s1 = train(model, a.dense_steps, a.lr, a.bs, a.train_seq,
                              a.n_keys, a.seed + 7, log_every=a.log_every,
                              answer_weight=a.answer_weight)
            model.config.rt_train_mode = "lsh"
            hist2, s2 = train(model, a.steps - a.dense_steps, a.lr, a.bs,
                              a.train_seq, a.n_keys, a.seed + 7,
                              log_every=a.log_every, answer_weight=a.answer_weight)
            hist, secs = hist1 + hist2, s1 + s2
        else:
            hist, secs = train(model, a.steps, a.lr, a.bs, a.train_seq, a.n_keys,
                               a.seed + 7, log_every=a.log_every,
                               answer_weight=a.answer_weight)
        print(f"entrenado en {secs/60:.1f} min", flush=True)
        torch.save(model.state_dict(), "/tmp/v5_kv_model.pt")
    else:
        model.load_state_dict(torch.load("/tmp/v5_kv_model.pt", map_location="cpu"))
        print("checkpoint cargado", flush=True)

    results = {"params": model.num_parameters(), "train_seconds": secs, "history": hist}
    n_eval = 32 if not a.full_eval else 8
    for seq, nks in ((a.train_seq, a.n_keys), (8192, 8), (16384, 8)):
        t0 = time.time()
        r = evaluate(model, seq, nks, n_samples=n_eval, seed=a.seed + 999,
                     fast=not a.full_eval)
        r["eval_seconds"] = time.time() - t0
        results[f"eval_{seq}"] = r
        dists = " ".join(f"d{q}={100 * r[f'd{q}']:4.0f}%"
                         for q in sorted(int(k[1:]) for k in r
                                         if k.startswith("d") and k[1:].isdigit()))
        print(f"[seq {seq:6d}] overall={100 * r['overall']:5.1f}%  "
              f"(azar {100 * CHANCE:.1f}%)  {dists}  [{r['eval_seconds']:.0f}s]", flush=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    ok = all(results[f"eval_{s}"]["overall"] >= 0.85
             for s in (8192, 16384) if f"eval_{s}" in results)
    print("OBJETIVO >=85%:", "CUMPLE" if ok else "NO CUMPLE",
          f"({'completa' if a.full_eval else 'rapida'})")
