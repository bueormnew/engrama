"""ENGRAMA V5.5 — benchmark de recuperacion KV en contextos ENORMES + POLISEMIA.

Dos protocolos:

1. **KV copia** (extension del benchmarks/kv_retrieval.py): N pares clave-valor,
   consultas a distancias crecientes. Entrena en seq=2048 y evalua en 8192 y
   16384. La lectura dura no extrapola (argmax no decae). Objetivo >=95 %.

2. **Polisemia/desambiguacion** (nuevo V5.5): la MISMA clave aparece en DOS
   contextos con valores distintos. Solo el eje de SENTIDO (K_sense contextual)
   puede desambiguar; el eje lexico aislado (V5) no. Mide si V5.5 elige el valor
   correcto por contexto.

Semantica LM estandar: logits = model(x[:, :-1]); logits[i] predice x[i+1].
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

from engrama.v55 import EngraModelV55, V55Config

VOCAB = 64
FILL_LO, FILL_HI = 1, 15
KEY_LO, KEY_HI = 20, 29
VAL_LO, VAL_HI = 40, 55
CTX_LO, CTX_HI = 56, 63           # tokens de CONTEXTO para la tarea polisemica
BOS = 0
CHANCE_KV = 1 / 16


def make_kv_sample(rng, seq_len, n_keys, query_pos):
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


def make_polyseme_sample(rng, seq_len, n_pairs):
    """La MISMA clave K aparece en dos contextos C1, C2 con valores V1, V2.

    Patron: ``C1 K V1 ... C2 K V2 ... <consulta> C? K ???``.
    El modelo debe predecir el valor segun el contexto (C1->V1, C2->V2). El eje
    lexico solo ve K->? (empate entre V1 y V2); el eje de sentido desempata.
    """
    seq = [BOS] + [rng.randint(FILL_LO, FILL_HI) for _ in range(seq_len - 1)]
    used = {0}
    c1, c2 = rng.sample(range(CTX_LO, CTX_HI + 1), 2)
    key = rng.randint(KEY_LO, KEY_HI)
    v1, v2 = rng.sample(range(VAL_LO, VAL_HI + 1), 2)
    # dos definiciones
    p1 = 4
    seq[p1], seq[p1 + 1], seq[p1 + 2] = c1, key, v1
    used.update({p1, p1 + 1, p1 + 2})
    p2 = seq_len // 2
    seq[p2], seq[p2 + 1], seq[p2 + 2] = c2, key, v2
    used.update({p2, p2 + 1, p2 + 2})
    # consulta: contexto aleatorio -> esperar el valor correspondiente
    ctx_q = rng.choice([c1, c2])
    val_q = v1 if ctx_q == c1 else v2
    q = seq_len - 6
    seq[q], seq[q + 1], seq[q + 2] = ctx_q, key, val_q
    used.update({q, q + 1, q + 2})
    return seq, [(q + 1, val_q)]   # predecir la posicion q+1 (el valor)


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
        synapse_rank=16, num_candidates=1, d_recall=32, d_sense=32,
        rt_score_chunk=512, rt_temperature=0.5, rt_train_mode="dense",
        rt_sense_beta_init=0.3, page_size=256,
    )
    kw.update(over)
    return EngraModelV55(V55Config(**kw))


def batch_kv(rng, bs, seq_len, n_keys, query_pos):
    seqs, answers = [], []
    for _ in range(bs):
        s, a = make_kv_sample(rng, seq_len, n_keys, query_pos)
        seqs.append(s); answers.append(a)
    return torch.tensor(seqs, dtype=torch.long), answers


def train_kv(model, steps, lr, bs, seq_len, n_keys, seed, log_every, answer_weight,
             retrieval_weight=0.3):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        lr_t = lr * min(1.0, step / 50) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr_t
        qp = train_positions(rng, seq_len, n_keys)
        x, answers = batch_kv(rng, bs, seq_len, n_keys, qp)
        xin, y = x[:, :-1], x[:, 1:]
        w = torch.ones_like(y, dtype=torch.float32)
        for r, ans in enumerate(answers):
            for pos, _ in ans:
                w[r, pos] = answer_weight
        logits = model(xin)
        raw = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1),
                              reduction="none").view(bs, -1)
        lm_loss = (raw * w).mean()
        # CE_retrieval (Seccion 6): entrena las proyecciones del tap directamente,
        # independiente de g_rt (que arranca en 0). Es lo que hace converger la
        # recuperacion en tareas de copia/KV.
        ret_loss = model.retrieval_loss(xin, y) if retrieval_weight > 0 else xin.new_zeros(())
        loss = lm_loss + retrieval_weight * ret_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            print(f"    [KV] paso {step:4d} lm {lm_loss.item():.4f} "
                  f"ret {float(ret_loss):.4f}", flush=True)
    return time.time() - t0


def train_polyseme(model, steps, lr, bs, seq_len, seed, log_every):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    for step in range(1, steps + 1):
        lr_t = lr * min(1.0, step / 50) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr_t
        seqs, answers = [], []
        for _ in range(bs):
            s, a = make_polyseme_sample(rng, seq_len, 1)
            seqs.append(s); answers.append(a)
        x = torch.tensor(seqs, dtype=torch.long)
        xin, y = x[:, :-1], x[:, 1:]
        w = torch.ones_like(y, dtype=torch.float32)
        for r, ans in enumerate(answers):
            for pos, _ in ans:
                w[r, pos] = 20.0
        logits = model(xin)
        raw = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1),
                              reduction="none").view(bs, -1)
        loss = (raw * w).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            print(f"    [POLI] paso {step:4d} loss {loss.item():.4f}", flush=True)


@torch.no_grad()
def eval_kv(model, seq_len, n_keys, n_samples, seed, bs=2, fast=True):
    rng = random.Random(seed)
    model.eval()
    qs = query_grid(seq_len, n_keys)
    per_q = [0] * len(qs)
    exact = total = 0
    done = 0
    while done < n_samples:
        b = min(bs, n_samples - done); done += b
        x, answers = batch_kv(rng, b, seq_len, n_keys, qs)
        xin = x[:, :-1]
        if fast:
            feats = model.forward_features(xin, score_rows=torch.tensor(qs))
            logits = model.evoker(feats, model.output_embeddings)
        else:
            logits = model(xin)
        pred = logits.argmax(-1)
        for r, ans in enumerate(answers):
            for qi, (pos, val) in enumerate(ans):
                hit = int(pred[r, pos].item() == val)
                per_q[qi] += hit; exact += hit; total += 1
    out = {"seq_len": seq_len, "overall": exact / max(1, total), "chance": CHANCE_KV}
    for qi, q in enumerate(qs):
        out[f"d{q}"] = per_q[qi] / max(1, n_samples)
    return out


@torch.no_grad()
def eval_polyseme(model, n_samples, seq_len, seed):
    rng = random.Random(seed)
    model.eval()
    correct = total = 0
    for _ in range(n_samples):
        s, ans = make_polyseme_sample(rng, seq_len, 1)
        x = torch.tensor([s], dtype=torch.long)
        logits = model(x[:, :-1])
        pred = logits.argmax(-1)
        for pos, val in ans:
            correct += int(pred[0, pos].item() == val); total += 1
    return correct / max(1, total)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--answer-weight", type=float, default=10.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--train-seq", type=int, default=2048)
    ap.add_argument("--n-keys", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="v55_kv_results.json")
    ap.add_argument("--polyseme-steps", type=int, default=0,
                    help="pasos de fine-tune en la tarea polisemica (0 = saltar)")
    ap.add_argument("--retrieval-weight", type=float, default=0.0,
                    help="peso de CE_retrieval (0 = LM puro, receta V5 probada)")
    args = ap.parse_args()

    model = build_model(args.seed)
    print(f"V5.5 params={model.num_parameters():,}", flush=True)
    print(model.config.describe(), flush=True)

    secs = train_kv(model, args.steps, args.lr, args.bs, args.train_seq,
                    args.n_keys, args.seed + 7, args.log_every, args.answer_weight,
                    retrieval_weight=args.retrieval_weight)
    print(f"entrenado KV en {secs/60:.1f} min", flush=True)

    results = {"params": model.num_parameters(), "train_seconds": secs}
    for seq, nks in ((args.train_seq, args.n_keys), (8192, 8), (16384, 8)):
        t0 = time.time()
        r = eval_kv(model, seq, nks, n_samples=32, seed=args.seed + 999, fast=True)
        r["eval_seconds"] = time.time() - t0
        results[f"eval_{seq}"] = r
        dists = " ".join(f"d{q}={100*r[f'd{q}']:4.0f}%"
                         for q in sorted(int(k[1:]) for k in r
                                         if k.startswith("d") and k[1:].isdigit()))
        print(f"[seq {seq:6d}] overall={100*r['overall']:5.1f}%  "
              f"(azar {100*CHANCE_KV:.1f}%)  {dists}  [{r['eval_seconds']:.0f}s]", flush=True)

    if args.polyseme_steps > 0:
        train_polyseme(model, args.polyseme_steps, args.lr, args.bs,
                       args.train_seq, args.seed + 11, args.log_every)
        acc = eval_polyseme(model, 64, args.train_seq, args.seed + 22)
        results["polyseme_acc"] = acc
        print(f"[polisemia] accuracy por contexto = {100*acc:5.1f}% "
              f"(azar 50%, eje lexico solo no desempata)", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    ok = all(results[f"eval_{s}"]["overall"] >= 0.95 for s in (8192, 16384))
    print("OBJETIVO >=95% (KV enorme):", "CUMPLE" if ok else "NO CUMPLE")
