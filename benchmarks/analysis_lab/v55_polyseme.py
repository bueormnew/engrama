"""Experimento de desambiguacion polisemica con CE_retrieval INTERNA.

Usa ``model.forward_loss`` (= CE_LM + lambda*CE_retrieval) directamente, SIN
perdida auxiliar a medida: es la prueba de que el sistema integrado (CE_retrieval
densa, activa por defecto) entrena el eje de sentido.

Tarea: M claves, cada una con 2 (contexto -> valor), definiciones repetidas y
repartidas. La consulta debe predecir el valor segun el contexto que precede a
la clave (azar 50 %). El eje lexico aislado siempre elige la ocurrencia mas
reciente del mismo token -> no puede desambiguar; el eje de sentido si.

  - sentido (beta entrenable, CE_retrieval activa): forward_loss por defecto
  - control lexico (beta == 0 congelado): CE_retrieval solo entrena lo lexico
"""
import argparse
import math
import random

import torch

from engrama.v55 import EngraModelV55, V55Config

VOCAB = 120
FILL = list(range(1, 16))
KEY = range(20, 40)
VAL = range(40, 60)
CTX = range(60, 80)


def build_sample(rng, seq_len, n_keys, repeats):
    seq = [0] + [rng.choice(FILL) for _ in range(seq_len - 1)]
    key_defs = {}
    for _ in range(n_keys):
        k = rng.choice(KEY)
        c1, c2 = rng.sample(CTX, 2)
        v1, v2 = rng.sample(VAL, 2)
        key_defs[k] = {c1: v1, c2: v2}
    total_defs = 2 * repeats * n_keys
    step = max(4, (seq_len - 8) // (total_defs + 1))
    pos = 4
    keys = list(key_defs.keys())
    rng.shuffle(keys)
    for _ in range(repeats):
        for k in keys:
            for c, v in key_defs[k].items():
                if pos + 2 >= seq_len - 6:
                    break
                seq[pos] = c
                seq[pos + 1] = k
                seq[pos + 2] = v
                pos += step
    k = rng.choice(keys)
    kv = key_defs[k]
    cq = rng.choice(list(kv.keys()))
    vq = kv[cq]
    q = seq_len - 4
    seq[q] = cq
    seq[q + 1] = k
    seq[q + 2] = vq
    return seq, q + 1, vq


def makecfg(seq_len, sense):
    return V55Config(
        vocab_size=VOCAB, d_model=64, d_gate=16, d_ff=256, num_cells=2,
        num_encoder_layers=1, num_consolidation_layers=7, context_length=seq_len,
        synapse_rank=16, num_candidates=1, d_recall=32, d_sense=32,
        rt_score_chunk=512, rt_train_mode='dense',
        rt_sense_beta_init=0.5 if sense else 0.0,
        rt_sense_beta_trainable=sense, rt_ctx_query='final', page_size=128,
        # cobertura completa: la CE_retrieval supervisa todas las posiciones con
        # candidatos mismo-token (incluye todas las definiciones de clave).
        retrieval_positions_frac=1.0,
    )


def train_eval(sense, steps, seed=0, rw=1.0):
    seq_len, n_keys, bs, repeats = 512, 6, 8, 3
    torch.manual_seed(seed)
    m = EngraModelV55(makecfg(seq_len, sense))
    if not sense:                      # control lexico: sentido desactivado
        with torch.no_grad():
            m.recall.beta.fill_(0.0)
        m.recall.beta.requires_grad_(False)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.01)
    rng = random.Random(seed)
    m.train()
    for step in range(1, steps + 1):
        lr = 2e-3 * min(1.0, step / 40) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g['lr'] = lr
        xs = []
        for r in range(bs):
            s, q, vq = build_sample(rng, seq_len, n_keys, repeats)
            xs.append(s)
        x = torch.tensor(xs)
        # forward_loss: CE_LM + lambda*CE_retrieval (INTERNA, por defecto).
        # El control lexico congela beta=0 -> CE_retrieval solo entrena lo lexico.
        loss = m.forward_loss(x[:, :-1], x[:, 1:], retrieval_weight=rw if sense else 0.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 150 == 0:
            print(f"  [sense={sense}] paso {step:4d} loss {loss.item():.4f} "
                  f"beta {m.recall.beta.item():.3f}", flush=True)
    m.eval()
    correct = tot = 0
    rng = random.Random(seed + 99)
    with torch.no_grad():
        for _ in range(96):
            s, q, vq = build_sample(rng, seq_len, n_keys, repeats)
            x = torch.tensor([s])
            lg = m(x[:, :-1])
            correct += int(lg.argmax(-1)[0, q].item() == vq)
            tot += 1
    return correct / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--rw', type=float, default=1.0, help='peso CE_retrieval (default 1.0 = sistema)')
    args = ap.parse_args()
    torch.set_num_threads(4)
    print(f"== sentido (CE_retrieval interna, rw={args.rw}) ==")
    acc_full = train_eval(True, args.steps, rw=args.rw)
    print("== control lexico (beta=0, CE_retrieval solo lexica) ==")
    acc_lex = train_eval(False, args.steps, rw=0.0)
    print(f"POLISEMIA (CE_retrieval interna rw={args.rw}):  sentido = {100*acc_full:.0f}%   "
          f"lexico-solo = {100*acc_lex:.0f}%   (azar 50%)")


if __name__ == '__main__':
    main()
