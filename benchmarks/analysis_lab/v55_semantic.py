"""Aguja semantica nativa: recuperacion asociativa por SIGNIFICADO (no token).

Tarea: K pares de alias fijos (a_i, b_i). En cada ejemplo, cada par recibe un
valor ALEATORIO v_i, y se inserta el hecho "a_i v_i". La consulta usa el ALIAS
"b_i" (token DISTINTO de a_i) y debe predecir v_i. Como v_i es aleatorio por
ejemplo, el modelo NO puede memorizar b_i->v_i: tiene que RETENER el hecho a_i y
puentear b_i~a_i por significado.

  - tap lexico: la consulta b_i nunca aparecio antes en el ejemplo -> no hay
    candidato mismo-token -> FALLA (~azar 1/|V|).
  - tap semantico: q_sem(b_i) debe aparecer a K_sem(a_i hecho) (aprendido via la
    consistencia del par alias a lo largo del dataset) -> lee v_i.

Entrena con forward_loss (CE_retrieval incluye el termino semantico asociativo).
"""
import argparse
import math
import random

import torch

from engrama.v55 import EngraModelV55, V55Config

K = 8               # pares de alias
NV = 16             # valores posibles
NFILL = 12


def vocab_layout():
    bos = 0
    a_tok = list(range(1, 1 + K))            # a_i
    b_tok = list(range(1 + K, 1 + 2 * K))    # b_i (alias de a_i)
    v_tok = list(range(1 + 2 * K, 1 + 2 * K + NV))   # valores
    fill = list(range(1 + 2 * K + NV, 1 + 2 * K + NV + NFILL))
    return bos, a_tok, b_tok, v_tok, fill


def build_sample(rng, seq_len, n_queries):
    bos, A, B, V, F = vocab_layout()
    seq = [bos] + [rng.choice(F) for _ in range(seq_len - 1)]
    # asignar valor aleatorio por par e insertar hechos "a_i v_i"
    pair_val = {}
    slots = list(range(4, seq_len - 8))
    step = max(3, len(slots) // (K + n_queries + 2))
    pos = slots[0]
    for i in range(K):
        v = rng.choice(V)
        pair_val[i] = v
        if pos + 1 < seq_len - 2:
            seq[pos] = A[i]
            seq[pos + 1] = v
        pos += step
    # consultas: alias b_i -> target v_i (el valor del par i)
    queries = []
    chosen = rng.sample(range(K), min(n_queries, K))
    for i in chosen:
        if pos + 1 >= seq_len - 1:
            break
        seq[pos] = B[i]          # alias (token distinto de a_i)
        seq[pos + 1] = pair_val[i]   # ground truth: v_i despues de b_i
        queries.append((pos, pair_val[i]))   # predecir en pos -> v_i
        pos += step
    return seq, queries


def makecfg(seq_len, semantic):
    return V55Config(
        vocab_size=1 + 2 * K + NV + NFILL, d_model=64, d_gate=16, d_ff=256,
        num_cells=2, num_encoder_layers=1, num_consolidation_layers=7,
        context_length=seq_len, synapse_rank=16, num_candidates=1,
        d_recall=32, d_sense=32, d_semantic=32, rt_score_chunk=512,
        rt_train_mode="dense", rt_ctx_query="final", page_size=128,
        semantic_recall_enabled=semantic, retrieval_positions_frac=1.0,
    )


def train_eval(semantic, steps, seed=0):
    seq_len, bs, n_q = 160, 8, 4
    torch.manual_seed(seed)
    m = EngraModelV55(makecfg(seq_len, semantic))
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.01)
    rng = random.Random(seed)
    m.train()
    for step in range(1, steps + 1):
        lr = 2e-3 * min(1.0, step / 40) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr
        xs = []
        for r in range(bs):
            s, _ = build_sample(rng, seq_len, n_q)
            xs.append(s)
        x = torch.tensor(xs)
        loss = m.forward_loss(x[:, :-1], x[:, 1:], retrieval_weight=1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 150 == 0:
            print(f"  [sem={semantic}] paso {step:4d} loss {loss.item():.4f}", flush=True)
    m.eval()
    correct = tot = 0
    rng = random.Random(seed + 99)
    with torch.no_grad():
        for _ in range(96):
            s, qs = build_sample(rng, seq_len, n_q)
            x = torch.tensor([s])
            lg = m(x[:, :-1])
            for (qp, vt) in qs:
                if qp < lg.size(1):
                    correct += int(lg.argmax(-1)[0, qp].item() == vt)
                    tot += 1
    return correct / max(1, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    torch.set_num_threads(4)
    print(f"== tap semantico ON (azar = {100/NV:.1f}%) ==")
    acc_sem = train_eval(True, args.steps)
    print("== control lexico (tap semantico OFF) ==")
    acc_lex = train_eval(False, args.steps)
    print(f"AGUJA SEMANTICA:  semantico = {100*acc_sem:.0f}%   "
          f"lexico-solo = {100*acc_lex:.0f}%   (azar {100/NV:.1f}%)")


if __name__ == "__main__":
    main()
