"""ENGRAMA V5 benchmark: long-range key-value retrieval by synaptic resonance.

Same protocol as ``benchmarks/kv_retrieval.py`` but for the V5 architecture.
Each sample binds keys to fresh random values in a header; queries appear later
at growing distances. The model must retrieve the exact value from context —
which V4 could not do (~chance) and V5 does by content-addressable resonance.

Usage::

    python benchmarks/kv_retrieval_v5.py --seq 256 --steps 3000
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from engrama.v5 import EngramaV5, EngramaV5Config

VOCAB = 48
FILLER_LO, FILLER_HI = 1, 15
KEY_LO, KEY_HI = 20, 27      # 8 possible keys
VAL_LO, VAL_HI = 32, 47      # 16 possible values (chance = 6.25%)
BOS = 0
N_KEYS = 3
ANSWER_WEIGHT = 10.0


def _layout(seq_len: int):
    header = [1, 3, 5][:N_KEYS]
    q0 = max(header) + 8
    queries = [q0 + i * ((seq_len - q0 - 2) // N_KEYS) for i in range(N_KEYS)]
    return header, queries


def make_sample(rng: random.Random, seq_len: int):
    header, queries = _layout(seq_len)
    keys = rng.sample(range(KEY_LO, KEY_HI + 1), N_KEYS)
    vals = [rng.randint(VAL_LO, VAL_HI) for _ in range(N_KEYS)]
    vo = dict(zip(keys, vals))
    seq = [BOS] + [rng.randint(FILLER_LO, FILLER_HI) for _ in range(seq_len - 1)]
    for slot, (k, v) in zip(header, zip(keys, vals)):
        seq[slot] = k
        seq[slot + 1] = v
    order = keys[:]
    rng.shuffle(order)
    targets: List[Tuple[int, int]] = []
    for pos, k in zip(queries, order):
        seq[pos] = k
        seq[pos + 1] = vo[k]
        targets.append((pos, vo[k]))
    return torch.tensor(seq, dtype=torch.long), targets


def make_batch(bs: int, rng: random.Random, seq_len: int):
    s = [make_sample(rng, seq_len) for _ in range(bs)]
    return torch.stack([x for x, _ in s]), [t for _, t in s]


def evaluate(model, rng, seq_len, device, samples=320) -> Dict[str, float]:
    model.eval()
    per = [0] * N_KEYS
    n = 0
    with torch.no_grad():
        left = samples
        while left > 0:
            b = min(16, left)
            left -= b
            seqs, ans = make_batch(b, rng, seq_len)
            preds = model(seqs[:, :-1].to(device)).argmax(-1)
            for r, a in enumerate(ans):
                for qi, (pos, v) in enumerate(a):
                    per[qi] += int(preds[r, pos].item() == v)
                    n += 1
    out = {"overall": sum(per) / max(1, n)}
    _, queries = _layout(seq_len)
    for qi in range(N_KEYS):
        out[f"dist_{queries[qi]}"] = per[qi] / (samples)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    cfg = EngramaV5Config(
        vocab_size=VOCAB, d_model=args.d_model, num_layers=args.layers,
        num_heads=args.heads, context_length=args.seq, num_candidates=1,
        norm_type="layernorm",
    )
    model = EngramaV5(cfg).to(device)
    print(f"ENGRAMA V5  seq={args.seq}  params={model.num_parameters():,}  device={device}")
    print(f"chance = {1.0/(VAL_HI-VAL_LO+1):.1%}")

    rng = random.Random(args.seed + 7)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        seqs, ans = make_batch(16, rng, args.seq)
        logits = model(seqs[:, :-1].to(device))
        tgt = seqs[:, 1:].to(device)
        raw = F.cross_entropy(
            logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none"
        ).view(16, -1)
        w = torch.ones_like(raw)
        for r, a in enumerate(ans):
            for pos, _ in a:
                w[r, pos] = ANSWER_WEIGHT
        loss = (raw * w).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 10) == 0:
            m = evaluate(model, random.Random(999), args.seq, device)
            print(f"  step {step:5d}  loss {loss.item():.3f}  recall {m['overall']:.1%}")
            model.train()

    m = evaluate(model, random.Random(999), args.seq, device, samples=640)
    print(f"\nFINAL recall = {m['overall']:.1%}  ({time.time()-t0:.0f}s)")
    for k, v in m.items():
        if k != "overall":
            print(f"  {k}: {v:.1%}")


if __name__ == "__main__":
    main()
