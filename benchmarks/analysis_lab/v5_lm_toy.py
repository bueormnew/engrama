"""LM toy controlado: V5 vs V4 vs transformer (misma receta, mismas semillas).

Corpus sintetico con tema persistente (identico al e2_toy_lm.py del analisis V4).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import torch

torch.set_num_threads(max(1, (torch.get_num_threads() + 1) // 2))

from engrama.config import EngramaConfig
from engrama.model import EngramaModel as ModelV4
from engrama.v5 import EngraModel as ModelV5, V5Config

V = 512
DOC = 128
FILLER = 300
TOPIC = V - 12


def build_corpus(n_docs, seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.0, size=(FILLER, FILLER))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    cum = np.cumsum(probs, axis=1)
    marks = {t: int(rng.integers(0, FILLER)) for t in range(TOPIC, V)}
    docs = np.empty((n_docs, DOC), dtype=np.int64)
    for d in range(n_docs):
        topics = rng.choice(np.arange(TOPIC, V), size=2, replace=False)
        t1 = int(rng.integers(10, DOC - 4))
        t2 = int(rng.integers(10, DOC - 4))
        toks = [-1] * DOC
        toks[0], toks[1] = topics[0], topics[1]
        for pos, tp in ((t1, topics[0]), (t2, topics[1])):
            toks[pos] = tp
            if pos + 1 < DOC:
                toks[pos + 1] = marks[tp]
        cur = int(rng.integers(0, FILLER))
        for i in range(DOC):
            if toks[i] == -1:
                toks[i] = cur
                cur = int(np.searchsorted(cum[cur], rng.random()))
            else:
                cur = int(rng.integers(0, FILLER))
        docs[d] = toks
    return torch.from_numpy(docs)


class MiniTransformer(torch.nn.Module):
    def __init__(self, d=64, layers=6, heads=4, dff=256):
        super().__init__()
        import torch.nn as nn
        import torch.nn.functional as F
        self.F = F
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(DOC, d)
        self.blocks = nn.ModuleList([Blk(d, heads, dff) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T))
        for b in self.blocks:
            h = b(h)
        return self.ln(h) @ self.emb.weight.T


class Blk(torch.nn.Module):
    def __init__(self, d, nh, dff):
        super().__init__()
        import torch.nn as nn
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.proj = nn.Linear(d, d, bias=False)
        self.f1 = nn.Linear(d, dff); self.f2 = nn.Linear(dff, d)
        self.nh = nh
    def forward(self, x):
        B, T, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).view(B, T, 3, self.nh, d // self.nh).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.f2(torch.nn.functional.gelu(self.f1(self.ln2(x))))


def make(kind, seed):
    torch.manual_seed(seed)
    if kind == "transformer":
        return MiniTransformer()
    if kind == "v4":
        return ModelV4(EngraConfig(
            vocab_size=V, d_model=64, d_gate=16, d_ff=256, num_cells=4,
            num_encoder_layers=1, num_consolidation_layers=7, context_length=DOC,
            num_candidates=4, candidate_aggregation="latent_fusion", synapse_rank=16,
            version="v4", offset_mode="resonant_multirate", norm_type="rmsnorm",
            tie_embeddings=True, stable_init=True, trace_tap=True))
    if kind == "v5":
        return ModelV5(V5Config(
            vocab_size=V, d_model=64, d_gate=16, d_ff=256, num_cells=4,
            num_encoder_layers=1, num_consolidation_layers=7, context_length=DOC,
            synapse_rank=16, num_candidates=4, d_recall=32, rt_layers=(3,),
            rt_score_chunk=128))
    raise ValueError(kind)


def train(kind, seed, steps=800, lr=3e-3, bs=8):
    torch.manual_seed(seed)
    model = make(kind, seed)
    tr = build_corpus(4000, seed=1)
    va = build_corpus(500, seed=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    for step in range(steps):
        lr_t = lr * min(1.0, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr_t
        i = torch.randint(0, len(tr) - 1, (bs,))
        x = tr[i][:, :-1]
        y = tr[i][:, 1:]
        if kind == "transformer":
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        else:
            loss = model.forward_loss(x, y, linear_chunk_size=4096, checkpoint_chunks=False)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(va) - 1, 64):
            x = va[i:i + 64, :-1]
            y = va[i:i + 64, 1:]
            if kind == "transformer":
                loss = torch.nn.functional.cross_entropy(
                    model(x).reshape(-1, V), y.reshape(-1), reduction="sum")
            else:
                loss = model.forward_loss(x, y, linear_chunk_size=4096) * y.numel()
            tot += float(loss.item())
            n += y.numel()
    return model, tot / n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="v5,v4,transformer")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--out", default="v5_lm_results.json")
    a = ap.parse_args()
    res = {}
    for kind in a.kinds.split(","):
        res[kind] = []
        for seed in [int(s) for s in a.seeds.split(",")]:
            t0 = time.time()
            m, val = train(kind, seed, steps=a.steps)
            res[kind].append({"seed": seed, "val_loss": val,
                              "params": sum(p.numel() for p in m.parameters()),
                              "minutes": (time.time() - t0) / 60})
            print(json.dumps(res[kind][-1]), flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("\n=== LM toy (menor = mejor) ===")
    for kind, rs in res.items():
        vals = [r["val_loss"] for r in rs]
        print(f"{kind:12s} params={rs[0]['params']:7,d} mean={sum(vals)/len(vals):.4f} "
              f"all={[round(v, 3) for v in vals]}")
