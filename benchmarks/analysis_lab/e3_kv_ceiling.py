"""E3: Techo arquitectonico de recuperacion KV (entrenando EN la tarea).

Protocolo = benchmarks/kv_retrieval.py (repo) + control Transformer + variantes.
Azar = 1/16 = 6.25%.
"""
import argparse, json, math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from engrama.config import EngramaConfig
from engrama.model import EngramaModel

VOCAB = 64
FILL_LO, FILL_HI = 1, 15
KEY_LO, KEY_HI = 20, 29
VAL_LO, VAL_HI = 40, 55
BOS = 0
N_KEYS = 4
SEQ = 192
HEADER_SLOTS = [0, 2, 4, 6]
QPOS = [32, 80, 128, 184]
W_ANS = 10.0

ENG = dict(d_model=64, d_gate=16, d_ff=256, num_cells=2, num_encoder_layers=1,
           num_consolidation_layers=8, context_length=SEQ, num_candidates=1,
           candidate_aggregation="mean", synapse_rank=16, tie_embeddings=True,
           stable_init=True)


def make_sample(rng: random.Random):
    keys = rng.sample(range(KEY_LO, KEY_HI + 1), N_KEYS)
    values = [rng.randint(VAL_LO, VAL_HI) for _ in range(N_KEYS)]
    value_of = dict(zip(keys, values))
    seq = [BOS] + [FILL_LO] * (SEQ - 1)
    for slot, (k, v) in zip(HEADER_SLOTS, zip(keys, values)):
        seq[1 + slot] = k; seq[1 + slot + 1] = v
    body = [rng.randint(FILL_LO, FILL_HI) for _ in range(8)]
    order = keys[:]; rng.shuffle(order)
    used = {0} | set(HEADER_SLOTS) | {s + 1 for s in HEADER_SLOTS}
    for pos, key in zip(QPOS, order):
        seq[pos] = key; seq[pos + 1] = value_of[key]
        used.add(pos); used.add(pos + 1)
    for i in range(9, SEQ):
        if i not in used:
            seq[i] = body[(i - 9) % 8]
    targets = [(pos, value_of[k]) for pos, k in zip(QPOS, order)]
    return seq, targets


def make_batch(bs, rng):
    seqs, answers = [], []
    for _ in range(bs):
        s, a = make_sample(rng)
        seqs.append(s); answers.append(a)
    return torch.tensor(seqs), answers


class TinyGPT(nn.Module):
    def __init__(self, d=64, layers=8, heads=4, dff=256):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(SEQ, d)
        self.blocks = nn.ModuleList([Blk(d, heads, dff) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))
        for b in self.blocks:
            h = b(h)
        return self.ln(h) @ self.emb.weight.T


class Blk(nn.Module):
    def __init__(self, d, nh, dff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.proj = nn.Linear(d, d, bias=False)
        self.f1 = nn.Linear(d, dff); self.f2 = nn.Linear(dff, d)
        self.nh = nh
    def forward(self, x):
        B, T, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).view(B, T, 3, self.nh, d // self.nh).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.f2(F.gelu(self.f1(self.ln2(x))))


def build(kind, seed):
    torch.manual_seed(seed)
    if kind == "transformer":
        return TinyGPT()
    kw = dict(ENG)
    if kind == "v4_resonant":
        kw.update(version="v4", offset_mode="resonant_multirate")  # dual + tap
    elif kind == "source_resonant":
        kw.update(version="v4", offset_mode="resonant_multirate", gating_mode="source")
    elif kind == "v4_dense":
        kw.update(version="v4", offset_mode="dense_dilated",
                  offsets=[0, 1, 2, 4, 8, 16, 32, 64, 128])
    elif kind == "v4_no_tap":
        kw.update(version="v4", offset_mode="resonant_multirate", trace_tap=False)
    return EngramaModel(EngramaConfig(vocab_size=VOCAB, **kw))


def run(kind, seed, steps=400, lr=4e-3, bs=16):
    t0 = time.time()
    model = build(kind, seed)
    params = sum(p.numel() for p in model.parameters())
    rng = random.Random(seed + 7)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    loss0 = lossN = float("nan")
    for step in range(1, steps + 1):
        seqs, answers = make_batch(bs, rng)
        logits = model(seqs[:, :-1])
        tgt = seqs[:, 1:]
        raw = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1),
                              reduction="none").view(bs, -1)
        w = torch.ones_like(raw)
        for r, ans in enumerate(answers):
            for pos, _ in ans:
                w[r, pos] = W_ANS
        loss = (raw * w).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1: loss0 = float(loss.item())
        lossN = float(loss.item())
    # eval
    model.eval()
    eval_rng = random.Random(seed + 999)
    corr = [0] * N_KEYS; tot = 0
    with torch.no_grad():
        for _ in range(200 // 16):
            seqs, answers = make_batch(16, eval_rng)
            logits = model(seqs[:, :-1])
            pred = logits.argmax(-1)
            for r, ans in enumerate(answers):
                for qi, (pos, val) in enumerate(ans):
                    corr[qi] += int(pred[r, pos].item() == val); tot += 1
    acc = {f"d{QPOS[qi]-8}": corr[qi] / 200 for qi in range(N_KEYS)}
    acc["overall"] = sum(corr) / max(1, tot)
    return dict(kind=kind, seed=seed, params=params, loss0=loss0, lossN=lossN,
                acc=acc, minutes=(time.time() - t0) / 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="v4_resonant,source_resonant,v4_dense,v4_no_tap,transformer")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default="e3_results.json")
    a = ap.parse_args()
    res = {}
    for kind in a.kinds.split(","):
        res[kind] = []
        for seed in [int(s) for s in a.seeds.split(",")]:
            r = run(kind, seed, steps=a.steps)
            res[kind].append(r)
            print(json.dumps(r), flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("\n=== KV entrenado (azar 6.25%) ===")
    for kind, rs in res.items():
        accs = [r["acc"]["overall"] for r in rs]
        d = rs[0]["acc"]
        print(f"{kind:18s} params={rs[0]['params']:7,d}  overall={100*sum(accs)/len(accs):5.1f}%  "
              f"cerca={100*d['d24']:.0f}%  lejos={100*d['d176']:.0f}%  lossN={rs[0]['lossN']:.3f}")
