"""E2: LM toy controlado — source vs dual vs transformer (mini), N semillas.

Corpus sintetico "tipo documento con tema persistente" (como TinyStories en miniatura):
- vocab 512, documentos de 128 tokens.
- 2 tokens-tema raros por doc que reaparecen mas adelante -> informacion util de largo alcance.
- relleno con bigramas Zipf.

Mide: val_loss final, saturacion de compuertas por capa, ganancia efectiva de ruta.
"""
import argparse, json, math, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/user/engrama/src")
from engrama.config import EngramaConfig
from engrama.model import EngramaModel
from engrama.losses import chunked_cross_entropy

V = 512
DOC = 128
FILLER = 300            # tokens de relleno (Zipf)
TOPIC = V - 12          # 512-12..511 = 12 tokens tema raros


def build_corpus(n_docs: int, seed: int):
    """Documentos con estructura local (bigrama estocastico, entropia alta)
    + estructura de largo alcance (2 temas raros que reaparecen y su marca)."""
    rng = np.random.default_rng(seed)
    # bigrama estocastico: fila t = distribucion sobre FILLER con entropia real
    logits = rng.normal(0, 1.0, size=(FILLER, FILLER))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    cum = np.cumsum(probs, axis=1)
    marks = {t: int(rng.integers(0, FILLER)) for t in range(TOPIC, V)}
    docs = np.empty((n_docs, DOC), dtype=np.int64)
    for d in range(n_docs):
        topics = rng.choice(np.arange(TOPIC, V), size=2, replace=False)
        t1_back = int(rng.integers(10, DOC - 4))
        t2_back = int(rng.integers(10, DOC - 4))
        toks = [-1] * DOC
        toks[0] = topics[0]
        toks[1] = topics[1]
        for pos, tp in ((t1_back, topics[0]), (t2_back, topics[1])):
            toks[pos] = tp
            if pos + 1 < DOC:
                toks[pos + 1] = marks[tp]
        cur = int(rng.integers(0, FILLER))
        for i in range(DOC):
            if toks[i] == -1:
                toks[i] = cur
                cur = int(np.searchsorted(cum[cur], rng.random()))
            else:
                cur = int(rng.integers(0, FILLER))  # re-sortear tras tema/marca
        docs[d] = toks
    return torch.from_numpy(docs)


def make_model(kind: str, seed: int):
    torch.manual_seed(seed)
    if kind == "transformer":
        return MiniTransformer()
    overrides = dict(d_model=64, d_gate=16, d_ff=256, num_cells=4,
                     num_encoder_layers=1, num_consolidation_layers=7,
                     context_length=DOC, num_candidates=4,
                     candidate_aggregation="latent_fusion", synapse_rank=16,
                     version="v4", offset_mode="resonant_multirate",
                     norm_type="rmsnorm", tie_embeddings=True, stable_init=True,
                     trace_tap=True)
    if kind == "source":
        overrides["gating_mode"] = "source"
    elif kind == "dual":
        overrides["gating_mode"] = "dual"
    elif kind == "no_tracetap":
        overrides["gating_mode"] = "dual"; overrides["trace_tap"] = False
    cfg = EngramaConfig(vocab_size=V, **overrides)
    return EngramaModel(cfg)


class MiniTransformer(nn.Module):
    def __init__(self, d=64, n_layers=6, n_heads=4, d_ff=256):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(DOC, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, d_ff) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, (nn.Linear,)):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))
        for b in self.blocks:
            h = b(h)
        return self.lnf(h) @ self.emb.weight.T


class Block(nn.Module):
    def __init__(self, d, nh, dff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, dff); self.fc2 = nn.Linear(dff, d)
        self.nh = nh
    def forward(self, x):
        B, T, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).view(B, T, 3, self.nh, d // self.nh).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        h = self.ln2(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


def train(kind: str, seed: int, steps=1200, lr=3e-3, batch=16):
    torch.manual_seed(seed)
    model = make_model(kind, seed)
    train_data = build_corpus(4000, seed=1)
    valid_data = build_corpus(500, seed=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    losses = []
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * sched(step)
        i = torch.randint(0, len(train_data) - 1, (batch,))
        x = train_data[i][:, :-1]
        y = train_data[i][:, 1:]
        if isinstance(model, MiniTransformer):
            logits = model(x)
            loss = chunked_cross_entropy(logits, y)
        else:
            loss = model.forward_loss(x, y, linear_chunk_size=2048, checkpoint_chunks=False)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            losses.append((step, float(loss.item())))
    # val loss
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(valid_data) - 1, 64):
            x = valid_data[i:i + 64, :-1]
            y = valid_data[i:i + 64, 1:]
            if isinstance(model, MiniTransformer):
                loss = chunked_cross_entropy(model(x), y, reduction="sum")
            else:
                loss = model.forward_loss(x, y, linear_chunk_size=2048, checkpoint_chunks=False, ) * y.numel()
            tot += float(loss.item()); n += y.numel()
    val = tot / n
    return model, val, losses


def gate_stats(model):
    """Saturacion y apertura media de compuertas por capa (con datos reales)."""
    out = []
    x = torch.randint(0, V, (2, DOC))
    with torch.no_grad():
        T0 = model.encoder(model.embeddings(x))
        t = T0
        for li, layer in enumerate(model.consolidation.layers):
            mix = layer.mix
            offsets = [p for p in mix.offsets if p < t.size(1)]
            keys = [str(p) for p in offsets]
            k_src = mix._causal_views(mix.p_g_src(t), offsets)
            ws = torch.stack([mix.gate_w_src[k] for k in keys])
            bs = torch.stack([mix.gate_b[k] for k in keys])
            pre = torch.einsum("bnpq,pqd->bnpd", k_src, ws) + bs
            if mix.gating_mode == "dual" and mix.p_g_tgt is not None:
                q = mix.p_g_tgt(t)
                wt = torch.stack([mix.gate_w_tgt[k] for k in keys])
                bil = (q.unsqueeze(2) * k_src).sum(-1, keepdim=True) / math.sqrt(mix.d_gate)
                pre = pre + torch.einsum("bnq,pqd->bnpd", q, wt) + bil
            g = torch.sigmoid(pre)
            rho = torch.sigmoid(torch.stack([mix.rho[k] for k in keys]))
            beta = torch.stack([mix.beta[k] for k in keys]).view(-1)
            out.append(dict(layer=li, offsets=offsets,
                            gate_mean=float(g.mean()), gate_sat=float(((g > .95) | (g < .05)).float().mean()),
                            rho_mean=float(rho.mean()), beta_mean=float(beta.mean()),
                            state_std=float(t.std())))
            t = layer.forward_train(t, T_0=T0)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="source,dual,transformer,no_tracetap")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--out", default="e2_results.json")
    a = ap.parse_args()
    results = {}
    for kind in a.kinds.split(","):
        results[kind] = []
        for seed in [int(s) for s in a.seeds.split(",")]:
            t0 = time.time()
            model, val, losses = train(kind, seed, steps=a.steps)
            rec = dict(kind=kind, seed=seed, val_loss=val, minutes=(time.time() - t0) / 60,
                       params=sum(p.numel() for p in model.parameters()),
                       train_loss_last=losses[-1][1])
            if hasattr(model, "consolidation"):
                rec["gates"] = gate_stats(model)
                rec["state_std_final"] = rec["gates"][-1]["state_std"]
            results[kind].append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != "gates"}), flush=True)
            if "gates" in rec:
                for gg in rec["gates"]:
                    print(f"   L{gg['layer']} gate={gg['gate_mean']:.2f} sat={gg['gate_sat']:.1%} rho={gg['rho_mean']:.2f} beta={gg['beta_mean']:.2f} std={gg['state_std']:.3f}", flush=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print("\n=== RESUMEN val_loss (media sobre semillas) ===")
    for kind, recs in results.items():
        vals = [r["val_loss"] for r in recs]
        print(f"{kind:14s} n={len(vals)}  mean={sum(vals)/len(vals):.4f}  all={['%.4f' % v for v in vals]}")
