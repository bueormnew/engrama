"""E1: ¿Saturan las compuertas del gating dual en la config exacta del compare?"""
import torch, math
from engrama.config import EngramaConfig
from engrama.model import EngramaModel

BASE = dict(d_model=256, d_gate=32, d_ff=1024, num_cells=8, num_encoder_layers=2,
            num_consolidation_layers=9, num_candidates=4, candidate_aggregation="latent_fusion",
            synapse_rank=32, version="v4", offset_mode="resonant_multirate",
            norm_type="rmsnorm", tie_embeddings=True, stable_init=True)

def probe(gating_mode, trace_tap=True, seed=0):
    torch.manual_seed(seed)
    cfg = EngramaConfig(vocab_size=1000, context_length=512, gating_mode=gating_mode,
                        trace_tap=trace_tap, **BASE)
    model = EngramaModel(cfg).eval()
    x = torch.randint(0, 1000, (2, 512))
    with torch.no_grad():
        emb = model.embeddings(x)              # (B,N,d)
        T0 = model.encoder(emb)                # huellas aisladas
        print(f"--- gating={gating_mode} tap={trace_tap} ---")
        print(f"|emb| std={emb.std():.3f}  |T0| std={T0.std():.3f}")
        t = T0
        for li, layer in enumerate(model.consolidation.layers):
            mix = layer.mix
            offsets = [p for p in mix.offsets if p < t.size(1)]
            keys = [str(p) for p in offsets]
            k_all = mix.p_g_src(t)
            k_src = mix._causal_views(k_all, offsets)
            gate_w_src = torch.stack([mix.gate_w_src[k] for k in keys])
            gate_b = torch.stack([mix.gate_b[k] for k in keys])
            g_src = torch.einsum("bnpq,pqd->bnpd", k_src, gate_w_src)
            terms = {"g_src": g_src}
            if gating_mode == "dual":
                q_tgt = mix.p_g_tgt(t)
                gate_w_tgt = torch.stack([mix.gate_w_tgt[k] for k in keys])
                g_tgt = torch.einsum("bnq,pqd->bnpd", q_tgt, gate_w_tgt)
                bilin = (q_tgt.unsqueeze(2) * k_src).sum(dim=-1, keepdim=True) / math.sqrt(mix.d_gate)
                terms["g_tgt"] = g_tgt; terms["bilinear"] = bilin.expand_as(g_src)
                pre = g_src + g_tgt + bilin + gate_b
            else:
                pre = g_src + gate_b
            g = torch.sigmoid(pre)
            sat = ((g > 0.95) | (g < 0.05)).float().mean().item()
            msg = " | ".join(f"{n}: m{v.mean():+.2f} s{v.std():.2f}" for n, v in terms.items())
            rho = torch.sigmoid(torch.stack([mix.rho[k] for k in keys]))
            print(f"L{li} off={offsets} | {msg} | gate mean={g.mean():.3f} sat5%={sat:.1%} | rho={rho.mean():.2f} | |T_in|={t.std():.3f}")
            t = layer.forward_train(t, T_0=T0)
            print(f"      -> salida capa std={t.std():.3f}")

probe("source")
probe("dual")
