"""E4: Cuantificacion de la perdida de informacion en la consolidacion.

Perturba T0[0] (primer token) con un delta pequenio y mide cuanta senal
llega a T_L[t] para t = 0..N-1, con y sin interferencia (relleno aleatorio).

Metricas:
- contribucion relativa: ||dT_L[t]|| / (||T_L[t]||)  (SNR del token lejano)
- comparacion con atencion ideal (peso 1/t del promedio uniforme: linea base
  de "compresion optimista sin decrimento") y con decaimiento exponencial.
"""
import json, math
import torch
from engrama.config import EngramaConfig
from engrama.model import EngramaModel

BASE = dict(d_model=256, d_gate=32, d_ff=1024, num_cells=8, num_encoder_layers=2,
            num_consolidation_layers=9, num_candidates=4, candidate_aggregation="latent_fusion",
            synapse_rank=32, version="v4", offset_mode="resonant_multirate",
            norm_type="rmsnorm", tie_embeddings=True, stable_init=True,
            gating_mode="source", trace_tap=True)

N = 512
EPS = 1e-3

def stack_forward(model, T0mod):
    t = T0mod
    for layer in model.consolidation.layers:
        t = layer.forward_train(t, T_0=T0mod)
    return t

def probe(seed=0, interferencia=True):
    torch.manual_seed(seed)
    cfg = EngramaConfig(vocab_size=1000, context_length=N, **BASE)
    model = EngramaModel(cfg).eval()
    x = torch.randint(0, 1000, (1, N))
    with torch.no_grad():
        T0 = model.encoder(model.embeddings(x))
        if not interferencia:
            # sin interferencia: contexto vacio (solo el token 0 distinto)
            T0 = torch.zeros_like(T0); T0[:, 0] = T0[:, 0]
        clean = stack_forward(model, T0)
        delta = torch.zeros_like(T0[0:1])
        delta[0, 0] = EPS
        pert = stack_forward(model, T0 + delta)
        dTL = (pert - clean)[0]           # (N, d)
        base = clean[0].norm(dim=-1).clamp_min(1e-9)
        contrib = dTL.norm(dim=-1) / base
    return contrib.tolist()

if __name__ == "__main__":
    out = {}
    for tag, inter in (("con_interferencia", True), ("solo_token0", False)):
        c = probe(interferencia=inter)
        # mostrar en decadas de distancia
        rows = []
        for t in [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 511]:
            rows.append((t, c[t]))
        out[tag] = rows
        print(f"--- {tag} ---")
        for t, v in rows:
            bar = "#" * max(0, min(60, int(-math.log10(max(v, 1e-12)) * 8)))
            print(f"t={t:4d}  contrib={v:9.2e}  10^(log) {math.log10(v):6.2f}  {bar}")
        # pendiente de decaimiento en log-log entre t=8 y t=511
        import math as m
        xs = [m.log(t) for t, v in rows if t >= 8 and v > 0]
        ys = [m.log(v) for t, v in rows if t >= 8 and v > 0]
        b = (len(xs) * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys)) / \
            (len(xs) * sum(x * x for x in xs) - sum(xs) ** 2)
        print(f"pendiente log-log contrib vs t (t>=8): {b:.3f}  (exp(-k*t^b))")
    with open("e4_results.json", "w") as f:
        json.dump(out, f, indent=1)
