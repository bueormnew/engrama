"""Confirmacion rapida (semantico OFF) de los benchmarks lexicos que NO cambiaron
con las ediciones V5.5 (tap lexico intacto). Polisemia (sentido) + KV."""
import sys, math, random, torch
sys.path.insert(0, "benchmarks/analysis_lab")
import v55_polyseme as P
import v55_kv_longcontext as KV
torch.set_num_threads(4)

# --- POLISEMIA (sentido ON vs lexico), semantico OFF ---
print("== POLISEMIA (semantico OFF; mide el tap lexico de sentido) ==")
_orig = P.makecfg
def _makecfg_off(sl, sense):
    c = _orig(sl, sense); c.semantic_recall_enabled = False; return c
P.makecfg = _makecfg_off
acc_full = P.train_eval(True, steps=300, rw=1.0)
acc_lex = P.train_eval(False, steps=300, rw=0.0)
print(f"  sentido (rw=1.0): {100*acc_full:.0f}%   lexico: {100*acc_lex:.0f}%")

# --- KV (copia lexica), semantico OFF, distancia moderada ---
print("== KV (copia lexica, semantico OFF) ==")
def makecfg_kv(**over):
    torch.manual_seed(0)
    kw = dict(vocab_size=KV.VOCAB, d_model=64, d_gate=16, d_ff=256, num_cells=2,
        num_encoder_layers=1, num_consolidation_layers=7, context_length=2048,
        synapse_rank=16, num_candidates=1, d_recall=32, d_sense=32,
        rt_score_chunk=512, rt_temperature=0.5, rt_train_mode="dense",
        rt_sense_beta_init=0.3, page_size=256, semantic_recall_enabled=False)
    kw.update(over)
    return KV.EngraModelV55(KV.V55Config(**kw))

for dist in [64, 256, 1024]:
    rng = random.Random(0); torch.manual_seed(0)
    m = makecfg_kv(); m.train()
    seq_len = dist + 16; n_keys = 8; bs = 4
    KV.train_kv(m, 150, 1e-3, bs, seq_len, n_keys, 0, 999, 10.0, retrieval_weight=0.3)
    m.eval(); correct = tot = 0
    with torch.no_grad():
        for _ in range(24):
            qp = KV.train_positions(rng, seq_len, n_keys)
            x, ans = KV.batch_kv(rng, 1, seq_len, n_keys, qp)
            lg = m(x[:, :-1])
            correct += int(lg.argmax(-1)[0, ans[0][0][0]] == ans[0][0][1]); tot += 1
    print(f"  distancia {dist:5d}: {100*correct/max(1,tot):.0f}% ({correct}/{tot})", flush=True)
