"""Tests de la arquitectura ENGRAMA V6 (puramente lineal, recall exacto).

Cubre:
  - invarianza causal (paralelo == incremental) en FP32 (<1e-6)
  - vocab_size correcto del evocador (fix del bug V5.5)
  - ausencia de NaN/Inf
  - que el recall LSH semantico recupera matches en tarea sintetica
  - que el indice invertido lexico no materializa N x N
"""
from __future__ import annotations

import unittest

import torch

from engrama.v6 import EngraModelV6, V6Config
from engrama.v6.lsh import V6LSHIndex, shared_planes, sign_codes
from engrama.v6.recall import _l2


def tiny_cfg(**over):
    kw = dict(
        vocab_size=128, d_model=32, d_gate=8, d_ff=64, num_cells=2,
        num_encoder_layers=1, num_consolidation_layers=4, context_length=128,
        synapse_rank=8, num_candidates=2, d_recall=16, d_sense=16,
        d_semantic=16, rt_score_chunk=64, page_size=16,
        rt_lsh_tables=8, rt_lsh_bits=12, rt_lsh_cap=4,
        rt_lsh_rescue_window=32,
    )
    kw.update(over)
    return V6Config(**kw)


class TestV6CausalInvariance(unittest.TestCase):
    def _check(self, model, x, atol=1e-6):
        model.eval()
        with torch.no_grad():
            par = model(x)
            cache = model.get_cache()
            steps = []
            for t in range(x.size(1)):
                lg, _ = model.step_forward(x[:, t:t + 1], cache, timestamp=t)
                steps.append(lg)
            steps = torch.stack(steps, dim=1).squeeze(2)
        self.assertEqual(par.shape, steps.shape)
        diff = float((par - steps).abs().max())
        self.assertLess(diff, atol, msg=f"max diff {diff:.3e}")

    def test_fp32_random(self):
        torch.manual_seed(0)
        self._check(EngraModelV6(tiny_cfg()), torch.randint(0, 128, (2, 40)), atol=1e-6)

    def test_fp32_repeats(self):
        torch.manual_seed(1)
        x = torch.randint(0, 8, (1, 40)).repeat(2, 1)
        self._check(EngraModelV6(tiny_cfg()), x, atol=1e-6)

    def test_fp16_invariance(self):
        torch.manual_seed(2)
        # FP16 tiene error de redondeo; exigimos <=2e-3 (mejor que V5.5)
        self._check(EngraModelV6(tiny_cfg(dtype="float16")),
                    torch.randint(0, 128, (1, 40)), atol=2e-3)


class TestV6VocabAndStability(unittest.TestCase):
    def test_evoker_vocab_matches_config(self):
        cfg = tiny_cfg(vocab_size=300)
        m = EngraModelV6(cfg).eval()
        x = torch.randint(0, 300, (1, 16))
        with torch.no_grad():
            y = m(x)
        self.assertEqual(y.shape[-1], 300)

    def test_no_nan_inf_extreme_inputs(self):
        cfg = tiny_cfg()
        m = EngraModelV6(cfg).eval()
        for scale in (0.0, 1.0, 1e3):
            with torch.no_grad():
                m.embeddings.weight.normal_(0, scale)
                x = torch.randint(0, 128, (2, 32))
                y = m(x)
            self.assertEqual(int(torch.isnan(y).sum()), 0)
            self.assertEqual(int(torch.isinf(y).sum()), 0)


class TestV6SemanticLSH(unittest.TestCase):
    def test_lsh_recovers_synthetic_concept(self):
        """Concepto + alias: el LSH debe recuperar una posicion del concepto."""
        torch.manual_seed(0)
        n, d = 2048, 32
        n_concepts = 64
        # construye secuencia concepto/alias como en el benchmark 4
        g = torch.Generator().manual_seed(42)
        bases = torch.randn(n_concepts, d, generator=g)
        bases = torch.nn.functional.normalize(bases, dim=-1)
        K = torch.zeros(n, d)
        Q = torch.zeros(n, d)
        valid = torch.zeros(n, dtype=torch.bool)
        i = c = 0
        while i + 1 < n:
            base = bases[c % n_concepts]
            K[i] = base
            a = base + 0.1 * torch.randn(d, generator=g)
            K[i + 1] = a
            Q[i + 1] = base
            valid[i + 1] = True
            i += 2; c += 1
        tokens = torch.arange(n) % n_concepts
        idx = V6LSHIndex.build(K, tokens, gap=1, n_tables=8, n_bits=12,
                               rescue_window=32, bucket_cap=4)
        planes = shared_planes(8, d, 12, K.device, K.dtype)
        qc = sign_codes(Q, planes)
        cand, ok = idx.candidates(qc)
        rows = valid.nonzero().flatten()
        # rerank exacto sobre candidatos
        chosen = []
        Kn = _l2(K)
        for s in range(0, rows.numel(), 512):
            ix = rows[s:s + 512]
            lc = cand[ix]; lv = ok[ix]; C = lc.size(1)
            ck = Kn[lc.clamp(min=0)].view(-1, C, d)
            sc = (_l2(Q[ix]).unsqueeze(1) * ck).sum(-1)
            sc = torch.where(lv, sc, torch.full_like(sc, -1e30))
            j = sc.argmax(dim=-1)
            chosen.append(lc.gather(1, j.view(-1, 1)).squeeze(1))
        chosen = torch.cat(chosen)
        # el elegido debe tener coseno >=0.99 con la consulta (es el concepto)
        cos = torch.nn.functional.cosine_similarity(K[chosen], Q[rows], dim=-1)
        self.assertGreaterEqual(float((cos >= 0.99).float().mean()), 0.99)


class TestV6LinearCost(unittest.TestCase):
    def test_candidate_count_bounded(self):
        """El numero de candidatos LSH debe estar acotado (O(N), no O(N^2))."""
        torch.manual_seed(0)
        for n in [1024, 4096, 16384]:
            d = 32
            K = torch.randn(n, d)
            tokens = torch.arange(n) % 64
            idx = V6LSHIndex.build(K, tokens, gap=1, n_tables=8, n_bits=12,
                                   rescue_window=32, bucket_cap=4)
            planes = shared_planes(8, d, 12, K.device, K.dtype)
            qc = sign_codes(K, planes)
            cand, ok = idx.candidates(qc)
            avg_c = float(ok.sum()) / n
            # C debe permanecer pequeno y acotado (< 200)
            self.assertLess(avg_c, 200,
                            msg=f"n={n} avg candidates {avg_c:.1f}")


if __name__ == "__main__":
    unittest.main()
