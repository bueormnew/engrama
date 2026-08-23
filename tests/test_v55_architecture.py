"""Tests de arquitectura ENGRAMA V5.5.

Cubre la invarianza causal estricta (paralelo == incremental, incluida la
lectura ASIMETRICA con q_ctx contextual y empates), aislamiento de codigos K_lex,
estabilidad anti-NaN (fp16 + extremos + LR alto), conteo de parametros y API.
"""
from __future__ import annotations

import unittest

import torch

from engrama.v55 import EngraModelV55, V55Config


def tiny_cfg(**over):
    kw = dict(
        vocab_size=64, d_model=32, d_gate=8, d_ff=64, num_cells=2,
        num_encoder_layers=1, num_consolidation_layers=4, context_length=64,
        synapse_rank=8, num_candidates=2, d_recall=16, d_sense=16,
        rt_score_chunk=64, rt_train_mode="dense", page_size=16,
    )
    kw.update(over)
    return V55Config(**kw)


class TestV55CausalInvariance(unittest.TestCase):
    def _check(self, model, x, label, atol=1e-4):
        model.eval()
        with torch.no_grad():
            par = model(x)
            cache = model.get_cache()
            steps = []
            for t in range(x.size(1)):
                lg, _ = model.step_forward(x[:, t:t + 1], cache, timestamp=t)
                steps.append(lg)
            steps = torch.stack(steps, dim=1)
        diff = float((par - steps).abs().max())
        self.assertLess(diff, atol, msg=f"{label}: max diff {diff:.3e}")

    def test_parallel_equals_incremental_random(self):
        torch.manual_seed(0)
        self._check(EngraModelV55(tiny_cfg()), torch.randint(0, 64, (2, 28)), "random B=2")

    def test_parallel_equals_incremental_with_repeats(self):
        # vocabulario chico -> muchas repeticiones -> empates en la lectura.
        torch.manual_seed(1)
        x = torch.randint(0, 8, (1, 28)).repeat(2, 1)
        self._check(EngraModelV55(tiny_cfg()), x, "repeats B=2")

    def test_parallel_equals_incremental_b1(self):
        torch.manual_seed(2)
        self._check(EngraModelV55(tiny_cfg()), torch.randint(0, 64, (1, 30)), "random B=1")

    def test_invariance_with_contextual_q_ctx(self):
        # q_ctx ve el estado consolidado final (contextual). La invarianza debe
        # mantenerse incluso con la consulta contextual (Pilar 4).
        torch.manual_seed(3)
        self._check(EngraModelV55(tiny_cfg(rt_ctx_query="final")),
                    torch.randint(0, 64, (2, 28)), "q_ctx=final")

    def test_invariance_across_page_boundaries(self):
        # longitud > page_size (16): fuerza cruzar varias paginas.
        torch.manual_seed(4)
        self._check(EngraModelV55(tiny_cfg(page_size=8)),
                    torch.randint(0, 64, (2, 40)), "page boundaries")

    def test_every_position_matches(self):
        torch.manual_seed(5)
        m = EngraModelV55(tiny_cfg()).eval()
        x = torch.randint(0, 64, (1, 32))
        with torch.no_grad():
            par = m(x)
            cache = m.get_cache()
            steps = []
            for t in range(32):
                lg, _ = m.step_forward(x[:, t:t + 1], cache, timestamp=t)
                steps.append(lg)
            steps = torch.stack(steps, dim=1)
        self.assertTrue(torch.allclose(par, steps, atol=1e-4),
                        msg=f"max diff {float((par - steps).abs().max()):.3e}")


class TestV55Stability(unittest.TestCase):
    def test_no_nan_fp16_forward(self):
        torch.manual_seed(7)
        m = EngraModelV55(tiny_cfg()).eval().half()
        x = torch.randint(0, 64, (2, 28))
        with torch.no_grad():
            logits = m(x)
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_no_nan_extreme_inputs(self):
        torch.manual_seed(8)
        m = EngraModelV55(tiny_cfg()).eval()
        with torch.no_grad():
            for x in (torch.zeros(1, 28, dtype=torch.long),
                      torch.full((1, 28), 63, dtype=torch.long),
                      torch.randint(0, 64, (1, 2))):
                self.assertTrue(torch.isfinite(m(x)).all().item())

    def test_training_high_lr_no_nan(self):
        torch.manual_seed(9)
        m = EngraModelV55(tiny_cfg())
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)   # 10x LR tipico
        for _ in range(20):
            x = torch.randint(0, 64, (4, 28))
            loss = m.forward_loss(x, x, retrieval_weight=0.0)
            self.assertTrue(torch.isfinite(loss).item())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()

    def test_residual_magnitude_stays_bounded(self):
        torch.manual_seed(10)
        m = EngraModelV55(tiny_cfg()).eval()
        x = torch.randint(0, 64, (2, 28))
        with torch.no_grad():
            t0 = m.footprints(x)
            t = t0
            stds = [float(t.std())]
            for layer in m.consolidation.layers:
                t = layer.forward_train(t, t0=t0)
                stds.append(float(t.std()))
        self.assertLess(max(stds) / max(stds[0], 1e-6), 15.0,
                        msg=f"magnitudes: {[round(s, 2) for s in stds]}")

    def test_softcap_bounds_logits(self):
        # Pilar 6: softcap mantiene los logits en [-C, C].
        torch.manual_seed(11)
        m = EngraModelV55(tiny_cfg(logit_cap=30.0)).eval()
        x = torch.randint(0, 64, (2, 28))
        with torch.no_grad():
            logits = m(x)
        self.assertLess(float(logits.abs().max()), 30.0 + 1e-3)


class TestV55ParamsAndAPI(unittest.TestCase):
    def test_param_count_near_v4(self):
        from engrama.config import EngramaConfig as C4
        from engrama.model import EngramaModel as M4
        torch.manual_seed(12)
        v55 = EngraModelV55(V55Config(
            vocab_size=50257, d_model=256, d_gate=32, d_ff=1024, num_cells=8,
            num_encoder_layers=2, num_consolidation_layers=9, context_length=8192,
            synapse_rank=32, d_recall=64, d_sense=64))
        v4 = M4(C4(vocab_size=50257, d_model=256, d_gate=32, d_ff=1024, num_cells=8,
                   num_encoder_layers=2, num_consolidation_layers=9,
                   context_length=512, num_candidates=4, version="v4"))
        p55, p4 = v55.num_parameters(), v4.num_parameters()
        self.assertLess(abs(p55 - p4) / p4, 0.08, msg=f"v55={p55:,} v4={p4:,}")

    def test_cache_memory_linear_in_context(self):
        from engrama.v55 import PagedDualTrace
        bytes_per_token = []
        for n in (64, 256, 1024):
            tr = PagedDualTrace(n, 32, 16, 16, horizons=[1] * 3, page_size=16)
            bytes_per_token.append(tr.memory_bytes() / n)
        slope = (bytes_per_token[-1] - bytes_per_token[0]) / (1024 - 64)
        self.assertLess(abs(slope), 5.0, msg=f"crecimiento no lineal: {bytes_per_token}")

    def test_forward_loss_backward_finite(self):
        torch.manual_seed(13)
        m = EngraModelV55(tiny_cfg())
        x = torch.randint(0, 64, (2, 28))
        loss = m.forward_loss(x, x)
        loss.backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all().item() for g in grads))

    def test_presets_api(self):
        for size in ("tiny", "small", "base", "large"):
            cfg = V55Config.from_preset(size, vocab_size=100)
            self.assertGreater(cfg.d_model, 0)
        m = EngraModelV55(V55Config.from_preset("tiny", vocab_size=64))
        self.assertGreater(m.num_parameters(), 0)

    def test_save_load_roundtrip(self):
        import tempfile, os
        torch.manual_seed(14)
        m = EngraModelV55(tiny_cfg()).eval()
        x = torch.randint(0, 64, (1, 12))
        with torch.no_grad():
            before = m(x)
        with tempfile.TemporaryDirectory() as d:
            m.save(d)
            m2 = EngraModelV55.load(d).eval()
            with torch.no_grad():
                after = m2(x)
        self.assertTrue(torch.allclose(before, after, atol=1e-5))

    def test_generate_runs(self):
        torch.manual_seed(15)
        m = EngraModelV55(tiny_cfg())
        out = m.generate([1, 2, 3], max_new_tokens=8, temperature=0.8)
        self.assertEqual(out[:3], [1, 2, 3])
        self.assertTrue(all(0 <= t < 64 for t in out))


if __name__ == "__main__":
    unittest.main()
