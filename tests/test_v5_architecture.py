"""Tests de arquitectura ENGRAMA V5.

Cubre: invarianza causal estricta (paralelo == incremental, incluida la
lectura dura del Recall Tap con empates), aislamiento de codigos K, estabilidad
anti-NaN (fp16 + extremos), conteo de parametros y API publica.
"""
from __future__ import annotations

import unittest

import torch

from engrama.v5 import EngraModel, RecallTap, V5Config, V5Trace


def tiny_cfg(**over):
    kw = dict(
        vocab_size=64, d_model=32, d_gate=8, d_ff=64, num_cells=2,
        num_encoder_layers=1, num_consolidation_layers=4, context_length=48,
        synapse_rank=8, num_candidates=2, d_recall=16, rt_layers=(2,),
        rt_score_chunk=16,
    )
    kw.update(over)
    return V5Config(**kw)


class TestV5CausalInvariance(unittest.TestCase):
    def test_parallel_equals_incremental_random(self):
        torch.manual_seed(0)
        model = EngraModel(tiny_cfg()).eval()
        x = torch.randint(0, 64, (2, 40))
        with torch.no_grad():
            par = model(x)
            cache = model.get_cache()
            for t in range(40):
                step_logits, _ = model.step_forward(x[:, t : t + 1], cache, timestamp=t)
        self.assertTrue(
            torch.allclose(par[:, -1], step_logits, atol=1e-4),
            msg=f"max diff {float((par[:, -1] - step_logits).abs().max()):.3e}",
        )

    def test_parallel_equals_incremental_with_repeats(self):
        # Tokens repetidos: fuerza empates exactos en la lectura dura.
        torch.manual_seed(1)
        model = EngraModel(tiny_cfg()).eval()
        base = torch.randint(0, 8, (1, 40))  # vocabulario chico -> repeticiones
        x = base.repeat(2, 1)
        with torch.no_grad():
            par = model(x)
            cache = model.get_cache()
            for t in range(40):
                step_logits, _ = model.step_forward(x[:, t : t + 1], cache, timestamp=t)
        self.assertTrue(
            torch.allclose(par[:, -1], step_logits, atol=1e-4),
            msg=f"max diff {float((par[:, -1] - step_logits).abs().max()):.3e}",
        )

    def test_every_position_matches(self):
        torch.manual_seed(2)
        model = EngraModel(tiny_cfg()).eval()
        x = torch.randint(0, 64, (1, 36))
        with torch.no_grad():
            par = model(x)
            cache = model.get_cache()
            steps = []
            for t in range(36):
                logits, _ = model.step_forward(x[:, t : t + 1], cache, timestamp=t)
                steps.append(logits)
        steps = torch.stack(steps, dim=1)
        self.assertTrue(
            torch.allclose(par, steps, atol=1e-4),
            msg=f"max diff {float((par - steps).abs().max()):.3e}",
        )


class TestV5Recall(unittest.TestCase):
    def test_hard_read_is_exact_induction(self):
        # La lectura dura debe recuperar el token siguiente a la ocurrencia
        # previa cuando q y k aprenden identidad (aqui se fuerza P_q=P_k=I parcial).
        torch.manual_seed(3)
        rt = RecallTap(8, 8, value="next", gap=1, score_chunk=8)
        with torch.no_grad():
            eye = torch.eye(8)
            rt.p_q.weight.copy_(eye)
            rt.p_k.weight.copy_(eye)
        t0 = torch.randn(1, 12, 8)
        t0[0, 4] = t0[0, 2]           # ocurrencia repetida (pos 2 -> 4)
        expected = torch.zeros_like(t0)
        expected[0, 4] = t0[0, 3]     # valor siguiente a la primera ocurrencia
        with torch.no_grad():
            reads = rt.forward_parallel(rt.queries(t0), rt.keys(t0), t0)
        self.assertTrue(torch.allclose(reads[0, 4], expected[0, 4], atol=1e-6))
        # posiciones sin historia valida leen 0
        self.assertTrue(torch.allclose(reads[0, 0], torch.zeros(8), atol=1e-6))

    def test_tie_break_prefers_most_recent(self):
        torch.manual_seed(4)
        rt = RecallTap(8, 8, value="next", gap=1, score_chunk=8)
        with torch.no_grad():
            rt.p_q.weight.copy_(torch.eye(8))
            rt.p_k.weight.copy_(torch.eye(8))
        t0 = torch.randn(1, 16, 8)
        t0[0, 10] = t0[0, 3]          # ocurrencia 1 (temprana)
        t0[0, 14] = t0[0, 3]          # ocurrencia 2 (reciente)
        with torch.no_grad():
            reads = rt.forward_parallel(rt.queries(t0), rt.keys(t0), t0)
        # en la posicion 15 la lectura debe venir de la ocurrencia MAS RECIENTE (14):
        # valor = T0[15]... con gap=1, j<=14, y el valor de j=14 es T0[15].
        # La ocurrencia temprana daria T0[4]; verificamos que no sea esa.
        self.assertFalse(torch.allclose(reads[0, 15], t0[0, 4], atol=1e-6))

    def test_step_read_matches_parallel_read(self):
        torch.manual_seed(5)
        rt = RecallTap(8, 8, value="next", gap=1, score_chunk=8)
        t0 = torch.randn(3, 20, 8)
        with torch.no_grad():
            par = rt.forward_parallel(rt.queries(t0), rt.keys(t0), t0)
            ring = V5Trace(20, 8, 8, horizons=[1] * 4)
            for t in range(20):
                k_t = rt.keys(t0[:, t])
                q_t = rt.queries(t0[:, t])
                ring.append_t0(t0[:, t], k_t)  # escribir antes de leer (contrato V5)
                r = rt.read_step(q_t, ring.k_ring, ring.t0_ring, ring.length)
                self.assertTrue(torch.allclose(par[:, t], r, atol=1e-5),
                                msg=f"pos {t} difiere")

    def test_key_codes_are_isolated(self):
        # K[j] no puede depender de otros tokens (pilar 1).
        torch.manual_seed(6)
        rt = RecallTap(8, 8)
        a = torch.randn(1, 10, 8)
        b = a.clone()
        b[0, 0] = torch.randn(8)      # perturba SOLO el token 0
        ka, kb = rt.keys(a), rt.keys(b)
        self.assertFalse(torch.allclose(ka[0, 0], kb[0, 0], atol=1e-6))
        self.assertTrue(torch.allclose(ka[0, 1:], kb[0, 1:], atol=1e-7))


class TestV5Stability(unittest.TestCase):
    def test_no_nan_fp16_forward(self):
        torch.manual_seed(7)
        model = EngraModel(tiny_cfg()).eval().half()
        x = torch.randint(0, 64, (2, 40))
        with torch.no_grad():
            logits = model(x)
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_no_nan_extreme_inputs(self):
        torch.manual_seed(8)
        model = EngraModel(tiny_cfg()).eval()
        with torch.no_grad():
            for x in (
                torch.zeros(1, 40, dtype=torch.long),
                torch.full((1, 40), 63, dtype=torch.long),
                torch.randint(0, 64, (1, 2)),
            ):
                logits = model(x)
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_training_high_lr_no_nan(self):
        # 10x el lr tipico: la mezcla normalizada y las cotas deben aguantar.
        torch.manual_seed(9)
        model = EngraModel(tiny_cfg())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        for _ in range(30):
            x = torch.randint(0, 64, (4, 32))
            loss = model.forward_loss(x, x)
            self.assertTrue(torch.isfinite(loss).item())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    def test_residual_magnitude_stays_bounded(self):
        # La normalizacion por conteo debe acotar el estado entre capas.
        torch.manual_seed(10)
        model = EngraModel(tiny_cfg()).eval()
        x = torch.randint(0, 64, (2, 40))
        with torch.no_grad():
            t0 = model.footprints(x)
            t = t0
            stds = [float(t.std())]
            for layer in model.consolidation.layers:
                t = layer.forward_train(t, t0=t0)
                stds.append(float(t.std()))
        self.assertLess(max(stds) / max(stds[0], 1e-6), 15.0,
                        msg=f"magnitudes por capa: {[round(s, 2) for s in stds]}")


class TestV5ParamsAndAPI(unittest.TestCase):
    def test_param_count_near_v4(self):
        from engrama.config import EngramaConfig as C4
        torch.manual_seed(11)
        v5 = EngraModel(V5Config(vocab_size=50257, d_model=256, d_gate=32,
                                 d_ff=1024, num_cells=8, num_encoder_layers=2,
                                 num_consolidation_layers=9, context_length=8192,
                                 synapse_rank=32, d_recall=64, rt_layers=(4,)))
        from engrama.model import EngramaModel as M4
        v4 = M4(C4(vocab_size=50257, d_model=256, d_gate=32, d_ff=1024, num_cells=8,
                   num_encoder_layers=2, num_consolidation_layers=9,
                   context_length=512, num_candidates=4, version="v4"))
        p5, p4 = v5.num_parameters(), v4.num_parameters()
        self.assertLess(abs(p5 - p4) / p4, 0.05,
                        msg=f"v5={p5:,} v4={p4:,}")

    def test_cache_memory_linear_in_context(self):
        bytes_per_token = []
        for n in (64, 256, 1024):
            tr = V5Trace(n, 32, 16, horizons=[1] * 3)
            bytes_per_token.append(tr.memory_bytes() / n)
        slope = (bytes_per_token[-1] - bytes_per_token[0]) / (1024 - 64)
        self.assertLess(abs(slope), 5.0, msg=f"crecimiento no lineal: {bytes_per_token}")
        self.assertTrue(all(b >= bytes_per_token[0] * 0.9 for b in bytes_per_token))

    def test_forward_loss_backward_finite(self):
        torch.manual_seed(12)
        model = EngraModel(tiny_cfg())
        x = torch.randint(0, 64, (2, 32))
        loss = model.forward_loss(x, x)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all().item() for g in grads))

    def test_presets_api(self):
        for size in ("tiny", "small", "base", "large"):
            cfg = V5Config.from_preset(size, vocab_size=100)
            self.assertGreater(cfg.d_model, 0)
        model = EngraModel(V5Config.from_preset("tiny", vocab_size=64))
        self.assertGreater(model.num_parameters(), 0)


if __name__ == "__main__":
    unittest.main()
