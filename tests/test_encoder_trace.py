"""Encoder isolation tests (Phase 1 theorem) and cache tests (Phase 2)."""

import unittest

import torch

from engrama.config import EngramaConfig
from engrama.encoder import IsolatedEncoder
from engrama.trace import CircularTrace, EngramaCache


def _config(**kw):
    base = dict(
        vocab_size=32, d_model=32, d_gate=8, d_ff=64, num_cells=4,
        num_encoder_layers=2, num_consolidation_layers=4, context_length=16,
        num_candidates=2,
    )
    base.update(kw)
    return EngramaConfig(**base)


class TestIsolatedEncoder(unittest.TestCase):
    def test_forward_shape(self):
        encoder = IsolatedEncoder(_config())
        out = encoder(torch.randn(2, 8, 32))
        self.assertEqual(out.shape, (2, 8, 32))

    def test_token_isolation_theorem(self):
        """T_0[i] must not change when any other token x_j (j != i) changes."""
        for mode in ("factorized", "dense"):
            torch.manual_seed(0)
            encoder = IsolatedEncoder(_config(synapse_mode=mode)).eval()
            x = torch.randn(1, 10, 32)
            with torch.no_grad():
                t0 = encoder(x)
                for j in (0, 5, 9):
                    x2 = x.clone()
                    x2[:, j] = torch.randn(32)
                    t0_2 = encoder(x2)
                    mask = torch.ones(10, dtype=torch.bool)
                    mask[j] = False
                    diff = (t0_2[:, mask] - t0[:, mask]).abs().max().item()
                    self.assertLess(
                        diff, 1e-6,
                        f"Isolation violated at j={j} in mode={mode}: {diff}",
                    )

    def test_batch_independence(self):
        """Samples in a batch must not influence each other."""
        encoder = IsolatedEncoder(_config()).eval()
        xa = torch.randn(1, 6, 32)
        xb = torch.randn(1, 6, 32)
        with torch.no_grad():
            joint = encoder(torch.cat([xa, xb], dim=0))
            self.assertLess(
                (joint[0:1] - encoder(xa)).abs().max().item(), 1e-6
            )
            self.assertLess(
                (joint[1:2] - encoder(xb)).abs().max().item(), 1e-6
            )


class TestCircularTrace(unittest.TestCase):
    def test_fifo_circular_overwrite(self):
        trace = CircularTrace(3)
        for i in range(5):
            trace.append(torch.tensor([float(i)]), i)
        self.assertEqual(len(trace), 3)
        self.assertEqual(trace.timestamps, [2, 3, 4])
        self.assertEqual(trace.values[0].item(), 2.0)
        self.assertEqual(trace.latest().item(), 4.0)

    def test_history_and_clear(self):
        trace = CircularTrace(4)
        for i in range(3):
            trace.append(torch.full((2,), float(i)), i)
        hist = trace.history(2)
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[-1].mean().item(), 2.0)
        trace.clear()
        self.assertEqual(len(trace), 0)


class TestEngramaCache(unittest.TestCase):
    def test_hierarchical_horizons_enforced(self):
        cfg = _config()
        cache = EngramaCache(
            N_max=cfg.context_length,
            num_layers=cfg.num_consolidation_layers,
            d_model=cfg.d_model,
            mode="hierarchical",
            horizons=cfg.cache_horizons(),
        )
        for t in range(30):
            t0 = torch.full((1, cfg.d_model), float(t))
            tls = [torch.full((1, cfg.d_model), float(t)) for _ in range(4)]
            cache.append(t0, tls, t)
        caps = cfg.cache_horizons()
        for l, buf in enumerate(cache.states.buffers):
            self.assertEqual(len(buf), caps[l])
        self.assertEqual(len(cache.trace), cfg.context_length)
        self.assertEqual(cache.tokens_written, 30)

    def test_full_mode_keeps_window(self):
        cache = EngramaCache(N_max=5, num_layers=2, d_model=4, mode="full")
        for t in range(8):
            cache.append(
                torch.full((1, 4), float(t)),
                [torch.full((1, 4), float(t))] * 2,
                t,
            )
        self.assertEqual(len(cache.trace), 5)
        self.assertEqual([len(b) for b in cache.states.buffers], [5, 5])

    def test_state_reduction(self):
        cfg = _config()
        cache = EngramaCache(
            N_max=cfg.context_length,
            num_layers=4,
            d_model=32,
            mode="hierarchical",
            horizons=cfg.cache_horizons(),
        )
        desc = cache.describe()
        self.assertGreater(desc["state_reduction_ratio"], 1.0)

    def test_views_and_device_move(self):
        cache = EngramaCache(N_max=4, num_layers=2, d_model=8, mode="full")
        for t in range(3):
            cache.append(torch.randn(1, 8), [torch.randn(1, 8)] * 2, t)
        cache.to("cpu")
        self.assertEqual(len(cache.T0), 3)
        self.assertEqual(len(cache.Tl[0]), 3)
        self.assertEqual(cache.timestamps, [0, 1, 2])
        self.assertGreater(cache.get_memory_footprint(), 0)


if __name__ == "__main__":
    unittest.main()
