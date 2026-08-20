"""Evoker tests: candidates, aggregations and V3 mean optimization (spec 14)."""

import math
import unittest
import warnings

import torch

from engrama.config import EngramaConfig
from engrama.evoker import MultiCandidateEvoker

warnings.simplefilter("ignore")


def _config(**kw):
    base = dict(
        vocab_size=24, d_model=32, d_gate=8, d_ff=64, num_cells=2,
        num_encoder_layers=1, num_consolidation_layers=2, context_length=16,
        num_candidates=4,
    )
    base.update(kw)
    return EngramaConfig(**base)


class TestEvokerShapes(unittest.TestCase):
    def test_logit_shapes(self):
        for mode in ("factorized", "dense"):
            ev = MultiCandidateEvoker(_config(evoker_mode=mode))
            h = torch.randn(2, 5, 32)
            E = torch.randn(24, 32)
            self.assertEqual(ev(h, E).shape, (2, 5, 24))

    def test_m1_collapses_to_linear_classifier(self):
        ev = MultiCandidateEvoker(_config(num_candidates=1))
        h = torch.randn(3, 32)
        E = torch.randn(24, 32)
        for agg in ("logsumexp", "max", "mean"):
            ev.aggregation = agg
            out1 = ev(h, E)
            c = ev.candidates_forward(h).squeeze(-2)
            ref = (c @ E.T) / math.sqrt(32)
            self.assertLess((out1 - ref).abs().max().item(), 1e-5)


class TestMeanOptimization(unittest.TestCase):
    """Spec 14.2/37: mean-aggregated candidates then one vocab matmul equals
    the naive per-candidate mean of logits (linearity)."""

    def test_mean_opt_equals_naive(self):
        ev = MultiCandidateEvoker(_config(candidate_aggregation="mean"))
        h = torch.randn(2, 7, 32)
        E = torch.randn(24, 32)

        fast = ev(h, E)

        cands = ev.candidates_forward(h)  # (..., M, d)
        scale = 1.0 / math.sqrt(32)
        naive = (torch.einsum("...md,vd->...mv", cands, E) * scale).mean(dim=-2)
        self.assertLess((fast - naive).abs().max().item(), 1e-5)


class TestLogSumExpStability(unittest.TestCase):
    def test_lse_matches_reference(self):
        ev = MultiCandidateEvoker(_config(candidate_aggregation="logsumexp"))
        h = torch.randn(2, 32)
        E = torch.randn(24, 32)
        out = ev(h, E)
        cands = ev.candidates_forward(h)
        scale = 1.0 / math.sqrt(32)
        logits = torch.einsum("bmd,vd->bmv", cands, E) * scale
        ref = torch.logsumexp(logits, dim=-2)
        self.assertLess((out - ref).abs().max().item(), 1e-5)

    def test_lse_large_values_no_overflow(self):
        ev = MultiCandidateEvoker(_config(candidate_aggregation="logsumexp"))
        h = torch.randn(2, 32) * 1e6
        E = torch.randn(24, 32) * 1e6
        out = ev(h, E.float())
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
