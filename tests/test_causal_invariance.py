"""Causal invariance test matrix (V3 spec sections 23-24, 45).

For every architecture mode combination, the parallel training forward must
equal the incremental cached step-by-step computation within 1e-4, under
both cache modes, including the FIFO overflow regime (when the dependency
cone of the compared positions fits inside the retained window).
"""

import unittest
import warnings

import torch

from engrama.config import EngramaConfig
from engrama.model import EngramaModel

warnings.simplefilter("ignore")

BASE = dict(
    vocab_size=40, d_model=48, d_gate=12, d_ff=96, num_cells=3,
    num_encoder_layers=2, num_consolidation_layers=4, context_length=32,
    num_candidates=3,
)

MODE_MATRIX = {
    "v3_default": {},
    "v3_binary_minimal": {"offset_mode": "binary_minimal"},
    "v3_dense_dilated": {"offset_mode": "dense_dilated"},
    "v3_no_hierarchical_gate": {"hierarchical_gate": False},
    "v3_no_identity_transport": {"identity_transport": False},
    "v3_global_anchor": {"global_anchor": True},
    "v3_unstable_init": {"stable_init": False},
    "v3_mean": {"candidate_aggregation": "mean"},
    "v3_max": {"candidate_aggregation": "max"},
    "v3_single_candidate": {"num_candidates": 1},
    "v2_preset": {"version": "v2"},
    "v1_preset": {"version": "v1"},
    "dense_synapses_ablation": {"synapse_mode": "dense", "cell_mode": "independent"},
    "factorized_dense_offsets_ablation": {
        "offset_mode": "dense_dilated", "cell_mode": "independent",
    },
    "untied_embeddings": {"tie_embeddings": False},
    "relu": {"activation": "relu"},
    "silu": {"activation": "silu"},
}


class TestCausalEquivalenceMatrix(unittest.TestCase):
    def _run_pair(self, name, overrides, seq_len=14):
        cfg = EngramaConfig(**{**BASE, **overrides})
        torch.manual_seed(0)
        model = EngramaModel(cfg).eval()
        seq = torch.randint(0, cfg.vocab_size, (1, seq_len))

        with torch.no_grad():
            logits_full = model.forward(seq)
            for mode in ("full", "hierarchical"):
                cache = model.get_cache(N_max=seq_len, mode=mode)
                step_logits = [
                    model.step_forward(seq[:, t : t + 1], cache, t)[0]
                    for t in range(seq_len)
                ]
                step = torch.stack(step_logits, dim=1)
                diff = (logits_full.float() - step.float()).abs().max().item()
                self.assertLess(
                    diff, 1e-4,
                    f"{name} [{mode} cache]: invariance failed, max diff {diff}",
                )

    def test_all_modes(self):
        for name, overrides in MODE_MATRIX.items():
            with self.subTest(mode=name):
                self._run_pair(name, overrides)


class TestStrictCausality(unittest.TestCase):
    """Spec section 4.5: changing the future must not alter past states."""

    def test_future_tokens_do_not_change_past(self):
        cfg = EngramaConfig(**BASE)
        torch.manual_seed(1)
        model = EngramaModel(cfg).eval()
        seq = torch.randint(0, cfg.vocab_size, (1, 20))
        seq2 = seq.clone()
        seq2[:, 10:] = torch.randint(0, cfg.vocab_size, (1, 10))
        with torch.no_grad():
            out1 = model.forward(seq)
            out2 = model.forward(seq2)
        diff = (out1[:, :10] - out2[:, :10]).abs().max().item()
        self.assertLess(diff, 1e-5)


class TestOverflowWindowEquivalence(unittest.TestCase):
    """After FIFO overflow, incremental output equals the windowed forward
    whenever the dependency cone fits inside the window (spec 26 rule)."""

    def test_overflow_equivalence(self):
        cfg = EngramaConfig(
            vocab_size=40, d_model=48, d_gate=12, d_ff=96, num_cells=2,
            num_encoder_layers=1, num_consolidation_layers=3,
            context_length=16, num_candidates=2,
        )
        torch.manual_seed(7)
        model = EngramaModel(cfg).eval()
        seq = torch.randint(0, cfg.vocab_size, (1, 24))
        with torch.no_grad():
            win_logits = model.forward(seq[:, 8:24])[:, -1, :]
            for mode in ("full", "hierarchical"):
                cache = model.get_cache(mode=mode)
                last = None
                for t in range(24):
                    last, _ = model.step_forward(seq[:, t : t + 1], cache, t)
                diff = (win_logits.float() - last.float()).abs().max().item()
                self.assertLess(diff, 1e-4, f"overflow mismatch [{mode}]: {diff}")


class TestGenerationEquivalence(unittest.TestCase):
    def test_cached_and_uncached_generation_match(self):
        cfg = EngramaConfig(**BASE)
        torch.manual_seed(0)
        model = EngramaModel(cfg).eval()
        prompt = [2, 5, 8, 13]
        torch.manual_seed(123)
        cached = model.generate(prompt, max_new_tokens=10, temperature=0.8,
                                top_k=5, use_cache=True)
        torch.manual_seed(123)
        plain = model.generate(prompt, max_new_tokens=10, temperature=0.8,
                               top_k=5, use_cache=False)
        self.assertEqual(cached, plain)


if __name__ == "__main__":
    unittest.main()
