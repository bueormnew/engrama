"""Canonical causal invariance verification (V1/V2 paper section 6, V3 spec
sections 23-24 and 45).

Guarantee: the parallel pass ``forward(seq)`` equals the incremental
``step_forward(seq)`` with max error < 1e-4, in both cache modes --
``full`` (V2) and ``hierarchical`` (V3 minimum-horizon theorem).
"""

import unittest
import warnings

import torch

from engrama.config import EngramaConfig
from engrama.model import EngramaModel

warnings.simplefilter("ignore")


def _build(version: str) -> EngramaModel:
    cfg = EngramaConfig(
        version=version,
        vocab_size=32, d_model=64, d_gate=16, d_ff=128, num_cells=4,
        num_encoder_layers=2, num_consolidation_layers=3, context_length=64,
        num_candidates=2, dropout=0.0,
    )
    torch.manual_seed(42)
    model = EngramaModel(cfg)
    model.eval()
    return model


class TestCausalInvariance(unittest.TestCase):
    def test_v3_both_cache_modes(self):
        self._verify(_build("v3"))

    def test_v2_full_cache(self):
        self._verify(_build("v2"), modes=("full",))

    def _verify(self, model: EngramaModel, modes=("full", "hierarchical")):
        torch.manual_seed(42)
        x = torch.randint(0, model.config.vocab_size, (20,))

        with torch.no_grad():
            logits_full = model.forward(x.unsqueeze(0))

            for mode in modes:
                with self.subTest(cache_mode=mode):
                    cache = model.get_cache(mode=mode)
                    for t in range(len(x)):
                        logits_step, _ = model.step_forward(
                            x[t : t + 1].unsqueeze(0), cache, timestamp=t
                        )
                        max_abs_diff = (
                            logits_full[:, t, :] - logits_step
                        ).abs().max().item()
                        self.assertLess(
                            max_abs_diff, 1e-4,
                            f"Invariance failed at t={t} [{mode}]: {max_abs_diff}",
                        )

            # Cached and uncached generation must agree token by token.
            prompt = [2, 5, 8]
            torch.manual_seed(123)
            gen_cached = model.generate(prompt, max_new_tokens=15,
                                        temperature=0.8, top_k=5, use_cache=True)
            torch.manual_seed(123)
            gen_plain = model.generate(prompt, max_new_tokens=15,
                                       temperature=0.8, top_k=5, use_cache=False)
            self.assertEqual(gen_cached, gen_plain)


if __name__ == "__main__":
    unittest.main()
