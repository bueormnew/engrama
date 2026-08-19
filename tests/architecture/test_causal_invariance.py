import unittest

import torch

from engrama.config import EngramaConfig
from engrama.model import EngramaModel


class TestCausalInvariance(unittest.TestCase):
    def test_causal_invariance(self):
        # 1. Initialize an ENGRAMA model with config version='v2'
        config = EngramaConfig(
            version="v2",
            vocab_size=32,
            d_model=64,
            d_gate=16,
            d_ff=128,
            num_cells=4,
            num_encoder_layers=2,
            num_consolidation_layers=3,
            context_length=64,
            offsets=[0, 1, 2, 4],
            num_candidates=2,
            dropout=0.0,
        )
        model = EngramaModel(config)
        model.eval()

        # 2. Generate or define a simple sequence x of length 20 (char ids)
        torch.manual_seed(42)
        x = torch.randint(0, config.vocab_size, (20,))

        # 3. forward_full = model.forward(x.unsqueeze(0)) (parallel)
        with torch.no_grad():
            logits_full = model.forward(x.unsqueeze(0))

        # 4 & 5. Do a manual generation step by step with step_forward
        cache = model.get_cache()
        with torch.no_grad():
            for t in range(len(x)):
                token_tensor = x[t : t + 1].unsqueeze(0)
                logits_step_t, _ = model.step_forward(token_tensor, cache, timestamp=t)

                # 6 & 7. Compare logits_full[:, t, :].detach() vs logits_step_t.detach()
                max_abs_diff = (logits_full[:, t, :].detach() - logits_step_t.detach()).abs().max().item()
                self.assertLess(
                    max_abs_diff,
                    1e-4,
                    f"Causal invariance failed at position {t} with diff {max_abs_diff}",
                )

        # 8. Assert model.generate(prompt, use_cache=True) == model.generate(prompt, use_cache=False)
        prompt = [2, 5, 8]
        torch.manual_seed(123)
        gen_cached = model.generate(prompt, max_new_tokens=15, temperature=0.8, top_k=5, use_cache=True)

        torch.manual_seed(123)
        gen_nocache = model.generate(prompt, max_new_tokens=15, temperature=0.8, top_k=5, use_cache=False)

        self.assertEqual(gen_cached, gen_nocache)


if __name__ == "__main__":
    unittest.main()
