"""Tests for ENGRAMA V4 architecture, dual gating, trace tap, and latent fusion."""

import unittest
import torch
import torch.nn.functional as F

from engrama.config import EngramaConfig
from engrama.model import EngramaModel
from engrama.trace import EngramaCache
from engrama.evoker import MultiCandidateEvoker


class TestV4Architecture(unittest.TestCase):
    def test_v4_config_defaults(self):
        cfg = EngramaConfig(version="v4")
        self.assertEqual(cfg.version, "v4")
        self.assertEqual(cfg.gating_mode, "dual")
        self.assertTrue(cfg.trace_tap)
        self.assertEqual(cfg.norm_type, "rmsnorm")
        self.assertEqual(cfg.candidate_aggregation, "latent_fusion")
        self.assertEqual(cfg.offset_mode, "resonant_multirate")

    def test_v4_forward_shape(self):
        cfg = EngramaConfig(
            vocab_size=100,
            d_model=64,
            d_gate=16,
            d_ff=256,
            num_cells=2,
            num_encoder_layers=1,
            num_consolidation_layers=4,
            context_length=64,
            version="v4",
        )
        model = EngramaModel(cfg)
        x = torch.randint(0, 100, (2, 32))
        logits = model(x)
        self.assertEqual(logits.shape, (2, 32, 100))

    def test_v4_causal_invariance_hierarchical_cache(self):
        """Forward train and step_forward with hierarchical cache match exactly."""
        cfg = EngramaConfig(
            vocab_size=64,
            d_model=48,
            d_gate=12,
            d_ff=192,
            num_cells=2,
            num_encoder_layers=1,
            num_consolidation_layers=4,
            context_length=32,
            version="v4",
            cache_mode="hierarchical",
        )
        model = EngramaModel(cfg)
        model.eval()

        torch.manual_seed(42)
        x = torch.randint(0, 64, (2, 24))

        with torch.no_grad():
            full_logits = model(x)
            cache = model.get_cache(N_max=32, mode="hierarchical")
            step_logits_list = []
            for t in range(24):
                tok = x[:, t : t + 1]
                log_t, _ = model.step_forward(tok, cache, timestamp=t)
                step_logits_list.append(log_t)
            step_logits = torch.stack(step_logits_list, dim=1)

        diff = (full_logits - step_logits).abs().max().item()
        self.assertLess(
            diff,
            1e-4,
            f"V4 causal invariance failed with diff {diff}",
        )

    def test_v4_causal_invariance_full_cache(self):
        """Forward train and step_forward with full cache match exactly."""
        cfg = EngramaConfig(
            vocab_size=64,
            d_model=48,
            d_gate=12,
            d_ff=192,
            num_cells=2,
            num_encoder_layers=1,
            num_consolidation_layers=4,
            context_length=32,
            version="v4",
            cache_mode="full",
        )
        model = EngramaModel(cfg)
        model.eval()

        torch.manual_seed(42)
        x = torch.randint(0, 64, (2, 24))

        with torch.no_grad():
            full_logits = model(x)
            cache = model.get_cache(N_max=32, mode="full")
            step_logits_list = []
            for t in range(24):
                tok = x[:, t : t + 1]
                log_t, _ = model.step_forward(tok, cache, timestamp=t)
                step_logits_list.append(log_t)
            step_logits = torch.stack(step_logits_list, dim=1)

        diff = (full_logits - step_logits).abs().max().item()
        self.assertLess(
            diff,
            1e-4,
            f"V4 full cache causal invariance failed with diff {diff}",
        )

    def test_v4_latent_fusion_evoker(self):
        cfg = EngramaConfig(
            vocab_size=500,
            d_model=64,
            num_candidates=4,
            candidate_aggregation="latent_fusion",
            version="v4",
        )
        evoker = MultiCandidateEvoker(cfg)
        h_star = torch.randn(2, 16, 64)
        emb_weights = torch.randn(500, 64)
        logits = evoker(h_star, emb_weights)
        self.assertEqual(logits.shape, (2, 16, 500))

    def test_v4_resonant_receptive_field(self):
        cfg = EngramaConfig(
            num_consolidation_layers=8,
            context_length=256,
            offset_mode="resonant_multirate",
            version="v4",
        )
        rf = cfg.receptive_field()
        self.assertTrue(rf["covers_context"])
        self.assertTrue(rf["dense_coverage"])
        self.assertGreaterEqual(rf["max_reach"], 255)


if __name__ == "__main__":
    unittest.main()

    def test_dual_bilinear_clamp_causal_invariance(self):
        """La variante opt-in dual_bilinear_clamp conserva la equivalencia
        forward_train == step_forward y acota la pre-activacion bilineal."""
        torch.manual_seed(11)
        cfg = EngramaConfig(
            vocab_size=48, d_model=32, d_gate=8, d_ff=64, num_cells=2,
            num_encoder_layers=1, num_consolidation_layers=4,
            context_length=24, num_candidates=2, version="v4",
            synapse_rank=8, dual_bilinear_clamp=4.0,
        )
        model = EngramaModel(cfg)
        model.eval()
        x = torch.randint(0, 48, (2, 20))
        with torch.no_grad():
            parallel = model(x)
            cache = model.get_cache(N_max=20, mode="hierarchical")
            for t in range(20):
                step_logits, _ = model.step_forward(x[:, t : t + 1], cache, timestamp=t)
        self.assertTrue(
            torch.allclose(parallel[:, -1], step_logits, atol=1e-4),
            msg="clamp rompe la invarianza causal",
        )
        # el clamp no debe cambiar el modo source ni el default (None)
        cfg_plain = EngramaConfig(
            vocab_size=48, d_model=32, d_gate=8, d_ff=64, num_cells=2,
            num_encoder_layers=1, num_consolidation_layers=4,
            context_length=24, num_candidates=2, version="v4", synapse_rank=8,
        )
        self.assertIsNone(cfg_plain.dual_bilinear_clamp)
