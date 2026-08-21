"""Equivalence tests for execution-only training optimizations."""

import copy
import unittest

import torch
import torch.nn.functional as F

from engrama import EngramaConfig, EngramaModel
from engrama.losses import chunked_cross_entropy, linear_cross_entropy
from engrama.optimization import DistributedContext, LanguageModelLoss, wrap_ddp


class TestOptimizedLosses(unittest.TestCase):
    def test_custom_chunked_ce_gradient(self):
        torch.manual_seed(12)
        logits = torch.randn(19, 257)
        targets = torch.randint(0, 257, (19,))
        targets[::7] = -100

        reference = logits.clone().requires_grad_(True)
        F.cross_entropy(reference, targets).backward()
        optimized = logits.clone().requires_grad_(True)
        chunked_cross_entropy(optimized, targets, chunk_size=61).backward()

        self.assertLess((reference.grad - optimized.grad).abs().max().item(), 2e-6)

    def test_linear_ce_values_and_gradients(self):
        torch.manual_seed(3)
        hidden = torch.randn(2, 5, 16)
        weight = torch.randn(71, 16)
        targets = torch.randint(0, 71, (2, 5))
        scale = 16 ** -0.5

        h_ref = hidden.clone().requires_grad_(True)
        w_ref = weight.clone().requires_grad_(True)
        ref_logits = F.linear(h_ref, w_ref) * scale
        ref = F.cross_entropy(ref_logits.reshape(-1, 71), targets.reshape(-1))
        ref.backward()

        h_got = hidden.clone().requires_grad_(True)
        w_got = weight.clone().requires_grad_(True)
        got = linear_cross_entropy(
            h_got, w_got, targets, scale=scale, chunk_size=3,
            checkpoint_chunks=True,
        )
        got.backward()

        self.assertAlmostEqual(ref.item(), got.item(), places=6)
        self.assertTrue(torch.allclose(h_ref.grad, h_got.grad, atol=2e-6, rtol=2e-5))
        self.assertTrue(torch.allclose(w_ref.grad, w_got.grad, atol=2e-6, rtol=2e-5))

    def test_model_forward_loss_is_exact(self):
        torch.manual_seed(8)
        config = EngramaConfig(
            vocab_size=97, d_model=32, d_gate=8, d_ff=64, num_cells=2,
            num_encoder_layers=1, num_consolidation_layers=4,
            context_length=16, num_candidates=3, version="v4",
        )
        reference = EngramaModel(config)
        optimized = copy.deepcopy(reference)
        x = torch.randint(0, config.vocab_size, (2, 12))
        y = torch.randint(0, config.vocab_size, (2, 12))

        ref_loss = F.cross_entropy(reference(x).reshape(-1, config.vocab_size), y.reshape(-1))
        ref_loss.backward()
        got_loss = optimized.forward_loss(x, y, linear_chunk_size=7)
        got_loss.backward()

        self.assertAlmostEqual(ref_loss.item(), got_loss.item(), places=5)
        for (name_a, param_a), (name_b, param_b) in zip(
            reference.named_parameters(), optimized.named_parameters()
        ):
            self.assertEqual(name_a, name_b)
            self.assertTrue(
                torch.allclose(param_a.grad, param_b.grad, atol=2e-5, rtol=2e-4),
                name_a,
            )

    def test_loss_wrapper_and_single_process_ddp_noop(self):
        config = EngramaConfig(
            vocab_size=41, d_model=24, d_gate=6, d_ff=48, num_cells=2,
            num_encoder_layers=1, num_consolidation_layers=3,
            context_length=8, version="v4", synapse_rank=8,
        )
        model = EngramaModel(config)
        wrapped = LanguageModelLoss(model, linear_chunk_size=4)
        self.assertIs(wrap_ddp(wrapped, DistributedContext()), wrapped)
        x = torch.randint(0, 41, (2, 8))
        self.assertTrue(torch.isfinite(wrapped(x, x)))


if __name__ == "__main__":
    unittest.main()
