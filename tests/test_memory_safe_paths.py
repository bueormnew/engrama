"""Tests for the memory-safe large-vocabulary paths (evoker + loss).

Covers:

- ``chunked_cross_entropy`` equivalence with ``F.cross_entropy``
  (values, ignore_index masking, reductions, gradients).
- The evoker's checkpointed chunked logsumexp/max aggregation: same
  logits as the plain path when forced, working backward pass, and
  preserved causal invariance under the chunked path.
- ``Trainer`` automatically switching to the chunked loss for large
  vocabularies.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""

import unittest
import warnings

import torch
import torch.nn.functional as F

import engrama
from engrama import EngramaConfig, EngramaModel, EngramaTokenizer, Trainer
from engrama.losses import chunked_cross_entropy

warnings.simplefilter("ignore")


class TestChunkedCrossEntropy(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.n, self.vocab = 37, 512
        self.logits = torch.randn(self.n, self.vocab, dtype=torch.float32)
        self.targets = torch.randint(0, self.vocab, (self.n,))

    def test_matches_cross_entropy_mean(self):
        ref = F.cross_entropy(self.logits, self.targets)
        got = chunked_cross_entropy(self.logits, self.targets, chunk_size=97)
        self.assertAlmostEqual(ref.item(), got.item(), places=5)

    def test_matches_cross_entropy_with_ignore(self):
        targets = self.targets.clone()
        targets[::5] = -100
        ref = F.cross_entropy(self.logits, targets, ignore_index=-100)
        got = chunked_cross_entropy(self.logits, targets, chunk_size=97)
        self.assertAlmostEqual(ref.item(), got.item(), places=5)

    def test_reductions(self):
        for reduction in ("mean", "sum"):
            ref = F.cross_entropy(self.logits, self.targets, reduction=reduction)
            got = chunked_cross_entropy(
                self.logits, self.targets, chunk_size=97, reduction=reduction
            )
            self.assertAlmostEqual(ref.item(), got.item(), places=5)

        per = chunked_cross_entropy(
            self.logits, self.targets, chunk_size=97, reduction="none"
        )
        self.assertEqual(per.shape, (self.n,))

    def test_gradient_matches(self):
        ref = self.logits.clone().requires_grad_(True)
        F.cross_entropy(ref, self.targets).backward()

        got = self.logits.clone().requires_grad_(True)
        chunked_cross_entropy(got, self.targets, chunk_size=97).backward()

        self.assertLess((ref.grad - got.grad).abs().max().item(), 1e-5)

    def test_shapes_any_leading_dims(self):
        logits = torch.randn(30, 512, dtype=torch.float32).reshape(2, 3, 5, -1)
        targets = torch.randint(0, 512, (2, 3, 5))
        ref = F.cross_entropy(logits.reshape(-1, 512), targets.reshape(-1))
        got = chunked_cross_entropy(logits, targets, chunk_size=64)
        self.assertAlmostEqual(ref.item(), got.item(), places=5)


class TestEvokerChunkedPath(unittest.TestCase):
    """Force the chunked path on a small model and compare with the plain one."""

    def _config(self, aggregation):
        return EngramaConfig(
            vocab_size=129,
            d_model=48,
            d_gate=12,
            d_ff=96,
            num_cells=2,
            num_encoder_layers=1,
            num_consolidation_layers=4,
            context_length=64,
            num_candidates=4,
            candidate_aggregation=aggregation,
            version="v3",
        )

    def test_logsumexp_plain_vs_chunked(self):
        import engrama.evoker as evoker_mod

        torch.manual_seed(3)
        cfg = self._config("logsumexp")
        model = EngramaModel(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 32))

        with torch.no_grad():
            plain = model(x)

        old_threshold = evoker_mod._MAX_AGGREGATE_ELEMENTS
        evoker_mod._MAX_AGGREGATE_ELEMENTS = 1  # force chunked path
        try:
            with torch.no_grad():
                chunked = model(x)
        finally:
            evoker_mod._MAX_AGGREGATE_ELEMENTS = old_threshold

        self.assertEqual(plain.shape, chunked.shape)
        self.assertTrue(torch.allclose(plain, chunked, atol=1e-5, rtol=1e-5))

    def test_max_plain_vs_chunked(self):
        import engrama.evoker as evoker_mod

        torch.manual_seed(3)
        cfg = self._config("max")
        model = EngramaModel(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 32))

        with torch.no_grad():
            plain = model(x)

        old_threshold = evoker_mod._MAX_AGGREGATE_ELEMENTS
        evoker_mod._MAX_AGGREGATE_ELEMENTS = 1
        try:
            with torch.no_grad():
                chunked = model(x)
        finally:
            evoker_mod._MAX_AGGREGATE_ELEMENTS = old_threshold

        self.assertTrue(torch.allclose(plain, chunked, atol=1e-5, rtol=1e-5))

    def test_chunked_path_backward_and_gradients_match(self):
        import engrama.evoker as evoker_mod

        torch.manual_seed(11)
        cfg = self._config("logsumexp")
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        y = torch.randint(0, cfg.vocab_size, (2, 16))

        model_plain = EngramaModel(cfg)
        loss_plain = chunked_cross_entropy(model_plain(x), y, chunk_size=64)
        loss_plain.backward()
        grads_plain = {k: v.grad.clone() for k, v in model_plain.named_parameters()}

        old_threshold = evoker_mod._MAX_AGGREGATE_ELEMENTS
        evoker_mod._MAX_AGGREGATE_ELEMENTS = 1
        try:
            model_chunked = EngramaModel(cfg)
            model_chunked.load_state_dict(model_plain.state_dict())
            loss_chunked = chunked_cross_entropy(
                model_chunked(x), y, chunk_size=64
            )
            loss_chunked.backward()
        finally:
            evoker_mod._MAX_AGGREGATE_ELEMENTS = old_threshold

        self.assertAlmostEqual(loss_plain.item(), loss_chunked.item(), places=4)
        for key, grad in grads_plain.items():
            got = dict(model_chunked.named_parameters())[key].grad
            self.assertIsNotNone(got, f"no grad on {key} in chunked path")
            self.assertTrue(
                torch.allclose(grad, got, atol=1e-4, rtol=1e-4),
                f"grad mismatch on {key}",
            )

    def test_causal_invariance_holds_under_chunked_path(self):
        import engrama.evoker as evoker_mod

        torch.manual_seed(5)
        cfg = self._config("logsumexp")
        model = EngramaModel(cfg).eval()
        x = torch.randint(0, cfg.vocab_size, (2, 32))

        old_threshold = evoker_mod._MAX_AGGREGATE_ELEMENTS
        evoker_mod._MAX_AGGREGATE_ELEMENTS = 1
        try:
            with torch.no_grad():
                full = model(x)
                cache = model.get_cache(N_max=32, mode="hierarchical")
                steps = [
                    model.step_forward(x[:, t : t + 1], cache, t)[0]
                    for t in range(32)
                ]
                inc = torch.stack(steps, dim=1)
        finally:
            evoker_mod._MAX_AGGREGATE_ELEMENTS = old_threshold

        self.assertLess((full - inc).abs().max().item(), 1e-4)


class TestTrainerLargeVocab(unittest.TestCase):
    def test_trainer_switches_to_chunked_loss(self):
        torch.manual_seed(1)
        vocab = 20000  # above _LARGE_VOCAB_THRESHOLD
        text = ("the quick brown fox jumps over the lazy dog. " * 8)
        tok = EngramaTokenizer().fit_on_text(text)
        cfg = EngramaConfig(
            vocab_size=vocab,
            d_model=32,
            d_gate=8,
            d_ff=64,
            num_cells=2,
            num_encoder_layers=1,
            num_consolidation_layers=4,
            context_length=32,
            num_candidates=2,
            candidate_aggregation="mean",
            tie_embeddings=True,
        )
        model = EngramaModel(cfg)
        # ids from the small char tokenizer stay far below the padded vocab.
        trainer = Trainer(model, lr=1e-3)
        import engrama.datasets as ds_mod

        ds = ds_mod.TextDataset(text, tok, sequence_length=16)
        history = trainer.fit(ds, batch_size=4, epochs=1)
        self.assertEqual(len(history), 1)
        self.assertTrue(torch.isfinite(torch.tensor(history[0])))


if __name__ == "__main__":
    unittest.main()
