"""End-to-end ecosystem tests: training, generation, serialization, quick API."""

import os
import shutil
import tempfile
import unittest
import warnings

import torch

import engrama
from engrama import (
    BenchmarkSuite,
    EngramaConfig,
    EngramaModel,
    EngramaTokenizer,
    Generator,
    TextDataset,
    Trainer,
    load_model,
    save_model,
)

warnings.simplefilter("ignore")

TEXT = (
    "el gato come pescado. la casa es grande. el perro ladra fuerte. "
    "el cielo es azul. la flor crece lento. el agua fluye clara. "
) * 30


def _small_config(vocab_size):
    return EngramaConfig(
        vocab_size=vocab_size, d_model=48, d_gate=12, d_ff=96, num_cells=2,
        num_encoder_layers=1, num_consolidation_layers=4, context_length=64,
        num_candidates=2,
    )


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.tok = EngramaTokenizer().fit_on_text(TEXT)
        self.cfg = _small_config(self.tok.vocab_size)
        torch.manual_seed(0)
        self.model = EngramaModel(self.cfg)

    def test_loss_decreases(self):
        ds = TextDataset(TEXT, self.tok, sequence_length=32)
        trainer = Trainer(self.model, lr=5e-3)
        history = trainer.fit(ds, batch_size=8, epochs=6)
        self.assertEqual(len(history), 6)
        self.assertLess(history[-1], history[0])
        # the run must show real learning, not just jitter
        self.assertLess(history[-1], 0.75 * history[0])

    def test_warmup_and_cosine_schedulers(self):
        ds = TextDataset(TEXT, self.tok, sequence_length=32)
        for sched in ("warmup", "cosine"):
            trainer = Trainer(self.model, lr=1e-3, scheduler=sched, warmup_steps=3)
            history = trainer.fit(ds, batch_size=8, epochs=2)
            self.assertEqual(len(history), 2)
        with self.assertRaises(ValueError):
            Trainer(self.model, scheduler="warpdrive")

    def test_evaluate(self):
        ds = TextDataset(TEXT, self.tok, sequence_length=32)
        trainer = Trainer(self.model, lr=5e-3)
        trainer.fit(ds, batch_size=8, epochs=2)
        self.assertIsInstance(trainer.evaluate(ds), float)


class TestSerialization(unittest.TestCase):
    def test_roundtrip_all_modes(self):
        for overrides in (
            {},
            {"version": "v2"},
            {"tie_embeddings": False},
            {"offset_mode": "binary_minimal", "cache_mode": "full"},
        ):
            with self.subTest(overrides=overrides):
                tok = EngramaTokenizer().fit_on_text(TEXT)
                cfg_kwargs = dict(
                    vocab_size=tok.vocab_size, d_model=48, d_gate=12, d_ff=96,
                    num_cells=2, num_encoder_layers=1, num_consolidation_layers=4,
                    context_length=64, num_candidates=2,
                )
                cfg_kwargs.update(overrides)
                cfg = EngramaConfig(**cfg_kwargs)
                torch.manual_seed(3)
                model = EngramaModel(cfg)
                tmp = tempfile.mkdtemp()
                try:
                    save_model(model, tmp, tok)
                    loaded, loaded_tok = load_model(tmp)
                    self.assertEqual(loaded_tok.vocab_size, tok.vocab_size)
                    self.assertEqual(
                        loaded.config.to_dict(), model.config.to_dict()
                    )
                    sd1, sd2 = model.state_dict(), loaded.state_dict()
                    for k in sd1:
                        self.assertTrue(
                            torch.equal(sd1[k], sd2[k]), f"weight mismatch: {k}"
                        )
                finally:
                    shutil.rmtree(tmp)

    def test_missing_files_raise(self):
        with self.assertRaises(FileNotFoundError):
            load_model("/tmp/engrama_does_not_exist_dir")


class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.tok = EngramaTokenizer().fit_on_text(TEXT)
        torch.manual_seed(0)
        self.model = EngramaModel(_small_config(self.tok.vocab_size))

    def test_generate_and_stream_counts(self):
        gen = Generator(self.model, self.tok)
        out = gen.generate("el", max_new_tokens=7)
        self.assertIsInstance(out, str)
        streamed = "".join(gen.generate_stream("el", max_new_tokens=5))
        # stream returns the completion (prompt excluded)
        self.assertEqual(len(streamed), 5)

    def test_eos_stop(self):
        ids = self.model.generate([2], max_new_tokens=10, eos_token_id=3)
        self.assertLessEqual(len(ids), 11)


class TestQuickAPI(unittest.TestCase):
    def test_create_model_all_sizes(self):
        for size in ("tiny", "small", "base"):
            model = engrama.create_model(size=size, vocab_size=32)
            self.assertIsInstance(model, EngramaModel)
            self.assertGreater(model.num_parameters(), 0)

    def test_quickstart_end_to_end(self):
        run = engrama.quickstart(
            TEXT, size="tiny", epochs=2, batch_size=8, verbose=False, seed=0
        )
        self.assertEqual(len(run.history), 2)
        self.assertLess(run.history[-1], run.history[0])
        self.assertIsInstance(run.generate("el cielo", max_new_tokens=4), str)
        self.assertIn("ENGRAMA", run.summary())
        with tempfile.TemporaryDirectory() as tmp:
            run.save(tmp)
            run2 = engrama.load_quick(tmp)
            self.assertEqual(
                run2.model.num_parameters(), run.model.num_parameters()
            )

    def test_list_sizes(self):
        sizes = engrama.list_sizes()
        self.assertEqual(set(sizes), {"tiny", "small", "base", "large"})


class TestBenchmarkHooks(unittest.TestCase):
    def test_verify_and_memory(self):
        model = EngramaModel(_small_config(40))
        res = BenchmarkSuite.verify_causal_invariance(model, seq_length=8)
        self.assertTrue(res["passed"])
        mem = BenchmarkSuite.benchmark_memory(model, seq_length=8)
        self.assertIn("parameter_bytes", mem)
        self.assertIn("cache_bytes_at_seq", mem)


if __name__ == "__main__":
    unittest.main()
