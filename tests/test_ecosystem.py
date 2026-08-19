import os
import shutil
import tempfile
import unittest
import torch

from engrama.config import EngramaConfig
from engrama.model import EngramaModel
from engrama.tokenizer import EngramaTokenizer
from engrama.datasets import TextDataset
from engrama.trainer import Trainer
from engrama.inference import Generator
from engrama.serialization import save_model, load_model
from engrama.inspection import EngramaInspector
from engrama.benchmarks import BenchmarkSuite


class TestEcosystem(unittest.TestCase):
    def setUp(self):
        self.text = "Hello ENGRAMA architecture world! This is a test dataset for training."
        self.tokenizer = EngramaTokenizer()
        self.tokenizer.fit_on_text(self.text)
        self.config = EngramaConfig(
            vocab_size=self.tokenizer.vocab_size,
            d_model=32,
            d_gate=8,
            num_cells=2,
            num_encoder_layers=1,
            num_consolidation_layers=2,
            context_length=16,
            offsets=[0, 1, 2],
            num_candidates=2,
        )
        self.model = EngramaModel(self.config)

    def test_datasets(self):
        dataset = TextDataset(self.text, self.tokenizer, sequence_length=8)
        self.assertGreater(len(dataset), 0)
        item = dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("target_ids", item)
        self.assertEqual(item["input_ids"].shape[0], 8)
        self.assertEqual(item["target_ids"].shape[0], 8)

    def test_trainer(self):
        dataset = TextDataset(self.text, self.tokenizer, sequence_length=8)
        trainer = Trainer(self.model, lr=1e-3)
        history = trainer.fit(dataset, batch_size=2, epochs=2)
        self.assertEqual(len(history), 2)
        val_loss = trainer.evaluate(dataset, batch_size=2)
        self.assertIsInstance(val_loss, float)

    def test_inference(self):
        generator = Generator(self.model, self.tokenizer)
        gen_text = generator.generate("Hello", max_new_tokens=5, use_cache=True)
        self.assertIsInstance(gen_text, str)

    def test_serialization(self):
        temp_dir = tempfile.mkdtemp()
        try:
            save_model(self.model, temp_dir, self.tokenizer)
            loaded_model, loaded_tok = load_model(temp_dir)
            self.assertIsNotNone(loaded_model)
            self.assertIsNotNone(loaded_tok)
            self.assertEqual(loaded_tok.vocab_size, self.tokenizer.vocab_size)
        finally:
            shutil.rmtree(temp_dir)

    def test_inspection(self):
        input_ids = torch.randint(0, self.config.vocab_size, (1, 8))
        activations = EngramaInspector.inspect_activations(self.model, input_ids)
        self.assertIn("T0", activations)
        self.assertIn("T1", activations)

        gates = EngramaInspector.inspect_gates(self.model, input_ids)
        self.assertIn("layer_0", gates)

    def test_benchmarks(self):
        lat = BenchmarkSuite.benchmark_latency(self.model, seq_length=8, num_runs=2)
        self.assertIn("parallel_tokens_per_sec", lat)
        self.assertIn("step_tokens_per_sec", lat)

        mem = BenchmarkSuite.benchmark_memory(self.model, seq_length=8)
        self.assertIn("num_parameters", mem)

        causal = BenchmarkSuite.verify_causal_invariance(self.model, seq_length=8)
        self.assertTrue(causal["passed"])


if __name__ == "__main__":
    unittest.main()
