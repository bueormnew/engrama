import unittest
import tempfile
import os
import torch

from engrama.config import EngramaConfig
from engrama.tokenizer import EngramaTokenizer
from engrama.model import EngramaModel


class TestEngramaTokenizer(unittest.TestCase):
    def test_fit_encode_decode(self):
        tokenizer = EngramaTokenizer()
        tokenizer.fit_on_text("hello world")
        self.assertGreater(tokenizer.vocab_size, 4)

        encoded = tokenizer.encode("hello", add_bos=True, add_eos=True)
        self.assertEqual(encoded[0], 2)
        self.assertEqual(encoded[-1], 3)

        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        self.assertEqual(decoded, "hello")

    def test_save_load(self):
        tokenizer = EngramaTokenizer()
        tokenizer.fit_on_text("abc")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "vocab.json")
            tokenizer.save(filepath)
            loaded = EngramaTokenizer.load(filepath)
            self.assertEqual(tokenizer.vocab_size, loaded.vocab_size)
            self.assertEqual(tokenizer.encode("abc"), loaded.encode("abc"))


class TestEngramaModel(unittest.TestCase):
    def setUp(self):
        self.config = EngramaConfig(
            vocab_size=32,
            d_model=32,
            d_gate=8,
            d_ff=64,
            num_cells=4,
            num_encoder_layers=2,
            num_consolidation_layers=2,
            context_length=64,
            offsets=[0, 1, 2],
            num_candidates=2,
        )
        self.model = EngramaModel(self.config)
        self.model.eval()

    def test_forward_shape(self):
        x = torch.randint(0, self.config.vocab_size, (2, 8))
        logits = self.model(x)
        self.assertEqual(logits.shape, (2, 8, self.config.vocab_size))

    def test_step_forward(self):
        cache = self.model.get_cache()
        token_id = torch.tensor([[5]])
        logits_t, t_l_t = self.model.step_forward(token_id, cache, timestamp=0)
        self.assertEqual(logits_t.shape, (1, self.config.vocab_size))
        self.assertEqual(t_l_t.shape, (1, self.config.d_model))
        self.assertEqual(len(cache), 1)

    def test_causal_equivalence(self):
        seq_len = 5
        seq = torch.randint(0, self.config.vocab_size, (1, seq_len))
        with torch.no_grad():
            full_logits = self.model(seq)

            cache = self.model.get_cache()
            step_logits_list = []
            for t in range(seq_len):
                token_id = seq[:, t : t + 1]
                logits_t, _ = self.model.step_forward(token_id, cache, timestamp=t)
                step_logits_list.append(logits_t.unsqueeze(1))
            step_logits = torch.cat(step_logits_list, dim=1)

        diff = (full_logits - step_logits).abs().max().item()
        self.assertLess(diff, 1e-4)

    def test_generation(self):
        prompt = [2, 4, 5]
        gen_cached = self.model.generate(prompt, max_new_tokens=4, use_cache=True)
        gen_nocache = self.model.generate(prompt, max_new_tokens=4, use_cache=False)
        self.assertEqual(len(gen_cached), len(prompt) + 4)
        self.assertEqual(len(gen_nocache), len(prompt) + 4)

    def test_inspect_trace(self):
        cache = self.model.get_cache()
        token_id = torch.tensor([[1]])
        self.model.step_forward(token_id, cache, timestamp=0)
        info = self.model.inspect_trace(cache)
        self.assertEqual(info["cache_length"], 1)
        self.assertEqual(info["num_layers"], self.config.num_consolidation_layers)


if __name__ == "__main__":
    unittest.main()
