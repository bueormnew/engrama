"""Tokenizer and TextDataset tests."""

import os
import tempfile
import unittest

import torch

from engrama.datasets import TextDataset
from engrama.tokenizer import EngramaTokenizer


class TestEngramaTokenizer(unittest.TestCase):
    def test_special_tokens(self):
        tok = EngramaTokenizer()
        self.assertEqual(tok.char_to_id["<pad>"], 0)
        self.assertEqual(tok.char_to_id["<unk>"], 1)
        self.assertEqual(tok.char_to_id["<bos>"], 2)
        self.assertEqual(tok.char_to_id["<eos>"], 3)
        self.assertEqual(tok.vocab_size, 4)

    def test_fit_encode_decode_roundtrip(self):
        tok = EngramaTokenizer().fit_on_text("hello world")
        self.assertGreater(tok.vocab_size, 4)
        ids = tok.encode("hello", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], 2)
        self.assertEqual(ids[-1], 3)
        self.assertEqual(tok.decode(ids), "hello")

    def test_unknown_char_maps_to_unk(self):
        tok = EngramaTokenizer().fit_on_text("abc")
        ids = tok.encode("aZb", add_bos=False)
        self.assertEqual(ids[1], 1)

    def test_batch_helpers(self):
        tok = EngramaTokenizer().fit_on_text("ab")
        seqs = tok.encode_batch(["ab", "ba"])
        self.assertEqual(len(seqs), 2)
        self.assertEqual(tok.decode_batch(seqs), ["ab", "ba"])

    def test_save_load(self):
        tok = EngramaTokenizer().fit_on_text("áéîöü 123")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vocab.json")
            tok.save(path)
            loaded = EngramaTokenizer.load(path)
        self.assertEqual(tok.vocab_size, loaded.vocab_size)
        self.assertEqual(tok.encode("áé 1"), loaded.encode("áé 1"))

    def test_vocab_file_constructor(self):
        tok = EngramaTokenizer().fit_on_text("xyz")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vocab.json")
            tok.save(path)
            loaded = EngramaTokenizer(vocab_file=path)
        self.assertEqual(tok.char_to_id, loaded.char_to_id)


class TestTextDataset(unittest.TestCase):
    def setUp(self):
        self.text = "abcdefghij" * 20
        self.tok = EngramaTokenizer().fit_on_text(self.text)

    def test_chunking_and_shift(self):
        ds = TextDataset(self.text, self.tok, sequence_length=16)
        self.assertGreater(len(ds), 0)
        item = ds[0]
        self.assertEqual(item["input_ids"].shape, (16,))
        self.assertEqual(item["target_ids"].shape, (16,))
        self.assertTrue(torch.equal(item["target_ids"][:-1], item["input_ids"][1:]))

    def test_from_file_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "corpus.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.text)
            ds = TextDataset(path, self.tok, sequence_length=8)
        self.assertGreater(len(ds), 0)

    def test_short_text_pads(self):
        ds = TextDataset("ab", self.tok, sequence_length=32)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]["input_ids"].shape, (32,))

    def test_stride(self):
        ds_a = TextDataset(self.text, self.tok, sequence_length=16, stride=16)
        ds_b = TextDataset(self.text, self.tok, sequence_length=16, stride=4)
        self.assertGreater(len(ds_b), len(ds_a))


if __name__ == "__main__":
    unittest.main()
