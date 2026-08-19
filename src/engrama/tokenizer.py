"""
ENGRAMA Character-Level Tokenizer Module
Author: BUEORM
License: AGPL-3.0
"""

import json
from typing import Dict, List, Optional, Union


class EngramaTokenizer:
    """Character-level tokenizer for ENGRAMA models.

    Supports special control tokens (<pad>, <unk>, <bos>, <eos>) and dynamic vocabulary fitting.
    """

    SPECIAL_TOKENS: Dict[str, int] = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
    }

    def __init__(self, vocab_file: Optional[str] = None):
        self.char_to_id: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.id_to_char: Dict[int, str] = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        if vocab_file is not None:
            self._load_from_file(vocab_file)

    def _load_from_file(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char_to_id = {k: int(v) for k, v in data.items()}
        self.id_to_char = {int(v): k for k, v in data.items()}

    def fit_on_text(self, text: Union[str, List[str]]) -> "EngramaTokenizer":
        """Fit tokenizer vocabulary on input text or list of texts."""
        texts = [text] if isinstance(text, str) else text
        for t in texts:
            for char in t:
                if char not in self.char_to_id:
                    idx = len(self.char_to_id)
                    self.char_to_id[char] = idx
                    self.id_to_char[idx] = char
        return self

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = False
    ) -> List[int]:
        """Encode text string into token ID sequence."""
        res: List[int] = []
        if add_bos:
            res.append(self.SPECIAL_TOKENS["<bos>"])
        for char in text:
            res.append(self.char_to_id.get(char, self.SPECIAL_TOKENS["<unk>"]))
        if add_eos:
            res.append(self.SPECIAL_TOKENS["<eos>"])
        return res

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ID sequence back into text string."""
        special_ids = set(self.SPECIAL_TOKENS.values())
        chars: List[str] = []
        for i in ids:
            if skip_special_tokens and i in special_ids:
                continue
            chars.append(self.id_to_char.get(i, "<unk>"))
        return "".join(chars)

    def encode_batch(
        self, texts: List[str], add_bos: bool = True, add_eos: bool = False
    ) -> List[List[int]]:
        """Encode batch of text strings into list of token ID sequences."""
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def decode_batch(
        self, sequences: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode batch of token ID sequences into list of text strings."""
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size."""
        return len(self.char_to_id)

    def save(self, filepath: str) -> None:
        """Save vocabulary mapping to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.char_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "EngramaTokenizer":
        """Load tokenizer instance from JSON vocabulary file."""
        tokenizer = cls()
        tokenizer._load_from_file(filepath)
        return tokenizer
