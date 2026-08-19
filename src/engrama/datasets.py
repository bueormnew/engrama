"""
ENGRAMA PyTorch Dataset Utilities Module
Author: BUEORM
License: AGPL-3.0
"""

import os
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from engrama.tokenizer import EngramaTokenizer


class TextDataset(Dataset):
    """PyTorch Dataset for text files or raw text strings.

    Creates fixed sequence length chunks with input_ids and shifted target_ids for
    autoregressive cross-entropy language modeling.

    Args:
        text (Union[str, os.PathLike]): Text string content or filepath.
        tokenizer (EngramaTokenizer): EngramaTokenizer instance.
        sequence_length (int): Context sequence length per sample chunk.
        stride (Optional[int]): Stride step size between adjacent chunks.
    """

    def __init__(
        self,
        text: Union[str, os.PathLike],
        tokenizer: EngramaTokenizer,
        sequence_length: int = 128,
        stride: Optional[int] = None,
    ):
        super().__init__()
        text_str = str(text)
        if os.path.isfile(text_str):
            with open(text_str, "r", encoding="utf-8") as f:
                raw_text = f.read()
        else:
            raw_text = text_str

        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.stride = stride if stride is not None else sequence_length

        tokens = self.tokenizer.encode(raw_text, add_bos=True, add_eos=True)
        chunk_len = sequence_length + 1
        self.chunks: List[torch.Tensor] = []

        if len(tokens) < chunk_len:
            pad_id = self.tokenizer.SPECIAL_TOKENS.get("<pad>", 0)
            tokens = tokens + [pad_id] * (chunk_len - len(tokens))
            self.chunks.append(torch.tensor(tokens, dtype=torch.long))
        else:
            for i in range(0, len(tokens) - chunk_len + 1, self.stride):
                chunk = tokens[i : i + chunk_len]
                self.chunks.append(torch.tensor(chunk, dtype=torch.long))
            if not self.chunks:
                chunk = tokens[:chunk_len]
                if len(chunk) < chunk_len:
                    pad_id = self.tokenizer.SPECIAL_TOKENS.get("<pad>", 0)
                    chunk = chunk + [pad_id] * (chunk_len - len(chunk))
                self.chunks.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        return {
            "input_ids": chunk[:-1],
            "target_ids": chunk[1:],
        }
