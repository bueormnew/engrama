"""
ENGRAMA PyTorch Dataset Utilities Module
Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""

import os
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from engrama.tokenizer import EngramaTokenizer

# Target id used for padding positions: F.cross_entropy ignores it by
# default (``ignore_index=-100``), so short/corpus-tail padding never
# contaminates the training or evaluation loss (HF convention).
IGNORE_INDEX = -100


class TextDataset(Dataset):
    """PyTorch Dataset for text files or raw text strings.

    Creates fixed sequence length chunks with input_ids and shifted target_ids for
    autoregressive cross-entropy language modeling.

    Padding behavior: when the corpus is shorter than one chunk, or the last
    chunk is incomplete, the remainder is padded with ``<pad>``. Every padded
    target position is set to ``IGNORE_INDEX`` (``-100``), which
    ``F.cross_entropy(..., ignore_index=-100)`` skips -- padding never
    contributes to the loss.

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
        self.pad_masks: List[torch.Tensor] = []

        def add_chunk(chunk: List[int], num_pad: int) -> None:
            """Register one chunk plus its trailing padding mask."""
            self.chunks.append(torch.tensor(chunk, dtype=torch.long))
            mask = torch.zeros(chunk_len, dtype=torch.bool)
            if num_pad > 0:
                mask[-num_pad:] = True
            self.pad_masks.append(mask)

        if len(tokens) < chunk_len:
            pad_id = self.tokenizer.SPECIAL_TOKENS.get("<pad>", 0)
            num_pad = chunk_len - len(tokens)
            add_chunk(tokens + [pad_id] * num_pad, num_pad)
        else:
            for i in range(0, len(tokens) - chunk_len + 1, self.stride):
                chunk = tokens[i : i + chunk_len]
                add_chunk(chunk, 0)
            if not self.chunks:
                chunk = tokens[:chunk_len]
                if len(chunk) < chunk_len:
                    pad_id = self.tokenizer.SPECIAL_TOKENS.get("<pad>", 0)
                    num_pad = chunk_len - len(chunk)
                    add_chunk(chunk + [pad_id] * num_pad, num_pad)
                else:
                    add_chunk(chunk, 0)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        mask = self.pad_masks[idx]
        target_ids = chunk[1:].clone()
        # Padded targets are never counted in the loss (F.cross_entropy
        # ignores IGNORE_INDEX by default).
        target_ids[mask[1:]] = IGNORE_INDEX
        return {
            "input_ids": chunk[:-1],
            "target_ids": target_ids,
        }
