"""ENGRAMA Text Generation Engine Module.

High-level text generation wrapper binding a model with its tokenizer.

Author: BUEORM
License: AGPL-3.0
"""

from __future__ import annotations

from typing import Generator as PyGenerator, Optional

from engrama.model import EngramaModel
from engrama.tokenizer import EngramaTokenizer


class Generator:
    """High-level text generation interface for ENGRAMA models.

    Args:
        model: ENGRAMA model instance.
        tokenizer: Matching tokenizer instance.
    """

    def __init__(self, model: EngramaModel, tokenizer: EngramaTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        use_cache: bool = True,
        stop_at_eos: bool = False,
    ) -> str:
        """Generate a text completion for ``prompt`` (prompt included)."""
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        output_ids = self.model.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_cache=use_cache,
            eos_token_id=self.tokenizer.SPECIAL_TOKENS["<eos>"] if stop_at_eos else None,
        )
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        stop_at_eos: bool = False,
    ) -> PyGenerator[str, None, None]:
        """Stream the completion token by token (character-level models: per char)."""
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        for token_id in self.model.generate_stream(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.SPECIAL_TOKENS["<eos>"] if stop_at_eos else None,
        ):
            yield self.tokenizer.decode([token_id], skip_special_tokens=True)
