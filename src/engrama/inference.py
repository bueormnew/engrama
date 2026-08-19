"""
ENGRAMA Text Generation Engine Module
Author: BUEORM
License: AGPL-3.0
"""

from typing import Generator as PyGenerator, Optional

from engrama.model import EngramaModel
from engrama.tokenizer import EngramaTokenizer


class Generator:
    """High-level text generation interface wrapper for ENGRAMA models.

    Args:
        model (EngramaModel): Loaded ENGRAMA model instance.
        tokenizer (EngramaTokenizer): Tokenizer instance.
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
    ) -> str:
        """Generate text completion for a given prompt string."""
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        output_ids = self.model.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_cache=use_cache,
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
    ) -> PyGenerator[str, None, None]:
        """Stream generated characters/tokens completion one by one."""
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        for token_id in self.model.generate_stream(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            char_str = self.tokenizer.decode([token_id], skip_special_tokens=True)
            yield char_str
