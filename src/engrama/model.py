"""ENGRAMA Neural Model Core Integration Module (V3).

Integrates the four architecture phases:

1. Isolated token encoding (:class:`~engrama.encoder.IsolatedEncoder`).
2. Circular trace + hierarchical minimum-horizon cache
   (:class:`~engrama.trace.EngramaCache`).
3. Hierarchical dilated consolidation
   (:class:`~engrama.consolidation.ConsolidationStack`).
4. Multi-candidate recall (:class:`~engrama.evoker.MultiCandidateEvoker`).

The parallel ``forward`` (training) and the incremental ``step_forward``
(inference) are exactly equivalent by causal invariance (V3 spec section
23), regardless of cache mode (V3 spec section 24).

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from engrama.config import EngramaConfig
from engrama.consolidation import ConsolidationStack
from engrama.encoder import IsolatedEncoder
from engrama.evoker import MultiCandidateEvoker
from engrama.trace import EngramaCache

# Default BOS token id used when ``generate``/``generate_stream`` receive an
# empty prompt. Matches ``EngramaTokenizer.SPECIAL_TOKENS["<bos>"]`` (id 2).
DEFAULT_BOS_TOKEN_ID = 2


def _validate_prompt_ids(prompt_ids: List[int], vocab_size: int) -> None:
    """Raise ``ValueError`` on empty, negative or out-of-range prompt ids."""
    for t in prompt_ids:
        if t < 0 or t >= vocab_size:
            raise ValueError(
                f"prompt_ids contains token id {t} outside [0, {vocab_size}) "
                f"(vocab_size={vocab_size}); re-encode the prompt with the "
                f"matching tokenizer"
            )


class EngramaModel(nn.Module):
    """ENGRAMA architecture integrator model.

    Args:
        config: Architecture configuration (see :class:`EngramaConfig`).
    """

    def __init__(self, config: EngramaConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = IsolatedEncoder(config)
        self.consolidation = ConsolidationStack(config)
        self.evoker = MultiCandidateEvoker(config)

        if config.tie_embeddings:
            self.output_projection: Optional[nn.Linear] = None
        else:
            self.output_projection = nn.Linear(
                config.d_model, config.vocab_size, bias=False
            )

        self._cache: Optional[EngramaCache] = None

        # Apply the configured numeric dtype to the whole model (V3 §38).
        dtype = config.torch_dtype()
        if dtype != torch.float32:
            self.to(dtype=dtype)

    # ------------------------------------------------------------------
    # Output embedding used by the evoker (tied or separate head)
    # ------------------------------------------------------------------
    @property
    def output_embeddings(self) -> torch.Tensor:
        if self.output_projection is not None:
            return self.output_projection.weight
        return self.embeddings.weight

    # ------------------------------------------------------------------
    # Parallel path (training / full-context inference)
    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Parallel autoregressive forward pass.

        Args:
            input_ids: Token tensor of shape ``(B, N)``.

        Returns:
            Logits tensor of shape ``(B, N, vocab_size)``.
        """
        x = self.embeddings(input_ids)
        T0 = self.encoder(x)
        T_L = self.consolidation.forward_train(T0, T0_pristine=T0)
        return self.evoker(T_L, self.output_embeddings)

    # ------------------------------------------------------------------
    # Incremental path (cached generation, V3 spec section 13)
    # ------------------------------------------------------------------
    def step_forward(
        self, token_id: torch.Tensor, cache: EngramaCache, timestamp: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Incremental forward pass for a single token using the trace cache.

        Args:
            token_id: Token tensor of shape ``(B, 1)`` (or ``(B,)``/scalar).
            cache: Active :class:`EngramaCache` instance.
            timestamp: Absolute timestamp ``t`` recorded in the trace.

        Returns:
            ``(logits_t, hidden_t)`` with shapes ``(B, vocab_size)`` and
            ``(B, d_model)``.
        """
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        elif token_id.dim() == 0:
            token_id = token_id.unsqueeze(0).unsqueeze(0)

        x_t = self.embeddings(token_id)
        t0_t = self.encoder(x_t)
        if t0_t.dim() == 3:
            t0_t = t0_t.squeeze(1)

        t_l_t, _ = self.consolidation.step_forward(
            cache, T0_current=t0_t, timestamp=timestamp, return_all_layers=True
        )
        logits_t = self.evoker(t_l_t, self.output_embeddings)
        return logits_t, t_l_t

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        generated_history: Optional[List[int]] = None,
    ) -> int:
        """Sample one token with temperature / top-k / top-p / penalties."""
        if logits.dim() == 2:
            logits = logits[0]
        logits = logits.float().clone()

        if repetition_penalty != 1.0 and generated_history:
            for token in set(generated_history):
                if logits[token] < 0:
                    logits[token] *= repetition_penalty
                else:
                    logits[token] /= repetition_penalty

        if temperature <= 0.0:
            return int(torch.argmax(logits, dim=-1).item())

        logits = logits / temperature

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(
                logits < v[-1],
                torch.tensor(-float("inf"), device=logits.device),
                logits,
            )

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        use_cache: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> List[int]:
        """Autoregressively generate tokens from prompt ids.

        Args:
            prompt_ids: Prompt token ids (empty list starts from the default
                BOS token, ``DEFAULT_BOS_TOKEN_ID``).
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (``<= 0`` => greedy argmax).
            top_k: Keep only the top-k most likely tokens (``None`` disables).
            top_p: Nucleus sampling threshold (``None`` disables).
            repetition_penalty: Penalty > 1.0 discourages repeated tokens.
            use_cache: Incremental generation with the trace cache (default).
                When ``False``, re-runs the parallel forward over the sliding
                window of the last ``context_length`` tokens.
            eos_token_id: Stop generation after emitting this token id.

        Returns:
            Prompt ids followed by the generated ids.
        """
        if not prompt_ids:
            prompt_ids = [DEFAULT_BOS_TOKEN_ID]
        _validate_prompt_ids(prompt_ids, self.config.vocab_size)
        generated = list(prompt_ids)
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        try:
            if use_cache:
                cache = self.get_cache()
                logits_t: Optional[torch.Tensor] = None
                with torch.no_grad():
                    for t, tok in enumerate(prompt_ids):
                        token_tensor = torch.tensor([[tok]], dtype=torch.long, device=device)
                        logits_t, _ = self.step_forward(token_tensor, cache, timestamp=t)

                    for i in range(max_new_tokens):
                        if logits_t is None:
                            break
                        next_token = self._sample_next_token(
                            logits_t, temperature, top_k, top_p,
                            repetition_penalty, generated,
                        )
                        generated.append(next_token)
                        if eos_token_id is not None and next_token == eos_token_id:
                            break
                        if i < max_new_tokens - 1:
                            token_tensor = torch.tensor(
                                [[next_token]], dtype=torch.long, device=device
                            )
                            logits_t, _ = self.step_forward(
                                token_tensor,
                                cache,
                                timestamp=len(prompt_ids) + i,
                            )
            else:
                with torch.no_grad():
                    for _ in range(max_new_tokens):
                        window = generated[-self.config.context_length:]
                        input_tensor = torch.tensor([window], dtype=torch.long, device=device)
                        logits = self.forward(input_tensor)
                        next_token = self._sample_next_token(
                            logits[:, -1, :], temperature, top_k, top_p,
                            repetition_penalty, generated,
                        )
                        generated.append(next_token)
                        if eos_token_id is not None and next_token == eos_token_id:
                            break
        finally:
            if was_training:
                self.train()

        return generated

    def generate_stream(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> Generator[int, None, None]:
        """Stream generated token ids one by one.

        Args:
            prompt_ids: Prompt token ids (empty list starts from the default
                BOS token, ``DEFAULT_BOS_TOKEN_ID``).
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (``<= 0`` => greedy argmax).
            top_k: Keep only the top-k most likely tokens (``None`` disables).
            top_p: Nucleus sampling threshold (``None`` disables).
            repetition_penalty: Penalty > 1.0 discourages repeated tokens.
            eos_token_id: Stop generation after emitting this token id.
            use_cache: Incremental generation with the trace cache (default).
                When ``False``, re-runs the parallel forward over the sliding
                window of the last ``context_length`` tokens.

        Yields:
            One generated token id at a time (prompt excluded).
        """
        if not prompt_ids:
            prompt_ids = [DEFAULT_BOS_TOKEN_ID]
        _validate_prompt_ids(prompt_ids, self.config.vocab_size)
        generated = list(prompt_ids)
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                if use_cache:
                    cache = self.get_cache()
                    logits_t: Optional[torch.Tensor] = None
                    for t, tok in enumerate(prompt_ids):
                        token_tensor = torch.tensor(
                            [[tok]], dtype=torch.long, device=device
                        )
                        logits_t, _ = self.step_forward(
                            token_tensor, cache, timestamp=t
                        )

                    for i in range(max_new_tokens):
                        if logits_t is None:
                            break
                        next_token = self._sample_next_token(
                            logits_t, temperature, top_k, top_p,
                            repetition_penalty, generated,
                        )
                        generated.append(next_token)
                        yield next_token
                        if eos_token_id is not None and next_token == eos_token_id:
                            return
                        if i < max_new_tokens - 1:
                            token_tensor = torch.tensor(
                                [[next_token]], dtype=torch.long, device=device
                            )
                            logits_t, _ = self.step_forward(
                                token_tensor, cache, timestamp=len(prompt_ids) + i
                            )
                else:
                    for _ in range(max_new_tokens):
                        window = generated[-self.config.context_length:]
                        input_tensor = torch.tensor(
                            [window], dtype=torch.long, device=device
                        )
                        logits = self.forward(input_tensor)
                        next_token = self._sample_next_token(
                            logits[:, -1, :], temperature, top_k, top_p,
                            repetition_penalty, generated,
                        )
                        generated.append(next_token)
                        yield next_token
                        if eos_token_id is not None and next_token == eos_token_id:
                            return
        finally:
            if was_training:
                self.train()

    # ------------------------------------------------------------------
    # Cache management (V3 sections 12, 22, 24)
    # ------------------------------------------------------------------
    def get_cache(
        self,
        N_max: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> EngramaCache:
        """Instantiate a fresh cache honoring the configured cache mode."""
        max_len = N_max if N_max is not None else self.config.context_length
        cache_mode = mode if mode is not None else self.config.cache_mode
        horizons = self.config.cache_horizons() if cache_mode == "hierarchical" else None
        self._cache = EngramaCache(
            N_max=max_len,
            num_layers=self.config.num_consolidation_layers,
            d_model=self.config.d_model,
            mode=cache_mode,
            horizons=horizons,
        )
        return self._cache

    def reset_cache(self) -> None:
        """Drop the internal cache reference."""
        self._cache = None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def num_parameters(self, only_trainable: bool = False) -> int:
        """Return the parameter count of the model."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def inspect_trace(self, cache: EngramaCache) -> Dict[str, Any]:
        """Inspect states and metadata inside the trace cache."""
        return cache.describe()

    def describe(self) -> str:
        """Human-readable model description."""
        return (
            self.config.describe()
            + f"\n  parameters={self.num_parameters():,} "
            f"(trainable {self.num_parameters(only_trainable=True):,})"
        )
