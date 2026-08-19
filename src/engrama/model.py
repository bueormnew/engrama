"""
ENGRAMA Neural Model Core Integration Module
Author: BUEORM
License: AGPL-3.0
"""

from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from engrama.config import EngramaConfig
from engrama.consolidation import ConsolidationStack
from engrama.encoder import IsolatedEncoder
from engrama.evoker import MultiCandidateEvoker
from engrama.trace import EngramaCache


class EngramaModel(nn.Module):
    """ENGRAMA Architecture Integrator Model.

    Combines:
    1. Isolated Token Encoder (Phase 1)
    2. Incremental Logarithmic Circular Cache Trace (Phase 2)
    3. Positional Dilated Gated Consolidation Stack (Phase 3)
    4. Multi-Candidate Memory Evoker (Phase 4)

    Args:
        config (EngramaConfig): Architecture configuration parameters.
    """

    def __init__(self, config: EngramaConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = IsolatedEncoder(config)
        self.consolidation = ConsolidationStack(config)
        self.evoker = MultiCandidateEvoker(config)
        self._cache: Optional[EngramaCache] = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Parallel autoregressive forward pass over sequence batch.

        Args:
            input_ids (Tensor): Input token tensor of shape (B, N).

        Returns:
            Tensor: Vocabulary logits tensor of shape (B, N, vocab_size).
        """
        x = self.embeddings(input_ids)
        T0 = self.encoder(x)
        T_L = self.consolidation.forward_train(T0)
        logits = self.evoker(T_L, self.embeddings.weight)
        return logits

    def step_forward(
        self, token_id: torch.Tensor, cache: EngramaCache, timestamp: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Incremental step forward pass for a single token using trace cache.

        Args:
            token_id (Tensor): Single token ID tensor of shape (B, 1) or (1, 1).
            cache (EngramaCache): Trace cache instance.
            timestamp (int): Current sequence position timestamp t.

        Returns:
            Tuple[Tensor, Tensor]: (logits_t of shape (B, vocab_size), final_hidden_state_t of shape (B, d_model)).
        """
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        elif token_id.dim() == 0:
            token_id = token_id.unsqueeze(0).unsqueeze(0)

        x_t = self.embeddings(token_id)
        t0_t = self.encoder(x_t)
        if t0_t.dim() == 3:
            t0_t = t0_t.squeeze(1)

        t_l_t, layer_outputs = self.consolidation.step_forward(
            cache, T0_current=t0_t, return_all_layers=True
        )
        logits_t = self.evoker(t_l_t, self.embeddings.weight)
        cache.append(t0_t, layer_outputs, timestamp)
        return logits_t, t_l_t

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        generated_history: Optional[List[int]] = None,
    ) -> int:
        """Sample next token with temperature, top-k, top-p, and repetition penalty."""
        if logits.dim() == 2:
            logits = logits[0]

        logits = logits.clone()

        # Repetition penalty
        if repetition_penalty != 1.0 and generated_history:
            for token in set(generated_history):
                if logits[token] < 0:
                    logits[token] *= repetition_penalty
                else:
                    logits[token] /= repetition_penalty

        if temperature <= 0.0:
            return int(torch.argmax(logits, dim=-1).item())

        logits = logits / temperature

        # Top-K filtering
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(
                logits < v[-1],
                torch.tensor(-float("Inf"), device=logits.device),
                logits,
            )

        # Top-P (Nucleus) filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        use_cache: bool = True,
    ) -> List[int]:
        """Autoregressively generate token sequence from prompt IDs."""
        if not prompt_ids:
            prompt_ids = [2]
        generated = list(prompt_ids)
        device = next(self.parameters()).device

        if use_cache:
            cache = self.get_cache()
            logits_t = None
            for t, tok in enumerate(prompt_ids):
                token_tensor = torch.tensor([[tok]], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits_t, _ = self.step_forward(token_tensor, cache, timestamp=t)

            for i in range(max_new_tokens):
                if logits_t is None:
                    break
                next_token = self._sample_next_token(
                    logits_t,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generated_history=generated,
                )
                generated.append(next_token)
                if i < max_new_tokens - 1:
                    token_tensor = torch.tensor(
                        [[next_token]], dtype=torch.long, device=device
                    )
                    with torch.no_grad():
                        logits_t, _ = self.step_forward(
                            token_tensor, cache, timestamp=len(prompt_ids) + i
                        )
        else:
            for _ in range(max_new_tokens):
                input_tensor = torch.tensor(
                    [generated], dtype=torch.long, device=device
                )
                with torch.no_grad():
                    logits = self.forward(input_tensor)
                logits_t = logits[:, -1, :]
                next_token = self._sample_next_token(
                    logits_t,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generated_history=generated,
                )
                generated.append(next_token)

        return generated

    def generate_stream(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
    ) -> Generator[int, None, None]:
        """Stream generated token IDs one by one using trace cache."""
        if not prompt_ids:
            prompt_ids = [2]
        generated = list(prompt_ids)
        device = next(self.parameters()).device

        cache = self.get_cache()
        logits_t = None
        for t, tok in enumerate(prompt_ids):
            token_tensor = torch.tensor([[tok]], dtype=torch.long, device=device)
            with torch.no_grad():
                logits_t, _ = self.step_forward(token_tensor, cache, timestamp=t)

        for i in range(max_new_tokens):
            if logits_t is None:
                break
            next_token = self._sample_next_token(
                logits_t,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_history=generated,
            )
            generated.append(next_token)
            yield next_token

            if i < max_new_tokens - 1:
                token_tensor = torch.tensor(
                    [[next_token]], dtype=torch.long, device=device
                )
                with torch.no_grad():
                    logits_t, _ = self.step_forward(
                        token_tensor, cache, timestamp=len(prompt_ids) + i
                    )

    def get_cache(self, N_max: Optional[int] = None) -> EngramaCache:
        """Instantiate and return a new EngramaCache trace buffer."""
        max_len = N_max if N_max is not None else self.config.context_length
        self._cache = EngramaCache(
            N_max=max_len,
            num_layers=self.config.num_consolidation_layers,
            d_model=self.config.d_model,
        )
        return self._cache

    def reset_cache(self) -> None:
        """Reset internal model cache reference."""
        self._cache = None

    def num_parameters(self, only_trainable: bool = False) -> int:
        """Return parameter count of the model."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def inspect_trace(self, cache: EngramaCache) -> Dict[str, Any]:
        """Inspect states and metadata inside trace cache."""
        return {
            "cache_length": len(cache),
            "timestamps": list(cache.timestamps),
            "T0_shapes": [list(t.shape) for t in cache.T0] if cache.T0 else [],
            "Tl_shapes": [
                [list(t.shape) for t in layer] for layer in cache.Tl
            ] if cache.Tl else [],
            "num_layers": cache.num_layers,
            "d_model": cache.d_model,
            "N_max": cache.N_max,
        }