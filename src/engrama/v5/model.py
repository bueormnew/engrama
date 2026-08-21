"""ENGRAMA V5 model — isolated encoder → synaptic-resonance stack → evoker.

Public, PyTorch-native API::

    model = EngramaV5(EngramaV5Config(...))
    logits = model(input_ids)                       # (B, N, vocab)
    loss   = model.loss(input_ids, targets)         # scalar
    ids    = model.generate([1, 2, 3], max_new_tokens=50)

Author: ENGRAMA contributors.
License: AGPL-3.0
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from engrama.v5.cache import ResonanceCache
from engrama.v5.config import EngramaV5Config
from engrama.v5.evoker import LatentFusionEvoker
from engrama.v5.primitives import Cell, IsolatedEncoder
from engrama.v5.resonance import SynapticResonance

DEFAULT_BOS_TOKEN_ID = 2


class ResonanceBlock(nn.Module):
    """One V5 block: synaptic resonance (synapse) + Cell (transformation)."""

    def __init__(self, cfg: EngramaV5Config):
        super().__init__()
        self.resonance = SynapticResonance(
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            read_norm=cfg.read_norm,
            tau_init=cfg.tau_init,
            norm_type=cfg.norm_type,
        )
        self.cell = Cell(cfg.d_model, cfg.d_ff, cfg.dropout, cfg.activation, cfg.norm_type)
        self.chunk_size = cfg.chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chunk_size and x.size(1) > self.chunk_size:
            x = self.resonance.forward_chunked(x, self.chunk_size)
        else:
            x = self.resonance(x)
        return self.cell(x)

    def step(self, x_t: torch.Tensor, layer_cache) -> torch.Tensor:
        x_t = self.resonance.step(x_t, layer_cache)
        return self.cell(x_t)


class EngramaV5(nn.Module):
    """ENGRAMA V5 — non-attention, non-compressive, content-addressable recall."""

    def __init__(self, cfg: EngramaV5Config):
        super().__init__()
        self.cfg = cfg
        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.encoder = IsolatedEncoder(
            cfg.d_model, cfg.d_ff, cfg.num_encoder_layers,
            cfg.dropout, cfg.activation, cfg.norm_type,
        )
        self.blocks = nn.ModuleList([ResonanceBlock(cfg) for _ in range(cfg.num_layers)])
        self.evoker = LatentFusionEvoker(cfg.d_model, cfg.num_candidates)

        if cfg.tie_embeddings:
            self.output_projection: Optional[nn.Linear] = None
        else:
            self.output_projection = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self._init_weights()
        dtype = cfg.torch_dtype()
        if dtype != torch.float32:
            self.to(dtype=dtype)

    def _init_weights(self) -> None:
        """Initialize embeddings at unit scale and projections at a modest std.

        The synaptic-resonance read relies on cosine similarity of q/k, so the
        *content* signal carried by embeddings must be strong at init. Shrinking
        embeddings to std=0.02 (GPT-2 style) collapses recall because both the
        input signal and the tied output logits become tiny. We initialize
        embeddings at unit variance and leave the resonance/cell projections at
        PyTorch defaults (Kaiming-uniform), which trains cleanly and NaN-free.
        """
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=1.0)
        if self.output_projection is not None:
            nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)

    @property
    def output_embeddings(self) -> torch.Tensor:
        if self.output_projection is not None:
            return self.output_projection.weight
        return self.embeddings.weight

    # ------------------------------------------------------------------
    def features(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_ids)
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.evoker(self.features(input_ids), self.output_embeddings)

    def loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Language-model cross-entropy loss."""
        logits = self.forward(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size),
            targets.reshape(-1),
            ignore_index=ignore_index,
        )

    # ------------------------------------------------------------------
    # Incremental generation with the native explicit-trace cache
    # ------------------------------------------------------------------
    def new_cache(self, n_max: Optional[int] = None) -> ResonanceCache:
        return ResonanceCache(self.cfg.num_layers, n_max or self.cfg.context_length)

    def _step_logits(self, token_id: torch.Tensor, cache: ResonanceCache) -> torch.Tensor:
        """Return logits (B, vocab) for one token, updating the cache."""
        if token_id.dim() == 0:
            token_id = token_id.view(1)
        x_t = self.embeddings(token_id)          # (B, d)
        x_t = self.encoder(x_t)
        for i, block in enumerate(self.blocks):
            x_t = block.step(x_t, cache.layer(i))
        return self.evoker(x_t, self.output_embeddings)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[int]:
        if not prompt_ids:
            prompt_ids = [DEFAULT_BOS_TOKEN_ID]
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            generated = list(prompt_ids)
            if use_cache:
                cache = self.new_cache()
                logits = None
                for tok in prompt_ids:
                    logits = self._step_logits(
                        torch.tensor([tok], device=device), cache
                    )
                for _ in range(max_new_tokens):
                    nxt = self._sample(logits, temperature, top_k, top_p)
                    generated.append(nxt)
                    if eos_token_id is not None and nxt == eos_token_id:
                        break
                    logits = self._step_logits(
                        torch.tensor([nxt], device=device), cache
                    )
            else:
                for _ in range(max_new_tokens):
                    window = generated[-self.cfg.context_length:]
                    ids = torch.tensor([window], device=device)
                    logits = self.forward(ids)[:, -1, :]
                    nxt = self._sample(logits, temperature, top_k, top_p)
                    generated.append(nxt)
                    if eos_token_id is not None and nxt == eos_token_id:
                        break
            return generated
        finally:
            if was_training:
                self.train()

    @staticmethod
    def _sample(logits, temperature, top_k, top_p) -> int:
        logits = logits[0].float()
        if temperature <= 0:
            return int(torch.argmax(logits).item())
        logits = logits / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.numel()))
            logits = torch.where(logits < v[-1], torch.full_like(logits, -1e30), logits)
        if top_p and 0.0 < top_p < 1.0:
            s, idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(s, dim=-1), dim=-1)
            rm = cum > top_p
            rm[1:] = rm[:-1].clone()
            rm[0] = False
            logits[idx[rm]] = -1e30
        probs = F.softmax(logits, dim=-1)
        if not torch.isfinite(probs).all() or probs.sum() <= 0:
            return int(torch.argmax(torch.nan_to_num(logits)).item())
        return int(torch.multinomial(probs, 1).item())

    # ------------------------------------------------------------------
    def num_parameters(self, only_trainable: bool = False) -> int:
        ps = self.parameters()
        return sum(p.numel() for p in ps if (p.requires_grad or not only_trainable))

    def describe(self) -> str:
        return self.cfg.describe() + f"\n  parameters={self.num_parameters():,}"
