"""ENGRAMA V5 — modelo integrador.

Flujo (todo paralelo en entrenamiento, incremental en generacion, identicos
por invarianza causal):

    T0 = EncoderAislado(embeddings(x))          (pilar 1: aislado por token)
    K  = P_k(T0); q = P_q(T0 o estado)          (codigos de la Traza)
    lecturas = RecallTap(q, K, T0)              (top-1 duro, sin softmax temporal)
    TL = ConsolidacionV5(T0, T0, lecturas)      (mezcla normalizada + tap + RT)
    logits = Evocador(TL, embeddings)           (fusion latente V4)

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from engrama.config import EngramaConfig
from engrama.encoder import IsolatedEncoder
from engrama.evoker import MultiCandidateEvoker
from engrama.losses import linear_cross_entropy
from engrama.v5.config import V5Config
from engrama.v5.consolidation import V5ConsolidationStack
from engrama.v5.recall import RecallTap
from engrama.v5.trace import V5Trace

DEFAULT_BOS = 2


class EngraModel(nn.Module):
    """ENGRAMA V5 (alias :class:`engrama.v5.EngraModel`)."""

    def __init__(self, config: V5Config):
        super().__init__()
        self.config = config
        # Submodulos reutilizados (ideas originales intactas) via config interna.
        self._inner = EngramaConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            d_gate=config.d_gate,
            d_ff=config.d_ff,
            num_cells=config.num_cells,
            num_encoder_layers=config.num_encoder_layers,
            num_consolidation_layers=config.num_consolidation_layers,
            context_length=256,  # irrelevante para encoder/evocador
            synapse_rank=config.synapse_rank,
            num_candidates=config.num_candidates,
            candidate_aggregation="latent_fusion",
            activation=config.activation,
            dropout=config.dropout,
            tie_embeddings=config.tie_embeddings,
            version="v4",
            stable_init=True,
        )
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = IsolatedEncoder(self._inner)
        self.consolidation = V5ConsolidationStack(config)
        self.evoker = MultiCandidateEvoker(self._inner)
        self.recall = (
            RecallTap(
                config.d_model,
                config.d_recall,
                value=config.rt_value,
                gap=config.rt_gap,
                temperature=config.rt_temperature,
                score_chunk=config.rt_score_chunk,
                init_std=config.rt_init_std,
                shared_init=config.rt_shared_init,
            )
            if config.recall_enabled
            else None
        )
        if not config.tie_embeddings:
            self.output_projection: Optional[nn.Linear] = nn.Linear(
                config.d_model, config.vocab_size, bias=False
            )
        else:
            self.output_projection = None
        self._cache: Optional[V5Trace] = None
        dtype = config.torch_dtype()
        if dtype != torch.float32:
            self.to(dtype=dtype)

    # ------------------------------------------------------------------
    @property
    def output_embeddings(self) -> torch.Tensor:
        if self.output_projection is not None:
            return self.output_projection.weight
        return self.embeddings.weight

    # ------------------------------------------------------------------
    # Paralelo (entrenamiento)
    # ------------------------------------------------------------------
    def footprints(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embeddings(input_ids))

    def forward_features(
        self, input_ids: torch.Tensor, *, score_rows: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        t0 = self.footprints(input_ids)
        reads = None
        if self.recall is not None:
            # consulta aislada: q depende solo de la huella del token actual.
            q = self.recall.queries(t0)
            k = self.recall.keys(t0)
            if self.config.rt_train_mode == "lsh" and self.training:
                reads = self.recall.forward_parallel_lsh(
                    q, k, t0, input_ids,
                    n_tables=self.config.rt_lsh_tables,
                    n_bits=self.config.rt_lsh_bits,
                    cap=self.config.rt_lsh_cap,
                    n_neg=self.config.rt_lsh_neg,
                )
            else:
                reads = self.recall.forward_parallel(q, k, t0, score_rows=score_rows)
        return self.consolidation.forward_train(t0, reads)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.evoker(self.forward_features(input_ids), self.output_embeddings)

    def forward_loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        *,
        linear_chunk_size: int = 2048,
        checkpoint_chunks: bool = False,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        states = self.forward_features(input_ids)
        latent = self.evoker.fused_latent(states)
        return linear_cross_entropy(
            latent,
            self.output_embeddings,
            targets,
            scale=1.0 / math.sqrt(self.config.d_model),
            chunk_size=linear_chunk_size,
            ignore_index=ignore_index,
            checkpoint_chunks=checkpoint_chunks,
        )

    # ------------------------------------------------------------------
    # Incremental (generacion) — cache nativa (Traza + anillo K + horizontes)
    # ------------------------------------------------------------------
    def get_cache(self, n_max: Optional[int] = None) -> V5Trace:
        n = n_max or self.config.context_length
        self._cache = V5Trace(
            n,
            self.config.d_model,
            self.config.d_recall if self.recall is not None else 1,
            self.config.cache_horizons(),
        )
        return self._cache

    def step_forward(
        self, token_id: torch.Tensor, cache: V5Trace, timestamp: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        elif token_id.dim() == 0:
            token_id = token_id.unsqueeze(0).unsqueeze(1)
        t0 = self.footprints(token_id)
        if t0.dim() == 3:
            t0 = t0.squeeze(1)
        k = self.recall.keys(t0) if self.recall is not None else None
        cache.append_t0(t0, k)  # primero escribir: el valor T0[j+1] puede ser el token actual
        read = None
        if self.recall is not None:
            q = self.recall.queries(t0)
            read = self.recall.read_step(q, cache.k_ring, cache.t0_ring, cache.length)
        t_last, _ = self.consolidation.step_forward(cache, read_now=read)
        logits = self.evoker(t_last, self.output_embeddings)
        return logits, t_last

    # ------------------------------------------------------------------
    # Generacion
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sample(self, logits: torch.Tensor, temperature: float, top_k: Optional[int]) -> int:
        row = logits.float().flatten()
        if temperature <= 0:
            return int(row.argmax().item())
        row = row / max(1e-6, temperature)
        if top_k:
            v, _ = torch.topk(row, min(top_k, row.numel()))
            row = torch.where(row < v[-1], torch.full_like(row, -float("inf")), row)
        probs = torch.softmax(row, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = probs / probs.sum().clamp_min(1e-9)
        return int(torch.multinomial(probs, 1).item())

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        use_cache: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> List[int]:
        if not prompt_ids:
            prompt_ids = [DEFAULT_BOS]
        for t in prompt_ids:
            if not (0 <= t < self.config.vocab_size):
                raise ValueError(f"token {t} fuera del vocabulario")
        self.eval()
        device = next(self.parameters()).device
        cache = self.get_cache()
        logits: Optional[torch.Tensor] = None
        out = list(prompt_ids)
        for t, tok in enumerate(prompt_ids):
            x = torch.tensor([[tok]], dtype=torch.long, device=device)
            logits, _ = self.step_forward(x, cache, timestamp=t)
        for i in range(max_new_tokens):
            if logits is None:
                break
            nxt = self._sample(logits, temperature, top_k)
            out.append(nxt)
            if eos_token_id is not None and nxt == eos_token_id:
                break
            if i < max_new_tokens - 1:
                x = torch.tensor([[nxt]], dtype=torch.long, device=device)
                logits, _ = self.step_forward(x, cache, timestamp=len(out) - 1)
        return out

    # ------------------------------------------------------------------
    def num_parameters(self, only_trainable: bool = False) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # API comoda (estilo librerias modernas)
    # ------------------------------------------------------------------
    @classmethod
    def from_preset(cls, size: str, **overrides) -> "EngraModel":
        """``EngraModel.from_preset("base", vocab_size=50257)``."""
        return cls(V5Config.from_preset(size, **overrides))

    def save(self, directory: str) -> None:
        """Guarda config.json + model.pt en ``directory``."""
        import json as _json
        import os as _os
        _os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), _os.path.join(directory, "model.pt"))
        with open(_os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
            _json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def load(cls, directory: str, map_location="cpu") -> "EngraModel":
        """Carga un modelo guardado con :meth:`save`."""
        import json as _json
        import os as _os
        with open(_os.path.join(directory, "config.json"), encoding="utf-8") as f:
            cfg = V5Config.from_dict(_json.load(f))
        model = cls(cfg)
        state = torch.load(_os.path.join(directory, "model.pt"), map_location=map_location)
        model.load_state_dict(state)
        return model

    def describe(self) -> str:
        rf = self.config.receptive_field()
        return (
            self.config.describe()
            + f"\n  parametros={self.num_parameters():,} alcance={rf['max_reach']}"
        )
