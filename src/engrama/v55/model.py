"""ENGRAMA V5.5 — modelo integrador.

Flujo (todo paralelo en entrenamiento, incremental en generacion, IDENTICOS por
invarianza causal)::

    T0          = EncoderV2(embeddings(x))          (pilar 1: aislado por token)
    T_L, T_sh   = ConsolidacionV5(T0)               (pilar 3: suavizado + tap)
    q_lex,K_lex = P_lex(T0),  P_lex(T0)             (eje lexico aislado)
    q_ctx,K_sen = P_ctx(T_L), P_sense(T_sh)         (eje de sentido contextual)
    lecturas    = RecallTapV2(...)                  (top-1 asimetrico, pilar 4)
    estado      = T_L + g_rt * W_r(lecturas)        (g_rt init 1.0, W_r std 0.02)
    logits      = softcap(Evocador(estado), C)      (pilar 6)

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import nn

from engrama.evoker import MultiCandidateEvoker
from engrama.config import EngramaConfig
from engrama.v55.config import V55Config
from engrama.v55.consolidation import V55ConsolidationStack
from engrama.v55.encoder import IsolatedEncoderV2
from engrama.v55.losses import (retrieval_cross_entropy, retrieval_cross_entropy_dense,
                                softcap_linear_cross_entropy)
from engrama.v55.primitives import softcap
from engrama.v55.recall import RecallTapV2
from engrama.v55.trace import PagedDualTrace

DEFAULT_BOS = 2


class EngraModelV55(nn.Module):
    """ENGRAMA V5.5 (sin atencion, sin compresion, recuperacion exacta)."""

    def __init__(self, config: V55Config):
        super().__init__()
        self.config = config
        self._inner = EngramaConfig(
            vocab_size=config.vocab_size, d_model=config.d_model,
            d_gate=config.d_gate, d_ff=config.d_ff, num_cells=config.num_cells,
            num_encoder_layers=config.num_encoder_layers,
            num_consolidation_layers=config.num_consolidation_layers,
            context_length=256, synapse_rank=config.synapse_rank,
            num_candidates=config.num_candidates,
            candidate_aggregation="latent_fusion", activation=config.activation,
            dropout=config.dropout, tie_embeddings=config.tie_embeddings,
            version="v4", stable_init=True,
        )
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = IsolatedEncoderV2(
            config.d_model, config.d_gate, config.num_cells, config.d_ff,
            config.num_encoder_layers, config.synapse_rank, config.dropout)
        self.consolidation = V55ConsolidationStack(config)
        self.evoker = MultiCandidateEvoker(self._inner)
        self.recall = (
            RecallTapV2(
                config.d_model, config.d_recall, config.d_sense,
                value=config.rt_value, gap=config.rt_gap,
                temperature=config.rt_temperature, score_chunk=config.rt_score_chunk,
                init_std=config.rt_init_std, shared_lex_init=config.rt_shared_lex_init,
                sense_beta_init=config.rt_sense_beta_init,
                sense_beta_trainable=config.rt_sense_beta_trainable,
                semantic_enabled=config.semantic_recall_enabled,
                d_semantic=config.d_semantic, sem_temperature=config.rt_sem_temperature,
            )
            if config.recall_enabled else None
        )
        if self.recall is not None:
            self.rt_gate = nn.Parameter(torch.tensor(float(_inv2sigmoid(config.rt_gate_init))))
            if config.semantic_recall_enabled:
                self.sem_gate = nn.Parameter(
                    torch.tensor(float(_inv2sigmoid(config.rt_sem_gate_init))))
        if not config.tie_embeddings:
            self.output_projection: Optional[nn.Linear] = nn.Linear(
                config.d_model, config.vocab_size, bias=False)
        else:
            self.output_projection = None
        self._cache: Optional[PagedDualTrace] = None
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

    def _sem_sources(self, t0: torch.Tensor, t_last: torch.Tensor):
        """Tensores fuente para claves/consultas semanticas segun config.
        ``"t0"`` = huella aislada pristina (mas fiel al aislamiento, pares
        distinguibles); ``"t_last"`` = consolidacion final (significado)."""
        cfg = self.config
        k_src = t0 if cfg.rt_sem_key_source == "t0" else t_last
        q_src = t0 if cfg.rt_sem_query_source == "t0" else t_last
        return q_src, k_src

    def _consolidated(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(t0, t_last, t_shallow)`` — compartido por features y la loss de
        recuperacion (evita recalcular huellas)."""
        t0 = self.footprints(input_ids)
        t_last, t_shallow = self.consolidation.forward_train(t0)
        return t0, t_last, t_shallow

    def _recall_projections(
        self, t0: torch.Tensor, t_last: torch.Tensor, t_shallow: torch.Tensor,
    ) -> Optional[dict]:
        """Calcula UNA vez todas las proyecciones del Recall Tap (lexico, sentido
        y semantico). Lo comparten la lectura (``_recall_state``) y la
        ``CE_retrieval`` (``_retrieval_loss_inner``) para evitar recalcular las
        mismas ``Linear`` dos veces por paso de entrenamiento."""
        rec = self.recall
        if rec is None:
            return None
        final = t_last if self.config.rt_ctx_query == "final" else t_shallow
        proj = {
            "q_lex": rec.queries_lex(t0),
            "k_lex": rec.keys_lex(t0),
            "q_ctx": rec.queries_ctx(final),
            "k_sen": rec.keys_sense(t_shallow),
            "sem_on": bool(rec.semantic_enabled),
        }
        if rec.semantic_enabled:
            q_sem_src, k_sem_src = self._sem_sources(t0, t_last)
            proj["q_sem"] = rec.queries_sem(q_sem_src)
            proj["k_sem"] = rec.keys_sem(k_sem_src)
        return proj

    def _identity_prev(self, input_ids: torch.Tensor) -> torch.Tensor:
        from engrama.v55.lsh import previous_same_occurrence
        b, n = input_ids.shape
        out = input_ids.new_full((b, n), -1)
        for bi in range(b):
            out[bi] = previous_same_occurrence(input_ids[bi], gap=self.config.rt_gap)
        return out

    def recall_reads(
        self, t0: torch.Tensor, t_last: torch.Tensor, t_shallow: torch.Tensor,
        input_ids: torch.Tensor, *, proj: Optional[dict] = None,
    ) -> torch.Tensor:
        """Calcula las lecturas del Recall Tap asimetrico (B,N,d). Si se pasa
        ``proj`` (proyecciones ya calculadas) las reutiliza en vez de recalcular."""
        if self.recall is None:
            return torch.zeros_like(t0)
        q_lex = proj["q_lex"] if proj is not None else self.recall.queries_lex(t0)
        k_lex = proj["k_lex"] if proj is not None else self.recall.keys_lex(t0)
        q_ctx = (proj["q_ctx"] if proj is not None
                 else self.recall.queries_ctx(t_last if self.config.rt_ctx_query == "final"
                                              else t_shallow))
        k_sen = proj["k_sen"] if proj is not None else self.recall.keys_sense(t_shallow)
        id_prev = self._identity_prev(input_ids)
        if self.config.rt_train_mode == "lsh" and self.training:
            return self.recall.forward_parallel_lsh(
                q_lex, k_lex, q_ctx, k_sen, t0, input_ids,
                n_tables=self.config.rt_lsh_tables, n_bits=self.config.rt_lsh_bits,
                cap=self.config.rt_lsh_cap, n_neg=self.config.rt_lsh_neg,
                hamming=self.config.rt_lsh_hamming)
        return self.recall.forward_parallel_dense(
            q_lex, k_lex, q_ctx, k_sen, t0, identity_prev=id_prev)

    def forward_features(
        self, input_ids: torch.Tensor, *, score_rows: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        t0, t_last, t_shallow = self._consolidated(input_ids)
        return self._recall_state(t0, t_last, t_shallow, input_ids, score_rows=score_rows)

    def _recall_state(
        self, t0: torch.Tensor, t_last: torch.Tensor, t_shallow: torch.Tensor,
        input_ids: torch.Tensor, *, score_rows: Optional[torch.Tensor] = None,
        proj: Optional[dict] = None,
    ) -> torch.Tensor:
        """Estado con los dos taps inyectados: lexico (copia exacta) + semantico
        (asociativo). ``score_rows`` restringe la lectura a filas de consulta.
        ``proj`` reutiliza proyecciones precalculadas (DRY con la CE_retrieval)."""
        if self.recall is None:
            return t_last
        p = proj if proj is not None else self._recall_projections(t0, t_last, t_shallow)
        if score_rows is not None:
            id_prev = self._identity_prev(input_ids)
            reads = self.recall.forward_parallel_dense(
                p["q_lex"], p["k_lex"], p["q_ctx"], p["k_sen"], t0,
                identity_prev=id_prev, score_rows=score_rows)
        else:
            reads = self.recall_reads(t0, t_last, t_shallow, input_ids, proj=p)
        state = self.recall.inject(t_last, reads, self.rt_gate)
        # tap semantico (asociativo): recupera por significado sobre todos
        if self.recall.semantic_enabled:
            q_sem, k_sem = p["q_sem"], p["k_sem"]
            if self.config.rt_sem_recall_mode == "lsh":
                sem_reads = self.recall.forward_semantic_lsh(
                    q_sem, k_sem, t0, input_ids,
                    n_tables=self.config.rt_lsh_tables,
                    n_bits=self.config.rt_lsh_bits, cap=self.config.rt_lsh_cap)
            else:
                sem_reads = self.recall.forward_semantic_dense(
                    q_sem, k_sem, t0, score_rows=score_rows)
            state = self.recall.inject_semantic(state, sem_reads, self.sem_gate)
        return state

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        state = self.forward_features(input_ids)
        logits = self.evoker(state, self.output_embeddings)
        return softcap(logits, self.config.logit_cap)

    def forward_loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        *,
        linear_chunk_size: int = 2048,
        checkpoint_chunks: bool = True,
        ignore_index: int = -100,
        retrieval_weight: Optional[float] = None,
        step: int = 0,
    ) -> torch.Tensor:
        """``CE_LM + lambda * CE_retrieval`` (Seccion 6) — **CE_retrieval es
        INTERNA y OBLIGATORIA por defecto** (auto-supervisada, sin etiquetas:
        es la senal que activa el eje de sentido).

        ``retrieval_weight=None`` usa ``config.retrieval_weight`` (0.2). El
        *annealing* a 0 solo aplica si ``config.retrieval_anneal_steps > 0``
        (por defecto 0 = siempre activa). Pasa ``retrieval_weight=0`` para
        desactivarla explicitamente (p. ej. tareas de copia pura donde no hace
        falta).
        """
        t0, t_last, t_shallow = self._consolidated(input_ids)
        proj = self._recall_projections(t0, t_last, t_shallow)
        state = self._recall_state(t0, t_last, t_shallow, input_ids, proj=proj)
        latent = self.evoker.fused_latent(state)
        lm_loss = softcap_linear_cross_entropy(
            latent, self.output_embeddings, targets,
            scale=1.0 / math.sqrt(self.config.d_model), cap=self.config.logit_cap,
            chunk_size=linear_chunk_size, ignore_index=ignore_index,
            checkpoint_chunks=checkpoint_chunks)
        if self.recall is None:
            return lm_loss
        # CE_retrieval auto-supervisado: OBLIGATORIO por defecto.
        if retrieval_weight is None:
            rw = self.config.retrieval_weight
            if self.config.retrieval_anneal_steps > 0 and step >= self.config.retrieval_anneal_steps:
                rw = 0.0
        else:
            rw = float(retrieval_weight)
        if rw <= 0:
            return lm_loss
        ret = self._retrieval_loss_inner(t0, t_last, t_shallow, input_ids, targets, proj=proj)
        return lm_loss + rw * ret

    def retrieval_loss(self, input_ids: torch.Tensor,
                       targets: torch.Tensor) -> torch.Tensor:
        """CE auto-supervisado del Recall Tap (Seccion 6) — version densa.

        Para cada posicion supervisada ``i`` (fraccion del config), puntua contra
        TODAS las posiciones previas ``j`` y empuja el score hacia los ``j`` cuyo
        token siguiente (``token[j+1]``) coincide con el objetivo ``y_i``. Es la
        senal que activa el eje de sentido (LM pura no basta). Vectorizada
        (matvec BLAS, sin construir indices LSH).
        """
        t0, t_last, t_shallow = self._consolidated(input_ids)
        return self._retrieval_loss_inner(t0, t_last, t_shallow, input_ids, targets)

    def _retrieval_loss_inner(
        self, t0: torch.Tensor, t_last: torch.Tensor, t_shallow: torch.Tensor,
        input_ids: torch.Tensor, targets: torch.Tensor,
        *, proj: Optional[dict] = None,
    ) -> torch.Tensor:
        """CE denso del Recall Tap. Reutiliza ``(t0, t_last, t_shallow)`` ya
        calculados (no recalcula huellas). Vectorizado por filas supervisadas.
        ``proj`` permite reutilizar las proyecciones ya calculadas para la
        lectura (evita recalcular las ``Linear``)."""
        from engrama.v55.recall import _l2
        cfg = self.config
        rec = self.recall
        b, n = input_ids.shape
        beta = rec.beta.float()
        p = proj if proj is not None else self._recall_projections(t0, t_last, t_shallow)
        ql_n, kl_n = _l2(p["q_lex"]), _l2(p["k_lex"])
        qc_n, ks_n = _l2(p["q_ctx"]), _l2(p["k_sen"])
        sem_on = p["sem_on"]
        if sem_on:
            qsm_n, ksm_n = _l2(p["q_sem"]), _l2(p["k_sem"])
        # next_tokens[j] = token en j+1 (lo que aporta leer la posicion j)
        nxt = torch.full_like(input_ids, ignore_idx_const())
        nxt[:, :-1] = input_ids[:, 1:]
        gen = torch.Generator(device="cpu").manual_seed(123 + step_seed(input_ids))
        mask = (torch.rand(b, n, generator=gen).to(input_ids.device)
                < cfg.retrieval_positions_frac)
        mask[:, :cfg.rt_gap] = False
        losses = []
        chunk = cfg.rt_score_chunk
        colj = torch.arange(n, device=t0.device)
        for bi in range(b):
            rows = mask[bi].nonzero().flatten()
            if rows.numel() == 0:
                continue
            klt = kl_n[bi].transpose(-1, -2)     # (dk, N)
            kst = ks_n[bi].transpose(-1, -2)     # (ds, N)
            ids_b = input_ids[bi]               # (N,)
            for start in range(0, rows.numel(), chunk):
                idx = rows[start:start + chunk]
                slex = ql_n[bi, idx] @ klt        # (m, N)  matvec BLAS
                ssen = qc_n[bi, idx] @ kst        # (m, N)
                s = slex * (1.0 + beta * ssen)
                causal = colj.unsqueeze(0) <= (idx.unsqueeze(1) - cfg.rt_gap)  # (m, N)
                if cfg.retrieval_same_token_only:
                    # solo candidatos del MISMO token: ahi el sentido desempata
                    # (no diluye el gradiente con copia de tokens distintos).
                    tok_i = ids_b[idx].unsqueeze(1)            # (m, 1)
                    same = ids_b.unsqueeze(0) == tok_i         # (m, N)
                    valid = causal & same
                else:
                    valid = causal
                losses.append(retrieval_cross_entropy_dense(
                    s.unsqueeze(0), valid.unsqueeze(0),
                    nxt[bi].unsqueeze(0), targets[bi, idx].unsqueeze(0),
                    temperature=cfg.retrieval_temperature))
        # --- termino SEMANTICO: asociativo, sobre TODOS los candidatos (causal) ---
        if sem_on:
            sem_losses = []
            for bi in range(b):
                rows = mask[bi].nonzero().flatten()
                if rows.numel() == 0:
                    continue
                ksm_t = ksm_n[bi].transpose(-1, -2)
                for start in range(0, rows.numel(), chunk):
                    idx = rows[start:start + chunk]
                    ssem = qsm_n[bi, idx] @ ksm_t          # (m, N) cos semantico
                    causal = colj.unsqueeze(0) <= (idx.unsqueeze(1) - cfg.rt_gap)
                    sem_losses.append(retrieval_cross_entropy_dense(
                        ssem.unsqueeze(0), causal.unsqueeze(0),
                        nxt[bi].unsqueeze(0), targets[bi, idx].unsqueeze(0),
                        temperature=cfg.retrieval_temperature))
            lex_mean = torch.stack(losses).mean() if losses else t0.new_zeros(())
            sem_mean = torch.stack(sem_losses).mean() if sem_losses else t0.new_zeros(())
            return lex_mean + sem_mean
        if not losses:
            return t0.new_zeros(())
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Incremental (generacion) — cache nativa paginada
    # ------------------------------------------------------------------
    def get_cache(self, n_max: Optional[int] = None) -> PagedDualTrace:
        n = n_max or self.config.context_length
        # store_dtype = dtype de computo -> invarianza causal EXACTA (paralelo
        # == incremental). Para ahorrar memoria en inferencia larga puede
        # pasarse una traza con store_dtype=float16 (divergencia ~1e-3).
        self._cache = PagedDualTrace(
            n, self.config.d_model,
            self.config.d_recall if self.recall is not None else 1,
            self.config.d_sense if self.recall is not None else 1,
            self.config.cache_horizons(), page_size=self.config.page_size,
            dtype=self.config.torch_dtype(),
            d_semantic=(self.config.d_semantic if self.recall is not None
                        and self.recall.semantic_enabled else 0),
        )
        return self._cache

    def step_forward(
        self, token_id: torch.Tensor, cache: PagedDualTrace, timestamp: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        elif token_id.dim() == 0:
            token_id = token_id.unsqueeze(0).unsqueeze(1)
        emb = self.embeddings(token_id)
        t0 = self.encoder(emb)
        if t0.dim() == 3:
            t0 = t0.squeeze(1)
        token_ids_b = token_id.reshape(-1).to(torch.long)      # (B,)
        b = token_ids_b.size(0)
        # fast path de identidad por lote: consultar ANTES de escribir.
        j_id = None
        if self.recall is not None:
            j_id = torch.tensor(
                [cache.last_occurrence(int(token_ids_b[bi].item()), batch_idx=bi,
                                       before_pos=cache.length)
                 for bi in range(b)],
                device=t0.device, dtype=torch.long)
        k_lex = self.recall.keys_lex(t0) if self.recall is not None else None
        cache.append_t0(t0, k_lex, token_id=token_ids_b)
        t_last, t_shallow = self.consolidation.step_forward(cache)
        read = None
        if self.recall is not None:
            k_sen = self.recall.keys_sense(t_shallow)
            cache.append_shallow(t_shallow, k_sen)
            q_lex = self.recall.queries_lex(t0)
            q_ctx = self.recall.queries_ctx(t_last)
            read = self.recall.read_step(
                q_lex, cache.linear_klex(), q_ctx, cache.linear_ksen(),
                cache.linear_t0(), cache.length, last_occurrence=j_id,
                threshold=self.config.rt_score_threshold,
                use_fast_path=self.config.rt_use_identity_fast_path)
        state = self.recall.inject(t_last, read, self.rt_gate) if self.recall else t_last
        if self.recall is not None and self.recall.semantic_enabled:
            q_sem_src, k_sem_src = self._sem_sources(t0, t_last)
            k_sem = self.recall.keys_sem(k_sem_src)
            cache.append_semantic(k_sem)
            q_sem = self.recall.queries_sem(q_sem_src)
            sem_read = self.recall.read_semantic_step(
                q_sem, cache.linear_ksem(), cache.linear_t0(), cache.length)
            state = self.recall.inject_semantic(state, sem_read, self.sem_gate)
        logits = self.evoker(state, self.output_embeddings)
        logits = softcap(logits, self.config.logit_cap)
        return logits, state

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
        eos_token_id: Optional[int] = None,
    ) -> List[int]:
        if not prompt_ids:
            prompt_ids = [DEFAULT_BOS]
        for t in prompt_ids:
            if not (0 <= t < self.config.vocab_size):
                raise ValueError(f"token {t} fuera del vocabulario")
        self.eval()
        device = next(self.parameters()).device
        needed = len(prompt_ids) + max_new_tokens
        n_max = max(self.config.context_length, needed)
        cache = self.get_cache(n_max=n_max)
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
    @classmethod
    def from_preset(cls, size: str, **overrides) -> "EngraModelV55":
        return cls(V55Config.from_preset(size, **overrides))

    def save(self, directory: str) -> None:
        import json as _json
        import os as _os
        _os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), _os.path.join(directory, "model.pt"))
        with open(_os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
            _json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def load(cls, directory: str, map_location="cpu") -> "EngraModelV55":
        import json as _json
        import os as _os
        with open(_os.path.join(directory, "config.json"), encoding="utf-8") as f:
            cfg = V55Config.from_dict(_json.load(f))
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


def _inv2sigmoid(y: float) -> float:
    y = min(max(y, 1e-3), 2.0 - 1e-3)
    p = y / 2.0
    return float(math.log(p / (1.0 - p)))


def step_seed(x: torch.Tensor) -> int:
    return int(torch.sum(x.to(torch.long)).item() % 100000)


def ignore_idx_const() -> int:
    return -100
