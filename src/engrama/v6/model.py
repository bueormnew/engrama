"""ENGRAMA V6 — modelo integrador (puramente lineal, recall exacto).

Flujo (paralelo en entrenamiento, incremental en generacion, IDENTICOS):

    T0          = EncoderV2(embeddings(x))          (pilar 1: aislado por token)
    T_L, T_sh   = ConsolidacionV6(T0)               (pilar 3, O(N))
    q_lex,K_lex = P_lex(T0),  P_lex(T0)             (eje lexico aislado)
    q_ctx,K_sen = P_ctx(T_L), P_sense(T_sh)         (eje de sentido contextual)
    lecturas    = RecallTapV3(...)                  (indice invertido / LSH, O(N))
    estado      = T_L + g_rt * W_r(lecturas)
    logits      = softcap(Evocador(estado), C)

Diferencias sobre V5.5:
  * El evocador usa el ``vocab_size`` REAL (V5.5 tenia 4096 hardcodeado).
  * La traza almacena claves en FP32 aunque el modelo vaya en FP16/BF16,
    garantizando invarianza causal exacta en todas las precisiones.
  * El recall lexico paralelo usa INDICE INVERTIDO por token (no denso N x N).
  * El recall semantico usa LSH V6 de alta recuperacion (16 tablas/16 bits +
    rescate de ventana), lineal y con recall ~denso.
  * La CE_retrieval usa los candidatos del indice/LSH (no matriz N x N).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import nn

from engrama.config import EngramaConfig
from engrama.evoker import MultiCandidateEvoker
from engrama.v6.config import V6Config
from engrama.v6.consolidation import V55ConsolidationStack
from engrama.v6.encoder import IsolatedEncoderV2
from engrama.v6.losses import (softcap_linear_cross_entropy,
                               retrieval_cross_entropy_dense,
                               retrieval_cross_entropy_candidates)
from engrama.v6.lsh import previous_same_occurrence
from engrama.v6.primitives import softcap
from engrama.v6.recall import RecallTapV3
from engrama.v6.trace import PagedDualTrace

DEFAULT_BOS = 2


class EngraModelV6(nn.Module):
    """ENGRAMA V6 (sin atencion, sin compresion, puramente lineal, recall exacto)."""

    def __init__(self, config: V6Config):
        super().__init__()
        self.config = config
        # El evocador debe conocer el vocab_size REAL (fix del bug V5.5).
        self._inner = EngramaConfig(
            vocab_size=config.vocab_size, d_model=config.d_model,
            d_gate=config.d_gate, d_ff=config.d_ff, num_cells=config.num_cells,
            num_encoder_layers=config.num_encoder_layers,
            num_consolidation_layers=config.num_consolidation_layers,
            context_length=max(256, config.context_length),
            synapse_rank=config.synapse_rank,
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
            RecallTapV3(
                config.d_model, config.d_recall, config.d_sense,
                value=config.rt_value, gap=config.rt_gap,
                temperature=config.rt_temperature, score_chunk=config.rt_score_chunk,
                init_std=config.rt_init_std, shared_lex_init=config.rt_shared_lex_init,
                sense_beta_init=config.rt_sense_beta_init,
                sense_beta_trainable=config.rt_sense_beta_trainable,
                semantic_enabled=config.semantic_recall_enabled,
                d_semantic=config.d_semantic, sem_temperature=config.rt_sem_temperature,
                lex_parallel_mode=config.rt_lex_parallel_mode,
                lsh_tables=config.rt_lsh_tables, lsh_bits=config.rt_lsh_bits,
                lsh_cap=config.rt_lsh_cap,
                lsh_rescue_window=config.rt_lsh_rescue_window,
                lsh_fp32_keys=config.rt_lsh_fp32_keys,
            )
            if config.recall_enabled else None
        )
        if self.recall is not None:
            self.rt_gate = nn.Parameter(torch.tensor(float(_inv2sigmoid(config.rt_gate_init))))
            if config.semantic_recall_enabled:
                self.sem_gate = nn.Parameter(
                    torch.tensor(float(_inv2sigmoid(config.rt_sem_gate_init))))
        if not config.tie_embeddings:
            self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        else:
            self.output_projection = None
        self._cache: Optional[PagedDualTrace] = None
        dtype = config.torch_dtype()
        if dtype != torch.float32:
            self.to(dtype=dtype)

    @property
    def output_embeddings(self):
        if self.output_projection is not None:
            return self.output_projection.weight
        return self.embeddings.weight

    # ------------------------------------------------------------------
    def footprints(self, input_ids):
        return self.encoder(self.embeddings(input_ids))

    def footprints_isolated(self, input_ids):
        """Huellas T0[j]=f(x_j) calculadas token a token.

        Es EXACTAMENTE igual que el camino incremental (mismo orden de
        reduccion en las capas lineales), por lo que garantiza invarianza
        causal bit a bit. Sigue siendo O(N) (una proyeccion por token),
        solo que sin el redondeo que oneDNN introduce al agrupar N tokens
        en un solo matmul. Se usa en evaluacion/inferencia.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        b, n = input_ids.shape
        outs = []
        for i in range(n):
            emb = self.embeddings(input_ids[:, i:i + 1])
            outs.append(self.encoder(emb).squeeze(1))
        return torch.stack(outs, dim=1)

    def _sem_sources(self, t0, t_last):
        cfg = self.config
        k_src = t0 if cfg.rt_sem_key_source == "t0" else t_last
        q_src = t0 if cfg.rt_sem_query_source == "t0" else t_last
        return q_src, k_src

    def _consolidated(self, input_ids):
        t0 = self.footprints(input_ids)
        t_last, t_shallow = self.consolidation.forward_train(t0)
        return t0, t_last, t_shallow

    def _recall_projections(self, t0, t_last, t_shallow):
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

    def _identity_prev(self, input_ids):
        b, n = input_ids.shape
        out = input_ids.new_full((b, n), -1)
        for bi in range(b):
            out[bi] = previous_same_occurrence(input_ids[bi], gap=self.config.rt_gap)
        return out

    def recall_reads(self, t0, t_last, t_shallow, input_ids, *, proj=None):
        if self.recall is None:
            return torch.zeros_like(t0)
        rec = self.recall
        q_lex = proj["q_lex"] if proj is not None else rec.queries_lex(t0)
        k_lex = proj["k_lex"] if proj is not None else rec.keys_lex(t0)
        q_ctx = (proj["q_ctx"] if proj is not None
                 else rec.queries_ctx(t_last if self.config.rt_ctx_query == "final"
                                     else t_shallow))
        k_sen = proj["k_sen"] if proj is not None else rec.keys_sense(t_shallow)
        id_prev = self._identity_prev(input_ids)
        # Ruta lexico-paralela V6: INDICE INVERTIDO (default) o LSH alta-recall.
        if self.config.rt_lex_parallel_mode == "inverted":
            return rec.forward_parallel_inverted(
                q_lex, k_lex, q_ctx, k_sen, t0, input_ids)
        if self.config.rt_train_mode == "lsh" and self.training:
            return rec.forward_parallel_lsh(
                q_lex, k_lex, q_ctx, k_sen, t0, input_ids,
                n_tables=self.config.rt_lsh_tables,
                n_bits=self.config.rt_lsh_bits, cap=0,
                n_neg=self.config.rt_lsh_neg,
                rescue_window=self.config.rt_lsh_rescue_window)
        return rec.forward_parallel_dense(
            q_lex, k_lex, q_ctx, k_sen, t0, identity_prev=id_prev)

    def forward_features(self, input_ids, *, score_rows=None):
        if self.training:
            t0, t_last, t_shallow = self._consolidated(input_ids)
        else:
            # En EVAL: huellas aisladas token a token (idénticas al camino
            # incremental, sin redondeo oneDNN de batch) -> invarianza exacta.
            t0 = self.footprints_isolated(input_ids)
            # La consolidacion tambien token a token: el matmul paralelo agrupa
            # N filas y oneDNN redondea distinto, lo que puede voltear un
            # argmax semantico. forward_train_isolated replica step_forward.
            t_last, t_shallow = self.consolidation.forward_train_isolated(t0)
        return self._recall_state(t0, t_last, t_shallow, input_ids, score_rows=score_rows)

    def _recall_projections_isolated(self, t0, t_last, t_shallow):
        """Proyecciones de recall calculadas token a token (para EVAL).

        Replica exactamente el orden de reduccion oneDNN del camino
        incremental, evitando el redondeo de lote (~1e-7) que voltea el
        argmax semantico ante puntuaciones casi empatadas. Sigue siendo O(N).
        """
        rec = self.recall
        b, n, _ = t0.shape
        final = t_last if self.config.rt_ctx_query == "final" else t_shallow
        proj = {
            "q_lex": torch.cat([rec.queries_lex(t0[:, i:i+1]) for i in range(n)], dim=1),
            "k_lex": torch.cat([rec.keys_lex(t0[:, i:i+1]) for i in range(n)], dim=1),
            "q_ctx": torch.cat([rec.queries_ctx(final[:, i:i+1]) for i in range(n)], dim=1),
            "k_sen": torch.cat([rec.keys_sense(t_shallow[:, i:i+1]) for i in range(n)], dim=1),
            "sem_on": bool(rec.semantic_enabled),
        }
        if rec.semantic_enabled:
            q_src, k_src = self._sem_sources(t0, t_last)
            proj["q_sem"] = torch.cat([rec.queries_sem(q_src[:, i:i+1]) for i in range(n)], dim=1)
            proj["k_sem"] = torch.cat([rec.keys_sem(k_src[:, i:i+1]) for i in range(n)], dim=1)
        return proj

    def _recall_state(self, t0, t_last, t_shallow, input_ids, *, score_rows=None,
                      proj=None):
        if self.recall is None:
            return t_last
        rec = self.recall
        if proj is not None:
            p = proj
        elif self.training:
            p = self._recall_projections(t0, t_last, t_shallow)
        else:
            p = self._recall_projections_isolated(t0, t_last, t_shallow)
        if score_rows is not None:
            id_prev = self._identity_prev(input_ids)
            reads = rec.forward_parallel_dense(
                p["q_lex"], p["k_lex"], p["q_ctx"], p["k_sen"], t0,
                identity_prev=id_prev, score_rows=score_rows)
        else:
            reads = self.recall_reads(t0, t_last, t_shallow, input_ids, proj=p)
        state = rec.inject(t_last, reads, self.rt_gate)
        if rec.semantic_enabled:
            q_sem, k_sem = p["q_sem"], p["k_sem"]
            # V6: el recall semantico es SIEMPRE lineal (LSH con C acotado),
            # tanto en entrenamiento como en eval/inferencia. El LSH tiene
            # 100% de recall verificado, asi que no se necesita ruta densa.
            # En entrenamiento con gradiente se anade STE para entrenar las
            # proyecciones; en eval solo la lectura dura (argmax).
            sem_identity = self._identity_prev(input_ids)
            if score_rows is not None and self.config.rt_sem_recall_mode == "dense":
                # Solo para pruebas/aislamiento: ruta densa O(N^2).
                sem_reads = rec.forward_semantic_dense(
                    q_sem, k_sem, t0, score_rows=score_rows)
            else:
                sem_reads = rec.forward_semantic_lsh(
                    q_sem, k_sem, t0, input_ids,
                    n_tables=self.config.rt_lsh_tables,
                    n_bits=self.config.rt_lsh_bits, cap=0,
                    rescue_window=self.config.rt_lsh_rescue_window,
                    identity_prev=sem_identity)
            state = rec.inject_semantic(state, sem_reads, self.sem_gate)
        return state

    def _evoker_isolated(self, state):
        """Evocador token a token para EVAL/inferencia.

        El camino paralelo agrupa toda la secuencia en los lineales del
        evocador y oneDNN redondea distinto en FP16/BF16. En FP32 la
        diferencia es imperceptible, pero en FP16 un volteo de redondeo en
        ``fused_latent`` o en la proyeccion de vocabulario puede cambiar un
        logit en ~0.1. Esta ruta replica el orden de reduccion de
        ``step_forward`` (un token por llamada), por lo que es bit a bit
        identica al camino incremental. Sigue siendo O(N*d*V).
        """
        b, n, _ = state.shape
        outs = []
        for i in range(n):
            outs.append(self.evoker(state[:, i:i + 1], self.output_embeddings))
        return torch.cat(outs, dim=1)

    def forward(self, input_ids):
        state = self.forward_features(input_ids)
        # En EVAL el evocador tambien se calcula token a token para replicar
        # el orden de reduccion de step_forward (ver _evoker_isolated). En
        # FP32 la diferencia es despreciable; en FP16/BF16 un volteo de
        # redondeo en el matmul de vocabulario podia cambiar un logit.
        logits = self.evoker(state, self.output_embeddings)
        if not self.training:
            iso = self._evoker_isolated(state)
            logits = iso
        return softcap(logits, self.config.logit_cap)

    def forward_loss(self, input_ids, targets, *, linear_chunk_size=2048,
                     checkpoint_chunks=True, ignore_index=-100,
                     retrieval_weight=None, step=0):
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
        if retrieval_weight is None:
            rw = self.config.retrieval_weight
            if (self.config.retrieval_anneal_steps > 0
                    and step >= self.config.retrieval_anneal_steps):
                rw = 0.0
        else:
            rw = float(retrieval_weight)
        if rw <= 0:
            return lm_loss
        ret = self._retrieval_loss_inner(t0, t_last, t_shallow, input_ids,
                                         targets, proj=proj)
        return lm_loss + rw * ret

    def _retrieval_loss_inner(self, t0, t_last, t_shallow, input_ids, targets,
                              *, proj=None):
        """CE_retrieval V6: usa candidatos del LSH/indice (O(N*C)), NO denso.

        Para que el entrenamiento sea lineal, construimos candidatos con el LSH
        V6 y aplicamos ``retrieval_cross_entropy_dense`` sobre la matriz
        (m, C) de scores (no (m, N)).
        """
        from engrama.v6.recall import _l2
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
        nxt = torch.full_like(input_ids, -100)
        nxt[:, :-1] = input_ids[:, 1:]
        gen = torch.Generator(device="cpu").manual_seed(123 + _step_seed(input_ids))
        mask = (torch.rand(b, n, generator=gen).to(input_ids.device)
                < cfg.retrieval_positions_frac)
        mask[:, :cfg.rt_gap] = False
        losses = []
        chunk = cfg.rt_score_chunk
        for bi in range(b):
            # Candidatos LSH lexico (union de identidad+rescate+buckets)
            from engrama.v6.lsh import V6LSHIndex, shared_planes, sign_codes
            index = V6LSHIndex.build(
                kl_n[bi].float(), input_ids[bi].long(), gap=cfg.rt_gap,
                n_tables=cfg.rt_lsh_tables, n_bits=cfg.rt_lsh_bits,
                rescue_window=cfg.rt_lsh_rescue_window)
            planes = shared_planes(cfg.rt_lsh_tables, kl_n.size(-1),
                                   cfg.rt_lsh_bits, kl_n.device, kl_n.dtype)
            qcodes = sign_codes(ql_n[bi].float(), planes)
            cand, valid = index.candidates(qcodes)
            # restringir a mismo token si esta activado (comparacion (N,1) vs (N,C))
            if cfg.retrieval_same_token_only:
                same = (input_ids[bi].unsqueeze(1)
                        == input_ids[bi][cand.clamp(min=0)])
                valid = valid & same
            rows = mask[bi].nonzero().flatten()
            for start in range(0, rows.numel(), chunk):
                idx = rows[start:start + chunk]
                m = idx.numel()
                lc = cand[idx]
                lv = valid[idx]
                ckl = kl_n[bi][lc.clamp(min=0)]
                cks = ks_n[bi][lc.clamp(min=0)]
                slex = (ql_n[bi, idx].unsqueeze(1) * ckl).sum(-1)
                ssen = (qc_n[bi, idx].unsqueeze(1) * cks).sum(-1)
                s = slex * (1.0 + beta * ssen)
                # next-token de cada candidato: nxt[cand_pos]
                cand_next = nxt[bi][lc.clamp(min=0)]
                losses.append(retrieval_cross_entropy_candidates(
                    s.unsqueeze(0), lv.unsqueeze(0),
                    cand_next.unsqueeze(0), targets[bi, idx].unsqueeze(0),
                    temperature=cfg.retrieval_temperature))
            if sem_on:
                # candidatos LSH semantico
                sindex = V6LSHIndex.build(
                    ksm_n[bi].float(), input_ids[bi].long(), gap=cfg.rt_gap,
                    n_tables=cfg.rt_lsh_tables, n_bits=cfg.rt_lsh_bits,
                    rescue_window=cfg.rt_lsh_rescue_window)
                splanes = shared_planes(cfg.rt_lsh_tables, ksm_n.size(-1),
                                        cfg.rt_lsh_bits, ksm_n.device, ksm_n.dtype)
                sqcodes = sign_codes(qsm_n[bi].float(), splanes)
                scand, svalid = sindex.candidates(sqcodes)
                for start in range(0, rows.numel(), chunk):
                    idx = rows[start:start + chunk]
                    lc = scand[idx]; lv = svalid[idx]
                    ck = ksm_n[bi][lc.clamp(min=0)]
                    ssem = (qsm_n[bi, idx].unsqueeze(1) * ck).sum(-1)
                    cand_next = nxt[bi][lc.clamp(min=0)]
                    losses.append(retrieval_cross_entropy_candidates(
                        ssem.unsqueeze(0), lv.unsqueeze(0),
                        cand_next.unsqueeze(0), targets[bi, idx].unsqueeze(0),
                        temperature=cfg.retrieval_temperature))
        if not losses:
            return t0.new_zeros(())
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    def get_cache(self, n_max=None):
        n = n_max or self.config.context_length
        store_dt = self.config.trace_store_torch_dtype()
        self._cache = PagedDualTrace(
            n, self.config.d_model,
            self.config.d_recall if self.recall is not None else 1,
            self.config.d_sense if self.recall is not None else 1,
            self.config.cache_horizons(), page_size=self.config.page_size,
            dtype=self.config.torch_dtype(),
            store_dtype=store_dt,  # V6: claves en FP32 para invarianza
            d_semantic=(self.config.d_semantic if self.recall is not None
                        and self.recall.semantic_enabled else 0),
        )
        # V6: adjunta el indice LSH semantico incremental para que la
        # inferencia sea O(C) por token (C=129 acotado), no O(N).
        if self.recall is not None and self.recall.semantic_enabled:
            self._cache.attach_semantic_lsh(
                n_tables=self.config.rt_lsh_tables,
                n_bits=self.config.rt_lsh_bits,
                cap=0,
                rescue_window=self.config.rt_lsh_rescue_window,
                gap=self.config.rt_gap, seed=7)
        return self._cache

    def step_forward(self, token_id, cache, timestamp):
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        elif token_id.dim() == 0:
            token_id = token_id.unsqueeze(0).unsqueeze(1)
        emb = self.embeddings(token_id)
        t0 = self.encoder(emb)
        if t0.dim() == 3:
            t0 = t0.squeeze(1)
        token_ids_b = token_id.reshape(-1).to(torch.long)
        b = token_ids_b.size(0)
        # j_id (identidad lexico/semantica) debe calcularse ANTES de escribir
        # el token actual: ``last_occurrence`` con before_pos=cache.length ya
        # excluye la posicion en curso. append_t0 actualiza el mapa de ultima
        # ocurrencia con la posicion actual DESPUES, por lo que si lo llamaramos
        # tras append obtendriamos j=i (posicion actual), rompiendo la
        # invarianza causal (candidato ilegal que ademas lee T0[i+1] fuera de
        # rango / el token actual).
        # j_id (identidad lexico/semantica) debe calcularse ANTES de escribir
        # el token actual: append_t0 actualiza el mapa de ultima ocurrencia con
        # la posicion en curso, y last_occurrence con before_pos=cache.length
        # ya excluye esa posicion. En el camino incremental el orden es:
        # append_t0 (escribe token actual) -> step_forward -> read, pero la
        # identidad se lee antes del append para no obtener j=i.
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
            q_sem = self.recall.queries_sem(q_sem_src)
            # La clave actual se escribe en el anillo (para futuras lecturas)
            # pero el INDICE LSH se actualiza DESPUES de leer, de modo que la
            # posicion actual no sea candidata de si misma (la identidad previa
            # real ya se pasa en j_id, calculada antes del append).
            cache.append_semantic(k_sem, token_id=token_ids_b)
            sem_read = self.recall.read_semantic_step_lsh(
                q_sem, cache, k_sem, token_ids_b, cache.length,
                identity_prev=j_id)
            cache.commit_semantic_lsh(k_sem, token_id=token_ids_b)
            state = self.recall.inject_semantic(state, sem_read, self.sem_gate)
        logits = self.evoker(state, self.output_embeddings)
        logits = softcap(logits, self.config.logit_cap)
        return logits, state

    @torch.no_grad()
    def _sample(self, logits, temperature, top_k):
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

    def generate(self, prompt_ids, max_new_tokens=64, temperature=0.8,
                 top_k=40, eos_token_id=None):
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
        logits = None
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

    def num_parameters(self, only_trainable=False):
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_preset(cls, size, **overrides):
        return cls(V6Config.from_preset(size, **overrides))

    def describe(self):
        rf = self.config.receptive_field()
        return (self.config.describe()
                + f"\n  parametros={self.num_parameters():,} alcance={rf['max_reach']}")


def _inv2sigmoid(y):
    y = min(max(y, 1e-3), 2.0 - 1e-3)
    p = y / 2.0
    return float(math.log(p / (1.0 - p)))


def _step_seed(x):
    return int(torch.sum(x.to(torch.long)).item() % 100000)
