"""ENGRAMA V6 — Recall Tap (Pilar 4), version PURAMENTE LINEAL.

Mismo principio que V5.5 (score lexico-dominante * (1 + beta*score_sentido),
argmax duro causal, lectura unica de T0[j*+1]), pero con TRES correcciones
estructurales:

1. **Lexico por INDICE INVERTIDO** (nunca denso O(N^2)): las posiciones con
   el mismo token se agrupan. Para cada consulta i solo se puntuan las
   ocurrencias de ``token[i]`` (mas el rescate de ventana). Coste
   ``O(N * f_i * d_k)`` donde ``f_i`` es la frecuencia del token; promedio
   ``O(N*d_k)`` y cero para tokens unicos.
2. **Semantico por LSH de alta recuperacion** (16 tablas, 16 bits, sin cap,
   con rescate denso de ventana fija). El LSH genera candidatos; sobre esos
   candidatos se hace el rerank exacto por coseno (que es O(N*C*d_sem) con C
   medio pequeno, NO O(N^2)). El rescate de W=256 posiciones recientes
   garantiza que ningun match reciente se pierde por hashing.
3. **Claves FP32 siempre**: los codigos y el argmax se calculan en FP32,
   incluso si el modelo va en FP16/BF16. Esto hace que paralelo e incremental
   sean bit-a-bit consistentes en todas las precisiones.

NO es atencion: sin softmax sobre el eje temporal, sin matriz N x N,
sin promedio ponderado. Solo productos punto + argmax duro + gather.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from engrama.v6.lsh import (
    V6LSHIndex, shared_planes, sign_codes, previous_same_occurrence,
)

_NEG = -1.0e30
SEM_TIE_DECIMALS = 4
# Por debajo de este coseno lexico NO hay induccion: las posiciones con el mismo
# token tienen K_lex IDENTICO => slex=1.0 exacto; cualquier otro token esta a
# ~1/sqrt(d_k) (<0.3). Si el maximo slex no supera el umbral, no hubo ocurrencia
# previa de MI token -> la lectura debe ser CERO (no un candidato arbitrario).
# Esto hace que el indice invertido y el camino incremental sean bit-idénticos.
_LEX_INDUCTION_THRESHOLD = 0.999


def _l2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalizacion L2 en FP32 SIEMPRE (clave para la invarianza multi-dtype)."""
    out_dtype = x.dtype
    return F.normalize(x.float(), dim=-1, eps=eps).to(out_dtype)


def _lex_dominant_argmax(slex, ssen, valid, positions=None, tol: float = 1e-3):
    """Seleccion lexico-dominante (igual que V5.5 pero tolerante a FP16).

    1. Solo candidatos con score_lex a tol del maximo (mismo token).
    2. Entre ellos, mayor score_sense (desempate contextual).
    3. Empate -> ocurrencia MAS RECIENTE.
    Operamos en FP32 internamente.
    """
    slex = slex.float()
    ssen = ssen.float()
    slex_m = torch.where(valid, slex, torch.full_like(slex, _NEG))
    max_lex = slex_m.max(dim=-1, keepdim=True).values
    lex_mask = (slex_m >= max_lex - tol) & valid
    key = torch.where(lex_mask, ssen, torch.full_like(ssen, _NEG))
    max_sen = key.max(dim=-1, keepdim=True).values
    sen_mask = lex_mask & (key >= max_sen - tol)
    if positions is None:
        idx = torch.arange(key.size(-1), device=key.device, dtype=torch.float32)
        poskey = torch.where(sen_mask, idx, torch.full_like(idx, -1.0))
    else:
        poskey = torch.where(sen_mask, positions.float(),
                            torch.full_like(positions.float(), -1.0))
    return poskey.argmax(dim=-1)


class RecallTapV3(nn.Module):
    """Recall Tap V6: lexico por indice invertido + semantico por LSH alta recall."""

    def __init__(
        self,
        d_model: int,
        d_recall: int,
        d_sense: int,
        *,
        value: str = "next",
        gap: int = 1,
        temperature: float = 0.5,
        score_chunk: int = 1024,
        init_std: float = 0.1,
        shared_lex_init: bool = True,
        sense_beta_init: float = 0.3,
        sense_beta_trainable: bool = True,
        semantic_enabled: bool = True,
        d_semantic: int = 64,
        sem_temperature: float = 0.1,
        # V6:
        lex_parallel_mode: str = "inverted",
        lsh_tables: int = 16,
        lsh_bits: int = 16,
        lsh_cap: int = 4,
        lsh_rescue_window: int = 256,
        lsh_fp32_keys: bool = True,
        lexical_rescue_window: int = 256,
    ):
        super().__init__()
        if value not in ("next", "self"):
            raise ValueError(f"rt_value debe ser 'next' o 'self', no {value!r}")
        if gap < 1:
            raise ValueError("rt_gap >= 1")
        if lex_parallel_mode not in ("inverted", "dense"):
            raise ValueError("lex_parallel_mode debe ser 'inverted' o 'dense'")
        self.d_model = d_model
        self.d_recall = d_recall
        self.d_sense = d_sense
        self.d_semantic = int(d_semantic)
        self.value = value
        self.gap = gap
        self.temperature = max(1e-2, float(temperature))
        self.score_chunk = max(64, int(score_chunk))
        self.lex_parallel_mode = lex_parallel_mode
        self.lsh_tables = int(lsh_tables)
        self.lsh_bits = int(lsh_bits)
        self.lsh_cap = int(lsh_cap)
        self.lsh_rescue_window = int(lsh_rescue_window)
        self.lsh_fp32_keys = bool(lsh_fp32_keys)
        self.lexical_rescue_window = int(lexical_rescue_window)

        self.p_q_lex = nn.Linear(d_model, d_recall, bias=False)
        self.p_k_lex = nn.Linear(d_model, d_recall, bias=False)
        nn.init.normal_(self.p_q_lex.weight, std=init_std)
        if shared_lex_init:
            with torch.no_grad():
                self.p_k_lex.weight.copy_(self.p_q_lex.weight)
        else:
            nn.init.normal_(self.p_k_lex.weight, std=init_std)

        self.p_q_ctx = nn.Linear(d_model, d_sense, bias=False)
        self.p_k_sense = nn.Linear(d_model, d_sense, bias=False)
        nn.init.normal_(self.p_q_ctx.weight, std=init_std)
        nn.init.normal_(self.p_k_sense.weight, std=init_std)

        self.beta = nn.Parameter(torch.tensor(float(sense_beta_init)))
        if not sense_beta_trainable:
            self.beta.requires_grad_(False)

        self.w_read = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.w_read.weight, std=0.02)

        self.semantic_enabled = bool(semantic_enabled)
        self.sem_temperature = max(1e-2, float(sem_temperature))
        if self.semantic_enabled:
            self.p_q_sem = nn.Linear(d_model, d_semantic, bias=False)
            self.p_k_sem = nn.Linear(d_model, d_semantic, bias=False)
            nn.init.normal_(self.p_q_sem.weight, std=init_std)
            nn.init.normal_(self.p_k_sem.weight, std=init_std)
            self.w_read_sem = nn.Linear(d_model, d_model, bias=False)
            nn.init.normal_(self.w_read_sem.weight, std=0.01)

    # ------------------------------------------------------------------
    # Proyecciones. Devuelven el dtype del modelo; la NORMALIZACION L2 y los
    # productos punto se hacen en FP32 dentro de _l2 y los scorers. La traza
    # almacena estas claves en FP32 (store_dtype) para invarianza multi-dtype.
    # ------------------------------------------------------------------
    def keys_lex(self, t0):
        return self.p_k_lex(t0)

    def queries_lex(self, t0):
        return self.p_q_lex(t0)

    def keys_sense(self, t_shallow):
        return self.p_k_sense(t_shallow)

    def queries_ctx(self, state):
        from engrama.v6.primitives import _rmsnorm
        return self.p_q_ctx(_rmsnorm(state))

    def keys_sem(self, x):
        from engrama.v6.primitives import _rmsnorm
        return self.p_k_sem(_rmsnorm(x))

    def queries_sem(self, x):
        from engrama.v6.primitives import _rmsnorm
        return self.p_q_sem(_rmsnorm(x))

    # ------------------------------------------------------------------
    def _value_matrix(self, t0):
        if self.value == "self":
            return t0
        return F.pad(t0, (0, 0, 0, 1))[..., 1:, :]

    def inject(self, state, reads, gate):
        if reads is None:
            return state
        g = 2.0 * torch.sigmoid(gate)
        return state + g * self.w_read(reads.to(state.dtype))

    def inject_semantic(self, state, reads, gate):
        if reads is None or not self.semantic_enabled:
            return state
        g = 2.0 * torch.sigmoid(gate)
        return state + g * self.w_read_sem(reads.to(state.dtype))

    @staticmethod
    def _argmax_recency(s_m):
        n = s_m.size(-1)
        return (n - 1) - s_m.flip(-1).argmax(dim=-1)

    # ------------------------------------------------------------------
    # INDICE INVERTIDO lexico
    # ------------------------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def build_lex_inverted_index(tokens: torch.Tensor, vocab_size: int):
        """Devuelve CSR por token: (offsets, members_positions).

        Los miembros estan ordenados por posicion (argsort estable por token).
        ``vocab_size`` es el tamano del vocabulario (no de la secuencia).
        """
        n = tokens.numel()
        counts = torch.bincount(tokens, minlength=vocab_size)
        offsets = torch.zeros(vocab_size + 1, dtype=torch.long, device=tokens.device)
        torch.cumsum(counts, dim=0, out=offsets[1:])
        members = torch.empty(n, dtype=torch.long, device=tokens.device)
        order = torch.argsort(tokens, stable=True)
        members[order] = torch.arange(n, device=tokens.device)
        return offsets, members

    # ------------------------------------------------------------------
    # Lexico paralelo por INDICE INVERTIDO (exacto, O(N) promedio)
    # ------------------------------------------------------------------
    def forward_parallel_inverted(
        self,
        q_lex: torch.Tensor,
        k_lex: torch.Tensor,
        q_ctx: torch.Tensor,
        k_sen: torch.Tensor,
        t0: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Lectura lexico-sentido paralela, exacta, sin matriz N x N.

        Para cada posicion i:
          - tomo SOLO las posiciones con token == tokens[i] (indice invertido)
          - anado un rescate de las ultimas W posiciones (cubre cualquier caso)
          - puntuo esas y elijo por el criterio lexico-dominante.
        Coste O(sum_i f_i * d_k) = O(N*d_k) en promedio.
        """
        b, n, _ = q_lex.shape
        device = q_lex.device
        out = torch.zeros_like(t0)
        v = self._value_matrix(t0)
        beta = float(self.beta.detach())
        W = min(self.lexical_rescue_window, n)

        for bi in range(b):
            ql = _l2(q_lex[bi].float())
            kl = _l2(k_lex[bi].float())
            qc = _l2(q_ctx[bi].float())
            ks = _l2(k_sen[bi].float())
            toks = tokens[bi].long()
            V = max(int(toks.max().item()) + 1, self.d_recall)
            offsets, members = self.build_lex_inverted_index(toks, V)
            v_b = v[bi]

            # Procesamos por filas en bloques para controlar memoria.
            for start in range(0, n, self.score_chunk):
                end = min(n, start + self.score_chunk)
                m = end - start
                row_pos = torch.arange(start, end, device=device)
                # Candidatos base: posiciones con el mismo token.
                starts_i = offsets[toks[start:end]]
                ends_i = offsets[toks[start:end] + 1]
                # Candidatos de rescate (ultimas W pos previos a i).
                rescue = (row_pos.unsqueeze(1)
                          - torch.arange(1, W + 1, device=device).unsqueeze(0))
                rescue = rescue.clamp(min=-1)
                # Unimos en una lista por fila (con padding).
                # Para evitar un bucle Python por fila, construimos un CSR.
                # Primero candidatos invertidos:
                inv_lists = []
                max_inv = 0
                for r in range(m):
                    s, e = int(starts_i[r]), int(ends_i[r])
                    inv_lists.append(members[s:e])
                    max_inv = max(max_inv, e - s)
                # tensor (m, max_inv + W)
                C = max_inv + W
                cand = torch.full((m, C), -1, dtype=torch.long, device=device)
                for r in range(m):
                    lst = inv_lists[r]
                    cand[r, :lst.numel()] = lst
                    cand[r, max_inv:max_inv + W] = rescue[r]
                # causal
                ar = row_pos.unsqueeze(1)
                valid = (cand >= 0) & (cand <= ar - self.gap)
                # Solo candidatos del MISMO token: el indice invertido ya los
                # dio; el rescate se excluye aqui porque K_lex de otro token no
                # alcanza el umbral de induccion. Asi se replica exactamente el
                # camino incremental (mismas posiciones -> mismo resultado).
                same = (toks[row_pos].unsqueeze(1) == toks[cand.clamp(min=0)])
                valid = valid & same
                # eliminar duplicados por fila (no estrictamente necesario, pero
                # reduce computo): conservar la primera aparicion.
                cand = torch.where(valid, cand, torch.full_like(cand, -1))
                # scores
                ckl = kl[cand.clamp(min=0)]           # (m,C,dk)
                cks = ks[cand.clamp(min=0)]
                slex = (ql[row_pos - start].unsqueeze(1) * ckl).sum(-1)
                ssen = (qc[row_pos - start].unsqueeze(1) * cks).sum(-1)
                s = slex * (1.0 + beta * ssen)
                row_ok = valid.any(dim=-1)
                s_m = torch.where(valid, s, torch.full_like(s, _NEG))
                # Umbral de induccion: si el mejor slex < umbral, no hubo token
                # previo identico (no deberia pasar con el indice invertido, pero
                # protege el caso de primera ocurrencia) -> lectura cero.
                best_lex = slex.masked_fill(~valid, -1e30).max(dim=-1).values
                induced = best_lex >= _LEX_INDUCTION_THRESHOLD
                row_ok = row_ok & induced
                j_star = self._lex_dom_argmax(slex, ssen, valid, positions=cand)
                # j_star es el INDICE LOCAL en la fila de candidatos; la posicion
                # global es cand[j_star]. El value ya esta desplazado (v_b), asi
                # que leemos v_b[global_pos] (= T0[global_pos+1]).
                global_pos = cand.gather(1, j_star.view(m, 1)).squeeze(1)
                global_pos = global_pos.clamp(0, n - 1)
                hard = v_b[global_pos]
                hard = hard * row_ok.unsqueeze(-1).to(hard.dtype)
                if self.training and torch.is_grad_enabled():
                    soft_w = F.softmax(s_m / self.temperature, dim=-1)
                    soft_w = torch.nan_to_num(soft_w, nan=0.0)
                    cv = v_b[cand.clamp(min=0)]            # (m,C,d)
                    soft = torch.einsum("mc,mcd->md", soft_w, cv)
                    reads = hard + (soft - soft.detach())
                else:
                    reads = hard
                out[bi, start:end] = reads.to(out.dtype)
        return out

    def _lex_dom_argmax(self, slex, ssen, valid, positions=None):
        return _lex_dominant_argmax(slex, ssen, valid, positions=positions)

    # ------------------------------------------------------------------
    # Lexico denso (solo validacion/benchmarks)
    # ------------------------------------------------------------------
    def forward_parallel_dense(
        self, q_lex, k_lex, q_ctx, k_sen, t0,
        *, token_ids=None, identity_prev=None, threshold=0.0, score_rows=None,
    ):
        b, n, _ = k_lex.shape
        device = k_lex.device
        v = self._value_matrix(t0)
        beta = float(self.beta.detach())
        out = torch.zeros_like(t0)
        ql = _l2(q_lex.float()); kl = _l2(k_lex.float())
        qc = _l2(q_ctx.float()); ks = _l2(k_sen.float())
        rows = (torch.arange(n, device=device) if score_rows is None
                else score_rows.to(device))
        for start in range(0, rows.numel(), self.score_chunk):
            idx = rows[start:start + self.score_chunk]
            c = idx.numel()
            pos = idx.view(1, c, 1)
            colj = torch.arange(n, device=device).view(1, 1, n)
            valid = colj <= (pos - self.gap)
            for bi in range(b):
                slex = ql[bi, idx] @ kl[bi].T
                ssen = qc[bi, idx] @ ks[bi].T
                s = slex * (1.0 + beta * ssen)
                row_ok = valid.any(dim=-1)
                j = _lex_dominant_argmax(slex, ssen, valid[bi])
                gather = j.clamp(0, n - 1)
                hard = v[bi, gather] * row_ok.unsqueeze(-1).to(v.dtype)
                if self.training and torch.is_grad_enabled():
                    s_m = torch.where(valid[bi], s, torch.full_like(s, _NEG))
                    soft_w = F.softmax(s_m / self.temperature, dim=-1)
                    soft_w = torch.nan_to_num(soft_w, nan=0.0)
                    soft = soft_w @ v[bi]
                    reads = hard + (soft - soft.detach())
                else:
                    reads = hard
                out[bi, idx] = reads.to(out.dtype)
        return out

    # ------------------------------------------------------------------
    # Lexico paralelo via LSH (el LSH lexico de V6, alta recuperacion)
    # ------------------------------------------------------------------
    def forward_parallel_lsh(self, q_lex, k_lex, q_ctx, k_sen, t0, tokens,
                             *, n_tables=16, n_bits=16, cap=0, n_neg=0,
                             hamming=0, identity_prev=None,
                             rescue_window=256):
        b, n, _ = k_lex.shape
        device = k_lex.device
        out = torch.zeros_like(t0)
        beta = float(self.beta.detach())
        for bi in range(b):
            # El LSH lexico se construye sobre K_lex; como K_lex es funcion solo
            # del token, el indice de tokens es identidad-exacto. Construimos el
            # V6LSHIndex con rescue_window grande para maxima recall.
            idx = V6LSHIndex.build(
                k_lex[bi].float(), tokens[bi].long(), gap=self.gap,
                n_tables=n_tables, n_bits=n_bits, rescue_window=rescue_window, bucket_cap=cap)
            # codigos de consulta con los MISMOS planos
            planes = shared_planes(n_tables, k_lex.size(-1), n_bits, device,
                                   k_lex.dtype)
            qcodes = sign_codes(q_lex[bi].float(), planes)
            cand, valid = idx.candidates(qcodes)
            if n_neg > 0:
                gen = torch.Generator(device="cpu").manual_seed(7 + 1234)
                lim = torch.arange(n, device=device) - self.gap
                r = torch.rand(n, n_neg, generator=gen).to(device)
                neg = (r * (lim.clamp(min=0) + 1).unsqueeze(1)).long()
                ok = (lim.unsqueeze(1) >= 0).expand(n, n_neg)
                cand = torch.cat([cand, neg], dim=1)
                valid = torch.cat([valid, ok], dim=1)
            self._score_and_read(
                q_lex[bi], k_lex[bi], q_ctx[bi], k_sen[bi], t0[bi],
                cand, valid, beta, out[bi], n)
        return out

    def _score_and_read(self, ql_b, kl_b, qc_b, ks_b, t0_b, cand, valid, beta,
                        out_b, n):
        v = self._value_matrix(t0_b.unsqueeze(0)).squeeze(0)
        ql = _l2(ql_b.float()); kl = _l2(kl_b.float())
        qc = _l2(qc_b.float()); ks = _l2(ks_b.float())
        c = cand.size(1)
        # chunked para no materializar (N, C, d) completo
        for start in range(0, n, self.score_chunk):
            end = min(n, start + self.score_chunk)
            m = end - start
            local_cand = cand[start:end]
            local_valid = valid[start:end]
            ck = kl[local_cand.clamp(min=0)].view(m, c, -1)
            cs = ks[local_cand.clamp(min=0)].view(m, c, -1)
            slex = (ql[start:end].unsqueeze(1) * ck).sum(-1)
            ssen = (qc[start:end].unsqueeze(1) * cs).sum(-1)
            s = slex * (1.0 + beta * ssen)
            row_ok = local_valid.any(dim=-1)
            s_m = torch.where(local_valid, s, torch.full_like(s, _NEG))
            j = _lex_dominant_argmax(slex, ssen, local_valid, positions=local_cand)
            jj = local_cand.gather(1, j.view(m, 1)).squeeze(1)
            jj = torch.where(row_ok, jj, torch.full_like(jj, -1))
            take = (jj + 1).clamp(max=n - 1) if self.value == "next" else jj.clamp(min=0)
            hard = v[take.clamp(min=0)] * (jj >= 0).unsqueeze(-1).to(v.dtype)
            if self.training and torch.is_grad_enabled():
                soft_w = F.softmax(s_m / self.temperature, dim=-1)
                soft_w = torch.nan_to_num(soft_w, nan=0.0)
                cv = v[local_cand.clamp(min=0)].view(m, c, -1)
                soft = torch.einsum("mc,mcd->md", soft_w, cv)
                reads = hard + (soft - soft.detach())
            else:
                reads = hard
            out_b[start:end] = reads.to(out_b.dtype)

    # ------------------------------------------------------------------
    # SEMANTICO: LSH alta recuperacion + rerank exacto sobre candidatos
    # ------------------------------------------------------------------
    def forward_semantic_lsh(self, q_sem, k_sem, t0, tokens, *,
                             n_tables=16, n_bits=16, cap=0,
                             rescue_window=256, seed=7,
                             identity_prev=None):
        b, n, _ = k_sem.shape
        device = k_sem.device
        out = torch.zeros_like(t0)
        bucket_cap = self.lsh_cap if self.lsh_cap and self.lsh_cap > 0 else 4
        planes = shared_planes(n_tables, k_sem.size(-1), n_bits, device,
                               k_sem.dtype, seed=seed)
        for bi in range(b):
            idx = V6LSHIndex.build(
                k_sem[bi].float(), tokens[bi].long(), gap=self.gap,
                n_tables=n_tables, n_bits=n_bits,
                rescue_window=rescue_window, seed=seed,
                bucket_cap=0)
            qcodes = sign_codes(q_sem[bi].float(), planes)
            cand, valid = idx.candidates(qcodes)
            if identity_prev is not None:
                ip = identity_prev[bi].to(cand.device).clamp(min=-1)
                cand[:, 0] = ip
                valid[:, 0] = ip >= 0
            self._sem_score_and_read(q_sem[bi], k_sem[bi], t0[bi], cand, valid,
                                     out[bi], n)
        return out

    # Tolerancia de redondeo para considerar dos cosenos como EMPATE. El camino
    # paralelo y el incremental calculan las claves a traves de la consolidacion
    # (oneDNN), que puede diferir en ~5e-7 por orden de reduccion; sin esta
    # tolerancia, empates coseno casi exactos volcarian el argmax de forma
    # distinta entre los dos caminos, rompiendo la invarianza causal.
    _COS_TIE_TOL = 1e-5

    @staticmethod
    def _argmax_cosine_then_recency(s_m, tol: float = 1e-5, positions=None):
        """Argmax por coseno; ante scores dentro de ``tol`` elige el mas reciente.

        Devuelve el INDICE DE COLUMNA del candidato ganador (para usarse con
        ``gather``). ``positions`` (misma forma que ``s_m``) son las posiciones
        GLOBALES de cada candidato: el desempate por recencia se decide por la
        posicion global mas alta, que es lo correcto cuando el orden de
        columnas de un LSH multi-tabla no es necesariamente creciente.
        """
        n = s_m.size(-1)
        valid = s_m.gt(_NEG / 2)
        row_max = s_m.masked_fill(~valid, _NEG).max(dim=-1, keepdim=True).values
        is_max = valid & ((row_max - s_m) <= tol)
        if positions is None:
            positions = torch.arange(n, device=s_m.device).view(
                *([1] * (s_m.dim() - 1)), n).expand_as(s_m)
        # entre los maximos, gana el de mayor posicion global; si ninguno es
        # valido, devolvemos la columna 0 (el llamante enmascara con row_ok).
        # Entre los candidatos que alcanzan el maximo (dentro de tol), elegimos
        # el de MAYOR posicion global (mas reciente); como los candidatos estan
        # deduplicados, esa posicion identifica una unica columna.
        masked_pos = torch.where(is_max, positions,
                                 torch.full_like(positions, -1))
        # argmax de la posicion global enmascarada da directamente el INDICE DE
        # COLUMNA del candidato mas reciente entre los ganadores.
        return masked_pos.argmax(dim=-1)

    @staticmethod
    def _dedup_candidates(cand, valid):
        """Elimina posiciones duplicadas por fila (primera aparicion).

        Las distintas tablas LSH pueden devolver la misma posicion; sin esta
        deduplicacion el candidato aparecia varias veces, alterando el STE
        y el argmax frente a la ruta incremental (que no duplica).
        """
        m, c = cand.shape
        # Para cada fila, marca como invalida la segunda aparicion en adelante.
        # Orden de columnas: identidad, rescate, tablas -> la primera aparicion
        # se conserva. Usamos scatter sobre una mascara de vistos por valor.
        sorted_cand, sort_idx = cand.sort(dim=1, stable=True)
        dup = torch.zeros_like(sorted_cand, dtype=torch.bool)
        dup[:, 1:] = sorted_cand[:, 1:] == sorted_cand[:, :-1]
        # deshacer el orden
        is_dup = torch.zeros_like(dup)
        is_dup.scatter_(1, sort_idx, dup)
        # -1 tambien es "duplicado" (todas las celdas invalidas); no las tocamos
        is_dup = is_dup & (cand >= 0)
        new_valid = valid & ~is_dup
        # Compactamos hacia la izquierda eliminando los huecos.
        keep_count = new_valid.sum(dim=1)
        new_c = int(keep_count.max().item()) if keep_count.numel() else 0
        out = torch.full((m, new_c), -1, dtype=cand.dtype, device=cand.device)
        out_v = torch.zeros((m, new_c), dtype=torch.bool, device=cand.device)
        # posicion de escritura por fila
        wpos = torch.zeros(m, dtype=torch.long, device=cand.device)
        for col in range(c):
            write = new_valid[:, col]
            if bool(write.any()):
                rows = write.nonzero(as_tuple=True)[0]
                out[rows, wpos[rows]] = cand[rows, col]
                out_v[rows, wpos[rows]] = True
                wpos[rows] += 1
        return out, out_v

    def _sem_score_and_read(self, q_b, k_b, t0_b, cand, valid, out_b, n):
        v = self._value_matrix(t0_b.unsqueeze(0)).squeeze(0)
        qs = _l2(q_b.float()); ks = _l2(k_b.float())
        for start in range(0, n, self.score_chunk):
            end = min(n, start + self.score_chunk)
            m = end - start
            lc_raw = cand[start:end]
            lv_raw = valid[start:end]
            # Dedup por fila: una misma posicion puede llegar por varias
            # tablas LSH; un candidato duplicado debe puntuar una sola vez
            # (si no, cambia el softmax/STE y el argmax frente a la ruta
            # incremental que no duplica). Mantenemos la primera aparicion.
            lc, lv = self._dedup_candidates(lc_raw, lv_raw)
            c = lc.size(1)
            if c == 0:
                out_b[start:end] = 0
                continue
            ck = ks[lc.clamp(min=0)].view(m, c, -1)
            s = (qs[start:end].unsqueeze(1) * ck).sum(-1)
            row_ok = lv.any(dim=-1)
            s_m = torch.where(lv, s, torch.full_like(s, _NEG))
            j = self._argmax_cosine_then_recency(s_m, positions=lc)
            take = lc.gather(1, j.view(m, 1)).squeeze(1)
            take = torch.where(row_ok, take, torch.full_like(take, -1))
            # v ya esta desplazado: v[j] = T0[j+1] (valor "next"). Por tanto
            # NO hay que sumar 1 otra vez (lo hacia y leia T0[j+2], rompiendo
            # la invarianza frente al incremental que hace t0_ring[j+1]).
            hard = v[take.clamp(min=0) if self.value == "next"
                     else take.clamp(min=0)]
            hard = hard * row_ok.unsqueeze(-1).to(hard.dtype)
            if self.training and torch.is_grad_enabled():
                soft_w = F.softmax(s_m / self.sem_temperature, dim=-1)
                soft_w = torch.nan_to_num(soft_w, nan=0.0)
                cv = v[lc.clamp(min=0)].view(m, c, -1)
                soft = torch.einsum("mc,mcd->md", soft_w, cv)
                reads = hard + (soft - soft.detach())
            else:
                reads = hard
            out_b[start:end] = reads.to(out_b.dtype)

    def forward_semantic_exact(self, q_sem, k_sem, t0, *, score_rows=None):
        """Lectura semantica EXACTA que replica bit a bit el camino incremental.

        Para cada posicion de consulta llama a la misma rutina de scoring
        (producto punto ``ksem @ q_i`` con la misma reduccion) y al mismo
        ``_argmax_cosine_then_recency`` que ``read_semantic_step``. De este
        modo el camino paralelo es numericamente identico al incremental
        incluso ante empates numericos.

        Coste O(N * W * d) (W = ventana causal). No se usa ningun matmul
        N x N cuyo orden de reduccion pudiera cambiar el argmax.
        """
        b, n, _ = k_sem.shape
        device = k_sem.device
        out = torch.zeros_like(t0)
        v = t0
        qs_n = _l2(q_sem.float()); ks_n = _l2(k_sem.float())
        rows = (torch.arange(n, device=device) if score_rows is None
                else score_rows.to(device))
        for bi in range(b):
            # Recorremos fila a fila replicando EXACTAMENTE las formas 3D y
            # la reduccion einsum del camino incremental (read_semantic_step)
            # para garantizar identidad numerica incluso ante empates.
            for i in rows.tolist():
                limit = i - self.gap
                if limit < 0:
                    continue
                # (W,1,d) y (1,d): igual que ksem_ring / q_sem_t en generacion
                kf = ks_n[bi, :limit + 1].unsqueeze(1)   # (W,1,d)
                qf = qs_n[bi, i:i + 1]                   # (1,d)
                s = torch.einsum("jbd,cd->cj", kf, qf)   # (1,W)
                j = int(self._argmax_cosine_then_recency(s).item())
                take = j
                if self.value == "next":
                    take = min(j + 1, n - 1)
                out[bi, i] = v[bi, take].to(out.dtype)
        return out

    def forward_semantic_dense(self, q_sem, k_sem, t0, *, score_rows=None):
        b, n, _ = k_sem.shape
        device = k_sem.device
        v = self._value_matrix(t0)
        out = torch.zeros_like(t0)
        qs_n = _l2(q_sem.float()); ks_n = _l2(k_sem.float())
        rows = (torch.arange(n, device=device) if score_rows is None
                else score_rows.to(device))
        for start in range(0, rows.numel(), self.score_chunk):
            idx = rows[start:start + self.score_chunk]
            c = idx.numel()
            pos = idx.view(1, c, 1)
            colj = torch.arange(n, device=device).view(1, 1, n)
            valid = (colj <= (pos - self.gap))  # (1,c,n)
            for bi in range(b):
                ssem = qs_n[bi, idx] @ ks_n[bi].T   # (c,n)
                vb = valid[bi] if valid.size(0) == b else valid[0]
                row_ok = vb.any(dim=-1)
                s_m = torch.where(vb, ssem, torch.full_like(ssem, _NEG))
                j = self._argmax_cosine_then_recency(s_m)
                gather = j.clamp(0, n - 1)
                hard = v[bi, gather] * row_ok.unsqueeze(-1).to(v.dtype)
                if self.training and torch.is_grad_enabled():
                    soft_w = F.softmax(s_m / self.sem_temperature, dim=-1)
                    soft_w = torch.nan_to_num(soft_w, nan=0.0) * row_ok.unsqueeze(-1).to(soft_w.dtype if False else soft_w.dtype)
                    soft = soft_w @ v[bi]
                    reads = hard + (soft - soft.detach())
                else:
                    reads = hard
                out[bi, idx] = reads.to(out.dtype)
        return out

    # ------------------------------------------------------------------
    # Incremental (generacion) — SIEMPRE exacto, O(N*d), con indice invertido
    # ------------------------------------------------------------------
    def read_step(self, q_lex_t, klex_ring, q_ctx_t, ksen_ring, t0_ring,
                  length, last_occurrence=None, threshold=0.0,
                  use_fast_path=True):
        """Lectura incremental. Si ``last_occurrence`` viene dado (>=0) y la
        clave coincide, esa es la candidata de induccion (O(1)); en caso
        contrario se devuelve CERO (no hay induccion) para mantener la
        invarianza exacta con el camino paralelo por indice invertido.
        """
        limit = length - 1 - self.gap
        b = q_lex_t.size(0)
        if limit < 0:
            return q_lex_t.new_zeros(b, t0_ring.size(-1))
        beta = float(self.beta.detach())
        kf_lex = _l2(klex_ring[:limit + 1].float())
        kf_sen = _l2(ksen_ring[:limit + 1].float())
        qf_lex = _l2(q_lex_t.float())
        qf_ctx = _l2(q_ctx_t.float())
        if kf_lex.dim() == 3:
            slex = torch.einsum("jbd,bd->bj", kf_lex, qf_lex)
            ssen = torch.einsum("jbd,bd->bj", kf_sen, qf_ctx)
        else:
            slex = (kf_lex @ qf_lex.unsqueeze(-1)).squeeze(-1)
            ssen = (kf_sen @ qf_ctx.unsqueeze(-1)).squeeze(-1)
        colj = torch.arange(slex.size(-1), device=slex.device)
        # Restringir SOLO a candidatos del MISMO token (indice invertido): K_lex
        # es identico para el mismo token => slex ~= 1.0. Cualquier otro token
        # tiene slex muy por debajo de _LEX_INDUCTION_THRESHOLD. Eliminamos toda
        # otra posicion para que el camino incremental sea exacto al paralelo.
        valid = (colj.unsqueeze(0) <= limit)
        lex_max = slex.masked_fill(~valid, -1e30).max(dim=-1, keepdim=True).values
        same_token = (slex >= lex_max - 1e-3) & valid & (lex_max >= _LEX_INDUCTION_THRESHOLD)
        # Si no hay ningun candidato del mismo token, lectura cero.
        has = same_token.any(dim=-1)
        j_star = _lex_dominant_argmax(slex, ssen, same_token)
        j_star = j_star.clamp(0, limit)
        idx = (j_star if self.value == "self"
               else (j_star + 1).clamp(max=length - 1))
        if t0_ring.dim() == 3:
            rows = torch.arange(b, device=t0_ring.device)
            src = t0_ring[idx, rows]
        else:
            src = t0_ring[idx]
        src = src * has.unsqueeze(-1).to(src.dtype)
        return src.to(q_lex_t.dtype)

    def read_semantic_step_lsh(self, q_sem_t, cache, k_sem_t, token_ids_b, length,
                               identity_prev=None):
        """Lectura semantica incremental O(C) por token usando el indice LSH.

        El cache mantiene un :class:`IncrementalLSHIndex` por elemento del
        batch con las ``cap`` posiciones mas recientes por bucket mas rescate
        de ventana. ``identity_prev`` (b,) es la ultima ocurrencia previa del
        mismo token (ya excluye la posicion actual); se usa como candidato de
        identidad. Calculamos el coseno solo contra C candidatos acotados.
        """
        b = q_sem_t.size(0)
        dq = q_sem_t.size(-1)
        device = q_sem_t.device
        t0_ring = cache.linear_t0()
        d_model = t0_ring.size(-1)
        out = q_sem_t.new_zeros(b, d_model)
        limit = length - 1 - self.gap
        # Valor "next": T0[j+1]. El camino paralelo (_sem_score_and_read)
        # construye v = pad(t0)[1:] sobre las N filas, de modo que v[j]=T0[j+1]
        # y en j=n-1 repite T0[n-1]. En incremental el anillo YA contiene a
        # T0 en orden 0..length-1, asi que t0_ring[j+1] es exactamente el
        # valor "next" (para j=limit el maximo es length-1 = T0[length-1],
        # el token actual, que es lo mismo que el clamp del paralelo).
        from engrama.v6.lsh import shared_planes, sign_codes
        planes = shared_planes(self.lsh_tables, dq, self.lsh_bits, device,
                               q_sem_t.dtype, seed=7)
        qcodes = sign_codes(q_sem_t.float(), planes)   # (b,t)
        for bi in range(b):
            if limit < 0:
                continue
            idx = cache.semantic_lsh(bi)
            if identity_prev is not None:
                ip = int(identity_prev[bi].item())
                identity_pos = ip if (0 <= ip <= limit) else -1
            else:
                token_id = int(token_ids_b[bi].item())
                identity_pos = idx.identity_for(token_id)
                if identity_pos > limit:
                    identity_pos = -1
            cand, valid = idx.candidates_for(limit, qcodes[bi],
                                             identity_pos=identity_pos)
            # Deduplicar (igual que la ruta paralela): distintas tablas pueden
            # traer la misma posicion; sin esto el desempate por recencia podia
            # volcar a una posicion duplicada con score identico.
            cand, valid = self._dedup_candidates(
                cand.unsqueeze(0), valid.unsqueeze(0))
            cand = cand[0]; valid = valid[0]
            if not bool(valid.any()):
                continue
            # rerank exacto sobre candidatos (coseno FP32). La CONSULTA es
            # q_sem y las CLAVES candidatas son k_sem (igual que la ruta
            # paralela ``_sem_score_and_read``): puntuamos q contra k.
            qf = _l2(q_sem_t[bi:bi + 1].float())
            ksem_ring = cache.linear_ksem()
            if ksem_ring.dim() == 3:
                ck = ksem_ring[cand.clamp(min=0), bi].float()
            else:
                ck = ksem_ring[cand.clamp(min=0)].float()
            ck = _l2(ck)
            s = (qf.unsqueeze(1) * ck.unsqueeze(0)).sum(-1)  # (1,C)
            s = torch.where(valid.unsqueeze(0), s,
                            torch.full_like(s, _NEG))
            j = int(self._argmax_cosine_then_recency(
                s, positions=cand.unsqueeze(0)).item())
            jpos = int(cand[j].item())
            if self.value == "next":
                # El valor "next" del candidato en posicion j es T0[j+1].
                # TANTO el camino paralelo como el incremental leen T0[j+1].
                # En paralelo v = F.pad(t0)[1:] => v[j] = t0[j+1].
                # En incremental el anillo contiene T0[0..length-1] ya
                # escrito, y j+1 <= limit+1 = length-gap <= length-1.
                jpos = jpos + 1
            if t0_ring.dim() == 3:
                out[bi] = t0_ring[jpos, bi]
            else:
                out[bi] = t0_ring[jpos]
        return out.to(q_sem_t.dtype)

    def read_semantic_value(self, q_sem_t, cache, k_sem_t, token_ids_b, length,
                            identity_prev=None):
        """Helper de diagnostico: devuelve (read, j, score, cand, valid)."""
        b = q_sem_t.size(0)
        limit = length - 1 - self.gap
        from engrama.v6.lsh import shared_planes, sign_codes
        planes = shared_planes(self.lsh_tables, q_sem_t.size(-1), self.lsh_bits,
                               q_sem_t.device, q_sem_t.dtype, seed=7)
        qcodes = sign_codes(q_sem_t.float(), planes)
        bi = 0
        idx = cache.semantic_lsh(bi)
        ip = int(identity_prev[bi].item()) if identity_prev is not None else -1
        identity_pos = ip if 0 <= ip <= limit else -1
        cand, valid = idx.candidates_for(limit, qcodes[bi], identity_pos=identity_pos)
        cand, valid = self._dedup_candidates(cand.unsqueeze(0), valid.unsqueeze(0))
        cand = cand[0]; valid = valid[0]
        if not bool(valid.any()):
            return q_sem_t.new_zeros(q_sem_t.size(-1)), -1, None, cand, valid
        qf = _l2(q_sem_t[bi:bi + 1].float())
        ksem_ring = cache.linear_ksem()
        ck = ksem_ring[cand.clamp(min=0), bi].float() if ksem_ring.dim() == 3 else ksem_ring[cand.clamp(min=0)].float()
        ck = _l2(ck)
        s = (qf.unsqueeze(1) * ck.unsqueeze(0)).sum(-1)
        s = torch.where(valid.unsqueeze(0), s, torch.full_like(s, _NEG))
        j = int(self._argmax_cosine_then_recency(s, positions=cand.unsqueeze(0)).item())
        jpos = int(cand[j].item())
        if self.value == "next":
            jpos = jpos + 1
        t0_ring = cache.linear_t0()
        return t0_ring[jpos, bi], jpos, s[0, j].item(), cand, valid

    def read_semantic_step(self, q_sem_t, ksem_ring, t0_ring, length):
        limit = length - 1 - self.gap
        b = q_sem_t.size(0)
        if limit < 0:
            return q_sem_t.new_zeros(b, t0_ring.size(-1))
        kf = _l2(ksem_ring[:limit + 1].float())
        qf = _l2(q_sem_t.float())
        if kf.dim() == 3:
            s = torch.einsum("jbd,bd->bj", kf, qf)
        else:
            s = (kf @ qf.unsqueeze(-1)).squeeze(-1)
        j = self._argmax_cosine_then_recency(s).clamp(0, limit)
        idx = (j if self.value == "self" else (j + 1).clamp(max=length - 1))
        if t0_ring.dim() == 3:
            rows = torch.arange(b, device=t0_ring.device)
            src = t0_ring[idx, rows]
        else:
            src = t0_ring[idx]
        return src.to(q_sem_t.dtype)
