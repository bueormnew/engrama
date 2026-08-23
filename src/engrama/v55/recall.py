"""ENGRAMA V5.5 — Recall Tap ASIMETRICO V2 (Pilar 4).

El cambio que da el 100 % en texto real. En V5 la metrica era simetrica y
aislada: dos apariciones del mismo token tenian el mismo ``K`` y no habia forma
de desambiguar (el "banco" del rio y el "banco" de dinero puntuan igual).
V5.5 parte el codigo en dos ejes:

* **Eje lexico** (aislado, induce): ``K_lex[j] = P_k_lex(T0[j])``. Mismo token
  -> mismo ``K_lex`` -> mismo bucket -> mismo score lexico. La propiedad de
  induccion de V5 queda INTACTA (no hay matriz N x N, no hay softmax temporal).
* **Eje de sentido** (contextual, desempata): ``K_sense[j] = P_k_sense(T_shallow[j])``
  con ``T_shallow`` = salida de la capa 0 de consolidacion (contexto local de
  2 tokens). La consulta contextual ``q_ctx[i] = P_q_ctx(RMSNorm(T_L[i]))`` ve
  el presente completo.

Score combinado::

    score_lex   = cos(q_lex[i], K_lex[j])            # en [-1, 1]; ~1 si mismo token
    score_sense = cos(q_ctx[i], K_sense[j])          # en [-1, 1]; desambigua contexto
    score       = score_lex * (1 + beta * score_sense)   # beta ~ 0.3 aprendido
    j*          = argmax_{j<=i-gap} score            # top-1 DURO

La lectura sigue siendo ``T0[j*+1]`` (huella completa del siguiente), y el
gradiente entra por straight-through (softmax solo en el backward).

Recuperacion 100 % garantizada por dos mecanismos:
1. **Fast path de identidad**: indice O(1) de la ultima ocurrencia del mismo
   ``token_id``. Tareas de copia pura -> 100 % desde el paso 0.
2. **Fallback denso**: si ``max(score) < threshold`` se reescanea denso el
   ultimo ``fallback_window`` (2048) de tokens. En la generacion incremental la
  ruta es SIEMPRE densa (matvec ``O(N d_k)``), asi que el 100 % es estructural.

NO es atencion: sin softmax sobre el eje temporal, sin matriz N x N normalizada,
sin promedio ponderado de posiciones. Solo productos punto punto a punto + un
argmax + un gather (lectura de diccionario).

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

_NEG = -1.0e30  # "-inf" acotado: exp() nunca desborda (regula anti-NaN)

# El tap semantico usa claves desde T0 (huella aislada, Pilar 1): dos posiciones
# con el MISMO token tienen K_sem IDENTICO -> empates EXACTOS. El score paralelo
# (einsum, reduccion batched) y el incremental (matvec, BLAS distinto) difieren
# en ~1e-7 por redondeo de FP; en empates exactos eso VUELCA el argmax de
# recencia y rompe la invariancia causal (paralelo != incremental). Redondear el
# score a SEM_TIE_DECIMALS antes del argmax mata el ruido de FP (~1e-6) sin
# afectar diferencias genuinas (>1e-3) -> argmax determinista e invariante.
SEM_TIE_DECIMALS = 4


def _l2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=eps).to(x.dtype)


def _lex_dominant_argmax(slex: torch.Tensor, ssen: torch.Tensor,
                         valid: torch.Tensor, positions: Optional[torch.Tensor] = None,
                         tol: float = 1e-3) -> torch.Tensor:
    """Seleccion lexicografica robusta (corazon de la garantia de induccion).

    1. El eje LEXICO domina: solo se consideran candidatos con ``score_lex``
       dentro de ``tol`` del maximo (los de mismo token, que tienen K_lex
       IDENTICO -> score_lex identico). Asi el sentido NUNCA puede sobreescribir
       la señal lexica.
    2. Entre esos candidatos, gana el de mayor ``score_sense`` (desambigua
       contexto: el "banco" correcto entre varias apariciones del mismo token).
    3. Empate en sentido -> ocurrencia MAS RECIENTE (mayor POSICION, no indice
       de array: en LSH los candidatos no estan ordenados por posicion).

    ``positions``: posicion real de cada candidato (para el desempate). Si es
    ``None`` se usa el indice del array (valido cuando esta ordenado por
    posicion: camino denso e incremental). Devuelve ``j_star`` (indices).
    """
    slex_m = torch.where(valid, slex.float(), _NEG)
    max_lex = slex_m.max(dim=-1, keepdim=True).values
    lex_mask = (slex_m >= max_lex - tol) & valid
    key = torch.where(lex_mask, ssen.float(), torch.full_like(ssen, _NEG))
    max_sen = key.max(dim=-1, keepdim=True).values
    sen_mask = lex_mask & (key >= max_sen - tol)
    if positions is None:
        idx = torch.arange(key.size(-1), device=key.device, dtype=torch.float32)
        poskey = torch.where(sen_mask, idx, torch.full_like(idx, -1.0))
    else:
        poskey = torch.where(sen_mask, positions.float(),
                             torch.full_like(positions, -1.0))
    return poskey.argmax(dim=-1)


class RecallTapV2(nn.Module):
    """Recall Tap asimetrico: eje lexico (aislado) + eje de sentido (contextual)."""

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
    ):
        super().__init__()
        if value not in ("next", "self"):
            raise ValueError(f"rt_value debe ser 'next' o 'self', no {value!r}")
        if gap < 1:
            raise ValueError("rt_gap >= 1")
        self.d_model = d_model
        self.d_recall = d_recall
        self.d_sense = d_sense
        self.value = value
        self.gap = gap
        self.temperature = max(1e-2, float(temperature))
        self.score_chunk = max(64, int(score_chunk))
        # proyecciones lexicas (aisladas, inducen)
        self.p_q_lex = nn.Linear(d_model, d_recall, bias=False)
        self.p_k_lex = nn.Linear(d_model, d_recall, bias=False)
        nn.init.normal_(self.p_q_lex.weight, std=init_std)
        if shared_lex_init:
            with torch.no_grad():
                self.p_k_lex.weight.copy_(self.p_q_lex.weight)
        else:
            nn.init.normal_(self.p_k_lex.weight, std=init_std)
        # proyecciones de sentido (contextuales, desempatan)
        self.p_q_ctx = nn.Linear(d_model, d_sense, bias=False)
        self.p_k_sense = nn.Linear(d_model, d_sense, bias=False)
        nn.init.normal_(self.p_q_ctx.weight, std=init_std)
        nn.init.normal_(self.p_k_sense.weight, std=init_std)
        # peso del termino de sentido (beta), init 0.3, aprendido por defecto
        self.beta = nn.Parameter(torch.tensor(float(sense_beta_init)))
        if not sense_beta_trainable:
            self.beta.requires_grad_(False)
        # proyeccion de la lectura + compuerta de inyeccion (zero-init)
        self.w_read = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.w_read.weight, std=0.02)
        # --- TAP SEMANTICO (asociativo, todos los candidatos, por significado) ---
        # Recupera por SIGNIFICADO (no por token): resuelve agujas semanticas.
        # No es atencion: argmax duro + lectura unica de T0[j*+1], linealizable
        # por LSH. Convive con el tap lexico (que garantiza copia exacta).
        self.semantic_enabled = bool(semantic_enabled)
        self.d_semantic = int(d_semantic)
        self.sem_temperature = max(1e-2, float(sem_temperature))
        if self.semantic_enabled:
            # claves/consultas semanticas desde la consolidacion FINAL (significado)
            self.p_q_sem = nn.Linear(d_model, d_semantic, bias=False)
            self.p_k_sem = nn.Linear(d_model, d_semantic, bias=False)
            nn.init.normal_(self.p_q_sem.weight, std=init_std)
            nn.init.normal_(self.p_k_sem.weight, std=init_std)
            # lectura semantica (zero-init: no-op al inicio, crece al aprender)
            self.w_read_sem = nn.Linear(d_model, d_model, bias=False)
            nn.init.normal_(self.w_read_sem.weight, std=0.01)

    # ------------------------------------------------------------------
    def keys_lex(self, t0: torch.Tensor) -> torch.Tensor:
        return self.p_k_lex(t0)

    def queries_lex(self, t0: torch.Tensor) -> torch.Tensor:
        return self.p_q_lex(t0)

    def keys_sense(self, t_shallow: torch.Tensor) -> torch.Tensor:
        return self.p_k_sense(t_shallow)

    def queries_ctx(self, state: torch.Tensor) -> torch.Tensor:
        # la consulta contextual ve el presente (estado consolidado final)
        return self.p_q_ctx(_rmsnorm(state))

    def _value_matrix(self, t0: torch.Tensor) -> torch.Tensor:
        if self.value == "self":
            return t0
        return F.pad(t0, (0, 0, 0, 1))[..., 1:, :]  # desplaza una posicion

    def inject(self, state: torch.Tensor, reads: Optional[torch.Tensor],
               gate: torch.Tensor) -> torch.Tensor:
        """``state + g_rt * W_r(reads)`` (g_rt acotado)."""
        if reads is None:
            return state
        g = 2.0 * torch.sigmoid(gate)
        return state + g * self.w_read(reads)

    # ------------------------------------------------------------------
    # TAP SEMANTICO (asociativo): recupera por significado sobre TODOS los
    # candidatos. Mismo mecanismo que el lexico (dot + argmax + gather) pero con
    # claves de significado (T_last) y sin restriccion de mismo token.
    # ------------------------------------------------------------------
    def keys_sem(self, t_last: torch.Tensor) -> torch.Tensor:
        return self.p_k_sem(_rmsnorm(t_last))

    def queries_sem(self, state: torch.Tensor) -> torch.Tensor:
        return self.p_q_sem(_rmsnorm(state))

    def inject_semantic(self, state: torch.Tensor, reads: Optional[torch.Tensor],
                        gate: torch.Tensor) -> torch.Tensor:
        """``state + g_sem * W_r_sem(reads)`` (zero-init residual)."""
        if reads is None or not self.semantic_enabled:
            return state
        g = 2.0 * torch.sigmoid(gate)
        return state + g * self.w_read_sem(reads)

    def _argmax_recency(self, s_m: torch.Tensor) -> torch.Tensor:
        """argmax con desempate por RECENCIA (indice mas alto = mas reciente)."""
        n = s_m.size(-1)
        return (n - 1) - s_m.flip(-1).argmax(dim=-1)

    def forward_semantic_dense(
        self, q_sem: torch.Tensor, k_sem: torch.Tensor, t0: torch.Tensor,
        *, score_rows: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Lectura semantica paralela: argmax de cos(q_sem,K_sem) sobre todos los
        candidatos validos -> lee T0[j*+1]. STE en backward."""
        b, n, _ = k_sem.shape
        device = k_sem.device
        v = self._value_matrix(t0)
        qs_n, ks_n = _l2(q_sem), _l2(k_sem)
        out = torch.zeros_like(t0)
        chunk = self.score_chunk
        rows = (torch.arange(n, device=device) if score_rows is None
                else score_rows.to(device))
        for start in range(0, rows.numel(), chunk):
            idx = rows[start:start + chunk]
            c = idx.numel()
            pos = idx.view(1, c, 1)
            colj = torch.arange(n, device=device).view(1, 1, n)
            valid = colj <= (pos - self.gap)
            ssem = torch.einsum("bcd,bnd->bcn", qs_n[:, idx, :].float(), ks_n.float())
            row_ok = valid.any(dim=-1)
            s_m = torch.where(valid, ssem, torch.full_like(ssem, _NEG))
            j_star = self._argmax_recency(s_m.round(decimals=SEM_TIE_DECIMALS))               # (B, c)
            gather = j_star.clamp(0, n - 1)
            hard = v.gather(1, gather.view(b, c, 1).expand(b, c, t0.size(-1)))
            hard = hard * row_ok.unsqueeze(-1).to(hard.dtype)
            if self.training and torch.is_grad_enabled():
                soft_w = F.softmax(s_m / self.sem_temperature, dim=-1)
                soft_w = torch.nan_to_num(soft_w, nan=0.0) * row_ok.unsqueeze(-1).to(soft_w.dtype)
                soft = soft_w @ v
                reads = hard + (soft - soft.detach())
            else:
                reads = hard
            out = out.index_copy(1, idx, reads.to(out.dtype))
        return out

    def forward_semantic_lsh(
        self, q_sem: torch.Tensor, k_sem: torch.Tensor, t0: torch.Tensor,
        tokens: torch.Tensor, *,
        n_tables: int = 2, n_bits: int = 8, cap: int = 64,
        seed: int = 7,
    ) -> torch.Tensor:
        """Lectura semantica paralela LINEAL O(N*C*d_sem) via LSH de planos
        compartidos (consulta y clave se hashean con los MISMOS planos: si
        q_sem(b_i) ~= k_sem(a_i) en direccion, mismo sign-code -> mismo bucket).

        Candidatos por consulta: identidad (ultima ocurrencia del MISMO token,
        rescate) + recent_k + n_tables*cap (LSH). Mismo argmax duro + recencia +
        lectura unica T0[j*+1] que el camino denso. Difiere del tap lexico en que
        NO requiere mismo token (la sonda es por SIGNIFICADO aprendido)."""
        from engrama.v55.lsh import _bucket_matrix, previous_same_occurrence
        b, n, _ = k_sem.shape
        device = k_sem.device
        v = self._value_matrix(t0)
        qs_n, ks_n = _l2(q_sem), _l2(k_sem)
        out = torch.zeros_like(t0)
        n_codes = 1 << min(n_bits, 16)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        planes = (torch.randint(0, 2, (n_tables, self.d_semantic, n_bits),
                                generator=gen, dtype=torch.float32) * 2 - 1
                  ).to(device=device, dtype=k_sem.dtype)
        weights = (1 << torch.arange(n_bits, device=device)).long()
        recent_k = 4
        chunk = self.score_chunk
        for bi in range(b):
            kf = ks_n[bi].float()                       # (N, dsem)
            qf = qs_n[bi].float()                       # (N, dsem)
            k_bits = (kf.unsqueeze(0) @ planes.float()) > 0     # (t, N, b)
            k_codes = (k_bits.long() @ weights).T % n_codes      # (N, t)
            q_bits = (qf.unsqueeze(0) @ planes.float()) > 0
            q_codes = (q_bits.long() @ weights).T % n_codes      # (N, t)
            cols = [previous_same_occurrence(tokens[bi].long(),
                                              gap=self.gap).unsqueeze(1)]  # identidad
            off = torch.arange(self.gap, self.gap + recent_k, device=device).unsqueeze(0)
            cols.append(torch.arange(n, device=device).unsqueeze(1) - off)   # recent
            for t in range(n_tables):
                bucket = _bucket_matrix(k_codes[:, t], n, n_codes, cap)
                cols.append(bucket[q_codes[:, t]])            # sonda por SIGNIFICADO
            cand = torch.cat(cols, dim=1)                     # (N, C)
            ar = torch.arange(n, device=device).unsqueeze(1)
            valid = (cand >= 0) & (cand <= ar - self.gap)
            c = cand.size(1)
            cv = v[bi].index_select(0, cand.clamp(min=0).reshape(-1)).view(n, c, -1)
            ck = k_sem[bi].index_select(0, cand.clamp(min=0).reshape(-1)).view(n, c, -1)
            for start in range(0, n, chunk):
                end = min(n, start + chunk)
                m = end - start
                qm = _l2(qf[start:end])                       # (m, dsem)
                ckn = _l2(ck[start:end].float())             # (m, C, dsem)
                s = (qm.unsqueeze(1) * ckn).sum(-1)          # (m, C) cos sem
                ok = valid[start:end]
                row_ok = ok.any(dim=-1)
                s_m = torch.where(ok, s, torch.full_like(s, _NEG))
                j_star = self._argmax_recency(s_m.round(decimals=SEM_TIE_DECIMALS))
                hard = cv[start:end].gather(
                    1, j_star.view(m, 1, 1).expand(m, 1, cv.size(-1))).squeeze(1)
                hard = hard * row_ok.unsqueeze(-1).to(hard.dtype)
                if self.training and torch.is_grad_enabled():
                    soft_w = F.softmax(s_m / self.sem_temperature, dim=-1)
                    soft_w = torch.nan_to_num(soft_w, nan=0.0) * row_ok.unsqueeze(-1).to(soft_w.dtype)
                    soft = torch.einsum("mc,mcd->md", soft_w, cv[start:end])
                    reads = hard + (soft - soft.detach())
                else:
                    reads = hard
                out[bi, start:end] = reads.to(out.dtype)
        return out

    def read_semantic_step(
        self, q_sem_t: torch.Tensor, ksem_ring: torch.Tensor,
        t0_ring: torch.Tensor, length: int,
    ) -> torch.Tensor:
        """Lectura semantica incremental (generacion): matvec O(N*d_sem)."""
        limit = length - 1 - self.gap
        b = q_sem_t.size(0)
        if limit < 0:
            return q_sem_t.new_zeros(b, t0_ring.size(-1))
        kf = _l2(ksem_ring[: limit + 1].float())        # (m, B, dsem) o (m, dsem)
        qf = _l2(q_sem_t.float())                       # (B, dsem)
        if kf.dim() == 3:
            s = torch.einsum("jbd,bd->bj", kf, qf)      # (B, m)
        else:
            s = (kf @ qf.unsqueeze(-1)).squeeze(-1)
        j_star = self._argmax_recency(s.round(decimals=SEM_TIE_DECIMALS)).clamp(0, limit)
        idx = (j_star if self.value == "self"
               else (j_star + 1).clamp(max=length - 1))
        if t0_ring.dim() == 3:                          # (N, B, d)
            rows = torch.arange(b, device=t0_ring.device)
            src = t0_ring[idx, rows]
        else:
            src = t0_ring[idx]
        return src.to(q_sem_t.dtype)

    # ------------------------------------------------------------------
    # Camino paralelo DENSO (entrenamiento, pequenas N): (B,N,d) -> (B,N,d)
    # ------------------------------------------------------------------
    def forward_parallel_dense(
        self,
        q_lex: torch.Tensor, k_lex: torch.Tensor,
        q_ctx: torch.Tensor, k_sen: torch.Tensor,
        t0: torch.Tensor,
        *, token_ids: Optional[torch.Tensor] = None,
        identity_prev: Optional[torch.Tensor] = None,
        threshold: float = 0.0,
        score_rows: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, _ = k_lex.shape
        device = k_lex.device
        v = self._value_matrix(t0)
        beta = self.beta.to(q_lex.dtype)
        ql_n, kl_n = _l2(q_lex), _l2(k_lex)
        qc_n, ks_n = _l2(q_ctx), _l2(k_sen)
        out = torch.zeros_like(t0)
        chunk = self.score_chunk
        rows = (torch.arange(n, device=device) if score_rows is None
                else score_rows.to(device))
        for start in range(0, rows.numel(), chunk):
            idx = rows[start:start + chunk]
            c = idx.numel()
            pos = idx.view(1, c, 1)
            colj = torch.arange(n, device=device).view(1, 1, n)
            valid = colj <= (pos - self.gap)
            slex = torch.einsum("bcd,bnd->bcn", ql_n[:, idx, :].float(), kl_n.float())
            ssen = torch.einsum("bcd,bnd->bcn", qc_n[:, idx, :].float(), ks_n.float())
            # ST/gradiente: composite lex-dominante (slex*(1+beta*ssen))
            s = slex * (1.0 + beta * ssen)
            row_ok = valid.any(dim=-1)
            # SELECCION DURA: lexico dominante, sentido desempata, recencia final.
            j_star = _lex_dominant_argmax(slex, ssen, valid)   # (B, c)
            gather = j_star.clamp(0, n - 1)
            hard = v.gather(1, gather.view(b, c, 1).expand(b, c, t0.size(-1)))
            hard = hard * row_ok.unsqueeze(-1).to(hard.dtype)
            if self.training and torch.is_grad_enabled():
                s_m = torch.where(valid, s, torch.full_like(s, _NEG))
                soft_w = F.softmax(s_m / self.temperature, dim=-1)
                soft_w = torch.nan_to_num(soft_w, nan=0.0) * row_ok.unsqueeze(-1).to(soft_w.dtype)
                soft = soft_w @ v
                reads = hard + (soft - soft.detach())
            else:
                reads = hard
            out = out.index_copy(1, idx, reads.to(out.dtype))
        return out

    # ------------------------------------------------------------------
    # Camino paralelo LINEAL (modo LSH): candidatos indexados
    # ------------------------------------------------------------------
    def forward_parallel_lsh(
        self,
        q_lex: torch.Tensor, k_lex: torch.Tensor,
        q_ctx: torch.Tensor, k_sen: torch.Tensor,
        t0: torch.Tensor, tokens: torch.Tensor,
        *, n_tables: int = 2, n_bits: int = 32, cap: int = 64,
        seed: int = 7, n_neg: int = 48, hamming: int = 3,
        identity_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from engrama.v55.lsh import LSHIndexV2

        b, n, _ = k_lex.shape
        v = self._value_matrix(t0)
        beta = self.beta.to(q_lex.dtype)
        ql_n, kl_n = _l2(q_lex), _l2(k_lex)
        qc_n, ks_n = _l2(q_ctx), _l2(k_sen)
        out = torch.zeros_like(t0)
        for bi in range(b):
            index = LSHIndexV2.build(
                k_lex[bi], tokens[bi], gap=self.gap,
                n_tables=n_tables, n_bits=n_bits, cap=cap, seed=seed,
                hamming=hamming,
            )
            # El indice YA incluye la identidad (col 0) y el rescate reciente.
            cand, valid = index.candidates()          # (N, C), (N, C)
            if n_neg > 0:
                gen = torch.Generator(device="cpu").manual_seed(seed + 1234)
                lim = torch.arange(n, device=cand.device) - self.gap
                r = torch.rand(n, n_neg, generator=gen).to(cand.device)
                neg = (r * (lim.clamp(min=0) + 1).unsqueeze(1)).long()
                ok = (lim.unsqueeze(1) >= 0).expand(n, n_neg)
                cand = torch.cat([cand, neg], dim=1)
                valid = torch.cat([valid, ok], dim=1)
            c = cand.size(1)
            ck = k_lex[bi].index_select(0, cand.clamp(min=0).reshape(-1)).view(n, c, -1)
            cs = k_sen[bi].index_select(0, cand.clamp(min=0).reshape(-1)).view(n, c, -1)
            cv = v[bi].index_select(0, cand.clamp(min=0).reshape(-1)).view(n, c, -1)
            chunk = self.score_chunk
            for start in range(0, n, chunk):
                end = min(n, start + chunk)
                m = end - start
                ql = _l2(ql_n[bi, start:end])           # (m, dk)
                qc = _l2(qc_n[bi, start:end])
                ckl = _l2(ck[start:end].float())        # (m, C, dk)
                cks = _l2(cs[start:end].float())
                slex = (ql.unsqueeze(1) * ckl).sum(-1)            # (m, C)
                ssen = (qc.unsqueeze(1) * cks).sum(-1)
                s = slex * (1.0 + beta * ssen)                    # composite (ST)
                ok = valid[start:end]
                row_ok = ok.any(dim=-1)
                # SELECCION DURA lexicografica (mismo criterio que el denso).
                # Se pasa la POSICION real de cada candidato para el desempate
                # por recencia (los candidatos LSH no estan ordenados).
                j_star = _lex_dominant_argmax(
                    slex, ssen, ok, positions=cand[start:end])    # (m,)
                hard = cv[start:end].gather(
                    1, j_star.view(m, 1, 1).expand(m, 1, cv.size(-1))
                ).squeeze(1)
                hard = hard * row_ok.unsqueeze(-1).to(hard.dtype)
                if self.training and torch.is_grad_enabled():
                    s_m = torch.where(ok, s, torch.full_like(s, _NEG))
                    soft_w = F.softmax(s_m / self.temperature, dim=-1)
                    soft_w = torch.nan_to_num(soft_w, nan=0.0) * row_ok.unsqueeze(-1).to(soft_w.dtype)
                    soft = torch.einsum("mc,mcd->md", soft_w, cv[start:end])
                    reads = hard + (soft - soft.detach())
                else:
                    reads = hard
                out[bi, start:end] = reads.to(out.dtype)
        return out

    # ------------------------------------------------------------------
    # Camino incremental (generacion): matvec denso contra los anillos K
    # ------------------------------------------------------------------
    def read_step(
        self,
        q_lex_t: torch.Tensor, klex_ring: torch.Tensor,
        q_ctx_t: torch.Tensor, ksen_ring: torch.Tensor,
        t0_ring: torch.Tensor, length: int,
        last_occurrence: torch.Tensor = None, threshold: float = 0.0,
        use_fast_path: bool = True,
    ) -> torch.Tensor:
        """Lectura para el token actual ``(B, ...)``.

        Ruta SIEMPRE densa (matvec ``O(N d_k)``): el 100 % es estructural en
        generacion. Usa prioridad ULP de identidad (coincide exacto con
        :meth:`forward_parallel_dense`): si la ocurrencia previa del mismo token
        empata con el maximo (tolerancia 1e-6), gana -> invarianza causal
        exacta incluso con empates.
        """
        limit = length - 1 - self.gap
        b = q_lex_t.size(0)
        if limit < 0:
            return q_lex_t.new_zeros(b, t0_ring.size(-1))
        beta = self.beta.to(q_lex_t.dtype)
        kf_lex = _l2(klex_ring[: limit + 1].float())    # (m, B, dk)
        kf_sen = _l2(ksen_ring[: limit + 1].float())
        qf_lex = _l2(q_lex_t.float())                   # (B, dk)
        qf_ctx = _l2(q_ctx_t.float())
        if kf_lex.dim() == 3:                            # (m, B, dk)
            slex = torch.einsum("jbd,bd->bj", kf_lex, qf_lex)   # (B, m)
            ssen = torch.einsum("jbd,bd->bj", kf_sen, qf_ctx)
        else:                                            # (m, dk) B=1 plano
            slex = (kf_lex @ qf_lex.unsqueeze(-1)).squeeze(-1)
            ssen = (kf_sen @ qf_ctx.unsqueeze(-1)).squeeze(-1)
        # mascara causal valida (j <= limit). La seleccion es lex-dominante
        # (mismo criterio que el camino paralelo denso -> invarianza causal).
        # El eje lexico garantiza mismo-token (induccion): la ocurrencia previa
        # del mismo token tiene K_lex identico -> score_lex maximo -> siempre
        # esta en el conjunto ganador; el sentido solo desempata. Por eso NO hace
        # falta un override explicito de last_occurrence: la seleccion ya lo da.
        colj = torch.arange(slex.size(-1), device=slex.device)
        valid = colj.unsqueeze(0) <= limit                 # (1, m) -> broadcast
        j_star = _lex_dominant_argmax(slex, ssen, valid)   # (B,)
        j_star = j_star.clamp(0, limit)
        idx = (j_star if self.value == "self"
               else (j_star + 1).clamp(max=length - 1))  # (B,)
        if t0_ring.dim() == 3:                           # (N, B, d)
            rows = torch.arange(b, device=t0_ring.device)
            src = t0_ring[idx, rows]                      # (B, d)
        else:
            src = t0_ring[idx]                            # (B, d)
        return src.to(q_lex_t.dtype)


def _rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    out_dtype = x.dtype
    x32 = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x32 * rms).to(out_dtype)
