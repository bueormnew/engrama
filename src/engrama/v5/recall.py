"""ENGRAMA V5 — Recall Tap: lectura dura direccionada desde la Traza.

``K[j] = P_k(T_0[j])``  (codigo de la huella pristina; aislado por token)
``q_i  = P_q(T_0[i])``  (consulta aislada)  o  ``P_q(estado)``
``s[i,j] = <q_i, K[j]> / sqrt(d_k)``  para ``j <= i - gap``  (causal)
``j* = argmax_j s[i,j]``   (top-1 DURO; empates -> ocurrencia mas reciente)
``r_i = T_0[j* + 1]``       (valor = huella del token siguiente al match)

El forward NUNCA hace softmax sobre el eje temporal ni promedia posiciones:
es una lectura de diccionario (argmax + gather) sobre la memoria explicita.
El gradiente entra por straight-through: ``r = r_duro + (r_suave - sg(r_suave))``
donde ``r_suave`` solo participa del backward.

Complejidad: paralela O(N^2 d_k) troceada en filas (VRAM O(chunk*N));
incremental O(N d_k) por token (un matvec contra el anillo K) — el
"KV-cache nativo" de ENGRAMA: un solo eje K, sin matrices por capa.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

_NEG = -1.0e30  # "menos infinito" acotado: exp() nunca desborda (regla anti-NaN)


class RecallTap(nn.Module):
    """Proyecciones globales de consulta/clave + lectura dura (top-1).

    Una sola instancia por modelo: ``P_q``/``P_k`` se comparten entre todas las
    capas que inyectan la lectura (los codigos K dependen solo del token).
    """

    def __init__(
        self,
        d_model: int,
        d_recall: int,
        *,
        value: str = "next",
        gap: int = 1,
        temperature: float = 0.5,
        score_chunk: int = 1024,
        init_std: float = 0.1,
        shared_init: bool = True,
    ):
        super().__init__()
        if value not in ("next", "self"):
            raise ValueError(f"rt_value debe ser 'next' o 'self', no {value!r}")
        if gap < 1:
            raise ValueError("rt_gap >= 1")
        self.d_model = d_model
        self.d_recall = d_recall
        self.value = value
        self.gap = gap
        self.temperature = max(1e-2, float(temperature))
        self.score_chunk = max(64, int(score_chunk))
        self.p_q = nn.Linear(d_model, d_recall, bias=False)
        self.p_k = nn.Linear(d_model, d_recall, bias=False)
        nn.init.normal_(self.p_q.weight, std=init_std)
        if shared_init:
            # Inicializacion simetrica: con P_q = P_k, el score de un token
            # consigo mismo (norma al cuadrado) tiende a dominar desde el paso 0,
            # dando al argmax un arranque correcto que el CE solo tiene que afinar.
            with torch.no_grad():
                self.p_k.weight.copy_(self.p_q.weight)
        else:
            nn.init.normal_(self.p_k.weight, std=init_std)
        self.scale = 1.0 / math.sqrt(d_recall)

    # ------------------------------------------------------------------
    def keys(self, t0: torch.Tensor) -> torch.Tensor:
        """Codigos K de una traza ``t0`` (aislado: depende solo de cada token)."""
        return self.p_k(t0)

    def queries(self, source: torch.Tensor) -> torch.Tensor:
        """Consultas q (desde T0 o desde el estado consolidado)."""
        return self.p_q(source)

    def _value_matrix(self, t0: torch.Tensor) -> torch.Tensor:
        """V[j] = T0[j+1] (modo 'next') o T0[j] (modo 'self'); ultima fila a 0."""
        if self.value == "self":
            return t0
        v = F.pad(t0, (0, 0, 0, 1))[..., 1:, :]  # desplaza una posicion
        return v

    # ------------------------------------------------------------------
    # Camino paralelo (entrenamiento): (B, N, d) -> lecturas (B, N, d)
    # ------------------------------------------------------------------
    def forward_parallel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        t0: torch.Tensor,
        *,
        score_rows: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Lectura dura en todas (o en ``score_rows``) las posiciones.

        Args:
            q: consultas (B, N, d_k).
            k: codigos de la traza (B, N, d_k).
            t0: huellas pristinas (B, N, d) (matriz de valores).
            score_rows: indices de posiciones a leer (por defecto, todas).
                Evaluar solo las filas necesarias es el camino rapido de
                evaluacion y el coste O(filas * N) en vez de O(N^2).
        """
        b, n, _ = k.shape
        device = k.device
        v = self._value_matrix(t0)
        rows = (
            torch.arange(n, device=device) if score_rows is None
            else score_rows.to(device)
        )
        out = torch.zeros_like(t0)
        chunk = self.score_chunk
        flat_q = q  # (B, N, dk)
        for start in range(0, rows.numel(), chunk):
            idx = rows[start : start + chunk]                     # (C,)
            c = idx.numel()
            # scores (B, C, N) en fp32 (regla anti-NaN)
            s = torch.einsum("bcd,bnd->bcn", flat_q[:, idx, :].float(), k.float()) * self.scale
            pos = idx.view(1, c, 1)
            colj = torch.arange(n, device=device).view(1, 1, n)
            valid = colj <= (pos - self.gap)                       # causal
            s = torch.where(valid, s, torch.full_like(s, _NEG))
            row_ok = valid.any(dim=-1)                             # (B, C)
            # --- lectura dura: argmax con desempate por ocurrencia mas reciente
            j_star = (n - 1) - s.flip(-1).argmax(dim=-1)       # (B, C)
            gather = j_star.clamp(0, n - 1)
            hard = v.gather(
                1, gather.view(b, c, 1).expand(b, c, t0.size(-1))
            )                                                      # (B, C, d)
            hard = hard * row_ok.unsqueeze(-1).to(hard.dtype)
            # --- gradiente straight-through (solo backward; valor = lectura dura)
            if self.training and torch.is_grad_enabled():
                soft_w = F.softmax(s / self.temperature, dim=-1)
                soft_w = torch.nan_to_num(soft_w, nan=0.0) * row_ok.unsqueeze(-1).to(soft_w.dtype)
                soft = soft_w @ v
                reads = hard + (soft - soft.detach())
            else:
                reads = hard
            out = out.index_copy(
                1, idx, reads.to(out.dtype)
            )
        return out

    # ------------------------------------------------------------------
    # Camino paralelo LINEAL (modo LSH): candidatos indexados
    # ------------------------------------------------------------------
    def forward_parallel_lsh(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        t0: torch.Tensor,
        tokens: torch.Tensor,
        *,
        n_tables: int = 2,
        n_bits: int = 8,
        cap: int = 64,
        seed: int = 7,
        n_neg: int = 32,
    ) -> torch.Tensor:
        """Lectura dura identica en semantica a ``forward_parallel``, con
        candidatura indexada (indice exacto de misma-token U buckets LSH).

        Coste O(N (1 + t*cap) d_k) — lineal en N. El candidato de
        induccion/ligadura (ocurrencia previa del MISMO token) esta garantizado
        por construccion: mismos tokens -> mismo codigo K -> mismo bucket, mas
        el indice exacto de ultima ocurrencia.
        """
        from engrama.v5.lsh import LSHIndex

        b, n, _ = k.shape
        v = self._value_matrix(t0)
        out = torch.zeros_like(t0)
        for bi in range(b):
            index = LSHIndex.build(
                k[bi], tokens[bi], gap=self.gap, n_tables=n_tables,
                n_bits=n_bits, cap=cap, seed=seed,
            )
            cand, valid = index.candidates()          # (N, C), (N, C)
            if n_neg > 0:
                # negativos muestreados uniformes por fila (sampled softmax):
                # mantienen a raya los scores de tokens fuera de bucket.
                gen = torch.Generator(device="cpu").manual_seed(seed + 1234)
                n_ = cand.size(0)
                lim = torch.arange(n_, device=cand.device) - self.gap
                r = torch.rand(n_, n_neg, generator=gen).to(cand.device)
                neg = (r * (lim.clamp(min=0) + 1).unsqueeze(1)).long()
                ok = (lim.unsqueeze(1) >= 0).expand(n_, n_neg)
                cand = torch.cat([cand, neg], dim=1)
                valid = torch.cat([valid, ok], dim=1)
            c = cand.size(1)
            cand_k = k[bi].index_select(0, cand.clamp(min=0).reshape(-1)).view(n, c, -1)
            cand_v = v[bi].index_select(0, cand.clamp(min=0).reshape(-1)).view(n, c, -1)
            chunk = self.score_chunk
            for start in range(0, n, chunk):
                end = min(n, start + chunk)
                m = end - start
                s = torch.einsum("md,mcd->mc", q[bi, start:end].float(),
                                 cand_k[start:end].float()) * self.scale
                ok = valid[start:end]                  # (m, C)
                s = torch.where(ok, s, torch.full_like(s, _NEG))
                row_ok = ok.any(dim=-1)
                # Prioridad estructural de la IDENTIDAD (columna 0): los
                # empates exactos no sobreviven al GEMM (1 ULP entre columnas
                # con K identico), asi que la identidad gana si empata dentro
                # de una tolerancia de ULP. Fija la semantica del camino denso
                # (ocurrencia MAS RECIENTE del propio token); el resto de
                # candidatos (rescate/buckets) va reciente-primero.
                s0 = s[:, 0]
                if s.size(1) > 1:
                    best_rest = s[:, 1:].max(dim=-1).values
                    rest_arg = 1 + s[:, 1:].argmax(dim=-1)
                else:
                    best_rest = torch.full_like(s0, _NEG)
                    rest_arg = torch.zeros_like(s0, dtype=torch.long)
                use_id = valid[start:end, 0] & (s0 >= best_rest - 1e-6)
                j_star = torch.where(use_id, torch.zeros_like(rest_arg), rest_arg)
                hard = cand_v[start:end].gather(
                    1, j_star.view(m, 1, 1).expand(m, 1, cand_v.size(-1))
                ).squeeze(1)
                hard = hard * row_ok.unsqueeze(-1).to(hard.dtype)
                if self.training and torch.is_grad_enabled():
                    soft_w = F.softmax(s / self.temperature, dim=-1)
                    soft_w = torch.nan_to_num(soft_w, nan=0.0) * row_ok.unsqueeze(-1).to(soft_w.dtype)
                    soft = torch.einsum("mc,mcd->md", soft_w, cand_v[start:end])
                    reads = hard + (soft - soft.detach())
                else:
                    reads = hard
                out[bi, start:end] = reads.to(out.dtype)
        return out

    # ------------------------------------------------------------------
    # Camino incremental (generacion): matvec contra el anillo K
    # ------------------------------------------------------------------
    def read_step(
        self,
        q_t: torch.Tensor,
        k_ring: torch.Tensor,
        t0_ring: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        """Lectura para el token actual.

        Args:
            q_t: consulta (B, d_k).
            k_ring: anillo preasignado (N_max, d_k) con ``length`` codigos escritos.
            t0_ring: anillo preasignado (N_max, d) con ``length`` huellas.
            length: tokens ya escritos en la traza (incluido el actual).
        """
        limit = length - 1 - self.gap              # j <= i - gap (i = length-1)
        if limit < 0:
            return q_t.new_zeros(q_t.size(0), t0_ring.size(-1))
        kf = k_ring[: limit + 1].float()                  # (m, B, dk) o (m, dk)
        qf = q_t.float()                                  # (B, dk)
        if kf.dim() == 3:
            s = torch.einsum("jbd,bd->bj", kf, qf) * self.scale
        else:
            s = (kf @ qf.unsqueeze(-1)).squeeze(-1) * self.scale   # (B, m)
        j_star = limit - s.flip(-1).argmax(-1)             # empate -> mas reciente
        j_star = j_star.clamp(0, limit)
        idx = (j_star if self.value == "self"
               else (j_star + 1).clamp(max=length - 1))     # (B,)
        if t0_ring.dim() == 3:                              # anillo por lotes (N, B, d)
            rows = torch.arange(t0_ring.size(1), device=t0_ring.device)
            src = t0_ring[idx, rows]                        # (B, d)
        else:                                               # anillo B=1 (N, d)
            src = t0_ring[idx]                              # (1, d)
        return src.to(q_t.dtype)

    # ------------------------------------------------------------------
    def extra_state_dict_prefixes(self):  # pragma: no cover - helper de depuracion
        return ("p_q", "p_k")
