"""ENGRAMA V5 — Indice LSH + ocurrencia-identica para el Recall Tap lineal.

Clave de la arquitectura (que hace esto posible y EXACTO para el caso
importante): por aislamiento (pilar 1), ``K[j] = P_k(T_0[token_j])`` depende
SOLO del token j. Por tanto:

* dos posiciones con el mismo token tienen codigos K IDENTICOS -> cualquier
  hash determinista de K las manda SIEMPRE al mismo bucket (recall 1.0 para
  matching por identidad, sin probabilidad);
* el candidato de induccion/ligadura ("la ocurrencia previa de MI token") se
  puede garantizar con un indice exacto de ultima ocurrencia por token.

Candidatos por consulta (t tablas LSH de b bits + indice exacto):

    C(i) = { ultima ocurrencia de token_i con j <= i-gap }        (exacto, 1)
         U { posiciones de los buckets LSH de K[i], j <= i-gap }  (t * cap)

Coste total: hashing O(N d_k t b) + sort/scatter O(N log N) + puntuacion
O(N (1 + t cap) d_k) -> LINEAL en N con constante (1 + t cap) d_k.
A N=8192, d_k=64, t=2, b=8, cap=64: ~67 MFLOPs vs 8.6 GFLOPs densos (~128x).

La lectura DURA conserva su semantica (argmax, empates -> mas reciente);
el gradiente straight-through se calcula solo sobre los candidatos.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def _cumcount_offsets(sorted_keys: torch.Tensor, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dado un tensor 1D ordenado, devuelve (group_id, inicio_de_grupo) por elemento."""
    device = sorted_keys.device
    change = torch.ones(n, dtype=torch.bool, device=device)
    if n > 1:
        change[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_id = torch.cumsum(change.long(), 0) - 1
    starts = torch.zeros(int(group_id[-1].item()) + 1, dtype=torch.long, device=device)
    starts.scatter_(0, group_id[change].long(), change.nonzero().flatten())
    return group_id, starts[group_id]


@torch.no_grad()
def previous_same_occurrence(tokens: torch.Tensor, gap: int = 1) -> torch.Tensor:
    """Ultima posicion j <= i-gap con el MISMO token que i (o -1).

    ``tokens``: (N,) long. Induce exacto O(N log N) via ordenacion estable.
    """
    n = tokens.numel()
    device = tokens.device
    if n == 0:
        return tokens.new_empty(0)
    order = torch.argsort(tokens, stable=True)
    pos_sorted = order  # posiciones ordenadas por token
    prev_sorted = torch.full_like(pos_sorted, -1)
    if n > 1:
        same = tokens[pos_sorted[1:]] == tokens[pos_sorted[:-1]]
        prev_sorted[1:] = torch.where(same, pos_sorted[:-1], prev_sorted[1:])
    p1 = torch.full_like(pos_sorted, -1)   # p1[i] = ocurrencia previa de token_i (< i)
    p1.scatter_(0, pos_sorted, prev_sorted)
    # La ocurrencia mas reciente con j <= i-gap es p1[i-gap+1] (p1[k] < k <= i-gap+1
    # implica p1[k] <= i-gap). Truco exacto, sin cadenas ni bucles.
    shift = gap - 1
    if shift == 0:
        return p1
    out = torch.full_like(p1, -1)
    out[shift:] = p1[:-shift]
    return out


@torch.no_grad()
def _bucket_matrix(codes: torch.Tensor, n: int, n_codes: int, cap: int) -> torch.Tensor:
    """Matriz (n_codes, cap) con las posiciones mas RECIENTES de cada bucket.

    Vacia = -1. Cada bucket queda ordenado de mas reciente a mas antiguo.
    """
    device = codes.device
    # ordenar por (codigo asc, posicion desc): empaquetar en una sola clave
    key = codes * n + (n - 1 - torch.arange(n, device=device))
    order = torch.argsort(key)
    group_id, starts = _cumcount_offsets(codes[order], n)
    slot = torch.arange(n, device=device) - starts  # 0 = mas reciente del bucket
    keep = slot < cap
    bucket = torch.full((n_codes, cap), -1, dtype=torch.long, device=device)
    bucket[codes[order][keep], slot[keep]] = order[keep]
    return bucket


class LSHIndex:
    """Construye y consulta los candidatos del Recall Tap (modo lineal).

    Uso::

        idx = LSHIndex.build(k, tokens, gap=1, n_tables=2, n_bits=8, cap=64)
        cand, valid = idx.candidates()      # (N, 1 + t*cap), bool
        scores = torch.einsum("nd,ncd->nc", q, k[cand.clamp(0)])
    """

    def __init__(self, cand: torch.Tensor, valid: torch.Tensor):
        self.cand = cand   # (N, C) long; -1 = hueco
        self.valid = valid  # (N, C) bool

    @classmethod
    def build(
        cls,
        k: torch.Tensor,          # (N, d_k) codigos de la traza
        tokens: torch.Tensor,     # (N,) ids de token
        *,
        gap: int = 1,
        n_tables: int = 2,
        n_bits: int = 8,
        cap: int = 64,
        seed: int = 7,
    ) -> "LSHIndex":
        n, d_k = k.shape
        device = k.device
        n_codes = 1 << n_bits
        gen = torch.Generator(device="cpu").manual_seed(seed)
        planes = torch.randint(0, 2, (n_tables, d_k, n_bits), generator=gen,
                               dtype=torch.float32) * 2 - 1
        planes = planes.to(device=device, dtype=k.dtype)
        bits = (k.float().unsqueeze(0) @ planes.float()) > 0        # (t, N, b)
        weights = (1 << torch.arange(n_bits, device=device)).long()
        codes = (bits.long() @ weights).T                            # (N, t)

        cols = [previous_same_occurrence(tokens.long(), gap=gap).unsqueeze(1)]
        # rescate: las `recent_k` posiciones mas recientes (cubre esquinas
        # degeneradas — metrica plana, o sin vecino de bucket — y garantiza
        # que ninguna fila con historia quede vacia).
        recent_k = 4
        idx = torch.arange(n, device=device).unsqueeze(1)
        offsets = torch.arange(gap, gap + recent_k, device=device).unsqueeze(0)
        cols.append(idx - offsets)                                   # (N, recent_k)
        for t in range(n_tables):
            bucket = _bucket_matrix(codes[:, t], n, n_codes, cap)
            cols.append(bucket[codes[:, t]])                         # (N, cap)
        cand = torch.cat(cols, dim=1)                                # (N, 1+k+t*cap)
        idx = torch.arange(n, device=device).unsqueeze(1)
        valid = (cand >= 0) & (cand <= idx - gap)
        return cls(cand, valid)

    # ------------------------------------------------------------------
    def candidates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cand, self.valid
