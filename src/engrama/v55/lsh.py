"""ENGRAMA V5.5 — Indice LSH cuantizado para el Recall Tap asimetrico (Pilar 5).

Por aislamiento (pilar 1), ``K_lex[j] = P_k_lex(T0[token_j])`` depende SOLO del
token ``j``. Por tanto:

* dos posiciones con el mismo token tienen ``K_lex`` IDENTICOS -> cualquier
  hash determinista las manda SIEMPRE al mismo bucket (recall 1.0 para
  matching lexico, sin probabilidad);
* el candidato de induccion (ocurrencia previa de MI token) se garantiza con
  un indice exacto de ultima ocurrencia por ``token_id`` (O(1)).

**Cuantizacion** (V5.5): ``K_lex`` se binariza con ``sign(K_lex)`` y se empaqueta
en palabras de 64 bits. El bucket es el codigo de signos; mismo token -> mismo
signo -> mismo bucket. Con ``t`` tablas de proyecciones aleatorias se obtiene
recuperacion robusta a distancia de Hamming (multi-probe LSH), que es la forma
eficiente y lineal del "hamming < umbral". En GPU, el popcount (XOR+bitcount) es
1 instruccion (kernel fusionado en :mod:`engrama.v55.kernels`).

Candidatos por consulta: ``1 (identidad) + recent_k (rescate) + t*cap (LSH)``.
Con los ``n_neg`` negativos muestreados del recall -> 181 como V5.

Coste total: hashing ``O(N d_k t b)`` + sort/scatter ``O(N log N)`` + puntuacion
``O(N (1 + t cap) d_k)`` -> LINEAL en N. La lectura dura conserva su semantica
(argmax, empates -> mas reciente); el straight-through se calcula solo sobre los
candidatos.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def _cumcount_offsets(sorted_keys: torch.Tensor, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
    """Ultima posicion ``j <= i-gap`` con el MISMO token que ``i`` (o -1).

    ``tokens``: ``(N,)`` long. Induce exacto ``O(N log N)`` via ordenacion
    estable + truco del desfase (demostrado en tests).
    """
    n = tokens.numel()
    device = tokens.device
    if n == 0:
        return tokens.new_empty(0)
    order = torch.argsort(tokens, stable=True)
    prev_sorted = torch.full_like(order, -1)
    if n > 1:
        same = tokens[order[1:]] == tokens[order[:-1]]
        prev_sorted[1:] = torch.where(same, order[:-1], prev_sorted[1:])
    p1 = torch.full_like(order, -1)
    p1.scatter_(0, order, prev_sorted)
    shift = gap - 1
    if shift == 0:
        return p1
    out = torch.full_like(p1, -1)
    out[shift:] = p1[:-shift]
    return out


@torch.no_grad()
def _bucket_matrix(codes: torch.Tensor, n: int, n_codes: int, cap: int) -> torch.Tensor:
    """Matriz ``(n_codes, cap)`` con las posiciones mas RECIENTES de cada bucket."""
    device = codes.device
    key = codes * n + (n - 1 - torch.arange(n, device=device))
    order = torch.argsort(key)
    group_id, starts = _cumcount_offsets(codes[order], n)
    slot = torch.arange(n, device=device) - starts
    keep = slot < cap
    bucket = torch.full((n_codes, cap), -1, dtype=torch.long, device=device)
    bucket[codes[order][keep], slot[keep]] = order[keep]
    return bucket


@torch.no_grad()
def sign_bitpack(k: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Cuantiza ``k`` (N, d_k) a codigos de signo de ``n_bits`` bits por tabla.

    Devuelve ``(N,)`` long con el entero del codigo binario (signo de la
    proyeccion aleatoria). Determinista (semilla fija): mismo ``K_lex`` ->
    mismo codigo siempre (propiedad de induccion).
    """
    n, d_k = k.shape
    device = k.device
    gen = torch.Generator(device="cpu").manual_seed(7)
    planes = torch.randint(0, 2, (1, d_k, n_bits), generator=gen,
                           dtype=torch.float32) * 2 - 1
    planes = planes.to(device=device, dtype=k.dtype)
    bits = (k.float().unsqueeze(0) @ planes.float()) > 0        # (1, N, b)
    weights = (1 << torch.arange(n_bits, device=device)).long()
    return (bits.long() @ weights).squeeze(0)                   # (N,)


class LSHIndexV2:
    """Construye y consulta los candidatos del Recall Tap asimetrico (lineal).

    Uso::

        idx = LSHIndexV2.build(k_lex, tokens, gap=1, n_tables=2, n_bits=32, cap=64)
        cand, valid = idx.candidates()      # (N, 1 + recent_k + t*cap)
    """

    def __init__(self, cand: torch.Tensor, valid: torch.Tensor):
        self.cand = cand    # (N, C) long; -1 = hueco
        self.valid = valid  # (N, C) bool

    @classmethod
    def build(
        cls,
        k_lex: torch.Tensor,     # (N, d_k) codigos lexicos de la traza
        tokens: torch.Tensor,    # (N,) ids de token
        *,
        gap: int = 1,
        n_tables: int = 2,
        n_bits: int = 8,
        cap: int = 64,
        seed: int = 7,
        hamming: int = 3,        # reservado: numero de tablas aproxima hamming
    ) -> "LSHIndexV2":
        n = k_lex.size(0)
        device = k_lex.device
        n_codes = 1 << min(n_bits, 16)   # tabla de buckets acotada (max 65536)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        planes = torch.randint(0, 2, (n_tables, k_lex.size(1), n_bits),
                               generator=gen, dtype=torch.float32) * 2 - 1
        planes = planes.to(device=device, dtype=k_lex.dtype)
        bits = (k_lex.float().unsqueeze(0) @ planes.float()) > 0   # (t, N, b)
        weights = (1 << torch.arange(n_bits, device=device)).long()
        codes = (bits.long() @ weights).T % n_codes                # (N, t) en [0,n_codes)

        cols = [previous_same_occurrence(tokens.long(), gap=gap).unsqueeze(1)]
        recent_k = 4
        idx = torch.arange(n, device=device).unsqueeze(1)
        offsets = torch.arange(gap, gap + recent_k, device=device).unsqueeze(0)
        cols.append(idx - offsets)                                  # (N, recent_k)
        for t in range(n_tables):
            bucket = _bucket_matrix(codes[:, t], n, n_codes, cap)
            cols.append(bucket[codes[:, t]])                        # (N, cap)
        cand = torch.cat(cols, dim=1)                               # (N, C)
        idx = torch.arange(n, device=device).unsqueeze(1)
        valid = (cand >= 0) & (cand <= idx - gap)
        return cls(cand, valid)

    def candidates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cand, self.valid
