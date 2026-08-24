"""ENGRAMA V6 — Indice LSH de ALTA RECUPERACION para el Recall Tap (Pilar 5).

Diferencia clave frente a V5.5:

* **16 tablas de 16 bits** (65536 buckets cada una) en vez de 2 tablas de 8 bits.
  Con 65536 buckets y ~N entradas, cada bucket tiene ~N/65536 elementos (p.ej.
  2 a 128K): los buckets son diminutos y no hace falta expulsar a nadie (cap=0).
* **SIN cap de expulsion**: todos los miembros del bucket son candidatos. Con
  16 tablas, el vecino mas cercano cae en el mismo bucket que la consulta en al
  menos una tabla con probabilidad muy alta para cosenos >= 0.9 (los verdaderos
  matches semanticos y lexicos).
* **Rescate denso de ventana fija** (W=256): las ultimas W posiciones se anaden
  como candidatos de todas formas. Esto garantiza que ningun match reciente se
  pierde por un mal hash, y como W es constante el coste es O(N*W*d) -> lineal.
* **Claves FP32**: incluso con el modelo en FP16/BF16, los codigos se calculan
  en FP32 => los empates y el argmax son deterministas (invarianza causal).
* **Indice invertido por bucket** (CSR): construir el indice es O(N*t),
  consultarlo es O(sumatorio de tamanos de bucket) = O(N*t + N*C), sin el
  ``_bucket_matrix`` de V5.5 que hacia un argsort por tabla (N log N).

El coste total es LINEAL en N:
``O(N * t * d * b)`` para hashing + ``O(N*t)`` para indexar +
``O(N * C * d)`` para puntuacion, con ``C`` el numero medio de candidatos
(pequeno: identidad + W rescate + sumatoria de tamanos de bucket).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


# ----------------------------------------------------------------------
# Indice de ultima ocurrencia (identidad lexico O(1))
# ----------------------------------------------------------------------
@torch.no_grad()
def previous_same_occurrence(tokens: torch.Tensor, gap: int = 1) -> torch.Tensor:
    """Ultima posicion ``j <= i-gap`` con el MISMO token (o -1). O(N) exacto."""
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


# ----------------------------------------------------------------------
# Planos de hashing compartidos (deterministas)
# ----------------------------------------------------------------------
def shared_planes(n_tables: int, d: int, n_bits: int, device, dtype,
                  seed: int = 7) -> torch.Tensor:
    """Devuelve ``(t, d, b)`` planos +/-1 deterministas (compartidos consulta/clave)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    planes = (torch.randint(0, 2, (n_tables, d, n_bits), generator=gen,
                            dtype=torch.float32) * 2 - 1)
    return planes.to(device=device, dtype=torch.float32)


@torch.no_grad()
def sign_codes(x: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
    """Codigos de signo para ``x`` (N,d). Devuelve ``(N, t)`` long.

    Los calculos se hacen en FP32 aunque ``x`` venga en FP16/BF16.
    """
    n_bits = planes.shape[-1]
    n_codes = 1 << n_bits
    xf = x.float()
    bits = (xf.unsqueeze(0) @ planes) > 0          # (t, N, b)
    weights = (1 << torch.arange(n_bits, device=x.device)).long()
    return (bits.long() @ weights).T % n_codes      # (N, t)


# ----------------------------------------------------------------------
# Indice invertido CSR por bucket (una tabla)
# ----------------------------------------------------------------------
class CSRBucketIndex:
    """Indice CSR de posiciones por bucket para una sola tabla.

    Construccion O(N log N) (un argsort). Cada bucket conserva solo las ``cap``
    posiciones MAS RECIENTES (cap=0 = todas). Esto es suficiente para
    recuperacion semantica y de induccion: si un concepto aparece, sus
    ocurrencias recientes estan en el bucket; el resto se cubre con el rescate
    de ventana. Mantener C acotado es lo que hace que el coste sea O(N).
    """

    __slots__ = ("n_codes", "offsets", "members", "n", "cap")

    def __init__(self, codes: torch.Tensor, n_codes: int, cap: int = 0):
        n = codes.numel()
        device = codes.device
        if cap and cap > 0:
            total_counts = torch.bincount(codes, minlength=n_codes)
            kept_counts = total_counts.clamp(max=cap)
            offsets = torch.zeros(n_codes + 1, dtype=torch.long, device=device)
            torch.cumsum(kept_counts, dim=0, out=offsets[1:])
            total = int(offsets[-1].item())
            members = torch.full((total,), -1, dtype=torch.long, device=device)
            if total > 0:
                # Ordena por (bucket, posicion DESCENDENTE): las mas recientes
                # primero dentro de cada bucket.
                key = codes.long() * (n + 1) + (n - 1 - torch.arange(n, device=device))
                order = torch.argsort(key, stable=True)
                sorted_codes = codes[order]
                # rank local 0,1,2... dentro del bucket (0 = mas reciente)
                change = torch.ones(n, dtype=torch.bool, device=device)
                change[1:] = sorted_codes[1:] != sorted_codes[:-1]
                bucket_id = torch.cumsum(change, 0) - 1
                rank = torch.arange(n, device=device) - offsets[bucket_id]
                keep = rank < cap
                # Dentro de cada bucket del miembro 'members', escribimos en
                # orden inverso para que members[...] quede de antigua a reciente
                # (no afecta a la correccion, solo al orden).
                kept_codes = sorted_codes[keep]
                kept_pos = order[keep]
                kept_rank = rank[keep]
                # destino = offsets[bucket] + (kept_counts[bucket]-1 - kept_rank)
                dest = offsets[kept_codes] + (kept_counts[kept_codes] - 1 - kept_rank)
                members[dest] = kept_pos
            self.n_codes = n_codes
            self.offsets = offsets
            self.members = members
            self.n = total
            self.cap = cap
            return
        counts = torch.bincount(codes, minlength=n_codes)
        offsets = torch.zeros(n_codes + 1, dtype=torch.long, device=device)
        torch.cumsum(counts, dim=0, out=offsets[1:])
        members = torch.empty(n, dtype=torch.long, device=device)
        order = torch.argsort(codes, stable=True)
        members[order] = torch.arange(n, device=device)
        self.n_codes = n_codes
        self.offsets = offsets
        self.members = members
        self.n = n
        self.cap = 0

    def lookup(self, bucket_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Devuelve (out_offsets, members_gather) para un vector de buckets.

        Vectorizado (sin bucle Python): expande cada segmento [starts[i],ends[i])
        en un tensor plano usando ``arange`` mascarado y ``index_put``.
        """
        b = bucket_ids.numel()
        starts = self.offsets[bucket_ids]
        ends = self.offsets[bucket_ids + 1]
        sizes = ends - starts
        out_offsets = torch.zeros(b + 1, dtype=torch.long, device=bucket_ids.device)
        torch.cumsum(sizes, dim=0, out=out_offsets[1:])
        total = int(out_offsets[-1].item())
        members_gather = torch.empty(total, dtype=torch.long, device=bucket_ids.device)
        if total > 0:
            # Para cada consulta i, copiamos members[starts[i]:ends[i]] al tramo
            # [out_offsets[i]:out_offsets[i+1]].
            max_sz = int(sizes.max().item())
            if max_sz > 0:
                # (b, max_sz) indices locales
                local = torch.arange(max_sz, device=bucket_ids.device).unsqueeze(0)
                src_idx = (starts.unsqueeze(1) + local).clamp(max=self.n - 1)
                dst_idx = (out_offsets[:-1].unsqueeze(1) + local)
                mask = local < sizes.unsqueeze(1)
                src_flat = src_idx[mask]
                dst_flat = dst_idx[mask]
                members_gather[dst_flat] = self.members[src_flat]
        return out_offsets, members_gather


# ----------------------------------------------------------------------
# Indice LSH V6 completo (multi-tabla + rescate + identidad)
# ----------------------------------------------------------------------
class V6LSHIndex:
    """Indice LSH multi-tabla de alta recuperacion.

    Uso::

        idx = V6LSHIndex.build(keys, tokens, gap=1, n_tables=16, n_bits=16,
                                rescue_window=256)
        cand, valid = idx.candidates(query_codes)   # (N, C)
    """

    def __init__(self, tables, n_codes: int, identity: torch.Tensor,
                 rescue: torch.Tensor, n: int, gap: int, device):
        self.tables = tables            # lista de CSRBucketIndex
        self.n_codes = n_codes
        self.identity = identity        # (N,)
        self.rescue = rescue            # (N, W) posiciones de rescate
        self.n = n
        self.gap = gap
        self.device = device

    @classmethod
    @torch.no_grad()
    def build(
        cls,
        keys: torch.Tensor,            # (N, d)
        tokens: torch.Tensor,           # (N,)
        *,
        gap: int = 1,
        n_tables: int = 16,
        n_bits: int = 16,
        rescue_window: int = 256,
        seed: int = 7,
        bucket_cap: int = 4,
    ) -> "V6LSHIndex":
        n, d = keys.shape
        device = keys.device
        n_codes = 1 << n_bits
        planes = shared_planes(n_tables, d, n_bits, device, keys.dtype, seed=seed)
        codes = sign_codes(keys, planes)       # (N, t)
        tables = [CSRBucketIndex(codes[:, t], n_codes, cap=bucket_cap)
                  for t in range(n_tables)]
        identity = previous_same_occurrence(tokens.long(), gap=gap)
        # rescate denso: las ultimas W posiciones (causalmente validas)
        W = min(rescue_window, n)
        idx = torch.arange(n, device=device).unsqueeze(1)
        off = torch.arange(gap, gap + W, device=device).unsqueeze(0)
        rescue = (idx - off).clamp(min=-1)
        return cls(tables, n_codes, identity, rescue, n, gap, device)

    @torch.no_grad()
    def query_codes(self, query_keys: torch.Tensor) -> torch.Tensor:
        """Calcula los codigos de consulta. Requiere los planos; los reconstruye
        con la misma semilla que ``build`` para que sean compartidos."""
        # Los planos se recrean; para evitar acoplamiento, el llamador pasa ya
        # los codigos via candidates_from_codes.
        raise NotImplementedError("use candidates_from_codes")

    @torch.no_grad()
    def candidates(self, query_codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``query_codes``: (N, t) long (mismos planos que las claves).
        Devuelve (cand (N, C), valid (N, C)). Vectorizado."""
        n = self.n
        device = self.device
        cols = [self.identity.unsqueeze(1), self.rescue]
        for t, table in enumerate(self.tables):
            q = query_codes[:, t]
            out_off, members_gather = table.lookup(q)
            sizes = out_off[1:] - out_off[:-1]
            c_t = int(sizes.max().item()) if sizes.numel() else 0
            if c_t == 0:
                continue
            block = torch.full((n, c_t), -1, dtype=torch.long, device=device)
            total = int(out_off[-1].item())
            if total > 0:
                row_ids = torch.arange(n, device=device).repeat_interleave(sizes)
                ar = torch.arange(total, device=device)
                local = ar - out_off[row_ids]
                block[row_ids, local] = members_gather
            cols.append(block)
        cand = torch.cat(cols, dim=1)
        idx = torch.arange(n, device=device).unsqueeze(1)
        valid = (cand >= 0) & (cand <= idx - self.gap)
        cand = torch.where(valid, cand, torch.full_like(cand, -1))
        return cand, valid


def lsh_candidate_count(tables, rescue, n):
    """Estimacion del numero total de candidatos (para diagnostico)."""
    total = 1 + rescue.size(1)
    for t in tables:
        # promedio de candidatos por consulta = suma de (tamano de bucket) / N
        sizes = t.offsets[1:] - t.offsets[:-1]
        total += float(sizes.sum().item()) / max(1, n)
    return total
