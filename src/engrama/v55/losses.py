"""ENGRAMA V5.5 — perdidas con softcap (Pilar 6).

* :func:`softcap` (reexportada de :mod:`engrama.v55.primitives`).
* :func:`softcap_linear_cross_entropy` — proyeccion lineal + softcap + CE en
  trozos de posiciones (sin materializar todos los logits), para estabilidad a
  LR alto.
* :func:`retrieval_cross_entropy` — CE sobre los scores del Recall Tap que
  ensena al tap a apuntar a la posicion correcta (auto-supervisado: la posicion
  cuyo token siguiente == objetivo). Seccion 6.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from engrama.v55.primitives import softcap


def _softcap_ce_chunk(hidden, weight, targets, scale, cap, ignore_index):
    logits = F.linear(hidden, weight) * scale
    logits = softcap(logits, cap)
    return F.cross_entropy(logits.float(), targets,
                           ignore_index=ignore_index, reduction="sum")


def softcap_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    scale: float = 1.0,
    cap: float = 30.0,
    chunk_size: int = 2048,
    ignore_index: int = -100,
    checkpoint_chunks: bool = True,
) -> torch.Tensor:
    """Proyeccion lineal + softcap + cross-entropy, troceada por posiciones.

    Numerikamente equivalente a ``CE(softcap(hidden @ W^T * scale))`` sin
    materializar la matriz ``(..., vocab)`` completa.
    """
    flat_h = hidden.reshape(-1, hidden.size(-1))
    flat_y = targets.reshape(-1)
    scale_t = flat_h.new_tensor(scale)
    cap_t = flat_h.new_tensor(cap)
    n_tokens = flat_h.size(0)

    def _one(h, y):
        if checkpoint_chunks and torch.is_grad_enabled():
            return checkpoint(_softcap_ce_chunk, h, weight, y, scale_t, cap_t,
                              ignore_index, use_reentrant=False)
        return _softcap_ce_chunk(h, weight, y, scale_t, cap_t, ignore_index)

    if n_tokens <= chunk_size:
        total = _one(flat_h, flat_y)
    else:
        total = flat_h.new_zeros((), dtype=torch.float32)
        for start in range(0, n_tokens, chunk_size):
            total = total + _one(flat_h[start:start + chunk_size],
                                 flat_y[start:start + chunk_size])
    denom = (flat_y != ignore_index).sum().clamp_min(1)
    return total / denom


def retrieval_cross_entropy(
    scores: torch.Tensor,        # (B, N, N) o (B, N, C) scores del tap
    candidates: torch.Tensor,    # (B, N, C) long, posiciones candidatas (-1 hueco)
    valid: torch.Tensor,         # (B, N, C) bool
    targets: torch.Tensor,       # (B, N) long, next-token id objetivo
    next_tokens: torch.Tensor,   # (B, N) long, token en j+1 de cada posicion j (=T0 index)
    *, gap: int = 1, temperature: float = 1.0,
    ignore_index: int = -100,
    position_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CE auto-supervisado del Recall Tap (Seccion 6).

    Para cada posicion ``i`` con objetivo ``y_i``, las posiciones candidatas
    ``j`` cuyo token siguiente (``next_tokens[j]``) coincide con ``y_i`` son
    objetivos positivos; se empuja ``scores[i, j]`` hacia arriba con CE sobre
    los candidatos. Entrena el tap para recuperar el sitio cuyo SIGUIENTE token
    es el objetivo (senal de induccion/copia), sin etiquetas externas.

    Solo se aplica donde ``position_mask`` es True (por defecto todas las
    posiciones con historia).
    """
    b, n, c = candidates.shape
    device = scores.device
    if position_mask is None:
        position_mask = torch.ones(b, n, dtype=torch.bool, device=device)
    # cand_tok[i,c] = next_tokens[candidate j]: el token que aporta leer la pos j
    nt = next_tokens.unsqueeze(-1).expand(-1, -1, c)        # (B,N,C)
    cand_tok = nt.gather(1, candidates.clamp(min=0))         # (B,N,C)
    target_exp = targets.view(b, n, 1)
    correct = (cand_tok == target_exp) & valid & (targets != ignore_index).unsqueeze(-1)
    has_pos = correct.any(dim=-1) & position_mask
    if not has_pos.any():
        return scores.new_zeros(())
    # log-softmax sobre candidatos validos
    s = scores.float().masked_fill(~valid, -1e9) / max(1e-2, temperature)
    logp = F.log_softmax(s, dim=-1)
    # loss = -log sum_exp(logp[correct]) (one positive es lo tipico)
    pos = (logp * correct.float()).sum(dim=-1)            # suma de logp positivos
    n_pos = correct.float().sum(dim=-1).clamp_min(1.0)
    per_pos = -(pos / n_pos)
    per_pos = per_pos * has_pos.float()
    return per_pos.sum() / has_pos.float().sum().clamp_min(1)


def retrieval_cross_entropy_dense(
    scores: torch.Tensor,      # (B, M, N) scores compuestos (cualquier escala)
    valid: torch.Tensor,       # (B, M, N) bool, mascara causal (j <= i-gap)
    next_tokens: torch.Tensor,  # (B, N) long, token en j+1 de cada posicion j
    targets: torch.Tensor,     # (B, M) long, objetivo y_i en cada fila supervisada
    *, temperature: float = 1.0, ignore_index: int = -100,
) -> torch.Tensor:
    """CE denso auto-supervisado del Recall Tap (version interna RAPIDA).

    Igual objetivo que :func:`retrieval_cross_entropy` pero sin construir
    candidatos LSH: puntua cada fila supervisada ``i`` contra TODAS las
    posiciones previas ``j`` y empuja ``scores[i, j]`` hacia arriba donde el
    token siguiente de ``j`` (``next_tokens[j]``) coincide con ``y_i``. Vectorizado
    (matvec BLAS, sin bucles Python por candidato). Es la senal que entrena las
    proyecciones lexicas/de sentido DIRECTAMENTE y que activa el eje de sentido
    (LM pura no basta: el argmax duro bloquea el gradiente via STE debil).
    """
    b, m, n = scores.shape
    target_exp = targets.view(b, m, 1)
    correct = (next_tokens.unsqueeze(1) == target_exp) & valid
    correct = correct & (targets != ignore_index).view(b, m, 1)
    has_pos = correct.any(dim=-1)
    if not has_pos.any():
        return scores.new_zeros(())
    s = scores.float().masked_fill(~valid, -1e9) / max(1e-2, temperature)
    logp = F.log_softmax(s, dim=-1)
    pos = (logp * correct.float()).sum(dim=-1)
    n_pos = correct.float().sum(dim=-1).clamp_min(1.0)
    per_pos = -(pos / n_pos)
    per_pos = per_pos * has_pos.float()
    return per_pos.sum() / has_pos.float().sum().clamp_min(1)
