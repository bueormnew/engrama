"""Memory-friendly loss helpers for large vocabularies.

``F.cross_entropy`` internally materializes a second copy of the logits
tensor for ``log_softmax``. With a GPT-2-sized vocabulary (50,257) and
training shapes like (32, 512), that copy alone is ~3.3 GB in fp32 and is
a common source of CUDA OOM on 16 GB cards -- even though the model itself
may only have a few million parameters.

:func:`chunked_cross_entropy` computes exactly the same loss by streaming
over the vocabulary in chunks: each position keeps a running log-sum-exp
normalizer (one scalar per position) and the logit of its target token.
Peak memory is ``O(N * chunk_size)`` instead of ``O(N * V)``, with the same
values as ``F.cross_entropy`` up to floating-point summation order.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""

from __future__ import annotations

from typing import Optional

import torch


def chunked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 2048,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy over the vocabulary axis, processed in chunks.

    Equivalent to::

        F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                        ignore_index=ignore_index, reduction=reduction)

    but avoids materializing the full ``log_softmax`` copy of the logits.

    Args:
        logits: Logits of shape ``(..., V)`` (any leading dims are
            flattened together).
        targets: Long ids of shape ``(...,)`` matching the leading dims of
            ``logits``.
        chunk_size: Number of vocabulary entries processed at once.
        ignore_index: Target value whose positions are excluded from the
            loss (``None`` disables masking; only ``-100`` is supported as
            a mask value for numerical output).
        reduction: ``"mean"`` | ``"sum"`` | ``"none"`` (``"none"`` returns
            one value per flattened position; ignored positions are -inf).

    Returns:
        Scalar loss for ``"mean"``/``"sum"``, or a ``(N,)`` tensor for
        ``"none"``.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("reduction must be 'mean', 'sum' or 'none'")

    flat = logits.reshape(-1, logits.size(-1))
    n, vocab = flat.shape
    flat_targets = targets.reshape(-1)

    m_run = flat.new_full((n,), -float("inf"))
    s_run = flat.new_zeros((n,))
    selected = flat.new_full((n,), -float("inf"))
    valid = torch.ones(n, dtype=torch.bool, device=flat.device)
    if ignore_index is not None:
        valid &= flat_targets != ignore_index

    for v0 in range(0, vocab, chunk_size):
        v1 = min(vocab, v0 + chunk_size)
        l_chunk = flat[:, v0:v1]  # (N, c) view -- no copy
        m_chunk = l_chunk.max(dim=-1).values  # (N,)
        m_new = torch.maximum(m_run, m_chunk)  # (N,)
        # Running log-sum-exp over chunks (numerically stable).
        s_run = s_run * torch.exp(m_run - m_new) + torch.exp(
            l_chunk - m_new.unsqueeze(-1)
        ).sum(dim=-1)
        m_run = m_new
        # Collect the logit of the target token when it falls in this chunk.
        hit = valid & (flat_targets >= v0) & (flat_targets < v1)
        if hit.any():
            selected[hit] = l_chunk[hit, flat_targets[hit] - v0]

    loss = (m_run + torch.log(s_run)) - selected  # (N,)

    if reduction == "none":
        return loss
    if not valid.any():
        return flat.new_zeros(())
    reduced = loss[valid].sum()
    if reduction == "sum":
        return reduced
    return reduced / valid.sum()
