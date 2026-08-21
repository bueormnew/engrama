"""Memory-efficient loss helpers for large vocabularies.

The implementation below intentionally uses a custom autograd function.  A
Python/autograd graph containing one ``exp``, ``max`` and mask operation per
vocabulary chunk is surprisingly expensive (and retains every temporary until
backward).  :func:`chunked_cross_entropy` instead saves only the logits,
targets and one log-normalizer per token.  Backward streams over the vocabulary
again and constructs the exact softmax gradient directly.

This keeps peak *additional* memory at ``O(tokens * chunk_size)`` rather than
``O(tokens * vocab)`` and turns hundreds of tiny autograd operations into one
node.  It is useful when logits already exist.  For latent-fusion models that
must avoid materialising logits entirely, use
:meth:`engrama.model.EngramaModel.forward_loss` with ``linear_chunk_size``.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class _ChunkedCrossEntropy(torch.autograd.Function):
    """Streaming log-sum-exp with a hand-written streaming backward."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        chunk_size: int,
        ignore_index: Optional[int],
        reduction: str,
    ) -> torch.Tensor:
        flat = logits.reshape(-1, logits.size(-1))
        y = targets.reshape(-1)
        n, vocab = flat.shape
        valid = torch.ones(n, dtype=torch.bool, device=flat.device)
        if ignore_index is not None:
            valid = y != ignore_index
        safe_y = torch.where(valid, y, torch.zeros_like(y))
        if valid.any() and ((safe_y[valid] < 0).any() or (safe_y[valid] >= vocab).any()):
            raise IndexError("target id is outside the vocabulary")

        # The target score needs one gather, not one synchronising ``hit.any``
        # per vocabulary slice.
        selected = flat.gather(1, safe_y.unsqueeze(1)).squeeze(1).float()
        log_z = torch.full((n,), -float("inf"), dtype=torch.float32, device=flat.device)
        for v0 in range(0, vocab, chunk_size):
            chunk_lse = torch.logsumexp(flat[:, v0 : v0 + chunk_size].float(), dim=-1)
            log_z = torch.logaddexp(log_z, chunk_lse)

        losses = log_z - selected
        losses = torch.where(valid, losses, torch.zeros_like(losses))
        count = valid.sum().clamp_min(1)

        ctx.save_for_backward(logits, safe_y, log_z, valid)
        ctx.chunk_size = int(chunk_size)
        ctx.reduction = reduction
        ctx.count = count

        if reduction == "none":
            return losses.reshape(targets.shape)
        total = losses.sum()
        return total if reduction == "sum" else total / count

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        logits, safe_y, log_z, valid = ctx.saved_tensors
        flat = logits.reshape(-1, logits.size(-1))
        n, vocab = flat.shape
        grad = torch.empty_like(flat)

        if ctx.reduction == "none":
            multiplier = grad_output.reshape(-1).float()
        else:
            multiplier = grad_output.float().expand(n)
            if ctx.reduction == "mean":
                multiplier = multiplier / ctx.count
        multiplier = multiplier * valid.float()

        # d CE / d logit = softmax(logit) - one_hot(target).  Each output
        # vocabulary slice is written once, so no full fp32 softmax is kept.
        for v0 in range(0, vocab, ctx.chunk_size):
            v1 = min(vocab, v0 + ctx.chunk_size)
            g = torch.exp(flat[:, v0:v1].float() - log_z.unsqueeze(1))
            local = (safe_y - v0).clamp(0, v1 - v0 - 1)
            hit = valid & (safe_y >= v0) & (safe_y < v1)
            g.scatter_add_(1, local.unsqueeze(1), -hit.float().unsqueeze(1))
            g.mul_(multiplier.unsqueeze(1))
            grad[:, v0:v1] = g.to(flat.dtype)

        return grad.reshape_as(logits), None, None, None, None


def chunked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 4096,
    ignore_index: Optional[int] = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy streamed over the vocabulary dimension.

    It is numerically equivalent to ``F.cross_entropy`` (up to floating-point
    summation order), supports AMP logits, and creates only one autograd node.
    ``chunk_size`` controls temporary memory; 4096 is a good T4 default.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("reduction must be 'mean', 'sum' or 'none'")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"targets shape {tuple(targets.shape)} must match logits leading shape "
            f"{tuple(logits.shape[:-1])}"
        )
    return _ChunkedCrossEntropy.apply(logits, targets, chunk_size, ignore_index, reduction)


def _linear_ce_chunk(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    scale: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    logits = F.linear(hidden, weight) * scale
    # Native CE is considerably faster than a vocabulary-streaming loss when
    # the position chunk has been sized to fit memory.  fp32 keeps T4 AMP runs
    # stable for a 50k vocabulary.
    return F.cross_entropy(logits.float(), targets, ignore_index=ignore_index, reduction="sum")


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    scale: float = 1.0,
    chunk_size: int = 2048,
    ignore_index: int = -100,
    checkpoint_chunks: bool = True,
) -> torch.Tensor:
    """Fused linear projection + CE, chunked over token positions.

    No global ``(..., vocab)`` tensor is returned. With checkpointing, no
    vocabulary-logit chunk is retained either: backward recomputes one position
    chunk at a time, trading roughly one extra
    vocabulary GEMM for a drastic reduction in peak VRAM.  Set
    ``checkpoint_chunks=False`` when memory allows and maximum throughput is
    preferred.
    """
    flat_h = hidden.reshape(-1, hidden.size(-1))
    flat_y = targets.reshape(-1)
    if flat_h.size(0) != flat_y.numel():
        raise ValueError("hidden and targets contain a different number of positions")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    scale_t = flat_h.new_tensor(scale)
    n_tokens = flat_h.size(0)

    def _one(h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if checkpoint_chunks and torch.is_grad_enabled():
            return checkpoint(
                _linear_ce_chunk,
                h,
                weight,
                y,
                scale_t,
                ignore_index,
                use_reentrant=False,
            )
        return _linear_ce_chunk(h, weight, y, scale_t, ignore_index)

    # One shot when the position chunk covers the whole batch: avoids a Python
    # loop (and torch.compile graph breaks) of 4+ tiny vocabulary GEMMs.
    if n_tokens <= chunk_size:
        total = _one(flat_h, flat_y)
    else:
        total = flat_h.new_zeros((), dtype=torch.float32)
        for start in range(0, n_tokens, chunk_size):
            total = total + _one(
                flat_h[start : start + chunk_size],
                flat_y[start : start + chunk_size],
            )
    denominator = (flat_y != ignore_index).sum().clamp_min(1)
    return total / denominator
