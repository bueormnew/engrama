"""High-throughput training utilities for ENGRAMA.

The module deliberately keeps execution policy separate from architecture:
DDP, compilation, AMP and fused optimizers can be enabled or removed without
changing a model checkpoint or any ENGRAMA equation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Tuple

import torch
from torch import nn

from engrama.losses import chunked_cross_entropy


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    distributed: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(timeout_minutes: int = 30) -> DistributedContext:
    """Initialize ``torchrun``'s process group, or return a single-process context."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext()
    if not torch.cuda.is_available():
        raise RuntimeError("multi-process ENGRAMA training currently requires CUDA/NCCL")

    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=timeout_minutes))
    return DistributedContext(rank, local_rank, world_size, True)


def destroy_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def configure_cuda(*, deterministic: bool = False) -> None:
    """Set safe throughput-oriented CUDA backend flags."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = not deterministic
    # These flags affect Ampere+ only; they are harmless on Turing/T4.
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = not deterministic
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def compile_model(
    model: nn.Module,
    *,
    enabled: bool = True,
    mode: str = "max-autotune",
    fullgraph: bool = False,
) -> nn.Module:
    """Compile a model when PyTorch 2 is available, otherwise return it unchanged."""
    if not enabled or not hasattr(torch, "compile"):
        return model
    return torch.compile(model, mode=mode, fullgraph=fullgraph, dynamic=False)


def adamw(
    parameters,
    *,
    lr: float,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: Optional[bool] = None,
) -> torch.optim.AdamW:
    """Construct AdamW with its fused CUDA implementation when supported."""
    parameters = list(parameters)
    if fused is None:
        fused = bool(parameters and parameters[0].is_cuda)
    kwargs = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    if fused:
        try:
            return torch.optim.AdamW(parameters, fused=True, **kwargs)
        except (TypeError, ValueError, RuntimeError):
            pass
    return torch.optim.AdamW(parameters, **kwargs)


class LanguageModelLoss(nn.Module):
    """Put loss inside the module so DDP never communicates giant logits."""

    def __init__(
        self,
        model: nn.Module,
        *,
        linear_chunk_size: int = 2048,
        checkpoint_chunks: bool = True,
        use_fused_linear_loss: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.linear_chunk_size = linear_chunk_size
        self.checkpoint_chunks = checkpoint_chunks
        self.use_fused_linear_loss = use_fused_linear_loss

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.use_fused_linear_loss and hasattr(self.model, "forward_loss"):
            return self.model.forward_loss(
                input_ids,
                targets,
                linear_chunk_size=self.linear_chunk_size,
                checkpoint_chunks=self.checkpoint_chunks,
            )
        return chunked_cross_entropy(self.model(input_ids), targets)


def wrap_ddp(
    module: nn.Module,
    context: DistributedContext,
    *,
    static_graph: bool = True,
    gradient_as_bucket_view: bool = True,
) -> nn.Module:
    """Wrap a persistent model replica in DDP (never per-step replication)."""
    if not context.distributed:
        return module
    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(
        module,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
    )
