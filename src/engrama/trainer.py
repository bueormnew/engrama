"""ENGRAMA Trainer Module.

High-level training engine with AdamW, gradient clipping, AMP (Automatic Mixed
Precision) and learning-rate schedules (constant, linear-warmup, or warmup +
cosine decay).

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from engrama.losses import chunked_cross_entropy
from engrama.model import EngramaModel
from engrama.optimization import adamw

# Above this vocabulary size the Trainer switches to chunked cross-entropy
_LARGE_VOCAB_THRESHOLD = 16384


def _cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy, memory-friendly for large vocabularies.

    Large vocabs go through :func:`chunked_cross_entropy`, which upcasts
    each vocabulary slice to fp32 (never a full ``(N, V)`` clone). Small
    vocabs use fused ``F.cross_entropy`` in fp32.
    """
    if logits.size(-1) > _LARGE_VOCAB_THRESHOLD:
        return chunked_cross_entropy(logits, targets)
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


class Trainer:
    """High-level training engine for ENGRAMA models.

    Args:
        model: Model instance to train.
        optimizer: Optional custom PyTorch optimizer.
        lr: Learning rate when the default AdamW is built. Default: 1e-3.
        device: Target device (``"cpu"``, ``"cuda"``, ...). Default: ``"cpu"``.
        gradient_clip: Global grad-norm clip (0 disables). Default: 1.0.
        weight_decay: AdamW weight decay. Default: 0.01.
        scheduler: ``"none"`` | ``"warmup"`` | ``"cosine"``. Default: ``"none"``.
        warmup_steps: Linear warmup steps for the chosen schedule. Default: 0.
        use_amp: Enable Automatic Mixed Precision (FP16 on CUDA). Default: False.
    """

    def __init__(
        self,
        model: EngramaModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-3,
        device: Union[str, torch.device] = "cpu",
        gradient_clip: float = 1.0,
        weight_decay: float = 0.01,
        scheduler: str = "none",
        warmup_steps: int = 0,
        use_amp: bool = False,
        fused_optimizer: Optional[bool] = None,
        fused_linear_loss: bool = True,
        linear_chunk_size: int = 2048,
        checkpoint_loss_chunks: bool = False,
        non_blocking: bool = True,
    ):
        if scheduler not in ("none", "warmup", "cosine"):
            raise ValueError("scheduler must be 'none', 'warmup' or 'cosine'")
        self.device = torch.device(device) if isinstance(device, str) else device
        self.gradient_clip = gradient_clip
        self.scheduler_name = scheduler
        self.warmup_steps = max(0, warmup_steps)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.non_blocking = bool(non_blocking and self.device.type == "cuda")
        self.fused_linear_loss = fused_linear_loss
        self.linear_chunk_size = linear_chunk_size
        self.checkpoint_loss_chunks = checkpoint_loss_chunks
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) if self.use_amp else None
        self.model = model.to(self.device)
        if optimizer is None:
            self.optimizer = adamw(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
                fused=(self.device.type == "cuda") if fused_optimizer is None else fused_optimizer,
            )
        else:
            self.optimizer = optimizer
        self.base_lr = lr
        self._global_step = 0

    # ------------------------------------------------------------------
    def _lr_factor(self, step: int, total_steps: Optional[int]) -> float:
        if self.scheduler_name == "none":
            return 1.0
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        if self.scheduler_name == "warmup":
            return 1.0
        if total_steps is None or total_steps <= self.warmup_steps:
            return 1.0
        progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def _apply_lr(self, total_steps: Optional[int]) -> float:
        factor = self._lr_factor(self._global_step, total_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr * factor
        return self.base_lr * factor

    @property
    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    # ------------------------------------------------------------------
    def _unpack_batch(
        self, batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            return (
                batch["input_ids"].to(self.device, non_blocking=self.non_blocking),
                batch["target_ids"].to(self.device, non_blocking=self.non_blocking),
            )
        return (
            batch[0].to(self.device, non_blocking=self.non_blocking),
            batch[1].to(self.device, non_blocking=self.non_blocking),
        )

    def _forward_loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        if (
            self.fused_linear_loss
            and self.model.config.vocab_size > _LARGE_VOCAB_THRESHOLD
            and self.model.evoker.aggregation in ("latent_fusion", "mean")
        ):
            return self.model.forward_loss(
                input_ids,
                target_ids,
                linear_chunk_size=self.linear_chunk_size,
                checkpoint_chunks=self.checkpoint_loss_chunks,
            )
        return _cross_entropy(self.model(input_ids), target_ids)

    def train_epoch(
        self, dataloader: DataLoader, total_steps: Optional[int] = None
    ) -> float:
        """Run a single training epoch; returns the mean loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            input_ids, target_ids = self._unpack_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    loss = self._forward_loss(input_ids, target_ids)
                self.scaler.scale(loss).backward()
                if self.gradient_clip and self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self._forward_loss(input_ids, target_ids)
                loss.backward()
                if self.gradient_clip and self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                self.optimizer.step()

            self._apply_lr(total_steps)
            self._global_step += 1
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def fit(
        self,
        dataset: Dataset,
        batch_size: int = 16,
        epochs: int = 10,
        shuffle: bool = True,
        callbacks: Optional[List[Callable[[int, float], Any]]] = None,
    ) -> List[float]:
        """Fit the model; returns the per-epoch loss history."""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        total_steps = len(dataloader) * epochs
        loss_history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = self.train_epoch(dataloader, total_steps=total_steps)
            loss_history.append(epoch_loss)
            if callbacks:
                for callback in callbacks:
                    callback(epoch, epoch_loss)

        return loss_history

    def evaluate(self, dataset: Dataset, batch_size: int = 16) -> float:
        """Evaluate mean cross-entropy loss on a dataset."""
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        try:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for batch in dataloader:
                    input_ids, target_ids = self._unpack_batch(batch)
                    logits = self.model(input_ids)
                    loss = _cross_entropy(logits, target_ids)
                    total_loss += loss.item()
                    num_batches += 1
        finally:
            if was_training:
                self.model.train()

        return total_loss / max(1, num_batches)
