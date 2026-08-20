"""ENGRAMA Trainer Module.

High-level training engine with AdamW, gradient clipping and an optional
learning-rate schedule (constant, linear-warmup, or warmup + cosine decay,
as recommended for V3 -- see paper section 8 and V3 spec section 32).

Author: BUEORM
License: AGPL-3.0
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from engrama.model import EngramaModel


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
    ):
        if scheduler not in ("none", "warmup", "cosine"):
            raise ValueError("scheduler must be 'none', 'warmup' or 'cosine'")
        self.device = torch.device(device) if isinstance(device, str) else device
        self.gradient_clip = gradient_clip
        self.scheduler_name = scheduler
        self.warmup_steps = max(0, warmup_steps)
        self.model = model.to(self.device)
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
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
            return batch["input_ids"].to(self.device), batch["target_ids"].to(self.device)
        return batch[0].to(self.device), batch[1].to(self.device)

    def train_epoch(
        self, dataloader: DataLoader, total_steps: Optional[int] = None
    ) -> float:
        """Run a single training epoch; returns the mean loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            input_ids, target_ids = self._unpack_batch(batch)
            self.optimizer.zero_grad()
            logits = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), target_ids.view(-1)
            )
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
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids, target_ids = self._unpack_batch(batch)
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), target_ids.view(-1)
                )
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(1, num_batches)
