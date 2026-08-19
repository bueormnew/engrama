"""
ENGRAMA Trainer Module
Author: BUEORM
License: AGPL-3.0
"""

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from engrama.model import EngramaModel


class Trainer:
    """High-level Training Engine for ENGRAMA models.

    Args:
        model (EngramaModel): Model instance to train.
        optimizer (Optional[Optimizer]): PyTorch optimizer instance.
        lr (float): Learning rate if default AdamW is constructed. Default: 1e-3.
        device (Union[str, torch.device]): Device target ('cuda', 'cpu', etc.). Default: 'cpu'.
        gradient_clip (float): Maximum norm for gradient clipping. Default: 1.0.
    """

    def __init__(
        self,
        model: EngramaModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-3,
        device: Union[str, torch.device] = "cpu",
        gradient_clip: float = 1.0,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.gradient_clip = gradient_clip
        self.model = model.to(self.device)
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run single training epoch over dataloader."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
            else:
                input_ids, target_ids = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
            )
            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()
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
        """Fit model on dataset over specified epochs."""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        loss_history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = self.train_epoch(dataloader)
            loss_history.append(epoch_loss)
            if callbacks:
                for callback in callbacks:
                    callback(epoch, epoch_loss)

        return loss_history

    def evaluate(self, dataset: Dataset, batch_size: int = 16) -> float:
        """Evaluate model loss on validation dataset."""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(self.device)
                    target_ids = batch["target_ids"].to(self.device)
                else:
                    input_ids, target_ids = batch[0].to(self.device), batch[1].to(self.device)

                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                )
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(1, num_batches)
