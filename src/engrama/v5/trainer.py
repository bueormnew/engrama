"""ENGRAMA V5 minimal, professional training loop.

Deliberately small and dependency-free (pure PyTorch). It covers the common
case: AdamW + cosine schedule + grad clipping + AMP + non-finite skipping.

Author: ENGRAMA contributors.
License: AGPL-3.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import torch

from engrama.v5.model import EngramaV5


@dataclass
class V5TrainConfig:
    lr: float = 3e-3
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 1000
    amp: bool = False
    log_every: int = 50


class V5Trainer:
    """Tiny trainer. ``batch_fn(step) -> (input_ids, targets)``."""

    def __init__(self, model: EngramaV5, cfg: Optional[V5TrainConfig] = None,
                 device: str = "cpu"):
        self.model = model.to(device)
        self.cfg = cfg or V5TrainConfig()
        self.device = device
        self.opt = torch.optim.AdamW(
            model.parameters(), lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay, betas=self.cfg.betas,
        )
        try:
            self.scaler = torch.amp.GradScaler(
                "cuda", enabled=self.cfg.amp and device == "cuda"
            )
        except (AttributeError, TypeError):  # pragma: no cover - older torch
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=self.cfg.amp and device == "cuda"
            )
        self._step = 0

    def _lr_at(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup_steps:
            return c.lr * step / max(1, c.warmup_steps)
        prog = (step - c.warmup_steps) / max(1, c.max_steps - c.warmup_steps)
        return 0.5 * c.lr * (1 + math.cos(math.pi * min(1.0, prog)))

    def train_step(self, input_ids: torch.Tensor, targets: torch.Tensor,
                   weight: Optional[torch.Tensor] = None) -> float:
        self.model.train()
        for g in self.opt.param_groups:
            g["lr"] = self._lr_at(self._step)
        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)
        use_amp = self.cfg.amp and self.device == "cuda"
        with torch.autocast(device_type="cuda", enabled=use_amp):
            if weight is None:
                loss = self.model.loss(input_ids, targets)
            else:
                logits = self.model(input_ids)
                raw = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, self.model.cfg.vocab_size),
                    targets.reshape(-1), reduction="none",
                ).view_as(targets)
                loss = (raw * weight.to(self.device)).mean()
        if not torch.isfinite(loss):
            self.opt.zero_grad(set_to_none=True)
            self._step += 1
            return float("nan")
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.cfg.grad_clip:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self._step += 1
        return float(loss.item())

    def fit(self, batch_fn: Callable[[int], Tuple[torch.Tensor, torch.Tensor]],
            steps: Optional[int] = None) -> None:
        steps = steps or self.cfg.max_steps
        for s in range(steps):
            ids, tgt = batch_fn(s)
            loss = self.train_step(ids, tgt)
            if s % self.cfg.log_every == 0:
                print(f"step {s:5d}  loss {loss:.4f}  lr {self._lr_at(self._step):.2e}")
