"""ENGRAMA Quick API -- the fastest path from text to a trained model.

Two usage modes of the library:

1. **Quick mode** (this module): pick a size preset, pass text, get a
   trained model. No architecture knowledge required::

       import engrama

       run = engrama.quickstart("corpus.txt", size="small", epochs=5)
       print(run.generate("Once upon a time"))
       run.save("./my_model")

       # or just instantiate:
       model = engrama.create_model(size="tiny", vocab_size=128)

2. **Expert mode**: full control through :class:`EngramaConfig` -- every
   architecture mode (version presets, synapse/cell/offset/cache/evoker
   modes, identity transport, hierarchical gates, global anchor, ranks...)
   is configurable, enabling the V3 ablation suites directly.

Author: BUEORM
License: AGPL-3.0
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from engrama.config import EngramaConfig
from engrama.datasets import TextDataset
from engrama.model import EngramaModel
from engrama.serialization import load_model, save_model
from engrama.tokenizer import EngramaTokenizer
from engrama.trainer import Trainer

_SIZE_PRESETS = ("tiny", "small", "base", "large")

# Sensible default learning rates per preset (empirically validated).
_DEFAULT_LR = {"tiny": 5e-3, "small": 3e-3, "base": 1e-3, "large": 3e-4}


def default_lr(size: str) -> float:
    """Return the recommended default learning rate for a size preset."""
    if size not in _DEFAULT_LR:
        raise ValueError(f"Unknown size preset {size!r}; choose from {_SIZE_PRESETS}")
    return _DEFAULT_LR[size]


def list_sizes() -> Dict[str, Dict[str, Any]]:
    """Describe the available size presets."""
    out: Dict[str, Dict[str, Any]] = {}
    for size in _SIZE_PRESETS:
        cfg = EngramaConfig.preset(size)
        out[size] = {
            "d_model": cfg.d_model,
            "num_cells": cfg.num_cells,
            "num_encoder_layers": cfg.num_encoder_layers,
            "num_consolidation_layers": cfg.num_consolidation_layers,
            "context_length": cfg.context_length,
            "synapse_rank": cfg.synapse_rank,
            "version": cfg.version,
        }
    return out


def create_model(
    size: str = "small",
    vocab_size: int = 256,
    context_length: Optional[int] = None,
    **overrides: Any,
) -> EngramaModel:
    """Create an ENGRAMA model from a size preset -- no config needed.

    Args:
        size: ``"tiny"`` | ``"small"`` | ``"base"`` | ``"large"``.
        vocab_size: Vocabulary size of the model.
        context_length: Optional context window override.
        **overrides: Any additional :class:`EngramaConfig` field, e.g.
            ``num_candidates=4`` or ``offset_mode="dense_dilated"``.

    Returns:
        A ready-to-train :class:`EngramaModel`.
    """
    if context_length is not None:
        overrides["context_length"] = context_length
    config = EngramaConfig.preset(size, vocab_size=vocab_size, **overrides)
    return EngramaModel(config)


@dataclass
class QuickRun:
    """Result bundle of :func:`quickstart` -- model, tokenizer and trainer."""

    model: EngramaModel
    tokenizer: EngramaTokenizer
    trainer: Trainer
    dataset: Optional[TextDataset] = None
    history: List[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def config(self) -> EngramaConfig:
        return self.model.config

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text completion for ``prompt``."""
        from engrama.inference import Generator

        generator = Generator(self.model, self.tokenizer)
        return generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs,
        )

    def stream(self, prompt: str, max_new_tokens: int = 50, **kwargs: Any):
        """Stream a completion character by character."""
        from engrama.inference import Generator

        generator = Generator(self.model, self.tokenizer)
        return generator.generate_stream(
            prompt, max_new_tokens=max_new_tokens, **kwargs
        )

    def evaluate(self, text: Union[str, os.PathLike], sequence_length: int = 128) -> float:
        """Cross-entropy loss of the model on new text."""
        dataset = TextDataset(text, self.tokenizer, sequence_length=sequence_length)
        return self.trainer.evaluate(dataset)

    def save(self, save_dir: str) -> None:
        """Save model + config + tokenizer to ``save_dir``."""
        save_model(self.model, save_dir, tokenizer=self.tokenizer)

    def summary(self) -> str:
        """Short training/architecture summary."""
        params = self.model.num_parameters()
        last_loss = f"{self.history[-1]:.4f}" if self.history else "n/a"
        return (
            f"ENGRAMA {self.config.version.upper()} | params={params:,} | "
            f"d_model={self.config.d_model} C={self.config.num_cells} "
            f"L={self.config.num_consolidation_layers} "
            f"N_max={self.config.context_length} | final_loss={last_loss}"
        )


def quickstart(
    data: Union[str, os.PathLike],
    size: str = "small",
    epochs: int = 5,
    batch_size: int = 16,
    lr: Optional[float] = None,
    sequence_length: Optional[int] = None,
    device: Optional[str] = None,
    seed: Optional[int] = 42,
    callbacks: Optional[List[Callable[[int, float], Any]]] = None,
    verbose: bool = True,
    **config_overrides: Any,
) -> QuickRun:
    """Train an ENGRAMA model on a text file or string in a single call.

    Args:
        data: Path to a text file, or a raw text string.
        size: Size preset (``"tiny"`` | ``"small"`` | ``"base"`` | ``"large"``).
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate (preset-aware default when omitted).
        sequence_length: Training window (defaults to the preset context).
        device: ``"cpu"`` | ``"cuda"`` (auto-detected when omitted).
        seed: Torch seed for reproducibility (``None`` disables).
        callbacks: Optional ``callback(epoch, loss)`` hooks per epoch.
        verbose: Print progress lines.
        **config_overrides: Extra :class:`EngramaConfig` overrides.

    Returns:
        A :class:`QuickRun` with the trained model, tokenizer and history.
    """
    if seed is not None:
        torch.manual_seed(seed)

    text_str = str(data)
    if os.path.isfile(text_str):
        with open(text_str, "r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        raw_text = text_str
    if not raw_text:
        raise ValueError("quickstart received empty training data")

    tokenizer = EngramaTokenizer().fit_on_text(raw_text)

    config = EngramaConfig.preset(size, vocab_size=tokenizer.vocab_size, **config_overrides)
    model = EngramaModel(config)

    seq_len = sequence_length or min(128, config.context_length)
    dataset = TextDataset(raw_text, tokenizer, sequence_length=seq_len)
    if len(dataset) < 1:
        raise ValueError("Training data produced zero samples; pass more text.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    learn_rate = lr if lr is not None else default_lr(size)

    trainer = Trainer(model, lr=learn_rate, device=device)

    if verbose:
        print(
            f"[ENGRAMA quickstart] size={size} params={model.num_parameters():,} "
            f"device={device} samples={len(dataset)} seq_len={seq_len} "
            f"lr={learn_rate:g} epochs={epochs}"
        )

    history = trainer.fit(
        dataset,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
    )
    if verbose:
        print(
            f"[ENGRAMA quickstart] done: loss {history[0]:.4f} -> {history[-1]:.4f}"
        )

    return QuickRun(
        model=model,
        tokenizer=tokenizer,
        trainer=trainer,
        dataset=dataset,
        history=history,
    )


def load_quick(load_dir: str, device: Union[str, torch.device] = "cpu") -> QuickRun:
    """Load a bundle saved with :meth:`QuickRun.save`."""
    model, tokenizer = load_model(load_dir, device=device)
    if tokenizer is None:
        tokenizer = EngramaTokenizer()
    trainer = Trainer(model, device=device)
    return QuickRun(model=model, tokenizer=tokenizer, trainer=trainer)
