"""ENGRAMA V5 quickstart — train a tiny model and generate.

Run::

    python examples/v5_quickstart.py
"""

from __future__ import annotations

import torch

from engrama.v5 import EngramaV5, EngramaV5Config, V5Trainer
from engrama.v5.trainer import V5TrainConfig


def main() -> None:
    torch.manual_seed(0)

    # 1. Configure a small model (no attention, no compression).
    cfg = EngramaV5Config(
        vocab_size=256,
        d_model=128,
        num_layers=4,
        num_heads=4,
        context_length=512,
    )
    model = EngramaV5(cfg)
    print(model.describe())

    # 2. A toy "copy the first token later" dataset.
    def batch_fn(step: int):
        ids = torch.randint(0, 256, (16, 64))
        ids[:, 32] = ids[:, 0]              # position 32 must recall token 0
        targets = ids.clone()
        targets[:, :-1] = ids[:, 1:]
        return ids, targets

    # 3. Train.
    trainer = V5Trainer(model, V5TrainConfig(max_steps=200, log_every=50, lr=3e-3))
    trainer.fit(batch_fn, steps=200)

    # 4. Generate with the native explicit-trace cache.
    out = model.generate([1, 2, 3, 4], max_new_tokens=20, temperature=0.8, top_k=20)
    print("generated:", out)


if __name__ == "__main__":
    main()
