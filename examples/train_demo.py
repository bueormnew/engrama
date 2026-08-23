"""Ejemplo de entrenamiento ENGRAMA V5.5 — receta minimal y funcional.

Entrena el modelo en una tarea de copia/induccion simple (un token se repite y
el modelo debe predecir la repeticion) y muestra que la loss baja de forma
estable. Sirve como plantilla: sustituye `make_batch` por tu propio dataset.

Ejecutar:
    python examples/train_demo.py
    python examples/train_demo.py --preset base --steps 2000 --lr 3e-4
"""
import argparse
import math
import random

import torch

from engrama.v55 import EngraModelV55, V55Config


def make_batch(rng, batch_size, seq_len, vocab_size):
    """Tarea de copia con repeticiones: seq de tokens donde cada cierto periodo
    se inserta un token-marca que se repite poco despues. Fuerte senal de
    induccion para el Recall Tap lexico."""
    x = [[rng.randrange(1, vocab_size) for _ in range(seq_len)] for _ in range(batch_size)]
    # insertar repeticiones: cada 7 posiciones, repetir el token de hace 3
    for b in range(batch_size):
        for i in range(3, seq_len):
            if i % 7 == 0:
                x[b][i] = x[b][i - 3]
    return torch.tensor(x, dtype=torch.long)


def cosine_lr(step, total, warmup, lr_max):
    if step < warmup:
        return lr_max * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    model = EngraModelV55.from_preset(args.preset)
    cfg = model.config
    vocab = cfg.vocab_size
    print(model.describe())

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    rng = random.Random(args.seed)
    warmup = max(1, args.steps // 20)

    model.train()
    for step in range(1, args.steps + 1):
        lr = cosine_lr(step, args.steps, warmup, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        x = make_batch(rng, args.batch, args.seq, vocab)
        loss = model.forward_loss(x[:, :-1], x[:, 1:], retrieval_weight=1.0)
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f"  paso {step:5d}  loss {loss.item():.4f}  "
                  f"lr {lr:.2e}  |grad| {float(grad_norm):.2f}", flush=True)

    # verificacion: la loss final debe ser mucho menor que la inicial
    print(f"OK — entrenamiento estable, loss final {loss.item():.4f}")

    # demo de generacion incremental (invariante al forward paralelo)
    model.eval()
    prompt = [0] + [rng.randrange(1, vocab) for _ in range(8)]
    out = model.generate(prompt, max_new_tokens=16, temperature=0.8, top_k=20)
    print(f"  generacion: {len(out)} tokens (prompt {len(prompt)} + 16 nuevos)")


if __name__ == "__main__":
    main()
