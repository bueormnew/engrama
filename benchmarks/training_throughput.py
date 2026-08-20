#!/usr/bin/env python3
"""Reproducible ENGRAMA training-step throughput/VRAM benchmark.

Examples:
  python benchmarks/training_throughput.py --profile optimized --steps 100
  python benchmarks/training_throughput.py --profile baseline --steps 100
"""

import argparse
import time

import torch

from engrama import EngramaConfig, EngramaModel, adamw, chunked_cross_entropy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=("baseline", "optimized", "checkpoint"), default="optimized")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--chunk-size", type=int, default=2048)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA")
    device = torch.device("cuda")
    cfg = EngramaConfig(
        vocab_size=args.vocab_size, context_length=args.seq_len,
        d_model=256, d_gate=32, d_ff=1024, num_cells=8,
        num_encoder_layers=2, num_consolidation_layers=9,
        num_candidates=4, candidate_aggregation="latent_fusion",
        synapse_rank=32, version="v4",
    )
    model = EngramaModel(cfg).to(device)
    optimizer = adamw(model.parameters(), lr=3e-4, fused=True)
    scaler = torch.cuda.amp.GradScaler(init_scale=2**12)
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            if args.profile == "baseline":
                loss = chunked_cross_entropy(model(x), y)
            else:
                loss = model.forward_loss(
                    x, y, linear_chunk_size=args.chunk_size,
                    checkpoint_chunks=args.profile == "checkpoint",
                )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    seconds = elapsed / args.steps
    tokens = args.batch_size * args.seq_len
    print(f"profile={args.profile} steps={args.steps} sec/step={seconds:.4f} "
          f"tokens/s={tokens / seconds:,.0f} "
          f"peak_allocated={torch.cuda.max_memory_allocated()/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
