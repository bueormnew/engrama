#!/usr/bin/env python3
"""Production-style 1/2+ GPU ENGRAMA TinyStories trainer.

Prepare flat GPT-2 token-id files as little-endian int32 (the Kaggle notebook
already creates these), then launch:

    torchrun --standalone --nproc_per_node=2 examples/train_tinystories_ddp.py \
        --train tinystories_train.ids --valid tinystories_valid.ids

Unlike ``nn.DataParallel``, every process owns a persistent model replica and
reads a disjoint data shard.  Only gradient buckets cross GPUs.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from engrama import (
    EngramaConfig,
    EngramaModel,
    LanguageModelLoss,
    adamw,
    compile_model,
    configure_cuda,
    destroy_distributed,
    init_distributed,
    wrap_ddp,
)


class TokenWindows(Dataset):
    """Non-overlapping zero-copy-on-disk windows over a flat int32 memmap."""

    def __init__(self, path: str, sequence_length: int):
        num_tokens = os.path.getsize(path) // 4
        self.tokens = torch.from_file(
            path, shared=False, size=num_tokens, dtype=torch.int32
        )
        self.sequence_length = sequence_length
        self.window = sequence_length + 1
        self.n = len(self.tokens) // self.window

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        start = index * self.window
        # One contiguous int32 -> int64 conversion, then overlapping views.
        values = self.tokens[start : start + self.window].to(torch.int64)
        return values[:-1], values[1:]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--output", default="engrama_v4_20m_gpt2")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--batch-size", type=int, default=16, help="per-GPU batch")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=25)
    p.add_argument("--linear-chunk-size", type=int, default=2048)
    p.add_argument("--checkpoint-loss", action="store_true", help="minimum VRAM, ~extra projection compute")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--compile-mode", default="max-autotune", choices=("default", "reduce-overhead", "max-autotune"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return value


@torch.inference_mode()
def evaluate(model, loader, device, amp, max_batches, world_size):
    model.eval()
    total = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = model(x, y)
        total += loss.detach()
        count += 1
    if world_size > 1:
        dist.all_reduce(total)
        dist.all_reduce(count)
    model.train()
    return (total / count.clamp_min(1)).item()


def main():
    args = arguments()
    ctx = init_distributed()
    configure_cuda()
    device = torch.device("cuda", ctx.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    amp = device.type == "cuda"
    torch.manual_seed(args.seed + ctx.rank)

    config = EngramaConfig(
        vocab_size=args.vocab_size,
        context_length=args.seq_len,
        d_model=256,
        d_gate=32,
        d_ff=1024,
        num_cells=8,
        num_encoder_layers=2,
        num_consolidation_layers=9,
        num_candidates=4,
        candidate_aggregation="latent_fusion",
        synapse_rank=32,
        version="v4",
        offset_mode="resonant_multirate",
        gating_mode="dual",
        trace_tap=True,
        norm_type="rmsnorm",
    )
    raw_model = EngramaModel(config).to(device)
    optimizer = adamw(
        raw_model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95), fused=amp,
    )
    loss_model = LanguageModelLoss(
        raw_model,
        linear_chunk_size=args.linear_chunk_size,
        checkpoint_chunks=args.checkpoint_loss,
        use_fused_linear_loss=True,
    )
    # Compile before DDP: graph capture stays local and DDP hooks stay outside.
    loss_model = compile_model(
        loss_model, enabled=not args.no_compile and amp, mode=args.compile_mode
    )
    train_model = wrap_ddp(loss_model, ctx, static_graph=True)

    train_ds = TokenWindows(args.train, args.seq_len)
    valid_ds = TokenWindows(args.valid, args.seq_len)
    train_sampler = DistributedSampler(
        train_ds, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True,
        seed=args.seed, drop_last=True,
    ) if ctx.distributed else None
    valid_sampler = DistributedSampler(
        valid_ds, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=False,
    ) if ctx.distributed else None
    loader_kw = dict(num_workers=args.workers, pin_memory=amp)
    if args.workers > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=train_sampler is None, drop_last=True, **loader_kw,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.eval_batch_size, sampler=valid_sampler,
        shuffle=False, drop_last=False, **loader_kw,
    )

    output = Path(args.output)
    if ctx.is_main:
        output.mkdir(parents=True, exist_ok=True)
        config.save(str(output / "config.json"))
    if ctx.distributed:
        dist.barrier()
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=2**12, growth_interval=2000)
    start_step, best = 0, float("inf")
    state_path = output / "trainer_state.pt"
    if args.resume and state_path.exists():
        checkpoint = torch.load(state_path, map_location=device)
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step, best = checkpoint["step"], checkpoint["best"]

    steps_per_epoch = len(train_loader)
    total_steps = start_step + steps_per_epoch * args.epochs
    if ctx.is_main:
        print(f"DDP={ctx.world_size} GPU(s) | local batch={args.batch_size} | "
              f"global batch={args.batch_size * ctx.world_size} | steps={total_steps}")
        print(f"parameters={raw_model.num_parameters():,} | compile={not args.no_compile and amp} | "
              f"checkpoint_loss={args.checkpoint_loss}")

    def learning_rate(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    step, started = start_step, time.perf_counter()
    train_model.train()
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for x, y in train_loader:
            lr = learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = train_model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip, foreach=True)
            # GradScaler already performs a fused non-finite check.  Do not scan
            # every parameter and synchronise Python with CUDA a second time.
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if step % args.log_every == 0:
                logged = reduce_mean(loss.detach(), ctx.world_size).item()
                if ctx.is_main:
                    elapsed = time.perf_counter() - started
                    print(f"step {step:6d}/{total_steps} | loss {logged:.4f} | "
                          f"lr {lr:.2e} | {elapsed / (step-start_step):.3f}s/step")

            if step % args.eval_every == 0 or step == total_steps:
                val = evaluate(train_model, valid_loader, device, amp, args.eval_batches, ctx.world_size)
                if ctx.is_main:
                    print(f"  [eval] step {step}: loss={val:.4f} ppl={math.exp(min(20, val)):.2f}")
                    if val < best:
                        best = val
                        payload = dict(
                            model=raw_model.state_dict(), optimizer=optimizer.state_dict(),
                            scaler=scaler.state_dict(), step=step, best=best,
                        )
                        torch.save(payload, state_path)
                        torch.save(raw_model.state_dict(), output / "best_model.pt")
        if step >= total_steps:
            break

    if ctx.is_main:
        torch.save(raw_model.state_dict(), output / "model.pt")
        print(f"done in {(time.perf_counter()-started)/60:.1f} min | best={best:.4f}")
    destroy_distributed()


if __name__ == "__main__":
    main()
