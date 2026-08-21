#!/usr/bin/env python3
"""DDP trainer for the 2x T4 ENGRAMA-vs-Transformer comparison notebook.

Trains one architecture replica per process. Architecture equations are
unchanged; this file only owns execution (DDP, AMP, fused CE, compile).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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
    linear_cross_entropy,
    wrap_ddp,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TokenWindows(Dataset):
    """Non-overlapping windows over a flat little-endian int32 token file."""

    def __init__(self, path: str, sequence_length: int, max_tokens: int | None = None):
        n = os.path.getsize(path) // 4
        if max_tokens is not None:
            n = min(n, int(max_tokens))
        self.tokens = torch.from_file(path, shared=False, size=n, dtype=torch.int32)
        self.sequence_length = sequence_length
        self.window = sequence_length + 1
        self.n = len(self.tokens) // self.window

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        start = index * self.window
        values = self.tokens[start : start + self.window].to(torch.int64)
        return values[:-1], values[1:]

    def __getitems__(self, indices):
        """Vectorized window gather so DataLoader workers do one index, not B."""
        starts = torch.as_tensor(indices, dtype=torch.long).unsqueeze(1) * self.window
        gather = starts + torch.arange(self.window, dtype=torch.long)
        values = self.tokens[gather.reshape(-1)].to(torch.int64).view(len(indices), self.window)
        return [(values[i, :-1], values[i, 1:]) for i in range(values.size(0))]


# ---------------------------------------------------------------------------
# Transformer baseline (RoPE GPT, tied embeddings, RMSNorm)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        x32 = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        y = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(out_dtype)


def _rope_cache(head_dim: int, seq_len: int, device, dtype, theta: float = 10000.0):
    half = head_dim // 2
    freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freq)  # (T, d/2)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D) with even D; rotate (x1, x2) pairs in the last dim.
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    cos = cos[: x.size(-2)].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.size(-2)].unsqueeze(0).unsqueeze(0)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.stack((rot1, rot2), dim=-1)
    return out.flatten(-2)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = _apply_rope(q.transpose(1, 2), cos, sin)
        k = _apply_rope(k.transpose(1, 2), cos, sin)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.drop(self.proj(attn))
        h = self.ln2(x)
        x = x + self.drop(self.fc2(F.gelu(self.fc1(h))))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer sized to match ENGRAMA (~20M, GPT-2 vocab)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 9,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self._rope_len = 0
        self._rope_key = None
        self._cos = None
        self._sin = None
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_parameters(self, only_trainable: bool = False) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        if t > self.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len={self.max_seq_len}")
        x = self.drop(self.tok_emb(input_ids))
        key = (t, x.device, x.dtype)
        if self._rope_key != key:
            self._cos, self._sin = _rope_cache(self.head_dim, t, x.device, x.dtype)
            self._rope_len = t
            self._rope_key = key
        for block in self.blocks:
            x = block(x, self._cos, self._sin)
        return self.ln_f(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.linear(self.forward_features(input_ids), self.tok_emb.weight)

    def forward_loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        *,
        linear_chunk_size: int = 2048,
        checkpoint_chunks: bool = False,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        hidden = self.forward_features(input_ids)
        return linear_cross_entropy(
            hidden,
            self.tok_emb.weight,
            targets,
            scale=1.0,
            chunk_size=linear_chunk_size,
            ignore_index=ignore_index,
            checkpoint_chunks=checkpoint_chunks,
        )


# ---------------------------------------------------------------------------
# Architecture factory
# ---------------------------------------------------------------------------
ENGRAMA_BASE = dict(
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
    tie_embeddings=True,
    stable_init=True,
)

ARCH_SPECS = {
    "engrama_v4": {
        "kind": "engrama",
        "title": "ENGRAMA V4 completo",
        "overrides": {},
    },
    "engrama_source_gate": {
        "kind": "engrama",
        "title": "ENGRAMA sin dual gating (source-only)",
        "overrides": {"gating_mode": "source"},
    },
    "engrama_no_tracetap": {
        "kind": "engrama",
        "title": "ENGRAMA sin Trace Tap T0",
        "overrides": {"trace_tap": False},
    },
    "transformer": {
        "kind": "transformer",
        "title": "Transformer decoder (RoPE, RMSNorm)",
        "overrides": {},
    },
}


def build_raw_model(arch: str, vocab_size: int, seq_len: int) -> nn.Module:
    if arch not in ARCH_SPECS:
        raise ValueError(f"unknown arch {arch!r}; choose from {tuple(ARCH_SPECS)}")
    spec = ARCH_SPECS[arch]
    if spec["kind"] == "engrama":
        kw = dict(ENGRAMA_BASE)
        kw.update(spec["overrides"])
        cfg = EngramaConfig(vocab_size=vocab_size, context_length=seq_len, **kw)
        return EngramaModel(cfg)
    return TransformerLM(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=9,
        n_heads=8,
        d_ff=1024,
        max_seq_len=max(2048, seq_len),
    )


def model_card(arch: str, model: nn.Module) -> dict:
    spec = ARCH_SPECS[arch]
    card = {
        "arch": arch,
        "title": spec["title"],
        "kind": spec["kind"],
        "parameters": int(model.num_parameters()),
        "overrides": spec["overrides"],
    }
    if spec["kind"] == "engrama":
        cfg = model.config
        card.update(
            {
                "version": cfg.version,
                "d_model": cfg.d_model,
                "num_cells": cfg.num_cells,
                "num_encoder_layers": cfg.num_encoder_layers,
                "num_consolidation_layers": cfg.num_consolidation_layers,
                "gating_mode": cfg.gating_mode,
                "trace_tap": bool(cfg.trace_tap),
                "offset_mode": cfg.offset_mode,
                "norm_type": cfg.norm_type,
                "candidate_aggregation": cfg.candidate_aggregation,
                "receptive_field": cfg.receptive_field(),
            }
        )
    else:
        card.update(
            {
                "d_model": model.d_model,
                "n_layers": model.n_layers,
                "n_heads": model.n_heads,
                "positional": "rope",
                "attention": "causal_sdpa",
                "norm_type": "rmsnorm",
            }
        )
    return card


# ---------------------------------------------------------------------------
# CLI / training
# ---------------------------------------------------------------------------
def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True, choices=tuple(ARCH_SPECS))
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--batch-size", type=int, default=16, help="per-GPU batch")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=25)
    p.add_argument("--linear-chunk-size", type=int, default=8192)
    p.add_argument("--max-train-tokens", type=int, default=None)
    p.add_argument("--max-valid-tokens", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=0, help="0 = full epoch(s)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--compile-mode", default="reduce-overhead",
                   choices=("default", "reduce-overhead", "max-autotune"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint-loss", action="store_true")
    return p.parse_args()


def planned_total_steps(steps_per_epoch: int, epochs: int, max_steps: int = 0) -> int:
    """Absolute last step (not ``start_step + remaining``). Resume must not retrain."""
    planned = int(steps_per_epoch) * int(epochs)
    if max_steps and int(max_steps) > 0:
        planned = min(planned, int(max_steps))
    return planned


def make_grad_scaler(amp: bool):
    try:
        return torch.amp.GradScaler(
            "cuda", enabled=amp, init_scale=2**12, growth_interval=2000
        )
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler(
            enabled=amp, init_scale=2**12, growth_interval=2000
        )


def shutdown_loader(loader) -> None:
    """Join DataLoader workers so torchrun can exit (persistent_workers hang otherwise)."""
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        try:
            iterator._shutdown_workers()
        except Exception:
            pass


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
        if not torch.isfinite(loss):
            continue
        total += loss.detach()
        count += 1
    if world_size > 1:
        dist.all_reduce(total)
        dist.all_reduce(count)
    model.train()
    return (total / count.clamp_min(1)).item()


def save_payload(raw_model, output: Path, card: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save(raw_model.state_dict(), output / "model.pt")
    if hasattr(raw_model, "config"):
        raw_model.config.save(str(output / "config.json"))
    else:
        with open(output / "config.json", "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2)
    with open(output / "arch.json", "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)


def main():
    args = arguments()
    # Kaggle 2x T4 often has flaky P2P/IB; disable to keep NCCL stable.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    ctx = init_distributed()
    configure_cuda()
    if torch.cuda.is_available():
        mm = torch.backends.cuda.matmul
        if hasattr(mm, "allow_fp16_reduced_precision_reduction"):
            mm.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            # T4 (SM75) has no FlashAttention; keep mem-efficient + math.
            torch.backends.cuda.enable_flash_sdp(False)

    device = (
        torch.device("cuda", ctx.local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    amp = device.type == "cuda"
    torch.manual_seed(args.seed + ctx.rank)

    raw_model = build_raw_model(args.arch, args.vocab_size, args.seq_len).to(device)
    card = model_card(args.arch, raw_model)
    optimizer = adamw(
        raw_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        fused=amp,
    )
    loss_model = LanguageModelLoss(
        raw_model,
        linear_chunk_size=args.linear_chunk_size,
        checkpoint_chunks=args.checkpoint_loss,
        use_fused_linear_loss=True,
    )
    compiled = False
    compile_mode = args.compile_mode
    if not args.no_compile and amp:
        os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
        try:
            loss_model = compile_model(loss_model, enabled=True, mode=compile_mode)
            compiled = True
        except Exception as exc:
            if compile_mode != "default":
                if ctx.is_main:
                    print(
                        f"[compile] {compile_mode} failed ({type(exc).__name__}); "
                        "falling back to default",
                        flush=True,
                    )
                try:
                    loss_model = compile_model(loss_model, enabled=True, mode="default")
                    compiled = True
                    compile_mode = "default"
                except Exception as exc2:
                    if ctx.is_main:
                        print(f"[compile] disabled ({type(exc2).__name__}: {exc2})", flush=True)
                    compiled = False
            else:
                if ctx.is_main:
                    print(f"[compile] disabled ({type(exc).__name__}: {exc})", flush=True)
                compiled = False
    train_model = wrap_ddp(loss_model, ctx, static_graph=True)

    train_ds = TokenWindows(args.train, args.seq_len, args.max_train_tokens)
    valid_ds = TokenWindows(args.valid, args.seq_len, args.max_valid_tokens)
    train_sampler = (
        DistributedSampler(
            train_ds,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        if ctx.distributed
        else None
    )
    valid_sampler = (
        DistributedSampler(
            valid_ds,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=False,
        )
        if ctx.distributed
        else None
    )
    loader_kw = dict(num_workers=args.workers, pin_memory=amp)
    if args.workers > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        **loader_kw,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.eval_batch_size,
        sampler=valid_sampler,
        shuffle=False,
        drop_last=False,
        **loader_kw,
    )

    output = Path(args.output)
    if ctx.is_main:
        output.mkdir(parents=True, exist_ok=True)
        save_payload(raw_model, output, card)

    if ctx.distributed:
        dist.barrier()

    scaler = make_grad_scaler(amp)
    start_step, best = 0, float("inf")
    state_path = output / "trainer_state.pt"
    if args.resume and state_path.exists():
        checkpoint = torch.load(state_path, map_location=device)
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step, best = int(checkpoint["step"]), float(checkpoint["best"])

    steps_per_epoch = len(train_loader)
    total_steps = planned_total_steps(steps_per_epoch, args.epochs, args.max_steps)
    tokens_per_step = args.batch_size * ctx.world_size * args.seq_len

    if ctx.is_main:
        print(
            f"arch={args.arch} params={card['parameters']:,} DDP={ctx.world_size} "
            f"local_batch={args.batch_size} global_batch={args.batch_size * ctx.world_size} "
            f"seq={args.seq_len} steps={total_steps} start_step={start_step} "
            f"compile={compiled}/{compile_mode} amp={amp} ce_chunk={args.linear_chunk_size}",
            flush=True,
        )
        print(f"train_windows={len(train_ds):,} valid_windows={len(valid_ds):,}", flush=True)
        if start_step >= total_steps:
            print(
                f"already finished {args.arch} at step {start_step}/{total_steps}; "
                "writing metrics and exiting",
                flush=True,
            )

    def learning_rate(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    history = []
    skipped = 0
    step, started = start_step, time.perf_counter()
    tokens_seen = 0
    steady_started = None
    peak_gb = 0.0
    last_loss = float("nan")
    last_val = float("nan")
    mark_cudagraph = hasattr(torch, "compiler") and hasattr(
        torch.compiler, "cudagraph_mark_step_begin"
    )
    train_model.train()

    if amp:
        torch.cuda.reset_peak_memory_stats(device)

    stop = False
    if start_step >= total_steps:
        stop = True
    for epoch in range(args.epochs):
        if stop:
            break
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for x, y in train_loader:
            if step >= total_steps:
                stop = True
                break
            lr = learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if mark_cudagraph and compiled:
                torch.compiler.cudagraph_mark_step_begin()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = train_model(x, y)
            # Do not .item()/isfinite the loss every step: that stalls the GPU.
            # GradScaler already skips the Adam update on inf/NaN grads.
            scale_before = scaler.get_scale() if amp else 1.0
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(), args.grad_clip, foreach=True
                )
            scaler.step(optimizer)
            scaler.update()
            if amp and scaler.get_scale() < scale_before:
                skipped += 1

            step += 1
            tokens_seen += tokens_per_step
            if step == start_step + 20:
                steady_started = time.perf_counter()

            if step % args.log_every == 0:
                logged = reduce_mean(loss.detach(), ctx.world_size).item()
                last_loss = logged
                if amp and ctx.is_main:
                    peak_gb = max(peak_gb, torch.cuda.max_memory_allocated(device) / 2**30)
                if ctx.is_main:
                    elapsed = time.perf_counter() - started
                    done = max(1, step - start_step)
                    sps = elapsed / done
                    tps = tokens_per_step / max(1e-9, sps)
                    history.append({"step": step, "loss": logged, "lr": lr, "tok_s": tps})
                    extra = f" | skip {skipped}" if skipped else ""
                    print(
                        f"step {step:6d}/{total_steps} | loss {logged:.4f} | "
                        f"lr {lr:.2e} | {sps:.3f}s/step | {tps:,.0f} tok/s{extra}",
                        flush=True,
                    )

            if step % args.eval_every == 0 or step == total_steps:
                last_val = evaluate(
                    train_model,
                    valid_loader,
                    device,
                    amp,
                    args.eval_batches,
                    ctx.world_size,
                )
                if ctx.is_main:
                    ppl = (
                        math.exp(min(20.0, last_val))
                        if math.isfinite(last_val)
                        else float("inf")
                    )
                    print(
                        f"  [eval] step {step}: val_loss={last_val:.4f} ppl={ppl:.2f}",
                        flush=True,
                    )
                    if math.isfinite(last_val) and last_val < best:
                        best = last_val
                        torch.save(
                            dict(
                                model=raw_model.state_dict(),
                                optimizer=optimizer.state_dict(),
                                scaler=scaler.state_dict(),
                                step=step,
                                best=best,
                            ),
                            state_path,
                        )
                        torch.save(raw_model.state_dict(), output / "best_model.pt")
                        save_payload(raw_model, output, card)
        if stop:
            break

    elapsed = time.perf_counter() - started
    done_steps = max(1, step - start_step)
    overall_tps = (tokens_seen / elapsed) if elapsed > 0 else 0.0
    if steady_started is not None:
        steady_elapsed = max(1e-9, time.perf_counter() - steady_started)
        steady_tokens = max(0, tokens_seen - 20 * tokens_per_step)
        steady_tps = steady_tokens / steady_elapsed
    else:
        steady_tps = overall_tps
    try:
        # evaluate() all_reduces: every rank must enter or none must.
        # Reuse the in-loop eval at total_steps so rank 0 does not wait alone.
        if not math.isfinite(last_val):
            last_val = evaluate(
                train_model, valid_loader, device, amp, args.eval_batches, ctx.world_size
            )
        val = last_val
        if ctx.is_main:
            torch.save(raw_model.state_dict(), output / "model.pt")
            save_payload(raw_model, output, card)
            if math.isfinite(val) and val < best:
                best = val
            metrics = {
                "arch": args.arch,
                "title": card["title"],
                "kind": card["kind"],
                "parameters": card["parameters"],
                "card": card,
                "seq_len": args.seq_len,
                "vocab_size": args.vocab_size,
                "epochs": args.epochs,
                "steps": step,
                "planned_steps": total_steps,
                "global_batch": args.batch_size * ctx.world_size,
                "tokens_per_step": tokens_per_step,
                "tokens_seen": tokens_seen,
                "train_windows": len(train_ds),
                "valid_windows": len(valid_ds),
                "final_train_loss": last_loss,
                "best_val_loss": best if math.isfinite(best) else None,
                "best_val_ppl": (
                    math.exp(min(20.0, best)) if math.isfinite(best) else None
                ),
                "last_val_loss": val if math.isfinite(val) else None,
                "seconds": elapsed,
                "minutes": elapsed / 60.0,
                "sec_per_step": elapsed / done_steps,
                "tokens_per_sec_overall": overall_tps,
                "tokens_per_sec_steady": steady_tps,
                "peak_train_vram_gb": peak_gb,
                "skipped_nonfinite": skipped,
                "world_size": ctx.world_size,
                "compiled": compiled,
                "compile_mode": compile_mode,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "history": history,
            }
            with open(output / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(
                f"done {args.arch} in {elapsed/60:.1f} min | best_val={best:.4f} | "
                f"steady {steady_tps:,.0f} tok/s | skip={skipped}",
                flush=True,
            )
            sys.stdout.flush()
        if ctx.distributed:
            dist.barrier()
    finally:
        shutdown_loader(train_loader)
        shutdown_loader(valid_loader)
        destroy_distributed()


if __name__ == "__main__":
    main()
