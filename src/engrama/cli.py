"""ENGRAMA Command Line Interface (CLI).

Quick commands (safe defaults -- train uses the ``small`` preset so it runs
on modest machines) and full expert control via flags.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""

from __future__ import annotations

import argparse
import sys

import torch

from engrama import __version__
from engrama.benchmarks import BenchmarkSuite
from engrama.config import EngramaConfig
from engrama.datasets import TextDataset
from engrama.inference import Generator
from engrama.inspection import EngramaInspector
from engrama.model import EngramaModel
from engrama.quick import list_sizes
from engrama.serialization import load_model, save_model
from engrama.tokenizer import EngramaTokenizer
from engrama.trainer import Trainer


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--text-file", type=str, required=True, help="Plain text training file")
    p.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=None, help="Learning rate (preset default if omitted)")
    p.add_argument("--size", type=str, default="small",
                   choices=list(list_sizes()), help="Size preset")
    p.add_argument("--context-len", type=int, default=None, help="Context window (N_max)")
    p.add_argument("--d-model", type=int, default=None, help="Override hidden dimension")
    p.add_argument("--num-cells", type=int, default=None, help="Override encoder cells")
    p.add_argument("--consolidation-layers", type=int, default=None,
                   help="Override consolidation depth L")
    p.add_argument("--scheduler", type=str, default="none",
                   choices=["none", "warmup", "cosine"])
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="cpu/cuda (auto if omitted)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="engrama",
        description="ENGRAMA: Non-Attention Autoregressive Neural Network Tool (V3)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train an ENGRAMA model on text")
    _add_train_args(train_parser)

    gen_parser = subparsers.add_parser("generate", help="Generate text from a checkpoint")
    gen_parser.add_argument("--model-dir", type=str, required=True)
    gen_parser.add_argument("--prompt", type=str, required=True)
    gen_parser.add_argument("--max-tokens", type=int, default=50)
    gen_parser.add_argument("--temperature", type=float, default=1.0)
    gen_parser.add_argument("--top-k", type=int, default=None)
    gen_parser.add_argument("--top-p", type=float, default=None)
    gen_parser.add_argument("--stream", action="store_true")
    gen_parser.add_argument("--no-cache", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate loss on a text file")
    eval_parser.add_argument("--model-dir", type=str, required=True)
    eval_parser.add_argument("--text-file", type=str, required=True)

    bench_parser = subparsers.add_parser(
        "benchmark", help="Latency, memory & causal invariance benchmarks"
    )
    bench_parser.add_argument("--seq-len", type=int, default=256)
    bench_parser.add_argument("--runs", type=int, default=10)
    bench_parser.add_argument("--size", type=str, default="small", choices=list(list_sizes()))
    bench_parser.add_argument(
        "--cache-mode", type=str, default=None, choices=["full", "hierarchical"]
    )

    inspect_parser = subparsers.add_parser(
        "inspect", help="Architecture, activations, gates & synapse fidelity"
    )
    inspect_parser.add_argument("--model-dir", type=str, required=True)
    inspect_parser.add_argument("--sample-text", type=str, default="ENGRAMA inspection")

    subparsers.add_parser("sizes", help="List size presets")
    subparsers.add_parser("info", help="Environment, author, license & version")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    if args.command == "train":
        print(f"[ENGRAMA] Loading text data from {args.text_file}...")
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()

        tokenizer = EngramaTokenizer().fit_on_text(text)
        overrides = {}
        if args.context_len is not None:
            overrides["context_length"] = args.context_len
        if args.d_model is not None:
            overrides["d_model"] = args.d_model
        if args.num_cells is not None:
            overrides["num_cells"] = args.num_cells
        if args.consolidation_layers is not None:
            overrides["num_consolidation_layers"] = args.consolidation_layers

        config = EngramaConfig.preset(
            args.size, vocab_size=tokenizer.vocab_size, **overrides
        )
        model = EngramaModel(config)
        seq_len = min(config.context_length, 128)
        dataset = TextDataset(text, tokenizer, sequence_length=seq_len)
        if len(dataset) == 0:
            print("[ENGRAMA] ERROR: dataset is empty; provide more text.")
            sys.exit(1)

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        from engrama.quick import default_lr
        lr = args.lr if args.lr is not None else default_lr(args.size)

        print(f"[ENGRAMA] {config.version.upper()} model "
              f"({model.num_parameters():,} params, size={args.size}, "
              f"L={config.num_consolidation_layers}, N_max={config.context_length}) "
              f"on {device} for {args.epochs} epochs (lr={lr:g})...")
        trainer = Trainer(
            model, lr=lr, device=device,
            scheduler=args.scheduler, warmup_steps=args.warmup_steps,
        )
        history = trainer.fit(dataset, batch_size=args.batch_size, epochs=args.epochs)
        save_model(model, args.output_dir, tokenizer)
        print(
            f"[ENGRAMA] Saved to '{args.output_dir}'. "
            f"Loss: {history[0]:.4f} -> {history[-1]:.4f}"
        )

    elif args.command == "generate":
        model, tokenizer = load_model(args.model_dir)
        if tokenizer is None:
            tokenizer = EngramaTokenizer()
        generator = Generator(model, tokenizer)
        if args.stream:
            print(args.prompt, end="", flush=True)
            for char_token in generator.generate_stream(
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                use_cache=not args.no_cache,
            ):
                print(char_token, end="", flush=True)
            print()
        else:
            print(
                generator.generate(
                    prompt=args.prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    use_cache=not args.no_cache,
                )
            )

    elif args.command == "evaluate":
        model, tokenizer = load_model(args.model_dir)
        if tokenizer is None:
            tokenizer = EngramaTokenizer()
        dataset = TextDataset(
            args.text_file, tokenizer,
            sequence_length=min(model.config.context_length, 128),
        )
        trainer = Trainer(model)
        loss = trainer.evaluate(dataset)
        print(f"[ENGRAMA] Validation Loss: {loss:.4f}")

    elif args.command == "benchmark":
        print(f"[ENGRAMA] Benchmark (size={args.size}, seq={args.seq_len}, runs={args.runs})...")
        config = EngramaConfig.preset(args.size)
        model = EngramaModel(config)
        lat = BenchmarkSuite.benchmark_latency(model, seq_length=args.seq_len, num_runs=args.runs)
        mem = BenchmarkSuite.benchmark_memory(model, seq_length=args.seq_len)
        causal_full = BenchmarkSuite.verify_causal_invariance(
            model, seq_length=min(20, args.seq_len), cache_mode="full"
        )
        causal_hier = BenchmarkSuite.verify_causal_invariance(
            model, seq_length=min(20, args.seq_len), cache_mode="hierarchical"
        )
        print("Latency:", lat)
        print("Memory:", mem)
        print("Causal invariance (full cache):", causal_full)
        print("Causal invariance (hierarchical cache):", causal_hier)

    elif args.command == "inspect":
        model, tokenizer = load_model(args.model_dir)
        if tokenizer is None:
            tokenizer = EngramaTokenizer()
        summary = EngramaInspector.inspect_model_summary(model)
        print("[ENGRAMA] Model summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        sample_ids = torch.tensor([tokenizer.encode(args.sample_text)], dtype=torch.long)
        print("[ENGRAMA] Activations:")
        for level, stats in EngramaInspector.inspect_activations(model, sample_ids).items():
            print(f"  {level}: mean={stats['mean']:.4f} std={stats['std']:.4f} "
                  f"norm={stats['norm']:.2f}")
        print("[ENGRAMA] Synapse fidelity:")
        for group, entries in EngramaInspector.inspect_synapses(model).items():
            print(f"  {group}: {entries}")

    elif args.command == "sizes":
        print("[ENGRAMA] Size presets:")
        for name, desc in list_sizes().items():
            print(f"  {name:>6}: d_model={desc['d_model']} C={desc['num_cells']} "
                  f"L_enc={desc['num_encoder_layers']} L={desc['num_consolidation_layers']} "
                  f"N_max={desc['context_length']} r={desc['synapse_rank']}")

    elif args.command == "info":
        print("=" * 64)
        print("ENGRAMA Neural Architecture & Library")
        print("Author: Gerson Fabian Buenahora Ormaza (BUEORM)")
        print("License: AGPL-3.0 (GNU Affero General Public License v3.0)")
        print(f"Version: {__version__} (V3 architecture)")
        print("-" * 64)
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print("Spec: ENGRAMA-V3-Teorica.md (no attention, no QK^T, hierarchical cache)")
        print("=" * 64)


if __name__ == "__main__":
    main()
