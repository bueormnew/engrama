"""
ENGRAMA Command Line Interface (CLI) Engine
Author: BUEORM
License: AGPL-3.0
"""

import argparse
import sys

import torch

from engrama.benchmarks import BenchmarkSuite
from engrama.config import EngramaConfig
from engrama.datasets import TextDataset
from engrama.inference import Generator
from engrama.inspection import EngramaInspector
from engrama.model import EngramaModel
from engrama.serialization import load_model, save_model
from engrama.tokenizer import EngramaTokenizer
from engrama.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ENGRAMA: Non-Attention Autoregressive Neural Network Tool"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train an ENGRAMA model on a text file")
    train_parser.add_argument("--text-file", type=str, required=True, help="Path to plain text training file")
    train_parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory to save model")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--context-len", type=int, default=128, help="Context sequence length")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text from a trained ENGRAMA model")
    gen_parser.add_argument("--model-dir", type=str, required=True, help="Directory of saved model checkpoint")
    gen_parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    gen_parser.add_argument("--max-tokens", type=int, default=50, help="Maximum new tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    gen_parser.add_argument("--top-k", type=int, default=None, help="Top-K sampling filter")
    gen_parser.add_argument("--top-p", type=float, default=None, help="Top-P nucleus sampling filter")
    gen_parser.add_argument("--stream", action="store_true", help="Stream generated output tokens")
    gen_parser.add_argument("--no-cache", action="store_true", help="Disable step trace cache")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model loss on text dataset")
    eval_parser.add_argument("--model-dir", type=str, required=True, help="Directory of saved model checkpoint")
    eval_parser.add_argument("--text-file", type=str, required=True, help="Path to text dataset for evaluation")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run latency, memory & causal invariance benchmarks")
    bench_parser.add_argument("--seq-len", type=int, default=256, help="Sequence length for benchmark")
    bench_parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect model architecture, activations & gate states")
    inspect_parser.add_argument("--model-dir", type=str, required=True, help="Directory of saved model checkpoint")
    inspect_parser.add_argument("--sample-text", type=str, default="ENGRAMA test prompt", help="Sample text for activation inspection")

    # Info command
    subparsers.add_parser("info", help="Print ENGRAMA environment, author, license & version details")

    args = parser.parse_args()

    if args.command == "train":
        print(f"[ENGRAMA] Loading text data from {args.text_file}...")
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer = EngramaTokenizer()
        tokenizer.fit_on_text(text)
        config = EngramaConfig(vocab_size=tokenizer.vocab_size, context_length=args.context_len)
        model = EngramaModel(config)
        dataset = TextDataset(text, tokenizer, sequence_length=config.context_length)
        print(f"[ENGRAMA] Training model ({model.num_parameters()} params) for {args.epochs} epochs...")
        trainer = Trainer(model, lr=args.lr)
        history = trainer.fit(dataset, batch_size=args.batch_size, epochs=args.epochs)
        save_model(model, args.output_dir, tokenizer)
        print(f"[ENGRAMA] Model and tokenizer saved successfully to '{args.output_dir}'. Final Loss: {history[-1]:.4f}")

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
            ):
                print(char_token, end="", flush=True)
            print()
        else:
            res = generator.generate(
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                use_cache=not args.no_cache,
            )
            print(res)

    elif args.command == "evaluate":
        model, tokenizer = load_model(args.model_dir)
        if tokenizer is None:
            tokenizer = EngramaTokenizer()
        dataset = TextDataset(args.text_file, tokenizer, sequence_length=model.config.context_length)
        trainer = Trainer(model)
        loss = trainer.evaluate(dataset)
        print(f"[ENGRAMA] Validation Loss: {loss:.4f}")

    elif args.command == "benchmark":
        print(f"[ENGRAMA] Benchmark Suite (seq_length={args.seq_len}, runs={args.runs})...")
        config = EngramaConfig()
        model = EngramaModel(config)
        lat = BenchmarkSuite.benchmark_latency(model, seq_length=args.seq_len, num_runs=args.runs)
        mem = BenchmarkSuite.benchmark_memory(model, seq_length=args.seq_len)
        causal = BenchmarkSuite.verify_causal_invariance(model, seq_length=min(20, args.seq_len))
        print("Latency Results:", lat)
        print("Memory Footprint:", mem)
        print("Causal Invariance Check:", causal)

    elif args.command == "inspect":
        model, tokenizer = load_model(args.model_dir)
        if tokenizer is None:
            tokenizer = EngramaTokenizer()
        summary = EngramaInspector.inspect_model_summary(model)
        print("[ENGRAMA] Model Architecture Summary:", summary)
        sample_ids = torch.tensor([tokenizer.encode(args.sample_text)], dtype=torch.long)
        activations = EngramaInspector.inspect_activations(model, sample_ids)
        print("[ENGRAMA] Activations Statistics (T0..TL):", activations)

    elif args.command == "info":
        print("=" * 60)
        print("ENGRAMA Neural Architecture & Library")
        print("Author: BUEORM")
        print("License: AGPL 3.0 (GNU Affero General Public License v3.0)")
        print("Version: 0.1.0")
        print("-" * 60)
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print("Architecture Spec: Non-Attention Autoregressive with Isolated Encoding")
        print("=" * 60)


if __name__ == "__main__":
    main()
