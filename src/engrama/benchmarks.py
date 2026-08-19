"""
ENGRAMA Benchmarking and Verification Suite Module
Author: BUEORM
License: AGPL-3.0
"""

import time
from typing import Any, Dict

import torch

from engrama.model import EngramaModel


class BenchmarkSuite:
    """Performance benchmarking and causal invariance verification suite for ENGRAMA."""

    @staticmethod
    def benchmark_latency(
        model: EngramaModel, seq_length: int = 256, num_runs: int = 10
    ) -> Dict[str, Any]:
        """Benchmark parallel forward pass vs step-by-step cached generation latency."""
        model.eval()
        device = next(model.parameters()).device
        input_ids = torch.randint(
            0, model.config.vocab_size, (1, seq_length), device=device, dtype=torch.long
        )

        with torch.no_grad():
            model(input_ids)

            start = time.perf_counter()
            for _ in range(num_runs):
                model(input_ids)
            t_parallel = time.perf_counter() - start

            start = time.perf_counter()
            for _ in range(num_runs):
                cache = model.get_cache(N_max=seq_length)
                for t in range(seq_length):
                    model.step_forward(input_ids[:, t : t + 1], cache, t)
            t_step = time.perf_counter() - start

        total_tokens = num_runs * seq_length
        return {
            "parallel_tokens_per_sec": float(total_tokens / max(1e-9, t_parallel)),
            "parallel_ms_per_token": float((t_parallel * 1000) / max(1, total_tokens)),
            "step_tokens_per_sec": float(total_tokens / max(1e-9, t_step)),
            "step_ms_per_token": float((t_step * 1000) / max(1, total_tokens)),
            "seq_length": seq_length,
            "num_runs": num_runs,
        }

    @staticmethod
    def benchmark_memory(
        model: EngramaModel, seq_length: int = 256
    ) -> Dict[str, Any]:
        """Benchmark peak execution memory footprint."""
        model.eval()
        device = next(model.parameters()).device
        num_params = model.num_parameters()

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            input_ids = torch.randint(
                0, model.config.vocab_size, (1, seq_length), device=device, dtype=torch.long
            )
            with torch.no_grad():
                model(input_ids)
            peak_memory = torch.cuda.max_memory_allocated(device)
        else:
            peak_memory = sum(p.numel() * p.element_size() for p in model.parameters())

        return {
            "device": str(device),
            "peak_memory_bytes": int(peak_memory),
            "num_parameters": num_params,
            "seq_length": seq_length,
        }

    @staticmethod
    def verify_causal_invariance(
        model: EngramaModel, seq_length: int = 20, atol: float = 1e-4
    ) -> Dict[str, Any]:
        """Strictly verify mathematical causal invariance rule.

        Ensures: forward(seq) == step_forward(seq) with max absolute tolerance < atol.
        """
        model.eval()
        device = next(model.parameters()).device
        seq = torch.randint(
            0, model.config.vocab_size, (1, seq_length), device=device, dtype=torch.long
        )

        with torch.no_grad():
            logits_full = model.forward(seq)
            cache = model.get_cache(N_max=seq_length)
            logits_steps = []
            for t in range(seq_length):
                tok_t = seq[:, t : t + 1]
                logits_t, _ = model.step_forward(tok_t, cache, timestamp=t)
                logits_steps.append(logits_t)

            if logits_steps[0].dim() == 2:
                logits_step = torch.stack(logits_steps, dim=1)
            else:
                logits_step = torch.cat(logits_steps, dim=1)

            diff = torch.abs(logits_full - logits_step)
            max_diff = float(diff.max().item())
            passed = bool(max_diff < atol)

        return {
            "passed": passed,
            "max_diff": max_diff,
            "tolerance": float(atol),
        }
