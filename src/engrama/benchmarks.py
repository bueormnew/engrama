"""ENGRAMA Benchmarking and Verification Suite.

Latency, memory and causal-invariance measurements. Memory reporting is
explicit about what is measured on each device (model weights vs. true
CUDA peak allocations vs. cache footprint).

Author: BUEORM
License: AGPL-3.0
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch

from engrama.model import EngramaModel


class BenchmarkSuite:
    """Performance benchmarking and causal invariance verification suite."""

    # ------------------------------------------------------------------
    @staticmethod
    def benchmark_latency(
        model: EngramaModel, seq_length: int = 256, num_runs: int = 10
    ) -> Dict[str, Any]:
        """Benchmark parallel forward vs cached step-by-step generation."""
        model.eval()
        device = next(model.parameters()).device
        input_ids = torch.randint(
            0, model.config.vocab_size, (1, seq_length), device=device, dtype=torch.long
        )

        with torch.no_grad():
            model(input_ids)  # warmup

            start = time.perf_counter()
            for _ in range(num_runs):
                model(input_ids)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_parallel = time.perf_counter() - start

            start = time.perf_counter()
            for _ in range(num_runs):
                cache = model.get_cache(N_max=seq_length)
                for t in range(seq_length):
                    model.step_forward(input_ids[:, t : t + 1], cache, t)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_step = time.perf_counter() - start

        total_tokens = num_runs * seq_length
        return {
            "parallel_tokens_per_sec": float(total_tokens / max(1e-9, t_parallel)),
            "parallel_ms_per_token": float((t_parallel * 1000) / max(1, total_tokens)),
            "step_tokens_per_sec": float(total_tokens / max(1e-9, t_step)),
            "step_ms_per_token": float((t_step * 1000) / max(1, total_tokens)),
            "seq_length": seq_length,
            "num_runs": num_runs,
            "cache_mode": model.config.cache_mode,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def benchmark_memory(
        model: EngramaModel, seq_length: int = 256
    ) -> Dict[str, Any]:
        """Memory footprint report (weights, cache, and true peaks on CUDA)."""
        model.eval()
        device = next(model.parameters()).device
        num_params = model.num_parameters()
        params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

        cache = model.get_cache(N_max=seq_length)
        with torch.no_grad():
            for t in range(seq_length):
                model.step_forward(torch.zeros(1, 1, dtype=torch.long, device=device), cache, t)
        cache_bytes = cache.get_memory_footprint()

        result: Dict[str, Any] = {
            "device": str(device),
            "num_parameters": num_params,
            "parameter_bytes": int(params_bytes),
            "cache_bytes_at_seq": int(cache_bytes),
            "cache_mode": cache.mode,
            "seq_length": seq_length,
        }

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            input_ids = torch.randint(
                0, model.config.vocab_size, (1, seq_length),
                device=device, dtype=torch.long,
            )
            with torch.no_grad():
                model(input_ids)
            result["peak_forward_bytes_cuda"] = int(torch.cuda.max_memory_allocated(device))
        else:
            # Honest estimate on CPU: PyTorch does not expose allocator peaks;
            # report weights + cache instead of pretending to measure peaks.
            result["peak_forward_bytes_cuda"] = None
            result["note"] = (
                "CPU execution: exact activation peaks are not exposed by the "
                "allocator; parameter_bytes and cache_bytes_at_seq are exact."
            )
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def verify_causal_invariance(
        model: EngramaModel,
        seq_length: int = 20,
        atol: float = 1e-4,
        cache_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify causal invariance: forward(seq) == step_forward(seq).

        Tests V3 theorem 1 (section 23) under the requested cache mode
        (theorem 2, section 24, for ``hierarchical``).
        """
        model.eval()
        device = next(model.parameters()).device
        seq = torch.randint(
            0, model.config.vocab_size, (1, seq_length), device=device, dtype=torch.long
        )

        with torch.no_grad():
            logits_full = model.forward(seq)
            cache = model.get_cache(N_max=seq_length, mode=cache_mode)
            logits_steps = []
            for t in range(seq_length):
                logits_t, _ = model.step_forward(seq[:, t : t + 1], cache, timestamp=t)
                logits_steps.append(logits_t)
            logits_step = torch.stack(logits_steps, dim=1)

            diff = torch.abs(
                logits_full.float() - logits_step.float()
            )
            max_diff = float(diff.max().item())
            mean_diff = float(diff.mean().item())

        return {
            "passed": bool(max_diff < atol),
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "tolerance": float(atol),
            "cache_mode": cache.mode,
        }
