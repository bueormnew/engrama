"""
ENGRAMA Architecture Inspector and Diagnostic Utilities Module
Author: BUEORM
License: AGPL-3.0
"""

from typing import Any, Dict

import torch

from engrama.model import EngramaModel


class EngramaInspector:
    """Diagnostic inspector for ENGRAMA internal representations, gate states, and metrics."""

    @staticmethod
    def inspect_activations(
        model: EngramaModel, input_ids: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """Compute statistical summary of hidden state activations across layers."""
        with torch.no_grad():
            x = model.embeddings(input_ids)
            t0 = model.encoder(x)
            res = {
                "T0": {
                    "mean": float(t0.mean().item()),
                    "std": float(t0.std().item()),
                    "min": float(t0.min().item()),
                    "max": float(t0.max().item()),
                    "norm": float(torch.norm(t0).item()),
                }
            }
            t = t0
            for l, layer in enumerate(model.consolidation.layers):
                t = layer.forward_train(t)
                res[f"T{l+1}"] = {
                    "mean": float(t.mean().item()),
                    "std": float(t.std().item()),
                    "min": float(t.min().item()),
                    "max": float(t.max().item()),
                    "norm": float(torch.norm(t).item()),
                }
            return res

    @staticmethod
    def inspect_gates(
        model: EngramaModel, input_ids: torch.Tensor
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Inspect dynamic gating activity across positional offsets in consolidation layers."""
        with torch.no_grad():
            x = model.embeddings(input_ids)
            t = model.encoder(x)
            b, n, d = t.shape
            gates_dict: Dict[str, Dict[str, Dict[str, float]]] = {}
            for l, layer in enumerate(model.consolidation.layers):
                mix = layer.mix
                layer_gates: Dict[str, Dict[str, float]] = {}
                for p in mix.offsets:
                    str_p = str(p)
                    if p == 0:
                        t_shifted = t
                    elif p < n:
                        t_shifted = torch.cat(
                            [
                                torch.zeros(b, p, d, device=t.device, dtype=t.dtype),
                                t[:, :-p, :],
                            ],
                            dim=1,
                        )
                    else:
                        continue
                    h_g = mix.p_g(t_shifted)
                    gate_logits = (
                        torch.matmul(h_g, mix.gate_w[str_p]) + mix.gate_b[str_p]
                    )
                    g = torch.sigmoid(gate_logits)
                    layer_gates[str_p] = {
                        "mean": float(g.mean().item()),
                        "std": float(g.std().item()),
                        "min": float(g.min().item()),
                        "max": float(g.max().item()),
                    }
                gates_dict[f"layer_{l}"] = layer_gates
                t = layer.forward_train(t)
            return gates_dict

    @staticmethod
    def inspect_model_summary(model: EngramaModel) -> Dict[str, Any]:
        """Return structural architecture summary of model."""
        return {
            "total_parameters": model.num_parameters(only_trainable=False),
            "trainable_parameters": model.num_parameters(only_trainable=True),
            "vocab_size": model.config.vocab_size,
            "d_model": model.config.d_model,
            "d_gate": model.config.d_gate,
            "num_cells": model.config.num_cells,
            "num_encoder_layers": model.config.num_encoder_layers,
            "num_consolidation_layers": model.config.num_consolidation_layers,
            "context_length": model.config.context_length,
            "offsets": model.config.offsets,
            "num_candidates": model.config.num_candidates,
            "candidate_aggregation": model.config.candidate_aggregation,
        }
