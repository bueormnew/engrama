"""ENGRAMA Architecture Inspector and Diagnostic Utilities.

V3 interpretability (spec section 50): every route keeps inspectable,
position-stable state, and the factorized synapses expose their alpha
gates, identity betas and low-rank scales.

Author: BUEORM
License: AGPL-3.0
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from engrama.model import EngramaModel


def _stats(t: torch.Tensor) -> Dict[str, float]:
    t = t.detach().float()
    return {
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "norm": float(torch.norm(t).item()),
    }


class EngramaInspector:
    """Diagnostic inspector for ENGRAMA internal representations and gates."""

    # ------------------------------------------------------------------
    @staticmethod
    def inspect_activations(
        model: EngramaModel, input_ids: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """Statistical summary of hidden states per level (T0..T_L)."""
        with torch.no_grad():
            x = model.embeddings(input_ids)
            t = model.encoder(x)
            res = {"T0": _stats(t)}
            for l, layer in enumerate(model.consolidation.layers):
                t = layer.forward_train(t)
                res[f"T{l+1}"] = _stats(t)
            return res

    # ------------------------------------------------------------------
    @staticmethod
    def inspect_gates(
        model: EngramaModel, input_ids: torch.Tensor
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Gate activity per offset and consolidation layer."""
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
                    g = mix._gate(t_shifted, str_p)
                    entry = _stats(g)
                    rho = mix._scale_gate(str_p)
                    if not isinstance(rho, float):
                        entry["rho"] = float(rho.detach().item())
                    layer_gates[f"offset_{p}"] = entry
                gates_dict[f"layer_{l}"] = layer_gates
                t = layer.forward_train(t)
            return gates_dict

    # ------------------------------------------------------------------
    @staticmethod
    def inspect_synapses(model: EngramaModel) -> Dict[str, Any]:
        """Identity-route fidelity and low-rank magnitudes (V3 spec 31/50)."""
        report: Dict[str, Any] = {"encoder": [], "consolidation": []}
        for idx, layer in enumerate(model.encoder.layers):
            entry: Dict[str, Any] = {"layer": idx, "mode": layer.synapse_mode}
            if layer.synapse_mode == "factorized":
                entry["beta_mean"] = layer.identity_fidelity()
                entry["s_scale_abs_mean"] = float(
                    layer.s_scale.detach().abs().mean().item()
                )
            report["encoder"].append(entry)

        for idx, layer in enumerate(model.consolidation.layers):
            mix = layer.mix
            entry = {"layer": idx, "mode": mix.synapse_mode, "offsets": mix.offsets}
            fid = mix.identity_fidelity()
            if fid is not None:
                entry["beta_mean"] = fid
                entry["s_scale_abs_mean"] = float(
                    torch.stack(
                        [mix.s_scale[s].detach().abs().mean() for s in mix.s_scale]
                    )
                    .mean()
                    .item()
                )
            if mix.hierarchical_gate and mix.rho is not None:
                entry["rho"] = {
                    s: float(torch.sigmoid(mix.rho[s]).item()) for s in mix.rho
                }
            report["consolidation"].append(entry)
        return report

    # ------------------------------------------------------------------
    @staticmethod
    def inspect_model_summary(model: EngramaModel) -> Dict[str, Any]:
        """Structural architecture summary of the model."""
        cfg = model.config
        return {
            "version": cfg.version,
            "total_parameters": model.num_parameters(only_trainable=False),
            "trainable_parameters": model.num_parameters(only_trainable=True),
            "vocab_size": cfg.vocab_size,
            "d_model": cfg.d_model,
            "d_gate": cfg.d_gate,
            "d_ff": cfg.d_ff,
            "num_cells": cfg.num_cells,
            "num_encoder_layers": cfg.num_encoder_layers,
            "num_consolidation_layers": cfg.num_consolidation_layers,
            "context_length": cfg.context_length,
            "num_candidates": cfg.num_candidates,
            "candidate_aggregation": cfg.candidate_aggregation,
            "synapse_mode": cfg.synapse_mode,
            "synapse_rank": cfg.synapse_rank,
            "identity_transport": cfg.identity_transport,
            "cell_mode": cfg.cell_mode,
            "offset_mode": cfg.offset_mode,
            "cache_mode": cfg.cache_mode,
            "evoker_mode": cfg.evoker_mode,
            "hierarchical_gate": cfg.hierarchical_gate,
            "global_anchor": cfg.global_anchor,
            "tie_embeddings": cfg.tie_embeddings,
            "layer_offsets": cfg.layer_offsets(),
            "cache_horizons": cfg.cache_horizons(),
            "receptive_field": cfg.receptive_field(),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def inspect_cache(cache: Any) -> Dict[str, Any]:
        """Cache state, horizons and memory savings (V3 sections 12/24)."""
        return cache.describe()
