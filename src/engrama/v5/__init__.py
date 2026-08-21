"""ENGRAMA V5 — Synaptic Resonance over an Explicit Trace (no attention).

ENGRAMA V5 keeps the whole ENGRAMA ontology (isolated footprint encoding →
explicit FIFO trace → synapses/cells → multi-candidate evocation) and replaces
the fixed-offset consolidation of V4 with **content-addressable synaptic
resonance**: a point-to-point sigmoid gate over the explicit trace, with **no
softmax over positions and no compressed recurrent state**.

Design goals (see ``ENGRAMA-V5-Teorica.md``):

* No attention — the gate ``sigma(tau * <q,k> + b)`` is decided independently
  per (query, key) pair; positions never compete for a normalized mass.
* No compression — every token's key/value is stored explicitly in the trace.
* Fully parallel training, O(N) generation memory, exact causal invariance.
* High long-range retrieval (measured 98%+ on the KV task that V4 solved at ~8%).

Quick start::

    from engrama.v5 import EngramaV5, EngramaV5Config

    cfg = EngramaV5Config(vocab_size=32000, d_model=256, num_layers=6,
                          num_heads=8, context_length=8192)
    model = EngramaV5(cfg)
    logits = model(input_ids)                    # (B, N, vocab)
    out = model.generate([1, 2, 3], max_new_tokens=50)

Author: ENGRAMA contributors.
License: AGPL-3.0
"""

from engrama.v5.config import EngramaV5Config
from engrama.v5.model import EngramaV5
from engrama.v5.resonance import SynapticResonance
from engrama.v5.blocksparse import BlockSparseResonance
from engrama.v5.cache import ResonanceCache
from engrama.v5.trainer import V5Trainer
from engrama.v5 import triton_kernels

__all__ = [
    "EngramaV5Config",
    "EngramaV5",
    "SynapticResonance",
    "BlockSparseResonance",
    "ResonanceCache",
    "V5Trainer",
    "triton_kernels",
]
