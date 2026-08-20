"""ENGRAMA: Non-Attention Autoregressive Neural Network Architecture & Library.

ENGRAMA V3 -- Isolated Footprint Encoding, Circular Trace, Hierarchical
Dyadic Consolidation with Factorized Synapses and Identity Transport,
Minimum-Horizon Cache and Factorized Multi-Candidate Evocation. Pure
PyTorch, no attention anywhere.

Quick mode (no architecture knowledge required)::

    import engrama

    run = engrama.quickstart("corpus.txt", size="small", epochs=5)
    print(run.generate("Once upon a time"))
    run.save("./my_model")

Expert mode (full control)::

    from engrama import EngramaConfig, EngramaModel, Trainer, TextDataset

    config = EngramaConfig(
        vocab_size=128, d_model=256, num_cells=8,
        offset_mode="hierarchical_dyadic", cache_mode="hierarchical",
    )
    model = EngramaModel(config)

Author: BUEORM
License: AGPL-3.0
"""

from engrama.benchmarks import BenchmarkSuite
from engrama.config import VERSION_PRESETS, EngramaConfig
from engrama.consolidation import (
    ConsolidationLayer,
    ConsolidationStack,
    PositionalDilatedMix,
)
from engrama.datasets import TextDataset
from engrama.encoder import IsolatedEncoder
from engrama.evoker import MultiCandidateEvoker
from engrama.inference import Generator
from engrama.inspection import EngramaInspector
from engrama.model import EngramaModel
from engrama.primitives import (
    Cell,
    EngramaLayerNorm,
    SharedCoreCellGroup,
    SynapseLayer,
)
from engrama.quick import (
    QuickRun,
    create_model,
    default_lr,
    list_sizes,
    load_quick,
    quickstart,
)
from engrama.serialization import load_model, save_model
from engrama.tokenizer import EngramaTokenizer
from engrama.trace import CircularTrace, EngramaCache, HierarchicalStateCache
from engrama.trainer import Trainer

__version__ = "0.3.0"
__author__ = "BUEORM"
__license__ = "AGPL-3.0"

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    # Core architecture
    "EngramaConfig",
    "EngramaModel",
    "EngramaCache",
    "CircularTrace",
    "HierarchicalStateCache",
    "IsolatedEncoder",
    "ConsolidationLayer",
    "ConsolidationStack",
    "PositionalDilatedMix",
    "MultiCandidateEvoker",
    "Cell",
    "SharedCoreCellGroup",
    "SynapseLayer",
    "EngramaLayerNorm",
    "VERSION_PRESETS",
    # Data & training
    "EngramaTokenizer",
    "TextDataset",
    "Trainer",
    # Inference & tooling
    "Generator",
    "EngramaInspector",
    "BenchmarkSuite",
    "save_model",
    "load_model",
    # Quick mode
    "quickstart",
    "create_model",
    "load_quick",
    "list_sizes",
    "default_lr",
    "QuickRun",
]
