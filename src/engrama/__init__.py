"""ENGRAMA: Non-Attention Autoregressive Neural Network Architecture & Library.

ENGRAMA V4 -- Isolated Footprint Encoding, Circular Trace, Hierarchical
Consolidation with Dual Target-Source Gating, Direct Trace Tap, Resonant
Multirate Offsets, RMSNorm, Minimum-Horizon Cache and Latent-Fusion
Multi-Candidate Evocation. Pure PyTorch, no attention anywhere.

Quick mode (no architecture knowledge required)::

    import engrama

    run = engrama.quickstart("corpus.txt", size="small", epochs=5)
    print(run.generate("Once upon a time"))
    run.save("./my_model")

Expert mode (full control)::

    from engrama import EngramaConfig, EngramaModel, Trainer, TextDataset

    config = EngramaConfig(
        vocab_size=128, d_model=256, num_cells=8,
        version="v4", offset_mode="resonant_multirate",
    )
    model = EngramaModel(config)

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
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
from engrama.losses import chunked_cross_entropy, linear_cross_entropy
from engrama.model import EngramaModel
from engrama.optimization import (
    DistributedContext,
    LanguageModelLoss,
    adamw,
    compile_model,
    configure_cuda,
    destroy_distributed,
    init_distributed,
    wrap_ddp,
)
from engrama.primitives import (
    Cell,
    EngramaLayerNorm,
    EngramaRMSNorm,
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

__version__ = "0.5.0"
__author__ = "Gerson Fabian Buenahora Ormaza (BUEORM)"
__license__ = "AGPL-3.0"

try:  # V5: arquitectura sin atencion con recuperacion exacta (ver docs/ENGRAMA-V5-Teorica.md)
    from engrama.v5 import EngraModel as EngraModelV5, V5Config, RecallTap, V5Trace
except Exception:  # pragma: no cover
    pass

__all__ = [
    "EngraModelV5",
    "V5Config",
    "RecallTap",
    "V5Trace",
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
    "EngramaRMSNorm",
    "VERSION_PRESETS",
    # Data & training
    "EngramaTokenizer",
    "TextDataset",
    "Trainer",
    "chunked_cross_entropy",
    "linear_cross_entropy",
    "DistributedContext",
    "LanguageModelLoss",
    "init_distributed",
    "destroy_distributed",
    "configure_cuda",
    "compile_model",
    "wrap_ddp",
    "adamw",
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
