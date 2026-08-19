"""
ENGRAMA: Non-Attention Autoregressive Neural Network Architecture & Library

Author: BUEORM
License: AGPL-3.0
"""

from engrama.benchmarks import BenchmarkSuite
from engrama.config import EngramaConfig
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
from engrama.primitives import Cell, EngramaLayerNorm, SynapseLayer
from engrama.serialization import load_model, save_model
from engrama.tokenizer import EngramaTokenizer
from engrama.trace import EngramaCache
from engrama.trainer import Trainer

__version__ = "0.1.0"
__author__ = "BUEORM"
__license__ = "AGPL-3.0"

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "BenchmarkSuite",
    "Cell",
    "ConsolidationLayer",
    "ConsolidationStack",
    "EngramaCache",
    "EngramaConfig",
    "EngramaInspector",
    "EngramaLayerNorm",
    "EngramaModel",
    "EngramaTokenizer",
    "Generator",
    "IsolatedEncoder",
    "MultiCandidateEvoker",
    "PositionalDilatedMix",
    "SynapseLayer",
    "TextDataset",
    "Trainer",
    "load_model",
    "save_model",
]
