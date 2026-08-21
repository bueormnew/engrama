"""ENGRAMA V5 — arquitectura sin atencion, sin compresion, recuperacion exacta.

API rapida::

    from engrama.v5 import EngraModel, V5Config

    model = EngraModel(V5Config.from_preset("base", vocab_size=50257))
    loss  = model.forward_loss(x, y)                 # entrenamiento paralelo
    ids   = model.generate(prompt_ids, max_new_tokens=200)   # cache incremental

Ver ``docs/ENGRAMA-V5-Teorica.md`` para el diseno completo.
"""
from engrama.v5.config import V5Config
from engrama.v5.model import EngraModel
from engrama.v5.recall import RecallTap
from engrama.v5.trace import V5Trace

EngraModelV5 = EngraModel  # alias retrocompatible

__all__ = ["EngraModel", "EngraModelV5", "V5Config", "RecallTap", "V5Trace"]
