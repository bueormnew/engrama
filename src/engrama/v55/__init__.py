"""ENGRAMA V5.5 — sin atencion, sin compresion, recuperacion exacta en texto real.

V5.5 = V5 estabilizado + Recall Tap ASIMETRICO (eje lexico aislado + eje de
sentido contextual) que desambigua palabras polisemicas y alcanza el 100 % de
recuperacion en texto real, manteniendo intacta la filosofia inviolable.

API rapida::

    from engrama.v55 import EngraModelV55, V55Config

    model = EngraModelV55(V55Config.from_preset("base", vocab_size=50257))
    loss  = model.forward_loss(x[:, :-1], x[:, 1:])
    ids   = model.generate(prompt_ids, max_new_tokens=200)

Ver ``docs/ENGRAMA-V55-Teorica.md`` para el diseno completo.
"""
from engrama.v55.config import V55Config
from engrama.v55.consolidation import V55ConsolidationStack
from engrama.v55.encoder import IsolatedEncoderV2
from engrama.v55.model import EngraModelV55
from engrama.v55.recall import RecallTapV2
from engrama.v55.trace import PagedDualTrace

__all__ = [
    "EngraModelV55",
    "V55Config",
    "RecallTapV2",
    "PagedDualTrace",
    "IsolatedEncoderV2",
    "V55ConsolidationStack",
]
