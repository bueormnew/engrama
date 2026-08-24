"""ENGRAMA V6 — version DEFINITIVA: puramente lineal, recall exacto.

V6 = V5.5 estabilizado + tres correcciones estructurales:

1. Recall lexico por INDICE INVERTIDO de token (exacto, O(N) promedio).
2. Recall semantico por LSH de alta recuperacion (16 tablas x 16 bits, sin cap,
   con rescate denso de ventana fija) -> lineal y con ~100% de recall.
3. Claves FP32 en traza y LSH -> invarianza causal exacta en FP16/BF16.

Ademas se corrige el bug de vocab_size del evocador (V5.5 usaba 4096 fijo).

API::

    from engrama.v6 import EngraModelV6, V6Config

    model = EngraModelV6(V6Config.from_preset("base", vocab_size=50257))
    logits = model(input_ids)
    ids = model.generate(prompt_ids, max_new_tokens=200)
"""
from engrama.v6.config import V6Config
from engrama.v6.consolidation import V55ConsolidationStack as V6ConsolidationStack
from engrama.v6.encoder import IsolatedEncoderV2
from engrama.v6.model import EngraModelV6
from engrama.v6.recall import RecallTapV3
from engrama.v6.trace import PagedDualTrace

__all__ = [
    "EngraModelV6",
    "V6Config",
    "RecallTapV3",
    "PagedDualTrace",
    "IsolatedEncoderV2",
    "V6ConsolidationStack",
]
