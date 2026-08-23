"""ENGRAMA V5.5 — configuracion.

V5.5 = V5 estabilizado y acelerado mas el **Recall Tap asimetrico** que
recupera el 100 % en texto real desambiguando por *sentido* (contexto local),
manteniendo intacta la filosofia inviolable de ENGRAMA:

* Huella aislada ``T0[j] = f(x_j)`` (cero mezcla temporal en Fase 1).
* Traza FIFO explicita, sin compresion, almacena ``T0`` pristino.
* Cero ``QK^T`` ``N x N``, cero softmax sobre el eje temporal.
* Consolidacion por offsets causales fijos ``D_l``.
* Invarianza causal exacta: forward paralelo == generacion incremental.

Las 3 claves que dan el 100 % en texto real:

1. **Recall Tap asimetrico**: ``score = score_lex * (1 + beta*score_sense)``.
   ``K_lex`` aislado (induccion garantizada) desempata con ``K_sense``
   (contexto local de 2-3 tokens). Un mismo token en dos contextos ya no
   colisiona.
2. **Fallback exacto garantizado**: indice de ultima ocurrencia del mismo
   token (O(1)) + escaneo denso de los ultimos ``fallback_window`` tokens si
   el mejor score cae bajo umbral. 100 % teorico.
3. **Estabilidad absoluta**: RMSNorm en toda entrada a productos punto,
   zero-init en todos los residuales/compuertas, evocador con *softcap*.

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

_OFFSET_MODES = ("resonant_multirate", "dense_dilated")
_RT_TRAIN_MODE = ("dense", "lsh")
_ACTIVATIONS = ("gelu", "relu", "silu")


@dataclass
class V55Config:
    """Configuracion de ENGRAMA V5.5.

    Arquitectura:

    * **Encoder V2** (Pilar 1): RMSNorm + enrutado sinaptico ``C x C``
      factorizado + celula SwiGLU + residual ``tanh(gamma)`` zero-init.
    * **Traza dual paginada** (Pilar 2): guarda ``T0`` pristino + ``T_shallow``
      (salida de la capa 0, contexto local de 2 tokens) + codigos ``K`` en
      paginas de ``page_size`` tokens. Append O(1), sin ``torch.cat``.
    * **Consolidacion V5.5** (Pilar 3): mezcla NORMALIZADA por conteo + compuerta
      dual acotada con RMSNorm + residual zero-init. 100 % torch.compile.
    * **Recall Tap asimetrico** (Pilar 4): ``score_lex * (1 + beta*score_sense)``
      con fallback exacto.
    * **LSH cuantizado** (Pilar 5): ``K_lex`` binarizado (sign -> 64 bits
      bitpacked), bucket por distancia de Hamming, kernel fusionado.
    * **Evocador con softcap** (Pilar 6): ``logits = softcap(c_bar @ E^T/sqrt(d), C)``.

    Args:
        recall_enabled: Activar el Recall Tap asimetrico.
        d_recall: Dimension ``d_k`` de los codigos LEXICALES ``K_lex`` (aislados).
        d_sense: Dimension de los codigos de SENTIDO ``K_sense`` y la consulta
            contextual ``q_ctx``. Si es None usa ``d_recall``.
        rt_sense_beta_init: Valor inicial del peso ``beta`` del termino de
            sentido (``score = score_lex*(1+beta*score_sense)``).
        rt_sense_beta_trainable: Si ``beta`` se aprende durante el entrenamiento.
        rt_gate_init: Valor inicial de la compuerta de inyeccion ``g_rt`` (valor
            final de ``2*sigmoid(param)``, en [0,2]). Por defecto 1.0: la
            recuperacion aporta desde el paso 0, pero suavemente (``W_r`` se
            inicia con std 0.02). Pasa 0.0 para arrancar de LM pura y dejar que
            la recuperacion arranque sola.
        rt_score_threshold: Umbral del mejor score (en escala coseno ~[0,1+beta])
            por debajo del cual se dispara el escaneo denso de respaldo.
        rt_fallback_window: Ventana del escaneo denso de respaldo (tokens).
        rt_gap: Distancia minima (en tokens) entre la posicion actual y el match.
        rt_temperature: Temperatura del gradiente straight-through (solo backward).
        rt_score_chunk: Filas de puntuacion por trozo (control de VRAM).
        rt_train_mode: ``dense`` (exacto O(N^2 d_k), pequenas N) o ``lsh``
            (lineal O(N*C*d_k)). La generacion incremental es siempre exacta.
        logit_cap: Cota del softcap del evocador (0 desactiva).
        page_size: Tamano de pagina de la traza dual (tokens).
    """

    vocab_size: int = 256
    d_model: int = 256
    d_gate: int = 32
    d_ff: int = 1024
    num_cells: int = 8
    num_encoder_layers: int = 2
    num_consolidation_layers: int = 9
    context_length: int = 8192
    synapse_rank: int = 32
    num_candidates: int = 4
    dropout: float = 0.0
    activation: str = "silu"
    tie_embeddings: bool = True
    offset_mode: str = "resonant_multirate"
    offsets: Optional[Sequence[int]] = None
    dual_bilinear_clamp: float = 4.0
    count_normalize: bool = True
    trace_tap: bool = True
    dtype: str = "float32"

    # --- Recall Tap asimetrico (Pilar 4) -----------------------------------
    recall_enabled: bool = True
    d_recall: int = 64
    d_sense: Optional[int] = None
    rt_value: str = "next"
    rt_gap: int = 1
    rt_temperature: float = 0.5
    rt_score_chunk: int = 1024
    rt_gate_init: float = 1.0
    rt_init_std: float = 0.1
    rt_shared_lex_init: bool = True
    rt_sense_beta_init: float = 0.3
    rt_sense_beta_trainable: bool = True
    rt_score_threshold: float = 0.5
    rt_fallback_window: int = 2048
    rt_use_identity_fast_path: bool = True
    rt_ctx_query: str = "final"   # "final" (T_L) | "shallow" (capa 0)

    # --- Tap SEMANTICO (asociativo, agujas semanticas) ---------------------
    # Recupera por SIGNIFICADO sobre todos los candidatos (no mismo token).
    # Convive con el tap lexico (copia exacta). No es atencion: argmax duro +
    # lectura unica, linealizable por LSH. Claves desde la consolidacion FINAL.
    semantic_recall_enabled: bool = True
    d_semantic: int = 64
    rt_sem_temperature: float = 0.1
    rt_sem_gate_init: float = 1.0
    # Fuente de las claves/consultas semanticas. "t_last" = consolidacion final
    # (significado contextual); "t0" = huella AISLADA pristina T0[j]=f(x_j) (Pilar
    # 1): retiene la identidad de token -> claves distinguibles por par, lo que
    # vuelve separables las asociaciones alias (b_i~a_i). Ambas respetan el
    # aislamiento; "t0" es la opcion por defecto (mas fiel al aislamiento y
    # empiricamente resuelve la aguja asociativa).
    rt_sem_key_source: str = "t0"
    rt_sem_query_source: str = "t0"
    # Modo de la lectura semantica paralela: "dense" = exacto O(N^2) (validado,
    # 100% en la aguja asociativa); "lsh" = lineal O(N*C*d_sem) aproximado (LSH
    # de planos compartidos, util a contexto largo con cierta perdida). El camino
    # incremental (generacion) es siempre exacto O(N) por token.
    rt_sem_recall_mode: str = "dense"

    # --- LSH cuantizado (Pilar 5) ------------------------------------------
    rt_train_mode: str = "lsh"
    rt_lsh_tables: int = 2
    rt_lsh_bits: int = 8           # bits por tabla (proyeccion del sign-code de 64)
    rt_lsh_cap: int = 64
    rt_lsh_neg: int = 48
    rt_lsh_hamming: int = 3

    # --- Evocador con softcap (Pilar 6) ------------------------------------
    logit_cap: float = 30.0

    # --- Traza paginada (Pilar 2) ------------------------------------------
    page_size: int = 256

    # --- Entrenamiento: CE_retrieval auto-supervisado (Seccion 6) ----------
    # INTERNA y OBLIGATORIA por defecto: senal auto-supervisada (sin etiquetas)
    # que activa el eje de sentido. retrieval_anneal_steps=0 => siempre activa.
    # rw=1.0 (default): la recuperacion pesa igual que el LM. Probado: desambigua
    # polisemia al 99 % (rw=0.3 -> 52 %, rw=2.0 -> 100 %). Bajalo (0.2) para
    # tareas de copia pura donde no hace falta sentido.
    retrieval_weight: float = 1.0
    retrieval_anneal_steps: int = 0
    retrieval_positions_frac: float = 0.25
    # Restringe la CE a candidatos del MISMO token (donde el sentido desempata):
    # concentra el gradiente en el eje de sentido (no lo diluye con copia de
    # tokens distintos, que ya aprende la LM). Temperatura aguda para que la
    # softmax diferencie entre apariciones del mismo token por contexto.
    retrieval_same_token_only: bool = True
    retrieval_temperature: float = 0.1

    def __post_init__(self) -> None:
        if self.offset_mode not in _OFFSET_MODES:
            raise ValueError(f"offset_mode debe ser {_OFFSET_MODES}, no {self.offset_mode!r}")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"activation debe ser {_ACTIVATIONS}, no {self.activation!r}")
        if self.rt_value not in ("next", "self"):
            raise ValueError("rt_value debe ser 'next' o 'self'")
        if self.rt_ctx_query not in ("final", "shallow"):
            raise ValueError("rt_ctx_query debe ser 'final' o 'shallow'")
        if self.rt_train_mode not in _RT_TRAIN_MODE:
            raise ValueError(f"rt_train_mode debe ser {_RT_TRAIN_MODE}, no {self.rt_train_mode!r}")
        if self.rt_sem_recall_mode not in _RT_TRAIN_MODE:
            raise ValueError(f"rt_sem_recall_mode debe ser {_RT_TRAIN_MODE}")
        if self.rt_sem_key_source not in ("t0", "t_last"):
            raise ValueError("rt_sem_key_source debe ser 't0' o 't_last'")
        if self.rt_sem_query_source not in ("t0", "t_last"):
            raise ValueError("rt_sem_query_source debe ser 't0' o 't_last'")
        if self.d_gate >= self.d_model:
            raise ValueError("d_gate debe ser < d_model")
        if not 1 <= self.synapse_rank <= self.d_model:
            raise ValueError("synapse_rank debe estar en [1, d_model]")
        if self.d_sense is None:
            object.__setattr__(self, "d_sense", self.d_recall)
        if self.rt_gap < 1:
            raise ValueError("rt_gap >= 1 (el valor leido debe ser causal)")
        if not 1 <= self.d_semantic <= 4096:
            raise ValueError("d_semantic debe estar en [1, 4096]")
        if not 0.0 <= self.retrieval_positions_frac <= 1.0:
            raise ValueError("retrieval_positions_frac debe estar en [0, 1]")
        if not 0.0 < self.retrieval_temperature <= 1.0:
            raise ValueError("retrieval_temperature debe estar en (0, 1]")
        if self.context_length < 2:
            raise ValueError("context_length >= 2")
        if self.page_size < 8:
            raise ValueError("page_size >= 8")

    # ------------------------------------------------------------------
    # Offsets por capa (suavizado multiescala; la recuperacion es del RT)
    # ------------------------------------------------------------------
    def layer_offsets(self, layer_idx: int) -> List[int]:
        if not 0 <= layer_idx < self.num_consolidation_layers:
            raise IndexError("layer_idx fuera de rango")
        if self.offset_mode == "dense_dilated":
            base = list(self.offsets or [0, 1, 2, 4, 8, 16, 32, 64, 128])
        else:  # resonant_multirate  D_l = {0, 1, 2^{l-1}, 2^l}
            if layer_idx == 0:
                base = [0, 1]
            else:
                base = [0, 1, 2 ** (layer_idx - 1), 2 ** layer_idx]
        cap = 2 ** max(1, self.num_consolidation_layers - 1)
        return sorted({p for p in base if p <= cap})

    def layer_offsets_all(self) -> List[List[int]]:
        return [self.layer_offsets(l) for l in range(self.num_consolidation_layers)]

    def receptive_field(self) -> Dict[str, object]:
        reachable = {0}
        for offs in self.layer_offsets_all():
            reachable = {r + p for r in reachable for p in offs}
        reach = max(reachable)
        return {
            "max_reach": reach,
            "dense_coverage": all(i in reachable for i in range(reach + 1)),
            "layers": self.num_consolidation_layers,
            "layer_offsets": self.layer_offsets_all(),
        }

    def cache_horizons(self) -> List[int]:
        offs = self.layer_offsets_all()
        return [max(offs[l + 1]) + 1 if l + 1 < len(offs) else 1
                for l in range(len(offs))]

    def torch_dtype(self):
        import torch
        return getattr(torch, self.dtype)

    def describe(self) -> str:
        rf = self.receptive_field()
        return (
            f"ENGRAMA V5.5  d={self.d_model} dg={self.d_gate} r={self.synapse_rank} "
            f"L={self.num_consolidation_layers} (alcance {rf['max_reach']}) N_max={self.context_length}\n"
            f"  mezcla: count_normalize={self.count_normalize} "
            f"clamp={self.dual_bilinear_clamp} trace_tap={self.trace_tap} "
            f"norm=rmsnorm cell=swiglu\n"
            f"  recall asim: d_k={self.d_recall} d_sense={self.d_sense} "
            f"beta0={self.rt_sense_beta_init} trainable={self.rt_sense_beta_trainable}\n"
            f"  fallback: threshold={self.rt_score_threshold} window={self.rt_fallback_window} "
            f"identity={self.rt_use_identity_fast_path}\n"
            f"  lsh: mode={self.rt_train_mode} t={self.rt_lsh_tables} b={self.rt_lsh_bits} "
            f"cap={self.rt_lsh_cap} neg={self.rt_lsh_neg} hamming={self.rt_lsh_hamming}\n"
            f"  evoker: logit_cap={self.logit_cap}  trace: page={self.page_size}"
        )

    def to_dict(self) -> Dict[str, object]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "V55Config":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    @classmethod
    def from_preset(cls, size: str, **overrides) -> "V55Config":
        presets = {
            "tiny": dict(d_model=64, d_gate=16, d_ff=256, num_cells=2,
                         num_encoder_layers=1, num_consolidation_layers=6,
                         context_length=256, synapse_rank=16, d_recall=32,
                         d_sense=32, rt_layers_note="single tap"),
            "small": dict(d_model=128, d_gate=16, d_ff=512, num_cells=4,
                          num_encoder_layers=1, num_consolidation_layers=8,
                          context_length=1024, synapse_rank=16, d_recall=64,
                          d_sense=64),
            "base": dict(d_model=256, d_gate=32, d_ff=1024, num_cells=8,
                         num_encoder_layers=2, num_consolidation_layers=9,
                         context_length=8192, synapse_rank=32, d_recall=64,
                         d_sense=64),
            "large": dict(d_model=512, d_gate=64, d_ff=2048, num_cells=16,
                          num_encoder_layers=2, num_consolidation_layers=11,
                          context_length=32768, synapse_rank=32, d_recall=96,
                          d_sense=96),
        }
        if size not in presets:
            raise ValueError(f"preset {size!r} desconocido: {tuple(presets)}")
        kw = {k: v for k, v in presets[size].items() if k != "rt_layers_note"}
        kw.update(overrides)
        return cls(**kw)
