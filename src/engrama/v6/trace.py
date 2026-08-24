"""ENGRAMA V5.5 — Traza dual PAGINADA (Pilar 2).

Memoria explicita, SIN comprimir, escalable por paginacion (estilo
PagedAttention). Cada posicion guarda una tupla:

    Trace[j] = ( T0[j] pristino,           # huella aislada (d)  -- pilar 2 V5
                 T_shallow[j],             # salida de capa 0 (d), contexto 2 tok
                 K_lex[j],                 # codigo lexico aislado (d_k)
                 K_sense[j],               # codigo de sentido (d_sense)
                 token_id[j], timestamp )

* ``T0`` y ``T_shallow`` en la precision de computo (1.2 KB/token en fp16,
  2.4 KB/token en fp32; lineal en cualquier caso). 16k ~ 20-40 MB.
* Paginas de ``page_size`` (256) tokens preasignadas: append es O(1) (escribir
  en el slot de la pagina actual), cero ``torch.cat``. Cuando se llena una
  pagina se abre una nueva; bajo capacidad finita se recicla la mas antigua
  (FIFO/ventana deslizante).
* Almacenamiento BATCHED ``(page, B, d)``: soporta inferencia/generacion por
  lotes (igual que el anillo de V5), preservando la invarianza causal.

La lectura lineal concatena las paginas activas en un tensor contiguo. 100 %
compatible con el ``get_cache()`` de V5: mismos horizontes minimos por capa de
consolidacion.

El *fast path* de identidad guarda ``{(batch, token_id) -> ultima posicion}``
para recuperar en O(1) la ocurrencia previa del mismo token (Pilar 4).

Author: Gerson Fabian Buenahora Ormaza (BUEORM)
License: AGPL-3.0
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional, Sequence, Tuple

import torch


class PagedDualTrace:
    """Traza explicita paginada de ENGRAMA V5.5 (nunca comprime)."""

    def __init__(
        self,
        n_max: int,
        d_model: int,
        d_recall: int,
        d_sense: int,
        horizons: Sequence[int],
        *,
        page_size: int = 256,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        store_dtype: Optional[torch.dtype] = None,
        d_semantic: int = 0,
    ):
        if n_max < 2:
            raise ValueError("n_max >= 2")
        self.n_max = int(n_max)
        self.d_model = int(d_model)
        self.d_recall = int(d_recall)
        self.d_sense = int(d_sense)
        self.d_semantic = int(d_semantic)
        self.horizons = [int(h) for h in horizons]
        self.page_size = int(page_size)
        self._device = device
        self.store_dtype = store_dtype if store_dtype is not None else dtype
        self._dtype = dtype
        self._batch = None              # B (lazy, se fija en el primer append)

        self._pages: List[dict] = []
        self.length = 0          # tokens escritos (logico)
        self._start = 0          # tokens descartados por FIFO
        self.layer_buffers: List[deque] = [deque(maxlen=h) for h in self.horizons]
        self._last_pos_of_token: dict = {}    # {(bi, tid): pos}
        # Indice LSH incremental para recall semantico O(C) por token en
        # inferencia (uno por elemento del batch). Se crea perezosamente.
        self._lsh_indexes: Optional[list] = None
        self._lsh_cfg: Optional[dict] = None

    def attach_semantic_lsh(self, *, n_tables: int, n_bits: int, cap: int,
                            rescue_window: int, gap: int, seed: int = 7,
                            batch_size: Optional[int] = None) -> None:
        """Configura (o reemplaza) el indice LSH semantico incremental.

        Se llama una vez tras ``get_cache``. Cada elemento del batch tendra
        su propio :class:`IncrementalLSHIndex` con los mismos planos
        deterministas que el indice estatico usado en entrenamiento.
        """
        from engrama.v6.lsh import IncrementalLSHIndex
        b = int(batch_size) if batch_size is not None else (
            self._batch if self._batch is not None else 1)
        n_codes = 1 << int(n_bits)
        self._lsh_cfg = dict(n_tables=int(n_tables), n_bits=int(n_bits),
                             cap=int(cap), rescue_window=int(rescue_window),
                             gap=int(gap), seed=int(seed))
        self._lsh_indexes = [
            IncrementalLSHIndex(n_codes, int(n_tables), int(cap),
                                int(rescue_window), int(gap),
                                self.n_max, self._device, seed=int(seed))
            for _ in range(b)]

    def semantic_lsh(self, batch_idx: int = 0):
        """Devuelve el indice LSH incremental de un elemento del batch."""
        if self._lsh_indexes is None:
            return None
        return self._lsh_indexes[batch_idx]

    def ensure_semantic_lsh_batch(self, batch_size: int) -> None:
        """Amplia perezosamente los indices LSH si el batch real es mayor.

        ``get_cache`` no conoce el tamano de batch antes del primer append,
        asi que inicialmente crea un solo indice. Al llegar el primer token
        con B>1 hay que crear los indices faltantes con la MISMA config.
        """
        if self._lsh_indexes is None or self._lsh_cfg is None:
            return
        have = len(self._lsh_indexes)
        if batch_size <= have:
            return
        from engrama.v6.lsh import IncrementalLSHIndex
        cfg = self._lsh_cfg
        n_codes = 1 << int(cfg["n_bits"])
        for _ in range(batch_size - have):
            self._lsh_indexes.append(
                IncrementalLSHIndex(
                    n_codes, int(cfg["n_tables"]), int(cfg["cap"]),
                    int(cfg["rescue_window"]), int(cfg["gap"]),
                    self.n_max, self._device, seed=int(cfg["seed"])))

    # ------------------------------------------------------------------
    def _alloc_page(self) -> None:
        dev, st = self._device, self.store_dtype
        b = self._batch if self._batch is not None else 1
        page = {
            "t0": torch.zeros(self.page_size, b, self.d_model, device=dev, dtype=st),
            "ts": torch.zeros(self.page_size, b, self.d_model, device=dev, dtype=st),
            "klex": torch.zeros(self.page_size, b, self.d_recall, device=dev, dtype=st),
            "ksen": torch.zeros(self.page_size, b, self.d_sense, device=dev, dtype=st),
            "tok": torch.full((self.page_size, b), -1, dtype=torch.long, device=dev),
        }
        if self.d_semantic > 0:
            page["ksem"] = torch.zeros(self.page_size, b, self.d_semantic,
                                       device=dev, dtype=st)
        self._pages.append(page)

    def _ensure_batch(self, t0: torch.Tensor) -> None:
        b = t0.size(0) if t0.dim() >= 2 else 1
        if self._batch is None:
            self._batch = b
            self._alloc_page()
        elif b != self._batch:
            raise ValueError(f"batch inconsistente en la traza: {self._batch} vs {b}")

    def _page_slot(self, logical: int) -> Tuple[int, int]:
        return logical // self.page_size, logical % self.page_size

    # ------------------------------------------------------------------
    def append_t0(self, t0: torch.Tensor, k_lex: Optional[torch.Tensor] = None,
                  token_id: Optional[torch.Tensor] = None) -> None:
        """Escribe la huella del token actual ``t0`` ``(B, d)`` (o ``(d,)``).

        ``k_lex`` ``(B, d_k)`` y ``token_id`` ``(B,)`` opcionales. T_shallow y
        K_sense se anaden con :meth:`append_shallow`.
        """
        if t0.dim() == 1:
            t0 = t0.unsqueeze(0)
        self._ensure_batch(t0)
        self.ensure_semantic_lsh_batch(t0.size(0))
        if self.length - self._start >= self.n_max:
            self._start += 1
        logical = self.length
        pi, si = self._page_slot(logical)
        while pi >= len(self._pages):
            self._alloc_page()
        page = self._pages[pi]
        page["t0"][si] = t0.to(self.store_dtype)
        if k_lex is not None:
            if k_lex.dim() == 1:
                k_lex = k_lex.unsqueeze(0)
            page["klex"][si] = k_lex.to(self.store_dtype)
        if token_id is not None:
            tid = token_id.reshape(-1).to(torch.long)
            page["tok"][si] = tid
            for bi in range(tid.numel()):
                self._last_pos_of_token[(bi, int(tid[bi].item()))] = logical
        self.length += 1

    def append_shallow(self, t_shallow: torch.Tensor,
                       k_sense: Optional[torch.Tensor] = None) -> None:
        """Escribe T_shallow (salida de capa 0) y K_sense del ULTIMO token."""
        if self.length == 0:
            return
        if t_shallow.dim() == 1:
            t_shallow = t_shallow.unsqueeze(0)
        logical = self.length - 1
        pi, si = self._page_slot(logical)
        page = self._pages[pi]
        page["ts"][si] = t_shallow.to(self.store_dtype)
        if k_sense is not None:
            if k_sense.dim() == 1:
                k_sense = k_sense.unsqueeze(0)
            page["ksen"][si] = k_sense.to(self.store_dtype)

    def append_semantic(self, k_sem: torch.Tensor,
                        token_id: Optional[torch.Tensor] = None) -> None:
        """Escribe K_sem (clave semantica) del ULTIMO token en el anillo.

        NO inserta aun en el indice LSH (ver :meth:`commit_semantic_lsh`):
        la lectura semantica del token actual debe ver solo posiciones
        previas, asi que el indice se actualiza despues de leer.
        """
        if self.length == 0 or self.d_semantic <= 0:
            return
        if k_sem.dim() == 1:
            k_sem = k_sem.unsqueeze(0)
        logical = self.length - 1
        pi, si = self._page_slot(logical)
        page = self._pages[pi]
        page["ksem"][si] = k_sem.to(self.store_dtype)

    def commit_semantic_lsh(self, k_sem: torch.Tensor,
                            token_id: Optional[torch.Tensor] = None) -> None:
        """Inserta la clave semantica del ULTIMO token en el indice LSH.

        Debe llamarse DESPUES de la lectura semantica del token actual para
        preservar la mascara causal (el indice no debe contener la posicion
        que esta leyendo).
        """
        if (self.length == 0 or self.d_semantic <= 0
                or self._lsh_indexes is None or self._lsh_cfg is None):
            return
        if k_sem.dim() == 1:
            k_sem = k_sem.unsqueeze(0)
        from engrama.v6.lsh import shared_planes, sign_codes
        cfg = self._lsh_cfg
        b = k_sem.size(0)
        d = k_sem.size(-1)
        planes = shared_planes(cfg["n_tables"], d, cfg["n_bits"],
                               k_sem.device, k_sem.dtype, seed=cfg["seed"])
        codes = sign_codes(k_sem.float(), planes)   # (b, t)
        tids = (token_id.reshape(-1).tolist()
                if token_id is not None else [-1] * b)
        for bi in range(b):
            self._lsh_indexes[bi].add(codes[bi], int(tids[bi]))

    def append_layer(self, layer: int, state: torch.Tensor) -> None:
        self.layer_buffers[layer].append(state)

    # ------------------------------------------------------------------
    def last_occurrence(self, token_id: int, batch_idx: int = 0,
                        before_pos: int = -1) -> int:
        """Ultima posicion logica con ``token_id`` (lote ``batch_idx``) y
        ``< before_pos`` (O(1)). -1 si no existe o fue descartada por FIFO.

        Debe llamarse ANTES de escribir el token actual.
        """
        pos = self._last_pos_of_token.get((int(batch_idx), int(token_id)), -1)
        if pos < 0 or pos < self._start:
            return -1
        if before_pos >= 0 and pos >= before_pos:
            return -1
        return pos

    def _linear(self, key: str, count: Optional[int] = None) -> torch.Tensor:
        n = count if count is not None else (self.length - self._start)
        b = self._batch or 1
        last_dim = self._dim_of(key)
        if n <= 0 or not self._pages:
            return torch.zeros(0, b, last_dim, device=self._device,
                               dtype=self.store_dtype)
        start = self._start
        first_pi, first_si = self._page_slot(start)
        last_pi, last_si = self._page_slot(start + n - 1)
        pieces: List[torch.Tensor] = []
        pi, si = first_pi, first_si
        while True:
            page = self._pages[pi]
            if pi == last_pi:
                pieces.append(page[key][si:last_si + 1])
                break
            pieces.append(page[key][si:])
            pi += 1
            si = 0
            if pi >= len(self._pages):
                break
        return torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]

    # ------------------------------------------------------------------
    def linear_t0(self, count: Optional[int] = None) -> torch.Tensor:
        return self._linear("t0", count)

    def _dim_of(self, key: str) -> int:
        return {"t0": self.d_model, "ts": self.d_model,
                "klex": self.d_recall, "ksen": self.d_sense,
                "ksem": self.d_semantic, "tok": 1}[key]

    def linear_shallow(self, count: Optional[int] = None) -> torch.Tensor:
        return self._linear("ts", count)

    def linear_klex(self, count: Optional[int] = None) -> torch.Tensor:
        return self._linear("klex", count)

    def linear_ksen(self, count: Optional[int] = None) -> torch.Tensor:
        return self._linear("ksen", count)

    def linear_ksem(self, count: Optional[int] = None) -> torch.Tensor:
        if self.d_semantic <= 0:
            b = self._batch or 1
            return torch.zeros(0, b, 1, device=self._device, dtype=self.store_dtype)
        return self._linear("ksem", count)

    def linear_tokens(self, count: Optional[int] = None) -> torch.Tensor:
        return self._linear("tok", count)

    # ------------------------------------------------------------------
    def t0_history(self, count: int) -> List[torch.Tensor]:
        """Ultimas ``count`` huellas ``(B, d)``, de la mas antigua a reciente."""
        count = max(0, min(count, self.length - self._start))
        out: List[torch.Tensor] = []
        base = self.length
        for i in range(count):
            logical = base - count + i
            pi, si = self._page_slot(logical)
            out.append(self._pages[pi]["t0"][si])
        return out

    def layer_history(self, layer: int, count: int) -> List[torch.Tensor]:
        buf = self.layer_buffers[layer]
        return list(buf)[-max(0, count):]

    # ------------------------------------------------------------------
    @property
    def k_ring(self):
        return self.linear_klex()

    @property
    def t0_ring(self):
        return self.linear_t0()

    # ------------------------------------------------------------------
    def memory_bytes(self) -> int:
        active = max(0, self.length - self._start)
        b = self._batch or 1
        per_token = (2 * self.d_model + self.d_recall + self.d_sense) * b
        store = active * per_token * _elem_size(self.store_dtype)
        store += len(self._pages) * self.page_size * max(b, 1) * 8
        layers = sum(x.numel() * x.element_size()
                     for buf in self.layer_buffers for x in buf)
        return int(store + layers)

    def describe(self) -> dict:
        return {
            "n_max": self.n_max, "length": self.length,
            "page_size": self.page_size, "n_pages": len(self._pages),
            "batch": self._batch,
            "bytes_per_token": (2 * self.d_model + self.d_recall + self.d_sense)
                               * _elem_size(self.store_dtype),
            "total_bytes": self.memory_bytes(), "horizons": self.horizons,
        }

    def __len__(self) -> int:
        return max(0, self.length - self._start)


def _elem_size(dtype: torch.dtype) -> int:
    try:
        return torch.tensor([], dtype=dtype).element_size()
    except Exception:
        return 4
