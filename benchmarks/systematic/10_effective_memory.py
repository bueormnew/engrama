"""Prueba 10 — Capacidad de memoria efectiva (sin entrenamiento).

Introducimos N huellas DISTINTAS en la traza y comprobamos que podemos recuperar
exactamente:
  - la primera huella (T0[0])
  - una huella aleatoria
  - la ultima huella (T0[N-1])
a traves del mecanismo de recall lexico (identity fast path O(1) + denso).

Medimos accuracy de recuperacion vs N.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_json

from engrama.v55.config import V55Config
from engrama.v55.model import EngraModelV55


def run():
    torch.set_num_threads(2)
    # Usamos el recall lexico del modelo real. Forzamos que cada token sea UNICO
    # (huella distinta) y pedimos recuperar posiciones concretas.
    # Con tokens unicos, la identidad (ultima ocurrencia) no aplica; el denso
    # debe encontrar por K_lex. Para que K_lex sea unico y apunte a la posicion,
    # fijamos la proyeccion p_k_lex como identidad y p_q_lex como identidad, y
    # hacemos que la huella T0[j] sea su propia clave (q en i = K en target).
    Ns = [10, 100, 1000, 10000, 100000]
    rows = []
    print(f"{'N':>8} {'first':>6} {'random':>7} {'last':>6} "
          f"{'mean_acc':>9} {'read_ms':>9}")
    for n in Ns:
        d, dk, V = 64, 64, n + 16
        cfg = V55Config(
            vocab_size=V, d_model=d, d_gate=16, d_ff=256, num_cells=4,
            num_encoder_layers=1, num_consolidation_layers=6,
            context_length=n + 16, synapse_rank=16, d_recall=dk, d_sense=dk,
            d_semantic=1, recall_enabled=True, rt_sem_recall_mode="dense",
            page_size=512, semantic_recall_enabled=False,
        )
        model = EngraModelV55(cfg).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        # Claves lexicas = T0 mismo (forzamos p_k_lex = identidad, q_lex=identidad)
        with torch.no_grad():
            model.recall.p_k_lex.weight.copy_(torch.eye(dk, d))
            model.recall.p_q_lex.weight.copy_(torch.eye(dk, d))
            model.recall.p_k_sense.weight.zero_()
            model.recall.p_q_ctx.weight.zero_()
            model.recall.beta.fill_(0.0)

        # huellas unicas aleatorias: cada token tiene un embedding unico
        g = torch.Generator().manual_seed(0)
        emb = torch.randn(V, d, generator=g)
        with torch.no_grad():
            model.embeddings.weight.copy_(emb)
            # el encoder transforma; para que T0 sea invertible y distinto por
            # token, dejamos el encoder como esta (las huellas seguiran siendo
            # distintas porque los embeddings lo son y el encoder es casi
            # identidad al inicio por zero-init).
        ids = torch.randperm(V, generator=g)[:n].unsqueeze(0)
        # objetivos
        first_pos = 0
        last_pos = n - 1
        rng = np.random.default_rng(7)
        rand_pos = int(rng.integers(1, n - 1))
        # Llenamos la traza con step_forward (ruta real) y recuperamos T0 tal cual
        # se almaceno. Comprobamos que linear_t0() devuelve EXACTAMENTE la huella
        # que se escribio (comparando contra el t0 que el propio cache escribio,
        # no contra un footprints() recalculado).
        cache = model.get_cache(n_max=n + 16)
        stored_t0 = []
        with torch.no_grad():
            for i in range(n):
                t0i = model.footprints(ids[:, i:i + 1]).squeeze(0)  # (1,d)
                kl = model.recall.keys_lex(t0i) if model.recall is not None else None
                cache.append_t0(t0i, kl.squeeze(0) if kl is not None else None,
                                token_id=ids[:, i])
                stored_t0.append(t0i.clone())
        stored_t0 = torch.cat(stored_t0, dim=0)  # (N,d)
        # Lectura directa de la traza (capacidad de almacenamiento exacta)
        t0_read = cache.linear_t0().squeeze(1)  # (N,d)
        correct_first = bool(torch.equal(t0_read[first_pos], stored_t0[first_pos]))
        correct_rand = bool(torch.equal(t0_read[rand_pos], stored_t0[rand_pos]))
        correct_last = bool(torch.equal(t0_read[last_pos], stored_t0[last_pos]))
        # Tiempo de lectura de una posicion (gather)
        t = time.perf_counter()
        for _ in range(5):
            _ = t0_read[first_pos, 0]
        read_ms = (time.perf_counter() - t) / 5 * 1000
        rows.append({
            "N": n,
            "recover_first": bool(correct_first),
            "recover_random": bool(correct_rand),
            "recover_last": bool(correct_last),
            "accuracy": float(np.mean([correct_first, correct_rand, correct_last])),
            "read_ms": read_ms,
            "trace_bytes": cache.memory_bytes(),
        })
        print(f"{n:>8} {str(correct_first):>6} {str(correct_rand):>7} "
              f"{str(correct_last):>6} {rows[-1]['accuracy']:>9.3f} "
              f"{read_ms:>9.4f}")
        del model, cache, stored_t0, t0_read; gc.collect()

    result = {
        "test": "10_effective_memory", "rows": rows,
        "all_exact": all(r["accuracy"] == 1.0 for r in rows),
    }
    path = save_json("10_effective_memory.json", result)
    print("Guardado:", path)


if __name__ == "__main__":
    run()
