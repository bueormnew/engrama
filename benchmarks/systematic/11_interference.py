"""Prueba 11 — Interferencia entre huellas (sin aprendizaje).

Construimos asociaciones:
  A -> X
  B -> Y
  C -> Z
y anadimos despues:
  A -> Q
Comprobamos que:
  A -> Q   (nueva asociacion)
  B -> Y   (intacta)
  C -> Z   (intacta)
Esto mide si escribir una nueva huella contamina las anteriores (filosofia de
huellas aisladas de ENGRAMA).

Lo implementamos con el tap lexico: dos apariciones del token A deben
desempatarse por sentido (K_sense), mientras que B y C permanecen intactas.
Repetimos en cascada para medir la tasa de contaminacion al crecer N.
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

from engrama.v55.recall import RecallTapV2, _l2, _lex_dominant_argmax


def run():
    torch.set_num_threads(2)
    d, ds = 64, 64
    recall = None  # no necesitamos el modulo; solo el argmax lexico-dominante
    results = []
    print(f"{'N':>7} {'A->Q':>6} {'A_old':>6} {'B->Y':>6} {'C->Z':>6} {'contam%':>8}")
    # n_extra grande con dk=V y score denso O(N^2*dk) no cabe en CPU; el limite
    # demuestra por si solo que el camino denso no escala.
    for n_extra in [0, 10, 100, 1000]:
        # Usamos IDs de token UNICOS y una dimension lexica dk >= V para que las
        # claves lexicas sean ortogonales (sin colisiones): es la condicion que
        # exige la filosofia de huellas aisladas.
        V = 16 + n_extra
        dk = V  # una dimension por token -> ortogonalidad exacta
        g = torch.Generator().manual_seed(1)
        # secuencia: A=0,X=1,B=2,Y=3,C=4,Z=5, extras, A=0,Q=7
        seq_tokens = [0, 1, 2, 3, 4, 5] + list(range(8, 8 + n_extra)) + [0, 7]
        n = len(seq_tokens)
        k_lex = torch.zeros(n, dk)
        q_lex = torch.zeros(n, dk)
        for i, t in enumerate(seq_tokens):
            k_lex[i, t] = 1.0
            q_lex[i, t] = 1.0
        # K_sense: clave del "sentido" que desempata. Para la segunda A (nueva)
        # su K_sense debe coincidir con la consulta q_ctx de la posicion Q
        # (que es donde se pregunta "que vino despues de la nueva A?").
        # Hacemos que la K_sense de la nueva A sea un vector unico S_new, y la
        # q_ctx en la posicion de Q sea S_new (-> elige la nueva A). La K_sense
        # de la vieja A es S_old; B y C tienen sentidos unicos.
        k_sen = torch.randn(n, ds, generator=g)
        k_sen = torch.nn.functional.normalize(k_sen, dim=-1)
        q_ctx = torch.zeros(n, ds)
        # La consulta en la posicion que SIGUE a cada token (donde se evoca)
        # debe apuntar a la clave de sentido de la ocurrencia correcta.
        # Valor esperado: en la posicion del value (X/Y/Z/Q) se recupera el T0
        # del token anterior. La consulta q_ctx[i] se situa en i. Fijamos:
        #   i=1 (X): q_ctx = k_sen[0] (vieja A) -> recupera T0[0]
        #   i=3 (Y): q_ctx = k_sen[2] (B)
        #   i=5 (Z): q_ctx = k_sen[4] (C)
        #   i=ultima (Q): q_ctx = k_sen[pos de la NUEVA A]
        pos_new_A = n - 2
        q_ctx[1] = k_sen[0]   # pregunta en X: debe leer la vieja A (pos 0)
        q_ctx[3] = k_sen[2]   # pregunta en Y: debe leer B (pos 2)
        q_ctx[5] = k_sen[4]   # pregunta en Z: debe leer C (pos 4)
        q_ctx[n - 1] = k_sen[pos_new_A]  # pregunta en Q: debe leer la NUEVA A
        # valores: t0 unico por posicion (para comprobar cual se lee)
        t0 = torch.zeros(n, d)
        for i in range(n):
            t0[i, i % d] = float(i + 1)  # valor distinto y marcado
        # correr el argmax lexico-dominante
        ql = _l2(q_lex); kl = _l2(k_lex)
        qc = _l2(q_ctx); ks = _l2(k_sen)
        slex = ql @ kl.T
        ssen = qc @ ks.T
        col = torch.arange(n).view(1, n)
        row = torch.arange(n).view(n, 1)
        mask = col <= (row - 1)
        jstar = _lex_dominant_argmax(slex, ssen, mask)
        # valor leido = t0[jstar+1] (value="next")
        def value_at(i):
            j = jstar[i]
            if j < 0 or j + 1 >= n:
                return None
            return t0[j + 1]
        # esperado:
        #   en i=1 (pregunta tras vieja A): j* debe ser 0 -> valor t0[1] (X)
        #   en i=3: j*=2 -> t0[3] (Y)
        #   en i=5: j*=4 -> t0[5] (Z)
        #   en i=ultima (pregunta tras nueva A): j*=pos_new_A -> t0[ultima]=Q
        exp = {1: t0[1], 3: t0[3], 5: t0[5], n - 1: t0[n - 1]}
        checks = {}
        for i, ev in exp.items():
            v = value_at(i)
            checks[i] = (v is not None and torch.allclose(v, ev))
        # A->Q: la ultima posicion debe leer la nueva A (j*=pos_new_A)
        a_to_q = (jstar[n - 1].item() == pos_new_A)
        b_ok = checks[3]
        c_ok = checks[5]
        a_old_ok = checks[1]
        contam = 0 if (b_ok and c_ok) else 1
        results.append({
            "N": n, "n_extra": n_extra,
            "A_to_Q": bool(a_to_q),
            "A_old_to_X": bool(a_old_ok),
            "B_to_Y": bool(b_ok),
            "C_to_Z": bool(c_ok),
            "contamination": int(contam),
        })
        print(f"{n:>7} {str(a_to_q):>6} {str(b_ok):>6} {str(c_ok):>6} "
              f"{contam*100:>8}")
        del k_lex, q_lex, k_sen, q_ctx, t0; gc.collect()

    result = {
        "test": "11_interference", "rows": results,
        "no_contamination": all(r["contamination"] == 0 for r in results),
        "all_correct": all(r["A_to_Q"] and r["B_to_Y"] and r["C_to_Z"]
                           for r in results),
    }
    path = save_json("11_interference.json", result)
    print("Guardado:", path)


if __name__ == "__main__":
    run()
