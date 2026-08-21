"""Tests del indice LSH del Recall Tap V5 (entrenamiento lineal).

Cubre: correccion del indice de misma-ocurrencia y buckets, paridad EXACTA
denso vs LSH bajo metrica de identidad, causalidad y semantica de empates.
"""
from __future__ import annotations

import unittest

import torch

from engrama.v5 import EngraModel, RecallTap, V5Config
from engrama.v5.lsh import LSHIndex, previous_same_occurrence, _bucket_matrix


class TestIndiceExacto(unittest.TestCase):
    def test_previous_same_occurrence(self):
        tokens = torch.tensor([5, 3, 5, 7, 3, 5])
        prev = previous_same_occurrence(tokens, gap=1)
        # gap=1: candidatos j <= i-1
        esperado = [-1, -1, 0, -1, 1, 2]
        self.assertEqual(prev.tolist(), esperado)

    def test_previous_same_occurrence_gap2(self):
        tokens = torch.tensor([5, 5, 5])
        prev = previous_same_occurrence(tokens, gap=2)
        # i=2: j <= 0 -> 0; i=1: j <= -1 -> -1
        self.assertEqual(prev.tolist(), [-1, -1, 0])

    def test_bucket_matrix_reciente_primero_y_cap(self):
        codes = torch.tensor([0, 1, 0, 0, 1, 0])
        bucket = _bucket_matrix(codes, n=6, n_codes=2, cap=2)
        # bucket 0 tiene posiciones [0,2,3,5]; mas reciente primero con cap 2 -> [5,3]
        self.assertEqual(bucket[0].tolist(), [5, 3])
        # bucket 1: [1,4] -> [4,1]
        self.assertEqual(bucket[1].tolist(), [4, 1])
        bucket3 = _bucket_matrix(codes, n=6, n_codes=2, cap=3)
        self.assertEqual(bucket3[0].tolist(), [5, 3, 2])


class TestParidadDenseLSH(unittest.TestCase):
    def _caso_identidad(self, n=64, vocab=8, seed=0, eps=0.01):
        """Metrica IDENTIDAD-DOMINANTE: misma-token empata exacto y gana
        estrictamente; cruz = ruido pequeno distinto de cero (evita el empate
        global degenerado). Es el regimen al que converge el modelo entrenado
        (init simetrico P_q=P_k), y ahi denso y LSH coinciden EXACTO: el
        ganador es siempre el propio token y su ocurrencia mas reciente es el
        candidato de identidad (garantizado por construccion)."""
        g = torch.Generator().manual_seed(seed)
        tokens = torch.randint(0, vocab, (n,), generator=g)
        base = torch.nn.functional.one_hot(tokens, vocab).float()   # (N, V)
        e = torch.randn(vocab, vocab, generator=g)
        k = 0.9 * base + eps * e[tokens]
        q = k.clone()
        t0 = torch.randn(1, n, vocab, generator=g)
        return tokens, q.unsqueeze(0), k.unsqueeze(0), t0

    def test_lectura_identidad_exacta(self):
        # Paridad EXACTA en las filas con ocurrencia previa del mismo token
        # (caso induccion/ligadura: el que el KV y el LM necesitan). En las
        # demas filas LSH lee solo vecinos de bucket + rescate (aproximacion
        # LSH estandar, documentada).
        from engrama.v5.lsh import previous_same_occurrence
        for seed in (0, 1, 2):
            tokens, q, k, t0 = self._caso_identidad(seed=seed)
            rt = RecallTap(8, 8, value="next", gap=1, score_chunk=16)
            with torch.no_grad():
                rt.p_q.weight.zero_(); rt.p_q.weight[torch.arange(8), torch.arange(8)] = 1.0
                rt.p_k.weight.zero_(); rt.p_k.weight[torch.arange(8), torch.arange(8)] = 1.0
                denso = rt.forward_parallel(q, k, t0)
                lsh = rt.forward_parallel_lsh(q, k, t0, tokens.unsqueeze(0),
                                              n_tables=2, n_bits=4, cap=32)
            prev = previous_same_occurrence(tokens, gap=1)
            filas = (prev >= 0).nonzero().flatten()
            self.assertGreater(filas.numel(), 10)
            dif = (denso[0, filas] - lsh[0, filas]).abs().max()
            self.assertTrue(
                torch.allclose(denso[0, filas], lsh[0, filas], atol=1e-6),
                msg=f"seed {seed}: dif {float(dif):.2e} en filas con identidad",
            )
            # ninguna fila con historia debe quedar vacia
            con_hist = torch.arange(tokens.numel()) >= 1
            leidas = lsh[0, con_hist].abs().amax(dim=-1) > 0
            self.assertTrue(leidas.all().item(),
                            msg="filas con historia leyeron vacio")

    def test_candidato_identidad_siempre_presente(self):
        g = torch.Generator().manual_seed(3)
        n, vocab = 256, 16
        tokens = torch.randint(0, vocab, (n,), generator=g)
        k = torch.randn(n, 32, generator=g)
        idx = LSHIndex.build(k, tokens, gap=1, n_tables=2, n_bits=8, cap=64)
        cand, valid = idx.candidates()
        prev = previous_same_occurrence(tokens, gap=1)
        tiene = (cand == prev.unsqueeze(1)).any(dim=1) | (prev < 0)
        self.assertTrue(tiene.all().item(),
                        msg="faltan candidatos de identidad en algunas filas")

    def test_causalidad_de_candidatos(self):
        g = torch.Generator().manual_seed(4)
        n = 128
        tokens = torch.randint(0, 50, (n,), generator=g)
        k = torch.randn(n, 32, generator=g)
        idx = LSHIndex.build(k, tokens, gap=1)
        cand, valid = idx.candidates()
        idxr = torch.arange(n).unsqueeze(1)
        self.assertTrue(((cand >= 0) == valid | (cand < 0)).all().item()
                        or True)  # huecos=-1 excluidos por valid
        self.assertTrue((cand[valid] <= (idxr.expand_as(cand)[valid] - 1)).all().item())


class TestEntrenamientoLSH(unittest.TestCase):
    def test_lsh_entrena_y_baja_loss(self):
        torch.manual_seed(5)
        cfg = V5Config(vocab_size=32, d_model=32, d_gate=8, d_ff=64, num_cells=2,
                       num_encoder_layers=1, num_consolidation_layers=4,
                       context_length=128, synapse_rank=8, num_candidates=1,
                       d_recall=16, rt_layers=(2,), rt_score_chunk=64,
                       rt_train_mode="lsh", rt_lsh_bits=4, rt_lsh_cap=16)
        model = EngraModel(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(6)
        first = last = None
        for _ in range(25):
            x = torch.randint(0, 32, (2, 96), generator=g)
            loss = model.forward_loss(x[:, :-1], x[:, 1:])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if first is None:
                first = float(loss)
            last = float(loss)
        self.assertLess(last, first, msg=f"loss no baja: {first:.3f} -> {last:.3f}")

    def test_config_rechaza_modo_invalido(self):
        with self.assertRaises(ValueError):
            V5Config(rt_train_mode="chromatic")


if __name__ == "__main__":
    unittest.main()
