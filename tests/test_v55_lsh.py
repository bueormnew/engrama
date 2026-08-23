"""Tests del indice LSH V5.5 cuantizado (Pilar 5).

Cubre: indice exacto de misma-ocurrencia, buckets, candidato de identidad
siempre presente, causalidad y paridad densa vs LSH bajo metrica lexica
identidad-dominante (el regimen al que converge el modelo).
"""
from __future__ import annotations

import unittest

import torch

from engrama.v55 import RecallTapV2, V55Config
from engrama.v55.lsh import LSHIndexV2, _bucket_matrix, previous_same_occurrence


class TestIndiceExacto(unittest.TestCase):
    def test_previous_same_occurrence(self):
        tokens = torch.tensor([5, 3, 5, 7, 3, 5])
        self.assertEqual(previous_same_occurrence(tokens, gap=1).tolist(),
                         [-1, -1, 0, -1, 1, 2])

    def test_previous_same_occurrence_gap2(self):
        tokens = torch.tensor([5, 5, 5])
        self.assertEqual(previous_same_occurrence(tokens, gap=2).tolist(),
                         [-1, -1, 0])

    def test_bucket_matrix_reciente_primero(self):
        codes = torch.tensor([0, 1, 0, 0, 1, 0])
        bucket = _bucket_matrix(codes, n=6, n_codes=2, cap=2)
        self.assertEqual(bucket[0].tolist(), [5, 3])
        self.assertEqual(bucket[1].tolist(), [4, 1])

    def test_sign_bitpack_deterministic(self):
        from engrama.v55.lsh import sign_bitpack
        k = torch.randn(16, 8)
        c1 = sign_bitpack(k, n_bits=4)
        c2 = sign_bitpack(k, n_bits=4)
        self.assertTrue(torch.equal(c1, c2))   # determinista (semilla fija)


class TestCandidatos(unittest.TestCase):
    def test_identidad_siempre_presente(self):
        g = torch.Generator().manual_seed(3)
        n, vocab = 256, 16
        tokens = torch.randint(0, vocab, (n,), generator=g)
        k = torch.randn(n, 32, generator=g)
        idx = LSHIndexV2.build(k, tokens, gap=1, n_tables=2, n_bits=8, cap=32)
        cand, valid = idx.candidates()
        prev = previous_same_occurrence(tokens, gap=1)
        tiene = (cand == prev.unsqueeze(1)).any(dim=1) | (prev < 0)
        self.assertTrue(tiene.all().item(),
                        msg="faltan candidatos de identidad")

    def test_causalidad_candidatos(self):
        g = torch.Generator().manual_seed(4)
        n = 128
        tokens = torch.randint(0, 50, (n,), generator=g)
        k = torch.randn(n, 32, generator=g)
        idx = LSHIndexV2.build(k, tokens, gap=1)
        cand, valid = idx.candidates()
        idxr = torch.arange(n).unsqueeze(1)
        self.assertTrue((cand[valid] <= (idxr.expand_as(cand)[valid] - 1)).all().item())


class TestParidadDenseLSH(unittest.TestCase):
    def _caso_identidad(self, n=64, vocab=8, seed=0, eps=0.01):
        g = torch.Generator().manual_seed(seed)
        tokens = torch.randint(0, vocab, (n,), generator=g)
        base = torch.nn.functional.one_hot(tokens, vocab).float()
        e = torch.randn(vocab, vocab, generator=g)
        k = 0.9 * base + eps * e[tokens]
        q = k.clone()
        t0 = torch.randn(1, n, vocab, generator=g)
        return tokens, q.unsqueeze(0), k.unsqueeze(0), t0

    def test_lectura_identidad_exacta(self):
        # Bajo metrica identidad-dominante, denso y LSH coinciden EXACTO en las
        # filas con ocurrencia previa del mismo token (caso induccion).
        for seed in (0, 1, 2):
            tokens, q_lex, k_lex, t0 = self._caso_identidad(seed=seed)
            q_ctx = q_lex.clone()
            k_sen = k_lex.clone()
            rt = RecallTapV2(8, 8, 8, value="next", gap=1, score_chunk=16)
            with torch.no_grad():
                rt.p_q_lex.weight.zero_(); rt.p_q_lex.weight[torch.arange(8), torch.arange(8)] = 1.0
                rt.p_k_lex.weight.zero_(); rt.p_k_lex.weight[torch.arange(8), torch.arange(8)] = 1.0
                denso = rt.forward_parallel_dense(q_lex, k_lex, q_ctx, k_sen, t0)
                lsh = rt.forward_parallel_lsh(q_lex, k_lex, q_ctx, k_sen, t0,
                                              tokens.unsqueeze(0),
                                              n_tables=2, n_bits=4, cap=32, n_neg=0)
            prev = previous_same_occurrence(tokens, gap=1)
            filas = (prev >= 0).nonzero().flatten()
            self.assertGreater(filas.numel(), 10)
            self.assertTrue(
                torch.allclose(denso[0, filas], lsh[0, filas], atol=1e-6),
                msg=f"seed {seed}: dif {float((denso[0, filas]-lsh[0, filas]).abs().max()):.2e}",
            )


class TestEntrenamientoLSH(unittest.TestCase):
    def test_lsh_entrena_y_baja_loss(self):
        torch.manual_seed(5)
        cfg = V55Config(vocab_size=32, d_model=32, d_gate=8, d_ff=64, num_cells=2,
                        num_encoder_layers=1, num_consolidation_layers=4,
                        context_length=128, synapse_rank=8, num_candidates=1,
                        d_recall=16, d_sense=16, rt_score_chunk=64,
                        rt_train_mode="lsh", rt_lsh_bits=4, rt_lsh_cap=16,
                        rt_lsh_neg=4, page_size=16)
        from engrama.v55 import EngraModelV55
        model = EngraModelV55(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(6)
        first = last = None
        for _ in range(20):
            x = torch.randint(0, 32, (2, 64), generator=g)
            loss = model.forward_loss(x[:, :-1], x[:, 1:], retrieval_weight=0.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            v = float(loss.detach())
            first = v if first is None else first
            last = v
        self.assertLess(last, first, msg=f"loss no baja: {first:.3f}->{last:.3f}")

    def test_config_rechaza_modo_invalido(self):
        with self.assertRaises(ValueError):
            V55Config(rt_train_mode="chromatic")


if __name__ == "__main__":
    unittest.main()
