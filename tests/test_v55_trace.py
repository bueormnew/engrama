"""Tests de la Traza Dual PAGINADA V5.5 (Pilar 2).

Cubre: append O(1) por paginacion, vistas lineales correctas, ventana FIFO,
indice de identidad O(1), almacenamiento batched y memoria lineal.
"""
from __future__ import annotations

import unittest

import torch

from engrama.v55.trace import PagedDualTrace


class TestPagedAppend(unittest.TestCase):
    def test_append_and_linear_view(self):
        tr = PagedDualTrace(32, 4, 4, 4, horizons=[1], page_size=8)
        for t in range(20):
            v = torch.full((1, 4), float(t))
            tr.append_t0(v, torch.zeros(1, 4), token_id=torch.tensor([t]))
            tr.append_shallow(v * 0.5, torch.zeros(1, 4))
        t0 = tr.linear_t0()
        self.assertEqual(tuple(t0.shape), (20, 1, 4))
        # la posicion i debe contener el valor i
        for i in range(20):
            self.assertTrue(torch.allclose(t0[i, 0], torch.tensor(float(i))))

    def test_crosses_page_boundaries(self):
        # page_size=8, escribe 30 -> 4 paginas; las vistas deben ser contiguas.
        tr = PagedDualTrace(64, 3, 3, 3, horizons=[1], page_size=8)
        for t in range(30):
            tr.append_t0(torch.full((1, 3), float(t)),
                         token_id=torch.tensor([t % 7]))
        t0 = tr.linear_t0()
        self.assertEqual(t0.size(0), 30)
        seq = t0[:, 0, 0].tolist()
        self.assertEqual(seq, [float(i) for i in range(30)])

    def test_append_o1_no_cat(self):
        # Append no debe llamar torch.cat sobre el anillo (paginacion).
        tr = PagedDualTrace(1024, 4, 4, 4, horizons=[1], page_size=256)
        cats = 0
        orig = torch.cat

        def spy(*a, **k):
            nonlocal cats
            cats += 1
            return orig(*a, **k)
        torch.cat = spy
        try:
            for t in range(300):   # > 1 pagina
                tr.append_t0(torch.zeros(1, 4))
        finally:
            torch.cat = orig
        # _linear si usa cat al leer (es esperado), pero append no.
        self.assertEqual(cats, 0)


class TestFIFO(unittest.TestCase):
    def test_fifo_window_drops_oldest(self):
        tr = PagedDualTrace(8, 4, 4, 4, horizons=[1], page_size=4)  # n_max=8
        for t in range(12):
            tr.append_t0(torch.full((1, 4), float(t)),
                         token_id=torch.tensor([t]))
        # solo quedan los ultimos 8 (tokens 4..11)
        self.assertEqual(len(tr), 8)
        t0 = tr.linear_t0()
        self.assertEqual(t0[0, 0, 0].item(), 4.0)
        self.assertEqual(t0[-1, 0, 0].item(), 11.0)

    def test_identity_dict_after_fifo(self):
        tr = PagedDualTrace(8, 4, 4, 4, horizons=[1], page_size=4)
        for t in range(12):
            tr.append_t0(torch.zeros(1, 4), token_id=torch.tensor([t % 3]))
        # token 0 aparece en pos 0,3,6,9; la ultima valida (tras FIFO, start=4)
        # es la 9. last_occurrence debe dar 9.
        lo = tr.last_occurrence(0, batch_idx=0, before_pos=12)
        self.assertEqual(lo, 9)


class TestIdentityIndex(unittest.TestCase):
    def test_last_occurrence_o1(self):
        # Simula el uso real: se consulta ANTES de escribir el token actual.
        tr = PagedDualTrace(32, 4, 4, 4, horizons=[1], page_size=8)
        toks = [5, 3, 5, 7, 3, 5]
        for t in toks:
            tr.append_t0(torch.zeros(1, 4), token_id=torch.tensor([t]))
        # tras escribir todo, la ultima ocurrencia de 5 (< length=6) es la 5.
        self.assertEqual(tr.last_occurrence(5, before_pos=6), 5)
        self.assertEqual(tr.last_occurrence(3, before_pos=6), 4)
        self.assertEqual(tr.last_occurrence(9, before_pos=6), -1)
        # uso real: antes de escribir la 3a aparicion de 5 (pos 5), la previa es la 2
        tr2 = PagedDualTrace(32, 4, 4, 4, horizons=[1], page_size=8)
        for t in [5, 3, 5, 7, 3]:           # aun NO escribimos la pos 5
            tr2.append_t0(torch.zeros(1, 4), token_id=torch.tensor([t]))
        self.assertEqual(tr2.last_occurrence(5, before_pos=tr2.length), 2)


class TestBatched(unittest.TestCase):
    def test_batched_storage(self):
        tr = PagedDualTrace(16, 4, 4, 4, horizons=[1], page_size=8)
        b = 3
        for t in range(10):
            v = torch.full((b, 4), float(t))
            tr.append_t0(v, torch.zeros(b, 4), token_id=torch.tensor([t] * b))
        t0 = tr.linear_t0()
        self.assertEqual(tuple(t0.shape), (10, b, 4))
        self.assertTrue(torch.allclose(t0[5], torch.full((b, 4), 5.0)))


class TestMemory(unittest.TestCase):
    def test_memory_linear(self):
        bpt = []
        for n in (64, 256, 1024):
            tr = PagedDualTrace(n, 32, 16, 16, horizons=[1] * 3, page_size=16)
            bpt.append(tr.memory_bytes() / n)
        slope = (bpt[-1] - bpt[0]) / (1024 - 64)
        self.assertLess(abs(slope), 5.0, msg=f"no lineal: {bpt}")


if __name__ == "__main__":
    unittest.main()
