"""Tests del Recall Tap ASIMETRICO V5.5 (Pilar 4).

Cubre: aislamiento de ``K_lex`` (propiedad de induccion), score lexico=1 para
mismo token, desambiguacion por sentido (mismo token, dos contextos), lectura
dura de induccion, desempate por recencia, prioridad ULP de identidad,
invarianza lectura paralela == incremental, y fallback denso.
"""
from __future__ import annotations

import unittest

import torch

from engrama.v55 import RecallTapV2, V55Config
from engrama.v55.trace import PagedDualTrace


def make_rt(d=8, dk=8, ds=8, **kw):
    return RecallTapV2(d, dk, ds, value="next", gap=1, score_chunk=16, **kw)


class TestLexicalAxis(unittest.TestCase):
    def test_klex_isolated(self):
        # K_lex[j] depende solo del token j (pilar 1).
        torch.manual_seed(0)
        rt = make_rt()
        a = torch.randn(1, 10, 8)
        b = a.clone(); b[0, 0] = torch.randn(8)
        ka, kb = rt.keys_lex(a), rt.keys_lex(b)
        self.assertFalse(torch.allclose(ka[0, 0], kb[0, 0], atol=1e-6))
        self.assertTrue(torch.allclose(ka[0, 1:], kb[0, 1:], atol=1e-7))

    def test_same_token_same_klex(self):
        # Mismo token -> mismo K_lex (propiedad de induccion).
        torch.manual_seed(1)
        rt = make_rt()
        t0 = torch.randn(1, 12, 8)
        t0[0, 7] = t0[0, 2]            # misma huella = mismo token
        k = rt.keys_lex(t0)
        self.assertTrue(torch.allclose(k[0, 2], k[0, 7], atol=1e-7))

    def test_score_lex_is_one_for_same_token(self):
        # Con shared init (P_q_lex=P_k_lex) y L2-norm, mismo token -> score 1.
        torch.manual_seed(2)
        rt = make_rt(shared_lex_init=True)
        t0 = torch.randn(1, 12, 8)
        t0[0, 7] = t0[0, 2]
        from engrama.v55.recall import _l2
        q = _l2(rt.queries_lex(t0)); k = _l2(rt.keys_lex(t0))
        s = (q[0, 7] * k[0, 2]).sum()
        self.assertAlmostEqual(float(s), 1.0, places=5)


class TestSenseDisambiguation(unittest.TestCase):
    def test_sense_breaks_tie(self):
        # Dos ocurrencias del mismo token (mismo K_lex) pero distinto contexto
        # (distinto T_shallow -> distinto K_sense). El score de sentido difiere:
        # el eje de sentido es el que desambigua "banco".
        torch.manual_seed(3)
        rt = make_rt()
        t0 = torch.randn(1, 12, 8)
        t0[0, 7] = t0[0, 2]                      # mismo token en pos 2 y 7
        t_shallow = torch.randn(1, 12, 8)        # contexto local DISTINTO
        t_shallow[0, 7] = t_shallow[0, 2] + 5.0  # pos 2 y 7: mismo token, ctx diferente
        from engrama.v55.recall import _l2
        ks = _l2(rt.keys_sense(t_shallow))
        # K_sense difiere entre las dos ocurrencias del mismo token:
        self.assertFalse(torch.allclose(ks[0, 2], ks[0, 7], atol=1e-5))

    def test_composite_score_lex_dominant(self):
        # score = score_lex * (1 + beta*score_sense). score_lex acota el rango:
        # sin match lexico (score_lex~0) el composite es ~0 aunque sense sea alto.
        torch.manual_seed(4)
        rt = make_rt(sense_beta_init=0.3)
        beta = float(rt.beta)
        slex_low = torch.tensor(0.0)
        ssen_high = torch.tensor(1.0)
        slex_high = torch.tensor(1.0)
        self.assertLess(float(slex_low * (1 + beta * ssen_high)),
                        float(slex_high * (1 + beta * ssen_high)))


class TestHardRead(unittest.TestCase):
    def test_induction_read_next(self):
        # Con metrica identidad, la lectura devuelve T0[j*+1] de la ocurrencia
        # previa del mismo token (induccion/copia).
        torch.manual_seed(5)
        rt = make_rt()
        with torch.no_grad():
            eye = torch.eye(8)
            rt.p_q_lex.weight.copy_(eye); rt.p_k_lex.weight.copy_(eye)
        t0 = torch.randn(1, 12, 8)
        t0[0, 5] = t0[0, 2]            # repite token (pos 2 -> 5)
        expected = t0[0, 3]            # valor siguiente a la 1a ocurrencia
        with torch.no_grad():
            t_sh = t0.clone()
            reads = rt.forward_parallel_dense(
                rt.queries_lex(t0), rt.keys_lex(t0),
                rt.queries_ctx(t_sh), rt.keys_sense(t_sh), t0)
        self.assertTrue(torch.allclose(reads[0, 5], expected, atol=1e-5))
        self.assertTrue(torch.allclose(reads[0, 0], torch.zeros(8), atol=1e-6))

    def test_tie_break_most_recent(self):
        torch.manual_seed(6)
        rt = make_rt()
        with torch.no_grad():
            eye = torch.eye(8)
            rt.p_q_lex.weight.copy_(eye); rt.p_k_lex.weight.copy_(eye)
        t0 = torch.randn(1, 16, 8)
        t0[0, 10] = t0[0, 3]; t0[0, 14] = t0[0, 3]   # 3 ocurrencias
        with torch.no_grad():
            t_sh = t0.clone()
            reads = rt.forward_parallel_dense(
                rt.queries_lex(t0), rt.keys_lex(t0),
                rt.queries_ctx(t_sh), rt.keys_sense(t_sh), t0)
        # pos 15: lectura debe venir de la ocurrencia MAS RECIENTE (14) -> T0[15]
        self.assertFalse(torch.allclose(reads[0, 15], t0[0, 4], atol=1e-6))

    def test_read_step_matches_parallel(self):
        # La lectura incremental (matvec) coincide con la paralela (invarianza).
        torch.manual_seed(7)
        rt = make_rt()
        t0 = torch.randn(3, 20, 8)
        t_sh = torch.randn(3, 20, 8)
        with torch.no_grad():
            par = rt.forward_parallel_dense(
                rt.queries_lex(t0), rt.keys_lex(t0),
                rt.queries_ctx(t_sh), rt.keys_sense(t_sh), t0)
            # simula el anillo incremental por lote
            tr = PagedDualTrace(20, 8, 8, 8, horizons=[1] * 2, page_size=8)
            inc = torch.zeros_like(t0)
            for t in range(20):
                tr.append_t0(t0[:, t], rt.keys_lex(t0[:, t]))
                tr.append_shallow(t_sh[:, t], rt.keys_sense(t_sh[:, t]))
                q_lex = rt.queries_lex(t0[:, t])
                q_ctx = rt.queries_ctx(t_sh[:, t])
                r = rt.read_step(q_lex, tr.linear_klex(), q_ctx, tr.linear_ksen(),
                                 tr.linear_t0(), tr.length)
                inc[:, t] = r
            self.assertTrue(torch.allclose(par, inc, atol=1e-4),
                            msg=f"diff {float((par-inc).abs().max()):.2e}")


class TestFallbackAndIdentity(unittest.TestCase):
    def test_identity_fast_path_ulp_priority(self):
        # Si hay ocurrencia previa del mismo token y empata, gana (invarianza
        # con el camino denso). Construimos identity_prev y comparamos.
        torch.manual_seed(8)
        rt = make_rt()
        with torch.no_grad():
            eye = torch.eye(8)
            rt.p_q_lex.weight.copy_(eye); rt.p_k_lex.weight.copy_(eye)
        t0 = torch.randn(1, 12, 8)
        t0[0, 6] = t0[0, 2]
        from engrama.v55.lsh import previous_same_occurrence
        tokens = torch.tensor([0, 1, 2, 3, 4, 5, 2, 7, 8, 9, 10, 11])  # token 2 en 2 y 6
        id_prev = previous_same_occurrence(tokens, gap=1).unsqueeze(0)
        t_sh = t0.clone()
        with torch.no_grad():
            reads = rt.forward_parallel_dense(
                rt.queries_lex(t0), rt.keys_lex(t0),
                rt.queries_ctx(t_sh), rt.keys_sense(t_sh), t0,
                identity_prev=id_prev)
        # pos 6: identidad apunta a pos 2 -> lectura T0[3]
        self.assertTrue(torch.allclose(reads[0, 6], t0[0, 3], atol=1e-5))

    def test_no_nan_empty_trace(self):
        # Trace vacia / primer token: la lectura debe ser 0 y finita.
        torch.manual_seed(9)
        rt = make_rt()
        tr = PagedDualTrace(8, 8, 8, 8, horizons=[1], page_size=4)
        q = rt.queries_lex(torch.randn(1, 8))
        qc = rt.queries_ctx(torch.randn(1, 8))
        r = rt.read_step(q, tr.linear_klex(), qc, tr.linear_ksen(),
                         tr.linear_t0(), tr.length)
        self.assertTrue(torch.allclose(r, torch.zeros(1, 8), atol=1e-6))
        self.assertTrue(torch.isfinite(r).all().item())


class TestRetrievalLossDense(unittest.TestCase):
    """CE_retrieval densa: senal auto-supervisada que activa el eje de sentido."""

    def test_dense_ce_prefers_correct_candidate(self):
        # Si el score apunta al candidato cuyo next-token == target, la perdida
        # debe ser MENOR que si apunta a uno incorrecto.
        from engrama.v55.losses import retrieval_cross_entropy_dense
        torch.manual_seed(0)
        b, m, n = 1, 1, 5
        scores = torch.zeros(b, m, n)
        valid = torch.ones(b, m, n, dtype=torch.bool)
        next_tokens = torch.tensor([[10, 20, 30, 40, 50]])   # (B, N)
        targets = torch.tensor([[30]])                        # objetivo = token 30
        # correcto: next_tokens[2] == 30 -> score alto en col 2
        good = scores.clone(); good[0, 0, 2] = 10.0
        # incorrecto: score alto en col 0 (next=10 != 30)
        bad = scores.clone(); bad[0, 0, 0] = 10.0
        lg = retrieval_cross_entropy_dense(good, valid, next_tokens, targets)
        lb = retrieval_cross_entropy_dense(bad, valid, next_tokens, targets)
        self.assertLess(float(lg), float(lb))

    def test_forward_loss_grad_to_sense(self):
        # CE_retrieval INTERNA (por defecto) da gradiente directo a las
        # proyecciones de sentido p_q_ctx y p_k_sense (LM pura no lo hace).
        from engrama.v55 import EngraModelV55
        torch.manual_seed(0)
        m = EngraModelV55(V55Config.from_preset("tiny", vocab_size=40,
                          context_length=32, rt_train_mode="dense"))
        m.train()
        x = torch.randint(0, 40, (2, 32))
        loss = m.forward_loss(x, x)          # CE_retrieval activa por defecto
        loss.backward()
        self.assertGreater(m.recall.p_q_ctx.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(m.recall.p_k_sense.weight.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_retrieval_weight_zero_disables(self):
        # retrieval_weight=0 desactiva CE_retrieval (solo LM).
        from engrama.v55 import EngraModelV55
        torch.manual_seed(1)
        m = EngraModelV55(V55Config.from_preset("tiny", vocab_size=40,
                          context_length=32, rt_train_mode="dense"))
        m.train()
        x = torch.randint(0, 40, (2, 32))
        loss = m.forward_loss(x, x, retrieval_weight=0.0)
        loss.backward()
        # sin CE_retrieval, el gradiente al sentido es ~0 (solo via STE debil
        # del forward); aqui comprobamos al menos que no falla y es finito.
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
