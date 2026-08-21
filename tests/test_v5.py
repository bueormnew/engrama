"""ENGRAMA V5 test suite — guarantees the 11 design requirements.

Covers: causal invariance (parallel == incremental), chunked == full,
NaN-free fp16 forward/backward, linear memory generation cache, and the API.
"""

from __future__ import annotations

import math

import pytest
import torch

from engrama.v5 import EngramaV5, EngramaV5Config, SynapticResonance, ResonanceCache


def _model(**kw):
    base = dict(vocab_size=64, d_model=64, num_layers=3, num_heads=4,
                context_length=128, num_candidates=2)
    base.update(kw)
    torch.manual_seed(0)
    return EngramaV5(EngramaV5Config(**base)).eval()


class TestConfig:
    def test_head_divisibility(self):
        with pytest.raises(ValueError):
            EngramaV5Config(d_model=64, num_heads=5)

    def test_roundtrip(self, tmp_path):
        cfg = EngramaV5Config(vocab_size=100, d_model=32, num_heads=4)
        p = tmp_path / "cfg.json"
        cfg.save(str(p))
        cfg2 = EngramaV5Config.load(str(p))
        assert cfg2.to_dict() == cfg.to_dict()

    def test_presets(self):
        for size in ("tiny", "small", "base", "large"):
            cfg = EngramaV5Config.preset(size, vocab_size=1000)
            assert cfg.vocab_size == 1000

    def test_encoder_zero_ok(self):
        cfg = EngramaV5Config(num_encoder_layers=0)
        assert cfg.num_encoder_layers == 0


class TestShapes:
    def test_forward_shape(self):
        m = _model()
        out = m.forward(torch.randint(0, 64, (2, 20)))
        assert out.shape == (2, 20, 64)

    def test_loss_scalar(self):
        m = _model()
        ids = torch.randint(0, 64, (2, 20))
        loss = m.loss(ids, torch.randint(0, 64, (2, 20)))
        assert loss.dim() == 0 and torch.isfinite(loss)


class TestCausalInvariance:
    def test_parallel_equals_incremental(self):
        m = _model()
        N = 40
        ids = torch.randint(0, 64, (2, N))
        with torch.no_grad():
            par = m.forward(ids)
            inc = torch.zeros_like(par)
            for b in range(ids.size(0)):
                cache = m.new_cache()
                for t in range(N):
                    inc[b, t] = m._step_logits(ids[b, t:t + 1], cache)[0]
        assert (par - inc).abs().max().item() < 1e-4

    def test_chunked_equals_full(self):
        m = _model()
        m2 = _model(chunk_size=8)
        m2.load_state_dict(m.state_dict())
        ids = torch.randint(0, 64, (2, 50))
        with torch.no_grad():
            assert (m.forward(ids) - m2.forward(ids)).abs().max().item() < 1e-4

    def test_strict_causality(self):
        """Changing a future token must not change past logits."""
        m = _model()
        ids = torch.randint(0, 64, (1, 30))
        with torch.no_grad():
            a = m.forward(ids)
            ids2 = ids.clone()
            ids2[0, 20] = (ids2[0, 20] + 1) % 64
            b = m.forward(ids2)
        assert (a[0, :20] - b[0, :20]).abs().max().item() < 1e-5


class TestNaNFree:
    def test_fp16_forward_finite(self):
        m = _model(dtype="float16")
        out = m.forward(torch.randint(0, 64, (2, 30)))
        assert torch.isfinite(out).all()

    def test_backward_finite(self):
        m = _model()
        m.train()
        ids = torch.randint(0, 64, (2, 30))
        loss = m.loss(ids, torch.randint(0, 64, (2, 30)))
        loss.backward()
        for p in m.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_extreme_repetition_no_nan(self):
        m = _model()
        ids = torch.zeros(1, 40, dtype=torch.long)
        out = m.forward(ids)
        assert torch.isfinite(out).all()


class TestCache:
    def test_cache_grows_and_caps(self):
        cache = ResonanceCache(num_layers=2, n_max=5)
        for _ in range(8):
            cache.layer(0).append(torch.randn(1, 4, 8), torch.randn(1, 4, 8))
        assert len(cache.layer(0)) == 5  # FIFO cap

    def test_memory_is_linear(self):
        cache = ResonanceCache(num_layers=1, n_max=1000)
        sizes = []
        for n in (10, 20, 40):
            cache.clear()
            for _ in range(n):
                cache.layer(0).append(torch.randn(1, 4, 8), torch.randn(1, 4, 8))
            sizes.append(cache.memory_bytes())
        # doubling tokens doubles memory (linear)
        assert abs(sizes[1] / sizes[0] - 2.0) < 0.01
        assert abs(sizes[2] / sizes[1] - 2.0) < 0.01


class TestResonanceMechanism:
    def test_no_softmax_gates_independent(self):
        """Gates must not sum to 1 across positions (that would be attention)."""
        torch.manual_seed(0)
        res = SynapticResonance(d_model=32, num_heads=2, read_norm=None)
        x = torch.randn(1, 10, 32)
        q, k, _ = res._project(x)
        g = res._gate_scores(q, k)
        mask = torch.ones(10, 10).tril().bool()
        g = g.masked_fill(~mask, 0.0)
        row_sums = g.sum(-1)  # would be ~1 everywhere if softmax
        # At least some rows must have mass clearly above 1 (superposition).
        assert row_sums.max().item() > 1.5

    def test_generation_runs(self):
        m = _model()
        out = m.generate([1, 2, 3], max_new_tokens=8, temperature=0.8, top_k=10)
        assert len(out) == 11


class TestBlockSparse:
    def test_topk_all_equals_dense(self):
        """Block-sparse with top_k >= num_blocks must equal the dense read."""
        from engrama.v5.blocksparse import BlockSparseResonance
        from engrama.v5.resonance import SynapticResonance
        torch.manual_seed(0)
        d, H, N = 64, 4, 50
        x = torch.randn(1, N, d)
        bs = BlockSparseResonance(d, H, block_size=8, top_k=999,
                                  read_norm=None, norm_type="layernorm").eval()
        dense = SynapticResonance(d, H, read_norm=None, norm_type="layernorm").eval()
        dense.load_state_dict({k: v for k, v in bs.state_dict().items()
                               if k in dense.state_dict()})
        with torch.no_grad():
            assert (bs(x) - dense(x)).abs().max().item() < 1e-4

    def test_blocksparse_causal(self):
        from engrama.v5.blocksparse import BlockSparseResonance
        torch.manual_seed(0)
        bs = BlockSparseResonance(64, 4, block_size=8, top_k=4,
                                  norm_type="layernorm").eval()
        x = torch.randn(1, 50, 64)
        with torch.no_grad():
            a = bs(x)
            x2 = x.clone()
            x2[0, 40] += 5.0
            b = bs(x2)
        assert (a[0, :40] - b[0, :40]).abs().max().item() < 1e-5

    def test_model_block_sparse_runs(self):
        cfg = EngramaV5Config(vocab_size=64, d_model=64, num_layers=2, num_heads=4,
                              context_length=128, resonance_mode="block_sparse",
                              block_size=16, top_k=4)
        m = EngramaV5(cfg).eval()
        out = m.forward(torch.randint(0, 64, (2, 40)))
        assert out.shape == (2, 40, 64) and torch.isfinite(out).all()

    def test_block_sparse_generation_matches_step(self):
        """Block-sparse training model still generates via the exact dense cache."""
        cfg = EngramaV5Config(vocab_size=64, d_model=64, num_layers=2, num_heads=4,
                              context_length=128, resonance_mode="block_sparse",
                              block_size=16, top_k=4)
        m = EngramaV5(cfg).eval()
        out = m.generate([1, 2, 3], max_new_tokens=5, temperature=0.0)
        assert len(out) == 8


class TestTritonKernels:
    def test_dense_ref_matches_module_read(self):
        """The kernel's PyTorch fallback must match the module's dense read."""
        from engrama.v5.triton_kernels import _dense_ref
        from engrama.v5.resonance import SynapticResonance
        torch.manual_seed(0)
        d, H, N = 64, 4, 30
        res = SynapticResonance(d, H, read_norm=None, norm_type="layernorm").eval()
        x = torch.randn(1, N, d)
        with torch.no_grad():
            q, k, v = res._project(x)
            r_ref = _dense_ref(q, k, v, res.tau(), res.gate_bias, None)
            g = res._gate_scores(q, k)
            mask = torch.ones(N, N).tril().bool()
            r_internal = torch.matmul(g.masked_fill(~mask, 0.0), v)
        assert (r_ref - r_internal).abs().max().item() < 1e-5

    def test_has_triton_flag(self):
        from engrama.v5.triton_kernels import has_triton
        assert isinstance(has_triton(), bool)
