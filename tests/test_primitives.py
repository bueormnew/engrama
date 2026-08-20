"""Primitive block tests: cells, shared cores, synapse routing (V3 spec 5-7, 31)."""

import unittest

import torch

from engrama.primitives import Cell, EngramaLayerNorm, SharedCoreCellGroup, SynapseLayer


def _layer(mode="factorized", cells=3, **kw):
    cfg = dict(
        d_model=32, d_gate=8, num_cells=cells, d_ff=64,
        synapse_mode=mode, **kw,
    )
    return SynapseLayer(**cfg)


class TestCell(unittest.TestCase):
    def test_residual_shape(self):
        cell = Cell(32, 64)
        x = torch.randn(2, 5, 32)
        out = cell(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.allclose(out, x))  # block actually transforms

    def test_layernorm_standardizes(self):
        ln = EngramaLayerNorm(16)
        x = torch.randn(4, 16) * 17 + 5
        y = ln(x)
        self.assertLess(abs(y.mean().item()), 1e-5)
        self.assertLess(abs(y.std(unbiased=False).item() - 1.0), 1e-4)


class TestSharedCoreCellGroup(unittest.TestCase):
    def test_shapes_and_identity_params(self):
        group = SharedCoreCellGroup(4, 32, 64)
        u = torch.randn(2, 7, 4, 32)
        self.assertEqual(group(u).shape, u.shape)

    def test_zero_scale_is_residual_identity(self):
        group = SharedCoreCellGroup(2, 16, 32)
        with torch.no_grad():
            group.s_scale.zero_()  # s_b = 0 -> Cell(x) = x
        u = torch.randn(3, 2, 16)
        self.assertLess((group(u) - u).abs().max().item(), 1e-6)


class TestSynapseLayerShapes(unittest.TestCase):
    def test_factorized_shapes_3d_4d(self):
        layer = _layer("factorized")
        self.assertEqual(layer(torch.randn(2, 5, 3, 32)).shape, (2, 5, 3, 32))
        self.assertEqual(layer(torch.randn(2, 3, 32)).shape, (2, 3, 32))

    def test_dense_shapes(self):
        layer = _layer("dense", cells=2, cell_mode="independent")
        self.assertEqual(layer(torch.randn(2, 4, 2, 32)).shape, (2, 4, 2, 32))


class TestGateFromSource(unittest.TestCase):
    """Spec section 7: alpha_{a->b} depends on the SOURCE cell a only."""

    def test_perturbing_source_changes_only_its_outgoing_gates(self):
        layer = _layer(cells=3)
        h = torch.randn(1, 2, 3, 32)
        alpha = layer.gates(h)  # (B, N, a, b)

        h2 = h.clone()
        h2[:, :, 1, :] += 0.5  # perturb source cell 1
        alpha2 = layer.gates(h2)

        self.assertGreater((alpha2[:, :, 1, :] - alpha[:, :, 1, :]).abs().max().item(), 1e-4)
        self.assertLess((alpha2[:, :, 0, :] - alpha[:, :, 0, :]).abs().max().item(), 1e-6)
        self.assertLess((alpha2[:, :, 2, :] - alpha[:, :, 2, :]).abs().max().item(), 1e-6)


class TestIdentityTransportRoute(unittest.TestCase):
    """Spec section 31: with beta=1, alpha~1, s~0 a synapse transports identity."""

    def test_single_synapse_is_identity(self):
        layer = _layer(cells=1)
        with torch.no_grad():
            layer.gate_b.fill_(20.0)  # alpha ~ 1
            layer.cell_group.s_scale.zero_()  # Cell ~ residual identity
        h = torch.randn(2, 5, 1, 32)
        self.assertLess((layer(h) - h).abs().max().item(), 1e-5)

    def test_stable_init_gives_zero_s_and_unit_beta(self):
        layer = _layer(stable_init=True)
        self.assertLess(layer.s_scale.detach().abs().max().item(), 1e-8)
        self.assertLess((layer.beta.detach() - 1).abs().max().item(), 1e-8)

    def test_unstable_init_differs(self):
        layer = _layer(stable_init=False)
        self.assertGreater(layer.s_scale.detach().abs().max().item(), 1e-4)


class TestSynapseModesEquivalence(unittest.TestCase):
    """Dense and factorized synapses must both be position-independent ops."""

    def test_token_isolation_both_modes(self):
        for mode in ("factorized", "dense",):
            layer = _layer(mode, cells=2).eval()
            h = torch.randn(1, 4, 2, 32)
            out = layer(h)
            h2 = h.clone()
            h2[:, 3] = torch.randn(2, 32)
            out2 = layer(h2)
            # position 3 changed -> earlier positions must remain identical
            self.assertLess((out2[:, :3] - out[:, :3]).abs().max().item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
