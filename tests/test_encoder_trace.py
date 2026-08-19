import unittest
import torch
from engrama.encoder import IsolatedEncoder
from engrama.trace import EngramaCache
from engrama.config import EngramaConfig


class TestIsolatedEncoder(unittest.TestCase):
    def test_forward_shape(self):
        config = EngramaConfig(
            d_model=32,
            d_gate=8,
            num_cells=4,
            num_encoder_layers=2,
            d_ff=64,
        )
        encoder = IsolatedEncoder(config)
        x = torch.randn(2, 8, 32)
        out = encoder(x)
        self.assertEqual(out.shape, (2, 8, 32))


class TestEngramaCache(unittest.TestCase):
    def test_cache_fifo(self):
        N_max = 3
        num_layers = 2
        d_model = 16
        cache = EngramaCache(N_max=N_max, num_layers=num_layers, d_model=d_model)

        for i in range(5):
            t0 = torch.tensor([float(i)])
            tl = [torch.tensor([float(i)]), torch.tensor([float(i * 2)])]
            cache.append(t0, tl, i)

        self.assertEqual(len(cache), 3)
        self.assertEqual(cache.timestamps, [2, 3, 4])
        self.assertEqual(cache.T0[0].item(), 2.0)
        self.assertEqual(cache.T0[2].item(), 4.0)


if __name__ == "__main__":
    unittest.main()