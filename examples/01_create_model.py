import torch
from engrama.config import EngramaConfig
from engrama.model import EngramaModel

config = EngramaConfig(vocab_size=100, d_model=64, d_gate=16, num_cells=4, version="v2")
model = EngramaModel(config)
print("Model initialized:", model)

x = torch.randint(0, 100, (1, 10))
logits = model(x)
print("Logits shape:", logits.shape)
assert logits.shape == (1, 10, 100)
print("Example 01 successful!")
