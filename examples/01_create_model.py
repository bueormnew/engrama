"""Ejemplo 01 — Crear un modelo ENGRAMA con el modo rápido.

Ejecuta:  python examples/01_create_model.py
"""

import torch

import engrama

# Presets de tamaño disponibles
for size, info in engrama.list_sizes().items():
    print(f"  {size:5s}: {info}")

# Crear un modelo listo para entrenar con una sola línea
model = engrama.create_model(size="tiny", vocab_size=100)
print(f"\nModelo 'tiny': {model.num_parameters():,} parámetros")

x = torch.randint(0, 100, (1, 10))
logits = model(x)
print("Logits shape:", tuple(logits.shape))
assert logits.shape == (1, 10, 100)
print("Ejemplo 01 OK")
