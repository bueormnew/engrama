"""ENGRAMA V5 — quickstart (4 lineas al modelo).

    pip install -e . && python examples/04_v5_quickstart.py
"""
from __future__ import annotations

import torch

from engrama import EngraModelV5, V5Config


def main() -> None:
    # 1) Modelo desde preset (tiny|small|base|large) o config manual.
    model = EngraModelV5.from_preset("tiny", vocab_size=256, context_length=512)
    print(model.describe())

    # 2) Entrenamiento paralelo (todo el contexto de una vez, sin atencion).
    x = torch.randint(0, 256, (8, 128))
    y = torch.randint(0, 256, (8, 128))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(5):
        loss = model.forward_loss(x[:, :-1], y[:, 1:])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        print(f"paso {step} loss {float(loss):.4f}")

    # 3) Generacion incremental con cache nativa (Traza + anillo K):
    #    consolidacion O(1)/token, lectura dura O(N d_k)/token — sin recalcular.
    model.eval()
    ids = model.generate([1, 2, 3], max_new_tokens=16, temperature=0.8, top_k=20)
    print("generado:", ids)

    # 4) Guardar/cargar.
    model.save("/tmp/engrama_v5_demo")
    model2 = EngraModelV5.load("/tmp/engrama_v5_demo")
    with torch.no_grad():
        ok = torch.allclose(model.eval()(torch.tensor([[1, 2, 3]])),
                            model2.eval()(torch.tensor([[1, 2, 3]])))
    print("save/load exacto:", ok)


if __name__ == "__main__":
    main()
