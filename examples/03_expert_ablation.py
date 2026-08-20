"""Ejemplo 03 — Modo experto: configuración completa y ablación V2/V3.

Muestra cómo EngramaConfig expone cada modo de arquitectura, cómo el
preset `version` se combina con overrides explícitos, y verifica la
invarianza causal (forward paralelo == inferencia incremental).

Ejecuta:  python examples/03_expert_ablation.py
"""

import torch

from engrama import EngramaConfig, EngramaModel

BASE = dict(vocab_size=128, d_model=256, num_cells=8,
            context_length=256, num_candidates=4)

# --- Presets de versión (V3 factorizado vs V2 denso) ----------------------
cfg_v3 = EngramaConfig(version="v3", **BASE)
cfg_v2 = EngramaConfig(version="v2", **BASE)
m_v3, m_v2 = EngramaModel(cfg_v3), EngramaModel(cfg_v2)
print(f"Parámetros V3: {m_v3.num_parameters():,}")
print(f"Parámetros V2: {m_v2.num_parameters():,}")

# --- Conectividad diádica y horizontes de caché ---------------------------
print("\nOffsets por capa (V3):",
      [cfg_v3.get_layer_offsets(l) for l in range(cfg_v3.num_consolidation_layers)])
print("Horizontes de caché jerárquico:", cfg_v3.cache_horizons())
print("Campo receptivo:", cfg_v3.receptive_field()["max_reach"])

# --- Ablación: V3 con offsets densos y caché completo (mezcla de modos) ---
cfg_abl = EngramaConfig(version="v3", offset_mode="dense_dilated",
                        cache_mode="full", **BASE)
m_abl = EngramaModel(cfg_abl)
print(f"\nAblación (V3 + offsets densos): {m_abl.num_parameters():,} parámetros")

# --- Invarianza causal -----------------------------------------------------
model = m_v3.eval()
x = torch.randint(0, 128, (2, 32))
with torch.no_grad():
    full = model(x)
    cache = model.get_cache(N_max=32)
    steps = [model.step_forward(x[:, t:t + 1], cache, t)[0] for t in range(32)]
    inc = torch.stack(steps, dim=1)
diff = (full - inc).abs().max().item()
print(f"Invarianza causal (max |diff|): {diff:.2e}")
assert diff < 1e-4
print("Ejemplo 03 OK")
