# Guía Completa de Uso — ENGRAMA V3

Esta guía cubre los dos modos de uso de la librería, la semántica de cada
parámetro de configuración, y los flujos completos de entrenamiento,
inferencia, serialización e inspección. Todo el código mostrado está
verificado contra la versión 0.3.0 de la librería.

## 1. Filosofía y principios fundamentales

ENGRAMA es una arquitectura autorregresiva **sin mecanismos de atención**:
no existe $QK^T$, ni matrices de afinidad $N \times N$, ni softmax sobre la
dimensión temporal. La versión 0.3.0 implementa la especificación
**ENGRAMA V3** (`ENGRAMA-V3-Teorica.md`):

- **Fase 1 — Codificación aislada**: cada token $x_i$ produce su huella
  $T_0[i]$ de forma independiente (células + sinapsis de enrutamiento
  $C \times C$ con compuertas latentes $d_g \ll d$). Paralelización total.
- **Fase 2 — Traza circular**: buffer FIFO que **solo almacena** huellas con
  timestamp exacto; nunca las transforma (`collections.deque(maxlen=N)`, O(1)).
- **Fase 3 — Consolidación diádica jerárquica**: mezcla causal relativa con
  offsets $D_l = \{0, 1, 2^l\}$ por capa. Las sinapsis factorizadas son
  **compartidas por capa** ($U_l, V_l$) con espectro por arista
  $s_{a \to b}$ y ruta identidad $\beta_{a \to b}$ (transporte exacto, §31).
  La compuerta de cada arista se calcula **desde la fuente**:
  $\alpha_{a \to b} = \sigma(w_{a \to b} \cdot (P_g h_a) + b_{a \to b})$.
- **Fase 4 — Evocación multi-candidato**: $M \in [1,8]$ candidatos
  factorizados sobre un núcleo $W_{shared}$ común, agregados con
  `logsumexp` (default), `max` o `mean` (esta última permite factorizar el
  matmul de vocabulario a $O(|V| d)$).
- **Invarianza causal obligatoria**: el paso paralelo `forward(seq)` y la
  generación incremental `step_forward` coinciden con error < 1e-4
  (medido: 5.96e-07 en float32). Testeado en 17 combinaciones de modos.

## 2. Instalación

```bash
pip install git+https://github.com/bueormnew/engrama.git
# o para desarrollo:
git clone https://github.com/bueormnew/engrama.git && cd engrama && pip install -e .
```

> El nombre `engrama` en PyPI pertenece a un paquete no relacionado;
> instala siempre desde GitHub.

## 3. Modo rápido

```python
import engrama

# 3.1 De texto a modelo entrenado en una llamada
run = engrama.quickstart(
    "corpus.txt",     # ruta o string con el texto
    size="small",     # tiny | small | base | large
    epochs=10,
    batch_size=16,
    lr=None,          # default por preset si se omite
    device=None,      # auto cuda/cpu
    seed=42,
)
print(run.summary())
run.generate("Hola", max_new_tokens=50, temperature=0.8, top_k=40)
run.evaluate("texto de validación")     # cross-entropy
run.save("./mi_modelo")                  # modelo + config + tokenizer

run2 = engrama.load_quick("./mi_modelo") # recarga el bundle completo

# 3.2 Solo crear el modelo
model = engrama.create_model(size="tiny", vocab_size=256,
                             num_candidates=4)   # overrides opcionales
print(engrama.list_sizes())                     # describe los presets
print(engrama.default_lr("small"))              # lr recomendado
```

`quickstart` acepta además cualquier override de `EngramaConfig` como
keyword (p. ej. `offset_mode="dense_dilated"`).

## 4. Modo experto

### 4.1 Configuración

```python
from engrama import EngramaConfig, EngramaModel

cfg = EngramaConfig(
    vocab_size=256,
    d_model=256, d_gate=32, d_ff=1024,
    num_cells=8,
    num_encoder_layers=2,
    num_consolidation_layers=8,
    context_length=256,
    num_candidates=4,
    candidate_aggregation="logsumexp",  # logsumexp | max | mean
    activation="gelu",                  # gelu | relu | silu
    dropout=0.0,
    dtype="float32",                    # float32/64/16, bfloat16
    version="v3",                       # preset arquitectónico
    tie_embeddings=True,
    synapse_rank=32,
    global_anchor=False,
    stable_init=True,
    # modos (None => heredado del preset `version`):
    synapse_mode=None, cell_mode=None, offset_mode=None,
    cache_mode=None, evoker_mode=None,
    identity_transport=None, hierarchical_gate=None,
)
model = EngramaModel(cfg)
```

### 4.2 Presets de versión (ablaciones integradas)

| Modo | v1 / v2 | v3 |
|---|---|---|
| `synapse_mode` | `dense` | `factorized` |
| `cell_mode` | `independent` | `shared_core` |
| `offset_mode` | `dense_dilated` | `hierarchical_dyadic` |
| `cache_mode` | `full` | `hierarchical` |
| `evoker_mode` | `dense` | `factorized` |
| `identity_transport` | `False` | `True` |
| `hierarchical_gate` | `False` | `True` |

Cualquier modo explícito sobrescribe al preset:

```python
# V3 con offsets densos (ablación D/E de la spec)
abl = EngramaConfig(version="v3", offset_mode="dense_dilated")

# V3 sin transporte identidad (áblation §43)
abl2 = EngramaConfig(version="v3", identity_transport=False)
```

### 4.3 Utilidades de conectividad

```python
cfg.get_layer_offsets(3)      # [0, 1, 8]  (D_l de la capa 3)
cfg.cache_horizons()          # [3, 5, 9, 17, 33, 65, 129, 1]
cfg.receptive_field()         # dict con max_reach, cobertura, etc.
```

La **regla de profundidad** (spec §26) exige $L \ge \lceil \log_2 N \rceil$
para cobertura binaria completa; violarla emite un `warnings.warn`
descriptivo (con `global_anchor=True` la cobertura la da el ancla).

### 4.4 Entrenamiento

```python
from engrama import EngramaTokenizer, TextDataset, Trainer
import torch

with open("corpus.txt", encoding="utf-8") as f:
    texto = f.read()

tokenizer = EngramaTokenizer().fit_on_text(texto)
dataset = TextDataset(texto, tokenizer, sequence_length=128)
trainer = Trainer(model, lr=1e-3,
                  scheduler="cosine",   # none | warmup | cosine
                  warmup_steps=200,
                  device="cuda" if torch.cuda.is_available() else "cpu")
history = trainer.fit(dataset, batch_size=16, epochs=10,
                      callbacks=[lambda e, l: print(e, l)])
val_loss = trainer.evaluate(dataset)
```

### 4.5 Inferencia incremental y generación

```python
from engrama import Generator

# forward paralelo (entrenamiento/evaluación)
input_ids = torch.tensor([tokenizer.encode("Hola mundo")])
logits = model(input_ids)                    # (B, N, vocab)

# incremental (invarianza causal garantizada)
cache = model.get_cache(N_max=256)
logits_t, hidden_t = model.step_forward(input_ids[:, 0:1], cache, 0)

gen = Generator(model, tokenizer)
salida = gen.generate("Hola", max_new_tokens=100,
                      temperature=0.8, top_k=40, top_p=0.95)
for chunk in gen.generate_stream("Hola", max_new_tokens=100):
    print(chunk, end="")
```

### 4.6 Serialización

```python
from engrama import save_model, load_model

save_model(model, "./ckpt", tokenizer=tokenizer)
model2, tokenizer2 = load_model("./ckpt")
```

### 4.7 Inspección

```python
from engrama import EngramaInspector

ids = torch.randint(0, cfg.vocab_size, (1, 16))
cache = model.get_cache(N_max=64)

EngramaInspector.inspect_model_summary(model)     # estructura + parámetros
EngramaInspector.inspect_synapses(model)          # fidelidad identidad β, |s|, ρ (§50)
EngramaInspector.inspect_activations(model, ids)  # stats de T0..T_L
EngramaInspector.inspect_gates(model, ids)        # actividad de compuertas por offset
EngramaInspector.inspect_cache(cache)             # estado del caché
```

## 5. CLI

```bash
engrama sizes | info
engrama train --text-file corpus.txt --size small --epochs 10 \
    [--lr 3e-3] [--scheduler cosine --warmup-steps 100] \
    [--d-model 256 --num-cells 8 --consolidation-layers 8 --context-len 256] \
    --output-dir checkpoints/run1
engrama generate --model-dir checkpoints/run1 --prompt "Hola" \
    [--max-tokens 100 --temperature 0.8 --top-k 40 --top-p 0.95 --stream]
engrama evaluate --model-dir checkpoints/run1 --text-file valid.txt
engrama inspect --model-dir checkpoints/run1 --sample-text "Hola"
engrama benchmark --size small --seq-len 256 --runs 10 [--no-cache]
```

## 6. Verificación y tests

```bash
cd tests && python -m unittest discover -q   # 68 tests
python benchmarks/kv_retrieval.py --steps 600 --seed 1234
```

Ver `docs/VERIFICACION.md` para la lista completa de garantías testeadas y
`benchmarks/KV_RETRIEVAL_REPORT.md` para el benchmark de largo alcance.
