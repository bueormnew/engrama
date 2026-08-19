# ENGRAMA 🧠⚡

**ENGRAMA** es una arquitectura neuronal autorregresiva avanzada y librería de aprendizaje profundo implementada en **PyTorch puro**, diseñada sin mecanismos de atención dinámica ($QK^T$), matrices de afinidad $N \times N$ ni softmax sobre la dimensión de secuencia.

- **Autor**: BUEORM
- **Licencia**: GNU Affero General Public License v3.0 (AGPL-3.0)
- **Versión**: 0.1.0

---

## 🌟 Características Principales

1. **Cero Atención Dinámica ($QK^T$)**: Sustituye la complejidad cuadrática $O(N^2)$ de los Transformers por sinapsis de enrutamiento celular estático y mezclas de posición causales dilatadas.
2. **Fase 1 - Codificación Aislada (`IsolatedEncoder`)**: Procesamiento token por token con 0% de fuga de contexto en la etapa inicial.
3. **Fase 2 - Traza Circular Incremental (`EngramaCache`)**: Buffer circular FIFO de capacidad $N_{max}$ con timestamps exactos para inferencia incremental.
4. **Fase 3 - Consolidación Causal Dilatada (`ConsolidationStack`)**: Mezcla relativa en offsets en potencias de 2 ($P_l(i) = [0, 1, 2, 4, 8, \dots]$) mediante compuertas latentes $d_g \ll d_{model}$.
5. **Fase 4 - Evocador Multi-Candidato (`MultiCandidateEvoker`)**: Generación de $M \in [1, 8]$ candidatos proyectados y aglomeración de similitud mediante LogSumExp, Max o Mean.
6. **Invarianza Causal Obligatoria**: Garantía matemática de que el paso paralelo `forward(seq)` equivale exactamente a la generación incremental `step_forward(seq)` con error $< 1e-4$.
7. **Generación con Streaming y Muestreo Avanzado**: Soporte para muestreo Top-K, Top-P (Nucleus Sampling), temperatura, penalización de repetición y streaming de caracteres/tokens en tiempo real.
8. **CLI Completa e Inspector Interno**: Herramientas para entrenar, evaluar, inspeccionar estados ocultos/compuertas y ejecutar suites de benchmark de latencia y memoria.

---

## 📦 Instalación

### Desde PyPI (Producción)
```bash
pip install engrama
```

### Desde el Código Fuente
```bash
git clone https://github.com/BUEORM/ENGRAMA.git
cd ENGRAMA
pip install -e .
```

---

## 🚀 Guía Rápida de Uso

### 1. Inicialización del Modelo y Tokenizador

```python
import torch
from engrama import EngramaConfig, EngramaModel, EngramaTokenizer

# 1. Crear e inicializar el tokenizador de caracteres
text_data = "Hola mundo, esta es la arquitectura neuronal ENGRAMA."
tokenizer = EngramaTokenizer().fit_on_text(text_data)

# 2. Configurar los hiperparámetros
config = EngramaConfig(
    vocab_size=tokenizer.vocab_size,
    d_model=512,
    d_gate=64,
    d_ff=2048,
    num_cells=16,
    num_encoder_layers=2,
    num_consolidation_layers=6,
    context_length=2048,
    num_candidates=4,
    candidate_aggregation="logsumexp",
)

# 3. Construir el modelo PyTorch
model = EngramaModel(config)
print(f"Parámetros totales: {model.num_parameters():,}")
```

---

### 2. Entrenamiento con `Trainer` y `TextDataset`

```python
from engrama import TextDataset, Trainer

# Crear dataset autorregresivo a partir de un archivo o string
dataset = TextDataset(text_data, tokenizer, sequence_length=64)

# Inicializar entrenador con optimizador AdamW y gradient clipping
trainer = Trainer(model, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu")

# Entrenar por 5 épocas
history = trainer.fit(dataset, batch_size=4, epochs=5)
print("Pérdidas por época:", history)
```

---

### 3. Generación de Texto y Streaming

```python
from engrama import Generator

generator = Generator(model, tokenizer)

# Generación estándar
prompt = "Hola"
completion = generator.generate(
    prompt=prompt,
    max_new_tokens=30,
    temperature=0.8,
    top_k=10,
    top_p=0.9,
    use_cache=True,
)
print("Resultado:", completion)

# Streaming token a token en tiempo real
print(prompt, end="", flush=True)
for token_char in generator.generate_stream(prompt, max_new_tokens=20, temperature=0.7):
    print(token_char, end="", flush=True)
print()
```

---

### 4. Guardado y Carga de Checkpoints

```python
from engrama import save_model, load_model

# Guardar modelo, configuración y tokenizador
save_model(model, save_dir="./checkpoint_engrama", tokenizer=tokenizer)

# Cargar checkpoint
model_loaded, tokenizer_loaded = load_model("./checkpoint_engrama", device="cpu")
```

---

### 5. Verificación de Invarianza Causal y Benchmarks

```python
from engrama import BenchmarkSuite

# 1. Verificar invarianza causal (forward paralelo vs step_forward incremental)
causal_res = BenchmarkSuite.verify_causal_invariance(model, seq_length=20)
print("Prueba de invarianza causal superada:", causal_res["passed"])
print("Diferencia máxima observada:", causal_res["max_diff"])

# 2. Benchmarks de latencia y memoria
lat = BenchmarkSuite.benchmark_latency(model, seq_length=128, num_runs=5)
mem = BenchmarkSuite.benchmark_memory(model, seq_length=128)
print("Latencia paralelo (tok/s):", lat["parallel_tokens_per_sec"])
print("Latencia por token incremental (tok/s):", lat["step_tokens_per_sec"])
print("Huella de memoria (bytes):", mem["peak_memory_bytes"])
```

---

## 🛠️ Interfaz de Línea de Comandos (CLI)

ENGRAMA incluye los comandos `engrama` y `engramacli`:

```bash
# Información del entorno y la librería
engrama info

# Entrenar un modelo sobre un archivo de texto
engrama train --text-file corpus.txt --output-dir ./mi_modelo --epochs 10 --batch-size 16

# Generar texto interactivo con streaming
engrama generate --model-dir ./mi_modelo --prompt "El futuro de la IA" --max-tokens 50 --stream

# Evaluar pérdida en un conjunto de validación
engrama evaluate --model-dir ./mi_modelo --text-file validacion.txt

# Ejecutar benchmarks de rendimiento e invarianza causal
engrama benchmark --seq-len 256 --runs 10

# Inspeccionar arquitectura y activaciones del modelo
engrama inspect --model-dir ./mi_modelo --sample-text "Prueba de inspección"
```

---

## 🧪 Pruebas Unitarias

Para ejecutar la suite completa de pruebas unitarias:

```bash
python -m unittest discover -s tests
```

---

## 📜 Licencia

Este proyecto está bajo la Licencia **GNU Affero General Public License v3.0 (AGPL-3.0)**.
Derechos de autor (C) 2026 **BUEORM**.
