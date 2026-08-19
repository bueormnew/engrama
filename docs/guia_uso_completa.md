# Documentación Completa de la Librería ENGRAMA

ENGRAMA es una arquitectura autoregresiva **sin mecanismos de atención**, diseñada sobre los principios de **Codificación de Huella Aislada**, **Traza Circular FIFO**, **Consolidación Dilatada Causal Relativa** y **Caché Logarítmico Incremental $O(L \cdot K \cdot d)$**.

## 1. Filosofía y Principios Fundamentales
- **Cero Atención:** Sin matrices $QK^T$, ni softmax sobre la dimensión temporal. La interacción depende únicamente de desplazamientos relativos estáticos (potencias de 2) con compuertas de gating.
- **Codificación Aislada (Fase 1):** Cada token $x_i$ produce su representación $T_0[i]$ de manera independiente, permitiendo una paralelización 100% libre de contexto pasado o futuro.
- **Traza Circular (Fase 2):** Buffer FIFO con corresponsabilidad temporal estable que reemplaza la atención dinámica guardando $N_{max}$ estados sin transformarlos.
- **Invarianza Causal (Fase 3):** Los estados pasados son inmutables al generar nuevos tokens. Garantiza equivalencia exacta entre el entrenamiento paralelo y la inferencia token-a-token incremental.

## 2. Creación y Configuración del Modelo
```python
from engrama.config import EngramaConfig
from engrama.model import EngramaModel

# Modelo Ultra-Pequeño (~1.5M - 3M parámetros)
config = EngramaConfig(
    vocab_size=256,
    d_model=128,
    d_gate=16,
    num_cells=4,
    num_encoder_layers=1,
    num_consolidation_layers=2,
    context_length=64,
    version="v2"
)

model = EngramaModel(config)
```

## 3. Preparación de Datos y Tokenización
```python
from engrama.tokenizer import EngramaTokenizer
from engrama.datasets import TextDataset

tokenizer = EngramaTokenizer()
tokenizer.fit_on_text("1 + 1 = 2\n2 + 2 = 4\n")

dataset = TextDataset(
    text="1 + 1 = 2\n2 + 2 = 4\n",
    tokenizer=tokenizer,
    sequence_length=32
)
```

## 4. Entrenamiento Autoregresivo (Teacher Forcing)
```python
from engrama.trainer import Trainer

trainer = Trainer(model, lr=1e-3, device="cpu")
loss_history = trainer.fit(dataset, batch_size=16, epochs=5)
```

## 5. Inferencia Incremental con Caché V2
```python
from engrama.inference import Generator

generator = Generator(model, tokenizer)
resultado = generator.generate(
    prompt="1 + 1 =",
    max_new_tokens=5,
    temperature=0.7,
    use_cache=True # Inferencia O(L * K * d) por token
)
print(resultado)
```

## 6. Guardado y Carga de Checkpoints
```python
from engrama.serialization import save_model, load_model

# Guardar
save_model(model, "checkpoint_math", tokenizer=tokenizer)

# Cargar
model_cargado, tokenizer_cargado = load_model("checkpoint_math")
```

## 7. Benchmark Real: Generación Matemática Sintética Autoregresiva
Se ejecutó un benchmark completo sobre un **modelo de ~1.37 Millones de Parámetros** entrenado durante 60 épocas en un dataset sintético de operaciones matemáticas (`A + B = C`).

### Configuración del Benchmark
- **Arquitectura:** `ENGRAMA V2` (Sin Atención, Invarianza Causal).
- **Parámetros totales:** `1,371,056`
- **Configuración:** `d_model=160, d_gate=20, d_ff=320, C=4, L_enc=1, L=3, M=2`
- **Vocabulario:** 17 tokens de caracteres (dígitos y operadores).
- **Épocas:** 60 sobre 4,500 líneas de datos.
- **Tiempo de Entrenamiento:** 376.1 segundos ( CPU ).

### Resultados Cuantitativos
- **Loss inicial:** `1.5722`
- **Loss final (Época 60):** `0.5540`
- **Reducción de Loss:** `64.8%`
- **Loss de evaluación:** `0.5353`
- **Precisión en Generación Autoregresiva Token-a-Token:** **75.0%** (15/20 aciertos exactos).
- **Verificación de Invarianza Causal V2:** **PASÓ** (`max_diff = 1.62e-05` < tolerancia `1e-4`).
- **Verificación de Serialización:** **OK** (`max_diff = 0.00e+00`).

### Ejemplos de Evaluación de Generación (Token por Token con Caché V2)
```text
  Prompt: "5+5="    -> Generado: "10"  [OK]
  Prompt: "10+10="  -> Generado: "20"  [OK]
  Prompt: "15+14="  -> Generado: "29"  [OK]
  Prompt: "0+0="    -> Generado: "0"   [OK]
  Prompt: "20+9="   -> Generado: "29"  [OK]
  Prompt: "25+25="  -> Generado: "50"  [OK]
  Prompt: "12+12="  -> Generado: "24"  [OK]
  Prompt: "9+9="    -> Generado: "18"  [OK]
  Prompt: "6+7="    -> Generado: "13"  [OK]
  Prompt: "11+11="  -> Generado: "22"  [OK]
  Prompt: "13+13="  -> Generado: "26"  [OK]
```

## 8. Análisis de Costes Computacionales
- **Codificación:** $O(L_{enc} \cdot C^2 \cdot d)$ por token (totalmente paralelizable).
- **Consolidación en Inferencia:** $O(L \cdot K \cdot d)$ por token (independiente de $N$, donde $K = |offsets| \approx \log_2 N$).
- **Memoria de Caché:** $O((L + 1) \cdot N \cdot d)$ floats en RAM/VRAM.
