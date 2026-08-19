# AGENTS.md - Guía Estricta de Trabajo y Arquitectura ENGRAMA

Instrucciones compactas de alto nivel para agentes de OpenCode trabajando en este repositorio.

## 1. Reglas Absolutas (Cero Excepciones)
- **Prohibición de Atención / $QK^T$:** No calcular similitudes dinámicas $QK^T$, matrices de afinidad $N \times N$, ni softmax sobre dimensión de secuencia.
- **Prohibición de Arquitecturas Existentes:** No usar Mamba (SSM), RWKV (decaimiento exponencial), RetNet, ni WaveNet puro. Basarse 100% en `ENGRAMA-Paper-Final-Verificado.md`.
- **PyTorch Puro:** Cero dependencias de `transformers` (HuggingFace) para la arquitectura base.
- **Fidelidad Matemática:** Implementación exacta de ecuaciones de Célula, Sinapsis, Mezcla Dilatada y Evocador sin simplificaciones de rendimiento.
- **Invarianza Causal Obligatoria:** Las pruebas deben verificar que `forward(seq)` (paralelo) == `step_forward(seq)` (incremental con `EngramaCache`) con error máximo $< 1e-4$.

## 2. Estructura de Módulos y Arquitectura
- `engrama/`
  - `config.py`: Dataclass `EngramaConfig` ($d=512, C=16, d_g=64, N_{max}=2048, M=4$).
  - `primitives.py`: `LayerNorm`, `Cell` ($d \to 4d \to d$), `SynapseLayer` ($C \times C$ con gating $d_g \ll d$).
  - `encoder.py`: `IsolatedEncoder` (Fase 1: $T_0[i]$ depende exclusivamente del token $x_i$, 0% contexto).
  - `trace.py`: `EngramaCache` (Fase 2: Buffer FIFO circular de tamaño $N_{max}$ con timestamps).
  - `consolidation.py`: `PositionalDilatedMix` y `ConsolidationStack` (Fase 3: offsets en potencias de 2 $P_l(i)$, pesos estáticos por offset relativo $W_{l,p}$).
  - `evoker.py`: `MultiCandidateEvoker` (Fase 4: $M \in [1, 8]$ candidatos con LogSumExp/Max/Mean).
  - `tokenizer.py`: `EngramaTokenizer` (Basado en caracteres).
  - `model.py`: `EngramaModel` (Integrador end-to-end con `forward` y `step_forward`).
  - `trainer.py` & `io.py`: Entrenador CrossEntropy autoregresivo y guardado `.pt` + JSON.
- `tests/`: Pruebas matemáticas, de aislamiento, invarianza causal y tensores.
- `examples/demo.py`: Ciclo de entrenamiento y generación token a token.

## 3. Comandos de Ejecución y Verificación
- **Ejecutar pruebas unitarias:** `python -m unittest discover tests`
- **Ejecutar demo y verificación end-to-end:** `python examples/demo.py`
- **Flujo de Trabajo:** Consultar `PLAN.md` para especificaciones y `TASKS.md` para el desglose granular de tareas y verificaciones antes de marcar avances.
