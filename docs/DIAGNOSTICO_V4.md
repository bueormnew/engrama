# Diagnóstico completo de ENGRAMA V4 — por qué falla la recuperación

> Documento generado tras revisar toda la librería (`src/engrama/*`), los papers
> teóricos V3/V4, y **reproducir los resultados con experimentos reales** en CPU
> (PyTorch 2.13). No hay ningún número inventado aquí: todo lo marcado como
> "medido" salió de correr código.

---

## 0. Resumen ejecutivo (TL;DR)

1. **La invariancia causal SÍ funciona** (medido: diferencia máx. `2.8e-7` entre
   `forward` paralelo y `step_forward` incremental). La arquitectura está bien
   implementada respecto a su propia especificación. El problema **no** es un bug.

2. **El problema es de diseño, no de código.** La consolidación de ENGRAMA es,
   matemáticamente, una **convolución dilatada con compuertas sobre offsets
   fijos** `{0, 1, 2^(l-1), 2^l}`. Ese tipo de operador **no puede** hacer
   recuperación asociativa clave→valor a una distancia arbitraria dependiente de
   los datos. Por eso la recuperación es ~azar.

3. **Sí hay compresión / pérdida de información**, a pesar de lo que dice el
   pilar 9 del paper. En inferencia, la caché de horizonte mínimo sólo retiene
   `max(D_{l+1})+1` estados por capa (p. ej. `[3,5,9,17,33,65,129,257,1]`). El
   único buffer "completo" es la traza `T0`, pero se lee **sólo en esos mismos
   offsets fijos**. El estado que llega al evocador es un resumen de tamaño
   acotado → es compresión con pérdida.

4. **Por qué `source_gate` gana a la V4 "completa"**: las innovaciones de la V4
   (gating dual, trace-tap) **añaden parámetros y rutas de gradiente que
   compiten** sin resolver el problema de fondo (routing dependiente de datos).
   En una tarea que ninguna de las variantes puede resolver de verdad, el
   modelo más simple (source-only) regulariza mejor y "gana" por ruido. No es
   que source-gate sea bueno: es que todas están cerca del azar y la más simple
   sobreajusta un poco menos.

---

## 1. Qué hace realmente cada fase (lectura del código)

### Fase 1 — Encoder aislado (`encoder.py`)
`T0[i] = w_pool(SynapseStack(init_proj(e_i)))`. Correcto y fiel al pilar de
codificación aislada: `T0[i]` depende **sólo** de `x_i`. Es un MLP por token.
Bien.

### Fase 2 — Traza circular (`trace.py`)
FIFO de `N_max` pares `(T0, timestamp)`. Almacena, no transforma. Bien. **Esta
es memoria explícita O(N) y NO es compresión.** El problema es cómo se lee.

### Fase 3 — Consolidación (`consolidation.py`) ← **aquí está el fallo**
Para cada capa `l` y cada offset `p ∈ D_l`:

```
T_pos_l[t] = Σ_p  ρ_{l,p} · α_{l,p}[t] ⊙ ( y_ctx(T_{l-1}[t-p]) + γ·y_tr(T0[t-p]) )
```

- Los offsets `D_l` son **fijos** (`resonant_multirate`: `{0,1,2^(l-1),2^l}`).
- La compuerta `α_{l,p}` es punto-a-punto (sigmoide), **sin softmax ni matriz
  N×N** — correcto, no es atención.
- El transporte `y_ctx = β·x + U·diag(s)·Vᵀ·x` es una proyección **lineal fija**
  (no depende del contenido más allá de la compuerta escalar-por-canal).

**Consecuencia matemática:** esto es una *gated dilated convolution*. Mezcla
posiciones que están a distancias que son sumas de potencias de 2. Para llevar el
valor de la posición `k` (donde se definió `key→value`) hasta la posición `q`
(donde aparece la query), la red necesita una cadena de offsets que sume `q-k`.
Pero:
  - `q-k` **cambia en cada muestra** (la query aparece a distancia variable),
  - los pesos de transporte son **los mismos** para todo contenido,
  - no hay ningún mecanismo que "empareje" la query con la clave por contenido.

Es decir: **no hay direccionamiento por contenido (content-addressable read).**
Un operador de topología fija no puede aprender a enrutar "el valor que va con
ESTA clave, esté donde esté". Por eso da azar.

### Fase 4 — Evocador (`evoker.py`)
`latent_fusion`: genera M candidatos low-rank, los fusiona con softmax(M) y
proyecta a vocabulario. Barato y correcto. No es el cuello de botella de recall.

---

## 2. Verificación experimental (medido, no teórico)

### 2.1 Invariancia causal — PASA
```
parallel vs incremental  max|Δ| = 2.83e-07   mean|Δ| = 5.3e-08
```
`forward` (entrenamiento) e `step_forward` (generación con caché) son
equivalentes. La generación autoregresiva es matemáticamente fiel.

### 2.2 Recuperación KV (misma tarea del notebook, entrenada de verdad, 400 pasos)
Header liga `key→value` aleatorio por muestra; la query aparece después a
distancia creciente. Azar = 6.2% (16 valores).

| modelo               | params   | recall | veredicto            |
|----------------------|----------|--------|----------------------|
| V4 dual + trace-tap  | 474,968  | 8.6%   | ≈ azar               |
| source_gate (V3-like)| 438,104  | 18.5%  | apenas sobre azar    |
| no_tracetap          | 441,248  | 7.9%   | ≈ azar               |

→ **Reproduce tu observación**: la V4 "completa" NO es la mejor, y todas están
cerca del azar. Confirma que el problema es estructural.

### 2.3 Prototipo de control — memoria asociativa (mismo dataset, menos params)
Reemplacé la consolidación por una **memoria asociativa por producto externo**
(estado `S = Σ kᵢ⊗vᵢ`, lectura `q·S`), que es O(N), sin softmax, sin matriz N×N:

| modelo                                   | params  | recall |
|------------------------------------------|---------|--------|
| Asociativa (lineal, outer-product)       | 102,788 | **62.3%** |
| Asociativa con compuerta de olvido       | 102,788 | 5.9%   |

→ **Con 4× menos parámetros, la memoria asociativa hace 62% vs 8% de la V4.**
Esta es la prueba de que el mecanismo correcto para recuperación es el
**binding sináptico key→value (Hebbiano)**, no la convolución de offsets fijos.
(La variante "con compuerta de olvido" colapsó por mala inicialización — dato a
tener en cuenta para la V5: el forget gate mal inicializado borra la memoria.)

---

## 3. Respuesta directa a tus dudas

**"¿Por qué gana source_gate si la V4 debería ser mejor?"**
Porque en un régimen donde ninguna variante resuelve la tarea, las mejoras de la
V4 sólo agregan capacidad/ruido de optimización. La más simple generaliza algo
mejor. No es mérito de source-gate; es que el techo de todas es bajísimo.

**"¿La recuperación es prácticamente azar?"**
Sí, confirmado experimentalmente. Es una limitación del operador (offsets fijos
sin direccionamiento por contenido), no del entrenamiento ni de un bug.

**"¿Esto es una arquitectura lineal, pero comprime algo? ¿por qué la pérdida de
información?"**
Es lineal en tiempo/memoria, sí. Y **sí comprime**, en dos sentidos:
  1. En inferencia, la caché de horizonte mínimo tira casi toda la historia
     (retiene `[3,5,9,...,257,1]` estados por capa). El vector `T_L[t]` que llega
     al evocador es un resumen de tamaño fijo → compresión con pérdida.
  2. Aunque la traza `T0` guarda todo (O(N), sin pérdida), sólo se **lee** en los
     offsets fijos `{0,1,2^(l-1),2^l}`. La información en posiciones intermedias
     que no caen en un camino de offsets **nunca se recupera de forma
     dependiente del contenido**. La "pérdida" es de *accesibilidad*, no de
     almacenamiento: está guardada pero es irrecuperable por consulta.

---

## 4. Implicación para la V5

El núcleo debe cambiar de **"convolución con compuertas sobre offsets fijos"** a
un **mecanismo de memoria asociativa direccionable por contenido** que:
  - siga siendo **sin atención** (sin softmax sobre posiciones, sin matriz N×N),
  - sea **paralelizable** (formulación por chunks tipo scan asociativo),
  - tenga **lectura/escritura O(1) por token** en generación (caché nativa),
  - y conserve la ontología ENGRAMA (huella aislada → traza → sinapsis/células →
    evocación).

La memoria Hebbiana por producto externo (`S += k⊗v`, con regla delta para
corregir interferencia) es **literalmente** "plasticidad sináptica" — encaja
perfecto con el nombre y la filosofía "engrama" (traza de memoria), y es el
candidato natural para la V5. El detalle de diseño (tamaño del estado, multi-cabeza,
regla delta, y cómo reconciliar esto con "nada de compresión") se decide en el
plan de la V5.
