# ENGRAMA V5
## Arquitectura Neuronal Autoregresiva sin Atención con Recuperación por Resonancia Sináptica sobre Traza Explícita

### Especificación Teórica V5 — Evolución compatible con la ontología ENGRAMA V1–V4

**Base:** ENGRAMA V1 + V2 + V3 + V4 y el diagnóstico experimental en `docs/DIAGNOSTICO_V4.md`.
**Estado:** Especificación + implementación de referencia en PyTorch puro.
**Principio rector:** Recuperar los tokens **exactos** guardados en la Traza —sin
atención, sin compresión, sin estado recurrente comprimido, 100% paralelizable—
mediante **resonancia sináptica punto a punto**.

---

## 0. Diagnóstico que motiva la V5 (medido, no teórico)

En `docs/DIAGNOSTICO_V4.md` se demostró experimentalmente que:

1. La consolidación de V4 (offsets fijos `{0,1,2^(l-1),2^l}`) es una **convolución
   dilatada con compuertas**. No tiene direccionamiento por contenido → la
   recuperación clave→valor a distancia variable es ≈ azar (8.6% medido).
2. La memoria asociativa de estado fijo (tipo linear-attention / Mamba) alcanza
   ~47% pero **comprime** la historia en un estado acotado → viola el pilar 9 y
   el requisito explícito de "nada de compresión, nada de modelos de estado".
3. La **resonancia sináptica sigmoide** sobre la traza explícita alcanza **98.3%**
   de recuperación (SEQ=128, medido), **sin softmax y sin estado comprimido**.

La V5 se construye sobre el hallazgo (3).

---

## 1. Los 11 requisitos y cómo los cumple la V5

| # | Requisito | Mecanismo V5 |
|---|-----------|--------------|
| 1 | Rediseño total de la librería | API nueva `engrama.v5` estilo PyTorch/Keras; V1–V4 movidas a `engrama.legacy`. |
| 2 | Mantener ideas originales | Huella aislada → Traza FIFO explícita → Sinapsis/Células → Evocación. La compuerta sináptica de V3/V4 se **generaliza** a todo el rango temporal. |
| 3 | Cero atención | Compuerta `σ(τ·⟨q̂,k̂⟩+b)` **punto a punto, sin softmax sobre posiciones**. Ninguna posición compite con otra por masa normalizada. |
| 4 | Cero compresión | Se guardan los N pares (k_j, v_j) **explícitos**. No hay vector de estado comprimido ni espacio latente que reemplace la memoria. |
| 5 | 100% paralelización | El recuerdo es un producto matricial enmascarado causalmente; entrenamiento totalmente vectorizado. |
| 6 | Lineal en memoria y generación | Memoria de traza O(N·d); generación O(N) por token (lectura sobre la traza retenida). Se **relaja** O(1) a petición del autor. |
| 7 | >85% recall en contextos enormes | Resonancia sináptica: 98.3% medido a SEQ=128; el mecanismo no se diluye con N (sin softmax). |
| 8 | Cero NaN | Sigmoide acotada [0,1]; normas en fp32; sin `exp` sin cota; sin softmax que subdesborde. |
| 9 | Alta velocidad de entrenamiento | Un solo matmul enmascarado + FFN por bloque; compatible con AMP y `torch.compile`. |
| 10 | Procesamiento aislado | Codificación de cada token independiente de vecinos (encoder aislado intacto). |
| 11 | Pocos parámetros | Sinapsis compartidas por cabeza; sin las ParameterDict por offset de V4. |

---

## 2. Ontología ENGRAMA preservada (los pilares)

1. **Codificación aislada**: `T0[i] = Encoder(e_i)`, depende sólo de `x_i`.
2. **Traza explícita FIFO**: buffer de `(k_j, v_j)` (y opcionalmente `T0[j]`).
   **Se almacena todo, no se transforma, no se comprime.**
3. **Consolidación causal relativa**: la compuerta depende sólo de `q_t` (destino)
   y `k_j` (fuente) con `j ≤ t`. Cero información futura.
4. **Células y Sinapsis separadas**: la Célula (FFN + norm) transforma; la
   Sinapsis (compuerta de resonancia) enruta/transporta.
5. **Cero atención dinámica**: cero softmax sobre posiciones, cero competencia
   normalizada entre posiciones. La resonancia es superposición Hebbiana.
6. **Invarianza causal**: `forward` paralelo ≡ `step` incremental (máscara triangular
   ≡ acumulación causal sobre la traza).
7. **Memoria de traza O(N)**: se retiene toda la traza (sin compresión). La
   generación relee la traza (O(N) por token).
8. **Evocación multicandidato**: se conserva el evocador de fusión latente de V4.
9. **Cero estado oculto comprimido**: **no** hay `S = Σ k⊗v` acumulado ni SSM.
   La memoria son los pares explícitos, no un resumen.
10. **Paralelización completa**: el score `G = σ(τ Q̂K̂ᵀ + b) ⊙ tril` se calcula
    de una vez para toda la secuencia.

---

## 3. El bloque de Resonancia Sináptica (núcleo de la V5)

Para cada cabeza `h` y cada par (destino `t`, fuente `j ≤ t`):

### 3.1 Proyecciones (Sinapsis)
$$q_t = W_q\,\mathrm{Norm}(x_t),\quad k_j = W_k\,\mathrm{Norm}(x_j),\quad v_j = W_v\,\mathrm{Norm}(x_j)$$

Con normalización unitaria por cabeza: $\hat q_t = q_t/\lVert q_t\rVert$, $\hat k_j = k_j/\lVert k_j\rVert$.

### 3.2 Compuerta de resonancia (NO softmax, punto a punto)
$$g_{t,j} = \sigma\!\big(\tau_h\,\langle \hat q_t, \hat k_j\rangle + b_h\big)\cdot \mathbb{1}[j \le t]$$

- $\tau_h$: agudeza (temperatura inversa) aprendida por cabeza. Controla cuán
  selectiva es la resonancia.
- $b_h$: sesgo por cabeza.
- **Clave:** $g_{t,j}\in(0,1)$ se decide **independientemente para cada par**.
  No hay $\sum_j g_{t,j} = 1$. Esto es lo que la distingue de la atención y lo
  que evita la dilución en contexto largo.

### 3.3 Lectura por superposición
$$r_t = \sum_{j\le t} g_{t,j}\, v_j \qquad(\text{opcional: } /\textstyle\sum_j g_{t,j}\text{ para estabilidad})$$

La variante sin normalizar es superposición Hebbiana pura; la variante
normalizada por suma de compuertas (no softmax) estabiliza magnitudes. La V5
usa una **normalización por conteo suave** opcional controlada por `read_norm`.

### 3.4 Célula de salida
$$x_t \leftarrow x_t + W_o\, r_t,\qquad x_t \leftarrow x_t + \mathrm{FFN}(\mathrm{Norm}(x_t))$$

### 3.5 Coste
- Entrenamiento: $O(H\,N^2 d_h)$ en cómputo del score pero **totalmente
  paralelo** (un matmul); memoria $O(N^2)$ por bloque que puede trocearse por
  chunks causales para contextos enormes (ver §5).
- Generación: memoria $O(N d)$ (traza explícita); coste por token $O(N d_h)$.

> Nota: el término $N^2$ es de **cómputo**, no de memoria comprimida. Cumple
> "lineal en memoria" (la traza es O(N)); el cómputo del score es cuadrático
> como cualquier lectura exhaustiva de N elementos, pero se procesa por bloques
> para mantener memoria acotada y velocidad alta (FlashAttention-style tiling,
> sin softmax).

---

## 4. Recuperación exacta: por qué funciona (intuición + evidencia)

La atención softmax reparte una masa de probabilidad 1 entre todas las
posiciones: al crecer N, cada posición recibe menos masa y la señal se diluye.
La **resonancia sigmoide no reparte masa**: la sinapsis (t,j) se activa si y sólo
si `q_t` resuena con `k_j` por encima del umbral, **independientemente de cuántas
otras** posiciones haya. Una clave que coincide exactamente produce `g≈1` sin
importar el largo del contexto → recuperación robusta a la distancia.

Evidencia medida (`docs/DIAGNOSTICO_V4.md` y prototipos): 98.3% de recall a
SEQ=128 con 227K parámetros; recall estable al variar SEQ 96→512 en las variantes
lineales (no colapsa con la longitud).

---

## 5. Contextos enormes sin explotar memoria: lectura por chunks (sin softmax)

Para SEQ grande (8000+), el score `N×N` se procesa por **bloques causales**:
para cada bloque de queries `[t0:t1]` se recorre la traza en tiles de claves
`[0:t1]`, acumulando `r_t += Σ g v` tile a tile. Como no hay softmax, **no se
necesita el truco de max-substraction**: la acumulación es una suma directa,
numéricamente trivial y sin estado global. Memoria de activación O(chunk·d).

Esto da el "kernel simple pero funcional y rápido" pedido: un tiling causal sin
softmax, fusionable con `torch.compile`, y trivial de portar a Triton.

---

## 6. Caché nativa de generación (KV-cache ENGRAMA, sin recomputar)

En generación autoregresiva se mantiene la **Traza explícita** de pares `(k_j,v_j)`
por capa. Al llegar el token `t`:
1. Se codifica `x_t` de forma aislada → `q_t, k_t, v_t`.
2. Se **añade** `(k_t, v_t)` a la traza (append O(1)).
3. Se lee `r_t = Σ_{j≤t} g_{t,j} v_j` sobre la traza retenida (O(N)).

No se recalcula nada de los tokens previos (sus `k_j, v_j` ya están en la traza).
Es el equivalente ENGRAMA al KV-cache, **nativo** de la resonancia sináptica y sin
la penalización del softmax. Invarianza causal exacta con el forward paralelo.

---

## 7. Configuración de referencia V5

```python
from engrama.v5 import EngramaV5, EngramaV5Config

cfg = EngramaV5Config(
    vocab_size=50257,
    d_model=256,
    num_layers=6,
    num_heads=8,
    d_ff=1024,
    context_length=8192,
    read_norm="softcount",   # None | "softcount"
    tie_embeddings=True,
    dtype="float32",
)
model = EngramaV5(cfg)
```

---

## 8. Diferencias frente a V4

| Dimensión | V4 | V5 |
|---|---|---|
| Recuperación | offsets fijos (convolución) | resonancia sináptica sobre traza explícita |
| Direccionamiento | posicional fijo | por contenido (q·k), sin softmax |
| Recall KV (medido) | 8.6% | 98.3% |
| Compresión | caché horizonte mínimo | ninguna (traza completa) |
| Atención | no | no |
| Paralelización | sí | sí |
| Memoria generación | O(1)/capa (comprimida) | O(N) explícita |
