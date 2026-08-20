# ENGRAMA V3
## Arquitectura Neuronal Autoregresiva sin Atención con Huella Aislada, Traza Circular, Consolidación Jerárquica Discreta, Sinapsis Factorizadas, Transporte Residual de Fidelidad y Caché Jerárquico de Horizonte Mínimo

### Propuesta teórica V3 — Evolución compatible con ENGRAMA V1 + V2

**Autor de la propuesta:** Duvan Felipe Buenahora Ormaza
**Base:** ENGRAMA V1 + V2 proporcionado en `ENGRAMA-Paper-Final-Verificado.md`
**Estado:** Propuesta teórica V3; no constituye todavía validación experimental.
**Principio rector:** mejorar recuperación, eficiencia y número de parámetros sin introducir atención ni abandonar la filosofía huella → Traza → consolidación → evocación.

---

# 0. Declaración de compatibilidad

ENGRAMA V3 no sustituye la arquitectura central de ENGRAMA. La conserva y modifica únicamente mecanismos internos que presentan una oportunidad clara de mejora.

La V3 mantiene obligatoriamente:

1. Codificación aislada: cada token se codifica sin observar tokens vecinos.
2. Traza explícita y FIFO: la memoria de trabajo almacena huellas y timestamps y no transforma contenido.
3. Consolidación causal: una posición solo depende de posiciones pasadas o de sí misma.
4. Pesos relativos por offset: nunca pesos dependientes de posición absoluta.
5. Células y Sinapsis como separación conceptual entre transformación y conexión/gating.
6. Evocación desde el último vector consolidado.
7. Generación autoregresiva.
8. Caché basada en invariancia causal.
9. Entrenamiento completamente paralelizable sobre la secuencia.
10. Ausencia completa de atención dinámica, búsqueda `QK^T`, softmax sobre posiciones, estado recurrente comprimido y decaimiento exponencial.

La V3 se basa directamente en las propiedades establecidas en V1/V2: codificación independiente, Traza que almacena, pesos relativos por offset, consolidación causal y caché invariante. [V1/V2, líneas 30–43, 64–77, 216–244, 246–302, 329–394].

---

# 1. Problemas que V3 intenta resolver

La V2 ya resuelve el gran cuello de botella del recálculo mediante la caché causal, pero quedan cuatro problemas teóricos importantes.

## 1.1 Explosión paramétrica de Sinapsis

Con `C=64`, una conexión completa entre dos capas de Células contiene:

```text
64 × 64 = 4096 Sinapsis
```

Si cada Sinapsis utiliza una matriz `d × d`, el coste paramétrico es:

```text
4096 × d²
```

Para `d=1024`, eso equivale a aproximadamente 4.29 mil millones de parámetros por capa de conexiones antes de contar gating, células y demás parámetros.

La V2 ya reconoce esta dificultad y propone factorización práctica, pero V3 convierte esa idea en parte formal de la arquitectura de referencia. [V1/V2, líneas 99–141].

## 1.2 Coste repetido innecesariamente en consolidación

La V2 utiliza en cada capa el conjunto completo de anclas:

```text
{0,1,2,4,8,16,...}
```

Esto produce `K≈log N` operaciones por capa, aunque el alcance global puede obtenerse repartiendo las escalas entre las capas.

V3 propone una consolidación **jerárquica discreta**, donde cada capa se especializa en una escala principal. El campo receptivo completo se conserva por composición binaria, mientras el número de anclas por capa baja de aproximadamente `log N` a una cantidad constante.

## 1.3 Dilución de señales lejanas

El transporte de una huella distante a través de muchas mezclas puede modificarla repetidamente y dificultar la recuperación exacta.

V3 incorpora una ruta de identidad dentro de cada Sinapsis: una parte puede transportar directamente la representación fuente y otra parte puede transformarla mediante una proyección de bajo rango.

Esto no añade atención ni una memoria nueva. Es un mecanismo de **fidelidad de transporte** para la misma información ya almacenada.

## 1.4 Redundancia de caché

V2 propone guardar `T0` y una lista de longitud `N` para cada capa consolidada. Sin embargo, con la consolidación jerárquica V3 cada capa solo necesita consultar un horizonte temporal determinado por su propia dilatación.

V3 conserva toda la Traza como memoria explícita, pero almacena las representaciones consolidadas con **horizontes mínimos por capa**.

---

# 2. Idea central de V3

La V3 puede resumirse como:

```text
TOKEN
  │
  ▼
HUELLA AISLADA
  │
  ▼
TRAZA CIRCULAR FIFO
  │
  ▼
CONSOLIDACIÓN JERÁRQUICA
  │
  ├── escala local
  ├── escala binaria
  ├── escala binaria
  ├── ...
  └── escala global
  │
  ▼
TRANSPORTE CON IDENTIDAD + TRANSFORMACIÓN LOW-RANK
  │
  ▼
CACHE DE HORIZONTE MÍNIMO
  │
  ▼
EVOCADOR MULTI-CANDIDATO
  │
  ▼
TOKEN SIGUIENTE
```

La filosofía sigue siendo exactamente:

```text
huella aislada → almacén explícito → consolidación → evocación
```

V3 no introduce ninguna etapa de recuperación mediante atención.

---

# 3. Notación V3

Sea:

- `|V|`: tamaño del vocabulario.
- `d`: dimensión de la Célula.
- `d_g`: dimensión reducida del gating.
- `C`: número de Células.
- `L_enc`: capas de codificación.
- `L`: capas de consolidación.
- `N_max`: capacidad máxima de la Traza.
- `M`: número de candidatos del Evocador.
- `r`: rango de las proyecciones de Sinapsis, con `r << d`.
- `D_l`: offsets utilizados específicamente por la capa `l`.
- `R_l`: horizonte de dependencia necesario en la capa `l`.
- `T_0`: Traza codificada.
- `T_l`: representación consolidada de la capa `l`.

V3 conserva la definición de timestamp absoluto de V2, pero el timestamp continúa siendo metadata de la Traza y no entra como coordenada absoluta dentro de los pesos del modelo.

---

# 4. Invariantes que V3 jamás puede romper

## 4.1 No atención

No existe ninguna operación de la forma:

```text
Q = XWq
K = XWk
V = XWv
softmax(QKᵀ)V
```

ni equivalente funcional basada en similitud dinámica entre una posición y todas las demás.

## 4.2 No selección dinámica entre posiciones

La selección de anclas es determinada exclusivamente por offsets predefinidos por arquitectura.

Un gate puede depender del contenido de su fuente, como en V2, pero nunca se utiliza para comparar una consulta con un conjunto arbitrario de posiciones y escoger una posición según similitud global.

## 4.3 La Traza no transforma

La Traza continúa almacenando:

```text
(vector, timestamp)
```

No aplica redes, gates ni mezclas.

## 4.4 La consolidación no crea información externa

Todas sus entradas proceden de representaciones ya almacenadas o previamente consolidadas.

## 4.5 Causalidad estricta

Para toda posición `i` y tiempo futuro `t>i`:

```text
T_l^new[i] = T_l^old[i]
```

cuando no se modifica la propia ventana histórica por overflow FIFO.

---

# 5. V3 — Célula optimizada

La Célula V2 utiliza un FFN de dimensión `4d`:

$$
Cell_l(x)=x+W^{(2)}_l\phi(W^{(1)}_lLN(x)+b_l^{(1)})+b_l^{(2)}.
$$

V3 mantiene esta función conceptual, pero permite una **Célula de núcleo compartido por capa**.

## 5.1 Núcleo compartido

En lugar de aprender un FFN completamente distinto para cada Célula `b`, la capa `l` tiene:

$$
F_l(x)=W^{(2)}_l\phi(W^{(1)}_lLN(x)+b^{(1)}_l)+b^{(2)}_l.
$$

Cada Célula conserva identidad mediante una modulación diagonal:

$$
Cell_{l,b}(x)=x+s_{l,b}\odot F_l(n_{l,b}\odot x+q_{l,b}).
$$

Donde:

- `s_{l,b} ∈ R^d` es una escala por Célula.
- `n_{l,b} ∈ R^d` es una modulación de entrada.
- `q_{l,b}` puede ser omitido en el modo mínimo.

Esto conserva la existencia de Células individuales, pero elimina la necesidad de una matriz FFN completamente independiente por Célula.

## 5.2 Ventaja

En V2, `C` Células independientes pueden requerir aproximadamente:

$$
C\cdot 2d\cdot d_{ff}
$$

parámetros de FFN.

Con núcleo compartido:

$$
2d\cdot d_{ff}+O(Cd).
$$

La reducción es aproximadamente `C×` en el componente FFN cuando el término de modulación es pequeño.

## 5.3 Compatibilidad conceptual

La Célula continúa siendo una unidad independiente con estado propio y recibe información de otras Células únicamente a través de Sinapsis.

Lo que se comparte es el **mecanismo de transformación**, no el estado ni las conexiones.

---

# 6. V3 — Sinapsis Factorizada con Transporte de Fidelidad

Esta es una de las modificaciones principales.

## 6.1 Problema V2

Una Sinapsis puede definirse como:

$$
z_{a\to b}=W_{a\to b}h_a,
$$

con `W_{a→b} ∈ R^{d×d}`.

Esto es expresivo, pero demasiado costoso.

## 6.2 Factorización V3

V3 define:

$$
W_{a\to b}^{(l)}h
=
\beta_{a\to b}^{(l)}h
+
U_l\operatorname{Diag}(s_{a\to b}^{(l)})V_l^Th.
$$

Donde:

- `U_l ∈ R^{d×r}` es compartida por todas las Sinapsis de la capa.
- `V_l ∈ R^{d×r}` es compartida por todas las Sinapsis de la capa.
- `s_{a→b}^{(l)} ∈ R^r` es exclusivo de cada Sinapsis.
- `β_{a→b}^{(l)}` es un escalar exclusivo de cada Sinapsis.
- `r << d`.

Esto significa que cada Sinapsis sigue siendo individual, pero su transformación aprende dentro de un subespacio compartido.

## 6.3 Complejidad de una Sinapsis

En lugar de:

$$
O(d^2)
$$

la transformación se aproxima a:

$$
O(dr).
$$

Para `r=32` y `d=1024`, el orden de la reducción es aproximadamente:

```text
d² = 1,048,576
 d·r = 32,768
```

unas 32 veces menos operaciones para la transformación lineal, ignorando costes de memoria y fusión de kernels.

## 6.4 Por qué el término identidad es importante

El término:

$$
\beta h
$$

permite transportar una huella sin obligarla a atravesar una transformación de rango reducido.

Esto crea una ruta de fidelidad:

```text
huella original
   │
   ├──────── identidad ────────┐
   │                           │
   └── transformación low-rank │
                               ▼
                         siguiente Célula
```

La transformación puede aprender abstracción; la identidad puede conservar señal.

No es una memoria adicional.

No es atención.

Es una propiedad de transporte dentro de la misma Sinapsis.

---

# 7. Gating V3

V2 utiliza una proyección reducida y un vector de gate por conexión.

V3 conserva la idea de gate local, pero evita repetir proyecciones innecesarias.

Primero:

$$
q_{i,a}^{(l)}=P_gH_{i,a}^{(l)}
$$

una sola vez por Célula fuente.

Después, para cada Sinapsis:

$$
\alpha_{i,a\to b}^{(l)}
=
\sigma((w_{a\to b}^{(l)})^Tq_{i,a}^{(l)}+b_{a\to b}^{(l)}).
$$

La Sinapsis final es:

$$
o_{i,a\to b}^{(l)}=
\alpha_{i,a\to b}^{(l)}
\left(
\beta_{a\to b}^{(l)}h_{i,a}^{(l)}
+
U_l\operatorname{Diag}(s_{a\to b}^{(l)})V_l^Th_{i,a}^{(l)}
\right).
$$

La puerta continúa siendo local a una conexión.

---

# 8. V3 — Consolidación Jerárquica Discreta

Este es el segundo cambio central.

La V2 permite que cada capa utilice:

```text
0,1,2,4,8,16,...
```

La V3 observa que el campo receptivo global puede obtenerse repartiendo las escalas entre capas.

## 8.1 Conjunto de offsets por capa

Definimos:

$$
D_l=\{0,1,2^l\}
$$

para las capas internas, adaptando `2^l` al rango máximo de la Traza.

Por tanto:

```text
capa 0: {0,1,1} → se normaliza a {0,1}
capa 1: {0,1,2}
capa 2: {0,1,4}
capa 3: {0,1,8}
capa 4: {0,1,16}
...
```

El duplicado en la primera capa se elimina.

Una variante todavía más minimalista utiliza:

$$
D_l=\{0,2^l\}.
$$

La configuración `local+dyadic` `{0,1,2^l}` es el default recomendado de V3 porque mantiene una ruta local estable mientras conserva el crecimiento exponencial.

## 8.2 Consolidación

$$
T_{pos,l}[i]
=
\sum_{p\in D_l, p\le i}
G_{l,p}(T_{l-1}[i-p])
$$

con:

$$
G_{l,p}(x)=
\alpha_{l,p}(x)
\left(
\beta_{l,p}x+U_l\operatorname{Diag}(s_{l,p})V_l^Tx
\right).
$$

Después:

$$
T_l[i]=Cell_l(T_{pos,l}[i]).
$$

---

# 9. ¿Se conserva el campo receptivo?

Sí, en principio.

Si cada capa introduce una dependencia de offset `2^l`, una ruta puede acumular:

$$
0+1+2+4+8+...+2^{L-1}
$$

y por representación binaria pueden alcanzarse posiciones dentro de un rango exponencial.

Para una secuencia de `L` escalas:

$$
R_L \sim 2^L-1.
$$

La presencia adicional del offset `1` introduce rutas locales adicionales.

Por tanto V3 busca mantener alcance global mediante composición, en vez de hacer que cada capa consulte todos los niveles de escala.

**Esta es una hipótesis arquitectónica que debe ser validada experimentalmente; el mantenimiento del campo receptivo máximo no garantiza por sí mismo la misma calidad de recuperación que V2.**

---

# 10. Mejora de recuperación: Transporte Residual Multiescala

El problema no es solamente que una posición distante sea accesible. Debe llegar con suficiente señal.

V3 utiliza tres vías dentro de la misma consolidación:

```text
1. identidad
2. transformación low-rank
3. refinamiento local
```

## 10.1 Identidad

$$
I_l(h)=\beta_lh.
$$

## 10.2 Transformación

$$
F_l(h)=U_l\operatorname{Diag}(s_l)V_l^Th.
$$

## 10.3 Refinamiento local

El offset `1` actúa como conexión vecina fija cuando está presente.

La información remota puede atravesar varias capas mediante las conexiones diádicas y mantener una trayectoria de identidad en cada etapa.

## 10.4 Interpretación

Una huella puede seguir la ruta:

```text
T0[posición remota]
   ↓
identidad
   ↓
identidad
   ↓
low-rank
   ↓
identidad
   ↓
refinamiento local
   ↓
TL[t]
```

Sin atención y sin comparar contenido contra todas las posiciones.

---

# 11. V3 — Ancla Global Determinista Opcional

Aunque la jerarquía diádica proporciona alcance exponencial, V3 mantiene una opción explícita de ancla global para contextos donde se quiera acceso directo al extremo de la ventana.

Definimos opcionalmente:

$$
D_l=\{0,1,2^l,g(N)\}
$$

pero `g(N)` solo se utiliza en una cantidad pequeña de capas, idealmente una sola.

Por defecto:

```text
sin ancla global
```

Modo:

```text
with_global_anchor=True
```

No debe convertirse en una ruta obligatoria en todas las capas.

El objetivo es que el alcance global siga siendo fundamentalmente jerárquico y no un coste `K=log N` repetido.

---

# 12. V3 — Caché Jerárquico de Horizonte Mínimo

Esta mejora aprovecha directamente la nueva estructura de offsets.

## 12.1 Observación

Si una capa `l` solo consulta:

$$
T_{l-1}[t]
$$

y

$$
T_{l-1}[t-2^l],
$$

no necesita conservar todos los `N` estados de `T_{l-1}` para inferencia incremental.

Solo necesita un horizonte suficientemente grande para las futuras consultas de esa capa.

## 12.2 Horizonte

Definimos:

$$
H_l=\min(N_{max}, \max D_l).
$$

La cache de `T_l` puede tener longitud:

$$
C_l=H_{l+1}+1
$$

para poder servir a la siguiente etapa.

## 12.3 Ejemplo

Para:

```text
N=8192
L=12
```

una implementación diádica puede utilizar horizontes aproximados:

```text
1
2
4
8
16
32
64
128
256
512
1024
2048
```

en lugar de guardar 8192 elementos para cada capa.

La Traza continúa pudiendo almacenar la ventana completa; lo que se reduce es la memoria redundante de los estados consolidados.

## 12.4 Memoria teórica

La suma de horizontes diádicos es:

$$
\sum_{l=0}^{L-1}2^l=2^L-1.
$$

Por lo tanto la caché consolidada puede aproximarse a:

$$
O(Nd)
$$

cuando `2^L≈N`, pero con una constante mucho menor que almacenar `L·N·d`.

Esto es una reducción importante respecto a la forma directa de V2.

---

# 13. Algoritmo V3 de generación incremental

```python
def generate_step(token_t):
    # 1. Huella aislada.
    t0 = encoder(token_t)
    trace.write(t0, timestamp=current_time)

    # 2. Consolidación jerárquica.
    prev = trace.latest()

    for l in range(L):
        offsets = layer_offsets[l]
        acc = 0

        # Cada capa utiliza un conjunto pequeño de escalas.
        for p in offsets:
            if p > current_time:
                continue

            x = cache_or_trace(l, current_time - p)

            # Gate local de la Sinapsis posicional.
            q = gate_projection(x)
            alpha = sigmoid(gate_weight[l, p] @ q + gate_bias[l, p])

            # Transporte identidad + low-rank.
            y = beta[l, p] * x
            y = y + U[l] @ (scale[l, p] * (V[l].T @ x))

            acc = acc + alpha * y

        current = cell[l](acc)
        hierarchical_cache[l].append(current)
        prev = current

    # 3. Evocación.
    logits = evocator(current)
    next_id = sample_or_argmax(logits)

    # 4. Evict únicamente estados que ya no son necesarios.
    hierarchical_cache.prune_by_horizon()

    return next_id
```

La estructura conceptual sigue siendo V2:

```text
nuevo token → T0 → T1 → ... → TL → evocador
```

pero V3 limita el número de anclas y el tamaño físico de las cachés intermedias.

---

# 14. V3 — Evocador Multi-Candidato Factorizado

La V2 define:

$$
c_m=W_{evo}^{(m)}h_*+b_m.
$$

Con `M` candidatos, almacenar `M` matrices densas `d×d` puede ser innecesario.

V3 define:

$$
c_m=W_{shared}h_*
+U_e\operatorname{Diag}(s_m)V_e^Th_*
+b_m.
$$

Donde:

- `W_shared` se comparte entre candidatos.
- `U_e,V_e` se comparten.
- `s_m` es pequeño y exclusivo del candidato.

Así, cada candidato sigue siendo independiente, pero no necesita una matriz completa independiente.

## 14.1 Coste

En V2:

$$
M\cdot d^2
$$

para las proyecciones de candidatos.

En V3:

$$
O(d^2)+O(Mdr).
$$

Para `M=4` y `r<<d`, la diferencia puede ser grande.

## 14.2 Evocador y vocabulario

Se mantienen las tres agregaciones:

- Max.
- LogSumExp.
- Mean.

Pero V3 incorpora una optimización específica para `Mean`:

$$
\frac1M\sum_m\ell_{m,v}
=
\left\langle
E_v,
\frac1M\sum_m c_m
\right\rangle.
$$

Por tanto, en modo Mean, el vocabulario puede proyectarse una sola vez sobre el candidato promedio.

Esto puede reducir el coste de:

$$
O(M|V|d)
$$

a:

$$
O(|V|d).
$$

No se aplica a Max ni LogSumExp, porque sus operadores requieren conservar información independiente de los candidatos.

---

# 15. Recuperación sin atención: estrategia V3

ENGRAMA no puede recuperar mediante búsqueda dinámica basada en similitud global.

V3 no intenta convertir esa limitación en atención disfrazada.

En su lugar utiliza cuatro mecanismos compatibles:

## 15.1 Campo receptivo exponencial

Las potencias de dos garantizan acceso estructurado a distancias crecientes.

## 15.2 Ruta de identidad

La representación puede viajar sin sufrir una transformación completa en cada salto.

## 15.3 Gating local

Cada conexión puede decidir cuánto de la información de su fuente atraviesa el enlace.

El gate depende de la fuente y de la propia Sinapsis, no compara posiciones globalmente.

## 15.4 Ancla global opcional

Una única conexión directa muy lejana puede evitar que una posición antigua tenga que recorrer todas las escalas.

Estos mecanismos buscan maximizar la probabilidad de conservar y recuperar información relevante sin introducir selección dinámica.

---

# 16. Principio de preservación de identidad

Se introduce el siguiente principio de diseño V3:

> Toda ruta de consolidación debe poder transportar información sin destruirla completamente, aun cuando la arquitectura también aprenda una transformación abstractiva.

Formalmente, una Sinapsis puede representar:

$$
S(h)=\alpha\left(\beta h+F_{lowrank}(h)\right).
$$

En el régimen donde:

$$
\beta\approx1,
$$

la Sinapsis se comporta aproximadamente como una ruta de identidad.

En el régimen donde:

$$
\beta\approx0,
$$

la Sinapsis puede especializarse en transformación.

Por tanto, la misma conexión puede aprender a comportarse como:

```text
transportadora
transformadora
mixta
```

sin añadir una segunda arquitectura.

---

# 17. V3 — Gating por jerarquía

Además del gate de cada Sinapsis, V3 permite un gate escalar por escala:

$$
\rho_{l,p}=\sigma(a_{l,p}).
$$

Este gate es un parámetro aprendido estático por offset/capa, no una función de la secuencia.

La interacción total queda:

$$
T_{pos,l}[i]
=
\sum_{p\in D_l}
\rho_{l,p}
G_{l,p}(T_{l-1}[i-p]).
$$

Esto permite que el entrenamiento descubra que ciertas escalas son más o menos útiles sin hacer selección dinámica sobre posiciones.

---

# 18. Reducción formal de parámetros

## 18.1 Sinapsis V2

Aproximadamente:

$$
C^2d^2 + C^2d_g
$$

por capa de conexiones.

## 18.2 Sinapsis V3

La parte transformadora pasa a:

$$
2dr + C^2r + C^2
$$

si `U,V` son compartidas por capa y cada conexión posee `r` coeficientes de escala y un escalar de identidad.

El gating añade aproximadamente:

$$
C^2d_g.
$$

Así:

$$
P_{V3,synapse}
\approx
2dr+C^2(r+d_g+1).
$$

## 18.3 Ejemplo conceptual

Para:

```text
C=64
d=1024
r=32
dg=128
```

V2 tendría del orden de:

```text
4096 × 1024²
```

solo en matrices densas por conexión.

V3 sustituye la enorme familia de matrices por dos matrices compartidas `1024×32` y pequeños parámetros por conexión.

La reducción teórica puede ser de varios órdenes de magnitud en el componente de transformación de Sinapsis.

---

# 19. Reducción formal de operaciones

V2 utiliza aproximadamente:

$$
O(LKd^2)
$$

si cada offset ejecuta una matriz densa.

Con V3:

```text
K → 2–3 por capa
```

y:

```text
d² → dr
```

por la factorización.

Por tanto el núcleo de consolidación puede aproximarse a:

$$
O(LKdr).
$$

Con `K≈3`:

$$
O(Ldr).
$$

ignorando constantes, gates y operaciones de ensamblaje.

La mejora teórica respecto a una implementación densa de V2 puede aproximarse por:

$$
\frac{K_{V2}d^2}{K_{V3}dr}
=
\frac{K_{V2}}{K_{V3}}\frac{d}{r}.
$$

Para `d=1024`, `r=32`, `K_V2=16`, `K_V3=3`:

$$
\frac{16}{3}\times32≈171.
$$

Es decir, el núcleo de consolidación podría tener potencialmente alrededor de dos órdenes de magnitud menos operaciones aritméticas que una formulación densa de V2.

**Esto es una estimación teórica de operaciones algebraicas, no una medición de tokens/s.**

---

# 20. Paralelización V3 durante entrenamiento

Durante teacher forcing, la secuencia completa está disponible.

Por tanto:

```text
[B,N,d]
```

puede procesarse simultáneamente.

Para cada capa `l`, las operaciones son:

1. gather de `t` y `t-2^l`;
2. proyección `V_l^T` de ambas fuentes;
3. escalado por Sinapsis;
4. reconstrucción mediante `U_l`;
5. ruta identidad;
6. gate;
7. suma;
8. Cell.

No existe dependencia secuencial entre posiciones durante teacher forcing más allá de la máscara causal implícita en los offsets.

---

# 21. Paralelización V3 durante generación

La generación conserva una única dependencia secuencial externa:

```text
x_t → x_{t+1}
```

Pero dentro de cada paso:

```text
Sinapsis
Cálculo de gates
Proyecciones low-rank
Reconstrucción
Célula
Evocador
```

se pueden ejecutar vectorizados/batched.

En particular, las distintas escalas de `D_l` no deben implementarse como una cadena Python innecesariamente serial.

Deben combinarse en una operación vectorizada por escala o un kernel fusionado cuando resulte beneficioso.

---

# 22. Caché V3 y Traza: separación explícita

La arquitectura diferencia dos cosas:

## Traza

Memoria semántica de trabajo:

```text
(vector codificado, timestamp)
```

Puede cubrir toda la ventana `N_max`.

## Caché jerárquica

Memoria computacional de representaciones consolidadas necesarias para futuras operaciones.

Puede ser mucho menor gracias al horizonte específico de cada capa.

Esto evita almacenar múltiples copias completas del contexto.

---

# 23. Invarianza causal V3

## Teorema 1 — Invarianza causal

Sea:

$$
T_l[i]=Cell_l(f_l(T_{l-1}[i],T_{l-1}[i-p_1],...)).
$$

con todos los offsets `p_j≥0`.

Entonces para todo `t>i`:

$$
T_l[i]\perp x_t,x_{t+1},...
$$

Por inducción sobre las capas, todas las representaciones históricas siguen siendo invariantes mientras los tokens que las alimentan continúen presentes en la Traza.

## Consecuencia

Se puede hacer caching exacto.

La V3 no necesita recomputar posiciones antiguas.

---

# 24. Teorema 2 — Corrección de caché con horizonte mínimo

Sea `H_l` el máximo offset que una capa necesita para calcular cualquier posición futura dentro de su régimen operativo.

Entonces ningún estado anterior a `t-H_l` es consultado por esa capa en el siguiente paso.

Por tanto, almacenar únicamente:

$$
\{T_l[t-H_l],...,T_l[t]\}
$$

es suficiente para el algoritmo incremental.

Este teorema depende directamente de que V3 utilice un patrón de offsets fijo y causal.

Debe verificarse formal y experimentalmente para cada configuración de `D_l`.

---

# 25. Campo receptivo V3

Con:

$$
D_l=\{0,2^l\}
$$

se obtiene una composición de saltos binarios.

Para `L` capas, los offsets acumulables son combinaciones de:

```text
1,2,4,8,...,2^(L-1)
```

por lo que pueden representarse enteros dentro de:

$$
[0,2^L-1].
$$

Con `D_l={0,1,2^l}`, el rango máximo crece ligeramente y existe una ruta local adicional.

Ejemplo:

```text
L=8  → alcance ≈ 255+
L=12 → alcance ≈ 4095+
L=13 → alcance ≈ 8191+
```

Por tanto para `N=8192`, una configuración V3 natural es `L=13` si se desea cubrir la ventana completa mediante escalas binarias sin depender del ancla global.

Otra posibilidad es usar una capa final con ancla global.

---

# 26. Nueva regla de profundidad

V2 y V3 no deben escoger arbitrariamente `L` sin mirar `N`.

V3 recomienda:

$$
L\ge\lceil\log_2(N)\rceil
$$

si se desea cobertura binaria completa sin ancla global.

Esto convierte el número de capas en parte directa del diseño de memoria.

---

# 27. V3 — Política de offsets configurable

Se definen tres modos:

## `dense_dilated`

Compatible con V2:

```text
{0,1,2,4,...}
```

en cada capa.

## `hierarchical_dyadic`

Modo V3 recomendado:

```text
capa l → {0,1,2^l}
```

## `binary_minimal`

Modo ultra eficiente:

```text
capa l → {0,2^l}
```

Esto permite experimentar directamente con la hipótesis central de V3.

---

# 28. No se añade compresión a la Traza

La V3 mantiene deliberadamente:

```text
NO
compresión
resumen
pooling de memoria
state-space oculto
```

La Traza sigue representando explícitamente la ventana de trabajo.

Si un futuro sistema de memoria jerárquica llegara a existir, sería una arquitectura posterior y no parte de esta V3.

---

# 29. Recuperación: hipótesis experimental de V3

La V3 establece la siguiente hipótesis:

> La combinación de conectividad diádica, rutas de identidad y gates locales permitirá conservar mejor la información distante que una pila equivalente donde todas las transmisiones sean transformaciones densas sucesivas, aun utilizando menos parámetros y menos operaciones.

Esta afirmación es una **hipótesis**, no un teorema.

Debe medirse.

---

# 30. Tareas específicamente diseñadas para validar recuperación

La V3 debe evaluarse con tareas que midan recuperación y no solamente perplexity.

## 30.1 Needle in a haystack

Insertar un dato único en posiciones crecientemente lejanas y consultar al final.

Medir:

```text
accuracy vs distance
```

## 30.2 Copy exacto

```text
AAAA 739184 BBBB
```

y preguntar por `739184` después de una distancia grande.

## 30.3 Pares clave-valor

```text
A=17
B=43
C=91
...
```

consulta tardía.

## 30.4 Múltiples distractores

Muchos hechos similares con uno solo relevante.

Esto es especialmente importante porque ENGRAMA no tiene búsqueda dinámica.

## 30.5 Dependencia temporal

Medir tareas donde el orden de eventos sea decisivo.

## 30.6 Recuperación incremental

Comparar:

```text
full recomputation
```

contra:

```text
incremental cache
```

---

# 31. Prueba de preservación de señal

Crear una prueba matemática/numérica aislada para una Sinapsis.

Entrada:

$$
h\neq0.
$$

Configurar:

$$
\beta=1,
\alpha≈1,
U\operatorname{Diag}(s)V^Th≈0.
$$

y comprobar que:

$$
S(h)≈h.
$$

Después introducir perturbaciones y medir la distancia:

$$
\|S(h)-h\|.
$$

La prueba no demuestra que el entrenamiento aprenderá `β≈1`, pero demuestra que la arquitectura **puede representar** una ruta de identidad.

---

# 32. Estabilidad de entrenamiento V3

La factorización low-rank puede producir escalas inestables.

Se recomienda inicializar:

```text
s_ab ≈ 0
beta_ab ≈ valor positivo cercano a identidad
```

para comenzar cerca de una ruta estable.

Posteriormente el entrenamiento puede aprender la cantidad de transformación requerida.

Esto transforma la inicialización desde:

```text
red totalmente aleatoria
```

hacia:

```text
transporte estable + aprendizaje de desviación
```

La decisión debe ser configurable y debe probarse contra inicialización convencional.

---

# 33. Normalización V3

La V3 conserva LayerNorm dentro de la Célula.

No añade normalización sobre posiciones ni softmax sobre las anclas.

Opcionalmente puede utilizarse una normalización de magnitud por vector antes del gate local si pruebas experimentales muestran saturación del sigmoid.

Debe ser una transformación por posición, nunca una operación que compare múltiples posiciones.

---

# 34. V3 — Preservación semántica de Sinapsis

Aunque la matriz `W_{a→b}` deje de almacenarse como una matriz independiente completa, la Sinapsis sigue existiendo conceptualmente como una entidad:

```text
(source cell, destination cell, gate, identity coefficient, low-rank coefficients)
```

Por tanto:

```text
64 × 64 = 4096 Sinapsis
```

sigue siendo cierto.

La diferencia es que sus matrices se representan mediante un espacio de bases compartidas.

Esto conserva la topología original y reduce el coste.

---

# 35. V3 — Bases compartidas de Sinapsis

Para cada capa existe una pareja:

```text
U_l
V_l
```

que constituye el espacio de transformaciones.

Cada Sinapsis define su dirección dentro de ese espacio mediante:

```text
s_{a→b,l}
```

Esto puede interpretarse como:

```text
U,V = lenguaje general de transformaciones
s_ab = especialización de la conexión
```

Esta es la misma filosofía de pesos compartidos de ENGRAMA llevada a una parametrización mucho más eficiente.

---

# 36. Extensión opcional: bases múltiples por grupos

Si `r` resulta insuficiente, la arquitectura puede particionar las Células en grupos.

Cada grupo posee un par `U_g,V_g`.

Esto permite:

```text
más expresividad
```

sin volver a:

```text
C² matrices densas
```

Debe considerarse una extensión opcional y no parte obligatoria del V3 mínimo.

---

# 37. V3 — Evocador y compartir pesos

El embedding `E` continúa siendo:

$$
E\in R^{|V|×d}.
$$

Los candidatos del evocador viven en el mismo espacio `d`.

V3 recomienda además que, cuando se utilice `Mean`, la representación final se agregue antes del producto contra `E`:

$$
\bar c=\frac1M\sum_m c_m
$$

$$
\ell_v=\frac{\langle\bar c,E_v\rangle}{\sqrt d}.
$$

Esto hace que `M` no multiplique el coste del vocabulario en ese modo.

---

# 38. V3 — Perfil recomendado

Configuración de investigación inicial para `N≈8192`:

```text
N_max              = 8192
L                  = 13
C                  = 32 o 64
r                  = 16, 32 o 64
D_l                = {0,1,2^l}
cell_ff_multiplier  = 4
M                  = 1 o 4
evocator_agg       = mean para benchmark rápido
trace_dtype        = fp16/bf16 según backend
cache_dtype        = fp16/bf16
```

Estos valores no son afirmaciones de optimalidad.

Son puntos de partida experimentales.

---

# 39. Comparación conceptual V1 / V2 / V3

| Propiedad | V1 | V2 | V3 |
|---|---|---|---|
| Huella aislada | Sí | Sí | Sí |
| Traza explícita | Sí | Sí | Sí |
| Traza circular | No/variante | Sí | Sí |
| Timestamp | No/limitado | Sí | Sí |
| Offset relativo | Sí | Sí | Sí |
| Consolidación causal | Sí | Sí | Sí |
| Gating | Sí | Sí | Sí |
| Pesos `d×d` por Sinapsis | Sí | Conceptual | No, factorizados |
| Anclas por capa | muchas | `~log N` | `~2–3` |
| Caché incremental | No | Sí | Sí |
| Caché por horizonte mínimo | No | No | Sí |
| Ruta identidad en Sinapsis | implícita | opcional | explícita |
| Evocador multi-candidato | Sí | Sí | Sí |
| Evocador factororizado | No | No | Sí |
| Atención | No | No | No |
| Búsqueda dinámica | No | No | No |

---

# 40. Complejidad teórica V3

## 40.1 Entrenamiento

V3 busca:

$$
O(BNLKdr)
$$

para el núcleo de consolidación, con `K≈2–3`.

En comparación, la V2 con matrices densas puede escribirse aproximadamente como:

$$
O(BNLKd^2).
$$

La reducción proviene de dos cambios independientes:

```text
K: log N → constante
 d² → d·r
```

## 40.2 Inferencia

Para cada token:

$$
O(LKdr)
$$

con `K≈2–3`.

Si el Evocador usa Mean:

$$
O(|V|d)
$$

para el vocabulario después de combinar candidatos.

## 40.3 Memoria

Traza:

$$
O(Nd)
$$

Caché consolidada:

$$
O(Nd)
$$

cuando la jerarquía de horizontes cubre el contexto mediante potencias de dos.

Esto busca eliminar el factor `L` que aparece al almacenar una ventana completa para cada capa.

---

# 41. Qué no promete V3

V3 **no demuestra** que:

- sea mejor que Transformer en perplexity;
- sea mejor que Transformer en razonamiento;
- recupere arbitrariamente cualquier dato con la misma fiabilidad que atención;
- tenga mejores tokens/s en cualquier hardware;
- necesite siempre menos memoria total después de incluir embeddings y vocabulario;
- las aproximaciones low-rank sean suficientes para todas las escalas.

V3 formula mecanismos destinados a mejorar esas áreas, pero requieren validación.

---

# 42. Riesgo principal de V3

La mayor hipótesis que debe comprobarse es:

> ¿Puede una conectividad diádica escasa transmitir suficiente información de contexto largo sin que la representación pierda capacidad frente a una V2 con más anclas por capa?

La segunda hipótesis crítica es:

> ¿Puede una representación low-rank por Sinapsis conservar suficiente diversidad de transformaciones entre Células?

La tercera es:

> ¿La ruta de identidad mejora recuperación sin provocar que el modelo aprenda simplemente a copiar y reduzca la capacidad de abstracción?

Estas son las preguntas experimentales centrales de V3.

---

# 43. Suite experimental V3 obligatoria

Toda implementación de V3 debe comparar como mínimo:

```text
V2-densa
V2-factorizada
V3-densa
V3-factorizada
```

y:

```text
V3 dense_dilated
V3 hierarchical_dyadic
V3 binary_minimal
```

Medir:

- parámetros;
- FLOPs;
- memoria;
- tokens/s;
- ms/token;
- perplexity;
- exact retrieval;
- long-context accuracy;
- gradient norm;
- estabilidad de entrenamiento.

---

# 44. Ablaciones obligatorias

## A. Sin identidad

Eliminar `βh`.

## B. Sin low-rank

Usar solo identidad.

## C. Sin gating

Ver cuánto aporta el gate.

## D. Dense offsets

Comparar V2.

## E. Dyadic offsets

Propuesta principal V3.

## F. Sin ancla global

## G. Con ancla global

## H. Cell compartida

## I. Cell independiente

## J. Evocador `M=1`

## K. Evocador `M=4`

---

# 45. Prueba de equivalencia incremental

Para una misma secuencia:

```text
x0 x1 x2 ... xt
```

calcular:

```text
Forward completo
```

y:

```text
Step x0
Step x1
Step x2
...
Step xt
```

Debe cumplirse aproximadamente:

$$
T_l^{full}[i]\approx T_l^{incremental}[i]
$$

para las posiciones válidas.

El error debe reportarse por capa:

```text
max_abs_error
mean_abs_error
relative_error
```

---

# 46. Prueba de recuperación exacta

Construir secuencias artificiales con una señal fácilmente identificable.

Ejemplo:

```text
[random tokens]
SPECIAL_TOKEN_A
[random × d]
query
```

Incrementar `d` y comprobar el punto donde la recuperación comienza a fallar.

Repetir para:

```text
V1
V2
V3
```

La curva de precisión frente a distancia será uno de los principales indicadores de la utilidad real de V3.

---

# 47. Prueba de resistencia a interferencia

Introducir múltiples patrones similares:

```text
KEY=ALPHA VALUE=17
KEY=BETA  VALUE=92
KEY=GAMMA VALUE=44
...
```

y pedir uno específico.

Esto mide si el sistema conserva identidad de representaciones frente a interferencia.

---

# 48. Prueba de parámetro por capacidad

Comparar modelos con aproximadamente el mismo número total de parámetros:

```text
V2 grande
V3 pequeño
```

y también:

```text
V2 pequeño
V3 pequeño
```

La métrica crítica es la calidad por parámetro.

---

# 49. Prueba de eficiencia por calidad

Crear una curva:

```text
calidad
  │
  │       V3
  │      /
  │     /
  │ V2 /
  │   /
  └──────────────
       FLOPs
```

No se debe declarar superioridad si V3 únicamente reduce FLOPs pero pierde demasiada calidad.

---

# 50. V3 — Interpretabilidad adicional

La separación de rutas permite inspeccionar:

- gate `α`;
- peso identidad `β`;
- coeficientes low-rank `s`;
- escala de offset;
- trayectoria binaria;
- capa donde llega información distante.

Esto proporciona una nueva forma de análisis:

```text
¿qué parte de la información viajó por identidad?
¿qué parte fue transformada?
¿qué escalas utilizaron los gates?
```

La arquitectura sigue siendo inspeccionable porque las representaciones permanecen asociadas a posiciones estables.

---

# 51. Interpretación de la recuperación en V3

V3 no intenta contestar:

> “¿A qué posición debo atender?”

En su lugar intenta responder:

> “¿Cómo hago que la información de cualquier posición potencialmente relevante sobreviva por una ruta estructurada de bajo coste hasta llegar al presente?”

Esta diferencia es fundamental.

La recuperación es **estructural**, no basada en búsqueda dinámica.

---

# 52. Filosofía V3

La arquitectura continúa siguiendo una visión de memoria biológica simplificada:

```text
experiencia
   ↓
huella
   ↓
almacenamiento
   ↓
consolidación progresiva
   ↓
evocación
```

La V3 introduce además la idea:

```text
consolidación ≠ destruir identidad
```

Una huella puede:

```text
mantenerse
transformarse
ambas cosas simultáneamente
```

---

# 53. Por qué V3 no utiliza atención escondida

No se introduce ningún mecanismo donde una representación final consulte un conjunto de posiciones mediante similitud.

El sistema puede presentar gates que dependan del contenido de la fuente, pero cada gate pertenece a una conexión estructural predeterminada.

Por tanto:

```text
V3 gate:
fuente → gate de enlace fijo
```

no:

```text
query → similitud con N posiciones → selección
```

Esto preserva la identidad arquitectónica de ENGRAMA.

---

# 54. V3 — Modo compatible con V2

La librería deberá poder activar:

```python
version="v2"
```

y:

```python
version="v3"
```

Dentro de V3 deberá poder elegirse:

```python
synapse_mode="dense"
synapse_mode="factorized"

offset_mode="dense_dilated"
offset_mode="hierarchical_dyadic"
offset_mode="binary_minimal"

cache_mode="full"
cache_mode="hierarchical"

cell_mode="independent"
cell_mode="shared_core"
```

Esto permitirá aislar el beneficio de cada mejora.

---

# 55. Configuración matemática V3 recomendada

Una configuración podría representarse como:

```python
EngramaV3Config(
    vocab_size=...
    d_model=...
    d_gate=...
    num_cells=...
    num_encoder_layers=...
    num_consolidation_layers=...
    context_length=...
    offset_mode="hierarchical_dyadic"
    synapse_mode="factorized"
    synapse_rank=32
    identity_transport=True
    shared_cell_core=True
    candidate_count=1
    candidate_aggregation="mean"
    global_anchor=False
    hierarchical_cache=True
)
```

---

# 56. Estado de investigación V3

La V3 debe considerarse una arquitectura experimental hasta que se realicen al menos:

1. Pruebas de equivalencia incremental.
2. Benchmarks de recuperación.
3. Benchmarks de eficiencia.
4. Entrenamientos comparativos.
5. Ablaciones.
6. Análisis de estabilidad.

No utilizar números de rendimiento proyectados como resultados publicados.

---

# 57. Resultado esperado de V3

Si las hipótesis funcionan, V3 debería conseguir simultáneamente:

```text
menos parámetros
        +
menos operaciones
        +
menos memoria de caché
        +
mejor transporte de información distante
        +
misma ausencia de atención
        +
misma separación huella/Traza/consolidación/evocación
```

La ganancia no procede de introducir una nueva forma de atención.

Procede de hacer la arquitectura original más eficiente estructuralmente.

---

# 58. Comparación de los costes dominantes

## V2 conceptual

Sinapsis:

$$
C^2d^2
$$

Consolidación:

$$
L\log(N)d^2
$$

Cache:

$$
LNd
$$

## V3 propuesta

Sinapsis:

$$
2dr+C^2r+C^2d_g
$$

Consolidación:

$$
L\cdot O(1)\cdot dr
$$

Cache:

$$
O(Nd)
$$

El cuello de botella puede desplazarse al embedding/evocador y por ello deben medirse todos los componentes, no solamente la consolidación.

---

# 59. Limitaciones V3

V3 introduce nuevas limitaciones:

1. La factorización low-rank puede reducir capacidad de representación por conexión.
2. La distribución de escalas entre capas puede dificultar ciertas interacciones que V2 obtiene inmediatamente mediante múltiples anclas en cada capa.
3. Las rutas diádicas pueden generar cuellos de botella de señal.
4. El modo Mean del Evocador puede reducir diversidad respecto a Max/LogSumExp.
5. La caché de horizonte mínimo requiere una prueba formal por configuración de offsets.
6. Compartir la Célula puede producir subespecialización insuficiente si la modulación por Célula es demasiado pequeña.

Por tanto la V3 no debe venderse como automáticamente superior.

Debe tratarse como una hipótesis arquitectónica optimizada que necesita experimentación rigurosa.

---

# 60. Criterio de éxito científico

La V3 será considerada exitosa si demuestra al menos una de estas condiciones sin perder completamente las restantes:

### Condición A

Misma calidad aproximada que V2 con significativamente menos parámetros.

### Condición B

Misma calidad aproximada que V2 con significativamente menos FLOPs.

### Condición C

Mejor recuperación de contexto largo con igual o menor coste.

### Condición D

Mayor calidad por parámetro y por operación.

### Condición E

Inferencia significativamente más eficiente manteniendo calidad suficiente.

La meta ideal es conseguir varias simultáneamente.

---

# 61. Esquema definitivo V3

```text
                    ┌──────────────────────┐
                    │       TOKEN          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ EMBEDDING            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ HUELLA AISLADA       │
                    │ Cell + Synapses      │
                    │ paralela por token   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ TRAZA CIRCULAR       │
                    │ vector + timestamp   │
                    │ FIFO                 │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
        escala local                    escala diádica
                │                             │
                └──────────────┬──────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ SINAPSIS V3              │
                 │ gate local               │
                 │ identidad                │
                 │ low-rank                 │
                 └─────────────┬────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ CÉLULA V3                │
                 │ núcleo compartido        │
                 │ modulación por Célula   │
                 └─────────────┬────────────┘
                               │
                               ▼
                   T1 → T2 → ... → TL
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ CACHÉ JERÁRQUICO         │
                 │ horizonte mínimo         │
                 └─────────────┬────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ EVOCADOR V3              │
                 │ candidatos factorized   │
                 └─────────────┬────────────┘
                               │
                               ▼
                         siguiente token
```

---

# 62. Checklist V3 de fidelidad arquitectónica

1. ✅ No atención.
2. ✅ No `QK^T`.
3. ✅ No softmax sobre secuencia.
4. ✅ Codificación aislada.
5. ✅ Traza explícita.
6. ✅ Traza no transforma.
7. ✅ FIFO circular.
8. ✅ Timestamp absoluto para inspección.
9. ✅ Consolidación causal.
10. ✅ Offsets relativos.
11. ✅ Sinapsis individuales.
12. ✅ Gate local por Sinapsis.
13. ✅ Identidad dentro de la Sinapsis.
14. ✅ Transformación low-rank compartida.
15. ✅ Células preservadas conceptualmente.
16. ✅ Evocador sin mezcla de posiciones.
17. ✅ Multi-candidato conservado.
18. ✅ Caché causal.
19. ✅ Caché de horizonte mínimo.
20. ✅ Entrenamiento paralelizable.
21. ✅ Generación autoregresiva.
22. ✅ Sin compresión de la Traza.
23. ✅ Sin estado oculto recurrente sustituyendo la Traza.
24. ✅ Sin selección dinámica global de posiciones.

---

# 63. Conclusión

ENGRAMA V3 intenta demostrar que la arquitectura puede evolucionar sin abandonar su identidad.

La estrategia completa es:

```text
V1
  huella aislada
  + Traza
  + consolidación
  + evocación

V2
  + offsets relativos
  + Traza circular
  + caché causal

V3
  + Sinapsis low-rank
  + transporte identidad
  + consolidación diádica por capa
  + caché de horizonte mínimo
  + Célula con núcleo compartido
  + Evocador factorized
```

La V3 no intenta ganar inteligencia introduciendo una operación más sofisticada.

Intenta ganar eficiencia y recuperación mediante una mejor organización del mismo principio fundamental:

> una experiencia deja una huella aislada, la huella se conserva explícitamente, la consolidación construye contexto a través de conexiones causales estructuradas y la evocación transforma el contexto final en el siguiente símbolo.

La hipótesis central V3 es que **la arquitectura puede conservar capacidad de contexto largo si las rutas de información se diseñan como una red jerárquica de transporte con identidad y transformación de bajo rango**, evitando tanto la explosión paramétrica de las Sinapsis densas como el coste repetido de consultar todos los offsets en cada capa.

La hipótesis debe demostrarse experimentalmente.

Pero si funciona, V3 puede conservar la característica más importante de ENGRAMA —memoria explícita, consolidación causal e inferencia cacheable sin atención— mientras reduce simultáneamente parámetros, operaciones y memoria computacional.

---

# 64. Referencia base

Esta V3 deriva directamente del documento de ENGRAMA V1+V2 proporcionado por el autor, en particular de las definiciones de Célula/Sinapsis, Traza circular, consolidación por offsets relativos, caché causal, evocación multi-candidato y análisis de complejidad.

La V2 establece explícitamente que la codificación es independiente por posición, que la Traza solo almacena, que la consolidación usa pesos relativos y que la invarianza causal permite cachear todo el pasado. La V3 conserva esas propiedades y modifica únicamente la parametrización y la organización jerárquica de las conexiones para buscar menor coste y mejor preservación de señal.

**Estado final:** V3 teórica propuesta. Requiere implementación, pruebas de equivalencia, benchmarks y validación científica antes de considerarse una versión experimental confirmada.

---

# Fin de ENGRAMA V3
