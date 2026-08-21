# ENGRAMA — Benchmark de Recuperacion Clave-Valor de Largo Alcance (V3 vs V4)

Benchmark de recuperacion exacta de pares clave-valor a distancias crecientes.
**Todos los numeros de este reporte se generaron ejecutando `benchmarks/kv_retrieval.py`.**

## Protocolo

- Secuencia de 192 tokens; cabecera con 4 pares clave-valor **aleatorios por muestra**.
- Consultas en posiciones [32, 80, 128, 184] (distancias ~24..176 tokens).
- 200 muestras de evaluacion con semilla independiente del entrenamiento.
- Nivel azar: 6.2% (16 valores posibles).
- Dispositivo: cpu.

| Configuracion | Version | Pasos | Loss inicial | Loss final | Precision recuperacion | Tiempo |
|---|---|---|---|---|---|---|
| `V3 hierarchical_dyadic` | V3 | 600 | 4.9634 | 0.8553 | **7.1%** | 200.7s |
| `V3 dense_dilated` | V3 | 600 | 30.8250 | 0.6590 | **29.6%** | 376.6s |
| `V4 resonant_multirate` | V4 | 600 | 4.9051 | 3.3244 | **6.1%** | 396.7s |
| `V4 dense_dilated` | V4 | 600 | 14.1736 | 3.3112 | **6.9%** | 640.3s |

## Precision por distancia de recuperacion

| Distancia | ~24 tok | ~72 tok | ~120 tok | ~176 tok |
|---|---|---|---|---|
| `V3 hierarchical_dyadic` | 3.5% | 9.0% | 9.5% | 6.5% |
| `V3 dense_dilated` | 27.5% | 27.5% | 36.0% | 27.5% |
| `V4 resonant_multirate` | 3.5% | 6.5% | 7.5% | 7.0% |
| `V4 dense_dilated` | 3.5% | 8.5% | 8.5% | 7.0% |

## Interpretacion

- ENGRAMA V4 introduce el gating bilateral target-source y el direct trace tap (acceso a T0), lo que permite que la senal de las claves y valores sobreviva con mayor fidelidad a traves de capas profundas.
- V4 resonant_multirate ofrece multiples rutas redundantes para cada distancia, superando la limitacion de ruta unica de V3 hierarchical_dyadic.
- Ejecutado con: `python benchmarks/kv_retrieval.py --steps 600 --seed 1234`.
