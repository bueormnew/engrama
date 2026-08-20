# ENGRAMA V3 — Benchmark de recuperacion clave-valor de largo alcance

Benchmark de las tareas 30.3/43/47 de la especificacion V3.
**Todos los numeros de este reporte se generaron ejecutando `benchmarks/kv_retrieval.py`; ningun valor es proyectado.**

## Protocolo

- Secuencia de 192 tokens; cabecera con 4 pares clave-valor **aleatorios por muestra** (imposible memorizarlos).
- Consultas en posiciones [32, 80, 128, 184] (distancias ~24..176 tokens).
- 200 muestras de evaluacion con semilla independiente del entrenamiento.
- Nivel azar: 6.2% (16 valores posibles).
- Dispositivo: cpu.

| Configuracion | Pasos | Loss inicial | Loss final | Precision recuperacion | Tiempo |
|---|---|---|---|---|---|
| V3 `hierarchical_dyadic` (383,958 params) | 600 | 4.9478 | 0.8544 | **7.4%** | 190.9s |
| V3 `hierarchical_dyadic + ancla` (384,568 params) | 600 | 4.9711 | 0.8522 | **7.1%** | 185.2s |
| V3 `dense_dilated` (413,848 params) | 600 | 34.5609 | 0.6907 | **27.5%** | 357.5s |

## Precision por distancia de recuperacion

| Distancia | ~24 tok | ~72 tok | ~120 tok | ~176 tok |
|---|---|---|---|---|
| V3 `hierarchical_dyadic` | 4.0% | 8.0% | 9.0% | 8.5% |
| V3 `hierarchical_dyadic + ancla` | 4.5% | 7.0% | 9.5% | 7.5% |
| V3 `dense_dilated` | 27.0% | 36.0% | 17.5% | 29.5% |

## Interpretacion honesta

- La precision por encima del azar indica que la informacion del encabezado sobrevive el transporte hasta la consulta (hipotesis V3, seccion 29); la comparacion entre politicas de offsets mide el riesgo principal de V3 (seccion 42).
- Este benchmark mide una tarea sintetica; no demuestra equivalencia con atencion en lenguaje natural (V3 seccion 41).
- Ejecutado con: `python benchmarks/kv_retrieval.py --steps 600 --seed 1234` en este entorno (CPU).
