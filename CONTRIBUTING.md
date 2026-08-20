# Contribuyendo a ENGRAMA

¡Gracias por tu interés en contribuir a ENGRAMA! Este proyecto es una
investigación activa sobre arquitecturas autorregresivas **sin atención**, y
cada contribución — código, tests, benchmarks, documentación o reportes de
bugs — es bienvenida.

## Tabla de contenidos

- [Código de conducta](#código-de-conducta)
- [Configuración del entorno de desarrollo](#configuración-del-entorno-de-desarrollo)
- [Cómo ejecutar los tests](#cómo-ejecutar-los-tests)
- [Reportar bugs](#reportar-bugs)
- [Proponer cambios](#proponer-cambios)
- [Estándares de código](#estándares-de-código)
- [Política de honestidad científica](#política-de-honestidad-científica)
- [Licencia](#licencia)

## Código de conducta

Sé respetuoso, constructivo y basado en evidencia. Las discusiones sobre la
arquitectura deben apoyarse en la especificación (`ENGRAMA-V3-Teorica.md`),
ejecuciones reales o matemática verificable — no en opiniones sin sustento.

## Configuración del entorno de desarrollo

```bash
git clone https://github.com/bueormnew/engrama.git
cd engrama

# Entorno virtual (Python >= 3.9)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# PyTorch CPU (suficiente para desarrollo y tests; CUDA opcional)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Librería en modo editable + dependencias de desarrollo
pip install -e ".[dev]"
```

## Cómo ejecutar los tests

La suite completa son **68 tests** (~14 s en CPU):

```bash
cd tests && python -m unittest discover -q
# Ran 68 tests in ~14s — OK
```

Reglas:

- Todo cambio de código **debe** mantener los 68 tests en verde.
- Todo cambio de comportamiento **debe** añadir tests que lo cubran
  (`tests/` usa `unittest`; los tests de arquitectura canónica viven en
  `tests/architecture/`).
- Los tests deben ejecutarse en CPU (sin requerir CUDA).

## Reportar bugs

Abre un issue en <https://github.com/bueormnew/engrama/issues> incluyendo:

1. Versión de la librería (`engrama info`), versión de Python y de PyTorch.
2. Comando o snippet mínimo que reproduce el problema.
3. Salida completa (incluyendo warnings).
4. Comportamiento esperado vs. obtenido.

Si el bug afecta a la *invarianza causal* o a los *números publicados* en la
documentación, márcalo como de alta prioridad: la política del proyecto es que
**ningún número publicado sea proyectado**.

## Proponer cambios

1. Crea una rama con nombre descriptivo: `fix/descripcion` o `feat/descripcion`.
2. Haz commits pequeños y con mensajes claros (inglés o español, consistente
   con el archivo que toques).
3. Añade/actualiza tests y verifica la suite completa.
4. Si cambias la API pública (`engrama/__init__.py`), actualiza `__all__`,
   el docstring del módulo y la documentación afectada (README, `docs/`).
5. Si cambias números reportados (parámetros, errores, tiempos), regenéralos
   con una ejecución real y actualiza `docs/VERIFICACION.md`.
6. Abre un Pull Request describiendo qué cambia y por qué, y qué verificaste.

## Estándares de código

- Python 3.9+, tipado con `typing` en las firmas públicas.
- Docstrings de módulo y de clase/método en todos los objetos públicos,
  citando las secciones de la especificación V3 cuando aplique
  (p. ej. `V3 spec section 24`).
- Sin dependencias nuevas fuera de PyTorch para el núcleo de la librería.
- El código debe seguir siendo **puro PyTorch** y **sin atención**:
  prohibidos `QK^T`, matrices de afinidad N×N y softmax sobre la dimensión
  temporal. El único softmax permitido es el de la distribución final sobre
  el vocabulario.
- Formato: 88 columnas, nombres descriptivos. No hay formateador obligatorio,
  pero mantén el estilo del archivo que edites.

## Política de honestidad científica

Este repositorio sigue la *política de honestidad* de la especificación V3
(sección 56):

- Todo número publicado en README, docs o benchmarks debe provenir de una
  **ejecución real** reproducible (semilla, comando y entorno documentados).
- Está **prohibido** presentar números proyectados o estimados como medidos.
- Los benchmarks comparativos deben reportar también los resultados que no
  favorecen a la hipótesis (ver `benchmarks/KV_RETRIEVAL_REPORT.md`).

## Licencia

Al contribuir aceptas que tu código se distribuya bajo
**GNU Affero General Public License v3.0** (ver `LICENSE`).
Autor: Gerson Fabian Buenahora Ormaza (BUEORM), 2026.
