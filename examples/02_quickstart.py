"""Ejemplo 02 — Modo rápido completo: de texto a modelo entrenado.

Entrena un modelo 'tiny' sobre un corpus de juguete, genera texto,
evalúa y guarda/carga el bundle. Ejecuta:  python examples/02_quickstart.py
"""

import os
import tempfile

import engrama

CORPUS = (
    "el gato duerme en la casa. la casa es grande y azul.\n"
    "el perro corre por el parque. el parque tiene arboles verdes.\n"
    "la nina lee un libro. el libro cuenta historias del mar.\n"
    "el sol brilla sobre el campo. el campo esta lleno de flores.\n"
) * 8

with tempfile.TemporaryDirectory() as tmp:
    corpus_path = os.path.join(tmp, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(CORPUS)

    # 1) Entrenar en una llamada
    run = engrama.quickstart(corpus_path, size="tiny", epochs=5)

    # 2) Resumen + generación
    print(run.summary())
    print("Generación:", repr(run.generate("el gato", max_new_tokens=20)))

    # 3) Evaluar sobre texto nuevo
    loss = run.evaluate("el gato duerme en la casa.")
    print(f"Loss de evaluación: {loss:.4f}")

    # 4) Guardar y recargar
    save_dir = os.path.join(tmp, "mi_modelo")
    run.save(save_dir)
    run2 = engrama.load_quick(save_dir)
    assert run2.config.d_model == run.config.d_model
    print("Bundle guardado y recargado OK")

print("Ejemplo 02 OK")
