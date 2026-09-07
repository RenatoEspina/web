# Terraria ES: corpus inicial para SFT

96 ejemplos de entrenamiento y 16 de evaluación, en español, a partir de
28 artículos de [Official Terraria Wiki](https://terraria.wiki.gg/).
Es una **base de datos JSON/JSONL curada**, no un volcado de toda la wiki ni
un modelo experto. Las respuestas son sintéticas y parafraseadas; no son
respuestas oficiales escritas por los autores de la wiki.

## Archivos

- `curated.json`: base editable, con preguntas, respuestas, fuentes y partición.
- `../../examples/terraria-training.jsonl`: SFT compatible con la GUI; aparece
  automáticamente en **Dataset → Ejemplo · terraria-training.jsonl**.
- `evaluation.jsonl`: preguntas reservadas, referencias y comprobaciones básicas.
  Está fuera de las carpetas seleccionables de SFT. No lo subas para entrenar.
- `build.py`: reconstruye ambos JSONL sin red. `--check` verifica reproducibilidad.

Cada ejemplo conserva URL, historial de contribuciones, fecha de consulta,
licencia y atribución. La consulta directa de la wiki respondió HTTP 403;
se utilizaron extractos indexados de las páginas oficiales accesibles en el
buscador, sin eludir el bloqueo. **2026-09-07 es la fecha de consulta del índice,
no una revisión fija de la wiki.** Esto limita su actualidad y reproducibilidad
externa. `curated.json` sí permite reproducir exactamente los archivos entregados.

## Alcance y calidad

Terraria vanilla para PC moderno y mundos normales, salvo indicación expresa:
inicio, vida/maná, fabricación, NPC, casas, movilidad y progresión de jefes.
No incluye mods, una tabla completa de objetos/recetas, todas las semillas
especiales ni todas las diferencias entre parches/plataformas.

La separación es por artículo/tema: Suspicious Looking Eye, Queen Bee,
Duke Fishron y The Aether quedan exclusivamente en evaluación. Esto comprueba
comportamiento en temas no entrenados y posible regresión del conocimiento base;
**no demuestra que el LoRA haya memorizado nuevos hechos de esos temas**.
Con 16 casos, el resultado es exploratorio, no una métrica robusta.

`contains` en el evaluador es un smoke test léxico: puede aprobar una respuesta
contradictoria o suspender una paráfrasis correcta. Revisa las respuestas completas
contra `reference_answer` y su fuente. No ajustes repetidamente hiperparámetros
mirando este conjunto final; para experimentar, crea otro conjunto de desarrollo.

Para un asistente que consulte toda la wiki y cite información actualizada,
usa RAG con contenido obtenido y reutilizado conforme a sus condiciones.
Este SFT pequeño sirve para experimentar con el estilo y el dominio; no garantiza
precisión factual ni sustituye una base de conocimiento consultable.

## Licencia y atribución

Corpus derivado de los colaboradores de Official Terraria Wiki (wiki.gg).
Terraria y sus materiales pertenecen a Re-Logic. Cambios realizados: selección
de hechos, traducción/paráfrasis al español, redacción de preguntas y separación
de evaluación. No se incluyen imágenes, audio ni diálogos del juego.

Los textos que la wiki puede licenciar están bajo
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), según su
[página de copyright](https://terraria.wiki.gg/wiki/Terraria_Wiki%3ACopyrights).
Este corpus se distribuye bajo esa misma licencia: conserva atribución, enlaces,
indicación de cambios, uso no comercial y compartir igual. **No lo trates como
datos MIT aunque el código del proyecto tenga otra licencia.** Antes de distribuir
un modelo entrenado o usarlo comercialmente, revisa las condiciones de la wiki y
del modelo base; este repositorio no garantiza que ese uso esté autorizado.

## Entrenar

Consulta [la guía paso a paso](../../../docs/TERRARIA_FINE_TUNING.md).

Reconstrucción opcional, desde la raíz del repositorio:

```bash
python trainer/corpora/terraria/build.py
python trainer/corpora/terraria/build.py --check
python trainer/validate_dataset.py trainer/examples/terraria-training.jsonl
```
