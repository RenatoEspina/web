# Dolly EN: corpus de control y de replay

Adaptación de [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
al formato JSONL conversacional del proyecto. A diferencia de los corpus
`terraria` y `unidades`, este no enseña un dominio: existe para **medir el
pipeline** y para **conservar capacidad general** durante un fine-tuning de
dominio.

## Por qué este dataset

- **15 011 pares instrucción/respuesta escritos por humanos**, no generados por
  otro LLM. Se prohibió explícitamente el uso de IA generativa al redactarlos,
  así que no arrastra el estilo de un modelo maestro.
- **Documentado**: tiene dataset card pública, categorías derivadas del paper de
  InstructGPT y limitaciones declaradas por sus autores.
- **CC BY-SA 3.0**, que permite uso comercial. El corpus de Terraria es
  CC BY-NC-SA 4.0 y el de unidades es sintético: este es el único de los tres
  que se puede redistribuir sin la cláusula no comercial.
- **Entra en 6 GB de VRAM**: tras el filtro de longitud, ningún ejemplo supera
  los 1024 tokens del `--max-length` por defecto (mediana 148, p95 405).

## Fuente y reproducibilidad

`build.py` descarga una sola vez el archivo de una revisión inmutable y lo
verifica por SHA-256 antes de usarlo:

```text
repositorio  databricks/databricks-dolly-15k
revisión     bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a
archivo      databricks-dolly-15k.jsonl
sha256       2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec
```

La copia queda en `.cache/` (gitignorada). A partir de ahí el build es offline y
determinista: mismo seed, mismas particiones, mismos hashes en `manifest.json`.
Las GUI invocan `build.py --if-cached`, que no hace nada mientras la fuente no
esté descargada, para que abrir la GUI nunca dependa de la red.

## Adaptación aplicada

El formato original (`instruction`, `context`, `response`, `category`) se
convierte a `messages` sin reescribir el texto humano:

- `system`: un prompt general fijo, compartido por las tres particiones;
- `user`: la instrucción, y si hay contexto, la instrucción seguida de `---` y el
  pasaje;
- `assistant`: la respuesta tal cual.

Filtros, todos registrados en `manifest.json -> dropped`:

| filtro             | qué descarta                               | ejemplos |
| ------------------ | ------------------------------------------ | -------- |
| `too_long`         | el ejemplo completo supera 2200 caracteres | 1107     |
| `duplicate_prompt` | el prompt normalizado ya apareció          | 204      |
| `empty`            | prompt o respuesta en blanco               | 0        |

Quedan **13 700** ejemplos. Una respuesta de un solo carácter (`"7"`, `"F"`) es
legítima y se conserva; el umbral de longitud solo se aplica a los fragmentos
`contains` de la evaluación, que es donde un fragmento corto haría daño.

El tope de 2200 caracteres está calibrado contra el tokenizer de
`Qwen/Qwen3.5-0.8B`: el ejemplo más largo que lo pasa mide 1014 tokens. Es una
heurística de biblioteca estándar; la comprobación dura sigue siendo `train.py`,
que aborta si una respuesta `assistant` se trunca.

## Particiones

El split es por ejemplo, con seed fijo, estratificado por categoría y sin fuga:
los prompts se deduplican con la misma normalización que usa
`training_quality.count_prompt_overlaps`, así que el solape entre particiones es
cero por construcción.

| perfil   | train  | validation | evaluation |
| -------- | ------ | ---------- | ---------- |
| `laptop` | 3000   | 400        | 250        |
| `full`   | 12 794 | 600        | 300        |

`laptop` es el que produce la GUI por defecto: con batch 1 y gradient
accumulation 8 son ~375 pasos de optimizador por época.

La evaluación se reserva **antes** que el resto, porque solo puede salir del
subconjunto con respuesta verificable: categorías `classification`, `closed_qa` e
`information_extraction` cuya respuesta tiene como mucho tres palabras y aparece
literalmente en el prompt. Así `contains` comprueba selección o extracción y no
memoria del anotador. Sigue siendo un smoke test léxico: una respuesta puede
contener el fragmento esperado y aun así contradecirse.

## Los dos usos

### 1. Control del pipeline

Antes de culpar a un corpus de dominio, entrena un adapter con este y mira si el
modelo resultante se comporta. Si `dolly-training.jsonl` también produce un
modelo roto, el problema está en los hiperparámetros, el template de chat o la
carga del LoRA, no en los datos de dominio.

```bash
./scripts/train-adapter.sh trainer/examples/dolly-training.jsonl dolly-control
```

La validación se detecta sola y el manifest registra `qualitySelection`.

### 2. Replay contra el olvido catastrófico

Un corpus de dominio pequeño entrenado solo consigo mismo empuja al adapter a
responder siempre en ese dominio. `trainer/mix_replay.py` reinyecta una fracción
de datos generales:

```bash
python trainer/mix_replay.py \
  --domain trainer/examples/terraria-training.jsonl \
  --name terraria-dolly-mix \
  --ratio 0.2
```

Escribe `trainer/datasets/terraria-dolly-mix-{training,validation}.jsonl`. El
dominio conserva **todos** sus ejemplos y el replay se suma encima hasta ocupar
`--ratio` de la mezcla.

La validación se mezcla en la misma proporción a propósito. Si solo se mezclara
el entrenamiento, `eval_loss` mediría únicamente el dominio y la selección de
checkpoint premiaría justo el checkpoint que más capacidad general perdió.

Los `system` prompts distintos conviven a propósito: el modelo aprende a seguir
el prompt que recibe en vez de colapsar sobre uno solo. `train.py` lo detecta y
deja `systemPrompt` vacío en el manifest, que es el comportamiento correcto.

## Reconstruir y validar

```bash
python trainer/corpora/dolly/build.py
python trainer/corpora/dolly/build.py --check
python trainer/corpora/dolly/build.py --profile full
python trainer/validate_dataset.py trainer/examples/dolly-training.jsonl
python trainer/validate_dataset.py trainer/corpora/dolly/validation.jsonl
```

## Licencia y atribución

`databricks/databricks-dolly-15k` es **CC BY-SA 3.0**, © empleados de
Databricks, Inc. Los pasajes de contexto proceden de Wikipedia, también
CC BY-SA 3.0. Share-alike: si redistribuyes el corpus adaptado o un derivado
suyo, mantén la atribución y la misma licencia.

Sus autores declaran limitaciones que siguen aplicando aquí: sesgos y errores
factuales heredados de Wikipedia, anotadores que no siempre son hablantes
nativos de inglés, y una composición demográfica que es la de la plantilla de
Databricks. Es un corpus de instrucción general, no una fuente factual.
