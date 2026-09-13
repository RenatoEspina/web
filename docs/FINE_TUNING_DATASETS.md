# Datasets documentados para fine-tuning en un PC personal

Investigación y decisión sobre qué corpus público usar en este proyecto como
**control del pipeline** y como **replay** contra el olvido catastrófico, con el
hardware objetivo real:

```text
GPU   NVIDIA GeForce RTX 3060 Laptop, 6144 MiB VRAM
RAM   16 GB DDR5
CPU   Intel i7-12650H (16 hilos)
base  Qwen/Qwen3.5-0.8B, QLoRA 4-bit, max_length 1024
```

## Qué se buscaba

No un dataset de dominio más, sino uno que cumpla cuatro condiciones a la vez:

1. **documentado**: dataset card pública, procedencia declarada, limitaciones
   escritas por sus autores;
2. **licencia usable**: el corpus de Terraria de este repositorio es
   CC BY-NC-SA 4.0, así que la cláusula no comercial ya limita un flujo; no
   convenía añadir otra;
3. **escrito por humanos**: un corpus destilado de otro LLM enseña el estilo de
   ese modelo maestro además de la tarea;
4. **que entre en 6 GB**: después de filtrar por longitud, ningún ejemplo debe
   superar el `--max-length` de 1024 tokens.

## Candidatos evaluados

| dataset | tamaño | licencia | origen | veredicto |
| --- | --- | --- | --- | --- |
| [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 15 011 | CC BY-SA 3.0 | humano, empleados de Databricks | **elegido** |
| [`HuggingFaceH4/no_robots`](https://huggingface.co/datasets/HuggingFaceH4/no_robots) | 10 000 | CC BY-NC 4.0 | humano, anotadores profesionales | descartado: no comercial, y 4560 de sus 10 000 filas son generación larga que se sale del presupuesto de 1024 tokens |
| `OpenAssistant/oasst1` | 66 k conversaciones | Apache 2.0 | humano, voluntarios | descartado: árboles multivuelta con ratings, requiere aplanado y filtrado por idioma/calidad; más trabajo de adaptación que valor aquí |
| `yahma/alpaca-cleaned` | 52 k | CC BY-NC 4.0 | generado con GPT-3 | descartado: no comercial y destilado de otro modelo |

Dolly gana los cuatro criterios: sus autores prohibieron explícitamente usar IA
generativa al redactarlo, la licencia permite uso comercial, y sus ocho
categorías vienen del paper de InstructGPT, así que cubre generación, QA abierto
y cerrado, clasificación, extracción, resumen y brainstorming en un solo corpus.

Sobre viabilidad de hardware: el estudio de perfilado
[*Profiling LoRA/QLoRA Fine-Tuning Efficiency on Consumer GPUs: An RTX 4060 Case
Study*](https://arxiv.org/abs/2509.12229) mide QLoRA sobre Qwen2.5-1.5B en 8 GB y
concluye que con estrategias eficientes en parámetros se llega a secuencias de
2048 tokens. Nuestro caso es más holgado: 0.8B en vez de 1.5B, y 1024 tokens.

## Medición real del presupuesto

Tokenizando Dolly con el tokenizer de `Qwen/Qwen3.5-0.8B` y la plantilla de chat
del proyecto:

| corte | ejemplos que entran |
| --- | --- |
| ≤ 512 tokens | 92.6 % |
| ≤ 1024 tokens | 98.4 % |

Mediana 155 tokens, p90 441. El corpus adaptado aplica un tope de 2200
caracteres, calibrado para que el ejemplo más largo quede en 1014 tokens. Tras
construirlo, el split de entrenamiento mide **mediana 148, p95 405, máximo 716
tokens**: cero ejemplos truncados.

Esto no es un detalle cosmético. `train.py` aborta si una respuesta `assistant`
se trunca a `--max-length`, y con razón: un target cortado a mitad enseña al
modelo a parar donde no debe.

## Verificación en la máquina objetivo

El corpus se entrenó de verdad en la RTX 3060 Laptop, no solo se construyó:

| medida | valor |
| --- | --- |
| VRAM en uso durante el entrenamiento | **4977 MiB de 6144** (con el escritorio abierto) |
| ejemplos truncados | 0 de 3000 |
| velocidad | 0.94 ejemplos/s → **~53 min por época** sobre 3000 ejemplos |
| validación completa (400 ejemplos) | ~51 s |

Un run corto de 19 pasos de optimizador (`--epochs 0.05`, 2.5 minutos) ya mueve
la validación en la dirección correcta:

```json
{
  "baseline":  { "baseline_loss": 2.2175, "baseline_perplexity": 9.185 },
  "selected":  { "validation_loss": 2.0327, "validation_perplexity": 7.635 },
  "relativeLossImprovement": 0.0834
}
```

Eso es lo que se espera de un pipeline sano: con una fracción de época la
pérdida de validación baja de forma clara y estable. Es la referencia contra la
que comparar cualquier corpus de dominio.

Consecuencia práctica para el control: **usa `--epochs 1`**. 375 pasos de
optimizador dan señal de sobra y cuestan ~55 minutos; dos épocas sobre 3000
ejemplos pasan de una hora y media sin aportar nada al diagnóstico.

## Qué se implementó

Ver [`trainer/corpora/dolly/README.md`](../trainer/corpora/dolly/README.md) para
el detalle del corpus. Resumen:

- `trainer/corpora/dolly/build.py` descarga una revisión inmutable, la verifica
  por SHA-256 y genera particiones deterministas sin fuga de prompts entre
  ellas;
- perfil `laptop` por defecto: 3000 / 400 / 250 (train / validation /
  evaluation), ~375 pasos de optimizador por época con batch 1 y gradient
  accumulation 8;
- la evaluación sale solo del subconjunto con respuesta verificable, de modo que
  `contains` mide extracción o selección y no memoria del anotador;
- `trainer/mix_replay.py` mezcla cualquier dataset de dominio con replay
  general, entrenamiento y validación a la vez.

Se integra en lo que ya existía: las dos GUI lo reconstruyen al abrirse,
`scripts/train-adapter.sh` y su variante Axolotl detectan su validación sola, y
el dataset aparece en el selector como **Ejemplo · dolly-training.jsonl**.

## Diagnóstico: por qué el adapter de Terraria da problemas

El corpus de Terraria de este repositorio está bien construido —split por
artículo fuente, validación separada, evaluación retenida—, pero la
configuración con la que se entrena tiene tres problemas que se suman.

### 1. Demasiados pocos pasos de optimizador

160 ejemplos, batch 1, gradient accumulation 8, 2 épocas son **40 pasos de
optimizador en total**. Es un régimen en el que el adapter apenas se mueve o
sobreajusta el ruido de esos 160 ejemplos, sin punto intermedio estable. El
perfil `laptop` de Dolly da ~750 pasos en las mismas dos épocas.

### 2. Un único system prompt y un único dominio

Los 160 ejemplos comparten el mismo `system` y hablan todos de Terraria. Con
`target_modules="all-linear"` y rank 16, lo que el adapter aprende no es solo
"datos de Terraria": aprende "responde siempre en este registro, sobre este
tema". Eso es exactamente lo que se percibe como modelo roto al usarlo en el
chat normal.

La literatura sobre olvido catastrófico es consistente en esto: LoRA lo reduce
frente al fine-tuning completo, pero **no lo elimina**, y menos cuando se aplica
a todas las proyecciones lineales. La mitigación documentada es reinyectar una
fracción de datos generales; los rangos que se reportan van del 1–5 % al 10–20 %
según la tarea.

### 3. SFT no es el mecanismo para inyectar hechos

160 pares pregunta/respuesta no meten de forma fiable el contenido de 40
artículos de wiki en los pesos. SFT enseña sobre todo formato, registro y forma
de responder. El propio `trainer/corpora/terraria/README.md` ya lo dice: para
cobertura factual actualizada y con fuentes, la herramienta es RAG, que este
proyecto ya tiene.

## Recetas para este hardware

### Paso 1 — Aislar el pipeline

Antes de seguir tocando el corpus de Terraria, comprueba que el pipeline produce
modelos sanos con un dataset conocido-bueno:

```bash
python trainer/corpora/dolly/build.py
./scripts/train-adapter.sh trainer/examples/dolly-training.jsonl dolly-control \
  --epochs 1 --rank 16 --alpha 32 --learning-rate 1e-4 \
  --batch-size 1 --gradient-accumulation 8 --max-length 1024
```

Tarda alrededor de una hora en esta máquina. El script detiene vLLM antes de
empezar y lo reinicia al terminar, así que la GPU no está compartida.

Luego carga el adapter en vLLM y compáralo contra el modelo base:

```bash
python trainer/evaluate.py \
  --dataset trainer/corpora/dolly/evaluation.jsonl \
  --model Qwen/Qwen3.5-0.8B --compare-model dolly-control \
  --output outputs/dolly-control.json
```

La lectura es binaria y ahorra mucho tiempo:

- si `dolly-control` conversa con normalidad y su `passRate` no empeora, el
  pipeline está bien y el problema está en los datos o el tamaño del corpus de
  Terraria;
- si también sale roto, el fallo está en hiperparámetros, plantilla de chat o
  carga del LoRA, y ningún cambio en el dataset de Terraria lo va a arreglar.

### Paso 2 — Reentrenar Terraria con replay

```bash
python trainer/mix_replay.py \
  --domain trainer/examples/terraria-training.jsonl \
  --name terraria-dolly-mix --ratio 0.2

./scripts/train-adapter.sh trainer/datasets/terraria-dolly-mix-training.jsonl \
  terraria-mix-v1 --epochs 3
```

La mezcla conserva los 160 ejemplos de Terraria y añade 40 generales encima. La
validación se mezcla en la misma proporción, así que `eval_loss` ya no puede
premiar al checkpoint que más capacidad general perdió.

Con 200 ejemplos siguen siendo pocos pasos: sube `--epochs` a 3 o 4, o baja
`--gradient-accumulation` a 4 para duplicar los pasos por época. Cambia **una
variable a la vez**, como pide el plan experimental que ya está escrito en
`docs/TERRARIA_FINE_TUNING.md`.

### Paso 3 — Medir las dos cosas

Un adapter de dominio útil tiene que aprobar dos evaluaciones, no una:

```bash
# ¿aprendió el dominio?
python trainer/evaluate.py --dataset trainer/corpora/terraria/evaluation.jsonl \
  --model Qwen/Qwen3.5-0.8B --compare-model terraria-mix-v1 \
  --output outputs/terraria-mix.json

# ¿conservó la capacidad general?
python trainer/evaluate.py --dataset trainer/corpora/dolly/evaluation.jsonl \
  --model Qwen/Qwen3.5-0.8B --compare-model terraria-mix-v1 \
  --output outputs/terraria-mix-general.json
```

La segunda es la que faltaba. Una caída fuerte de `passRate` frente al base en la
evaluación de Dolly es la medida directa de "el fine-tuning rompió el modelo", y
hasta ahora no había forma de cuantificarla en este repositorio.

## Límites honestos de todo esto

- `contains` es un smoke test léxico en ambos corpus. Una respuesta puede
  contener el fragmento esperado y contradecirse, o fallar por parafrasear bien.
  Léelas.
- Dolly no es una fuente factual: hereda sesgos y errores de Wikipedia, y sus
  autores lo declaran. Sirve como corpus de instrucción general, no como verdad.
- El replay reduce el olvido, no lo elimina. Con un dominio de 160 ejemplos y un
  modelo de 0.8B, la vía correcta para cobertura factual sigue siendo RAG.
- Los números de tokens y de tiempo de este documento están medidos con
  `Qwen/Qwen3.5-0.8B` en esta RTX 3060 Laptop. Con otro modelo base, vuelve a
  medir antes de confiar en el tope de 2200 caracteres o en los minutos por
  época.
