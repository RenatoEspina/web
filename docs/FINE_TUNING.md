# Fine-tuning QLoRA

La implementación de fine-tuning entrena adaptadores PEFT mediante **SFT + QLoRA**. vLLM se utiliza únicamente para inferencia y para cargar/descargar los adaptadores resultantes.

La interfaz administrativa principal es la GUI local documentada en [`FINE_TUNING_GUI.md`](FINE_TUNING_GUI.md). El CLI de `comandos.fish` se mantiene para automatización y diagnóstico.

## Flujo recomendado

Desde la raíz del repositorio:

```fish
# 1. Preparar el entorno una vez
./comandos.fish fine-tune-setup

# 2. Comprobar CUDA y una cuantización NF4 real
./comandos.fish fine-tune-check

# 3. Validar el dataset antes de entrenar
./comandos.fish fine-tune-validate datasets/entrenamiento.jsonl

# 4. Entrenar
./comandos.fish fine-tune-train datasets/entrenamiento.jsonl qwen-dominio-v1 \
  --rank 16 --epochs 3

# 5. Listar adaptadores terminados
./comandos.fish fine-tune-list
```

También puedes iniciar la GUI local:

```bash
./fine-tune-gui
```

Desde ella se puede preparar el entorno, subir y validar datasets, entrenar, iniciar/detener vLLM y cargar o descargar LoRA dinámicamente.

## Entorno Python

El entorno preferido es `trainer/.venv`. Las dependencias están fijadas en `trainer/requirements.txt`.

```bash
python3 -m venv trainer/.venv
trainer/.venv/bin/python -m pip install -r trainer/requirements.txt
trainer/.venv/bin/python trainer/check_environment.py
```

`trainer/check_environment.py` es la única comprobación CUDA/NF4 usada por la GUI y por `comandos.fish`. Comprueba:

- PyTorch y CUDA;
- GPU visible;
- `bitsandbytes`;
- una operación real de cuantización NF4.

Si PyTorch utiliza CUDA 13.x, mantén `bitsandbytes` en la versión fijada por `trainer/requirements.txt` o una versión compatible posterior.

## Dataset

Cada línea del JSONL representa una conversación independiente:

```json
{"messages":[{"role":"user","content":"Pregunta"},{"role":"assistant","content":"Respuesta ideal"}]}
```

El validador canónico está en `trainer/dataset_validation.py`. Comprueba roles, contenidos, cantidad de mensajes, tamaño y que la respuesta final pertenezca al asistente.

Para validar manualmente:

```fish
./comandos.fish fine-tune-validate dataset.jsonl
```

El repositorio incluye `trainer/examples/base-training.jsonl` únicamente para probar el pipeline. No debe utilizarse para afirmar mejoras de calidad.

No reutilices ejemplos de entrenamiento en el conjunto de evaluación.

## Entrenamiento

El comando principal es:

```fish
./comandos.fish fine-tune-train dataset.jsonl qwen-dominio-v1 \
  --rank 16 \
  --epochs 3
```

Internamente llama a `scripts/train-adapter.sh`. Si vLLM está activo, el script lo detiene antes del entrenamiento y lo vuelve a iniciar al finalizar para liberar VRAM durante QLoRA.

El entrenador escribe primero en un directorio temporal oculto bajo `adapters/`. Solo cuando el entrenamiento, los pesos, tokenizer y `manifest.json` se han guardado correctamente publica el directorio final mediante un rename atómico. Un entrenamiento fallido no deja un adaptador parcial ocupando el nombre solicitado.

El resultado final tiene una estructura similar a:

```text
adapters/qwen-dominio-v1/
├── adapter_config.json
├── adapter_model.safetensors
├── manifest.json
├── tokenizer_config.json
└── ...
```

`manifest.json` registra modelo base, dataset, parámetros, métricas de entrenamiento y versión de PyTorch.

La semilla se aplica antes de cargar el modelo e inicializar LoRA, además de
pasarse a la configuración del entrenamiento. Repetir el experimento requiere
también los mismos pesos base, datos y versiones; el determinismo completo de
CUDA depende del hardware y de las operaciones utilizadas.

## Carga dinámica en vLLM

El flujo estático basado en editar `--lora-modules` manualmente está retirado del flujo normal.

`docker-compose.yml` habilita LoRA y la actualización runtime. La GUI local utiliza:

```text
POST /v1/load_lora_adapter
POST /v1/unload_lora_adapter
```

Al cargar un adaptador, la GUI también lo incorpora a `LLM_ADAPTER_MODELS` en `.env.local`. Al descargarlo, lo retira de esa allowlist. Si el gateway web ya estaba ejecutándose, reinícialo para que relea el entorno.

El puerto de vLLM permanece ligado a `127.0.0.1`; los endpoints administrativos de LoRA no deben exponerse mediante el túnel público.

Antes de cargar o comprobar un adaptador, la GUI contrasta
`adapter_config.json.base_model_name_or_path` y, si existe,
`manifest.json.baseModel` con el `root` del modelo base servido. Los nombres
publicados por vLLM pueden ser alias: la identidad comprobada es su `root`.
Un alias con el nombre de la base entrenada no permite usar otros pesos.

La procedencia del entrenamiento se valida en esa capa local, antes de llamar a
`/v1/load_lora_adapter` y antes de incorporar el nombre a `LLM_ADAPTER_MODELS`.
El gateway web puede ejecutarse mediante Vinext/Cloudflare Workers y por diseño
no intenta leer `adapters/` desde el filesystem del host. Antes de cada inferencia
consulta `/v1/models` y exige que el modelo base configurado esté servido, que el
adaptador seleccionado siga cargado, que exponga `parent` y `root`, y que el
`parent` resuelva al mismo `root` que la base configurada. Cualquier
inconsistencia runtime se rechaza con HTTP 409.

Las exportaciones Qwen 3.5 son content-addressed. Si se reemplazan los archivos
locales de un adaptador, la copia ya cargada continúa siendo la que usa vLLM
hasta descargar/recargar; la siguiente carga vuelve a validar la base y genera o
selecciona la exportación correspondiente al contenido actual. Los adaptadores
PEFT antiguos sin manifest pueden usarse si conservan un `adapter_config.json`
válido.

## Evaluación base contra LoRA

`trainer/evaluate.py` ejecuta casos independientes contra un modelo ya servido por vLLM.
Cada caso debe incluir una lista `contains` no vacía con fragmentos de texto
esperados, todos no vacíos. Una conversación de evaluación puede empezar con un
`system`, después alterna `user`/`assistant` y siempre termina en `user`: la
respuesta que se evalúa no debe estar ya incluida en la entrada.

Se valida todo el archivo antes de llamar al modelo. Los casos sin criterios,
con respuestas de referencia al final, formatos inválidos o un dataset vacío
se rechazan; nunca cuentan como aprobados. Los turnos assistant anteriores sí
se permiten como contexto de una evaluación de varios turnos.

```fish
./comandos.fish fine-tune-evaluate datasets/evaluacion.jsonl \
  Qwen/Qwen3.5-0.8B outputs/base.json

./comandos.fish fine-tune-evaluate datasets/evaluacion.jsonl \
  qwen-dominio-v1 outputs/lora.json
```

El reporte contiene:

- `passRate` por coincidencia de fragmentos;
- latencia por caso;
- respuesta completa.

Esta métrica es un **smoke test**, no una evaluación completa de calidad. Para resultados de investigación se debe complementar con revisión manual, calidad semántica, alucinaciones, degradación de capacidades generales y métricas de rendimiento.

## Comandos disponibles

```text
fine-tune-setup
fine-tune-check
fine-tune-validate
fine-tune-train
fine-tune-list
fine-tune-evaluate
```

Ejecuta:

```fish
./comandos.fish fine-tune-help
```

para ver la ayuda actual.

## Auditoría de calidad y reproducibilidad

Antes de reservar la GPU, el entrenador aplica la misma plantilla de chat a todos
los ejemplos y registra su distribución de tokens: total, tokens assistant,
p50, p95, truncamientos y respuestas que quedarían fuera de `--max-length`.
Por defecto aborta si una respuesta assistant sería truncada. Para aceptar ese
riesgo de manera consciente se puede usar:

```fish
./comandos.fish fine-tune-train dataset.jsonl adaptador-v1 \
  --max-length 1024 --allow-truncation
```

El `manifest.json` resultante incluye hashes SHA-256 del dataset y validación,
el commit del repositorio, la revisión del modelo, el hash del chat template,
versiones, hardware y los resúmenes de tokens. Para reproducibilidad en
Hugging Face, entrega una revisión fija:

```fish
./comandos.fish fine-tune-train dataset.jsonl adaptador-v1 \
  --model-revision <commit-o-revision>
```

Para medir si el LoRA mejora realmente las respuestas, sirve los dos modelos y
envía las mismas preguntas:

```fish
./comandos.fish fine-tune-evaluate trainer/corpora/terraria/evaluation.jsonl \
  qwen-terraria outputs/terraria-comparison.json \
  --compare-model Qwen/Qwen3.5-0.8B
```

El reporte conserva un resultado independiente por modelo y calcula
`passedDelta`, `passRateDelta` y la diferencia de latencia. Un `eval_loss`
menor o un `effect_detected` positivo no reemplaza esta comparación: solo
demuestra aprendizaje o modificación de probabilidades, no una mejora de
calidad.
