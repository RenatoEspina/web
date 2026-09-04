# Fine-tuning QLoRA

> Si PyTorch muestra CUDA 13.x, usa `bitsandbytes` 0.50.2 o posterior. Las versiones antiguas pueden intentar cargar un binario como `libbitsandbytes_cuda130.so` que no está incluido. Si ya instalaste las dependencias, actualízalo dentro del venv con `trainer/.venv/bin/python -m pip install --force-reinstall --no-cache-dir bitsandbytes==0.50.2` y vuelve a comprobar la importación.

Esta implementación entrena adaptadores reales PEFT mediante SFT + QLoRA. vLLM no entrena: carga el modelo base y el adaptador resultante para inferencia.

## Flujo recomendado, de principio a fin

Desde la raíz del repositorio, el camino normal es:

```fish
# 1) Una sola vez por máquina: entorno y dependencias
./comandos.fish fine-tune-setup

# 2) Antes de usar la GPU: comprobar CUDA y bitsandbytes
./comandos.fish fine-tune-check

# 3) Antes de entrenar: validar el JSONL
./comandos.fish fine-tune-validate datasets/entrenamiento.jsonl

# 4) Entrenar; el nombre se convertirá en adapters/<nombre>
./comandos.fish fine-tune-train datasets/entrenamiento.jsonl qwen-dominio-v1 \
  --rank 16 --epochs 3

# 5) Revisar los adaptadores terminados
./comandos.fish fine-tune-list

# 6) Obtener las instrucciones exactas para publicar uno en vLLM
./comandos.fish fine-tune-config qwen-dominio-v1
```

`fine-tune-train` valida otra vez el dataset, detiene el servicio vLLM del
proyecto `llm-bridge` si está activo para liberar la VRAM, ejecuta el trainer y
lo vuelve a iniciar al terminar (también si el entrenamiento falla). No
arranques otro entrenamiento en paralelo en la RTX 3060.

## Preparación

El entrenamiento usa la GPU y vLLM debe estar detenido. En la raíz:

```bash
python -m venv trainer/.venv
trainer/.venv/bin/pip install -r trainer/requirements.txt
trainer/.venv/bin/python trainer/validate_dataset.py dataset.jsonl
```

También puedes delegar la preparación y las comprobaciones al CLI de Fish del
proyecto:

```fish
./comandos.fish fine-tune-setup
./comandos.fish fine-tune-check
./comandos.fish fine-tune-validate dataset.jsonl
```

`fine-tune-check` no solo importa los paquetes: ejecuta una cuantización NF4
pequeña en CUDA, por lo que detecta el caso en que bitsandbytes imprime un
error pero deja continuar al intérprete Python.

El CLI usa `trainer/.venv` como ubicación preferida. Si ya tienes un `.venv`
en la raíz, lo reutiliza; también puedes indicar explícitamente otro intérprete
con `FINE_TUNE_PYTHON=/ruta/al/python`.

Si Fish no está instalado, usa los equivalentes Bash de las secciones
siguientes. El script `scripts/train-adapter.sh` también usa el nombre de
proyecto Compose `llm-bridge`, por lo que detecta correctamente el vLLM
iniciado por `comandos.fish`.

Cada línea del dataset contiene una conversación independiente:

```json
{"messages":[{"role":"user","content":"Pregunta"},{"role":"assistant","content":"Respuesta ideal"}]}
```

No uses las mismas preguntas para entrenamiento y evaluación. El ejemplo incluido solo verifica el flujo; dos registros no bastan para producir un adaptador útil.

## Entrenamiento

```bash
./comandos.fish fine-tune-train dataset.jsonl qwen-dominio-v1 --rank 16 --epochs 3
```

El comando anterior es un envoltorio de `scripts/train-adapter.sh`. Si
necesitas ejecutar el script directamente, conserva la forma equivalente:

```bash
./scripts/train-adapter.sh dataset.jsonl qwen-dominio-v1 --rank 16 --epochs 3
```

El script valida el dataset, detiene vLLM si estaba activo, ejecuta QLoRA y vuelve a iniciar vLLM al salir. El resultado queda en `adapters/qwen-dominio-v1/`, incluido `manifest.json` con parámetros y métricas.

Para 6 GB comienza con `rank=8` o `16`, `batch-size=1`, acumulación 8 y longitud 1024. Si aparece OOM, reduce primero `--max-length`; después usa rank 8. No subas los pesos al repositorio.

## Servir el adaptador

Compose ya monta `./adapters` como solo lectura y habilita LoRA. Añade temporalmente al comando de vLLM:

```yaml
- --lora-modules
- qwen-dominio-v1=/adapters/qwen-dominio-v1
```

Define además:

```dotenv
LLM_ADAPTER_MODELS=qwen-dominio-v1
```

Recrea vLLM y reinicia la web. El selector del chat ofrecerá el modelo base y el adaptador. La lista permitida impide que una petición externa seleccione nombres arbitrarios.

El comando `fine-tune-config` comprueba que existan `adapter_config.json` y
`adapter_model.safetensors` y muestra estos pasos sin modificar tus secretos ni
sobrescribir `.env.local`:

```fish
./comandos.fish fine-tune-config qwen-dominio-v1
```

Después de aplicar las dos líneas de configuración, recrea vLLM y verifica que
el adaptador aparece en `/v1/models`:

```bash
./comandos.fish stop
./comandos.fish vllm
curl -fsS http://127.0.0.1:8000/v1/models
```

Para varios adaptadores, sepáralos por comas en `LLM_ADAPTER_MODELS` y registra
cada par `nombre=/adapters/nombre` que quieras precargar en el comando de
vLLM. El nombre enviado por el gateway debe coincidir exactamente con el
nombre registrado.

## Evaluación base contra LoRA

Construye un archivo de casos nunca usados durante entrenamiento. Cada línea lleva `messages` y fragmentos esperados en `contains`. Luego ejecuta ambos modelos con temperatura cero:

```bash
python trainer/evaluate.py --dataset evaluacion.jsonl --model Qwen/Qwen3.5-0.8B --output outputs/base.json
python trainer/evaluate.py --dataset evaluacion.jsonl --model qwen-dominio-v1 --output outputs/lora.json
```

Considera el resultado un fine-tuning útil solo si mejora en un conjunto de prueba separado y no degrada de forma importante tareas generales. Revisa además respuestas manualmente: la tasa `contains` es reproducible, pero no mide por sí sola calidad, seguridad ni alucinaciones.

Para simplificar la evaluación desde Fish:

```fish
./comandos.fish fine-tune-evaluate evaluacion.jsonl Qwen/Qwen3.5-0.8B outputs/base.json
./comandos.fish fine-tune-evaluate evaluacion.jsonl qwen-dominio-v1 outputs/lora.json
```

Para listar adaptadores que ya tienen manifiesto:

```fish
./comandos.fish fine-tune-list
```

La evaluación necesita que vLLM esté activo y que el nombre del modelo ya esté
publicado. Compara siempre el mismo conjunto de casos, primero con el modelo
base y luego con el adaptador:

```fish
./comandos.fish fine-tune-evaluate datasets/evaluacion.jsonl \
  Qwen/Qwen3.5-0.8B outputs/base.json
./comandos.fish fine-tune-evaluate datasets/evaluacion.jsonl \
  qwen-dominio-v1 outputs/lora.json
```

El resultado JSON incluye `passRate`, latencia y la respuesta de cada caso.
Una mejora real requiere un conjunto de prueba separado del entrenamiento,
revisión manual de respuestas y comprobación de que el adaptador no empeora
tareas generales.

TRL integra `SFTTrainer` con PEFT para entrenar adaptadores sin actualizar todos los parámetros, y recomienda PEFT/LoRA como técnica para reducir memoria: [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) y [reducción de memoria](https://huggingface.co/docs/trl/reducing_memory_usage).
