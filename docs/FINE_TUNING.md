# Fine-tuning QLoRA

> Si PyTorch muestra CUDA 13.x, usa `bitsandbytes` 0.50.2 o posterior. Las versiones antiguas pueden intentar cargar un binario como `libbitsandbytes_cuda130.so` que no está incluido. Si ya instalaste las dependencias, actualízalo con `python -m pip install --force-reinstall --no-cache-dir bitsandbytes==0.50.2` y vuelve a comprobar la importación.

Esta implementación entrena adaptadores reales PEFT mediante SFT + QLoRA. vLLM no entrena: carga el modelo base y el adaptador resultante para inferencia.

## Preparación

El entrenamiento usa la GPU y vLLM debe estar detenido. En la raíz:

```bash
python -m venv trainer/.venv
trainer/.venv/bin/pip install -r trainer/requirements.txt
trainer/.venv/bin/python trainer/validate_dataset.py dataset.jsonl
```

Cada línea del dataset contiene una conversación independiente:

```json
{"messages":[{"role":"user","content":"Pregunta"},{"role":"assistant","content":"Respuesta ideal"}]}
```

No uses las mismas preguntas para entrenamiento y evaluación. El ejemplo incluido solo verifica el flujo; dos registros no bastan para producir un adaptador útil.

## Entrenamiento

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

## Evaluación base contra LoRA

Construye un archivo de casos nunca usados durante entrenamiento. Cada línea lleva `messages` y fragmentos esperados en `contains`. Luego ejecuta ambos modelos con temperatura cero:

```bash
python trainer/evaluate.py --dataset evaluacion.jsonl --model Qwen/Qwen3.5-0.8B --output outputs/base.json
python trainer/evaluate.py --dataset evaluacion.jsonl --model qwen-dominio-v1 --output outputs/lora.json
```

Considera el resultado un fine-tuning útil solo si mejora en un conjunto de prueba separado y no degrada de forma importante tareas generales. Revisa además respuestas manualmente: la tasa `contains` es reproducible, pero no mide por sí sola calidad, seguridad ni alucinaciones.

TRL integra `SFTTrainer` con PEFT para entrenar adaptadores sin actualizar todos los parámetros, y recomienda PEFT/LoRA como técnica para reducir memoria: [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) y [reducción de memoria](https://huggingface.co/docs/trl/reducing_memory_usage).
