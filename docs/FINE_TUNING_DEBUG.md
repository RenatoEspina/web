# Diagnóstico de calidad y rendimiento de LoRA

Esta guía separa cuatro causas que suelen confundirse: selección incorrecta del adapter, modelo base desincronizado, terminación temprana aprendida por el fine-tuning y rendimiento real de vLLM/GPU.

## 1. Confirmar el estado de vLLM

Desde la raíz del repositorio:

```bash
docker compose -p llm-bridge ps
curl -s http://127.0.0.1:8000/v1/models | jq
```

Debe aparecer el modelo base y, si el LoRA está cargado, una entrada adicional para el adapter. La entrada del adapter debe declarar como `parent` el mismo modelo base que usa el gateway.

Comprueba también la configuración persistida:

```bash
grep -E '^(LLM_MODEL|LLM_ADAPTER_MODELS)=' .env.local
```

El gateway ahora valida `/v1/models` antes de cada inferencia con vLLM. Si el modelo base de `.env.local` no coincide con el realmente servido, o si el `parent` del LoRA no coincide, `/api/chat` responde con HTTP 409 y un mensaje explícito en vez de generar en un estado incoherente.

## 2. Comprobar que el adapter pertenece al modelo base correcto

```bash
jq '{baseModel, parameters, metrics}' adapters/<adapter>/manifest.json
```

El campo `baseModel` debe ser exactamente el modelo base que está sirviendo vLLM.

Para adapters creados con el entrenador actual, confirma además:

```bash
jq '.parameters.assistantOnlyLoss' adapters/<adapter>/manifest.json
```

Debe devolver `true`. Un adapter antiguo entrenado antes de esta corrección debe reentrenarse para beneficiarse de ella.

## 3. Ejecutar el diagnóstico A/B del LoRA

Con el modelo base y el adapter cargados:

```bash
trainer/.venv/bin/python trainer/verify_lora.py \
  --base-model Qwen/Qwen3.5-0.8B \
  --adapter <adapter> \
  --output outputs/<adapter>-verify.json
```

El reporte contiene dos grupos de pruebas:

- `distributionCases`: confirma que el LoRA modifica realmente las probabilidades respecto del modelo base.
- `generationCases`: compara generaciones completas con el mismo prompt y registra latencia, `completionTokens`, `tokensPerSecond` y `finishReason`.

Si `diagnostics.earlyStopSuspected` es `true`, el adapter termina mucho antes que el modelo base de forma repetida. Eso apunta a dataset, EOS o sobreajuste, no a que vLLM esté usando menos GPU por token.

## 4. Comparar desde el frontend sin contaminación del historial

Reinicia la web principal después de cargar o descargar adapters para que relea `.env.local`. En la interfaz principal:

1. selecciona **Sin documentos** para excluir RAG/CAG;
2. elige el modelo base;
3. formula una pregunta de prueba;
4. registra los indicadores mostrados bajo la respuesta: modelo, latencia, tokens de salida, tok/s y motivo de finalización;
5. cambia al adapter; el cambio de modelo limpia automáticamente el historial;
6. formula exactamente la misma pregunta;
7. compara las métricas.

Interpretación rápida:

- **Tok/s parecido + muchos menos tokens + `finishReason=stop`:** el adapter está terminando temprano; revisar entrenamiento/dataset.
- **Mismos tokens aproximados + tok/s mucho menor:** investigar runtime, LoRA o GPU.
- **`finishReason=length`:** la salida chocó con `LLM_MAX_TOKENS`; no es EOS aprendido.
- **HTTP 409:** hay una desincronización entre gateway, vLLM o el `parent` del adapter.

## 5. Aislar completamente el frontend y el gateway

Haz la misma consulta directamente contra vLLM. Primero al modelo base:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"Qwen/Qwen3.5-0.8B",
    "messages":[{"role":"user","content":"EXACTAMENTE LA MISMA PREGUNTA"}],
    "temperature":0,
    "max_tokens":512
  }' | jq '{text:.choices[0].message.content, finish:.choices[0].finish_reason, usage:.usage}'
```

Después repite cambiando únicamente `model` por el nombre del adapter.

Si la respuesta pobre y corta también aparece directamente en vLLM, el frontend queda descartado como causa. Si vLLM directo funciona bien pero `/api/chat` no, revisa el prompt, contexto documental y configuración del gateway.

## 6. Observar GPU y logs durante exactamente la misma prueba

En una terminal:

```bash
watch -n 0.5 nvidia-smi
```

En otra:

```bash
docker compose -p llm-bridge logs -f vllm
```

No compares solo el porcentaje instantáneo de GPU. Una respuesta de 60 tokens naturalmente mantiene la GPU ocupada mucho menos tiempo que una de 400 tokens. La métrica útil para distinguir ambos casos es `tokensPerSecond` junto con `completionTokens`.

## 7. Revisar si el dataset enseña respuestas demasiado cortas

Para un JSONL conversacional:

```bash
jq -r '.messages[-1].content | length' trainer/datasets/<dataset>.jsonl \
  | awk '{sum+=$1; n++; if($1<min || n==1) min=$1; if($1>max) max=$1} END {print "ejemplos="n, "promedio_chars="sum/n, "min="min, "max="max}'
```

Si casi todas las respuestas del assistant son muy cortas, el LoRA puede aprender a emitir EOS pronto aunque el contenido sea correcto.

## 8. Criterio para decidir la causa

Antes de tocar hiperparámetros, reúne para la misma pregunta:

- respuesta del modelo base;
- respuesta del adapter;
- `completionTokens` de ambos;
- `tokensPerSecond` de ambos;
- `finishReason` de ambos;
- resultado de `earlyStopSuspected`;
- salida de `/v1/models` con `parent`;
- `assistantOnlyLoss` del manifest.

Con esos datos se puede distinguir de forma reproducible entre un problema de integración, un problema del runtime LoRA y un problema de entrenamiento.
