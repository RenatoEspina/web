# Diagnóstico de calidad y rendimiento de LoRA

Esta guía separa cuatro causas que suelen confundirse: selección incorrecta del adapter, modelo base desincronizado, terminación temprana aprendida por el fine-tuning y diferencias entre el contexto usado durante SFT y el usado al servir.

## 1. Confirmar el estado de vLLM

Desde la raíz del repositorio:

```bash
docker compose -p llm-bridge ps
curl -s http://127.0.0.1:8000/v1/models | jq
```

Debe aparecer el modelo base y, si el LoRA está cargado, una entrada adicional
para el adapter. Su `parent` debe apuntar a una base servida, pero **no prueba la
procedencia del entrenamiento**: en cargas dinámicas vLLM lo completa con la
base activa. El `root` de la entrada base identifica el modelo realmente servido;
su `id` puede ser un alias.

Comprueba también la configuración persistida:

```bash
grep -E '^(LLM_MODEL|LLM_ADAPTER_MODELS)=' .env.local
```

La procedencia del LoRA se valida en la GUI local antes de cargarlo y antes de
incorporarlo a `LLM_ADAPTER_MODELS`. El gateway no intenta leer `adapters/`
desde el filesystem del host porque puede ejecutarse dentro del runtime
Vinext/Cloudflare Workers. Antes de cada inferencia consulta `/v1/models` y
rechaza con HTTP 409 si el modelo base configurado no está servido, el adapter
seleccionado no está cargado, faltan `parent` o `root`, o el `parent` del LoRA
resuelve a un `root` diferente del modelo base configurado.

## 2. Comprobar que el adapter pertenece al modelo base correcto

```bash
jq '{baseModel, parameters, metrics}' adapters/<adapter>/manifest.json
```

El campo `baseModel` y `adapter_config.json.base_model_name_or_path` deben
coincidir con el `root` de la base servida. La GUI comprueba esto antes de cargar
LoRA; no basta con cambiar el alias del servidor para aparentar compatibilidad.

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
- **HTTP 409:** hay una desincronización entre gateway y vLLM: base esperada no servida, adapter no cargado o identidad runtime (`parent`/`root`) incoherente.

Para adapters Terraria, `/api/chat` hace además una clasificación de scope con el modelo base. Esa clasificación usa `temperature=0` y `maxTokens=8`; no reutiliza los parámetros de generación globales. Si `scope.allowed` es `false`, la consulta se rechaza antes de llegar al LoRA.

El clasificador conserva íntegra la consulta actual, incluido su final. Recorta
primero el historial para mantener el presupuesto agregado de 12.000 caracteres;
el límite por turno histórico sigue siendo 1.500 caracteres. Así no clasifica
solo la introducción de una pregunta larga. Estos límites son aproximaciones
por caracteres, no una garantía de que cualquier texto quepa en el contexto de
cualquier tokenizer.

## 5. Aislar completamente el frontend y el gateway

Para comprobar el efecto del LoRA sin contaminar la prueba con el gateway, envía la misma conversación directamente a vLLM.

En Terraria, el adapter fue entrenado siempre con el `system` de `trainer/corpora/terraria/data/metadata.json`. El serving normal sin documentos usa exactamente ese mismo texto. Para una comparación justa, úsalo tanto con el modelo base como con el adapter:

```bash
SYSTEM_PROMPT=$(jq -r '.system' trainer/corpora/terraria/data/metadata.json)

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "$(jq -n \
    --arg model 'Qwen/Qwen3.5-0.8B' \
    --arg system "$SYSTEM_PROMPT" \
    --arg question 'EXACTAMENTE LA MISMA PREGUNTA' \
    '{
      model:$model,
      messages:[
        {role:"system",content:$system},
        {role:"user",content:$question}
      ],
      temperature:0,
      max_tokens:512
    }')" \
  | jq '{text:.choices[0].message.content, finish:.choices[0].finish_reason, usage:.usage}'
```

Después repite cambiando únicamente `model` por el nombre del adapter. No cambies el `system`, la pregunta ni los parámetros entre ambas ejecuciones.

Si la respuesta pobre también aparece directamente en vLLM usando el mismo `system` que en SFT, el frontend y el clasificador quedan descartados como causa. Si el LoRA directo funciona bien pero `/api/chat` no, revisa `scope`, historial y modo documental.

Con RAG/CAG no existe paridad exacta con SFT porque necesariamente se añade contexto documental. Para Terraria, el gateway conserva el prompt SFT exacto como prefijo del único `system` y fusiona ahí las instrucciones documentales; evita enviar dos mensajes `system` consecutivos.

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
- `assistantOnlyLoss` del manifest;
- para adapters de dominio, decisión `scope` del gateway;
- confirmación de que la prueba directa usa el mismo `system` que el entrenamiento.

Con esos datos se puede distinguir de forma reproducible entre un problema de integración, un problema del runtime LoRA y un problema de entrenamiento.
