# GUI local de fine-tuning

El proyecto incluye un panel web **exclusivamente local** para administrar SFT + QLoRA sin exponer operaciones administrativas en el chat público.

## Arranque

Desde la raíz:

```bash
./fine-tune-gui
```

El lanzador:

1. inicia `trainer/gui_server.py` en `127.0.0.1:3031`;
2. abre el navegador;
3. crea una entrada local llamada **LLM Bridge Fine-tuning** en el menú de aplicaciones del usuario.

Para usar otro puerto:

```bash
FINE_TUNE_GUI_PORT=3040 ./fine-tune-gui
```

La ruta `/fine-tune` de la aplicación principal ya no implementa un segundo flujo de entrenamiento: solo informa cómo abrir esta GUI local.

## Flujo recomendado

### 1. Preparar entorno

Pulsa **Preparar entorno**. La GUI busca Python 3.13 y luego `python3`/`python`, crea `trainer/.venv`, actualiza `pip` e instala `trainer/requirements.txt`.

La verificación usa el mismo script que el CLI:

```text
trainer/check_environment.py
```

Este script comprueba PyTorch, CUDA, GPU, `bitsandbytes` y una cuantización NF4 real.

### 2. Subir el dataset

Selecciona un `.jsonl`. Antes de guardarlo en `trainer/datasets/`, la GUI lo valida con `trainer/dataset_validation.py`, que es el validador canónico del proyecto.

`trainer/examples/base-training.jsonl` sirve únicamente para comprobar el pipeline. No es un dataset suficiente para medir calidad real.

`evaluation.jsonl` queda reservado para evaluación y no aparece como dataset entrenable.

### 3. Entrenar

Configura:

- nombre del adaptador;
- modelo base;
- rank LoRA;
- épocas;
- opcionalmente alpha, dropout, learning rate, batch size, gradient accumulation, longitud máxima y seed.

Al pulsar **Preparar y entrenar**, la GUI:

1. prepara el entorno si todavía no existe;
2. comprueba CUDA/NF4;
3. ejecuta `scripts/train-adapter.sh`;
4. detiene vLLM si estaba activo para liberar GPU;
5. entrena QLoRA;
6. vuelve a iniciar vLLM si estaba activo antes del entrenamiento.

El entrenador trabaja primero en un directorio temporal oculto y solo publica `adapters/<nombre>/` después de guardar correctamente pesos, tokenizer y `manifest.json`. Un fallo no deja un adaptador parcial bloqueando el mismo nombre.

Solo se permite una operación pesada simultánea. El trabajo actual se puede cancelar desde la propia página.

### 4. Iniciar o detener vLLM

La GUI administra el servicio `vllm` del `docker-compose.yml`. Para modelos gated se puede introducir un token de Hugging Face en el formulario; el panel no lo persiste.

### 5. Cargar o descargar LoRA

Los adaptadores válidos terminados bajo `adapters/` aparecen en la GUI.

**Cargar en vLLM** utiliza:

```text
POST /v1/load_lora_adapter
```

y registra el nombre en `LLM_ADAPTER_MODELS` dentro de `.env.local`.

**Descargar de vLLM** utiliza:

```text
POST /v1/unload_lora_adapter
```

y retira el nombre de `LLM_ADAPTER_MODELS` para evitar que el selector del gateway anuncie un adaptador que ya no está cargado.

Si la aplicación web ya estaba ejecutándose, reiníciala después de modificar la allowlist para que relea `.env.local`.

## Seguridad local

El panel es una herramienta administrativa local:

- solo acepta `127.0.0.1`, `localhost` o `::1`;
- el lanzador usa `127.0.0.1`;
- las operaciones mutables requieren un token aleatorio de sesión en `X-Fine-Tune-Token`;
- no habilita CORS;
- los datasets solo provienen de `trainer/datasets/` o `trainer/examples/`;
- los adaptadores solo se cargan desde `adapters/` y sus nombres se validan;
- no existe un endpoint para ejecutar comandos arbitrarios;
- vLLM continúa publicado en el host únicamente como `127.0.0.1:8000`.

La carga dinámica de LoRA requiere `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`. Esa API no debe exponerse a clientes no confiables; en este proyecto permanece detrás de loopback.

## Cerrar el panel

Pulsa **Cerrar panel**. Esto detiene el servidor administrativo local; los adaptadores y servicios Docker quedan en el estado actual.

Para depurar el lanzador:

```text
.runtime/fine-tune-gui.log
```

## CLI como fallback

Para automatización o diagnóstico:

```text
fine-tune-setup
fine-tune-check
fine-tune-validate
fine-tune-train
fine-tune-list
fine-tune-evaluate
```

Consulta la ayuda con:

```fish
./comandos.fish fine-tune-help
```
