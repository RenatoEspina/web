# GUI local de fine-tuning

El proyecto incluye un panel web **exclusivamente local** para preparar y ejecutar fine-tuning SFT + QLoRA sin tener que copiar comandos entre terminales.

## Arranque

Desde la raíz del proyecto, la primera vez ejecuta:

```bash
./fine-tune-gui
```

El lanzador:

1. inicia `trainer/gui_server.py` en `127.0.0.1:3031`;
2. abre el navegador automáticamente;
3. crea una entrada local llamada **LLM Bridge Fine-tuning** en el menú de aplicaciones del usuario.

Después de esa primera ejecución se puede abrir el panel desde el lanzador de aplicaciones sin volver a usar la terminal. Si el puerto `3031` está ocupado, se puede definir otro antes de ejecutar el lanzador:

```bash
FINE_TUNE_GUI_PORT=3040 ./fine-tune-gui
```

## Flujo recomendado

### 1. Preparar entorno

Pulsa **Preparar entorno**. La GUI busca Python 3.13 primero y, si no está disponible, usa `python3`/`python`. Luego:

- crea `trainer/.venv`;
- actualiza `pip`;
- instala `trainer/requirements.txt`;
- comprueba PyTorch, CUDA, la GPU, `bitsandbytes` y una cuantización NF4 real.

El log se muestra dentro del navegador.

### 2. Subir el dataset

Selecciona un `.jsonl`. La GUI lo valida con la misma lógica del entrenador antes de persistirlo en `trainer/datasets/`.

También aparece `trainer/examples/base-training.jsonl`, pensado únicamente para comprobar que el pipeline funciona. No es un dataset suficiente para medir calidad real.

### 3. Entrenar

Configura como mínimo:

- nombre del adaptador;
- modelo base;
- rank LoRA;
- épocas.

Los parámetros avanzados permiten cambiar alpha, dropout, learning rate, batch size, gradient accumulation, longitud máxima y seed.

Al pulsar **Preparar y entrenar**, la GUI:

1. crea el entorno automáticamente si todavía no existe;
2. comprueba CUDA/NF4;
3. valida nuevamente el dataset;
4. ejecuta `scripts/train-adapter.sh`;
5. detiene vLLM si estaba usando la GPU;
6. entrena el adaptador;
7. vuelve a iniciar vLLM si el script detectó que estaba activo antes del entrenamiento.

Solo se permite una operación pesada al mismo tiempo. La operación actual se puede cancelar desde la propia página.

### 4. Iniciar vLLM

La GUI puede iniciar o detener únicamente el servicio `vllm` de `docker-compose.yml`. Para un modelo público el token de Hugging Face puede dejarse vacío. Para modelos gated se puede introducir el token en el formulario; el panel no lo persiste.

### 5. Cargar el adaptador

Los adaptadores válidos creados bajo `adapters/` aparecen en la GUI. Al pulsar **Cargar en vLLM** se usa el endpoint local `/v1/load_lora_adapter` y la ruta del contenedor `/adapters/<nombre>`.

Además, la GUI agrega el nombre a `LLM_ADAPTER_MODELS` dentro de `.env.local` para mantener la allowlist del gateway. Si la web ya estaba ejecutándose, hay que reiniciarla para que relea esa variable de entorno.

## Seguridad local

El panel está diseñado como una herramienta administrativa local, no como parte pública del chat:

- el servidor solo acepta `127.0.0.1`, `localhost` o `::1`;
- el lanzador usa `127.0.0.1`;
- las operaciones mutables requieren un token aleatorio de sesión enviado mediante `X-Fine-Tune-Token`;
- no se habilita CORS;
- los datasets solo se pueden seleccionar desde `trainer/datasets/` o `trainer/examples/`;
- los adaptadores solo se cargan desde `adapters/` y sus nombres se validan;
- no existe un endpoint para ejecutar comandos arbitrarios;
- vLLM continúa publicado en el host únicamente como `127.0.0.1:8000`.

La carga dinámica de LoRA de vLLM requiere `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`. vLLM advierte que esta función no debe exponerse a clientes no confiables. En este proyecto se habilita porque el puerto de vLLM y la GUI permanecen restringidos a loopback.

Documentación oficial: https://docs.vllm.ai/en/stable/features/lora/

## Cerrar el panel

Pulsa **Cerrar panel** en la esquina superior derecha. Esto detiene el servidor administrativo local. El adaptador entrenado y los servicios Docker continúan en el estado en que hayan quedado.

Si necesitas depurar un fallo de arranque, revisa:

```text
.runtime/fine-tune-gui.log
```

## CLI como fallback

La GUI no elimina los comandos existentes. Para automatización, scripts o diagnóstico siguen disponibles `fine-tune-setup`, `fine-tune-check`, `fine-tune-validate`, `fine-tune-train`, `fine-tune-list` y `fine-tune-evaluate` mediante `comandos.fish`.
