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

## Diseño del panel

La interfaz usa un **wizard de cinco pasos**. Solo se muestra la configuración correspondiente al paso actual, mientras una terminal integrada permanece visible en paralelo para seguir las operaciones en tiempo real.

En escritorio:

- panel izquierdo: paso actual y sus controles;
- panel derecho: terminal/log y cancelación de la operación actual.

En móvil ambos paneles se apilan verticalmente. Los botones **Atrás** y **Siguiente**, además del indicador superior de pasos, permiten navegar sin perder los valores introducidos en formularios anteriores.

Los pasos son:

1. Entorno;
2. Dataset;
3. Entrenamiento;
4. vLLM;
5. Adaptadores.

## Flujo recomendado

### 1. Preparar entorno

Pulsa **Preparar / actualizar entorno**. La GUI busca Python 3.13 y luego `python3`/`python`, crea `trainer/.venv`, actualiza `pip` e instala `trainer/requirements.txt`.

La verificación usa el mismo script que el CLI:

```text
trainer/check_environment.py
```

Este script comprueba PyTorch, CUDA, GPU, `bitsandbytes` y una cuantización NF4 real.

El estado distingue dos cosas: que el venv exista (**Entorno: sin verificar**) y que
la última comprobación real haya sido correcta (**Entorno: verificado**, en verde).
El resultado de **Comprobar GPU**, de **Preparar / actualizar entorno** y de la
verificación previa a cada entrenamiento se guarda en
`.runtime/environment-check.<script>.json`, uno por backend. Deja de contar si la
comprobación siguiente falla, si se vuelve a preparar el entorno o si el venv se
recrea. La comprobación por CLI (`./comandos.fish fine-tune-check`) no actualiza
ese estado.

### 2. Subir el dataset

Selecciona un `.jsonl`. Antes de guardarlo en `trainer/datasets/`, la GUI lo valida con `trainer/dataset_validation.py`, que es el validador canónico del proyecto.

Cualquier error al analizarlo, incluido un JSON con más de 512 niveles de anidamiento, se informa como dataset inválido; el archivo temporal de la subida se borra siempre.

`trainer/examples/base-training.jsonl` sirve únicamente para comprobar el pipeline. No es un dataset suficiente para medir calidad real.

`evaluation.jsonl` queda reservado para evaluación y no aparece como dataset entrenable.

### 3. Entrenar

Configura:

- nombre del adaptador;
- modelo base;
- rank LoRA;
- épocas;
- opcionalmente alpha, dropout, learning rate, batch size, gradient accumulation, longitud máxima y seed;
- opcionalmente un **HF token** para las descargas realizadas por Hugging Face durante esa ejecución.

#### Para qué sirve el HF token

El fine-tuning utiliza Hugging Face porque el modelo base (`Qwen/Qwen3.5-0.8B` por defecto), su configuración, tokenizer y pesos se resuelven mediante `transformers`/`huggingface_hub`.

El token puede ayudar durante la fase de descarga porque:

- autentica las solicitudes al Hub;
- evita depender de los límites anónimos;
- permite acceder a repositorios gated o privados cuando la cuenta tiene permiso.

El token **no aumenta los tokens por segundo del entrenamiento** una vez que el modelo ya está cargado en la GPU.

La GUI no guarda ese secreto. El servidor:

1. lo retira del payload de la operación al comenzar;
2. lo valida;
3. crea una copia del entorno del proceso hijo;
4. define únicamente `HF_TOKEN` y `HUGGING_FACE_HUB_TOKEN` en ese entorno;
5. ejecuta `scripts/train-adapter.sh` con ese entorno;
6. nunca añade el token a los argumentos de línea de comandos ni a los logs.

Si el campo queda vacío, el flujo continúa de manera anónima como antes.

Al pulsar **Preparar y entrenar**, la GUI:

1. prepara el entorno si todavía no existe;
2. comprueba CUDA/NF4;
3. ejecuta `scripts/train-adapter.sh`;
4. detiene vLLM si estaba activo para liberar GPU;
5. descarga el modelo si no está completo en caché, usando el HF token si fue proporcionado;
6. conserva los roles de la conversación y entrena la pérdida solo sobre las respuestas del asistente;
7. vuelve a iniciar vLLM si estaba activo antes del entrenamiento, espera a que responda y reconcilia la allowlist de adaptadores.

El entrenador trabaja primero en un directorio temporal oculto y solo publica `adapters/<nombre>/` después de guardar correctamente pesos, tokenizer y `manifest.json`. Un fallo no deja un adaptador parcial bloqueando el mismo nombre.

Solo se permite una operación simultánea, también entre procesos: la GUI básica,
la GUI Axolotl y `./comandos.fish fine-tune-train` comparten el lock
`.runtime/gpu.lock`. Mientras otro proceso lo tiene, la acción se rechaza con un
mensaje que nombra el lock; nunca queda esperando. Los scripts de entrenamiento
lanzados desde la GUI heredan el lock que ya tomó el panel.

El trabajo actual se puede cancelar desde la terminal integrada. La cancelación
envía `SIGINT` al grupo de procesos del job; si sigue activo tras unos 10 s envía
`SIGTERM`, y tras otros 5 s `SIGKILL`. El job queda como `cancelled` aunque un
subproceso desacoplado mantenga abierta la salida, pero nunca mientras quede vivo
algún proceso de su grupo (por ejemplo, un worker que ignora `SIGINT`): hasta
entonces sigue en curso, conserva `.runtime/gpu-job.json` y **Cerrar panel** se
rechaza.

### 4. Iniciar o detener vLLM

La GUI administra el servicio `vllm` del `docker-compose.yml`. Para modelos gated se puede introducir un token de Hugging Face en este paso; igual que el token del entrenamiento, no se persiste.

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

Reiniciar vLLM descarta los LoRA cargados dinámicamente. Por eso, después de cada
arranque con **Iniciar vLLM** o `./comandos.fish vllm`, y después del reinicio que
hace el entrenamiento, se ejecuta `trainer/runtime_adapters.py`: cada adaptador de
`LLM_ADAPTER_MODELS` que vLLM ya no sirve se recarga con la misma validación de base
que **Cargar en vLLM**. Solo se retira de la allowlist una entrada inválida en sí
misma: el adaptador ya no existe, sus metadatos son inválidos o vLLM sirve otra base.
Un fallo de infraestructura (intérprete o exportación, error de vLLM) nunca retira un
adaptador válido: se conserva y la operación termina con error. Desde la GUI Axolotl
la exportación usa el intérprete de `trainer/.venv-axolotl`. **Detener vLLM**
conserva la allowlist; el siguiente arranque la reconcilia.

Si se cancela un entrenamiento mientras vLLM se reinicia, la escalada de la
cancelación interrumpe la reconciliación. Por eso el entrenamiento deja la marca
`.runtime/adapters-reconcile-pending` al detener vLLM, y solo una reconciliación
completa la borra. Mientras el panel está abierto, cuando vLLM vuelve a responder y
no hay otra operación en curso, inicia solo una operación **Reconciliando allowlist
LoRA**. Si falla, la reintenta al cabo de un minuto.

Si un adaptador sigue en la allowlist sin estar cargado, la GUI muestra
**Retirar de la allowlist**: lo quita de `LLM_ADAPTER_MODELS` sin llamar a vLLM.

Para eliminar un adaptador, primero pulsa **Descargar de vLLM** si está activo y
después pulsa **Borrar**. La operación pide confirmación, retira el nombre de la
allowlist y borra también la exportación de compatibilidad de vLLM. No es reversible
desde la GUI.

#### Qwen3.5: carga real de LoRA y comprobación A/B

El modelo de texto de Transformers guarda módulos `model.layers.*`, mientras
vLLM 0.24.0 usa `language_model.model.layers.*` en su wrapper Qwen3.5, incluso
con `--language-model-only`. Registrar el adaptador en `/v1/models` no prueba
que los pesos estén aplicándose.

La GUI exporta Qwen3.5 automáticamente con el prefijo HF
`model.language_model.layers.*`, que el mapper de vLLM convierte al namespace
correcto. Guarda la copia en `adapters/.vllm-exports/<nombre>/<hash>/`, sin modificar
los archivos PEFT, checkpoints o manifiesto originales. La exportación se hace
en CPU, se publica al terminar y verifica hashes antes de reutilizarla. Un namespace
desconocido falla explícitamente. Los demás modelos no se renombrarán.

Si el LoRA ya estaba cargado con la ruta anterior, descárgalo y vuelve a cargarlo.
**Comprobar uso** ejecuta `trainer/verify_lora.py` y compara base repetida contra
adaptador en tres prompts. Detectar efecto no mide calidad ni garantiza que cada
tensor se aplique. La GUI informa una prueba inconclusa como tal.

La [prueba local registrada](verification/lora-runtime-2026-09-07.json) con
`qwen-es-v1` muestra 372/372 nombres exportados en el namespace esperado,
probabilidades idénticas al modelo base sin convertir y efecto detectado en
los tres prompts después de la exportación. Se usó vLLM 0.24.0.

Un fallo al actualizar la allowlist después de descargar restaura la ruta exacta
que tenía el runtime, incluida una exportación o una ruta heredada.

Para experimentar con el corpus incluido, sigue la
[guía de Terraria](TERRARIA_FINE_TUNING.md).

## Seguridad local

El panel es una herramienta administrativa local:

- solo acepta `127.0.0.1`, `localhost` o `::1` (con `--host ::1` escucha por IPv6 en `http://[::1]:<puerto>`);
- el lanzador usa `127.0.0.1`;
- las operaciones mutables requieren un token aleatorio de sesión en `X-Fine-Tune-Token`;
- no habilita CORS;
- los datasets solo provienen de `trainer/datasets/` o `trainer/examples/`;
- los adaptadores solo se cargan desde `adapters/` y sus nombres se validan;
- los HF tokens opcionales solo viven en memoria y en el entorno del proceso hijo correspondiente;
- no existe un endpoint para ejecutar comandos arbitrarios;
- cada acción valida su payload completo (nombre, modelo, token e hiperparámetros) antes de cualquier efecto, incluida la reparación de permisos de `adapters/` con un contenedor root;
- al reescribir `.env.local` (allowlist desde la GUI, `LLM_MODEL` desde `./comandos.fish vllm`) se conserva el modo del archivo; uno nuevo se crea con modo `0600` y un enlace simbólico se mantiene, actualizando el archivo al que apunta;
- vLLM continúa publicado en el host únicamente como `127.0.0.1:8000`.

La carga dinámica de LoRA requiere `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`. Esa API no debe exponerse a clientes no confiables; en este proyecto permanece detrás de loopback.

## Cerrar el panel

Pulsa **Cerrar panel**. Esto detiene el servidor administrativo local; los adaptadores y servicios Docker quedan en el estado actual.

Con una operación en curso, **Cerrar panel** se rechaza: cancélala o espera a que
termine. Si el servidor recibe `SIGTERM` o `Ctrl+C`, primero cancela el job activo
con la misma escalada y espera a que termine antes de salir.

Mientras corre, el PGID del job queda registrado en `.runtime/gpu-job.json`. Si un
proceso de una sesión anterior sigue vivo (por ejemplo, porque el panel se mató con
`SIGKILL`), al arrancar el panel lo informa en su log, la terminal lo muestra como
operación externa y las nuevas acciones se rechazan hasta que termine.

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
