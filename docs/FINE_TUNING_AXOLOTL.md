# Fine-tuning con Axolotl

El proyecto conserva dos backends de entrenamiento independientes que comparten los mismos datasets, adaptadores y administración de vLLM.

## Lanzadores

```bash
./fine-tune-gui
```

Abre la GUI básica existente en `127.0.0.1:3031` y mantiene el pipeline TRL + PEFT de `trainer/train.py`.

```bash
./fine-tune-gui-axolotl
```

Abre la variante Axolotl en `127.0.0.1:3032`. Esta GUI reutiliza el panel existente, pero prepara `trainer/.venv-axolotl` y redirige el entrenamiento a `trainer/train_axolotl.py`.

Ambas interfaces pueden existir instaladas en el menú de aplicaciones:

- **LLM Bridge Fine-tuning**
- **LLM Bridge Fine-tuning (Axolotl)**

## Entorno Axolotl

El botón **Preparar / actualizar entorno** de la variante Axolotl crea un entorno virtual separado para no modificar `trainer/.venv`.

El entorno de Axolotl queda fijado a **Python 3.12**. Si el equipo no tiene `python3.12` instalado, la GUI prepara `uv` y deja que `uv` descargue automáticamente una distribución compatible de Python 3.12. El Python del sistema solo se usa, si es necesario, para crear el pequeño bootstrap de `uv`; Axolotl no se instala en ese intérprete.

Si `trainer/.venv-axolotl` ya existe pero fue creado con otra versión, por ejemplo Python 3.14, el botón de preparación lo detecta como incompatible y reconstruye exclusivamente ese entorno dedicado. La GUI básica y `trainer/.venv` no se modifican.

La integración fija Axolotl `0.18.0` y PyTorch `2.12.1`. Por defecto usa `UV_TORCH_BACKEND=cu130`; puede sobrescribirse con `AXOLOTL_TORCH_BACKEND` si otra instalación requiere un backend distinto.

Este perfil **no instala el extra `deepspeed`**. El proyecto entrena QLoRA en una sola GPU y la configuración generada no contiene una sección `deepspeed:`. Evitar ese extra también evita exigir un CUDA Toolkit de desarrollo local (`CUDA_HOME`/`nvcc`) cuando PyTorch CUDA y `bitsandbytes` ya son suficientes para este flujo.

El botón **Comprobar GPU** valida Python 3.12, CUDA, `bitsandbytes`/NF4, el CLI de Axolotl y la versión esperada.

## Entrenamiento

La GUI Axolotl conserva los parámetros visibles del panel: modelo base, rank, alpha, dropout, épocas, learning rate, micro-batch, gradient accumulation, longitud máxima y seed.

El wrapper genera una configuración Axolotl por ejecución con:

- `adapter: qlora` y cuantización de 4 bits;
- `lora_target_linear: true`, equivalente práctico al objetivo `all-linear` del backend básico;
- dataset OpenAI `messages` con `roles_to_train: [assistant]`;
- `train_on_eos: turn` para no entrenar sobre los turnos `system`/`user`;
- `sample_packing: false`; para Qwen3.5 esto evita depender del packing optimizado en este perfil de una sola GPU;
- plantilla `qwen3_5` y `enable_thinking: false` cuando el modelo pertenece a Qwen3.5.

Si existe un dataset de validación separado para Terraria, Unidades o un archivo hermano `*-validation.jsonl`, se añade como `test_datasets`.

## Salida y serving

Los adapters terminados siguen publicándose en `adapters/<nombre>` y deben contener los archivos PEFT esperados por el proyecto. Además se escriben:

- `manifest.json` con `backend: axolotl` y metadatos de reproducibilidad;
- `axolotl-config.yaml` con la configuración efectiva del entrenamiento.

La carga, descarga, verificación y exportación para vLLM siguen usando la infraestructura existente. El entrenamiento mantiene el comportamiento de liberar vLLM antes de ocupar la GPU y restaurarlo al finalizar.

## Recuperación de un entorno creado con Python 3.14

Después de actualizar esta rama, cierra y vuelve a abrir la GUI Axolotl para cargar el servidor nuevo:

```bash
./fine-tune-gui-axolotl
```

Luego pulsa **Preparar / actualizar entorno**. No es necesario borrar manualmente `trainer/.venv-axolotl`: la GUI detectará el entorno 3.14 y lo reconstruirá con Python 3.12 antes de instalar Axolotl.
