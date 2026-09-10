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

El botón **Preparar / actualizar entorno** de la variante Axolotl crea un entorno virtual separado para no modificar `trainer/.venv`. La integración fija Axolotl `0.18.0` y prepara PyTorch antes de instalar Axolotl con `--no-build-isolation`.

El botón **Comprobar GPU** valida CUDA, `bitsandbytes`/NF4, el CLI de Axolotl y la versión esperada.

## Entrenamiento

La GUI Axolotl conserva los parámetros visibles del panel: modelo base, rank, alpha, dropout, épocas, learning rate, micro-batch, gradient accumulation, longitud máxima y seed.

El wrapper genera una configuración Axolotl por ejecución con:

- `adapter: qlora` y cuantización de 4 bits;
- `lora_target_linear: true`, equivalente práctico al objetivo `all-linear` del backend básico;
- dataset OpenAI `messages` con `roles_to_train: [assistant]`;
- `train_on_eos: turn` para no entrenar sobre los turnos `system`/`user`;
- `sample_packing: false`; para Qwen3.5 esto evita depender de FLA en el perfil de una sola GPU;
- plantilla `qwen3_5` y `enable_thinking: false` cuando el modelo pertenece a Qwen3.5.

Si existe un dataset de validación separado para Terraria, Unidades o un archivo hermano `*-validation.jsonl`, se añade como `test_datasets`.

## Salida y serving

Los adapters terminados siguen publicándose en `adapters/<nombre>` y deben contener los archivos PEFT esperados por el proyecto. Además se escriben:

- `manifest.json` con `backend: axolotl` y metadatos de reproducibilidad;
- `axolotl-config.yaml` con la configuración efectiva del entrenamiento.

La carga, descarga, verificación y exportación para vLLM siguen usando la infraestructura existente. El entrenamiento mantiene el comportamiento de liberar vLLM antes de ocupar la GPU y restaurarlo al finalizar.
