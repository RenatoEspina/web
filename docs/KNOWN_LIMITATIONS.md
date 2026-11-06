# Limitaciones aceptadas y errores conocidos

Registro central de problemas del proyecto. Separa dos categorías:

- **Limitaciones aceptadas:** decisiones deliberadas del alcance actual. No se consideran errores prioritarios mientras el perfil de investigación siga siendo el mismo.
- **Errores conocidos:** comportamientos incorrectos, inseguros o engañosos que deben corregirse, aunque no bloqueen el flujo principal actual.

Cuando una limitación aceptada pase a formar parte del alcance soportado, debe retirarse de la primera sección y tratarse como requisito funcional. Cuando un error se corrija con su regresión, debe moverse a [Correcciones recientes](#correcciones-recientes-relacionadas).

**Fuentes consolidadas (2026-11-05):** este documento reúne los problemas que antes estaban repartidos en `README.md`, `docs/FINE_TUNING_DATASETS.md`, `docs/FINE_TUNING_AXOLOTL.md`, `docs/FINE_TUNING_DEBUG.md`, `docs/TERRARIA_FINE_TUNING.md` y `docs/SCALING_MULTI_AGENT.md`, más una revisión de código de solo lectura del subsistema de fine-tuning. Los errores marcados **Confirmado** fueron trazados en el código; los marcados **Plausible** requieren una prueba en ejecución.

## Limitaciones aceptadas

### Gateway y documentos

#### Cambio del modelo de embeddings con un índice existente

Los documentos mantienen sus vectores en memoria junto con metadatos del proveedor, modelo y dimensión usados al indexarlos. El flujo actual presupone que el modelo de embeddings no cambia durante la vida del índice.

Cambiar `EMBEDDING_MODEL`, sus prefijos o el proveedor después de haber indexado documentos puede dejar vectores antiguos en memoria.

**Práctica aceptada actualmente:** reiniciar el gateway y volver a cargar los PDF después de cambiar la configuración de embeddings. Una política automática de reindexado o invalidación queda fuera de la prioridad inmediata.

#### Engines generativos alternativos

El gateway conserva un adaptador para Ollama y una abstracción OpenAI-compatible, pero las pruebas, el ciclo LoRA, CAG con prefix caching y la operación normal se priorizan para vLLM.

No se considera prioridad inmediata garantizar equivalencia funcional completa entre vLLM y otros engines.

#### Configuración específica del equipo de investigación

`docker-compose.yml` contiene parámetros deliberadamente ajustados al entorno de investigación, incluyendo límites de VRAM, CPU y runtime NVIDIA.

La generalización y optimización para hardware distinto se pospone hasta que la arquitectura funcional esté estabilizada.

#### Persistencia documental

Los PDF, chunks, embeddings y cachés RAG/CAG viven actualmente en memoria. Reiniciar el gateway elimina el índice.

Este comportamiento se acepta para el MVP local, aunque no sería apropiado para múltiples réplicas, serverless o una biblioteca documental persistente.

#### Presupuesto de contexto basado en caracteres

RAG y CAG limitan actualmente el contexto mediante cantidad de caracteres y no mediante tokens del tokenizer real.

Se acepta como aproximación para el perfil actual. Debe reemplazarse por presupuesto de tokens antes de realizar pruebas de contexto extensas o comparar modelos con tokenizadores muy distintos.

#### Historial conversacional después de cambiar documentos

Las respuestas anteriores permanecen en el historial aunque hayan sido obtenidas con ayuda de un PDF que posteriormente se desmarque o aunque el modo cambie a `none`.

Por ahora se considera un comportamiento conversacional aceptado: el modelo puede recordar información que ya apareció explícitamente en la conversación. La selección documental controla el **nuevo contexto recuperado**, no borra el contenido textual de turnos anteriores.

Si en el futuro se necesita aislamiento estricto por corpus, deberá versionarse el contexto de la conversación o reiniciarse el historial al cambiar de selección documental.

### Fine-tuning y evaluación

Origen: `docs/FINE_TUNING_DATASETS.md` (secciones *Diagnóstico* y *Límites honestos*) y `docs/SCALING_MULTI_AGENT.md`.

#### SFT no inyecta conocimiento factual

Un corpus pequeño de pares pregunta/respuesta (por ejemplo, los 160 ejemplos de Terraria) enseña sobre todo formato y registro, no hechos. Para cobertura factual con fuentes, la herramienta del proyecto es RAG.

#### Corpus de dominio pequeños producen pocos pasos de optimizador

160 ejemplos con batch 1, gradient accumulation 8 y 2 épocas son 40 pasos de optimizador. En ese régimen el adapter apenas se mueve o sobreajusta. La mitigación documentada es mezclar replay general (`trainer/mix_replay.py`) y subir épocas o bajar la acumulación, cambiando una variable a la vez.

#### El replay reduce el olvido catastrófico, no lo elimina

LoRA reduce el olvido frente al fine-tuning completo, pero no lo elimina. Todo adapter de dominio debe evaluarse dos veces: contra su corpus de dominio y contra `trainer/corpora/dolly/evaluation.jsonl` para medir la capacidad general conservada.

#### La métrica `contains` es un smoke test léxico

Una respuesta puede contener el fragmento esperado y contradecirse, o fallar por parafrasear bien. Los resultados de `trainer/evaluate.py` deben leerse, no solo contarse.

#### Dolly no es una fuente factual

Hereda sesgos y errores de Wikipedia declarados por sus autores. Se usa como corpus de instrucción general y de control, no como verdad de referencia.

#### Mediciones atadas al hardware de investigación

Los números de tokens, el tope de 2200 caracteres y los minutos por época están medidos con `Qwen/Qwen3.5-0.8B` en una RTX 3060 Laptop de 6 GB. Con otro modelo base o GPU deben volver a medirse.

#### La VRAM es el cuello de botella para escalar a varios agentes

Con 6 GB de VRAM, varias generaciones simultáneas compiten por memoria. Un diseño multiagente debe ejecutar especialistas en serie o compartir un único modelo base con adaptadores LoRA.

## Errores conocidos pendientes

### Resumen

| ID | Área | Severidad | Estado | Error |
|----|------|-----------|--------|-------|
| [FT-01](#ft-01-no-hay-exclusión-mutua-de-gpu-y-vllm-entre-procesos) | Fine-tuning | Alta | Confirmado (OOM plausible) | No hay exclusión mutua de GPU/vLLM entre procesos |
| [FT-02](#ft-02-cerrar-la-gui-deja-huérfano-el-entrenamiento) | Fine-tuning | Alta | Confirmado | Cerrar la GUI deja huérfano el entrenamiento |
| [FT-03](#ft-03-allowlist-lora-desincronizada-tras-reiniciar-vllm) | Fine-tuning | Media | Confirmado | Allowlist LoRA desincronizada tras reiniciar vLLM |
| [FT-04](#ft-04-envlocal-se-reescribe-sin-conservar-permisos) | Seguridad | Media | Confirmado (impacto plausible) | `.env.local` se reescribe sin conservar permisos |
| [FT-05](#ft-05-la-cancelación-no-escala-y-el-job-puede-quedar-running) | Fine-tuning | Media | Confirmado (bloqueo plausible) | La cancelación no escala y el job puede quedar `running` |
| [FT-06](#ft-06-el-backend-axolotl-entrena-all-linear-sobre-qwen35) | Fine-tuning | Media | Confirmado | El backend Axolotl entrena `all-linear` sobre Qwen3.5 |
| [FT-07](#ft-07-la-gui-marca-el-entorno-como-listo-sin-verificarlo) | Fine-tuning | Media | Confirmado | La GUI marca el entorno como listo sin verificarlo |
| [FT-08](#ft-08-el-staging-de-entrenamiento-no-se-limpia-al-cancelar) | Fine-tuning | Media-baja | Confirmado | El staging de entrenamiento no se limpia al cancelar |
| [FT-09](#ft-09-mix_replaypy-escribe-antes-de-validar) | Fine-tuning | Media-baja | Confirmado | `mix_replay.py` escribe antes de validar |
| [FT-10](#ft-10-la-subida-de-dataset-no-captura-todas-las-excepciones) | Fine-tuning | Baja | Confirmado | La subida de dataset no captura todas las excepciones |
| [FT-11](#ft-11-efectos-con-root-antes-de-validar-la-entrada) | Seguridad | Baja | Confirmado | Efectos con root antes de validar la entrada |
| [FT-12](#ft-12-requirementstxt-no-fija-torch) | Entorno | Baja | Confirmado | `requirements.txt` no fija torch |
| [FT-13](#ft-13---host-1-se-acepta-pero-falla) | Fine-tuning | Baja | Confirmado | `--host ::1` se acepta pero falla |
| [FT-14](#ft-14-solo-loopback-no-implica-solo-este-usuario) | Seguridad | Baja | Plausible | "Solo loopback" no implica "solo este usuario" |
| [GW-01](#gw-01-host-y-port-parecen-configurables-desde-envlocal-sin-afectar-a-vite) | Gateway | Baja | Confirmado | `HOST`/`PORT` de `.env.local` no afectan a Vite |
| [GW-02](#gw-02-la-interfaz-dice-embeddings-activos-sin-comprobar-el-servicio) | Gateway | Baja | Confirmado | "Embeddings activos" sin health check |
| [GW-03](#gw-03-la-ui-puede-mostrar-rag-sin-documentos-seleccionados) | Gateway | Baja | Confirmado | La UI muestra RAG sin documentos seleccionados |
| [DOC-01](#doc-01-documentación-desactualizada-sobre-all-linear) | Docs | Baja | Confirmado | Documentación desactualizada sobre `all-linear` |

Los errores GW-01, GW-02, GW-03 y FT-07 ya figuraban en este documento; se verificó el 2026-11-05 que siguen presentes en el código.

### Fine-tuning

#### FT-01: no hay exclusión mutua de GPU y vLLM entre procesos

`trainer/gui_server.py:69-70, 808-819`: `STATE_LOCK` y `CURRENT_JOB` son globales del proceso. La GUI básica (`:3031`), la GUI Axolotl (`:3032`) y `./comandos.fish fine-tune-train` pueden ejecutarse a la vez.

**Impacto:** la GUI Axolotl entrena y su script detiene vLLM (`scripts/train-adapter-axolotl.sh:84`). La GUI básica ve vLLM detenido y lanza otro entrenamiento: quedan dos QLoRA en 6 GB, con riesgo de OOM. Cuando termina el primero, su trap `restart_vllm` levanta vLLM (0.70 de VRAM) mientras el segundo sigue entrenando. Ambos procesos reescriben `.env.local`.

**Corrección prevista:** un lock entre procesos (`fcntl.flock` sobre `.runtime/gpu.lock`) tomado por `start_job` y por `scripts/train-adapter*.sh` (`flock`); rechazar la acción si está ocupado.

#### FT-02: cerrar la GUI deja huérfano el entrenamiento

`trainer/gui_server.py:976-979`: `/api/shutdown` apaga el servidor sin cancelar el job. `run_command` lanza el hijo con `start_new_session=True` (`:364`), así que sobrevive.

**Impacto:** al presionar **Cerrar panel** (o matar la GUI) durante un entrenamiento, este sigue ocupando la GPU. Al relanzar `./fine-tune-gui`, `CURRENT_JOB=None` y se permite otro entrenamiento o `start-vllm`, que compite con el huérfano. vLLM queda detenido hasta que el huérfano termine.

**Corrección prevista:** rechazar el shutdown con un job `running`, o cancelarlo y esperarlo; registrar handlers de `atexit` y SIGTERM; persistir el PGID del job en `.runtime/` y comprobarlo al arrancar.

#### FT-03: allowlist LoRA desincronizada tras reiniciar vLLM

`scripts/train-adapter.sh:150-163` (`restart_vllm`), `comandos.fish:378` (`compose up -d vllm`) y `trainer/gui_server.py:719-726` (`stop-vllm`) reinician vLLM. `docker-compose.yml` no usa `--lora-modules`, así que los adaptadores cargados dinámicamente se pierden.

**Impacto:** se carga el LoRA `a` (queda en `LLM_ADAPTER_MODELS`) y luego se entrena `b`. El reinicio descarga `a`, pero sigue en la allowlist: el gateway lo ofrece y responde 409 (`lib/llm/runtime.ts` valida contra `/v1/models`). **Descargar** no sirve porque `loaded_adapter_path` (`gui_server.py:466-473`) lanza una excepción antes de tocar la allowlist. Además, `start_vllm` cambia `LLM_MODEL` (`comandos.fish:388`) sin purgar los adaptadores entrenados sobre la base anterior.

**Corrección prevista:** reconciliar después de cada (re)arranque, recargando los adaptadores de la allowlist con validación de base o retirándolos; permitir "retirar de la allowlist" cuando el adaptador no esté cargado.

#### FT-05: la cancelación no escala y el job puede quedar `running`

`trainer/gui_server.py:822-834` envía SIGINT una sola vez. `run_command` (`:369`) lee stdout hasta EOF sin timeout.

**Impacto:** si un hijo ignora o maneja lento SIGINT (pip, `docker compose pull`, `fish` dentro de `wait_for_url`, workers de accelerate o Axolotl), el job nunca termina y todas las acciones responden "Ya hay una operación en curso" hasta matar procesos a mano.

**Corrección prevista:** SIGINT, esperar unos 10 s, SIGTERM, esperar y SIGKILL al grupo; marcar el job `cancelled` aunque el stdout siga abierto.

#### FT-06: el backend Axolotl entrena `all-linear` sobre Qwen3.5

`trainer/train_axolotl.py:111-112` fija `lora_target_linear: True`. `trainer/train.py:35-50` documenta que adaptar `in_proj_a`/`in_proj_b` de las capas de atención lineal de Qwen3.5 produce bucles de repetición, y por eso el backend básico usa `attention-mlp` por defecto. La medición de `docs/FINE_TUNING_DEBUG.md:175-176` lo respalda: `all-linear` obtuvo 3/60 frente a 39/60 de `attention-mlp`.

**Impacto:** el mismo dataset produce un adapter sano en la GUI básica y uno que repite en bucle en la GUI Axolotl. Axolotl tampoco aplica la auditoría de truncamiento de `train.py:174-249` y entrena el JSONL crudo, sin el `strip()` de `load_jsonl`.

**Corrección prevista:** para `qwen3_5`, emitir `lora_target_modules` con el preset `attention-mlp` y registrarlo en el manifest.

#### FT-07: la GUI marca el entorno como listo sin verificarlo

`trainer/gui_server.py:327` calcula `environmentReady` como `trainer_python().is_file()`. Encontrar el ejecutable Python del venv no garantiza que PyTorch, CUDA, `bitsandbytes` y NF4 funcionen.

**Impacto:** un entorno creado parcialmente o con dependencias rotas se muestra como **Entorno: listo** hasta ejecutar la comprobación real.

**Corrección prevista:** separar al menos `venvPresent` de `environmentVerified` y basar el estado verde en una comprobación real o en el resultado persistido de la última comprobación válida.

#### FT-08: el staging de entrenamiento no se limpia al cancelar

`trainer/train.py:577-579` y `trainer/train_axolotl.py:318-320` limpian con `except Exception`. La cancelación (SIGINT) llega como `KeyboardInterrupt`, que es `BaseException`.

**Impacto:** cada entrenamiento cancelado deja `adapters/.<name>.training-*` o `.axolotl-output-*` con checkpoints y estado del optimizador. `list_adapters` no los muestra (`SAFE_NAME` exige un primer carácter alfanumérico) y `delete-adapter` no los alcanza, así que el disco se va llenando.

**Corrección prevista:** `except BaseException`, o un `finally` con un flag de éxito.

#### FT-09: `mix_replay.py` escribe antes de validar

`trainer/mix_replay.py:121-122` escribe el split de training antes del chequeo de fuga (`:133-135`). La rama sin validación (`:139-146`) no borra un `<name>-validation.jsonl` anterior.

**Impacto:** si una corrida previa dejó `X-validation.jsonl` y la nueva aborta por fuga o no tiene validación, queda un `X-training.jsonl` nuevo junto a la validación vieja. `scripts/train-adapter.sh:95-99` los empareja: el checkpoint se elige con un split ajeno, o el chequeo de overlap aborta con un error confuso.

**Corrección prevista:** armar ambos splits en memoria, validarlos y escribirlos atómicamente; si no hay validación, borrar la obsoleta.

#### FT-10: la subida de dataset no captura todas las excepciones

`trainer/gui_server.py:162-167` captura solo `OSError`, `ValueError` y `JSONDecodeError`. Un JSON muy anidado lanza `RecursionError`, que no es `ValueError` (verificado).

**Impacto:** con una línea `{"messages":[[[[...` la excepción escapa de `do_POST` (`:981-986`), el cliente no recibe respuesta JSON y queda `trainer/datasets/.<name>.<hex>.tmp`. Relacionado: `int_arg` con `1e400` lanza `OverflowError` (`:138`), aunque dentro del job.

**Corrección prevista:** capturar `Exception` en `dataset_summary` y borrar el temporal en un `finally`.

#### FT-13: `--host ::1` se acepta pero falla

`trainer/gui_server.py:998-1001`: `ThreadingHTTPServer` usa `AF_INET` y falla con `gaierror` ante una dirección IPv6.

**Corrección prevista:** una subclase con `address_family = AF_INET6` cuando el host contiene `:`.

### Seguridad

#### FT-04: `.env.local` se reescribe sin conservar permisos

`trainer/gui_server.py:584-585` (`write_text` + `os.replace`) y `comandos.fish:260-270` (`awk > tmp; mv`) crean un archivo nuevo.

**Impacto:** si `.env.local` tiene modo 0600 y contiene `APP_TOKEN` o una API key (`lib/llm/config.ts`), tras cargar un adaptador o iniciar vLLM queda con el umask (normalmente 0644), legible por otros usuarios. Un symlink se reemplaza por un archivo regular. El modo actual del archivo no se pudo comprobar en la revisión.

**Corrección prevista:** crear el temporal con `os.open(tmp, O_CREAT | O_EXCL | O_WRONLY, 0o600)` o copiar el modo con `shutil.copymode`; usar `umask 077` en fish.

#### FT-11: efectos con root antes de validar la entrada

`trainer/gui_server.py:646-660` y `trainer/gui_server_axolotl.py:234-248` llaman a `ensure_adapter_dir_writable`, que puede lanzar un contenedor `--user 0:0` con `chown -R`, antes de `validate_adapter_name`, `validate_model` y los hiperparámetros.

**Corrección prevista:** validar todo el payload antes de cualquier efecto.

#### FT-14: "solo loopback" no implica "solo este usuario"

`GET /api/bootstrap` (`trainer/gui_server.py:907-908`) entrega el token CSRF a cualquier proceso local. En un host multiusuario, otro usuario puede entrenar, borrar adaptadores o iniciar Docker con la identidad del dueño de la GUI. Depende del host.

**Corrección prevista (opcional):** escuchar en un socket Unix con permisos 0600, o entregar el token mediante un archivo con modo 0600.

### Entorno

#### FT-12: `requirements.txt` no fija torch

`trainer/requirements.txt` no declara torch: pip lo resuelve de forma transitiva (vía `bitsandbytes` y `accelerate`) a la última versión. No es reproducible, puede no calzar con el driver CUDA, y `trainer/check_dependencies.py` no lo verifica.

**Corrección prevista:** fijar `torch==X` con el index CUDA explícito.

### Gateway y chat

#### GW-01: `HOST` y `PORT` parecen configurables desde `.env.local` sin afectar a Vite

`.env.example` presenta `HOST` y `PORT` como configuración normal, mientras `vite.config.ts:48-49` lee `process.env.HOST` y `process.env.PORT` durante la evaluación de la configuración.

**Impacto:** si esos valores existen únicamente en `.env.local`, pueden no modificar el servidor de desarrollo como espera el usuario.

**Corrección prevista:** cargar explícitamente los archivos de entorno en `vite.config.ts` mediante `loadEnv`, o retirar esas variables de la configuración documentada.

#### GW-02: la interfaz dice "embeddings activos" sin comprobar el servicio

`app/page.tsx:547` deriva el texto de `embedding.enabled`, que significa que los embeddings están habilitados por configuración. `/api/health` comprueba el proveedor generativo, no el servicio de embeddings.

**Impacto:** Ollama puede estar caído y la UI seguir mostrando **embeddings activos**. La indexación degrada correctamente a recuperación léxica, pero el estado visual es engañoso.

**Corrección prevista:** usar **habilitados** para el estado de configuración y reservar **activos/disponibles** para un health check real.

#### GW-03: la UI puede mostrar RAG sin documentos seleccionados

El selector (`app/page.tsx:172-173`) puede quedar en `RAG` con `selectedDocumentIds=[]`. El backend interpreta correctamente una selección vacía como ausencia de contexto documental y devuelve `mode=none`.

**Impacto:** la semántica del backend es correcta, pero la interfaz sugiere que la siguiente pregunta usará RAG cuando no usará ningún PDF.

**Corrección prevista:** mostrar "selecciona al menos un documento" o reflejar el modo efectivo `none` mientras la selección esté vacía.

### Documentación

#### DOC-01: documentación desactualizada sobre `all-linear`

Desde que `attention-mlp` es el valor por defecto (`docs/FINE_TUNING_DEBUG.md:183`), varios documentos siguen describiendo `all-linear` como configuración vigente:

- `docs/FINE_TUNING_AXOLOTL.md:45` presenta `lora_target_linear: true` como "equivalente práctico al objetivo `all-linear` del backend básico", cuando el backend básico ya no usa `all-linear` (ver FT-06).
- `docs/TERRARIA_FINE_TUNING.md:141` indica que se mantiene `all-linear` fijo en el plan experimental.
- `docs/FINE_TUNING_DATASETS.md:133` atribuye parte del problema de Terraria a `all-linear` sin aclarar que ya no es el valor por defecto.

**Corrección prevista:** actualizar esos pasajes junto con la corrección de FT-06.

## Tests faltantes

La revisión del 2026-11-05 no encontró tests para:

- la cancelación de jobs y su escalamiento (FT-05);
- el shutdown con un job activo (FT-02);
- el lock entre GUIs y CLI (FT-01);
- la conservación del modo de `.env.local` (FT-04);
- la limpieza del staging ante `KeyboardInterrupt` (FT-08);
- la reconciliación de la allowlist tras reiniciar vLLM (FT-03);
- los errores de upload que no son `ValueError` (FT-10);
- los target modules de Axolotl (FT-06);
- el orden de escritura de `mix_replay` (FT-09).

## Revisado sin problemas (2026-11-05)

La revisión del subsistema de fine-tuning verificó como correctos:

- el chequeo de `Host` (anti DNS-rebinding), el token CSRF en un header propio y la ausencia de CORS; el bind restringido a loopback; vLLM publicado en `127.0.0.1:8000` con `adapters:ro`;
- `dataset_path` con `resolve()`, que rechaza traversal, rutas absolutas y symlinks hacia afuera; las expresiones `SAFE_NAME`, `SAFE_MODEL` y `SAFE_DATASET`;
- subprocesos invocados siempre con listas argv, sin `shell=True`; el `chown` dentro de Docker con uid/gid enteros;
- el escape con `escapeHtml` en el frontend y los logs mostrados con `textContent`; el HF token no se persiste ni aparece en los snapshots;
- `vllm_export.py`: rechaza symlinks, es content-addressed, verifica hashes y hace rename atómico;
- `validate_adapter_base` compara el `root` y no el alias; load, unload y delete revierten vLLM si falla la allowlist;
- `train.py` no sobrescribe un adaptador existente y valida la máscara de assistant.

Ejecuciones de esa revisión: compilación de todos los `trainer/*.py` sin errores; `python -m unittest` con 46 OK y 5 omitidos; tests de Node del fine-tuning con 52 OK y 1 fallo de entorno (`finetune-review-regressions.test.mjs` requiere `node_modules/vite`).

## Correcciones recientes relacionadas

Los siguientes problemas ya no se consideran errores conocidos porque cuentan con corrección y regresión:

- una selección explícita `documentIds: []` ya no utiliza todos los documentos por accidente;
- RAG solo publica fuentes cuyos chunks realmente entraron al contexto;
- RAG sin candidatos ya no afirma en el `system` prompt que se recuperaron fragmentos relevantes;
- la carga y descarga dinámica de LoRA intenta revertir el estado de vLLM si falla la actualización de `LLM_ADAPTER_MODELS`, y reporta explícitamente si también falla el rollback;
- la API evita procesar un nuevo PDF cuando el workspace ya alcanzó el límite de documentos;
- el backend básico usa `attention-mlp` por defecto en lugar de `all-linear`, porque adaptar las puertas de estado de Qwen3.5 degeneraba en bucles de repetición (`docs/FINE_TUNING_DEBUG.md`);
- la GUI Axolotl reconstruye con Python 3.12 un entorno creado con Python 3.14 al pulsar **Preparar / actualizar entorno** (`docs/FINE_TUNING_AXOLOTL.md`).
