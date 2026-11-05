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

### Alcance de las correcciones de fine-tuning (2026-11-05)

Decisiones tomadas al corregir FT-01 a FT-13. No son errores, pero conviene conocerlas:

- **Lock de GPU:** `.runtime/gpu.lock` cubre todas las acciones de ambas GUIs y `scripts/train-adapter*.sh` (y por tanto `./comandos.fish fine-tune-train`). `./comandos.fish vllm` ejecutado directamente desde la CLI no lo toma. Si `flock` no está instalado, los scripts fallan con un mensaje explícito.
- **Cierre de la GUI y fin de un job:** `/api/shutdown` responde 409 mientras hay un job en curso. Un job no termina, ni se borra `.runtime/gpu-job.json`, mientras quede vivo algún proceso de su grupo; esto incluye procesos en segundo plano que el comando deje sin cancelar. Solo SIGTERM o Ctrl+C sobre el servidor cancelan el job y esperan a que el grupo termine; si la espera vence, envían SIGKILL al grupo antes de salir (código de salida 143 con SIGTERM). Un job huérfano de una sesión anterior se informa como `externalJob` y bloquea acciones nuevas hasta que termina.
- **Reconciliación de la allowlist:** se ejecuta después de `./comandos.fish vllm` y de `restart_vllm` en ambos scripts, y falla de forma cerrada: solo retira una entrada inválida en sí misma (nombre inválido, directorio o archivos PEFT ausentes, metadatos inválidos, o un vLLM que sirve otra base conocida). Un error de exportación, de intérprete o de vLLM conserva la entrada y hace fallar el job; `./comandos.fish vllm` sale con código 1 en ese caso. Los scripts de entrenamiento crean `.runtime/adapters-reconcile-pending` antes de detener vLLM y solo una reconciliación completa lo borra. Así, si una cancelación interrumpe la reconciliación, la completa automáticamente un panel de fine-tuning abierto (job `reconcile-adapters`, revisado cada 5 s y reintentado a los 60 s si falla), el siguiente `./comandos.fish vllm` o `python3 trainer/runtime_adapters.py`. Sin panel abierto y sin ejecutar esos comandos, el gateway conserva la allowlist desactualizada. Un `docker compose up` manual no reconcilia: usa **Retirar de la allowlist** o vuelve a iniciar vLLM desde la GUI o la CLI.
- **Permisos de `.env.local`:** se conserva el modo del archivo existente y un archivo nuevo se crea con 0600. Un `.env.local` que es symlink se mantiene y se reescribe su destino.
- **Validación de payloads:** sigue ocurriendo dentro del job (202 seguido de un job fallido), así que el contrato HTTP para payloads inválidos no cambió; lo nuevo es que ningún efecto ocurre antes de validar.
- **Estado del entorno:** `environmentReady` conserva su significado anterior (venv presente). La insignia verde usa `environmentVerified`. `./comandos.fish fine-tune-check` no actualiza el resultado persistido que lee la GUI.
- **Backend Axolotl:** la auditoría de truncamiento usa la plantilla de TRL y Axolotl tokeniza con la suya, así que los conteos pueden diferir en unos pocos tokens de formato. Requiere `trl.chat_template_utils.get_training_chat_template`, que no se pudo comprobar sin el venv de Axolotl; si falta, falla con un mensaje claro. El manifest de Axolotl no registra los resúmenes de la auditoría ni `allowTruncation`, a diferencia de `train.py`.
- **Comprobación de torch:** `trainer/check_dependencies.py` aplica compatibilidad binaria de CUDA (los wheels oficiales omiten arquitecturas como `sm_89`). Un torch sin CUDA o sin GPU visible solo genera una advertencia; la disponibilidad de GPU la verifica `trainer/check_environment.py`.

## Errores conocidos pendientes

| ID | Área | Severidad | Estado | Error |
|----|------|-----------|--------|-------|
| [FT-12](#ft-12-requirementstxt-no-fija-torch) | Entorno | Baja | Parcial | `requirements.txt` no fija torch |
| [FT-14](#ft-14-solo-loopback-no-implica-solo-este-usuario) | Seguridad | Baja | Plausible | "Solo loopback" no implica "solo este usuario" |

### FT-12: `requirements.txt` no fija torch

`trainer/requirements.txt` no declara torch: pip lo resuelve de forma transitiva (vía `bitsandbytes` y `accelerate`) a la última versión, lo que no es reproducible y puede no calzar con el driver CUDA.

**Estado:** `trainer/check_dependencies.py` ya informa la versión de torch y su build de CUDA, y falla si la GPU visible no está soportada. Falta fijar `torch==X` con su index CUDA, lo que requiere decidir y probar una versión en el venv real del trainer.

### FT-14: "solo loopback" no implica "solo este usuario"

`GET /api/bootstrap` (`trainer/gui_server.py`) entrega el token CSRF a cualquier proceso local. En un host multiusuario, otro usuario puede entrenar, borrar adaptadores o iniciar Docker con la identidad del dueño de la GUI. Depende del host.

**Pendiente de decisión:** cualquier corrección (socket Unix con permisos 0600, o token entregado por un archivo 0600) cambia cómo el navegador obtiene el token, así que primero hay que decidir el flujo de arranque de la GUI.

## Revisado sin problemas (2026-11-05)

La revisión del subsistema de fine-tuning verificó como correctos:

- el chequeo de `Host` (anti DNS-rebinding), el token CSRF en un header propio y la ausencia de CORS; el bind restringido a loopback; vLLM publicado en `127.0.0.1:8000` con `adapters:ro`;
- `dataset_path` con `resolve()`, que rechaza traversal, rutas absolutas y symlinks hacia afuera; las expresiones `SAFE_NAME`, `SAFE_MODEL` y `SAFE_DATASET`;
- subprocesos invocados siempre con listas argv, sin `shell=True`; el `chown` dentro de Docker con uid/gid enteros;
- el escape con `escapeHtml` en el frontend y los logs mostrados con `textContent`; el HF token no se persiste ni aparece en los snapshots;
- `vllm_export.py`: rechaza symlinks, es content-addressed, verifica hashes y hace rename atómico;
- `validate_adapter_base` compara el `root` y no el alias; load, unload y delete revierten vLLM si falla la allowlist;
- `train.py` no sobrescribe un adaptador existente y valida la máscara de assistant.

## Correcciones recientes relacionadas

Los siguientes problemas ya no se consideran errores conocidos porque cuentan con corrección y regresión.

### Corregidos el 2026-11-05

- **FT-01:** un lock entre procesos (`.runtime/gpu.lock`, `fcntl.flock`/`flock`) compartido por ambas GUIs y `scripts/train-adapter*.sh`; una segunda acción de GPU recibe 409 con un mensaje que nombra el lock.
- **FT-02:** `/api/shutdown` ya no deja huérfano un entrenamiento; el PGID del job se persiste en `.runtime/gpu-job.json` (0600) y se comprueba al arrancar.
- **FT-03:** la allowlist LoRA se reconcilia tras cada (re)arranque de vLLM (`trainer/runtime_adapters.py`); un adaptador de la allowlist que no está cargado se puede retirar.
- **FT-04:** reescribir `.env.local` ya no lo deja legible por otros usuarios (Python con `mkstemp` + `copymode`; fish con `umask 077`).
- **FT-05:** la cancelación escala SIGINT → SIGTERM (10 s) → SIGKILL (5 s) sobre el grupo de procesos, y el job termina `cancelled` aunque el stdout siga abierto.
- **FT-06:** el backend Axolotl usa el preset `attention-mlp` para Qwen3.5, lo registra en el manifest y aplica la misma normalización y auditoría de truncamiento que `train.py` (con `--allow-truncation`).
- **FT-07:** la GUI distingue `venvPresent` de `environmentVerified`; la insignia verde depende de una comprobación real persistida.
- **FT-08:** cancelar un entrenamiento (SIGINT o SIGTERM) borra el staging de esa corrida en ambos backends.
- **FT-09:** `mix_replay.py` valida ambos splits antes de escribir, los publica de forma atómica y borra una validación obsoleta.
- **FT-10:** cualquier error al parsear un dataset subido responde 422 en JSON y el temporal siempre se borra. En Python 3.14 `RecursionError` ya respondía 409, pero el temporal sí quedaba.
- **FT-11:** ambas GUIs validan todo el payload antes de cualquier efecto, incluido el contenedor root de `ensure_adapter_dir_writable`.
- **FT-13:** `--host ::1` sirve por IPv6.
- **GW-01:** `HOST` y `PORT` de `.env.local` llegan al servidor de desarrollo de Vite (`loadEnv`); las variables del proceso siguen teniendo prioridad.
- **GW-02:** `/api/health` informa la disponibilidad real de los embeddings sin cambiar su código de estado; la UI distingue *habilitados*, *disponibles* y *no disponibles*.
- **GW-03:** con RAG o CAG y ningún documento seleccionado, la UI pide seleccionar al menos uno.
- **DOC-01:** la documentación ya no presenta `all-linear` como valor por defecto.

### Anteriores

- una selección explícita `documentIds: []` ya no utiliza todos los documentos por accidente;
- RAG solo publica fuentes cuyos chunks realmente entraron al contexto;
- RAG sin candidatos ya no afirma en el `system` prompt que se recuperaron fragmentos relevantes;
- la carga y descarga dinámica de LoRA intenta revertir el estado de vLLM si falla la actualización de `LLM_ADAPTER_MODELS`, y reporta explícitamente si también falla el rollback;
- la API evita procesar un nuevo PDF cuando el workspace ya alcanzó el límite de documentos;
- el backend básico usa `attention-mlp` por defecto en lugar de `all-linear`, porque adaptar las puertas de estado de Qwen3.5 degeneraba en bucles de repetición (`docs/FINE_TUNING_DEBUG.md`);
- la GUI Axolotl reconstruye con Python 3.12 un entorno creado con Python 3.14 al pulsar **Preparar / actualizar entorno** (`docs/FINE_TUNING_AXOLOTL.md`).
