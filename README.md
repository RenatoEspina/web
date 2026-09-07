# LLM Bridge Chat

Aplicación local-first para conversar con un LLM alojado en el computador del usuario. El navegador llama únicamente al gateway propio; el gateway se comunica con el engine de inferencia y mantiene URLs y claves fuera del cliente.

El perfil principal del proyecto es:

- **Generación:** vLLM.
- **Embeddings:** Ollama en CPU con `qwen3-embedding:4b`.
- **Documentos:** PDF con RAG híbrido o CAG.
- **Fine-tuning:** SFT + QLoRA con adaptadores LoRA cargados dinámicamente en vLLM.
- **Exposición remota:** Cloudflare Tunnel apuntando al gateway, nunca directamente a vLLM/Ollama.

Las decisiones deliberadamente diferidas están registradas en [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md).

## Arquitectura

```text
Navegador
   │
   ▼
Gateway web :3000
   ├── /api/chat ───────────────► vLLM :8000
   ├── /api/documents
   │      ├── extracción PDF
   │      ├── chunking multipágina
   │      └── embeddings ───────► Ollama :11434
   └── RAG / CAG
```

La GUI administrativa de fine-tuning es un proceso separado que solo escucha en loopback:

```text
Navegador local ─► trainer/gui_server.py :3031
                      ├── datasets
                      ├── trainer QLoRA
                      ├── Docker/vLLM
                      └── load/unload LoRA
```

## Requisitos

- Node.js `>=22.13.0`
- Linux
- Docker + NVIDIA Container Toolkit para el flujo actual de vLLM
- `curl`, `flock` y GNU `timeout`
- Python para fine-tuning

## Inicio rápido

```bash
cp .env.example .env.local
npm ci
./comandos.fish vllm
npm run dev
```

Abre:

```text
http://127.0.0.1:3000
```

El comando `./comandos.fish vllm` inicia:

1. Ollama para embeddings;
2. descarga/verifica `qwen3-embedding:4b`;
3. inicia vLLM;
4. espera sus health checks.

La configuración principal correspondiente es:

```dotenv
LLM_PROVIDER=vllm
LLM_BASE_URL=http://127.0.0.1:8000
LLM_MODEL=Qwen/Qwen3.5-0.8B

EMBEDDING_ENABLED=true
EMBEDDING_PROVIDER=ollama
EMBEDDING_BASE_URL=http://127.0.0.1:11434
EMBEDDING_MODEL=qwen3-embedding:4b
```

## Comandos operativos

```fish
./comandos.fish vllm [modelo]
./comandos.fish embeddings [modelo]
./comandos.fish status
./comandos.fish stop
./comandos.fish down
```

El soporte para Ollama como engine generativo se conserva como alternativa, pero el flujo probado y prioritario del proyecto utiliza vLLM para generación y Ollama para embeddings.

## RAG

Al cargar un PDF:

```text
PDF
 ↓
extracción de texto
 ↓
normalización estructural
 ↓
chunks que pueden cruzar páginas
 ↓
embeddings por lote
 ↓
índice temporal en memoria
```

La normalización conserva saltos de línea y párrafos útiles para listas/tablas simples. Solo recompone palabras partidas por salto de línea en casos acotados, evitando borrar rangos o guiones con significado.

Durante una pregunta RAG se calculan dos rankings:

- señal léxica tipo TF-IDF;
- similitud coseno sobre embeddings.

Los candidatos léxicos positivos y los candidatos semánticos que superan `RAG_MIN_SEMANTIC_SCORE` se acotan antes de combinarse mediante **Reciprocal Rank Fusion (RRF)**. Esto evita mezclar directamente escalas incompatibles y reduce la posibilidad de que similitudes semánticas débiles entren al `topK` únicamente por tener una posición relativa.

Después de la fusión se toman los `topK` configurados y se construye el contexto respetando el presupuesto máximo. La API devuelve únicamente fuentes cuyos chunks realmente entraron al contexto enviado al LLM. Si ningún candidato semántico supera el umbral, el ranking vuelve a ser exclusivamente léxico y `embeddingUsed` se reporta como `false`.

Si el servicio de embeddings falla al subir un PDF, el documento no se rechaza: queda indexado para recuperación léxica y el fallo semántico se registra en logs.

El contenido documental se trata como datos. Si un PDF contiene literalmente las etiquetas `<documentos>` o `</documentos>`, el gateway las neutraliza antes de insertar el texto en el `system` prompt para impedir que el documento cierre los delimitadores usados por la aplicación.

### Selección de documentos

La selección es explícita:

- `documentIds` omitido: compatibilidad con clientes antiguos, usa los documentos disponibles;
- `documentIds: []`: no usa ningún documento;
- `documentIds: [id...]`: usa solo los seleccionados.

Las consultas a workspaces inexistentes no crean espacios vacíos. Solo una escritura documental crea un workspace, evitando que lecturas arbitrarias consuman el límite LRU o expulsen bibliotecas activas.

## CAG

CAG mantiene un contexto documental estable en orden de documento y lo reutiliza entre preguntas.

Con vLLM el Compose habilita:

```text
--enable-prefix-caching
```

por lo que el prefijo documental estable puede reutilizarse en inferencias posteriores.

Si el contexto excede `CAG_MAX_CONTEXT_CHARACTERS`, se trunca y la respuesta lo reporta mediante `contextTruncated`. CAG no cambia silenciosamente a RAG.

## PDF y límites

Solo se soportan PDF con texto seleccionable. Los documentos escaneados requieren OCR, que queda como evolución posterior.

Los defaults se encuentran en `.env.example`:

```dotenv
RAG_MAX_PDF_BYTES=10485760
RAG_MAX_PDF_PAGES=100
RAG_MAX_DOCUMENT_CHARACTERS=400000
RAG_MAX_DOCUMENTS=10
RAG_MAX_WORKSPACES=16
RAG_CHUNK_SIZE=1200
RAG_CHUNK_OVERLAP=180
RAG_TOP_K=4
RAG_SEMANTIC_WEIGHT=70
RAG_LEXICAL_WEIGHT=30
RAG_MIN_SEMANTIC_SCORE=0.2
RAG_MAX_CONTEXT_CHARACTERS=7000
CAG_MAX_CONTEXT_CHARACTERS=8000
```

`RAG_MIN_SEMANTIC_SCORE=0.2` corresponde al perfil actual con `qwen3-embedding:4b`. Si se cambia el modelo de embeddings, este umbral debe volver a validarse junto con el resto del índice.

El índice es temporal y vive en memoria. Reiniciar el gateway elimina documentos, chunks, embeddings y caché CAG.

## Docker

El perfil actual de vLLM está orientado al entorno de investigación y usa, por defecto:

```text
vllm/vllm-openai:v0.24.0
GPU memory utilization: 0.80
max model len: 4096
max num seqs: 1
max num batched tokens: 2048
enforce eager: activo
prefix caching: activo
LoRA: activo
```

Ollama se fuerza a CPU en Compose para reservar la GPU al engine generativo.

Los servicios quedan publicados únicamente en loopback:

```text
vLLM   127.0.0.1:8000
Ollama 127.0.0.1:11434
```

## Fine-tuning QLoRA

El flujo administrativo oficial está separado del chat público.

Inicia la GUI local con:

```bash
./fine-tune-gui
```

La GUI permite:

- preparar `trainer/.venv`;
- comprobar CUDA y NF4;
- subir/validar datasets;
- entrenar SFT + QLoRA;
- iniciar/detener vLLM;
- cargar/descargar adaptadores LoRA dinámicamente.

La comprobación del entorno está centralizada en:

```text
trainer/check_environment.py
```

El validador canónico de datasets está en:

```text
trainer/dataset_validation.py
```

El entrenador publica primero en un directorio temporal y solo crea `adapters/<nombre>/` cuando pesos, tokenizer y manifiesto se guardaron correctamente. Un entrenamiento fallido no deja un adaptador parcial bloqueando el nombre.

La carga dinámica usa los endpoints locales de vLLM. La GUI mantiene sincronizada la allowlist `LLM_ADAPTER_MODELS` al cargar y descargar adaptadores.

La guía completa está en:

- [`docs/FINE_TUNING.md`](docs/FINE_TUNING.md)
- [`docs/FINE_TUNING_GUI.md`](docs/FINE_TUNING_GUI.md)

### CLI

```fish
./comandos.fish fine-tune-setup
./comandos.fish fine-tune-check
./comandos.fish fine-tune-validate dataset.jsonl
./comandos.fish fine-tune-train dataset.jsonl qwen-dominio-v1 --rank 16 --epochs 3
./comandos.fish fine-tune-list
./comandos.fish fine-tune-evaluate evaluacion.jsonl MODELO salida.json
```

El antiguo flujo manual basado en editar `--lora-modules` fue retirado del camino normal; la carga/descarga runtime se administra desde la GUI local.

## Seguridad y exposición pública

No expongas directamente los puertos `8000` ni `11434`.

Para usar Cloudflare Tunnel:

1. configura una clave larga en `.env.local`:

   ```dotenv
   APP_TOKEN=una-clave-larga-y-aleatoria
   ```

2. inicia la aplicación:

   ```bash
   npm run dev
   ```

3. abre el túnel hacia el gateway:

   ```bash
   npm run tunnel
   ```

El destino debe ser siempre:

```text
http://127.0.0.1:3000
```

La GUI de fine-tuning no se publica mediante el túnel y permanece en loopback.

## Verificación

El flujo completo de comprobación local es:

```bash
npm run verify
```

Incluye:

- lint;
- typecheck;
- build;
- tests Node;
- tests Python de la GUI administrativa.

GitHub Actions ejecuta la misma verificación para los pull requests. Las comprobaciones que requieren GPU/CUDA siguen siendo locales y se ejecutan mediante `trainer/check_environment.py`.

## Estado y limitaciones

Las principales limitaciones deliberadas se documentan en [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md), incluyendo:

- cambio del modelo de embeddings con un índice ya creado;
- engines alternativos a vLLM;
- configuración ajustada al hardware de investigación;
- persistencia documental;
- presupuesto de contexto basado todavía en caracteres.
