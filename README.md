# LLM Bridge Chat

Aplicación local-first para conversar con un modelo alojado en el computador
del usuario. El navegador llama únicamente al backend propio (`/api/chat`); el
backend selecciona un adaptador para vLLM u Ollama y mantiene sus URLs y claves
fuera del navegador.

La interfaz está pensada para texto y permite cargar PDF con texto seleccionable
para consultarlos mediante RAG o CAG. El índice documental y sus embeddings
viven en la memoria del gateway y se separan por un identificador local del
navegador; no se guardan los PDF en una base de datos.

## Requisitos

- Node.js `>=22.13.0`
- Linux con `flock`, `curl` y GNU `timeout`
- vLLM 0.24.0 escuchando en `127.0.0.1:8000`, u Ollama escuchando en
  `127.0.0.1:11434`

## Arquitectura

```text
navegador → /api/chat → adaptador del gateway → vLLM / Ollama local
```

El adaptador OpenAI-compatible cubre vLLM y permite sustituirlo por otros
servidores que expongan `/v1/chat/completions`. El adaptador de Ollama usa
`/api/chat`. Para cambiar de proveedor se modifica el `.env.local`; no se toca
la interfaz.

## RAG y CAG para PDF

La biblioteca documental se encuentra en el mismo gateway y se accede desde la
interfaz con `Agregar PDF`. Cada PDF se procesa en el computador que ejecuta
LLM Bridge:

```text
navegador → POST /api/documents → extracción → fragmentación entre páginas → embeddings → índice temporal
chat      → POST /api/chat      → RAG/CAG → contexto → vLLM u Ollama
```

### RAG

RAG (Retrieval-Augmented Generation) genera un embedding por fragmento al
cargar el PDF y otro por cada pregunta. Después combina similitud coseno
semántica con coincidencia léxica tipo TF-IDF. Así puede recuperar una idea
expresada con palabras distintas y seguir encontrando nombres, cifras o
términos exactos. Las respuestas muestran el nombre del PDF y el rango de
páginas de los fragmentos usados.

El servicio de embeddings es independiente del modelo generativo. La
configuración predeterminada usa Ollama con `qwen3-embedding:4b`, un modelo
multilingüe adecuado para preguntas en español. Debes descargarlo una vez:

```bash
ollama pull qwen3-embedding:4b
```

También se puede usar un servidor vLLM separado que exponga
`/v1/embeddings`; en ese caso configura `EMBEDDING_PROVIDER=vllm`, su URL y el
modelo de embeddings. El vLLM que sirve `Qwen/Qwen3.5-0.8B` para chat no se
convierte automáticamente en un modelo de embeddings.

La integración con Ollama usa el endpoint actual `/api/embed` y puede enviar
varios fragmentos por lote. Si `EMBEDDING_ENABLED=false`, el gateway omite la
recuperación semántica y usa únicamente el recuperador léxico, aunque haya un
valor obsoleto o inválido en `EMBEDDING_PROVIDER`.

### CAG

CAG (Cache-Augmented Generation) prepara y mantiene en memoria el contexto de
los PDF seleccionados para reutilizarlo en preguntas posteriores. Si el
contexto completo cabe en el límite, conserva el comportamiento CAG original y
no necesita volver a consultar el embedding en cada pregunta. Si los PDF superan ese límite, CAG mantiene el orden documental y marca el contexto como truncado; no convierte silenciosamente la operación en RAG. Para recuperar selectivamente una ventana relevante se debe usar RAG. La caché de contexto del gateway se combina con la caché automática de prefijos de vLLM, que permite reutilizar el prefijo documental estable entre preguntas.

### Límites y alcance actual

- Solo se aceptan PDF con texto seleccionable. Un PDF escaneado necesita OCR,
  que queda como mejora posterior.
- Por defecto se admiten 10 MB, 100 páginas, 10 documentos por espacio,
  400.000 caracteres por documento y 16 espacios activos en memoria.
- El índice se pierde al reiniciar el gateway o el proceso Worker. Hay un
  espacio independiente por navegador para que dos usuarios del túnel no
  mezclen sus bibliotecas por accidente.
- Los embeddings y los PDF se mantienen en memoria y se pierden al reiniciar el
  gateway. Para una evolución con muchos documentos conviene reemplazar el
  almacén temporal por D1/R2 más un índice vectorial persistente.

Los límites se pueden ajustar en `.env.local`; las variables disponibles están
documentadas en `.env.example`.

## Ejecución local

Desde la raíz del proyecto:

```bash
cp .env.example .env.local
npm run dev
```

Abre `http://127.0.0.1:3000`. El archivo `.env.local` está ignorado por Git.

La configuración que coincide con tu vLLM actual es:

```dotenv
LLM_PROVIDER=vllm
LLM_BASE_URL=http://127.0.0.1:8000
LLM_MODEL=Qwen/Qwen3.5-0.8B
```

Para Ollama:

```dotenv
LLM_PROVIDER=ollama
LLM_BASE_URL=http://127.0.0.1:11434
LLM_MODEL=llama3.2:1b-instruct-fp16
```

Si tu modelo tiene otro tag, cambia únicamente `LLM_MODEL`.

### Configurar embeddings

Con el `.env.example` predeterminado, el modelo generativo puede seguir en
vLLM y los embeddings se generan mediante Ollama:

```dotenv
LLM_PROVIDER=vllm
LLM_BASE_URL=http://127.0.0.1:8000
EMBEDDING_PROVIDER=ollama
EMBEDDING_BASE_URL=http://127.0.0.1:11434
EMBEDDING_MODEL=qwen3-embedding:4b
```

Si usas Ollama para ambos servicios, deja `LLM_PROVIDER=ollama` y conserva la
misma configuración de embeddings. Si prefieres vLLM para ambos, levanta un
servidor de embeddings independiente y usa, por ejemplo:

```dotenv
EMBEDDING_PROVIDER=vllm
EMBEDDING_BASE_URL=http://127.0.0.1:8001
EMBEDDING_MODEL=intfloat/multilingual-e5-small
EMBEDDING_QUERY_PREFIX="query: "
EMBEDDING_DOCUMENT_PREFIX="passage: "
```

El modelo usado para indexar debe ser exactamente el mismo que se usa para
consultar. Si solo quieres conservar el recuperador léxico, define
`EMBEDDING_ENABLED=false`.

## Servidores locales con Docker

El archivo `docker-compose.yml` incluye vLLM 0.24.0 y Ollama. Ambos servicios
quedan publicados solamente en `127.0.0.1`; el navegador y el túnel deben
conectarse al gateway en el puerto `3000`, no directamente al puerto `8000` ni
al `11434`. Para habilitar los embeddings predeterminados descarga el modelo
en el contenedor una vez:

```bash
docker compose exec ollama ollama pull qwen3-embedding:4b
```

El perfil predeterminado usa `Qwen/Qwen3.5-0.8B` en modo solo texto. Para la
RTX 3060 Laptop de 6 GB se reservan como máximo aproximadamente el 60 % de la
VRAM para vLLM, con contexto de 2048 tokens, una sola secuencia y un lote de
1024 tokens. `--enforce-eager` reduce el consumo de memoria de los CUDA graphs.
El flag `--language-model-only` evita cargar el componente visual de Qwen3.5,
porque esta interfaz únicamente envía y recibe texto. También se fija
`enable_thinking=false` para que la respuesta visible no pierda tokens en un
bloque de razonamiento interno.

El script `comandos.fish` inicia Ollama para embeddings con `qwen3-embedding:4b` y la configuración de Compose fuerza Ollama a CPU para reservar la GPU a vLLM:

```bash
chmod +x comandos.fish

# vLLM; solicita HF_TOKEN si no está definido en la terminal
./comandos.fish vllm

# Opcional: probar otro tamaño. Debes poner el mismo nombre en .env.local.
# ./comandos.fish vllm Qwen/Qwen3.5-2B

# Ollama; el modelo es opcional y tiene un valor predeterminado
./comandos.fish ollama llama3.2:1b-instruct-fp16

./comandos.fish status
./comandos.fish stop
```

Después de iniciar vLLM, usa en `.env.local`:

```dotenv
LLM_PROVIDER=vllm
LLM_BASE_URL=http://127.0.0.1:8000
LLM_MODEL=Qwen/Qwen3.5-0.8B
```

Después de iniciar Ollama, usa:

```dotenv
LLM_PROVIDER=ollama
LLM_BASE_URL=http://127.0.0.1:11434
LLM_MODEL=llama3.2:1b-instruct-fp16
```

Los volúmenes `hf-cache` y `ollama-data` conservan los modelos aunque se
detengan los contenedores. La primera ejecución puede tardar bastante porque
debe descargar el modelo. Si ya existían contenedores llamados `vllm` u
`ollama` creados desde otro proyecto, detén ese proyecto antes de ejecutar este
Compose para evitar conflictos de puertos.

## Exposición pública gratuita

No conviene publicar directamente el puerto `8000`: expondría la API de
inferencia sin la protección del gateway. La opción gratuita más simple para
este MVP es un Cloudflare Quick Tunnel, que crea una URL HTTPS temporal sin
abrir puertos del router.

1. Instala `cloudflared` usando el paquete de tu distribución o la documentación
   oficial de Cloudflare.
2. Define una clave larga en `.env.local`:

   ```dotenv
   APP_TOKEN=una-clave-larga-y-dificil-de-adivinar
   HOST=127.0.0.1
   ```

3. Ejecuta la aplicación:

   ```bash
   npm run dev
   ```

4. En otra terminal ejecuta:

   ```bash
   npm run tunnel
   ```

Cloudflare mostrará una URL `https://...trycloudflare.com`. Comparte esa URL
junto con la clave de `APP_TOKEN`. La página pedirá la clave y la conservará
solo en `sessionStorage`.

El Quick Tunnel es temporal y su URL cambia al reiniciarlo. Para una URL fija
se puede usar un túnel nombrado con un subdominio propio; esa evolución no
cambia el gateway ni los adaptadores.

## Vercel más túnel hacia el gateway local

La interfaz también puede publicarse en Vercel manteniendo el gateway y el
modelo en el computador:

```text
Vercel → URL pública del túnel → gateway local:3000 → vLLM/Ollama local
```

El proyecto incluye `vercel.json` y `npm run build:vercel` para el despliegue
con Next.js en Vercel. La variable `NEXT_PUBLIC_GATEWAY_URL` se inserta en el
código del navegador, por lo que solo debe contener la URL del túnel; no es un
lugar para guardar tokens o claves.

### Configuración del gateway local

Después de obtener la URL del proyecto en Vercel, añade en `.env.local` el
origen exacto de la página:

```dotenv
APP_CORS_ORIGIN=https://tu-proyecto.vercel.app
```

Reinicia `npm run dev` después de cambiar esta variable.

### Configuración de Vercel

En las variables de entorno del proyecto Vercel define:

```text
NEXT_PUBLIC_GATEWAY_URL=https://URL-ACTUAL-DEL-TUNEL
```

No subas `.env.local` ni configures `HF_TOKEN`, `LLM_API_KEY` o `APP_TOKEN` en
Vercel. El `APP_TOKEN` permanece en el gateway local y se introduce en la
interfaz.

Un Quick Tunnel gratuito puede cambiar de URL al reiniciarse. Si cambia, hay
que actualizar `NEXT_PUBLIC_GATEWAY_URL` en Vercel y volver a desplegar. Más
adelante conviene usar un túnel con hostname estable.

### Subdominio propio mediante Cloudflare Tunnel

Una vez que `npm run dev` responde en `127.0.0.1:3000`, crea un túnel nombrado
con `cloudflared` y configura una entrada que apunte a:

```text
http://127.0.0.1:3000
```

No apuntes el túnel a `8000` o `11434`. Esos puertos son únicamente para la
comunicación privada entre el gateway y el proveedor.

Ejemplo de `~/.cloudflared/config.yml`:

```yaml
tunnel: UUID_DEL_TUNEL
credentials-file: /home/USUARIO/.cloudflared/UUID_DEL_TUNEL.json

ingress:
  - hostname: chat.tudominio.com
    service: http://127.0.0.1:3000
  - service: http_status:404
```

Luego asocia el DNS y ejecuta el túnel:

```bash
cloudflared tunnel route dns llm-bridge chat.tudominio.com
npm run tunnel:named
```

El computador debe permanecer encendido y con el gateway, el proveedor y
`cloudflared` ejecutándose. Para una URL fija no uses `npm run tunnel`, porque
ese script inicia un Quick Tunnel temporal.

## Producción local

```bash
npm run build
HOST=127.0.0.1 PORT=3000 npm start
```

Después, `npm run tunnel` puede apuntar al mismo puerto `3000`.

## Variables de entorno

| Variable | Uso |
| --- | --- |
| `LLM_PROVIDER` | `vllm` u `ollama`. |
| `LLM_BASE_URL` | URL raíz privada del proveedor. |
| `LLM_MODEL` | Nombre del modelo enviado al proveedor. |
| `LLM_API_KEY` | Clave opcional para vLLM/OpenAI-compatible. |
| `LLM_MAX_TOKENS` | Límite de tokens generados. |
| `LLM_TEMPERATURE` | Temperatura común a ambos adaptadores. |
| `LLM_TIMEOUT_MS` | Tiempo máximo de espera de una respuesta. |
| `EMBEDDING_ENABLED` | Activa o desactiva la generación semántica. |
| `EMBEDDING_PROVIDER` | `ollama` o `vllm` para el servicio de embeddings. |
| `EMBEDDING_BASE_URL` | URL raíz del servicio de embeddings. |
| `EMBEDDING_MODEL` | Modelo usado para indexar y consultar. |
| `EMBEDDING_API_KEY` | Clave opcional del servicio de embeddings. |
| `EMBEDDING_BATCH_SIZE` | Cantidad de fragmentos enviados por lote al indexar. |
| `EMBEDDING_TIMEOUT_MS` | Tiempo máximo de cada lote de embeddings. |
| `EMBEDDING_QUERY_PREFIX` | Prefijo opcional para embeddings de preguntas. |
| `EMBEDDING_DOCUMENT_PREFIX` | Prefijo opcional para embeddings de fragmentos. |
| `RAG_SEMANTIC_WEIGHT` | Peso porcentual de la similitud semántica; por defecto 70. |
| `RAG_LEXICAL_WEIGHT` | Peso porcentual de la coincidencia léxica; por defecto 30. |
| `APP_TOKEN` | Protege `/api/chat` y `/api/health`; recomendado con túnel. |
| `NEXT_PUBLIC_GATEWAY_URL` | URL pública del gateway para la interfaz desplegada en Vercel. |
| `APP_CORS_ORIGIN` | Origen de Vercel autorizado para llamar al gateway. |
| `HOST` / `PORT` | Escucha local de la interfaz. Mantén `HOST=127.0.0.1`. |

## Endpoints internos

- `GET /api/config`: devuelve proveedor, modelo y estado/modelo de embeddings;
  nunca devuelve las URL privadas ni las claves.
- `GET /api/health`: comprueba el proveedor configurado.
- `POST /api/chat`: recibe `{ "message": "...", "history": [] }` y devuelve
  `{ "message": "..." }`.

El gateway limita cada mensaje a 12.000 caracteres y el historial a los 20
mensajes más recientes. Cuando se usa RAG o CAG, el historial se reduce además
para reservar espacio al contexto documental. Las respuestas no se almacenan
en el servidor.

### Endpoints documentales

- `GET /api/documents`: lista los PDF indexados en el espacio del navegador.
- `POST /api/documents`: recibe un campo multipart llamado `file` y procesa un
  PDF.
- `DELETE /api/documents?id=...`: elimina un PDF del espacio actual.

Los endpoints documentales requieren el mismo `APP_TOKEN` que `/api/chat` y
usan el encabezado `x-workspace-id`, generado automáticamente por la interfaz.

## Verificación

```bash
npm run typecheck
npm run lint
npm test
```

## Sites

El proyecto conserva compatibilidad con el ciclo de vida de Sites. La parte
principal para el uso local es el gateway ejecutado en el computador; una
publicación de Sites no puede alcanzar automáticamente el `localhost` del
computador del usuario. Para el flujo público de este MVP se debe usar el
Quick Tunnel sobre la aplicación local.
