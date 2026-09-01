# LLM Bridge Chat

Aplicación local-first para conversar con un modelo alojado en el computador
del usuario. El navegador llama únicamente al backend propio (`/api/chat`); el
backend selecciona un adaptador para vLLM u Ollama y mantiene sus URLs y claves
fuera del navegador.

La interfaz está pensada para texto: no sube archivos, no guarda conversaciones
en una base de datos y conserva el historial solo en la sesión del navegador.

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
LLM_MODEL=meta-llama/Llama-3.2-1B-Instruct
```

Para Ollama:

```dotenv
LLM_PROVIDER=ollama
LLM_BASE_URL=http://127.0.0.1:11434
LLM_MODEL=llama3.2:1b-instruct-fp16
```

Si tu modelo tiene otro tag, cambia únicamente `LLM_MODEL`.

## Servidores locales con Docker

El archivo `docker-compose.yml` incluye vLLM 0.24.0 y Ollama. Ambos servicios
quedan publicados solamente en `127.0.0.1`; el navegador y el túnel deben
conectarse al gateway en el puerto `3000`, no directamente al puerto `8000` ni
al `11434`.

El script `comandos.fish` evita que vLLM y Ollama ocupen la GPU al mismo tiempo:

```bash
chmod +x comandos.fish

# vLLM; solicita HF_TOKEN si no está definido en la terminal
./comandos.fish vllm

# Ollama; el modelo es opcional y tiene un valor predeterminado
./comandos.fish ollama llama3.2:1b-instruct-fp16

./comandos.fish status
./comandos.fish stop
```

Después de iniciar vLLM, usa en `.env.local`:

```dotenv
LLM_PROVIDER=vllm
LLM_BASE_URL=http://127.0.0.1:8000
LLM_MODEL=meta-llama/Llama-3.2-1B-Instruct
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
| `APP_TOKEN` | Protege `/api/chat` y `/api/health`; recomendado con túnel. |
| `HOST` / `PORT` | Escucha local de la interfaz. Mantén `HOST=127.0.0.1`. |

## Endpoints internos

- `GET /api/config`: devuelve proveedor, modelo y si se requiere token; nunca
  devuelve la URL privada ni las claves.
- `GET /api/health`: comprueba el proveedor configurado.
- `POST /api/chat`: recibe `{ "message": "...", "history": [] }` y devuelve
  `{ "message": "..." }`.

El gateway limita cada mensaje a 12.000 caracteres y el historial a los 20
mensajes más recientes. Las respuestas no se almacenan en el servidor.

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
