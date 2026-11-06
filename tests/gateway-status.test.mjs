import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url)).replace(/\/$/, "");
const vite = await createServer({
  appType: "custom",
  configFile: false,
  root,
  resolve: { alias: { "@": root } },
  server: { middlewareMode: true },
});

after(async () => vite.close());

const MANAGED_ENV = [
  "HOST",
  "PORT",
  "APP_TOKEN",
  "LLM_PROVIDER",
  "LLM_BASE_URL",
  "LLM_MODEL",
  "EMBEDDING_ENABLED",
  "EMBEDDING_PROVIDER",
  "EMBEDDING_BASE_URL",
  "EMBEDDING_MODEL",
];

async function withEnv(values, run) {
  const previous = Object.fromEntries(MANAGED_ENV.map((key) => [key, process.env[key]]));
  const previousFetch = globalThis.fetch;
  for (const key of MANAGED_ENV) delete process.env[key];
  Object.assign(process.env, values);
  try {
    return await run();
  } finally {
    globalThis.fetch = previousFetch;
    for (const key of MANAGED_ENV) {
      if (previous[key] === undefined) delete process.env[key];
      else process.env[key] = previous[key];
    }
  }
}

function stubFetch(routes) {
  const calls = [];
  globalThis.fetch = async (input) => {
    const url = typeof input === "string" ? input : input.url;
    calls.push(url);
    const route = routes[url];
    if (!route) throw new TypeError(`fetch failed: ${url}`);
    return new Response(JSON.stringify(route.body ?? {}), {
      status: route.status ?? 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  return calls;
}

async function envDir(contents) {
  const dir = await mkdtemp(path.join(tmpdir(), "llm-bridge-env-"));
  await writeFile(path.join(dir, ".env.local"), contents);
  return dir;
}

test("HOST y PORT de .env.local llegan al servidor de desarrollo", async () => {
  const { resolveDevServerAddress } = await vite.ssrLoadModule("/build/dev-server-address.ts");
  const dir = await envDir("HOST=0.0.0.0\nPORT=3999\n");
  await withEnv({}, async () => {
    assert.deepEqual(resolveDevServerAddress("development", dir), { host: "0.0.0.0", port: 3999 });
  });
});

test("las variables del proceso siguen teniendo prioridad sobre .env.local", async () => {
  const { resolveDevServerAddress } = await vite.ssrLoadModule("/build/dev-server-address.ts");
  const dir = await envDir("HOST=0.0.0.0\nPORT=3999\n");
  await withEnv({ HOST: "127.0.0.2", PORT: "4001" }, async () => {
    assert.deepEqual(resolveDevServerAddress("development", dir), { host: "127.0.0.2", port: 4001 });
  });
});

test("sin configuración el servidor de desarrollo conserva 127.0.0.1:3000", async () => {
  const { resolveDevServerAddress } = await vite.ssrLoadModule("/build/dev-server-address.ts");
  const dir = await mkdtemp(path.join(tmpdir(), "llm-bridge-env-"));
  await withEnv({}, async () => {
    assert.deepEqual(resolveDevServerAddress("development", dir), { host: "127.0.0.1", port: 3000 });
  });
});

test("HOST= y PORT= vacíos en .env.local conservan 127.0.0.1:3000", async () => {
  const { resolveDevServerAddress } = await vite.ssrLoadModule("/build/dev-server-address.ts");
  const dir = await envDir("HOST=\nPORT=\n");
  await withEnv({}, async () => {
    assert.deepEqual(resolveDevServerAddress("development", dir), { host: "127.0.0.1", port: 3000 });
  });
});

test("HOST y PORT con solo espacios conservan 127.0.0.1:3000", async () => {
  const { resolveDevServerAddress } = await vite.ssrLoadModule("/build/dev-server-address.ts");
  const dir = await envDir("HOST=\"  \"\nPORT=\" \"\n");
  await withEnv({}, async () => {
    assert.deepEqual(resolveDevServerAddress("development", dir), { host: "127.0.0.1", port: 3000 });
  });
});

test("vite.config.ts resuelve host y puerto con los archivos de entorno", async () => {
  const { readFile } = await import("node:fs/promises");
  const config = await readFile(path.join(root, "vite.config.ts"), "utf8");
  assert.match(config, /resolveDevServerAddress\(mode, process\.cwd\(\)\)/);
  assert.doesNotMatch(config, /host: process\.env\.HOST/);
});

test("checkEmbeddings informa disponible cuando Ollama lista el modelo", async () => {
  await withEnv({ EMBEDDING_BASE_URL: "http://embed.test", EMBEDDING_MODEL: "qwen3-embedding:4b" }, async () => {
    const calls = stubFetch({
      "http://embed.test/api/tags": { body: { models: [{ name: "qwen3-embedding:4b", model: "qwen3-embedding:4b" }] } },
    });
    const { checkEmbeddings } = await vite.ssrLoadModule("/lib/embeddings/index.ts");
    assert.deepEqual(await checkEmbeddings(AbortSignal.timeout(1_000)), { enabled: true, available: true });
    assert.deepEqual(calls, ["http://embed.test/api/tags"]);
  });
});

test("checkEmbeddings informa no disponible si Ollama no tiene el modelo", async () => {
  await withEnv({ EMBEDDING_BASE_URL: "http://embed.test", EMBEDDING_MODEL: "qwen3-embedding:4b" }, async () => {
    stubFetch({ "http://embed.test/api/tags": { body: { models: [{ name: "otro:latest", model: "otro:latest" }] } } });
    const { checkEmbeddings } = await vite.ssrLoadModule("/lib/embeddings/index.ts");
    assert.deepEqual(await checkEmbeddings(AbortSignal.timeout(1_000)), {
      enabled: true,
      available: false,
      error: "ollama no tiene el modelo qwen3-embedding:4b.",
    });
  });
});

test("checkEmbeddings informa no disponible si el servicio no responde", async () => {
  await withEnv({ EMBEDDING_BASE_URL: "http://embed.test" }, async () => {
    stubFetch({});
    const { checkEmbeddings } = await vite.ssrLoadModule("/lib/embeddings/index.ts");
    assert.deepEqual(await checkEmbeddings(AbortSignal.timeout(1_000)), {
      enabled: true,
      available: false,
      error: "ollama no responde.",
    });
  });
});

test("checkEmbeddings acepta un modelo vLLM listado en /v1/models", async () => {
  await withEnv({
    EMBEDDING_PROVIDER: "vllm",
    EMBEDDING_BASE_URL: "http://embed.test",
    EMBEDDING_MODEL: "Qwen/Qwen3-Embedding-0.6B",
  }, async () => {
    stubFetch({ "http://embed.test/v1/models": { body: { data: [{ id: "Qwen/Qwen3-Embedding-0.6B" }] } } });
    const { checkEmbeddings } = await vite.ssrLoadModule("/lib/embeddings/index.ts");
    assert.deepEqual(await checkEmbeddings(AbortSignal.timeout(1_000)), { enabled: true, available: true });
  });
});

test("checkEmbeddings no consulta la red con embeddings desactivados", async () => {
  await withEnv({ EMBEDDING_ENABLED: "false" }, async () => {
    const calls = stubFetch({});
    const { checkEmbeddings } = await vite.ssrLoadModule("/lib/embeddings/index.ts");
    assert.deepEqual(await checkEmbeddings(AbortSignal.timeout(1_000)), { enabled: false, available: false });
    assert.deepEqual(calls, []);
  });
});

test("/api/health agrega el estado de embeddings sin cambiar el código del proveedor", async () => {
  await withEnv({
    LLM_PROVIDER: "vllm",
    LLM_BASE_URL: "http://llm.test",
    LLM_MODEL: "Qwen/Qwen3.5-0.8B",
    EMBEDDING_BASE_URL: "http://embed.test",
  }, async () => {
    stubFetch({ "http://llm.test/health": { body: {} } });
    const { GET } = await vite.ssrLoadModule("/app/api/health/route.ts");
    const response = await GET(new Request("http://localhost/api/health"));
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.ok, true);
    assert.equal(body.provider, "vllm");
    assert.deepEqual(body.embedding, { enabled: true, available: false, error: "ollama no responde." });
  });
});

test("/api/health sigue respondiendo 503 cuando el proveedor generativo falla", async () => {
  await withEnv({ LLM_PROVIDER: "vllm", LLM_BASE_URL: "http://llm.test", EMBEDDING_ENABLED: "false" }, async () => {
    stubFetch({ "http://llm.test/health": { status: 500, body: {} } });
    const { GET } = await vite.ssrLoadModule("/app/api/health/route.ts");
    const response = await GET(new Request("http://localhost/api/health"));
    assert.equal(response.status, 503);
    const body = await response.json();
    assert.equal(body.ok, false);
    assert.deepEqual(body.embedding, { enabled: false, available: false });
  });
});

test("la etiqueta distingue embeddings habilitados de disponibles", async () => {
  const { embeddingStatusLabel } = await vite.ssrLoadModule("/lib/ui/knowledge-status.ts");
  assert.equal(embeddingStatusLabel(false, null), "embeddings desactivados");
  assert.equal(embeddingStatusLabel(true, null), "embeddings habilitados");
  assert.equal(embeddingStatusLabel(true, true), "embeddings disponibles");
  assert.equal(embeddingStatusLabel(true, false), "embeddings no disponibles");
});

test("RAG o CAG sin documentos seleccionados piden seleccionar uno", async () => {
  const { knowledgeHint } = await vite.ssrLoadModule("/lib/ui/knowledge-status.ts");
  assert.equal(
    knowledgeHint("rag", 0),
    "Selecciona al menos un documento para usar RAG. Mientras tanto, el chat responderá sin consultar los PDF.",
  );
  assert.equal(
    knowledgeHint("cag", 0),
    "Selecciona al menos un documento para usar CAG. Mientras tanto, el chat responderá sin consultar los PDF.",
  );
});

test("las pistas de modo existentes se conservan con documentos seleccionados", async () => {
  const { knowledgeHint } = await vite.ssrLoadModule("/lib/ui/knowledge-status.ts");
  assert.equal(
    knowledgeHint("rag", 1),
    "RAG combina embeddings semánticos y coincidencia léxica para recuperar los fragmentos más relacionados.",
  );
  assert.equal(
    knowledgeHint("cag", 2),
    "CAG reutiliza el contexto documental en orden y lo trunca si supera el límite configurado.",
  );
  assert.equal(knowledgeHint("none", 0), "El chat responderá sin consultar los PDF.");
});

test("page.tsx usa la etiqueta y la pista centralizadas", async () => {
  const { readFile } = await import("node:fs/promises");
  const page = await readFile(path.join(root, "app/page.tsx"), "utf8");
  assert.match(page, /embeddingStatusLabel\(/);
  assert.match(page, /knowledgeHint\(knowledgeMode, selectedDocumentIds\.length\)/);
  assert.doesNotMatch(page, /embeddings \{config\?\.embedding\.enabled \? "activos"/);
});
