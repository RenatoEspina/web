import assert from "node:assert/strict";
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

function restoreEnv(name, value) {
  if (value === undefined) delete process.env[name];
  else process.env[name] = value;
}

test("chat reporta mode=none cuando no llegó a usar contexto documental", async () => {
  const previous = {
    provider: process.env.LLM_PROVIDER,
    baseUrl: process.env.LLM_BASE_URL,
    model: process.env.LLM_MODEL,
    token: process.env.APP_TOKEN,
  };
  const originalFetch = globalThis.fetch;
  const workspaceId = crypto.randomUUID();

  process.env.LLM_PROVIDER = "vllm";
  process.env.LLM_BASE_URL = "http://127.0.0.1:8000";
  process.env.LLM_MODEL = "Qwen/Qwen3.5-0.8B";
  delete process.env.APP_TOKEN;
  globalThis.fetch = async () => new Response(JSON.stringify({
    choices: [{ message: { content: "Respuesta sin documentos" } }],
  }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });

  try {
    const { POST } = await vite.ssrLoadModule("/app/api/chat/route.ts");
    const response = await POST(new Request("http://localhost/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-workspace-id": workspaceId,
      },
      body: JSON.stringify({
        message: "Hola",
        mode: "rag",
        documentIds: [],
      }),
    }));

    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.mode, "none");
    assert.deepEqual(body.sources, []);
    assert.equal(body.embeddingUsed, false);
  } finally {
    globalThis.fetch = originalFetch;
    restoreEnv("LLM_PROVIDER", previous.provider);
    restoreEnv("LLM_BASE_URL", previous.baseUrl);
    restoreEnv("LLM_MODEL", previous.model);
    restoreEnv("APP_TOKEN", previous.token);
  }
});

test("RAG no publica una fuente nueva si el presupuesto solo alcanza para una cabecera parcial", async () => {
  const previousLimit = process.env.RAG_MAX_CONTEXT_CHARACTERS;
  const previousEmbeddings = process.env.EMBEDDING_ENABLED;
  const workspaceId = crypto.randomUUID();
  const documentId = crypto.randomUUID();

  process.env.RAG_MAX_CONTEXT_CHARACTERS = "1000";
  process.env.EMBEDDING_ENABLED = "false";

  try {
    const { addDocument } = await vite.ssrLoadModule("/lib/documents/store.ts");
    const { buildKnowledgeContext } = await vite.ssrLoadModule("/lib/documents/retrieval.ts");

    addDocument(workspaceId, {
      id: documentId,
      name: "presupuesto.pdf",
      sizeBytes: 2000,
      pages: 2,
      characters: 2000,
      createdAt: Date.now(),
      chunks: [
        {
          id: `${documentId}-0`,
          documentId,
          documentName: "presupuesto.pdf",
          page: 1,
          index: 0,
          text: `objetivo ${"a".repeat(900)}`,
        },
        {
          id: `${documentId}-1`,
          documentId,
          documentName: "presupuesto.pdf",
          page: 2,
          index: 1,
          text: `objetivo ${"b".repeat(900)}`,
        },
      ],
    });

    const rag = await buildKnowledgeContext(workspaceId, "rag", "objetivo", [documentId]);
    assert.equal(rag?.sources.length, 1);
    assert.equal(rag?.sources[0]?.chunkId, `${documentId}-0`);
    assert.equal(rag?.truncated, true);
    assert.doesNotMatch(rag?.text ?? "", /página 2/);
  } finally {
    restoreEnv("RAG_MAX_CONTEXT_CHARACTERS", previousLimit);
    restoreEnv("EMBEDDING_ENABLED", previousEmbeddings);
  }
});

test("RAG_SEMANTIC_WEIGHT=0 evita que embeddings influyan en el ranking", async () => {
  const previous = {
    enabled: process.env.EMBEDDING_ENABLED,
    provider: process.env.EMBEDDING_PROVIDER,
    model: process.env.EMBEDDING_MODEL,
    semanticWeight: process.env.RAG_SEMANTIC_WEIGHT,
    lexicalWeight: process.env.RAG_LEXICAL_WEIGHT,
  };
  const originalFetch = globalThis.fetch;
  const workspaceId = crypto.randomUUID();
  const documentId = crypto.randomUUID();
  let embeddingCalls = 0;

  process.env.EMBEDDING_ENABLED = "true";
  process.env.EMBEDDING_PROVIDER = "ollama";
  process.env.EMBEDDING_MODEL = "qwen3-embedding:4b";
  process.env.RAG_SEMANTIC_WEIGHT = "0";
  process.env.RAG_LEXICAL_WEIGHT = "100";
  globalThis.fetch = async () => {
    embeddingCalls += 1;
    throw new Error("No debería consultar embeddings con peso semántico cero");
  };

  try {
    const { addDocument } = await vite.ssrLoadModule("/lib/documents/store.ts");
    const { buildKnowledgeContext } = await vite.ssrLoadModule("/lib/documents/retrieval.ts");

    addDocument(workspaceId, {
      id: documentId,
      name: "pesos.pdf",
      sizeBytes: 500,
      pages: 2,
      characters: 100,
      createdAt: Date.now(),
      embeddingProvider: "ollama",
      embeddingModel: "qwen3-embedding:4b",
      embeddingDimension: 2,
      chunks: [
        {
          id: `${documentId}-0`,
          documentId,
          documentName: "pesos.pdf",
          page: 1,
          index: 0,
          text: "objetivo alfa",
          embedding: [0, 1],
        },
        {
          id: `${documentId}-1`,
          documentId,
          documentName: "pesos.pdf",
          page: 2,
          index: 1,
          text: "objetivo beta",
          embedding: [1, 0],
        },
      ],
    });

    const rag = await buildKnowledgeContext(workspaceId, "rag", "objetivo", [documentId]);
    assert.equal(embeddingCalls, 0);
    assert.equal(rag?.embeddingUsed, false);
    assert.equal(rag?.sources[0]?.chunkId, `${documentId}-0`);
  } finally {
    globalThis.fetch = originalFetch;
    restoreEnv("EMBEDDING_ENABLED", previous.enabled);
    restoreEnv("EMBEDDING_PROVIDER", previous.provider);
    restoreEnv("EMBEDDING_MODEL", previous.model);
    restoreEnv("RAG_SEMANTIC_WEIGHT", previous.semanticWeight);
    restoreEnv("RAG_LEXICAL_WEIGHT", previous.lexicalWeight);
  }
});

test("la API rechaza una carga antes de procesarla si el workspace ya está lleno", async () => {
  const previousMaxDocuments = process.env.RAG_MAX_DOCUMENTS;
  const previousToken = process.env.APP_TOKEN;
  const workspaceId = crypto.randomUUID();
  const documentId = crypto.randomUUID();

  process.env.RAG_MAX_DOCUMENTS = "1";
  delete process.env.APP_TOKEN;

  try {
    const { addDocument } = await vite.ssrLoadModule("/lib/documents/store.ts");
    const { POST } = await vite.ssrLoadModule("/app/api/documents/route.ts");

    addDocument(workspaceId, {
      id: documentId,
      name: "existente.pdf",
      sizeBytes: 10,
      pages: 1,
      characters: 20,
      createdAt: Date.now(),
      chunks: [{
        id: `${documentId}-0`,
        documentId,
        documentName: "existente.pdf",
        page: 1,
        index: 0,
        text: "contenido existente",
      }],
    });

    const response = await POST(new Request("http://localhost/api/documents", {
      method: "POST",
      headers: { "x-workspace-id": workspaceId },
    }));

    assert.equal(response.status, 409);
    assert.match((await response.json()).error, /máximo de 1 documentos/);
  } finally {
    restoreEnv("RAG_MAX_DOCUMENTS", previousMaxDocuments);
    restoreEnv("APP_TOKEN", previousToken);
  }
});
