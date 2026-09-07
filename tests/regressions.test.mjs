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

after(async () => {
  await vite.close();
});

test("una selección documental explícitamente vacía no usa todos los documentos", async () => {
  const { addDocument, getDocuments } = await vite.ssrLoadModule("/lib/documents/store.ts");
  const { buildKnowledgeContext } = await vite.ssrLoadModule("/lib/documents/retrieval.ts");
  const workspaceId = crypto.randomUUID();
  const documentId = crypto.randomUUID();

  addDocument(workspaceId, {
    id: documentId,
    name: "privado.pdf",
    sizeBytes: 100,
    pages: 1,
    characters: 50,
    createdAt: Date.now(),
    chunks: [{
      id: `${documentId}-0`,
      documentId,
      documentName: "privado.pdf",
      page: 1,
      index: 0,
      text: "Este contenido no debe entrar si el documento fue desmarcado.",
    }],
  });

  assert.equal(getDocuments(workspaceId).length, 1);
  assert.deepEqual(getDocuments(workspaceId, []), []);
  assert.equal(await buildKnowledgeContext(workspaceId, "rag", "contenido", []), null);
});

test("RAG solo publica fuentes cuyos fragmentos entraron al contexto", async () => {
  const { addDocument } = await vite.ssrLoadModule("/lib/documents/store.ts");
  const { buildKnowledgeContext } = await vite.ssrLoadModule("/lib/documents/retrieval.ts");
  const previousContextLimit = process.env.RAG_MAX_CONTEXT_CHARACTERS;
  const previousEmbeddingEnabled = process.env.EMBEDDING_ENABLED;
  const workspaceId = crypto.randomUUID();
  const documentId = crypto.randomUUID();

  process.env.RAG_MAX_CONTEXT_CHARACTERS = "1000";
  process.env.EMBEDDING_ENABLED = "false";

  try {
    addDocument(workspaceId, {
      id: documentId,
      name: "limite.pdf",
      sizeBytes: 3000,
      pages: 2,
      characters: 3000,
      createdAt: Date.now(),
      chunks: [
        {
          id: `${documentId}-0`,
          documentId,
          documentName: "limite.pdf",
          page: 1,
          index: 0,
          text: `objetivo ${"contenido ".repeat(180)}`,
        },
        {
          id: `${documentId}-1`,
          documentId,
          documentName: "limite.pdf",
          page: 2,
          index: 1,
          text: `objetivo ${"información ".repeat(180)}`,
        },
      ],
    });

    const rag = await buildKnowledgeContext(workspaceId, "rag", "objetivo", [documentId]);
    assert.equal(rag?.truncated, true);
    assert.equal(rag?.sources.length, 1);
    assert.equal(rag?.sources[0]?.chunkId, `${documentId}-0`);
    assert.doesNotMatch(rag?.text ?? "", /información/);
  } finally {
    if (previousContextLimit === undefined) delete process.env.RAG_MAX_CONTEXT_CHARACTERS;
    else process.env.RAG_MAX_CONTEXT_CHARACTERS = previousContextLimit;
    if (previousEmbeddingEnabled === undefined) delete process.env.EMBEDDING_ENABLED;
    else process.env.EMBEDDING_ENABLED = previousEmbeddingEnabled;
  }
});

test("RRF excluye candidatos con similitud semántica no positiva y sin señal léxica", async () => {
  const { addDocument } = await vite.ssrLoadModule("/lib/documents/store.ts");
  const { buildKnowledgeContext } = await vite.ssrLoadModule("/lib/documents/retrieval.ts");
  const originalFetch = globalThis.fetch;
  const previousEmbeddingEnvironment = {
    enabled: process.env.EMBEDDING_ENABLED,
    provider: process.env.EMBEDDING_PROVIDER,
    model: process.env.EMBEDDING_MODEL,
  };
  const workspaceId = crypto.randomUUID();
  const documentId = crypto.randomUUID();

  process.env.EMBEDDING_ENABLED = "true";
  process.env.EMBEDDING_PROVIDER = "ollama";
  process.env.EMBEDDING_MODEL = "qwen3-embedding:4b";
  globalThis.fetch = async () => new Response(JSON.stringify({ embeddings: [[1, 0]] }), { status: 200 });

  try {
    addDocument(workspaceId, {
      id: documentId,
      name: "semantica.pdf",
      sizeBytes: 500,
      pages: 2,
      characters: 120,
      createdAt: Date.now(),
      embeddingProvider: "ollama",
      embeddingModel: "qwen3-embedding:4b",
      embeddingDimension: 2,
      chunks: [
        {
          id: `${documentId}-0`,
          documentId,
          documentName: "semantica.pdf",
          page: 1,
          index: 0,
          text: "Un fragmento conceptualmente relacionado sin palabras de la consulta.",
          embedding: [1, 0],
        },
        {
          id: `${documentId}-1`,
          documentId,
          documentName: "semantica.pdf",
          page: 2,
          index: 1,
          text: "Un fragmento conceptualmente opuesto y también sin coincidencias.",
          embedding: [-1, 0],
        },
      ],
    });

    const rag = await buildKnowledgeContext(workspaceId, "rag", "consulta ajena", [documentId]);
    assert.equal(rag?.embeddingUsed, true);
    assert.deepEqual(rag?.sources.map((source) => source.chunkId), [`${documentId}-0`]);
  } finally {
    globalThis.fetch = originalFetch;
    if (previousEmbeddingEnvironment.enabled === undefined) delete process.env.EMBEDDING_ENABLED;
    else process.env.EMBEDDING_ENABLED = previousEmbeddingEnvironment.enabled;
    if (previousEmbeddingEnvironment.provider === undefined) delete process.env.EMBEDDING_PROVIDER;
    else process.env.EMBEDDING_PROVIDER = previousEmbeddingEnvironment.provider;
    if (previousEmbeddingEnvironment.model === undefined) delete process.env.EMBEDDING_MODEL;
    else process.env.EMBEDDING_MODEL = previousEmbeddingEnvironment.model;
  }
});
