import assert from "node:assert/strict";
import test, { after } from "node:test";
import { createServer } from "vite";

const root = new URL("..", import.meta.url).pathname.replace(/\/$/, "");
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

test("crea un fragmento que puede cruzar páginas y conserva el rango de origen", async () => {
  const { createChunks } = await vite.ssrLoadModule("/lib/documents/chunking.ts");
  const chunks = createChunks("documento-1", "manual.pdf", [
    "Contenido de la primera página.",
    "Contenido de la segunda página.",
  ]);

  assert.equal(chunks.length, 1);
  assert.equal(chunks[0].page, 1);
  assert.equal(chunks[0].pageEnd, 2);
  assert.match(chunks[0].text, /primera página/);
  assert.match(chunks[0].text, /segunda página/);
});

test("RAG recupera el fragmento relevante y CAG reutiliza su caché", async () => {
  const { addDocument } = await vite.ssrLoadModule("/lib/documents/store.ts");
  const { buildKnowledgeContext } = await vite.ssrLoadModule("/lib/documents/retrieval.ts");
  const workspaceId = crypto.randomUUID();
  const documentId = crypto.randomUUID();
  const document = {
    id: documentId,
    name: "apuntes.pdf",
    sizeBytes: 500,
    pages: 1,
    characters: 94,
    createdAt: Date.now(),
    chunks: [{
      id: `${documentId}-0`,
      documentId,
      documentName: "apuntes.pdf",
      page: 1,
      index: 0,
      text: "La memoria caché conserva el contexto documental para reutilizarlo en preguntas posteriores.",
    }],
  };

  addDocument(workspaceId, document);

  const rag = await buildKnowledgeContext(workspaceId, "rag", "¿Qué conserva la memoria caché?", [documentId]);
  assert.equal(rag?.mode, "rag");
  assert.equal(rag?.sources[0]?.page, 1);
  assert.match(rag?.text ?? "", /contexto documental/);

  const firstCag = await buildKnowledgeContext(workspaceId, "cag", "resumen", [documentId]);
  const secondCag = await buildKnowledgeContext(workspaceId, "cag", "otra pregunta", [documentId]);
  assert.equal(firstCag?.cacheHit, false);
  assert.equal(secondCag?.cacheHit, true);
});

test("RAG combina embeddings con búsqueda léxica", async () => {
  const { addDocument } = await vite.ssrLoadModule("/lib/documents/store.ts");
  const { buildKnowledgeContext } = await vite.ssrLoadModule("/lib/documents/retrieval.ts");
  const workspaceId = crypto.randomUUID();
  const documentId = crypto.randomUUID();
  const originalFetch = globalThis.fetch;
  const previousEmbeddingEnvironment = {
    enabled: process.env.EMBEDDING_ENABLED,
    provider: process.env.EMBEDDING_PROVIDER,
    model: process.env.EMBEDDING_MODEL,
  };

  process.env.EMBEDDING_ENABLED = "true";
  process.env.EMBEDDING_PROVIDER = "ollama";
  process.env.EMBEDDING_MODEL = "embeddinggemma";
  globalThis.fetch = async (_input, init) => {
    const body = JSON.parse(String(init?.body));
    assert.equal(body.model, "embeddinggemma");
    return new Response(JSON.stringify({ embeddings: [[0, 1]] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  try {
    addDocument(workspaceId, {
      id: documentId,
      name: "semantica.pdf",
      sizeBytes: 500,
      pages: 1,
      characters: 100,
      createdAt: Date.now(),
      embeddingProvider: "ollama",
      embeddingModel: "embeddinggemma",
      embeddingDimension: 2,
      chunks: [
        {
          id: `${documentId}-0`,
          documentId,
          documentName: "semantica.pdf",
          page: 1,
          index: 0,
          text: "El felino doméstico duerme durante la tarde.",
          embedding: [1, 0],
        },
        {
          id: `${documentId}-1`,
          documentId,
          documentName: "semantica.pdf",
          page: 1,
          index: 1,
          text: "La bicicleta necesita mantenimiento periódico.",
          embedding: [0, 1],
        },
      ],
    });

    const rag = await buildKnowledgeContext(workspaceId, "rag", "¿Cómo mantener la bicicleta?", [documentId]);
    assert.equal(rag?.embeddingUsed, true);
    assert.equal(rag?.sources[0]?.chunkId, `${documentId}-1`);
    assert.match(rag?.text ?? "", /bicicleta/);
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
