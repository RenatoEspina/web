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

test("crea fragmentos con su página de origen", async () => {
  const { createChunks } = await vite.ssrLoadModule("/lib/documents/chunking.ts");
  const chunks = createChunks("documento-1", "manual.pdf", [
    "Contenido de la primera página.",
    "Contenido de la segunda página.",
  ]);

  assert.equal(chunks.length, 2);
  assert.equal(chunks[0].page, 1);
  assert.equal(chunks[1].page, 2);
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

  const rag = buildKnowledgeContext(workspaceId, "rag", "¿Qué conserva la memoria caché?", [documentId]);
  assert.equal(rag?.mode, "rag");
  assert.equal(rag?.sources[0]?.page, 1);
  assert.match(rag?.text ?? "", /contexto documental/);

  const firstCag = buildKnowledgeContext(workspaceId, "cag", "resumen", [documentId]);
  const secondCag = buildKnowledgeContext(workspaceId, "cag", "otra pregunta", [documentId]);
  assert.equal(firstCag?.cacheHit, false);
  assert.equal(secondCag?.cacheHit, true);
});
