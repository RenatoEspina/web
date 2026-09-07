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

test("RAG sin resultados no afirma que recuperó fragmentos relevantes", async () => {
  const { withKnowledge } = await vite.ssrLoadModule("/lib/documents/prompt.ts");

  const messages = [{ role: "user", content: "¿Cuál es la respuesta?" }];
  const result = withKnowledge(messages, {
    mode: "rag",
    text: "",
    sources: [],
    cacheHit: false,
    embeddingUsed: false,
    truncated: false,
  });

  const system = result[0]?.content ?? "";
  assert.match(system, /no se recuperaron fragmentos relevantes/i);
  assert.match(system, /No se encontraron fragmentos relevantes/i);
  assert.doesNotMatch(system, /Se recuperaron fragmentos relevantes mediante RAG/i);
});
