import assert from "node:assert/strict";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
const root = fileURLToPath(new URL("..", import.meta.url)).replace(/\/$/, "");
const vite = await createServer({ appType: "custom", configFile: false, root, resolve: { alias: { "@": root } }, server: { middlewareMode: true } });
after(async () => vite.close());
test("valida un dataset conversacional apto para SFT", async () => {
  const { validateJsonlDataset } = await vite.ssrLoadModule("/lib/fine-tuning/dataset.ts");
  const result = validateJsonlDataset('{"messages":[{"role":"user","content":"Hola"},{"role":"assistant","content":"Hola"}]}');
  assert.deepEqual(result, { valid: true, examples: 1, messages: 2, characters: 8, errors: [] });
});
test("rechaza ejemplos cuya respuesta final no pertenece al asistente", async () => {
  const { validateJsonlDataset } = await vite.ssrLoadModule("/lib/fine-tuning/dataset.ts");
  const result = validateJsonlDataset('{"messages":[{"role":"assistant","content":"Hola"},{"role":"user","content":"Hola"}]}');
  assert.equal(result.valid, false); assert.match(result.errors[0], /último mensaje debe ser assistant/);
});

test("chat devuelve JSON controlado si la configuración del proveedor es inválida", async () => {
  const previousProvider = process.env.LLM_PROVIDER;
  const previousToken = process.env.APP_TOKEN;
  process.env.LLM_PROVIDER = "proveedor-inexistente";
  delete process.env.APP_TOKEN;
  try {
    const { POST } = await vite.ssrLoadModule("/app/api/chat/route.ts");
    const response = await POST(new Request("http://localhost/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "Hola", mode: "none" }),
    }));
    assert.equal(response.status, 500);
    assert.deepEqual(await response.json(), { error: "La configuración del proveedor no es válida." });
  } finally {
    if (previousProvider === undefined) delete process.env.LLM_PROVIDER;
    else process.env.LLM_PROVIDER = previousProvider;
    if (previousToken === undefined) delete process.env.APP_TOKEN;
    else process.env.APP_TOKEN = previousToken;
  }
});
