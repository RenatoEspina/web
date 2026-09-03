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
