import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url)).replace(/\/$/, "");
const vite = await createServer({ appType: "custom", configFile: false, root, server: { middlewareMode: true } });
after(() => vite.close());
const { terrariaScopeMessages } = await vite.ssrLoadModule("/lib/llm/masterPrompt.ts");
const { validateRuntimeModelSelection } = await vite.ssrLoadModule("/lib/llm/runtime.ts");

test("scope preserves the complete current question while bounding earlier turns", () => {
  const prefix = "Estos son mis apuntes previos. ".repeat(55);
  const inside = prefix + "¿Cómo invoco al Ojo de Cthulhu en Terraria?";
  const outside = prefix + "¿Cómo resuelvo una integral por partes?";
  const history = [{ role: "system", content: "s".repeat(2000) },
    ...Array.from({ length: 12 }, () => ({ role: "assistant", content: "h".repeat(2000) }))];
  const classify = (content) => terrariaScopeMessages([...history, { role: "user", content }]);
  const first = classify(inside);
  assert.notDeepEqual(first, classify(outside));
  const conversation = JSON.parse(first[1].content.slice(first[1].content.indexOf("\n") + 1));
  assert.equal(conversation.length, 8);
  assert.equal(conversation[0].role, "system");
  assert.equal(conversation.at(-1).content, inside);
  assert.ok(conversation.slice(0, -1).every((item) => item.content.length <= 1500));
  assert.ok(conversation.reduce((total, item) => total + item.content.length, 0) <= 12000);
  const maximum = "a".repeat(11990) + " Terraria?";
  assert.ok(classify(maximum)[1].content.includes(maximum));
  const maxConversation = JSON.parse(classify(maximum)[1].content.split("\n").slice(1).join("\n"));
  assert.equal(maxConversation.length, 1);
});

function fixture(t, { served = "research/base-a", alias = served } = {}) {
  const name = "review-test-adapter";
  const entries = [
    { id: alias, root: served },
    { id: name, root: `/adapters/${name}`, parent: alias },
  ];
  t.mock.method(globalThis, "fetch", async () => Response.json({ data: entries }));
  const config = { provider: "vllm", model: alias, baseUrl: "http://unused", apiKey: "" };
  return {
    entries,
    name,
    validate: () => validateRuntimeModelSelection(config, name, AbortSignal.timeout(1000)),
    validateBase: () => validateRuntimeModelSelection(config, alias, AbortSignal.timeout(1000)),
  };
}

test("accepts a loaded adapter when its parent resolves to the configured served root", async (t) => {
  const f = fixture(t, { alias: "public-base" });
  assert.equal((await f.validate()).id, f.name);

  // vLLM may publish several aliases for the same root and use any one as parent.
  f.entries.push({ id: "other-alias", root: "research/base-a" });
  f.entries[1].parent = "other-alias";
  assert.equal((await f.validate()).id, f.name);
});

test("rejects an adapter whose parent alias resolves to another served root", async (t) => {
  const f = fixture(t, { alias: "public-base" });
  f.entries.push({ id: "wrong-parent", root: "research/base-b" });
  f.entries[1].parent = "wrong-parent";
  await assert.rejects(f.validate(), (error) => error.status === 409 && /base distinta/.test(error.message));
});

test("fails closed when vLLM omits runtime identity fields", async (t) => {
  const f = fixture(t);
  delete f.entries[0].root;
  await assert.rejects(f.validate(), { status: 409 });

  f.entries[0].root = "research/base-a";
  delete f.entries[1].root;
  await assert.rejects(f.validate(), { status: 409 });

  f.entries[1].root = `/adapters/${f.name}`;
  delete f.entries[1].parent;
  await assert.rejects(f.validate(), { status: 409 });
});

test("rejects a configured base or adapter that is not currently served", async (t) => {
  const f = fixture(t);
  assert.equal((await f.validateBase()).id, "research/base-a");

  f.entries.splice(1, 1);
  await assert.rejects(f.validate(), (error) => error.status === 409 && /no está cargado/.test(error.message));

  f.entries.splice(0, 1);
  await assert.rejects(f.validateBase(), (error) => error.status === 409 && /gateway espera/.test(error.message));
});

test("runtime validation is Worker-safe and never reads host adapters through node:fs", async () => {
  const runtime = await readFile(new URL("../lib/llm/runtime.ts", import.meta.url), "utf8");
  assert.doesNotMatch(runtime, /from ["']node:fs["']/);
  assert.doesNotMatch(runtime, /process\.cwd\(\)/);
  assert.doesNotMatch(runtime, /realpathSync|readFileSync/);
  assert.match(runtime, /verifyAdapterRuntimeIdentity/);
});
