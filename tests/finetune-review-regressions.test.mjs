import assert from "node:assert/strict";
import test, { after } from "node:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
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

async function fixture(t, { trained = "research/base-a", served = "research/base-a", alias = served } = {}) {
  const adapters = join(root, "adapters");
  await mkdir(adapters, { recursive: true });
  const directory = await mkdtemp(join(adapters, "review-test-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const name = basename(directory);
  const write = (file, data) => writeFile(join(directory, file), JSON.stringify(data));
  await write("adapter_config.json", { base_model_name_or_path: trained });
  await write("manifest.json", { baseModel: trained });
  const entries = [{ id: alias, root: served }, { id: name, root: `/adapters/${name}`, parent: alias }];
  t.mock.method(globalThis, "fetch", async () => Response.json({ data: entries }));
  const config = { provider: "vllm", model: alias, baseUrl: "http://unused", apiKey: "" };
  return { entries, name, directory, write,
    validate: () => validateRuntimeModelSelection(config, name, AbortSignal.timeout(1000)),
    validateBase: () => validateRuntimeModelSelection(config, alias, AbortSignal.timeout(1000)) };
}

test("rejects a different training base even when vLLM parent matches the gateway", async (t) => {
  const f = await fixture(t, { served: "research/base-b" });
  await assert.rejects(f.validate(), (error) => error.status === 409 && /entrenado/.test(error.message));
  assert.equal((await f.validateBase()).id, "research/base-b");
});

test("accepts verified training metadata with a served alias and matching root", async (t) => {
  const f = await fixture(t, { alias: "public-base" });
  assert.equal((await f.validate()).id, f.name);
  // vLLM may publish several aliases and use the first as the adapter parent.
  f.entries.push({ id: "other-alias", root: "research/base-a" });
  f.entries[1].parent = "other-alias";
  assert.equal((await f.validate()).id, f.name);
});

test("rejects an alias that disguises a different root", async (t) => {
  const f = await fixture(t, { served: "research/base-b", alias: "research/base-a" });
  await assert.rejects(f.validate(), { status: 409 });
});

test("checks fresh metadata and rejects absent, malformed or inconsistent identity", async (t) => {
  const f = await fixture(t);
  assert.equal((await f.validate()).id, f.name);
  await f.write("manifest.json", { baseModel: "research/base-b" });
  await assert.rejects(f.validate(), { status: 409 });
  await rm(join(f.directory, "manifest.json"));
  assert.equal((await f.validate()).id, f.name); // Legacy PEFT config is sufficient.
  for (const invalid of [{}, [], { base_model_name_or_path: "research/base-b" }]) {
    await f.write("adapter_config.json", invalid);
    await assert.rejects(f.validate(), { status: 409 });
  }
  await rm(join(f.directory, "adapter_config.json"));
  await assert.rejects(f.validate(), { status: 409 });
});

test("also verifies metadata in the runtime export instead of trusting a replaced original", async (t) => {
  const f = await fixture(t);
  const exported = join(f.directory, "export");
  await mkdir(exported);
  f.entries[1].root = `/adapters/${f.name}/export`;
  await writeFile(join(exported, "adapter_config.json"), JSON.stringify({ base_model_name_or_path: "research/base-b" }));
  await assert.rejects(f.validate(), { status: 409 });
  await writeFile(join(exported, "adapter_config.json"), JSON.stringify({ base_model_name_or_path: "research/base-a" }));
  assert.equal((await f.validate()).id, f.name);
});

test("fails closed when runtime identity or local metadata path cannot be verified", async (t) => {
  const f = await fixture(t);
  delete f.entries[0].root;
  await assert.rejects(f.validate(), { status: 409 });
  f.entries[0].root = "research/base-a";
  f.entries[1].root = "/adapters/../../unknown-adapter";
  await assert.rejects(f.validate(), { status: 409 });
});
