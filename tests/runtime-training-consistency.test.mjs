import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const root = new URL("../", import.meta.url);
const read = (path) => readFile(new URL(path, root), "utf8");

const masterPrompt = await read("lib/llm/masterPrompt.ts");
const chatRoute = await read("app/api/chat/route.ts");
const documentPrompt = await read("lib/documents/prompt.ts");
const commands = await read("comandos.fish");
const trainScript = await read("scripts/train-adapter.sh");
const verifyLora = await read("trainer/verify_lora.py");

test("serving recupera el system prompt del dataset del adapter y no solo de Terraria", () => {
  assert.match(masterPrompt, /function adapterSystemPrompt\(model: string\)/);
  assert.match(masterPrompt, /commonDatasetSystemPrompt\(manifest\.dataset\)/);
  assert.match(masterPrompt, /const prompt = adapterSystemPrompt\(model\)/);
  assert.match(masterPrompt, /currentDatasetPath/);
});

test("scope Terraria coincide con el corpus unmodded y conserva contexto documental", () => {
  assert.match(masterPrompt, /directly about unmodded Terraria/);
  assert.match(masterPrompt, /mod loader, third-party tool/);
  assert.match(masterPrompt, /messages\[0\]\?\.role === "system"/);
  assert.ok(!masterPrompt.includes("Terraria-specific mods or tooling"));
});

test("RAG CAG se construye antes del scope y los timeouts no comparten reloj", () => {
  const knowledgeIndex = chatRoute.indexOf("const knowledge =");
  const scopeIndex = chatRoute.indexOf("if (isTerrariaModel(selectedModel))");
  assert.ok(knowledgeIndex >= 0 && scopeIndex > knowledgeIndex);
  assert.match(chatRoute, /terrariaScopeMessages\(knowledgeMessages\)/);
  assert.match(chatRoute, /SCOPE_TIMEOUT_MS/);
  assert.match(chatRoute, /RUNTIME_VALIDATION_TIMEOUT_MS/);
  assert.match(chatRoute, /AbortSignal\.timeout\(config\.timeoutMs\)/);
});

test("las instrucciones documentales no contradicen el idioma del adapter", () => {
  assert.match(documentPrompt, /Respeta cualquier requisito de idioma o estilo que ya establezca el system prompt del modelo/);
  assert.ok(!documentPrompt.includes('"Responde en el idioma de la pregunta y sé preciso."'));
});

test("arrancar vLLM sincroniza el modelo persistido del gateway", () => {
  assert.match(commands, /function persist_llm_model/);
  assert.match(commands, /persist_llm_model "\$model"/);
  assert.match(commands, /LLM_MODEL sincronizado en \.env\.local/);
});

test("entrenar preserva el modelo base que estaba servido antes de liberar la GPU", () => {
  assert.match(trainScript, /http:\/\/127\.0\.0\.1:8000\/v1\/models/);
  assert.match(trainScript, /vllm_model=/);
  assert.match(trainScript, /VLLM_MODEL="\$vllm_model" compose up -d vllm/);
});

test("el diagnóstico LoRA aplica el mismo system prompt a base y adapter", () => {
  assert.match(verifyLora, /def adapter_system_prompt\(adapter: str\)/);
  assert.match(verifyLora, /system_prompt = adapter_system_prompt\(adapter\)/);
  assert.match(verifyLora, /baseline = probe\(base_url, base_model, prompt, system_prompt\)/);
  assert.match(verifyLora, /adapted = probe\(base_url, adapter, prompt, system_prompt\)/);
  assert.match(verifyLora, /"systemPromptApplied": bool\(system_prompt\)/);
});
