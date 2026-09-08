import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

const route = await readFile(new URL("../app/api/chat/route.ts", import.meta.url), "utf8");
const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
const runtime = await readFile(new URL("../lib/llm/runtime.ts", import.meta.url), "utf8");
const openai = await readFile(new URL("../lib/llm/openai-compatible.ts", import.meta.url), "utf8");
const verifier = await readFile(new URL("../trainer/verify_lora.py", import.meta.url), "utf8");

test("el gateway conserva telemetría útil para comparar base y LoRA", () => {
  assert.match(openai, /completion_tokens/);
  assert.match(openai, /finish_reason/);
  assert.match(openai, /tokensPerSecond/);
  assert.match(route, /inference:\s*\{/);
  assert.match(route, /completionTokens: completion\.usage\.completionTokens/);
  assert.match(route, /finishReason: completion\.finishReason/);
});

test("vLLM valida el modelo base y el parent del adapter antes de inferir", () => {
  assert.match(runtime, /\/v1\/models/);
  assert.match(runtime, /baseEntries\.find\(\(entry\) => entry\.id === config\.model\)/);
  assert.match(runtime, /selected\.parent !== config\.model/);
  assert.match(runtime, /No se realizará inferencia con un LoRA incompatible/);

  const validation = route.indexOf("validateRuntimeModelSelection(");
  const generation = route.indexOf("const completion = await complete(");
  assert.ok(validation >= 0);
  assert.ok(generation > validation);
  assert.match(route, /RUNTIME_VALIDATION_TIMEOUT_MS/);
  assert.match(
    route,
    /AbortSignal\.timeout\(Math\.min\(config\.timeoutMs, RUNTIME_VALIDATION_TIMEOUT_MS\)\)/,
  );
});

test("cambiar de modelo inicia una comparación con historial limpio", () => {
  assert.match(page, /function handleModelChange\(nextModel: string\)/);
  assert.match(page, /setSelectedModel\(nextModel\);\s+setMessages\(\[\]\);/);
  assert.match(page, /onValueChange=\{handleModelChange\}/);
});

test("el frontend hace visibles tokens, latencia, throughput y motivo de fin", () => {
  assert.match(page, /Métricas de inferencia/);
  assert.match(page, /completionTokens/);
  assert.match(page, /tokensPerSecond/);
  assert.match(page, /finishReason/);
  assert.match(page, /latencia/);
});

test("el verificador LoRA detecta respuestas anormalmente cortas", () => {
  assert.match(verifier, /GENERATION_MAX_TOKENS = 256/);
  assert.match(verifier, /generation_probe/);
  assert.match(verifier, /adapterMarkedlyShorter/);
  assert.match(verifier, /earlyStopSuspected/);
});
