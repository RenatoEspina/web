import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

const commands = await readFile(new URL("../comandos.fish", import.meta.url), "utf8");
const compose = await readFile(new URL("../docker-compose.yml", import.meta.url), "utf8");
const trainer = await readFile(new URL("../trainer/train.py", import.meta.url), "utf8");
const masterPrompt = await readFile(new URL("../lib/llm/masterPrompt.ts", import.meta.url), "utf8");
const chatRoute = await readFile(new URL("../app/api/chat/route.ts", import.meta.url), "utf8");

test("vLLM inicia antes del smoke test pesado de embeddings", () => {
  const prepare = commands.indexOf('prepare_embeddings "$default_embedding_model"');
  const vllmUp = commands.indexOf("compose up -d vllm", prepare);
  const vllmReady = commands.indexOf("wait_for_url vllm", vllmUp);
  const verify = commands.indexOf('verify_embeddings "$default_embedding_model"', vllmReady);

  assert.ok(prepare >= 0);
  assert.ok(vllmUp > prepare);
  assert.ok(vllmReady > vllmUp);
  assert.ok(verify > vllmReady);
  assert.match(commands, /ollama show "\$model"/);
  assert.match(commands, /--max-time 180/);
});

test("Ollama queda aislado en CPU con runner AVX2", () => {
  assert.match(compose, /OLLAMA_LLM_LIBRARY: "cpu_avx2"/);
  assert.match(compose, /CUDA_VISIBLE_DEVICES: "-1"/);
});

test("SFT valida la máscara assistant-only antes de cargar el modelo cuantizado", () => {
  const templateCheck = trainer.indexOf("prepare_assistant_only_template(tokenizer, examples)");
  const modelLoad = trainer.indexOf("AutoModelForCausalLM.from_pretrained");

  assert.ok(templateCheck >= 0);
  assert.ok(modelLoad > templateCheck);
  assert.match(trainer, /get_training_chat_template/);
  assert.match(trainer, /return_assistant_tokens_mask=True/);
  assert.match(trainer, /encoded\.get\("assistant_masks"\)/);
  assert.match(trainer, /if all\(assistant_mask\)/);
});

test("los modelos Terraria reciben master prompt y filtro duro de dominio", () => {
  assert.match(masterPrompt, /TERRARIA_MASTER_PROMPT/);
  assert.match(masterPrompt, /scope is strictly Terraria/i);
  assert.match(masterPrompt, /TERRARIA_OUT_OF_SCOPE_MESSAGE/);
  assert.match(masterPrompt, /Solo puedo responder preguntas relacionadas con Terraria\./);
  assert.match(masterPrompt, /LLM_TERRARIA_MODELS/);
  assert.match(masterPrompt, /TERRARIA_MODEL_PATTERN/);
  assert.match(masterPrompt, /TERRARIA_SCOPE_CLASSIFIER_PROMPT/);
  assert.match(masterPrompt, /Return exactly one label and nothing else: TERRARIA or OUTSIDE/);
  assert.match(masterPrompt, /parseTerrariaScopeDecision/);
  assert.match(masterPrompt, /content\.trim\(\)\.toUpperCase\(\) === "TERRARIA"/);

  const classify = chatRoute.indexOf("terrariaScopeMessages(messages)");
  const reject = chatRoute.indexOf("scope_rejected", classify);
  const knowledge = chatRoute.indexOf("buildKnowledgeContext", classify);
  const adapterCompletion = chatRoute.indexOf("complete(requestMessages", classify);

  assert.ok(classify >= 0);
  assert.ok(reject > classify);
  assert.ok(knowledge > reject);
  assert.ok(adapterCompletion > knowledge);
  assert.match(chatRoute, /complete\([\s\S]*terrariaScopeMessages\(messages\)[\s\S]*config\.model/);
  assert.match(chatRoute, /message: TERRARIA_OUT_OF_SCOPE_MESSAGE/);
  assert.match(chatRoute, /withMasterPrompt\(knowledgeMessages, selectedModel\)/);
});
