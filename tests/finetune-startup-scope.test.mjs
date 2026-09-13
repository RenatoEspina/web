import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

const commands = await readFile(new URL("../comandos.fish", import.meta.url), "utf8");
const compose = await readFile(new URL("../docker-compose.yml", import.meta.url), "utf8");
const trainer = await readFile(new URL("../trainer/train.py", import.meta.url), "utf8");
const masterPrompt = await readFile(new URL("../lib/llm/masterPrompt.ts", import.meta.url), "utf8");
const chatRoute = await readFile(new URL("../app/api/chat/route.ts", import.meta.url), "utf8");
const llmTypes = await readFile(new URL("../lib/llm/types.ts", import.meta.url), "utf8");
const llmIndex = await readFile(new URL("../lib/llm/index.ts", import.meta.url), "utf8");
const openAiProvider = await readFile(new URL("../lib/llm/openai-compatible.ts", import.meta.url), "utf8");
const ollamaProvider = await readFile(new URL("../lib/llm/ollama.ts", import.meta.url), "utf8");
const terrariaMetadata = JSON.parse(
  await readFile(new URL("../trainer/corpora/terraria/data/metadata.json", import.meta.url), "utf8"),
);

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

test("Terraria usa el mismo system prompt en SFT y serving", () => {
  assert.equal(typeof terrariaMetadata.system, "string");
  assert.ok(terrariaMetadata.system.length > 80);
  assert.match(masterPrompt, /metadata\.json/);
  assert.match(masterPrompt, /TERRARIA_ADAPTER_SYSTEM_PROMPT = terrariaMetadata\.system/);
  assert.match(masterPrompt, /TERRARIA_MASTER_PROMPT = TERRARIA_ADAPTER_SYSTEM_PROMPT/);
  assert.match(masterPrompt, /adapterSystemPrompt\(model\)/);
  assert.match(masterPrompt, /isTerrariaModel\(model\) \? TERRARIA_ADAPTER_SYSTEM_PROMPT : null/);
  assert.match(masterPrompt, /content: prompt/);
  assert.match(masterPrompt, /first\?\.role === "system"/);
  assert.match(masterPrompt, /content: `\$\{prompt\}\\n\\n\$\{first\.content\}`/);
});

test("los modelos Terraria mantienen filtro duro con clasificación determinista", () => {
  assert.match(masterPrompt, /TERRARIA_OUT_OF_SCOPE_MESSAGE/);
  assert.match(masterPrompt, /Solo puedo responder preguntas relacionadas con Terraria\./);
  assert.match(masterPrompt, /LLM_TERRARIA_MODELS/);
  assert.match(masterPrompt, /TERRARIA_MODEL_PATTERN/);
  assert.match(masterPrompt, /TERRARIA_SCOPE_CLASSIFIER_PROMPT/);
  assert.match(masterPrompt, /directly about unmodded Terraria/);
  assert.match(masterPrompt, /Return exactly one label and nothing else: TERRARIA or OUTSIDE/);
  assert.match(masterPrompt, /parseTerrariaScopeDecision/);
  assert.match(masterPrompt, /\.test\(content\)/);

  const knowledge = chatRoute.indexOf("const knowledge =");
  const classify = chatRoute.indexOf("terrariaScopeMessages(knowledgeMessages)", knowledge);
  const reject = chatRoute.indexOf("scope_rejected", classify);
  const adapterCompletion = chatRoute.indexOf("const completion = await complete(", reject);

  assert.ok(knowledge >= 0);
  assert.ok(classify > knowledge);
  assert.ok(reject > classify);
  assert.ok(adapterCompletion > reject);
  assert.match(chatRoute, /terrariaScopeMessages\(knowledgeMessages\)[\s\S]*config\.model[\s\S]*temperature: 0[\s\S]*maxTokens: 8/);
  assert.match(chatRoute, /message: TERRARIA_OUT_OF_SCOPE_MESSAGE/);
  assert.match(chatRoute, /withMasterPrompt\(knowledgeMessages, selectedModel\)/);
});

test("un adapter entrenado sobre varias tareas no hereda el alcance por nombre", () => {
  // El gate de Terraria se disparaba solo por el nombre del modelo, así que un
  // adapter de dominio + replay llamado terraria-mix-* veía rechazada toda
  // pregunta general antes de ejecutarse. La decisión ahora sale del manifest.
  assert.match(masterPrompt, /function trainedAcrossTasks\(model: string\): boolean/);
  assert.match(masterPrompt, /serving\.systemPrompt == null/);

  const configured = masterPrompt.indexOf("configuredTerrariaModels().has(model)");
  const acrossTasks = masterPrompt.indexOf("trainedAcrossTasks(model)", configured);
  const pattern = masterPrompt.indexOf("TERRARIA_MODEL_PATTERN.test(model)", acrossTasks);

  // Orden obligatorio: configuración explícita del operador, luego la evidencia
  // del manifest, y el patrón por nombre solo como último recurso.
  assert.ok(configured >= 0);
  assert.ok(acrossTasks > configured);
  assert.ok(pattern > acrossTasks);
  assert.match(masterPrompt, /if \(configuredTerrariaModels\(\)\.has\(model\)\) return true;/);
  assert.match(masterPrompt, /if \(trainedAcrossTasks\(model\)\) return false;/);
});

test("el manifest del adapter se lee una sola vez para prompt y alcance", () => {
  assert.match(masterPrompt, /adapterManifestCache = new Map<string, AdapterManifest \| null>\(\)/);
  assert.match(masterPrompt, /function adapterManifest\(model: string\): AdapterManifest \| null/);
  // Ambos consumidores comparten la misma lectura y los mismos límites de ruta.
  assert.match(masterPrompt, /const serving = adapterManifest\(model\)\?\.serving;/);
  assert.match(masterPrompt, /const manifest = adapterManifest\(model\);/);
  assert.match(masterPrompt, /SAFE_ADAPTER_NAME\.test\(model\)/);
  assert.match(masterPrompt, /pathInside\(adaptersRoot, manifestPath\)/);
});

test("los providers respetan overrides sin cambiar defaults globales", () => {
  assert.match(llmTypes, /interface CompletionOptions/);
  assert.match(llmIndex, /options\?: CompletionOptions/);
  assert.match(openAiProvider, /options\?\.temperature \?\? this\.config\.temperature/);
  assert.match(openAiProvider, /options\?\.maxTokens \?\? this\.config\.maxTokens/);
  assert.match(ollamaProvider, /options\?\.temperature \?\? this\.config\.temperature/);
  assert.match(ollamaProvider, /options\?\.maxTokens \?\? this\.config\.maxTokens/);
});
