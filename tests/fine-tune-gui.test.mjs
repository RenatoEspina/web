import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

const server = await readFile(new URL("../trainer/gui_server.py", import.meta.url), "utf8");
const launcher = await readFile(new URL("../fine-tune-gui", import.meta.url), "utf8");
const compose = await readFile(new URL("../docker-compose.yml", import.meta.url), "utf8");
const gui = await readFile(new URL("../trainer/gui/index.html", import.meta.url), "utf8");
const commands = await readFile(new URL("../comandos.fish", import.meta.url), "utf8");
const trainer = await readFile(new URL("../trainer/train.py", import.meta.url), "utf8");
const requirements = await readFile(new URL("../trainer/requirements.txt", import.meta.url), "utf8");
const dependencyCheck = await readFile(new URL("../trainer/check_dependencies.py", import.meta.url), "utf8");
const trainingScript = await readFile(new URL("../scripts/train-adapter.sh", import.meta.url), "utf8");
const ci = await readFile(new URL("../.github/workflows/ci.yml", import.meta.url), "utf8");

test("la GUI de fine-tuning queda restringida a loopback", () => {
  assert.match(server, /default="127\.0\.0\.1"/);
  assert.match(server, /Por seguridad, la GUI solo puede enlazarse a loopback/);
  assert.match(launcher, /HOST=127\.0\.0\.1/);
});

test("la GUI solo admite acciones administrativas predefinidas", () => {
  assert.match(server, /allowed = \{"setup", "check", "train", "start-vllm", "stop-vllm", "load-adapter", "unload-adapter", "verify-adapter", "delete-adapter"\}/);
  assert.match(server, /Dataset fuera de las carpetas permitidas/);
  assert.match(server, /x-fine-tune-token/i);
});

test("vLLM habilita LoRA dinámica únicamente detrás del puerto local", () => {
  assert.match(compose, /127\.0\.0\.1:8000:8000/);
  assert.match(compose, /VLLM_ALLOW_RUNTIME_LORA_UPDATING/);
});

test("el learning rate por defecto es válido para el control HTML", () => {
  assert.match(gui, /id="learningRate"[^>]+value="0\.0001"[^>]+step="any"/);
});

test("la sección de entrenamiento explica sus parámetros", () => {
  assert.match(gui, /class="field-help"[^>]+data-help="Identificador de la carpeta del adaptador/);
  assert.match(gui, /class="field-help"[^>]+data-help="Tamaño de cada actualización/);
  assert.match(gui, /class="field-help"[^>]+data-help="Acumula gradientes antes de actualizar/);
});

test("la GUI permite borrar adaptadores persistidos y los descarga si están activos", () => {
  assert.match(gui, /data-action="delete-adapter">Borrar/);
  assert.match(gui, /delete-adapter.*window\.confirm/);
  assert.match(server, /if action == "delete-adapter"/);
  assert.match(server, /Descargando \{name\} de vLLM antes de borrarlo/);
  assert.match(server, /post_vllm\("\/v1\/unload_lora_adapter"/);
  assert.match(server, /shutil\.rmtree\(directory\)/);
});

test("iniciar vLLM desde la GUI reutiliza comandos.fish y levanta embeddings", () => {
  assert.match(server, /\[fish, str\(ROOT \/ "comandos\.fish"\), "vllm", model\]/);
  assert.match(server, /env\["LLM_BRIDGE_NONINTERACTIVE"\] = "1"/);
  assert.match(server, /"ollama": "started"/);
  assert.match(server, /"embeddings": "ready"/);
  assert.match(commands, /start_embeddings/);
  assert.match(commands, /LLM_BRIDGE_NONINTERACTIVE/);
});

test("la GUI permite comprobar efecto LoRA además de registrarlo", () => {
  assert.match(gui, /data-action="verify-adapter">Comprobar uso/);
  assert.match(server, /if action == "verify-adapter"/);
  assert.match(server, /"inferenceVerified": False/);
  assert.match(server, /runtime_path = prepare_runtime_adapter\(job, name\)/);
});

test("evaluation.jsonl se identifica como evaluación y no se ofrece para SFT", () => {
  assert.match(server, /EVALUATION_DATASETS = \{"evaluation\.jsonl"\}/);
  assert.match(server, /evaluation\.jsonl es un dataset de evaluación y no se puede usar para SFT/);
  assert.match(server, /if path\.name\.casefold\(\) in EVALUATION_DATASETS:\s+continue/);
});

test("entrenamiento y vLLM preparan adapters sin borrar su contenido", () => {
  assert.equal((server.match(/ensure_adapter_dir_writable\(job\)/g) || []).length, 4);
  assert.match(server, /docker,[\s\S]+"run",[\s\S]+"--rm",[\s\S]+"--user",[\s\S]+"0:0"/);
  assert.match(server, /DEFAULT_VLLM_IMAGE = "vllm\/vllm-openai:v0\.24\.0"/);
  assert.match(server, /"--entrypoint",\s+"\/bin\/sh"/);
  assert.match(server, /chown -R \{os\.getuid\(\)\}:\{os\.getgid\(\)\} \/adapters && chmod -R u\+rwX \/adapters/);
  assert.match(server, /owner=\{owner\}:\{group\}.*mode=\{mode\}/);
  assert.match(server, /sudo chown -R/);
  assert.doesNotMatch(server, /rmtree\(ADAPTER_DIR/);
});

test("GUI y CLI reutilizan la misma comprobación CUDA y NF4", () => {
  assert.match(server, /check_environment\.py/);
  assert.match(commands, /trainer\/check_environment\.py/);
  assert.doesNotMatch(server, /CUDA_CHECK\s*=/);
  assert.doesNotMatch(commands, /quantize_4bit/);
});

test("el CLI ya no documenta el flujo LoRA estático deprecado", () => {
  assert.doesNotMatch(commands, /fine_tune_config/);
  assert.doesNotMatch(commands, /--lora-modules/);
});

test("train.py publica el adaptador solo después de completar el staging", () => {
  assert.match(trainer, /tempfile\.mkdtemp/);
  assert.match(trainer, /staging\.rename\(destination\)/);
  assert.match(trainer, /shutil\.rmtree\(staging, ignore_errors=True\)/);
});

test("el trainer usa una versión de datasets compatible con Python 3.14", () => {
  assert.match(requirements, /^datasets==4\.8\.5$/m);
  assert.match(dependencyCheck, /Dataset\.from_list\(\[\{"text": "compatibility-check"\}\]\)/);
  assert.match(trainingScript, /trainer\/check_dependencies\.py/);
  assert.match(trainingScript, /pip install -r trainer\/requirements\.txt/);
});

test("la GUI hace visible la compatibilidad y actualización del entorno", () => {
  assert.match(gui, /Python 3\.14 compatible/);
  assert.match(gui, /Preparar \/ actualizar entorno/);
  assert.match(gui, /datasets 4\.8\.5/);
  assert.match(gui, /sincroniza automáticamente un entorno desactualizado/);
});

test("el stack de fine-tuning soporta oficialmente Qwen3.5 y Python 3.14", () => {
  assert.match(requirements, /^transformers==5\.16\.1$/m);
  assert.match(requirements, /^trl==1\.11\.0$/m);
  assert.match(requirements, /^peft==0\.20\.0$/m);
  assert.match(requirements, /^accelerate==1\.14\.0$/m);
  assert.match(requirements, /^safetensors==0\.8\.0$/m);
  assert.match(dependencyCheck, /"qwen3_5" not in CONFIG_MAPPING/);
  assert.match(dependencyCheck, /Qwen3_5ForCausalLM/);
});

test("CI resuelve las dependencias del trainer con la misma versión mayor de Python", () => {
  assert.match(ci, /python-version: "3\.14"/);
  assert.match(ci, /pip install --dry-run -r trainer\/requirements\.txt/);
});

test("train.py valida la arquitectura antes de cargar el modelo y no usa Dataset.map con lambda", () => {
  assert.match(trainer, /validate_model_architecture\(args\.model, model_revision\)/);
  assert.match(trainer, /AutoConfig\.from_pretrained/);
  assert.match(trainer, /Dataset\.from_list\(examples\)/);
  assert.match(trainer, /assistant_only_loss=True/);
  assert.match(trainer, /audit_token_lengths/);\n  assert.match(trainer, /datasetSha256/);\n  assert.match(trainer, /model-revision/);
  assert.match(trainer, /processing_class=tokenizer/);
  assert.match(trainer, /dtype=compute_dtype/);
  assert.doesNotMatch(trainer, /Dataset\.from_list\(examples\)\.map/);
  assert.doesNotMatch(trainer, /lambda example/);
});

test("la GUI vuelve a ser un wizard de cinco pasos con terminal separada", () => {
  assert.equal((gui.match(/class="step-button[^\"]*"[^>]+data-step="[0-4]"/g) || []).length, 5);
  assert.equal((gui.match(/class="step-panel[^\"]*"[^>]+data-panel="[0-4]"/g) || []).length, 5);
  assert.match(gui, /id="prevStepBtn"[^>]*>Atrás/);
  assert.match(gui, /id="nextStepBtn"[^>]*>Siguiente/);
  assert.match(gui, /Terminal \/ logs/);
  assert.match(gui, /function showStep\(index\)/);
  assert.match(gui, /grid-template-columns:minmax\(0,1\.05fr\) minmax\(360px,\.8fr\)/);
});

test("el HF token de entrenamiento es efímero y no viaja por argv", () => {
  assert.match(gui, /id="trainHfToken" type="password"/);
  assert.match(gui, /hfToken:\$\("trainHfToken"\)\.value/);
  assert.equal((server.match(/payload\.pop\("hfToken", ""\)/g) || []).length, 2);
  assert.match(server, /env\["HF_TOKEN"\] = token/);
  assert.match(server, /env\["HUGGING_FACE_HUB_TOKEN"\] = token/);
  assert.match(server, /run_command\(job, command, env=training_env/);
  assert.doesNotMatch(server, /--hf-token/);
  assert.doesNotMatch(server, /os\.environ\["HF_TOKEN"\]\s*=/);
});
