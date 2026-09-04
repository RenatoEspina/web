import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

const server = await readFile(new URL("../trainer/gui_server.py", import.meta.url), "utf8");
const launcher = await readFile(new URL("../fine-tune-gui", import.meta.url), "utf8");
const compose = await readFile(new URL("../docker-compose.yml", import.meta.url), "utf8");
const gui = await readFile(new URL("../trainer/gui/index.html", import.meta.url), "utf8");

test("la GUI de fine-tuning queda restringida a loopback", () => {
  assert.match(server, /default="127\.0\.0\.1"/);
  assert.match(server, /Por seguridad, la GUI solo puede enlazarse a loopback/);
  assert.match(launcher, /HOST=127\.0\.0\.1/);
});

test("la GUI solo admite acciones administrativas predefinidas", () => {
  assert.match(server, /allowed = \{"setup", "check", "train", "start-vllm", "stop-vllm", "load-adapter", "unload-adapter"\}/);
  assert.match(server, /Dataset fuera de las carpetas permitidas/);
  assert.match(server, /x-fine-tune-token/i);
});

test("vLLM habilita LoRA dinámica únicamente detrás del puerto local", () => {
  assert.match(compose, /127\.0\.0\.1:8000:8000/);
  assert.match(compose, /VLLM_ALLOW_RUNTIME_LORA_UPDATING/);
});

test("el learning rate por defecto es válido para el control HTML", () => {
  assert.match(gui, /id="learningRate"[^>]+value="0\.0002"[^>]+step="any"/);
});

test("evaluation.jsonl se identifica como evaluación y no se ofrece para SFT", () => {
  assert.match(server, /EVALUATION_DATASETS = \{"evaluation\.jsonl"\}/);
  assert.match(server, /evaluation\.jsonl es un dataset de evaluación y no se puede usar para SFT/);
  assert.match(server, /if path\.name\.casefold\(\) in EVALUATION_DATASETS:\s+continue/);
});

test("entrenamiento y vLLM preparan adapters sin borrar su contenido", () => {
  assert.equal((server.match(/ensure_adapter_dir_writable\(job\)/g) || []).length, 2);
  assert.match(server, /docker,[\s\S]+"run",[\s\S]+"--rm",[\s\S]+"--user",[\s\S]+"0:0"/);
  assert.match(server, /DEFAULT_VLLM_IMAGE = "vllm\/vllm-openai:v0\.24\.0"/);
  assert.match(server, /"--entrypoint",\s+"\/bin\/sh"/);
  assert.match(server, /chown -R \{os\.getuid\(\)\}:\{os\.getgid\(\)\} \/adapters && chmod -R u\+rwX \/adapters/);
  assert.match(server, /owner=\{owner\}:\{group\}.*mode=\{mode\}/);
  assert.match(server, /sudo chown -R/);
  assert.doesNotMatch(server, /rmtree\(ADAPTER_DIR/);
});
