import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

const server = await readFile(new URL("../trainer/gui_server.py", import.meta.url), "utf8");
const launcher = await readFile(new URL("../fine-tune-gui", import.meta.url), "utf8");
const compose = await readFile(new URL("../docker-compose.yml", import.meta.url), "utf8");

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
