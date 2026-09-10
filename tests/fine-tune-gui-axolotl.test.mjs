import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

const basicLauncher = await readFile(new URL("../fine-tune-gui", import.meta.url), "utf8");
const axolotlLauncher = await readFile(new URL("../fine-tune-gui-axolotl", import.meta.url), "utf8");
const axolotlServer = await readFile(new URL("../trainer/gui_server_axolotl.py", import.meta.url), "utf8");
const axolotlTrain = await readFile(new URL("../trainer/train_axolotl.py", import.meta.url), "utf8");
const axolotlScript = await readFile(new URL("../scripts/train-adapter-axolotl.sh", import.meta.url), "utf8");

test("the original GUI remains the basic TRL/PEFT backend", () => {
  assert.match(basicLauncher, /FINE_TUNE_GUI_PORT:-3031/);
  assert.match(basicLauncher, /trainer\/gui_server\.py/);
  assert.doesNotMatch(basicLauncher, /gui_server_axolotl\.py/);
});

test("the Axolotl GUI uses an independent process and port", () => {
  assert.match(axolotlLauncher, /FINE_TUNE_AXOLOTL_GUI_PORT:-3032/);
  assert.match(axolotlLauncher, /trainer\/gui_server_axolotl\.py/);
  assert.match(axolotlServer, /\.venv-axolotl/);
  assert.match(axolotlServer, /train-adapter-axolotl\.sh/);
  assert.match(axolotlServer, /Backend: Axolotl/);
});

test("Axolotl training keeps QLoRA and assistant-only masking", () => {
  assert.match(axolotlTrain, /"adapter": "qlora"/);
  assert.match(axolotlTrain, /"roles_to_train": \["assistant"\]/);
  assert.match(axolotlTrain, /"train_on_eos": "turn"/);
  assert.match(axolotlTrain, /"sample_packing": False/);
  assert.match(axolotlTrain, /"backend": "axolotl"/);
  assert.match(axolotlScript, /trainer\/train_axolotl\.py/);
});
