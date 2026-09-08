import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

const trainer = await readFile(new URL("../trainer/train.py", import.meta.url), "utf8");
const trainingScript = await readFile(new URL("../scripts/train-adapter.sh", import.meta.url), "utf8");
const launcher = await readFile(new URL("../fine-tune-gui", import.meta.url), "utf8");
const builder = await readFile(new URL("../trainer/corpora/terraria/build.py", import.meta.url), "utf8");
const metadata = JSON.parse(await readFile(new URL("../trainer/corpora/terraria/data/metadata.json", import.meta.url), "utf8"));

test("SFT puede usar validación separada y selecciona el menor eval_loss", () => {
  assert.match(trainer, /--validation-dataset/);
  assert.match(trainer, /eval_dataset=validation_dataset/);
  assert.match(trainer, /eval_strategy="epoch" if use_validation else "no"/);
  assert.match(trainer, /load_best_model_at_end=use_validation/);
  assert.match(trainer, /metric_for_best_model="eval_loss" if use_validation else None/);
  assert.match(trainer, /greater_is_better=False if use_validation else None/);
});

test("el manifiesto conserva métricas de calidad de validación", () => {
  assert.match(trainer, /metric_key_prefix="baseline"/);
  assert.match(trainer, /metric_key_prefix="validation"/);
  assert.match(trainer, /qualitySelection/);
  assert.match(trainer, /relativeLossImprovement/);
  assert.match(trainer, /perplexity/);
  assert.match(trainer, /best_model_checkpoint/);
});

test("Terraria conecta automáticamente su validation set sin contaminar evaluación final", () => {
  assert.match(trainingScript, /trainer\/corpora\/terraria\/validation\.jsonl/);
  assert.match(trainingScript, /--validation-dataset/);
  assert.doesNotMatch(trainingScript, /terraria\/evaluation\.jsonl.*--validation-dataset/);
  assert.match(builder, /"train": \[\], "validation": \[\], "evaluation": \[\]/);
  assert.match(builder, /split in \{"train", "validation"\}/);
});

test("la GUI materializa el corpus inglés versionado antes de mostrar datasets", () => {
  assert.match(launcher, /trainer\/corpora\/terraria\/build\.py/);
  assert.match(builder, /DATA_DIR = DIRECTORY \/ "data"/);
  assert.match(builder, /SOURCE_GLOB = "sources-\*\.json"/);
  assert.match(metadata.system, /in English/);
  assert.match(metadata.split_policy, /Validation may be used for checkpoint selection/);
});
