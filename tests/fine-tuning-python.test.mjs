import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import test from "node:test";

const execute = promisify(execFile);

async function validate(contents) {
  const directory = await mkdtemp(join(tmpdir(), "llm-bridge-dataset-"));
  const dataset = join(directory, "dataset.jsonl");
  await writeFile(dataset, contents, "utf8");
  try {
    return await execute("python", ["trainer/validate_dataset.py", dataset], { cwd: process.cwd() });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

test("validador Python informa la línea del JSON inválido", async () => {
  await assert.rejects(validate('{"messages":'), (error) => {
    assert.match(error.stderr, /Línea 1: JSON inválido/);
    return true;
  });
});

test("validador Python rechaza mensajes que no son objetos", async () => {
  await assert.rejects(validate('{"messages":["texto",{"role":"assistant","content":"respuesta"}]}'), (error) => {
    assert.match(error.stderr, /Línea 1, mensaje 1: se esperaba un objeto/);
    return true;
  });
});
