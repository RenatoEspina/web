import type { DatasetValidation, TrainingExample, TrainingMessage, TrainingRole } from "./types";

const ROLES = new Set<TrainingRole>(["system", "user", "assistant"]);
const MAX_EXAMPLES = 50_000;
const MAX_MESSAGES_PER_EXAMPLE = 64;
const MAX_MESSAGE_CHARACTERS = 32_000;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function parseTrainingExample(value: unknown, line: number): TrainingExample {
  if (!isRecord(value) || !Array.isArray(value.messages)) throw new Error(`Línea ${line}: se esperaba un objeto con un arreglo messages.`);
  if (value.messages.length < 2 || value.messages.length > MAX_MESSAGES_PER_EXAMPLE) throw new Error(`Línea ${line}: messages debe contener entre 2 y ${MAX_MESSAGES_PER_EXAMPLE} elementos.`);
  const messages: TrainingMessage[] = value.messages.map((item, index) => {
    if (!isRecord(item) || !ROLES.has(item.role as TrainingRole) || typeof item.content !== "string") throw new Error(`Línea ${line}, mensaje ${index + 1}: role o content no es válido.`);
    const content = item.content.trim();
    if (!content || content.length > MAX_MESSAGE_CHARACTERS) throw new Error(`Línea ${line}, mensaje ${index + 1}: content está vacío o supera ${MAX_MESSAGE_CHARACTERS} caracteres.`);
    return { role: item.role as TrainingRole, content };
  });
  if (!messages.some((message) => message.role === "user")) throw new Error(`Línea ${line}: falta al menos un mensaje user.`);
  if (messages.at(-1)?.role !== "assistant") throw new Error(`Línea ${line}: el último mensaje debe ser assistant.`);
  return { messages };
}

export function parseJsonlDataset(text: string): TrainingExample[] {
  const examples: TrainingExample[] = [];
  for (const [index, raw] of text.split(/\r?\n/).entries()) {
    const source = raw.trim();
    if (!source) continue;
    if (examples.length >= MAX_EXAMPLES) throw new Error(`El dataset supera ${MAX_EXAMPLES} ejemplos.`);
    let value: unknown;
    try { value = JSON.parse(source); } catch { throw new Error(`Línea ${index + 1}: JSON inválido.`); }
    examples.push(parseTrainingExample(value, index + 1));
  }
  if (!examples.length) throw new Error("El dataset no contiene ejemplos.");
  return examples;
}

export function validateJsonlDataset(text: string): DatasetValidation {
  try {
    const examples = parseJsonlDataset(text);
    let messages = 0;
    let characters = 0;
    for (const example of examples) {
      messages += example.messages.length;
      for (const message of example.messages) characters += message.content.length;
    }
    return { valid: true, examples: examples.length, messages, characters, errors: [] };
  } catch (error) {
    return { valid: false, examples: 0, messages: 0, characters: 0, errors: [error instanceof Error ? error.message : "Dataset inválido."] };
  }
}
