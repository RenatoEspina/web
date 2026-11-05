import { endpoint } from "../llm/config";

import { getEmbeddingConfig } from "./config";
import type { EmbeddingConfig, EmbeddingInputType } from "./types";

type OllamaResponse = {
  embeddings?: unknown;
};

type VllmResponse = {
  data?: Array<{
    index?: unknown;
    embedding?: unknown;
  }>;
};

function prefixedInputs(texts: string[], config: EmbeddingConfig, inputType: EmbeddingInputType): string[] {
  const prefix = inputType === "query" ? config.queryPrefix : config.documentPrefix;
  return texts.map((text) => `${prefix}${text}`);
}

function vector(value: unknown): number[] | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  if (!value.every((item) => typeof item === "number" && Number.isFinite(item))) return null;
  return value as number[];
}

function validateVectors(value: unknown, expectedCount: number): number[][] {
  if (!Array.isArray(value) || value.length !== expectedCount) {
    throw new Error("El proveedor de embeddings devolvió una cantidad de vectores inesperada.");
  }

  const vectors = value.map(vector);
  if (vectors.some((item) => item === null)) {
    throw new Error("El proveedor de embeddings devolvió un vector inválido.");
  }

  const dimensions = vectors[0]?.length ?? 0;
  if (dimensions === 0 || vectors.some((item) => item?.length !== dimensions)) {
    throw new Error("El proveedor de embeddings devolvió vectores con dimensiones incompatibles.");
  }

  return vectors as number[][];
}

async function readJson(response: Response, provider: string): Promise<unknown> {
  const raw = await response.text();
  if (!response.ok) {
    console.error(`[llm-bridge] ${provider} embedding provider error`, response.status, raw.slice(0, 500));
    throw new Error(`El proveedor de embeddings respondió HTTP ${response.status}.`);
  }

  try {
    return JSON.parse(raw) as unknown;
  } catch {
    throw new Error("El proveedor de embeddings devolvió JSON inválido.");
  }
}

function headers(config: EmbeddingConfig): Record<string, string> {
  const result: Record<string, string> = { "Content-Type": "application/json" };
  if (config.apiKey) result.Authorization = `Bearer ${config.apiKey}`;
  return result;
}

async function embedWithOllama(
  texts: string[],
  config: EmbeddingConfig,
  signal: AbortSignal,
): Promise<number[][]> {
  const response = await fetch(endpoint(config.baseUrl, "/api/embed"), {
    method: "POST",
    headers: headers(config),
    body: JSON.stringify({ model: config.model, input: texts }),
    signal,
  });

  const data = await readJson(response, "Ollama") as OllamaResponse;
  return validateVectors(data.embeddings, texts.length);
}

async function embedWithVllm(
  texts: string[],
  config: EmbeddingConfig,
  signal: AbortSignal,
): Promise<number[][]> {
  const response = await fetch(endpoint(config.baseUrl, "/v1/embeddings"), {
    method: "POST",
    headers: headers(config),
    body: JSON.stringify({
      model: config.model,
      input: texts,
      encoding_format: "float",
    }),
    signal,
  });

  const data = await readJson(response, "vLLM") as VllmResponse;
  if (!Array.isArray(data.data)) {
    throw new Error("vLLM no devolvió el campo data de embeddings.");
  }

  const ordered = data.data
    .map((item, fallbackIndex) => ({
      index: typeof item.index === "number" ? item.index : fallbackIndex,
      embedding: item.embedding,
    }))
    .sort((left, right) => left.index - right.index)
    .map((item) => item.embedding);

  return validateVectors(ordered, texts.length);
}

export async function embedTexts(
  texts: string[],
  inputType: EmbeddingInputType,
  signal?: AbortSignal,
): Promise<number[][] | null> {
  if (texts.length === 0) return [];

  const config = getEmbeddingConfig();
  if (!config.enabled) return null;

  const inputs = prefixedInputs(texts, config, inputType);
  const vectors: number[][] = [];

  for (let start = 0; start < inputs.length; start += config.batchSize) {
    const batch = inputs.slice(start, start + config.batchSize);
    const requestSignal = signal ?? AbortSignal.timeout(config.timeoutMs);
    const batchVectors = config.provider === "ollama"
      ? await embedWithOllama(batch, config, requestSignal)
      : await embedWithVllm(batch, config, requestSignal);
    vectors.push(...batchVectors);
  }

  return validateVectors(vectors, texts.length);
}

export async function embedText(
  text: string,
  inputType: EmbeddingInputType,
  signal?: AbortSignal,
): Promise<number[] | null> {
  const result = await embedTexts([text], inputType, signal);
  return result?.[0] ?? null;
}

export type EmbeddingHealth = {
  enabled: boolean;
  available: boolean;
  error?: string;
};

type ModelListResponse = {
  models?: Array<{ name?: unknown; model?: unknown }>;
  data?: Array<{ id?: unknown }>;
};

function listedModels(provider: EmbeddingConfig["provider"], data: ModelListResponse): string[] {
  const names = provider === "ollama"
    ? (data.models ?? []).flatMap((item) => [item.name, item.model])
    : (data.data ?? []).map((item) => item.id);
  return names.filter((name): name is string => typeof name === "string");
}

/**
 * Checks that the embedding service answers and serves the configured model.
 * It never throws: indexing already degrades to lexical retrieval, so the
 * result only feeds the status shown in the interface.
 */
export async function checkEmbeddings(signal: AbortSignal): Promise<EmbeddingHealth> {
  let config: EmbeddingConfig;
  try {
    config = getEmbeddingConfig();
  } catch {
    return { enabled: true, available: false, error: "La configuración de embeddings no es válida." };
  }
  if (!config.enabled) return { enabled: false, available: false };

  const path = config.provider === "ollama" ? "/api/tags" : "/v1/models";
  let response: Response;
  try {
    response = await fetch(endpoint(config.baseUrl, path), {
      method: "GET",
      headers: headers(config),
      signal,
    });
  } catch {
    return { enabled: true, available: false, error: `${config.provider} no responde.` };
  }

  if (!response.ok) {
    return { enabled: true, available: false, error: `${config.provider} respondió HTTP ${response.status}.` };
  }

  let data: ModelListResponse;
  try {
    data = await response.json() as ModelListResponse;
  } catch {
    return { enabled: true, available: false, error: `${config.provider} devolvió JSON inválido.` };
  }

  const models = listedModels(config.provider, data ?? {});
  const served = models.includes(config.model) ||
    (config.provider === "ollama" && !config.model.includes(":") && models.includes(`${config.model}:latest`));
  return served
    ? { enabled: true, available: true }
    : { enabled: true, available: false, error: `${config.provider} no tiene el modelo ${config.model}.` };
}

export { getEmbeddingConfig, publicEmbeddingConfig } from "./config";
export type { EmbeddingConfig, EmbeddingInputType, EmbeddingProviderName } from "./types";
