import type { EmbeddingConfig, EmbeddingProviderName } from "./types";

const DEFAULTS: Record<EmbeddingProviderName, {
  baseUrl: string;
  model: string;
  queryPrefix: string;
  documentPrefix: string;
}> = {
  ollama: {
    baseUrl: "http://127.0.0.1:11434",
    model: "qwen3-embedding:4b",
    queryPrefix: "",
    documentPrefix: "",
  },
  vllm: {
    baseUrl: "http://127.0.0.1:8000",
    model: "intfloat/multilingual-e5-small",
    queryPrefix: "query: ",
    documentPrefix: "passage: ",
  },
};

function readEnv(name: string): string | undefined {
  if (typeof process === "undefined") return undefined;
  return process.env[name]?.trim() || undefined;
}

function readRawEnv(name: string): string | undefined {
  if (typeof process === "undefined") return undefined;
  return process.env[name];
}

function readBoolean(name: string, fallback: boolean): boolean {
  const value = readEnv(name)?.toLocaleLowerCase("en-US");
  if (!value) return fallback;
  if (["1", "true", "yes", "on"].includes(value)) return true;
  if (["0", "false", "no", "off"].includes(value)) return false;
  return fallback;
}

function readNumber(name: string, fallback: number, min: number, max: number): number {
  const value = Number(readEnv(name));
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Math.floor(value)));
}

export function getEmbeddingConfig(): EmbeddingConfig {
  const enabled = readBoolean("EMBEDDING_ENABLED", true);
  const requestedProvider = (readEnv("EMBEDDING_PROVIDER") ?? "ollama").toLocaleLowerCase("en-US");
  let provider: EmbeddingProviderName;

  if (requestedProvider === "ollama" || requestedProvider === "vllm") {
    provider = requestedProvider;
  } else if (enabled) {
    throw new Error("EMBEDDING_PROVIDER debe ser ollama o vllm.");
  } else {
    // A disabled index must remain a valid lexical-only configuration even if
    // a stale provider value remains in the environment.
    provider = "ollama";
  }

  const defaults = DEFAULTS[provider];

  return {
    enabled,
    provider,
    baseUrl: (readEnv("EMBEDDING_BASE_URL") ?? defaults.baseUrl).replace(/\/+$/, ""),
    model: readEnv("EMBEDDING_MODEL") ?? defaults.model,
    apiKey: readEnv("EMBEDDING_API_KEY") ?? (provider === "vllm" ? readEnv("LLM_API_KEY") ?? "" : ""),
    batchSize: readNumber("EMBEDDING_BATCH_SIZE", 16, 1, 64),
    timeoutMs: readNumber("EMBEDDING_TIMEOUT_MS", 120_000, 5_000, 600_000),
    queryPrefix: readRawEnv("EMBEDDING_QUERY_PREFIX") ?? defaults.queryPrefix,
    documentPrefix: readRawEnv("EMBEDDING_DOCUMENT_PREFIX") ?? defaults.documentPrefix,
  };
}

export function publicEmbeddingConfig() {
  const config = getEmbeddingConfig();
  return {
    enabled: config.enabled,
    provider: config.provider,
    model: config.model,
  };
}
