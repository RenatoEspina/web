import type { LlmConfig, ProviderName } from "./types";

const DEFAULTS: Record<ProviderName, { baseUrl: string; model: string }> = {
  vllm: {
    baseUrl: "http://127.0.0.1:8000",
    model: "Qwen/Qwen3.5-0.8B",
  },
  ollama: {
    baseUrl: "http://127.0.0.1:11434",
    model: "llama3.2:1b-instruct-fp16",
  },
};

function readEnv(name: string): string | undefined {
  if (typeof process === "undefined") return undefined;
  return process.env[name]?.trim() || undefined;
}

function numberEnv(name: string, fallback: number, min: number, max: number): number {
  const value = Number(readEnv(name));
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

export function getLlmConfig(): LlmConfig {
  const requestedProvider = (readEnv("LLM_PROVIDER") ?? "vllm").toLowerCase();
  if (requestedProvider !== "vllm" && requestedProvider !== "ollama") {
    throw new Error("LLM_PROVIDER must be either vllm or ollama.");
  }

  const provider = requestedProvider as ProviderName;
  const defaults = DEFAULTS[provider];

  return {
    provider,
    baseUrl: (readEnv("LLM_BASE_URL") ?? defaults.baseUrl).replace(/\/+$/, ""),
    model: readEnv("LLM_MODEL") ?? defaults.model,
    apiKey: readEnv("LLM_API_KEY") ?? "",
    maxTokens: numberEnv("LLM_MAX_TOKENS", 512, 1, 4096),
    temperature: numberEnv("LLM_TEMPERATURE", 1.0, 0, 2),
    timeoutMs: numberEnv("LLM_TIMEOUT_MS", 120_000, 5_000, 600_000),
  };
}

export function getAppToken(): string {
  return readEnv("APP_TOKEN") ?? "";
}

export function publicLlmConfig() {
  const config = getLlmConfig();
  return {
    provider: config.provider,
    model: config.model,
    authRequired: Boolean(getAppToken()),
  };
}

export function endpoint(baseUrl: string, path: string): string {
  return new URL(path.replace(/^\/+/, ""), `${baseUrl}/`).toString();
}
