import { getAllowedModels, getLlmConfig } from "./config";
import { OllamaProvider } from "./ollama";
import { OpenAiCompatibleProvider } from "./openai-compatible";
import type { ChatMessage, LlmProvider } from "./types";

export function getProvider(model?: string): LlmProvider {
  const config = getLlmConfig();
  const selectedModel = model?.trim() || config.model;
  if (!getAllowedModels(config.model).includes(selectedModel)) {
    throw new Error("El modelo o adaptador solicitado no está habilitado.");
  }
  const selectedConfig = { ...config, model: selectedModel };
  return config.provider === "ollama"
    ? new OllamaProvider(selectedConfig)
    : new OpenAiCompatibleProvider(selectedConfig);
}

export async function complete(messages: ChatMessage[], signal: AbortSignal, model?: string): Promise<string> {
  return getProvider(model).complete(messages, signal);
}

export async function checkProvider(signal: AbortSignal) {
  const config = getLlmConfig();
  const provider = config.provider === "ollama"
    ? new OllamaProvider(config)
    : new OpenAiCompatibleProvider(config);
  const health = await provider.health(signal);

  return {
    ok: health.ok,
    provider: config.provider,
    model: config.model,
  };
}

export type { ChatMessage } from "./types";
