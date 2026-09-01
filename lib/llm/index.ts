import { getLlmConfig } from "./config";
import { OllamaProvider } from "./ollama";
import { OpenAiCompatibleProvider } from "./openai-compatible";
import type { ChatMessage, LlmProvider } from "./types";

export function getProvider(): LlmProvider {
  const config = getLlmConfig();
  return config.provider === "ollama"
    ? new OllamaProvider(config)
    : new OpenAiCompatibleProvider(config);
}

export async function complete(messages: ChatMessage[], signal: AbortSignal): Promise<string> {
  return getProvider().complete(messages, signal);
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
