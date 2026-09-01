export type ProviderName = "vllm" | "ollama";

export type ChatRole = "system" | "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export interface LlmConfig {
  provider: ProviderName;
  baseUrl: string;
  model: string;
  apiKey: string;
  maxTokens: number;
  temperature: number;
  timeoutMs: number;
}

export interface ProviderHealth {
  ok: boolean;
  status?: number;
}

export interface LlmProvider {
  complete(messages: ChatMessage[], signal: AbortSignal): Promise<string>;
  health(signal: AbortSignal): Promise<ProviderHealth>;
}
