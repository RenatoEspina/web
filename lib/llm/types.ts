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

export interface CompletionOptions {
  temperature?: number;
  maxTokens?: number;
}

export interface ProviderHealth {
  ok: boolean;
  status?: number;
}

export interface CompletionUsage {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
}

export interface CompletionResult {
  content: string;
  latencyMs: number;
  finishReason?: string;
  usage: CompletionUsage;
  tokensPerSecond?: number;
}

export interface LlmProvider {
  complete(messages: ChatMessage[], signal: AbortSignal, options?: CompletionOptions): Promise<CompletionResult>;
  health(signal: AbortSignal): Promise<ProviderHealth>;
}
