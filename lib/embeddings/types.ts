export type EmbeddingProviderName = "ollama" | "vllm";

export type EmbeddingInputType = "query" | "document";

export interface EmbeddingConfig {
  enabled: boolean;
  provider: EmbeddingProviderName;
  baseUrl: string;
  model: string;
  apiKey: string;
  batchSize: number;
  timeoutMs: number;
  queryPrefix: string;
  documentPrefix: string;
}
