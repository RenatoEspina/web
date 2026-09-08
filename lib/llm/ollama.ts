import { endpoint } from "./config";
import type { ChatMessage, CompletionResult, LlmConfig, LlmProvider, ProviderHealth } from "./types";

type OllamaResponse = {
  message?: {
    content?: unknown;
  };
  done_reason?: unknown;
  prompt_eval_count?: unknown;
  eval_count?: unknown;
};

function tokenCount(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined;
}

export class OllamaProvider implements LlmProvider {
  constructor(private readonly config: LlmConfig) {}

  async complete(messages: ChatMessage[], signal: AbortSignal): Promise<CompletionResult> {
    const started = performance.now();
    const response = await fetch(endpoint(this.config.baseUrl, "/api/chat"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: this.config.model,
        messages,
        stream: false,
        options: {
          temperature: this.config.temperature,
          num_predict: this.config.maxTokens,
        },
      }),
      signal,
    });
    const latencyMs = Math.max(0, Math.round(performance.now() - started));

    const raw = await response.text();
    if (!response.ok) {
      console.error("[llm-bridge] Ollama provider error", response.status, raw.slice(0, 500));
      throw new Error(`Provider returned HTTP ${response.status}.`);
    }

    let data: OllamaResponse;
    try {
      data = JSON.parse(raw) as OllamaResponse;
    } catch {
      throw new Error("Provider returned invalid JSON.");
    }

    const content = data.message?.content;
    if (typeof content !== "string") {
      throw new Error("Provider response did not contain assistant text.");
    }

    const promptTokens = tokenCount(data.prompt_eval_count);
    const completionTokens = tokenCount(data.eval_count);
    const totalTokens = promptTokens !== undefined && completionTokens !== undefined
      ? promptTokens + completionTokens
      : undefined;
    const finishReason = typeof data.done_reason === "string" ? data.done_reason : undefined;
    const tokensPerSecond = completionTokens !== undefined && latencyMs > 0
      ? Number((completionTokens / (latencyMs / 1000)).toFixed(2))
      : undefined;

    return {
      content,
      latencyMs,
      finishReason,
      usage: {
        promptTokens,
        completionTokens,
        totalTokens,
      },
      tokensPerSecond,
    };
  }

  async health(signal: AbortSignal): Promise<ProviderHealth> {
    const response = await fetch(endpoint(this.config.baseUrl, "/api/tags"), {
      method: "GET",
      signal,
    });
    return { ok: response.ok, status: response.status };
  }
}
