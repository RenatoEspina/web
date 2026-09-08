import { endpoint } from "./config";
import type { ChatMessage, CompletionResult, LlmConfig, LlmProvider, ProviderHealth } from "./types";

type OpenAiResponse = {
  choices?: Array<{
    finish_reason?: unknown;
    message?: {
      content?: unknown;
    };
  }>;
  usage?: {
    prompt_tokens?: unknown;
    completion_tokens?: unknown;
    total_tokens?: unknown;
  };
};

function tokenCount(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined;
}

export class OpenAiCompatibleProvider implements LlmProvider {
  constructor(private readonly config: LlmConfig) {}

  async complete(messages: ChatMessage[], signal: AbortSignal): Promise<CompletionResult> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.config.apiKey) {
      headers.Authorization = `Bearer ${this.config.apiKey}`;
    }

    const started = performance.now();
    const response = await fetch(endpoint(this.config.baseUrl, "/v1/chat/completions"), {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: this.config.model,
        messages,
        temperature: this.config.temperature,
        max_tokens: this.config.maxTokens,
        stream: false,
      }),
      signal,
    });
    const latencyMs = Math.max(0, Math.round(performance.now() - started));

    const raw = await response.text();
    if (!response.ok) {
      console.error("[llm-bridge] OpenAI-compatible provider error", response.status, raw.slice(0, 500));
      throw new Error(`Provider returned HTTP ${response.status}.`);
    }

    let data: OpenAiResponse;
    try {
      data = JSON.parse(raw) as OpenAiResponse;
    } catch {
      throw new Error("Provider returned invalid JSON.");
    }

    const choice = data.choices?.[0];
    const content = choice?.message?.content;
    if (typeof content !== "string") {
      throw new Error("Provider response did not contain assistant text.");
    }

    const promptTokens = tokenCount(data.usage?.prompt_tokens);
    const completionTokens = tokenCount(data.usage?.completion_tokens);
    const totalTokens = tokenCount(data.usage?.total_tokens);
    const finishReason = typeof choice?.finish_reason === "string" ? choice.finish_reason : undefined;
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
    const response = await fetch(endpoint(this.config.baseUrl, "/health"), {
      method: "GET",
      signal,
    });
    return { ok: response.ok, status: response.status };
  }
}
