import { endpoint } from "./config";
import type { ChatMessage, LlmConfig, LlmProvider, ProviderHealth } from "./types";

type OpenAiResponse = {
  choices?: Array<{
    message?: {
      content?: unknown;
    };
  }>;
};

export class OpenAiCompatibleProvider implements LlmProvider {
  constructor(private readonly config: LlmConfig) {}

  async complete(messages: ChatMessage[], signal: AbortSignal): Promise<string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.config.apiKey) {
      headers.Authorization = `Bearer ${this.config.apiKey}`;
    }

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

    const content = data.choices?.[0]?.message?.content;
    if (typeof content !== "string") {
      throw new Error("Provider response did not contain assistant text.");
    }

    return content;
  }

  async health(signal: AbortSignal): Promise<ProviderHealth> {
    const response = await fetch(endpoint(this.config.baseUrl, "/health"), {
      method: "GET",
      signal,
    });
    return { ok: response.ok, status: response.status };
  }
}
