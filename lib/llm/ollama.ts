import { endpoint } from "./config";
import type { ChatMessage, LlmConfig, LlmProvider, ProviderHealth } from "./types";

type OllamaResponse = {
  message?: {
    content?: unknown;
  };
};

export class OllamaProvider implements LlmProvider {
  constructor(private readonly config: LlmConfig) {}

  async complete(messages: ChatMessage[], signal: AbortSignal): Promise<string> {
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

    return content;
  }

  async health(signal: AbortSignal): Promise<ProviderHealth> {
    const response = await fetch(endpoint(this.config.baseUrl, "/api/tags"), {
      method: "GET",
      signal,
    });
    return { ok: response.ok, status: response.status };
  }
}
