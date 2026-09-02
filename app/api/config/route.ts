import { publicLlmConfig } from "@/lib/llm/config";
import { publicEmbeddingConfig } from "@/lib/embeddings";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return Response.json({
      ...publicLlmConfig(),
      embedding: publicEmbeddingConfig(),
    });
  } catch (error) {
    console.error("[llm-bridge] Invalid configuration", error);
    return Response.json(
      { error: "La configuración del gateway no es válida." },
      { status: 500 },
    );
  }
}
