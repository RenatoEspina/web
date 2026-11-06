import { checkEmbeddings } from "@/lib/embeddings";
import { errorResponse, isAuthorized } from "@/lib/http/request";
import { checkProvider } from "@/lib/llm";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  if (!isAuthorized(request)) {
    return errorResponse("Se requiere una clave de acceso.", 401);
  }

  const timeout = AbortSignal.timeout(5_000);
  // The embedding check never throws and does not affect the status code:
  // the gateway stays usable with lexical retrieval when embeddings are down.
  const embeddingCheck = checkEmbeddings(timeout);

  try {
    const result = await checkProvider(timeout);
    return Response.json(
      { ...result, embedding: await embeddingCheck },
      { status: result.ok ? 200 : 503 },
    );
  } catch (error) {
    console.error("[llm-bridge] Provider health check failed", error);
    return Response.json(
      { ok: false, error: "El proveedor no está disponible.", embedding: await embeddingCheck },
      { status: 503 },
    );
  }
}
