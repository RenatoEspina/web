import { errorResponse, isAuthorized } from "@/lib/http/request";
import { checkProvider } from "@/lib/llm";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  if (!isAuthorized(request)) {
    return errorResponse("Se requiere una clave de acceso.", 401);
  }

  try {
    const timeout = AbortSignal.timeout(5_000);
    const result = await checkProvider(timeout);
    return Response.json(result, { status: result.ok ? 200 : 503 });
  } catch (error) {
    console.error("[llm-bridge] Provider health check failed", error);
    return Response.json({ ok: false, error: "El proveedor no está disponible." }, { status: 503 });
  }
}
