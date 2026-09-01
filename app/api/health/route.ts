import { checkProvider } from "@/lib/llm";
import { getAppToken } from "@/lib/llm/config";

export const dynamic = "force-dynamic";

function authorized(request: Request): boolean {
  const appToken = getAppToken();
  if (!appToken) return true;

  const authorization = request.headers.get("authorization") ?? "";
  const suppliedToken = authorization.startsWith("Bearer ")
    ? authorization.slice("Bearer ".length).trim()
    : request.headers.get("x-app-token") ?? "";

  return suppliedToken === appToken;
}

export async function GET(request: Request) {
  if (!authorized(request)) {
    return Response.json({ error: "Se requiere una clave de acceso." }, { status: 401 });
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
