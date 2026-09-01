import { complete } from "@/lib/llm";
import { getAppToken, getLlmConfig } from "@/lib/llm/config";
import type { ChatMessage, ChatRole } from "@/lib/llm/types";

export const dynamic = "force-dynamic";

const MAX_MESSAGE_CHARS = 12_000;
const MAX_HISTORY_ITEMS = 20;

function unauthorized(request: Request): boolean {
  const appToken = getAppToken();
  if (!appToken) return false;

  const authorization = request.headers.get("authorization") ?? "";
  const suppliedToken = authorization.startsWith("Bearer ")
    ? authorization.slice("Bearer ".length).trim()
    : request.headers.get("x-app-token") ?? "";

  return suppliedToken !== appToken;
}

function isRole(value: unknown): value is Exclude<ChatRole, "system"> {
  return value === "user" || value === "assistant";
}

function sanitizeHistory(value: unknown): ChatMessage[] {
  if (!Array.isArray(value)) return [];

  return value
    .slice(-MAX_HISTORY_ITEMS)
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
    .map((item) => ({
      role: item.role,
      content: typeof item.content === "string" ? item.content.trim() : "",
    }))
    .filter((item): item is ChatMessage => isRole(item.role) && Boolean(item.content))
    .filter((item) => item.content.length <= MAX_MESSAGE_CHARS);
}

export async function POST(request: Request) {
  if (unauthorized(request)) {
    return Response.json({ error: "Se requiere una clave de acceso." }, { status: 401 });
  }

  let body: { message?: unknown; history?: unknown };
  try {
    body = (await request.json()) as { message?: unknown; history?: unknown };
  } catch {
    return Response.json({ error: "El cuerpo de la petición no es JSON válido." }, { status: 400 });
  }

  const message = typeof body.message === "string" ? body.message.trim() : "";
  if (!message) {
    return Response.json({ error: "Escribe un mensaje antes de enviarlo." }, { status: 400 });
  }
  if (message.length > MAX_MESSAGE_CHARS) {
    return Response.json({ error: `El mensaje supera el límite de ${MAX_MESSAGE_CHARS} caracteres.` }, { status: 413 });
  }

  const messages: ChatMessage[] = [
    ...sanitizeHistory(body.history),
    { role: "user", content: message },
  ];

  try {
    const config = getLlmConfig();
    const answer = await complete(messages, AbortSignal.timeout(config.timeoutMs));
    return Response.json({ message: answer, provider: config.provider });
  } catch (error) {
    console.error("[llm-bridge] Chat request failed", error);
    if (error instanceof Error && error.name === "TimeoutError") {
      return Response.json({ error: "El modelo tardó demasiado en responder." }, { status: 504 });
    }
    return Response.json(
      { error: "No fue posible obtener una respuesta del proveedor configurado." },
      { status: 502 },
    );
  }
}
