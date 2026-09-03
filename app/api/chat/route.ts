import { buildKnowledgeContext, type KnowledgeMode } from "@/lib/documents";
import { withKnowledge } from "@/lib/documents/prompt";
import { complete } from "@/lib/llm";
import { getAllowedModels, getAppToken, getLlmConfig } from "@/lib/llm/config";
import type { ChatMessage, ChatRole } from "@/lib/llm/types";

export const dynamic = "force-dynamic";

const MAX_MESSAGE_CHARS = 12_000;
const MAX_HISTORY_ITEMS = 20;
const MAX_HISTORY_CHARS = 8_000;
const MAX_KNOWLEDGE_HISTORY_CHARS = 4_000;
const WORKSPACE_ID_PATTERN = /^[A-Za-z0-9_-]{16,80}$/;

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

function limitHistory(history: ChatMessage[], maxCharacters: number): ChatMessage[] {
  const result: ChatMessage[] = [];
  let characters = 0;

  for (const item of [...history].reverse()) {
    if (characters + item.content.length > maxCharacters) break;
    result.unshift(item);
    characters += item.content.length;
  }

  return result;
}

function knowledgeMode(value: unknown): KnowledgeMode {
  return value === "rag" || value === "cag" ? value : "none";
}

function documentIds(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(value
    .filter((item): item is string => typeof item === "string" && /^[A-Za-z0-9-]{16,100}$/.test(item))
    .slice(0, 20))];
}

export async function POST(request: Request) {
  if (unauthorized(request)) {
    return Response.json({ error: "Se requiere una clave de acceso." }, { status: 401 });
  }

  let body: { message?: unknown; history?: unknown; mode?: unknown; documentIds?: unknown; model?: unknown };
  try {
    body = (await request.json()) as { message?: unknown; history?: unknown; mode?: unknown; documentIds?: unknown; model?: unknown };
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

  const mode = knowledgeMode(body.mode);
  const model = typeof body.model === "string" ? body.model.trim() : "";
  let config: ReturnType<typeof getLlmConfig>;
  let allowedModels: string[];
  try {
    config = getLlmConfig();
    allowedModels = getAllowedModels(config.model);
  } catch (error) {
    console.error("[llm-bridge] Invalid LLM configuration", error);
    return Response.json({ error: "La configuración del proveedor no es válida." }, { status: 500 });
  }
  if (model && !allowedModels.includes(model)) {
    return Response.json({ error: "El modelo o adaptador solicitado no está habilitado." }, { status: 400 });
  }
  const selectedDocumentIds = documentIds(body.documentIds);
  const workspaceId = request.headers.get("x-workspace-id")?.trim() ?? "";
  if (mode !== "none" && !WORKSPACE_ID_PATTERN.test(workspaceId)) {
    return Response.json({ error: "El espacio de documentos no es válido." }, { status: 400 });
  }

  const history = limitHistory(
    sanitizeHistory(body.history),
    mode === "none" ? MAX_HISTORY_CHARS : MAX_KNOWLEDGE_HISTORY_CHARS,
  );
  const messages: ChatMessage[] = [
    ...history,
    { role: "user", content: message },
  ];

  try {
    const knowledge = mode === "none"
      ? null
      : await buildKnowledgeContext(workspaceId, mode, message, selectedDocumentIds);
    const requestMessages = knowledge ? withKnowledge(messages, knowledge) : messages;
    const selectedModel = model || config.model;
    const answer = await complete(requestMessages, AbortSignal.timeout(config.timeoutMs), selectedModel);
    return Response.json({
      message: answer,
      provider: config.provider,
      model: selectedModel,
      mode,
      sources: knowledge?.sources ?? [],
      cacheHit: knowledge?.cacheHit ?? false,
      embeddingUsed: knowledge?.embeddingUsed ?? false,
      contextTruncated: knowledge?.truncated ?? false,
    });
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
