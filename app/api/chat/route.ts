import { buildKnowledgeContext, type KnowledgeMode } from "@/lib/documents";
import { withKnowledge } from "@/lib/documents/prompt";
import { errorResponse, isAuthorized, workspaceIdFrom } from "@/lib/http/request";
import { complete } from "@/lib/llm";
import { getAllowedModels, getLlmConfig } from "@/lib/llm/config";
import {
  TERRARIA_OUT_OF_SCOPE_MESSAGE,
  isTerrariaModel,
  parseTerrariaScopeDecision,
  terrariaScopeMessages,
  withMasterPrompt,
} from "@/lib/llm/masterPrompt";
import { RuntimeModelError, validateRuntimeModelSelection } from "@/lib/llm/runtime";
import type { ChatMessage, ChatRole } from "@/lib/llm/types";

export const dynamic = "force-dynamic";

const MAX_MESSAGE_CHARS = 12_000;
const MAX_HISTORY_ITEMS = 20;
const MAX_HISTORY_CHARS = 8_000;
const MAX_KNOWLEDGE_HISTORY_CHARS = 4_000;

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

function documentIds(value: unknown): string[] | undefined {
  if (value === undefined) return undefined;
  if (!Array.isArray(value)) return [];
  return [...new Set(value
    .filter((item): item is string => typeof item === "string" && /^[A-Za-z0-9-]{16,100}$/.test(item))
    .slice(0, 20))];
}

export async function POST(request: Request) {
  if (!isAuthorized(request)) {
    return errorResponse("Se requiere una clave de acceso.", 401);
  }

  let body: { message?: unknown; history?: unknown; mode?: unknown; documentIds?: unknown; model?: unknown };
  try {
    body = (await request.json()) as { message?: unknown; history?: unknown; mode?: unknown; documentIds?: unknown; model?: unknown };
  } catch {
    return errorResponse("El cuerpo de la petición no es JSON válido.", 400);
  }

  const message = typeof body.message === "string" ? body.message.trim() : "";
  if (!message) {
    return errorResponse("Escribe un mensaje antes de enviarlo.", 400);
  }
  if (message.length > MAX_MESSAGE_CHARS) {
    return errorResponse(`El mensaje supera el límite de ${MAX_MESSAGE_CHARS} caracteres.`, 413);
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
    return errorResponse("La configuración del proveedor no es válida.", 500);
  }
  if (model && !allowedModels.includes(model)) {
    return errorResponse("El modelo o adaptador solicitado no está habilitado.", 400);
  }

  const selectedDocumentIds = documentIds(body.documentIds);
  const workspaceId = workspaceIdFrom(request);
  if (mode !== "none" && !workspaceId) {
    return errorResponse("El espacio de documentos no es válido.", 400);
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
    const selectedModel = model || config.model;
    const inferenceSignal = AbortSignal.timeout(config.timeoutMs);
    let scope: {
      checked: true;
      allowed: boolean;
      classifierModel: string;
      latencyMs: number;
    } | undefined;

    if (isTerrariaModel(selectedModel)) {
      const scopeCompletion = await complete(
        terrariaScopeMessages(messages),
        inferenceSignal,
        config.model,
      );
      const allowed = parseTerrariaScopeDecision(scopeCompletion.content);
      scope = {
        checked: true,
        allowed,
        classifierModel: config.model,
        latencyMs: scopeCompletion.latencyMs,
      };

      if (!allowed) {
        return Response.json({
          message: TERRARIA_OUT_OF_SCOPE_MESSAGE,
          provider: config.provider,
          model: selectedModel,
          mode: "none" satisfies KnowledgeMode,
          sources: [],
          cacheHit: false,
          embeddingUsed: false,
          contextTruncated: false,
          scope,
          inference: {
            latencyMs: 0,
            finishReason: "scope_rejected",
            promptTokens: 0,
            completionTokens: 0,
            totalTokens: 0,
            tokensPerSecond: undefined,
          },
        });
      }
    }

    await validateRuntimeModelSelection(config, selectedModel, inferenceSignal);

    const knowledge = mode === "none"
      ? null
      : await buildKnowledgeContext(workspaceId!, mode, message, selectedDocumentIds);
    const knowledgeMessages = knowledge ? withKnowledge(messages, knowledge) : messages;
    const requestMessages = withMasterPrompt(knowledgeMessages, selectedModel);
    const completion = await complete(requestMessages, inferenceSignal, selectedModel);
    const effectiveMode: KnowledgeMode = knowledge?.mode ?? "none";

    return Response.json({
      message: completion.content,
      provider: config.provider,
      model: selectedModel,
      mode: effectiveMode,
      sources: knowledge?.sources ?? [],
      cacheHit: knowledge?.cacheHit ?? false,
      embeddingUsed: knowledge?.embeddingUsed ?? false,
      contextTruncated: knowledge?.truncated ?? false,
      scope,
      inference: {
        latencyMs: completion.latencyMs,
        finishReason: completion.finishReason,
        promptTokens: completion.usage.promptTokens,
        completionTokens: completion.usage.completionTokens,
        totalTokens: completion.usage.totalTokens,
        tokensPerSecond: completion.tokensPerSecond,
      },
    });
  } catch (error) {
    console.error("[llm-bridge] Chat request failed", error);
    if (error instanceof RuntimeModelError) {
      return errorResponse(error.message, error.status);
    }
    if (error instanceof Error && error.name === "TimeoutError") {
      return errorResponse("El modelo tardó demasiado en responder.", 504);
    }
    return errorResponse("No fue posible obtener una respuesta del proveedor configurado.", 502);
  }
}
