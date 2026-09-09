import { endpoint } from "./config";
import type { LlmConfig } from "./types";

type VllmModelEntry = {
  id: string;
  parent?: string;
  root?: string;
};

type VllmModelsResponse = {
  data?: unknown;
};

export class RuntimeModelError extends Error {
  constructor(message: string, readonly status: 409 | 502) {
    super(message);
    this.name = "RuntimeModelError";
  }
}

function parseModelEntries(value: unknown): VllmModelEntry[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const record = item as Record<string, unknown>;
    if (typeof record.id !== "string" || !record.id.trim()) return [];
    return [{
      id: record.id,
      ...(typeof record.parent === "string" && record.parent ? { parent: record.parent } : {}),
      ...(typeof record.root === "string" && record.root ? { root: record.root } : {}),
    }];
  });
}

function verifyAdapterRuntimeIdentity(
  model: string,
  selected: VllmModelEntry,
  baseEntries: VllmModelEntry[],
  configuredBase: VllmModelEntry,
): void {
  // La procedencia de entrenamiento se valida en la GUI local antes de cargar
  // el LoRA y antes de incorporarlo a LLM_ADAPTER_MODELS. Este gateway puede
  // ejecutarse como Cloudflare Worker/Vinext y no tiene acceso al filesystem
  // del host, por lo que aquí solo se valida el estado que vLLM sirve ahora.
  if (!configuredBase.root || !selected.parent || !selected.root) {
    throw new RuntimeModelError(
      `No se pudo identificar la base servida para '${model}'. No se realizará inferencia con un LoRA en un estado runtime incompleto.`,
      409,
    );
  }

  const parent = baseEntries.find((entry) => entry.id === selected.parent);
  if (!parent?.root || parent.root !== configuredBase.root) {
    throw new RuntimeModelError(
      `El adaptador '${model}' está asociado a una base distinta de la configurada. No se realizará inferencia con un LoRA incompatible.`,
      409,
    );
  }
}

export async function validateRuntimeModelSelection(
  config: LlmConfig,
  selectedModel: string,
  signal: AbortSignal,
): Promise<VllmModelEntry | null> {
  if (config.provider !== "vllm") return null;

  const headers: Record<string, string> = {};
  if (config.apiKey) headers.Authorization = `Bearer ${config.apiKey}`;

  let response: Response;
  try {
    response = await fetch(endpoint(config.baseUrl, "/v1/models"), {
      method: "GET",
      headers,
      signal,
      cache: "no-store",
    });
  } catch (error) {
    if (error instanceof Error && error.name === "TimeoutError") throw error;
    throw new RuntimeModelError("No fue posible consultar los modelos cargados en vLLM.", 502);
  }

  if (!response.ok) {
    throw new RuntimeModelError(`vLLM no permitió verificar sus modelos cargados (HTTP ${response.status}).`, 502);
  }

  let payload: VllmModelsResponse;
  try {
    payload = await response.json() as VllmModelsResponse;
  } catch {
    throw new RuntimeModelError("vLLM devolvió una lista de modelos inválida.", 502);
  }

  const entries = parseModelEntries(payload.data);
  const baseEntries = entries.filter((entry) => !entry.parent);
  const configuredBase = baseEntries.find((entry) => entry.id === config.model);
  if (!configuredBase) {
    const served = baseEntries.map((entry) => entry.id).join(", ") || "ninguno";
    throw new RuntimeModelError(
      `El gateway espera el modelo base '${config.model}', pero vLLM sirve: ${served}. Reinicia o sincroniza el modelo antes de continuar.`,
      409,
    );
  }

  const selected = entries.find((entry) => entry.id === selectedModel);
  if (!selected) {
    throw new RuntimeModelError(
      `El modelo o adaptador '${selectedModel}' está permitido por el gateway pero no está cargado actualmente en vLLM.`,
      409,
    );
  }

  if (selectedModel !== config.model) {
    verifyAdapterRuntimeIdentity(selectedModel, selected, baseEntries, configuredBase);
  }

  return selected;
}
