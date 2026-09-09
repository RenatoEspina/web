import { readFileSync, realpathSync } from "node:fs";
import { isAbsolute, relative, resolve } from "node:path";

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

function verifyAdapterTrainingBase(model: string, runtimeRoot: string | undefined, base: string): void {
  try {
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(model) || !runtimeRoot) throw new Error("Nombre o ruta de adaptador inválidos.");
    const adaptersRoot = realpathSync(resolve(process.cwd(), "adapters"));
    const original = resolve(adaptersRoot, model);
    // Docker mounts the local adapters directory at /adapters. Also support
    // vLLM started on the host with an absolute path inside that directory.
    const loaded = runtimeRoot.startsWith("/adapters/")
      ? resolve(adaptersRoot, runtimeRoot.slice("/adapters/".length))
      : resolve(runtimeRoot);
    const readMetadata = (directory: string, filename: string): Record<string, unknown> => {
      const path = realpathSync(resolve(directory, filename));
      const within = relative(adaptersRoot, path);
      if (within.startsWith("..") || isAbsolute(within)) throw new Error("Metadatos fuera de adapters/.");
      const data: unknown = JSON.parse(readFileSync(path, "utf-8"));
      if (!data || typeof data !== "object" || Array.isArray(data)) throw new Error("Metadatos inválidos.");
      return data as Record<string, unknown>;
    };
    // Check the runtime copy too: an export may predate a replaced local adapter.
    for (const directory of new Set([original, loaded])) {
      const metadata = readMetadata(directory, "adapter_config.json");
      if (metadata.base_model_name_or_path !== base) throw new Error("adapter_config.json declara otra base o no la especifica.");
    }
    let manifest: Record<string, unknown> | undefined;
    try {
      manifest = readMetadata(original, "manifest.json");
    } catch (error) {
      // Legacy PEFT adapters can lack a manifest; adapter_config is mandatory.
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    if (manifest && "baseModel" in manifest && manifest.baseModel !== base) throw new Error("El manifest declara otra base.");
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Metadatos no disponibles.";
    throw new RuntimeModelError(
      `No se pudo verificar que '${model}' fue entrenado sobre '${base}'. No se realizará inferencia con un LoRA incompatible. ${detail}`,
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
    const parent = baseEntries.find((entry) => entry.id === selected.parent);
    if (!configuredBase.root || !parent || parent.root !== configuredBase.root) {
      throw new RuntimeModelError(
        `No se pudo identificar la base servida para '${selectedModel}'. No se realizará inferencia con un LoRA incompatible.`,
        409,
      );
    }
    verifyAdapterTrainingBase(selectedModel, selected.root, configuredBase.root);
  }

  return selected;
}
