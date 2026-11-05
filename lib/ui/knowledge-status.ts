import type { KnowledgeMode } from "@/lib/documents";

/**
 * `enabled` comes from configuration; `available` comes from `/api/health`
 * (`null` while it has not been checked). Only a real check may say the
 * service is available.
 */
export function embeddingStatusLabel(enabled: boolean, available: boolean | null): string {
  if (!enabled) return "embeddings desactivados";
  if (available === null) return "embeddings habilitados";
  return available ? "embeddings disponibles" : "embeddings no disponibles";
}

/**
 * The backend treats an empty selection as `mode=none`, so the hint must not
 * suggest that RAG or CAG will be used without selected documents.
 */
export function knowledgeHint(mode: KnowledgeMode, selectedCount: number): string {
  if (mode !== "none" && selectedCount === 0) {
    return `Selecciona al menos un documento para usar ${mode.toUpperCase()}. Mientras tanto, el chat responderá sin consultar los PDF.`;
  }
  if (mode === "cag") {
    return "CAG reutiliza el contexto documental en orden y lo trunca si supera el límite configurado.";
  }
  if (mode === "rag") {
    return "RAG combina embeddings semánticos y coincidencia léxica para recuperar los fragmentos más relacionados.";
  }
  return "El chat responderá sin consultar los PDF.";
}
