function readNumber(name: string, fallback: number, min: number, max: number): number {
  if (typeof process === "undefined") return fallback;

  const value = Number(process.env[name]);
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Math.floor(value)));
}

export function getDocumentConfig() {
  return {
    maxPdfBytes: readNumber("RAG_MAX_PDF_BYTES", 10 * 1024 * 1024, 1_024, 25 * 1024 * 1024),
    maxPdfPages: readNumber("RAG_MAX_PDF_PAGES", 100, 1, 500),
    maxDocumentCharacters: readNumber("RAG_MAX_DOCUMENT_CHARACTERS", 400_000, 10_000, 2_000_000),
    maxDocuments: readNumber("RAG_MAX_DOCUMENTS", 10, 1, 50),
    maxWorkspaces: readNumber("RAG_MAX_WORKSPACES", 16, 1, 100),
    chunkSize: readNumber("RAG_CHUNK_SIZE", 1_200, 400, 4_000),
    chunkOverlap: readNumber("RAG_CHUNK_OVERLAP", 180, 0, 800),
    topK: readNumber("RAG_TOP_K", 4, 1, 12),
    maxRagContextCharacters: readNumber("RAG_MAX_CONTEXT_CHARACTERS", 7_000, 1_000, 30_000),
    maxCagContextCharacters: readNumber("CAG_MAX_CONTEXT_CHARACTERS", 8_000, 1_000, 30_000),
  };
}

export function safeDocumentName(name: string): string {
  const trimmed = name.trim().replace(/[\\/\u0000-\u001f]/g, "_");
  return trimmed.slice(0, 180) || "documento.pdf";
}
