import { extractText, getDocumentProxy } from "unpdf";

import { getDocumentConfig, safeDocumentName } from "./config";
import { createChunks, normalizeExtractedPages } from "./chunking";
import type { IndexedDocument } from "./types";

export async function indexPdf(data: Uint8Array, originalName: string, sizeBytes: number): Promise<IndexedDocument> {
  const config = getDocumentConfig();
  const documentName = safeDocumentName(originalName);

  if (sizeBytes > config.maxPdfBytes) {
    throw new Error(`El PDF supera el límite de ${Math.round(config.maxPdfBytes / 1024 / 1024)} MB.`);
  }

  const pdf = await getDocumentProxy(data);
  try {
    if (pdf.numPages > config.maxPdfPages) {
      throw new Error(`El PDF supera el límite de ${config.maxPdfPages} páginas.`);
    }

    const extracted = await extractText(pdf, { mergePages: false });
    const pages = normalizeExtractedPages(extracted.text, config.maxDocumentCharacters);
    const characters = pages.reduce((total, page) => total + page.length, 0);

    if (characters < 20) {
      throw new Error("No se encontró texto seleccionable. Los PDF escaneados requieren OCR.");
    }

    const id = crypto.randomUUID();
    const chunks = createChunks(id, documentName, pages);
    if (chunks.length === 0) {
      throw new Error("No fue posible crear fragmentos de texto para este PDF.");
    }

    return {
      id,
      name: documentName,
      sizeBytes,
      pages: pdf.numPages,
      characters,
      chunks,
      createdAt: Date.now(),
    };
  } finally {
    await pdf.cleanup();
  }
}
