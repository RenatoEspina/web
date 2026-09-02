import { extractText, getDocumentProxy } from "unpdf";

import { embedTexts, getEmbeddingConfig } from "../embeddings";

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

    const embeddingConfig = getEmbeddingConfig();
    let embeddingProvider: string | undefined;
    let embeddingModel: string | undefined;
    let embeddingDimension: number | undefined;

    if (embeddingConfig.enabled) {
      try {
        const embeddings = await embedTexts(
          chunks.map((chunk) => chunk.text),
          "document",
          AbortSignal.timeout(embeddingConfig.timeoutMs),
        );

        if (!embeddings || embeddings.length !== chunks.length) {
          throw new Error("No se recibió un vector por cada fragmento.");
        }

        chunks.forEach((chunk, index) => {
          chunk.embedding = embeddings[index];
        });
        embeddingProvider = embeddingConfig.provider;
        embeddingModel = embeddingConfig.model;
        embeddingDimension = embeddings[0]?.length;
      } catch (error) {
        const detail = error instanceof Error ? ` ${error.message}` : "";
        throw new Error(
          `No fue posible generar embeddings para el PDF.${detail} ` +
          "Verifica que el proveedor y el modelo de embeddings estén disponibles.",
        );
      }
    }

    return {
      id,
      name: documentName,
      sizeBytes,
      pages: pdf.numPages,
      characters,
      chunks,
      createdAt: Date.now(),
      ...(embeddingProvider ? { embeddingProvider } : {}),
      ...(embeddingModel ? { embeddingModel } : {}),
      ...(embeddingDimension ? { embeddingDimension } : {}),
    };
  } finally {
    await pdf.cleanup();
  }
}
