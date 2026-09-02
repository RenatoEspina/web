import { getDocumentConfig } from "./config";
import type { DocumentChunk } from "./types";

function normalizeText(value: string): string {
  return value
    .replace(/\u00ad/g, "")
    .replace(/-\s+/g, "")
    .replace(/\s+/gu, " ")
    .trim();
}

function findBreak(text: string, desiredEnd: number, minimumEnd: number): number {
  if (desiredEnd >= text.length) return text.length;

  const candidates = [
    text.lastIndexOf(". ", desiredEnd),
    text.lastIndexOf("; ", desiredEnd),
    text.lastIndexOf(": ", desiredEnd),
    text.lastIndexOf(" ", desiredEnd),
  ];

  return candidates.find((position) => position >= minimumEnd) ?? desiredEnd;
}

function splitPage(text: string, chunkSize: number, overlap: number): string[] {
  const normalized = normalizeText(text);
  if (!normalized) return [];
  if (normalized.length <= chunkSize) return [normalized];

  const chunks: string[] = [];
  let start = 0;

  while (start < normalized.length) {
    const desiredEnd = Math.min(normalized.length, start + chunkSize);
    const end = findBreak(normalized, desiredEnd, start + Math.floor(chunkSize * 0.6));
    const chunk = normalized.slice(start, end).trim();
    if (chunk) chunks.push(chunk);

    if (end >= normalized.length) break;
    const nextStart = Math.max(start + 1, end - overlap);
    start = nextStart;
  }

  return chunks;
}

export function createChunks(
  documentId: string,
  documentName: string,
  pages: string[],
): DocumentChunk[] {
  const config = getDocumentConfig();
  const chunks: DocumentChunk[] = [];

  pages.forEach((pageText, pageIndex) => {
    splitPage(pageText, config.chunkSize, Math.min(config.chunkOverlap, config.chunkSize - 1)).forEach((text) => {
      chunks.push({
        id: `${documentId}-${chunks.length}`,
        documentId,
        documentName,
        page: pageIndex + 1,
        index: chunks.length,
        text,
      });
    });
  });

  return chunks;
}

export function normalizeExtractedPages(pages: string[], maxCharacters: number): string[] {
  let remaining = maxCharacters;
  const normalized: string[] = [];

  for (const page of pages) {
    if (remaining <= 0) break;
    const text = normalizeText(page).slice(0, remaining);
    if (text) {
      normalized.push(text);
      remaining -= text.length;
    } else {
      normalized.push("");
    }
  }

  return normalized;
}
