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

type PageSpan = {
  page: number;
  start: number;
  end: number;
};

type TextSegment = {
  text: string;
  start: number;
  end: number;
};

function joinPages(pages: string[]): { text: string; spans: PageSpan[] } {
  let text = "";
  const spans: PageSpan[] = [];

  pages.forEach((page, index) => {
    const normalized = normalizeText(page);
    if (!normalized) return;

    if (text) text += " ";
    const start = text.length;
    text += normalized;
    spans.push({ page: index + 1, start, end: text.length });
  });

  return { text, spans };
}

function splitText(text: string, chunkSize: number, overlap: number): TextSegment[] {
  if (!text) return [];
  if (text.length <= chunkSize) return [{ text, start: 0, end: text.length }];

  const segments: TextSegment[] = [];
  let start = 0;

  while (start < text.length) {
    const desiredEnd = Math.min(text.length, start + chunkSize);
    const end = findBreak(text, desiredEnd, start + Math.floor(chunkSize * 0.6));
    const rawChunk = text.slice(start, end);
    const chunk = rawChunk.trim();
    if (chunk) {
      const leadingWhitespace = rawChunk.search(/\S/u);
      const segmentStart = leadingWhitespace < 0 ? start : start + leadingWhitespace;
      segments.push({ text: chunk, start: segmentStart, end });
    }

    if (end >= text.length) break;
    const nextStart = Math.max(start + 1, end - overlap);
    start = nextStart;
  }

  return segments;
}

function pagesForSegment(segment: TextSegment, spans: PageSpan[]): { page: number; pageEnd?: number } {
  const covered = spans.filter((span) => span.end > segment.start && span.start < segment.end);
  const first = covered[0] ?? spans.find((span) => span.start >= segment.start);
  const last = covered.at(-1) ?? first;

  if (!first) return { page: 1 };
  return {
    page: first.page,
    ...(last && last.page > first.page ? { pageEnd: last.page } : {}),
  };
}

export function createChunks(
  documentId: string,
  documentName: string,
  pages: string[],
): DocumentChunk[] {
  const config = getDocumentConfig();
  const { text, spans } = joinPages(pages);
  const segments = splitText(
    text,
    config.chunkSize,
    Math.min(config.chunkOverlap, config.chunkSize - 1),
  );

  return segments.map((segment, index) => {
    const pageRange = pagesForSegment(segment, spans);
    return {
      id: `${documentId}-${index}`,
      documentId,
      documentName,
      ...pageRange,
      index,
      text: segment.text,
    };
  });
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
