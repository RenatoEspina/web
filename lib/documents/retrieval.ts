import { getDocumentConfig } from "./config";
import { getCachedCagContext, getDocuments, setCachedCagContext } from "./store";
import type { DocumentChunk, IndexedDocument, KnowledgeContext, KnowledgeSource } from "./types";

const STOP_WORDS = new Set([
  "a", "al", "algo", "con", "como", "cual", "de", "del", "el", "ella", "ellas", "ellos",
  "en", "es", "esta", "este", "estos", "ha", "hay", "la", "las", "lo", "los", "más", "me",
  "mi", "mis", "o", "para", "por", "que", "qué", "se", "su", "sus", "un", "una", "unas", "uno",
  "unos", "y", "ya", "the", "of", "to", "in", "is", "are", "and", "or", "for", "with",
]);

function terms(value: string): string[] {
  return value
    .toLocaleLowerCase("es-CL")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .split(/[^\p{L}\p{N}]+/gu)
    .filter((term) => term.length >= 2 && !STOP_WORDS.has(term));
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

function scoreChunks(query: string, chunks: DocumentChunk[]): Array<{ chunk: DocumentChunk; score: number }> {
  const queryTerms = unique(terms(query));
  if (queryTerms.length === 0) return [];

  const chunkTerms = chunks.map((chunk) => terms(chunk.text));
  const documentFrequency = new Map<string, number>();
  chunkTerms.forEach((chunkTokenList) => {
    unique(chunkTokenList).forEach((term) => documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1));
  });

  return chunks
    .map((chunk, index) => {
      const counts = new Map<string, number>();
      chunkTerms[index].forEach((term) => counts.set(term, (counts.get(term) ?? 0) + 1));
      const score = queryTerms.reduce((total, term) => {
        const frequency = counts.get(term) ?? 0;
        if (!frequency) return total;
        const idf = Math.log((chunks.length + 1) / ((documentFrequency.get(term) ?? 0) + 0.5) + 1);
        return total + (frequency / (frequency + 1.5)) * idf;
      }, 0);

      const normalizedQuery = query.trim().toLocaleLowerCase("es-CL");
      const normalizedChunk = chunk.text.toLocaleLowerCase("es-CL");
      const phraseBonus = normalizedQuery.length > 5 && normalizedChunk.includes(normalizedQuery) ? 1 : 0;

      return { chunk, score: score + phraseBonus };
    })
    .filter((result) => result.score > 0)
    .sort((left, right) => right.score - left.score);
}

function sourceFor(chunk: DocumentChunk, score?: number): KnowledgeSource {
  return {
    documentId: chunk.documentId,
    documentName: chunk.documentName,
    page: chunk.page,
    chunkId: chunk.id,
    snippet: chunk.text.slice(0, 280),
    ...(score === undefined ? {} : { score: Number(score.toFixed(4)) }),
  };
}

function selectedDocuments(workspaceId: string, ids?: string[]): IndexedDocument[] {
  return getDocuments(workspaceId, ids).sort((left, right) => left.createdAt - right.createdAt);
}

function buildRag(workspaceId: string, query: string, ids?: string[]): KnowledgeContext {
  const config = getDocumentConfig();
  const documents = selectedDocuments(workspaceId, ids);
  const ranked = scoreChunks(query, documents.flatMap((document) => document.chunks)).slice(0, config.topK);
  const sources = ranked.map(({ chunk, score }) => sourceFor(chunk, score));
  let remaining = config.maxRagContextCharacters;
  const blocks: string[] = [];

  for (const { chunk } of ranked) {
    if (remaining <= 0) break;
    const text = chunk.text.slice(0, remaining);
    blocks.push(`[Documento: ${chunk.documentName} | página ${chunk.page}]\n${text}`);
    remaining -= text.length;
  }

  return {
    mode: "rag",
    text: blocks.join("\n\n"),
    sources,
    cacheHit: false,
  };
}

function buildCag(workspaceId: string, ids?: string[]): KnowledgeContext {
  const config = getDocumentConfig();
  const documents = selectedDocuments(workspaceId, ids);
  const cached = getCachedCagContext(workspaceId, documents);
  if (cached !== undefined) {
    return {
      mode: "cag",
      text: cached,
      sources: documents.flatMap((document) => document.chunks.slice(0, 8).map((chunk) => sourceFor(chunk))),
      cacheHit: true,
    };
  }

  let remaining = config.maxCagContextCharacters;
  const blocks: string[] = [];
  const sources: KnowledgeSource[] = [];

  for (const document of documents) {
    for (const chunk of document.chunks) {
      if (remaining <= 0) break;
      const text = chunk.text.slice(0, remaining);
      blocks.push(`[Documento: ${chunk.documentName} | página ${chunk.page}]\n${text}`);
      sources.push(sourceFor(chunk));
      remaining -= text.length;
    }
    if (remaining <= 0) break;
  }

  const text = blocks.join("\n\n");
  setCachedCagContext(workspaceId, documents, text);

  return { mode: "cag", text, sources, cacheHit: false };
}

export function buildKnowledgeContext(
  workspaceId: string,
  mode: "rag" | "cag",
  query: string,
  ids?: string[],
): KnowledgeContext | null {
  const documents = selectedDocuments(workspaceId, ids);
  if (documents.length === 0) return null;
  return mode === "rag" ? buildRag(workspaceId, query, ids) : buildCag(workspaceId, ids);
}
