import { embedText } from "../embeddings";

import { getDocumentConfig } from "./config";
import { getCachedCagContext, getDocuments, setCachedCagContext } from "./store";
import type { DocumentChunk, IndexedDocument, KnowledgeContext, KnowledgeSource } from "./types";

// `terms()` elimina diacríticos antes de consultar este conjunto, por lo que
// las stop words se mantienen en la misma representación normalizada.
const STOP_WORDS = new Set([
  "a", "al", "algo", "con", "como", "cual", "cuando", "de", "del", "donde", "el", "ella", "ellas", "ellos",
  "en", "es", "esta", "este", "estos", "ha", "hay", "la", "las", "lo", "los", "mas", "me",
  "mi", "mis", "o", "para", "por", "que", "quien", "se", "su", "sus", "un", "una", "unas", "uno",
  "unos", "y", "ya", "the", "of", "to", "in", "is", "are", "and", "or", "for", "with",
]);

type RankedChunk = {
  chunk: DocumentChunk;
  score: number;
  lexicalScore: number;
  semanticScore?: number;
};

type RankedChunks = {
  ranked: RankedChunk[];
  embeddingUsed: boolean;
};

function terms(value: string): string[] {
  return value
    .toLocaleLowerCase("es-CL")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .split(/[^\p{L}\p{N}]+/gu)
    .filter((term) => term.length >= 2 && !STOP_WORDS.has(term));
}

function normalizeForSearch(value: string): string {
  return value
    .toLocaleLowerCase("es-CL")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\s+/gu, " ")
    .trim();
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

function scoreLexically(query: string, chunks: DocumentChunk[]): number[] {
  const queryTerms = unique(terms(query));
  const chunkTerms = chunks.map((chunk) => terms(chunk.text));
  const documentFrequency = new Map<string, number>();

  chunkTerms.forEach((chunkTokenList) => {
    unique(chunkTokenList).forEach((term) => {
      documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1);
    });
  });

  const normalizedQuery = normalizeForSearch(query);
  return chunks.map((chunk, index) => {
    const counts = new Map<string, number>();
    chunkTerms[index].forEach((term) => counts.set(term, (counts.get(term) ?? 0) + 1));
    const lexical = queryTerms.reduce((total, term) => {
      const frequency = counts.get(term) ?? 0;
      if (!frequency) return total;
      const idf = Math.log((chunks.length + 1) / ((documentFrequency.get(term) ?? 0) + 0.5) + 1);
      return total + (frequency / (frequency + 1.5)) * idf;
    }, 0);

    const normalizedChunk = normalizeForSearch(chunk.text);
    const phraseBonus = normalizedQuery.length > 5 && normalizedChunk.includes(normalizedQuery) ? 1 : 0;
    return lexical + phraseBonus;
  });
}

function cosineSimilarity(left: number[], right: number[]): number | null {
  if (left.length === 0 || left.length !== right.length) return null;

  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] * left[index];
    rightNorm += right[index] * right[index];
  }

  if (leftNorm === 0 || rightNorm === 0) return null;
  return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
}

function rankPositions(
  scores: Array<number | undefined>,
  include: (score: number) => boolean,
  limit: number,
): Array<number | undefined> {
  const rankedIndexes = scores
    .map((score, index) => ({ score, index }))
    .filter((item): item is { score: number; index: number } => (
      item.score !== undefined && include(item.score)
    ))
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .slice(0, limit);

  const ranks = scores.map(() => undefined as number | undefined);
  rankedIndexes.forEach((item, index) => {
    ranks[item.index] = index + 1;
  });
  return ranks;
}

const RRF_K = 60;

async function rankChunks(query: string, chunks: DocumentChunk[]): Promise<RankedChunks> {
  if (chunks.length === 0) return { ranked: [], embeddingUsed: false };

  const config = getDocumentConfig();
  const lexicalScores = scoreLexically(query, chunks);
  const semanticScores: Array<number | undefined> = chunks.map(() => undefined);

  const hasStoredEmbeddings = chunks.some((chunk) => Array.isArray(chunk.embedding) && chunk.embedding.length > 0);
  if (hasStoredEmbeddings && config.semanticWeight > 0) {
    try {
      const queryEmbedding = await embedText(query, "query");
      if (queryEmbedding) {
        chunks.forEach((chunk, index) => {
          if (!chunk.embedding) return;
          const score = cosineSimilarity(queryEmbedding, chunk.embedding);
          if (score !== null) semanticScores[index] = Math.max(-1, Math.min(1, score));
        });
      }
    } catch (error) {
      console.error("[llm-bridge] Semantic retrieval failed; using lexical retrieval", error);
    }
  }

  const weightTotal = config.semanticWeight + config.lexicalWeight;
  const semanticWeight = weightTotal > 0 ? config.semanticWeight / weightTotal : 0.7;
  const lexicalWeight = weightTotal > 0 ? config.lexicalWeight / weightTotal : 0.3;
  const semanticEnabled = semanticWeight > 0;
  const lexicalEnabled = lexicalWeight > 0;
  const candidateLimit = Math.min(chunks.length, Math.max(config.topK * 4, config.topK));
  const semanticRanks = semanticEnabled
    ? rankPositions(semanticScores, (score) => score >= config.minSemanticScore, candidateLimit)
    : semanticScores.map(() => undefined);
  const lexicalRanks = lexicalEnabled
    ? rankPositions(lexicalScores, (score) => score > 0, candidateLimit)
    : lexicalScores.map(() => undefined);
  const embeddingUsed = semanticEnabled && semanticRanks.some((rank) => rank !== undefined);

  const ranked = chunks
    .map((chunk, index) => {
      const fusedScore = (semanticRanks[index] !== undefined
        ? semanticWeight / (RRF_K + (semanticRanks[index] ?? RRF_K))
        : 0)
        + (lexicalRanks[index] !== undefined
          ? lexicalWeight / (RRF_K + (lexicalRanks[index] ?? RRF_K))
          : 0);

      return {
        chunk,
        // RRF fusiona rankings heterogéneos sin asumir que cosine y TF-IDF
        // comparten escala o significado. Si ningún embedding supera el umbral,
        // el ranking vuelve a ser exclusivamente léxico cuando esa señal está habilitada.
        score: embeddingUsed ? fusedScore : (lexicalEnabled ? lexicalScores[index] : 0),
        lexicalScore: lexicalScores[index],
        ...(semanticScores[index] === undefined ? {} : { semanticScore: semanticScores[index] }),
      };
    })
    .filter((result) => result.score > 0)
    .sort((left, right) => (
      right.score - left.score ||
      (embeddingUsed
        ? (right.semanticScore ?? -Infinity) - (left.semanticScore ?? -Infinity)
        : 0) ||
      (lexicalEnabled ? right.lexicalScore - left.lexicalScore : 0) ||
      left.chunk.index - right.chunk.index
    ));

  return { ranked, embeddingUsed };
}

function sourceFor(chunk: DocumentChunk, result?: RankedChunk): KnowledgeSource {
  return {
    documentId: chunk.documentId,
    documentName: chunk.documentName,
    page: chunk.page,
    ...(chunk.pageEnd && chunk.pageEnd > chunk.page ? { pageEnd: chunk.pageEnd } : {}),
    chunkId: chunk.id,
    snippet: chunk.text.slice(0, 280),
    ...(result ? {
      score: Number(result.score.toFixed(4)),
      lexicalScore: Number(result.lexicalScore.toFixed(4)),
      ...(result.semanticScore === undefined ? {} : { semanticScore: Number(result.semanticScore.toFixed(4)) }),
    } : {}),
  };
}

function pageLabel(chunk: DocumentChunk): string {
  return chunk.pageEnd && chunk.pageEnd > chunk.page
    ? `páginas ${chunk.page}-${chunk.pageEnd}`
    : `página ${chunk.page}`;
}

function neutralizeDocumentDelimiters(value: string): string {
  return value.replace(/<\s*(\/?)\s*documentos\s*>/giu, (_match, closing: string) => (
    closing ? "[/documentos]" : "[documentos]"
  ));
}

function contextBlock(chunk: DocumentChunk): string {
  const documentName = JSON.stringify(chunk.documentName);
  return `[Documento: ${documentName} | ${pageLabel(chunk)}]\n${neutralizeDocumentDelimiters(chunk.text)}`;
}

function selectedDocuments(workspaceId: string, ids?: string[]): IndexedDocument[] {
  return getDocuments(workspaceId, ids).sort((left, right) => left.createdAt - right.createdAt);
}

function estimatedContextCharacters(chunks: DocumentChunk[]): number {
  return chunks.reduce((total, chunk) => total + contextBlock(chunk).length + 2, 0);
}

async function buildRag(documents: IndexedDocument[], query: string): Promise<KnowledgeContext> {
  const config = getDocumentConfig();
  const { ranked: allRanked, embeddingUsed } = await rankChunks(query, documents.flatMap((document) => document.chunks));
  const ranked = allRanked.slice(0, config.topK);
  const sources: KnowledgeSource[] = [];
  let remaining = config.maxRagContextCharacters;
  const blocks: string[] = [];
  let contextWasCut = false;

  for (const result of ranked) {
    if (remaining <= 0) {
      contextWasCut = true;
      break;
    }

    const fullBlock = contextBlock(result.chunk);
    if (fullBlock.length > remaining && blocks.length > 0) {
      contextWasCut = true;
      break;
    }

    const block = fullBlock.slice(0, remaining);
    if (!block) {
      contextWasCut = true;
      break;
    }

    blocks.push(block);
    sources.push(sourceFor(result.chunk, result));
    if (block.length < fullBlock.length) contextWasCut = true;
    remaining -= block.length + 2;
  }

  if (sources.length < ranked.length) contextWasCut = true;

  return {
    mode: "rag",
    text: blocks.join("\n\n"),
    sources,
    cacheHit: false,
    embeddingUsed,
    truncated: contextWasCut,
  };
}

function buildCag(workspaceId: string, documents: IndexedDocument[]): KnowledgeContext {
  const config = getDocumentConfig();
  const allChunks = documents.flatMap((document) => document.chunks);
  const completeContextFits = estimatedContextCharacters(allChunks) <= config.maxCagContextCharacters;
  const cached = getCachedCagContext(workspaceId, documents);

  if (cached) {
    return {
      mode: "cag",
      text: cached.text,
      sources: cached.sources,
      cacheHit: true,
      embeddingUsed: false,
      truncated: cached.truncated,
    };
  }

  // CAG no usa la pregunta para seleccionar fragmentos: conserva un contexto
  // documental estable para que vLLM pueda reutilizar su prefijo.
  let remaining = config.maxCagContextCharacters;
  const blocks: string[] = [];
  const sources: KnowledgeSource[] = [];
  let truncated = !completeContextFits;

  for (const chunk of allChunks) {
    if (remaining <= 0) break;

    const fullBlock = contextBlock(chunk);
    if (fullBlock.length > remaining && blocks.length > 0) {
      truncated = true;
      break;
    }

    const block = fullBlock.slice(0, remaining);
    if (!block) {
      truncated = true;
      break;
    }

    blocks.push(block);
    sources.push(sourceFor(chunk));
    if (block.length < fullBlock.length) truncated = true;
    remaining -= block.length + 2;
  }

  const text = blocks.join("\n\n");
  setCachedCagContext(workspaceId, documents, {
    text,
    sources,
    truncated,
  });

  return { mode: "cag", text, sources, cacheHit: false, embeddingUsed: false, truncated };
}

export async function buildKnowledgeContext(
  workspaceId: string,
  mode: "rag" | "cag",
  query: string,
  ids?: string[],
): Promise<KnowledgeContext | null> {
  const documents = selectedDocuments(workspaceId, ids);
  if (documents.length === 0) return null;
  return mode === "rag" ? buildRag(documents, query) : buildCag(workspaceId, documents);
}
