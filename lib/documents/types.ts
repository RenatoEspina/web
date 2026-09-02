export type KnowledgeMode = "none" | "rag" | "cag";

export interface DocumentChunk {
  id: string;
  documentId: string;
  documentName: string;
  page: number;
  pageEnd?: number;
  index: number;
  text: string;
  embedding?: number[];
}

export interface IndexedDocument {
  id: string;
  name: string;
  sizeBytes: number;
  pages: number;
  characters: number;
  chunks: DocumentChunk[];
  createdAt: number;
  embeddingProvider?: string;
  embeddingModel?: string;
  embeddingDimension?: number;
}

export interface DocumentSummary {
  id: string;
  name: string;
  sizeBytes: number;
  pages: number;
  characters: number;
  chunks: number;
  createdAt: number;
  embeddingProvider?: string;
  embeddingModel?: string;
  embeddingDimension?: number;
}

export interface KnowledgeSource {
  documentId: string;
  documentName: string;
  page: number;
  pageEnd?: number;
  chunkId: string;
  snippet: string;
  score?: number;
  lexicalScore?: number;
  semanticScore?: number;
}

export interface KnowledgeContext {
  mode: Exclude<KnowledgeMode, "none">;
  text: string;
  sources: KnowledgeSource[];
  cacheHit: boolean;
  embeddingUsed: boolean;
  truncated: boolean;
}
