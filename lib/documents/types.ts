export type KnowledgeMode = "none" | "rag" | "cag";

export interface DocumentChunk {
  id: string;
  documentId: string;
  documentName: string;
  page: number;
  index: number;
  text: string;
}

export interface IndexedDocument {
  id: string;
  name: string;
  sizeBytes: number;
  pages: number;
  characters: number;
  chunks: DocumentChunk[];
  createdAt: number;
}

export interface DocumentSummary {
  id: string;
  name: string;
  sizeBytes: number;
  pages: number;
  characters: number;
  chunks: number;
  createdAt: number;
}

export interface KnowledgeSource {
  documentId: string;
  documentName: string;
  page: number;
  chunkId: string;
  snippet: string;
  score?: number;
}

export interface KnowledgeContext {
  mode: Exclude<KnowledgeMode, "none">;
  text: string;
  sources: KnowledgeSource[];
  cacheHit: boolean;
}
