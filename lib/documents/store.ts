import { getDocumentConfig } from "./config";
import type { DocumentSummary, IndexedDocument, KnowledgeSource } from "./types";

type Workspace = Map<string, IndexedDocument>;
type CagCacheEntry = {
  text: string;
  sources: KnowledgeSource[];
  truncated: boolean;
};

const workspaces = new Map<string, Workspace>();
const cagContextCache = new Map<string, CagCacheEntry>();
const workspaceActivity = new Map<string, number>();

function existingWorkspace(id: string): Workspace | undefined {
  const workspace = workspaces.get(id);
  if (workspace) workspaceActivity.set(id, Date.now());
  return workspace;
}

function workspaceForWrite(id: string): Workspace {
  const existing = existingWorkspace(id);
  if (existing) return existing;

  const config = getDocumentConfig();
  if (workspaces.size >= config.maxWorkspaces) {
    const oldest = [...workspaceActivity.entries()].sort((left, right) => left[1] - right[1])[0]?.[0];
    if (oldest) {
      workspaces.delete(oldest);
      workspaceActivity.delete(oldest);
      clearCagCache(oldest);
    }
  }

  const workspace = new Map<string, IndexedDocument>();
  workspaces.set(id, workspace);
  workspaceActivity.set(id, Date.now());
  return workspace;
}

function cacheKey(workspaceId: string, documents: IndexedDocument[]): string {
  const documentSignature = documents.map((document) => `${document.id}:${document.createdAt}`).join(",");
  return `${workspaceId}:${documentSignature}:complete`;
}

function summaryFor(document: IndexedDocument): DocumentSummary {
  return {
    id: document.id,
    name: document.name,
    sizeBytes: document.sizeBytes,
    pages: document.pages,
    characters: document.characters,
    chunks: document.chunks.length,
    createdAt: document.createdAt,
    ...(document.embeddingProvider ? { embeddingProvider: document.embeddingProvider } : {}),
    ...(document.embeddingModel ? { embeddingModel: document.embeddingModel } : {}),
    ...(document.embeddingDimension ? { embeddingDimension: document.embeddingDimension } : {}),
  };
}

export function listDocuments(workspaceId: string): DocumentSummary[] {
  const workspace = existingWorkspace(workspaceId);
  if (!workspace) return [];

  return [...workspace.values()]
    .sort((left, right) => right.createdAt - left.createdAt)
    .map(summaryFor);
}

export function addDocument(workspaceId: string, document: IndexedDocument): DocumentSummary {
  const config = getDocumentConfig();
  const workspace = workspaceForWrite(workspaceId);

  if (workspace.size >= config.maxDocuments) {
    throw new Error(`Este espacio ya contiene el máximo de ${config.maxDocuments} documentos.`);
  }

  workspace.set(document.id, document);
  workspaceActivity.set(workspaceId, Date.now());
  clearCagCache(workspaceId);

  return summaryFor(document);
}

export function removeDocument(workspaceId: string, documentId: string): boolean {
  const workspace = existingWorkspace(workspaceId);
  if (!workspace) return false;

  const removed = workspace.delete(documentId);
  if (removed) clearCagCache(workspaceId);
  return removed;
}

export function getDocuments(workspaceId: string, ids?: string[]): IndexedDocument[] {
  const workspace = existingWorkspace(workspaceId);
  if (!workspace) return [];
  if (ids === undefined) return [...workspace.values()];
  if (ids.length === 0) return [];

  const documents: IndexedDocument[] = [];
  for (const id of ids) {
    const document = workspace.get(id);
    if (document) documents.push(document);
  }
  return documents;
}

export function getCachedCagContext(
  workspaceId: string,
  documents: IndexedDocument[],
): CagCacheEntry | undefined {
  return cagContextCache.get(cacheKey(workspaceId, documents));
}

export function setCachedCagContext(
  workspaceId: string,
  documents: IndexedDocument[],
  value: CagCacheEntry,
): void {
  cagContextCache.set(cacheKey(workspaceId, documents), value);
}

export function clearCagCache(workspaceId: string): void {
  for (const key of cagContextCache.keys()) {
    if (key.startsWith(`${workspaceId}:`)) cagContextCache.delete(key);
  }
}
