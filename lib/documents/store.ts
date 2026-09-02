import { getDocumentConfig } from "./config";
import type { DocumentSummary, IndexedDocument } from "./types";

type Workspace = Map<string, IndexedDocument>;

const workspaces = new Map<string, Workspace>();
const cagContextCache = new Map<string, { signature: string; text: string }>();
const workspaceActivity = new Map<string, number>();

function workspaceFor(id: string): Workspace {
  const existing = workspaces.get(id);
  if (existing) {
    workspaceActivity.set(id, Date.now());
    return existing;
  }

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
  return `${workspaceId}:${documents.map((document) => `${document.id}:${document.createdAt}`).join(",")}`;
}

export function listDocuments(workspaceId: string): DocumentSummary[] {
  return [...workspaceFor(workspaceId).values()]
    .sort((left, right) => right.createdAt - left.createdAt)
    .map((document) => ({
      id: document.id,
      name: document.name,
      sizeBytes: document.sizeBytes,
      pages: document.pages,
      characters: document.characters,
      chunks: document.chunks.length,
      createdAt: document.createdAt,
    }));
}

export function addDocument(workspaceId: string, document: IndexedDocument): DocumentSummary {
  const config = getDocumentConfig();
  const workspace = workspaceFor(workspaceId);

  if (workspace.size >= config.maxDocuments) {
    throw new Error(`Este espacio ya contiene el máximo de ${config.maxDocuments} documentos.`);
  }

  workspace.set(document.id, document);
  workspaceActivity.set(workspaceId, Date.now());
  clearCagCache(workspaceId);

  return {
    id: document.id,
    name: document.name,
    sizeBytes: document.sizeBytes,
    pages: document.pages,
    characters: document.characters,
    chunks: document.chunks.length,
    createdAt: document.createdAt,
  };
}

export function removeDocument(workspaceId: string, documentId: string): boolean {
  const workspace = workspaceFor(workspaceId);
  const removed = workspace.delete(documentId);
  if (removed) clearCagCache(workspaceId);
  return removed;
}

export function getDocuments(workspaceId: string, ids?: string[]): IndexedDocument[] {
  const workspace = workspaceFor(workspaceId);
  if (!ids || ids.length === 0) return [...workspace.values()];

  const documents: IndexedDocument[] = [];
  for (const id of ids) {
    const document = workspace.get(id);
    if (document) documents.push(document);
  }
  return documents;
}

export function getCachedCagContext(workspaceId: string, documents: IndexedDocument[]): string | undefined {
  return cagContextCache.get(cacheKey(workspaceId, documents))?.text;
}

export function setCachedCagContext(workspaceId: string, documents: IndexedDocument[], text: string): void {
  const key = cacheKey(workspaceId, documents);
  cagContextCache.set(key, { signature: key, text });
}

export function clearCagCache(workspaceId: string): void {
  for (const key of cagContextCache.keys()) {
    if (key.startsWith(`${workspaceId}:`)) cagContextCache.delete(key);
  }
}
