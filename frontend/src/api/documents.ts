import { api, apiFetch, queryString } from "./client";
import type {
  Chunk,
  DocumentItem,
  DocumentStatus,
  DocumentStatusInfo,
  Insights,
  Page,
} from "./types";

export interface ListDocumentsParams {
  limit?: number;
  offset?: number;
  status?: DocumentStatus;
}

export function listDocuments(
  params: ListDocumentsParams = {},
  signal?: AbortSignal,
): Promise<Page<DocumentItem>> {
  return api<Page<DocumentItem>>(`/documents${queryString({ ...params })}`, { signal });
}

export function getDocument(id: string, signal?: AbortSignal): Promise<DocumentItem> {
  return api<DocumentItem>(`/documents/${id}`, { signal });
}

export function getDocumentStatus(id: string, signal?: AbortSignal): Promise<DocumentStatusInfo> {
  return api<DocumentStatusInfo>(`/documents/${id}/status`, { signal });
}

export function uploadDocument(file: File): Promise<DocumentItem> {
  const form = new FormData();
  form.append("file", file);
  return api<DocumentItem>("/documents", { method: "POST", form });
}

export function deleteDocument(id: string): Promise<void> {
  return api<void>(`/documents/${id}`, { method: "DELETE" });
}

export function reprocessDocument(id: string): Promise<DocumentItem> {
  return api<DocumentItem>(`/documents/${id}/reprocess`, { method: "POST" });
}

export function getChunk(documentId: string, chunkId: string, signal?: AbortSignal): Promise<Chunk> {
  return api<Chunk>(`/documents/${documentId}/chunks/${chunkId}`, { signal });
}

/** Download the original file (the request needs the Authorization header). */
export async function fetchDocumentFile(id: string): Promise<Blob> {
  const response = await apiFetch(`/documents/${id}/file`);
  return response.blob();
}

export function getInsights(documentId: string, signal?: AbortSignal): Promise<Insights> {
  return api<Insights>(`/documents/${documentId}/insights`, { signal });
}

export function regenerateInsights(documentId: string): Promise<Insights> {
  return api<Insights>(`/documents/${documentId}/insights/regenerate`, { method: "POST" });
}
