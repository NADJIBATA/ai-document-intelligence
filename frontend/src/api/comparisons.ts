import { api, queryString } from "./client";
import type { Aspect, Comparison, ComparisonSummary, Page } from "./types";

export function compareDocuments(
  documentAId: string,
  documentBId: string,
  aspects: Aspect[],
): Promise<Comparison> {
  return api<Comparison>("/documents/compare", {
    method: "POST",
    json: {
      document_a_id: documentAId,
      document_b_id: documentBId,
      aspects: aspects.length > 0 ? aspects : null,
    },
  });
}

export function listComparisons(
  params: { limit?: number; offset?: number } = {},
  signal?: AbortSignal,
): Promise<Page<ComparisonSummary>> {
  return api<Page<ComparisonSummary>>(`/comparisons${queryString(params)}`, { signal });
}

export function getComparison(id: string, signal?: AbortSignal): Promise<Comparison> {
  return api<Comparison>(`/comparisons/${id}`, { signal });
}

export function deleteComparison(id: string): Promise<void> {
  return api<void>(`/comparisons/${id}`, { method: "DELETE" });
}
