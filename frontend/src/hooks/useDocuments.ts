import { listDocuments, type ListDocumentsParams } from "../api/documents";
import type { DocumentItem, Page } from "../api/types";
import { useApi, type ApiState } from "./useApi";
import { usePolling } from "./usePolling";

const POLL_INTERVAL_MS = 2000;

export function isInProgress(document: DocumentItem): boolean {
  return document.status === "uploaded" || document.status === "processing";
}

/**
 * My documents. While at least one of them is being processed, the list is refreshed
 * every two seconds so that statuses update without reloading the page.
 */
export function useDocuments(params: ListDocumentsParams = {}): ApiState<Page<DocumentItem>> {
  const state = useApi(
    (signal) => listDocuments(params, signal),
    [params.limit, params.offset, params.status],
  );
  const inProgress = state.data?.items.some(isInProgress) ?? false;
  usePolling(state.reload, POLL_INTERVAL_MS, inProgress);
  return state;
}
