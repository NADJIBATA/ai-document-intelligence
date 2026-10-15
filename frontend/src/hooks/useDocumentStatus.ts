import { getDocumentStatus } from "../api/documents";
import type { DocumentStatusInfo } from "../api/types";
import { useApi, type ApiState } from "./useApi";
import { usePolling } from "./usePolling";

const POLL_INTERVAL_MS = 2000;

/** Processing status of one document, polled until it is indexed or failed. */
export function useDocumentStatus(documentId: string | null): ApiState<DocumentStatusInfo> {
  const state = useApi(
    documentId ? (signal) => getDocumentStatus(documentId, signal) : null,
    [documentId],
  );
  const status = state.data?.status;
  usePolling(state.reload, POLL_INTERVAL_MS, status === "uploaded" || status === "processing");
  return state;
}
