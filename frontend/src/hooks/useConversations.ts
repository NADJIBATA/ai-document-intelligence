import { listConversations } from "../api/chat";
import type { ConversationSummary, Page } from "../api/types";
import { useApi, type ApiState } from "./useApi";

export function useConversations(limit = 50): ApiState<Page<ConversationSummary>> {
  return useApi((signal) => listConversations({ limit }, signal), [limit]);
}
