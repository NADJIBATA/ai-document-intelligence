import { api, queryString } from "./client";
import type { ChatResponse, ConversationDetail, ConversationSummary, Page } from "./types";

export interface AskParams {
  question: string;
  conversationId?: string | null;
  documentIds?: string[];
}

export function askQuestion(params: AskParams, signal?: AbortSignal): Promise<ChatResponse> {
  return api<ChatResponse>("/chat", {
    method: "POST",
    json: {
      question: params.question,
      conversation_id: params.conversationId ?? null,
      document_ids: params.documentIds && params.documentIds.length > 0 ? params.documentIds : null,
    },
    signal,
  });
}

export function listConversations(
  params: { limit?: number; offset?: number } = {},
  signal?: AbortSignal,
): Promise<Page<ConversationSummary>> {
  return api<Page<ConversationSummary>>(`/conversations${queryString(params)}`, { signal });
}

export function getConversation(id: string, signal?: AbortSignal): Promise<ConversationDetail> {
  return api<ConversationDetail>(`/conversations/${id}`, { signal });
}

export function deleteConversation(id: string): Promise<void> {
  return api<void>(`/conversations/${id}`, { method: "DELETE" });
}
