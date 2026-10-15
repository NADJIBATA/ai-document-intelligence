import type { ChatMetadata, Source } from "../../api/types";

/** A message as displayed in the chat (server messages + local pending/error states). */
export interface UiMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  answered: boolean | null;
  /** The answer is being generated. */
  pending?: boolean;
  /** The request failed: `content` holds the error message. */
  failed?: boolean;
  metadata?: ChatMetadata;
}
