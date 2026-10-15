import { ApiError } from "../api/client";

const FRIENDLY_MESSAGES: Record<string, string> = {
  network_error: "The server is unreachable. Check your connection and try again.",
  not_authenticated: "Your session has expired. Please sign in again.",
  duplicate_document: "This file has already been uploaded.",
  llm_unavailable: "The language model is temporarily unavailable. Please try again later.",
  insights_in_progress: "The analysis of this document is already in progress.",
  document_not_indexed: "This document is not indexed yet.",
};

/** Message that can be shown to the user for any thrown value. */
export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 429) {
      const wait = error.retryAfter ? ` Try again in ${error.retryAfter} s.` : "";
      return `Too many requests.${wait}`;
    }
    if (error.code === "validation_error" && error.details.length > 0) {
      return error.details.map(describeDetail).join(" ");
    }
    return FRIENDLY_MESSAGES[error.code] ?? error.message;
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "An unexpected error occurred.";
}

function describeDetail(detail: { loc: string[]; msg: string }): string {
  // "body.password" -> "password"
  const field = detail.loc.filter((part) => part !== "body" && part !== "query").join(".");
  const message = detail.msg.replace(/^Value error, /, "");
  return field ? `${field}: ${message}` : message;
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

export function isNotFound(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404;
}
