import { describe, expect, it } from "vitest";

import { ApiError } from "../api/client";
import { errorMessage, isAbortError, isNotFound } from "../lib/errors";

describe("errorMessage", () => {
  it("uses the message sent by the API", () => {
    const error = new ApiError(409, "conflict", "Email already registered.");
    expect(errorMessage(error)).toBe("Email already registered.");
  });

  it("replaces some technical codes with friendlier messages", () => {
    const error = new ApiError(503, "llm_unavailable", "The language model is unavailable.");
    expect(errorMessage(error)).toBe(
      "The language model is temporarily unavailable. Please try again later.",
    );
  });

  it("lists the fields of a validation error", () => {
    const error = new ApiError(422, "validation_error", "The request payload is invalid.", {
      details: [
        { loc: ["body", "password"], msg: "Value error, Password is too short.", type: "value" },
        { loc: ["body", "email"], msg: "value is not a valid email address", type: "value" },
      ],
    });
    expect(errorMessage(error)).toBe(
      "password: Password is too short. email: value is not a valid email address",
    );
  });

  it("explains rate limiting", () => {
    expect(errorMessage(new ApiError(429, "rate_limit_exceeded", "Too many", { retryAfter: 42 }))).toBe(
      "Too many requests. Try again in 42 s.",
    );
    expect(errorMessage(new ApiError(429, "rate_limit_exceeded", "Too many"))).toBe(
      "Too many requests.",
    );
  });

  it("handles any thrown value", () => {
    expect(errorMessage(new Error("boom"))).toBe("boom");
    expect(errorMessage("a string")).toBe("An unexpected error occurred.");
    expect(errorMessage(undefined)).toBe("An unexpected error occurred.");
  });
});

describe("error predicates", () => {
  it("recognises 404 responses", () => {
    expect(isNotFound(new ApiError(404, "document_not_found", "Document not found."))).toBe(true);
    expect(isNotFound(new ApiError(500, "internal_error", "Oops"))).toBe(false);
    expect(isNotFound(new Error("x"))).toBe(false);
  });

  it("recognises aborted requests", () => {
    expect(isAbortError(new DOMException("Aborted", "AbortError"))).toBe(true);
    expect(isAbortError(new Error("x"))).toBe(false);
  });
});
