import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  api,
  ApiError,
  hasAccessToken,
  onSessionExpired,
  queryString,
  refreshSession,
  setAccessToken,
} from "../api/client";

interface Call {
  url: string;
  method: string;
  authorization: string | null;
  contentType: string | null;
  body: unknown;
}

type Handler = (call: Call) => Response;

function json(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

function tokens(accessToken: string): Response {
  return json(200, {
    access_token: accessToken,
    token_type: "bearer",
    expires_in: 900,
    refresh_token: "ignored-by-browsers",
    refresh_expires_in: 604800,
  });
}

/** Replace fetch with `handler` and record every call. */
function mockFetch(handler: Handler): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal("fetch", (input: RequestInfo | URL, init?: RequestInit) => {
    const headers = new Headers(init?.headers);
    const call: Call = {
      url: String(input),
      method: init?.method ?? "GET",
      authorization: headers.get("Authorization"),
      contentType: headers.get("Content-Type"),
      body: init?.body,
    };
    calls.push(call);
    return Promise.resolve(handler(call));
  });
  return calls;
}

beforeEach(() => {
  setAccessToken(null);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api", () => {
  it("sends JSON with the bearer token and returns the decoded body", async () => {
    setAccessToken("token-1");
    const calls = mockFetch(() => json(200, { answer: "ok" }));

    const result = await api<{ answer: string }>("/chat", {
      method: "POST",
      json: { question: "Hello?" },
    });

    expect(result).toEqual({ answer: "ok" });
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe("/api/chat");
    expect(calls[0]?.method).toBe("POST");
    expect(calls[0]?.authorization).toBe("Bearer token-1");
    expect(calls[0]?.contentType).toBe("application/json");
    expect(calls[0]?.body).toBe('{"question":"Hello?"}');
  });

  it("does not send the token to public endpoints", async () => {
    setAccessToken("token-1");
    const calls = mockFetch(() => json(200, {}));
    await api("/auth/login", { method: "POST", json: {}, auth: false });
    expect(calls[0]?.authorization).toBeNull();
  });

  it("lets the browser set the multipart content type for uploads", async () => {
    const calls = mockFetch(() => json(202, { id: "doc" }));
    const form = new FormData();
    form.append("file", new Blob(["hello"]), "a.txt");
    await api("/documents", { method: "POST", form });
    expect(calls[0]?.contentType).toBeNull();
    expect(calls[0]?.body).toBe(form);
  });

  it("returns undefined for 204 responses", async () => {
    mockFetch(() => new Response(null, { status: 204 }));
    expect(await api<void>("/documents/1", { method: "DELETE" })).toBeUndefined();
  });

  it("turns API errors into ApiError", async () => {
    mockFetch(() =>
      json(
        409,
        { error: { code: "duplicate_document", message: "Already uploaded.", request_id: "req-1" } },
        { "Retry-After": "30" },
      ),
    );
    const error = await api("/documents").catch((reason: unknown) => reason);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      status: 409,
      code: "duplicate_document",
      message: "Already uploaded.",
      requestId: "req-1",
      retryAfter: 30,
    });
  });

  it("copes with error responses that are not JSON", async () => {
    mockFetch(() => new Response("<html>Bad gateway</html>", { status: 502 }));
    const error = await api("/documents").catch((reason: unknown) => reason);
    expect(error).toMatchObject({ status: 502, code: "http_502" });
  });

  it("reports network failures", async () => {
    vi.stubGlobal("fetch", () => Promise.reject(new TypeError("Failed to fetch")));
    const error = await api("/documents").catch((reason: unknown) => reason);
    expect(error).toMatchObject({ status: 0, code: "network_error" });
  });
});

describe("session refresh", () => {
  it("refreshes the session once and replays the request after a 401", async () => {
    setAccessToken("expired");
    const calls = mockFetch((call) => {
      if (call.url === "/api/auth/refresh") {
        return tokens("fresh");
      }
      return call.authorization === "Bearer fresh"
        ? json(200, { items: [] })
        : json(401, { error: { code: "not_authenticated", message: "Expired." } });
    });

    expect(await api("/documents")).toEqual({ items: [] });
    expect(calls.map((call) => call.url)).toEqual([
      "/api/documents",
      "/api/auth/refresh",
      "/api/documents",
    ]);
    expect(calls[1]?.authorization).toBeNull(); // the refresh relies on the cookie only
    expect(calls[1]?.method).toBe("POST");
  });

  it("uses a single refresh for concurrent requests", async () => {
    setAccessToken("expired");
    const calls = mockFetch((call) => {
      if (call.url === "/api/auth/refresh") {
        return tokens("fresh");
      }
      return call.authorization === "Bearer fresh" ? json(200, {}) : json(401, {});
    });

    await Promise.all([api("/documents"), api("/conversations"), api("/dashboard/stats")]);
    expect(calls.filter((call) => call.url === "/api/auth/refresh")).toHaveLength(1);
  });

  it("signals an expired session when the refresh is refused", async () => {
    setAccessToken("expired");
    mockFetch(() => json(401, { error: { code: "not_authenticated", message: "Expired." } }));
    const listener = vi.fn();
    const unsubscribe = onSessionExpired(listener);

    const error = await api("/documents").catch((reason: unknown) => reason);
    unsubscribe();

    expect(error).toMatchObject({ status: 401 });
    expect(listener).toHaveBeenCalledTimes(1);
    expect(hasAccessToken()).toBe(false);
  });

  it("does not try to refresh after a failed login", async () => {
    const calls = mockFetch(() =>
      json(401, { error: { code: "invalid_credentials", message: "Invalid email or password." } }),
    );
    const error = await api("/auth/login", { method: "POST", json: {}, auth: false }).catch(
      (reason: unknown) => reason,
    );
    expect(error).toMatchObject({ message: "Invalid email or password." });
    expect(calls).toHaveLength(1);
  });

  it("refreshSession stores the new access token", async () => {
    mockFetch(() => tokens("fresh"));
    expect(await refreshSession()).toBe(true);
    expect(hasAccessToken()).toBe(true);
  });

  it("refreshSession resolves to false when the server is unreachable", async () => {
    vi.stubGlobal("fetch", () => Promise.reject(new TypeError("Failed to fetch")));
    expect(await refreshSession()).toBe(false);
  });
});

describe("queryString", () => {
  it("skips empty values", () => {
    expect(queryString({ limit: 20, offset: 0, status: undefined, q: "" })).toBe(
      "?limit=20&offset=0",
    );
    expect(queryString({})).toBe("");
  });
});
