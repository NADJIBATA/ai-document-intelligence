// Thin wrapper around fetch for the backend API.
//
// * The access token lives in memory only (never in localStorage): an XSS cannot read
//   it from storage, and it disappears with the tab.
// * The refresh token is an HttpOnly cookie scoped to /api/auth: JavaScript never sees
//   it. When a request gets a 401, the session is refreshed once and the request is
//   replayed.
// * Refreshes are serialised (one at a time, across tabs when the browser supports the
//   Web Locks API): the backend rotates refresh tokens and treats a replayed token as
//   a theft, so two concurrent refreshes must never use the same cookie.

import type { TokenResponse } from "./types";

const API_BASE = "/api";
const SESSION_HINT_KEY = "docintel.session";
const REFRESH_LOCK = "docintel.refresh";

export interface ApiErrorDetail {
  loc: string[];
  msg: string;
  type: string;
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: ApiErrorDetail[];
  readonly requestId: string | null;
  readonly retryAfter: number | null;

  constructor(
    status: number,
    code: string,
    message: string,
    options: {
      details?: ApiErrorDetail[];
      requestId?: string | null;
      retryAfter?: number | null;
    } = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = options.details ?? [];
    this.requestId = options.requestId ?? null;
    this.retryAfter = options.retryAfter ?? null;
  }
}

export interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  /** JSON body. */
  json?: unknown;
  /** Multipart body (file upload). */
  form?: FormData;
  /** Set to false for public endpoints (login, register). */
  auth?: boolean;
  signal?: AbortSignal;
}

let accessToken: string | null = null;
let refreshInFlight: Promise<boolean> | null = null;
const sessionExpiredListeners = new Set<() => void>();

export function setAccessToken(token: string | null): void {
  accessToken = token;
}

export function hasAccessToken(): boolean {
  return accessToken !== null;
}

/** Called when the session cannot be refreshed any more (the user must log in again). */
export function onSessionExpired(listener: () => void): () => void {
  sessionExpiredListeners.add(listener);
  return () => {
    sessionExpiredListeners.delete(listener);
  };
}

// A non-sensitive flag telling whether it is worth trying to restore a session when
// the application starts (it avoids a useless 401 for visitors who never logged in).
export function rememberSession(active: boolean): void {
  try {
    if (active) {
      window.localStorage.setItem(SESSION_HINT_KEY, "1");
    } else {
      window.localStorage.removeItem(SESSION_HINT_KEY);
    }
  } catch {
    // Storage disabled (private mode...): the session simply is not restored on reload.
  }
}

export function hasSessionHint(): boolean {
  try {
    return window.localStorage.getItem(SESSION_HINT_KEY) === "1";
  } catch {
    return false;
  }
}

async function toApiError(response: Response): Promise<ApiError> {
  const retryAfterHeader = response.headers.get("Retry-After");
  const retryAfter = retryAfterHeader ? Number.parseInt(retryAfterHeader, 10) : null;
  let code = `http_${response.status}`;
  let message = response.statusText || "The request failed.";
  let details: ApiErrorDetail[] = [];
  let requestId = response.headers.get("X-Request-ID");
  try {
    const body: unknown = await response.json();
    if (typeof body === "object" && body !== null && "error" in body) {
      const error = (body as { error: Record<string, unknown> }).error;
      if (typeof error.code === "string") code = error.code;
      if (typeof error.message === "string") message = error.message;
      if (typeof error.request_id === "string") requestId = error.request_id;
      if (Array.isArray(error.details)) details = error.details as ApiErrorDetail[];
    }
  } catch {
    // Not a JSON body (proxy error page...): keep the HTTP status text.
  }
  return new ApiError(response.status, code, message, {
    details,
    requestId,
    retryAfter: Number.isNaN(retryAfter) ? null : retryAfter,
  });
}

async function send(path: string, options: RequestOptions): Promise<Response> {
  const headers = new Headers({ Accept: "application/json" });
  let body: BodyInit | undefined;
  if (options.json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.json);
  } else if (options.form) {
    body = options.form; // the browser sets the multipart boundary itself
  }
  if (options.auth !== false && accessToken) {
    headers.set("Authorization", `Bearer ${accessToken}`);
  }
  try {
    return await fetch(`${API_BASE}${path}`, {
      method: options.method ?? "GET",
      headers,
      body,
      credentials: "same-origin",
      signal: options.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new ApiError(0, "network_error", "The server is unreachable. Check your connection.");
  }
}

async function doRefresh(): Promise<boolean> {
  let response: Response;
  try {
    response = await send("/auth/refresh", { method: "POST", auth: false });
  } catch {
    return false; // network error: keep the session hint, the user can retry
  }
  if (!response.ok) {
    setAccessToken(null);
    if (response.status === 401) {
      rememberSession(false);
    }
    return false;
  }
  const tokens = (await response.json()) as TokenResponse;
  setAccessToken(tokens.access_token);
  rememberSession(true);
  return true;
}

/** Exchange the refresh cookie for a new access token. Resolves to false on failure. */
export function refreshSession(): Promise<boolean> {
  if (refreshInFlight === null) {
    const locks = typeof navigator !== "undefined" ? navigator.locks : undefined;
    const run = locks
      ? (locks.request(REFRESH_LOCK, () => doRefresh()) as Promise<boolean>)
      : doRefresh();
    refreshInFlight = run.finally(() => {
      refreshInFlight = null;
    });
  }
  return refreshInFlight;
}

/** Low-level call returning the raw Response (used for file downloads). */
export async function apiFetch(path: string, options: RequestOptions = {}): Promise<Response> {
  let response = await send(path, options);
  if (response.status === 401 && options.auth !== false) {
    const refreshed = await refreshSession();
    if (refreshed) {
      response = await send(path, options);
    }
    if (response.status === 401) {
      setAccessToken(null);
      sessionExpiredListeners.forEach((listener) => listener());
    }
  }
  if (!response.ok) {
    throw await toApiError(response);
  }
  return response;
}

/** JSON call. Resolves to undefined for 204 responses. */
export async function api<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await apiFetch(path, options);
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export function queryString(params: Record<string, string | number | undefined | null>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      search.set(key, String(value));
    }
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}
