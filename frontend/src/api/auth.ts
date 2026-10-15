import { api, rememberSession, setAccessToken } from "./client";
import type { TokenResponse, User } from "./types";

export async function login(email: string, password: string): Promise<void> {
  const tokens = await api<TokenResponse>("/auth/login", {
    method: "POST",
    json: { email, password },
    auth: false,
  });
  // The refresh token of the JSON body is ignored on purpose: the browser keeps the
  // HttpOnly cookie set by the same response.
  setAccessToken(tokens.access_token);
  rememberSession(true);
}

export function register(email: string, password: string, fullName: string): Promise<User> {
  return api<User>("/auth/register", {
    method: "POST",
    json: { email, password, full_name: fullName.trim() || null },
    auth: false,
  });
}

export function fetchCurrentUser(signal?: AbortSignal): Promise<User> {
  return api<User>("/auth/me", { signal });
}

export async function logout(): Promise<void> {
  try {
    await api<void>("/auth/logout", { method: "POST", auth: false });
  } finally {
    setAccessToken(null);
    rememberSession(false);
  }
}
