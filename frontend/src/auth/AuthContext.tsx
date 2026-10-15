import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";

import * as authApi from "../api/auth";
import { hasSessionHint, onSessionExpired, refreshSession } from "../api/client";
import type { User } from "../api/types";
import { AuthContext, type AuthContextValue, type AuthStatus } from "./context";

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>("loading");
  const [user, setUser] = useState<User | null>(null);

  // Restore the session on startup: the refresh cookie (if any) gives a new access token.
  useEffect(() => {
    let cancelled = false;
    async function restore(): Promise<void> {
      if (!hasSessionHint()) {
        setStatus("anonymous");
        return;
      }
      try {
        const refreshed = await refreshSession();
        const current = refreshed ? await authApi.fetchCurrentUser() : null;
        if (!cancelled) {
          setUser(current);
          setStatus(current ? "authenticated" : "anonymous");
        }
      } catch {
        if (!cancelled) {
          setStatus("anonymous");
        }
      }
    }
    void restore();
    return () => {
      cancelled = true;
    };
  }, []);

  // The refresh token expired or was revoked: back to the login page.
  useEffect(
    () =>
      onSessionExpired(() => {
        setUser(null);
        setStatus("anonymous");
      }),
    [],
  );

  const login = useCallback(async (email: string, password: string) => {
    await authApi.login(email, password);
    const current = await authApi.fetchCurrentUser();
    setUser(current);
    setStatus("authenticated");
  }, []);

  const register = useCallback(
    async (email: string, password: string, fullName: string) => {
      await authApi.register(email, password, fullName);
      await login(email, password);
    },
    [login],
  );

  const logout = useCallback(async () => {
    try {
      await authApi.logout();
    } catch {
      // Even if the server cannot be reached, the local session is dropped.
    }
    setUser(null);
    setStatus("anonymous");
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({ status, user, login, register, logout }),
    [status, user, login, register, logout],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
