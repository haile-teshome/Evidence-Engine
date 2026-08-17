import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { apiFetch, getAuthToken, setAuthSession, clearAuthSession } from "./backendClient";

// Local accounts: each user signs up with an email + password (hashed on the
// backend) and logs in to their own account. The account id scopes all sessions
// and per-reviewer decisions, exactly as the old profile id did, so projects and
// autosave keep working per account. There is no cloud; everything is local.

export type AuthUser = { id: string; email: string; name?: string };
export type AuthStatus = "loading" | "authed" | "anon";

type AuthCtx = {
  user: AuthUser | null;
  status: AuthStatus;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string, name?: string) => Promise<void>;
  signOut: () => Promise<void>;
};

const Ctx = createContext<AuthCtx | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [status, setStatus] = useState<AuthStatus>("loading");

  // Restore the session on mount by validating any stored token with the backend.
  useEffect(() => {
    let active = true;
    (async () => {
      if (!getAuthToken()) { if (active) setStatus("anon"); return; }
      try {
        const r = await apiFetch("/auth/me");
        if (!active) return;
        setUser(r.user);
        setStatus("authed");
      } catch {
        clearAuthSession();
        if (active) setStatus("anon");
      }
    })();
    return () => { active = false; };
  }, []);

  async function adopt(r: { token: string; user: AuthUser }) {
    setAuthSession(r.token, r.user.id);
    setUser(r.user);
    setStatus("authed");
  }

  async function login(email: string, password: string) {
    const r = await apiFetch("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) });
    await adopt(r);
  }

  async function signup(email: string, password: string, name?: string) {
    const r = await apiFetch("/auth/signup", { method: "POST", body: JSON.stringify({ email, password, name: name || "" }) });
    await adopt(r);
  }

  async function signOut() {
    try { await apiFetch("/auth/logout", { method: "POST" }); } catch { /* ignore */ }
    clearAuthSession();
    setUser(null);
    setStatus("anon");
  }

  return (
    <Ctx.Provider value={{ user, status, login, signup, signOut }}>
      {children}
    </Ctx.Provider>
  );
}

const noopAuth: AuthCtx = {
  user: null,
  status: "anon",
  login: async () => { throw new Error("AuthProvider missing"); },
  signup: async () => { throw new Error("AuthProvider missing"); },
  signOut: async () => { /* noop */ },
};

export function useAuth() {
  return useContext(Ctx) ?? noopAuth;
}
