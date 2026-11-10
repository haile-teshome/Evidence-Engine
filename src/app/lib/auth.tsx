import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { apiFetch, getAuthToken, setAuthSession, clearAuthSession, type ApiError } from "./backendClient";
import { bootStep } from "./boot";

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
  //
  // The engine is still starting when the window opens, and until it answers,
  // every call fails. Reading that as "signed out" threw the reviewer back to
  // the login screen and deleted their token on the way, so opening the app
  // too quickly after launching it cost them their session. Only the engine
  // itself can end a session: a 401 or 403 means the token was rejected, and
  // anything else means we asked too early and should ask again.
  useEffect(() => {
    let active = true;
    (async () => {
      if (!getAuthToken()) { if (active) setStatus("anon"); return; }
      for (let attempt = 0; active; attempt++) {
        try {
          const r = await apiFetch("/auth/me");
          if (!active) return;
          setUser(r.user);
          setStatus("authed");
          return;
        } catch (e) {
          const status = (e as ApiError)?.status;
          if (status === 401 || status === 403) {   // the engine rejected the token
            clearAuthSession();
            if (active) setStatus("anon");
            return;
          }
          if (attempt >= 44) {                      // ~25s; let them sign in by hand
            if (active) setStatus("anon");          // the token stays, so a reload can still restore
            return;
          }
          await new Promise(r => setTimeout(r, attempt < 20 ? 250 : 1000));
        }
      }
    })();
    return () => { active = false; };
  }, []);

  // Whether the session came back or there was none to come back to, the
  // question has been answered and the startup screen can tick it off.
  useEffect(() => { bootStep("session", status === "loading" ? "running" : "done"); }, [status]);

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
