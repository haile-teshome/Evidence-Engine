import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { supabase, apiFetch, supabaseConfigured } from "./supabaseClient";
import { projectId, publicAnonKey } from "../../../utils/supabase/info";

type AuthUser = { id: string; email: string; name?: string };

export type OAuthProvider = "google" | "azure" | "github";
const SUPPORTED_PROVIDERS: OAuthProvider[] = ["google", "azure", "github"];

type AuthCtx = {
  user: AuthUser | null;
  loading: boolean;
  enabledProviders: Set<OAuthProvider>;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string, name: string) => Promise<void>;
  signInWithProvider: (provider: OAuthProvider) => Promise<void>;
  signOut: () => Promise<void>;
};

const Ctx = createContext<AuthCtx | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [enabledProviders, setEnabledProviders] = useState<Set<OAuthProvider>>(new Set());

  useEffect(() => {
    // No Supabase configured → render as a logged-out demo user, no network calls.
    if (!supabaseConfigured) {
      setLoading(false);
      return;
    }
    let active = true;
    supabase.auth.getSession().then(({ data }) => {
      if (!active) return;
      const u = data.session?.user;
      setUser(u ? { id: u.id, email: u.email || "", name: (u.user_metadata as any)?.name } : null);
      setLoading(false);
    });
    const { data: sub } = supabase.auth.onAuthStateChange((_evt, session) => {
      const u = session?.user;
      setUser(u ? { id: u.id, email: u.email || "", name: (u.user_metadata as any)?.name } : null);
    });
    // Probe Supabase Auth for which external providers are enabled. On failure
    // (network or misconfig) the set stays empty and OAuth buttons stay hidden.
    fetch(`https://${projectId}.supabase.co/auth/v1/settings`, {
      headers: { apikey: publicAnonKey },
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((data: { external?: Record<string, boolean> } | null) => {
        if (!active || !data?.external) return;
        setEnabledProviders(new Set(SUPPORTED_PROVIDERS.filter((p) => Boolean(data.external![p]))));
      })
      .catch(() => { /* silent — providers stay hidden */ });
    return () => { active = false; sub.subscription.unsubscribe(); };
  }, []);

  function ensureConfigured() {
    if (!supabaseConfigured) {
      throw new Error(
        "Supabase auth is not configured. Set VITE_SUPABASE_PROJECT_ID and " +
          "VITE_SUPABASE_ANON_KEY in .env.local to enable sign-in."
      );
    }
  }

  async function signIn(email: string, password: string) {
    ensureConfigured();
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) throw new Error(`Sign-in failed: ${error.message}`);
  }
  async function signUp(email: string, password: string, name: string) {
    ensureConfigured();
    await apiFetch("/signup", { method: "POST", body: JSON.stringify({ email, password, name }) });
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) throw new Error(`Auto sign-in after signup failed: ${error.message}`);
  }
  async function signInWithProvider(provider: OAuthProvider) {
    ensureConfigured();
    const { error } = await supabase.auth.signInWithOAuth({
      provider,
      options: { redirectTo: window.location.origin },
    });
    if (error) throw new Error(`OAuth (${provider}) failed: ${error.message}. Make sure the provider is enabled in your Supabase project.`);
  }
  async function signOut() {
    if (!supabaseConfigured) { setUser(null); return; }
    const { error } = await supabase.auth.signOut();
    if (error) console.error(`Sign-out error: ${error.message}`);
  }

  return <Ctx.Provider value={{ user, loading, enabledProviders, signIn, signUp, signInWithProvider, signOut }}>{children}</Ctx.Provider>;
}

const noopAuth: AuthCtx = {
  user: null,
  loading: false,
  enabledProviders: new Set(),
  signIn: async () => { throw new Error("AuthProvider missing"); },
  signUp: async () => { throw new Error("AuthProvider missing"); },
  signInWithProvider: async () => { throw new Error("AuthProvider missing"); },
  signOut: async () => { /* noop */ },
};

export function useAuth() {
  return useContext(Ctx) ?? noopAuth;
}
