import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { apiFetch, REVIEWER_ID_KEY, DEFAULT_REVIEWER_ID } from "./backendClient";

// Local reviewer profiles replace cloud accounts. A "user" is just a named
// profile (with an optional email) stored in the local backend. There are no
// passwords or sign-in — you pick or create a profile, and its id scopes your
// sessions and per-reviewer decisions in multi-reviewer projects.

export type AuthUser = { id: string; email: string; name?: string };

const DEFAULT_USER: AuthUser = { id: DEFAULT_REVIEWER_ID, email: "", name: "You" };

type AuthCtx = {
  user: AuthUser | null;
  loading: boolean;
  reviewers: AuthUser[];
  addReviewer: (name: string, email?: string) => Promise<void>;
  selectReviewer: (id: string) => void;
  signOut: () => void;                 // switch back to the default local profile
  refreshReviewers: () => Promise<void>;
};

const Ctx = createContext<AuthCtx | null>(null);

async function loadReviewers(): Promise<AuthUser[]> {
  try {
    const r = await apiFetch("/reviewers");
    const list: AuthUser[] = (r.reviewers || []).map((x: any) => ({
      id: x.id,
      email: x.email || "",
      name: x.name || "Reviewer",
    }));
    return list.length ? list : [DEFAULT_USER];
  } catch {
    // Backend not up yet (or offline) — fall back to the default profile so the
    // app is always usable locally.
    return [DEFAULT_USER];
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(DEFAULT_USER);
  const [reviewers, setReviewers] = useState<AuthUser[]>([DEFAULT_USER]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    (async () => {
      const list = await loadReviewers();
      if (!active) return;
      setReviewers(list);
      let savedId = DEFAULT_REVIEWER_ID;
      try { savedId = localStorage.getItem(REVIEWER_ID_KEY) || DEFAULT_REVIEWER_ID; } catch { /* ignore */ }
      const found = list.find(r => r.id === savedId) || list.find(r => r.id === DEFAULT_REVIEWER_ID) || list[0] || DEFAULT_USER;
      setUser(found);
      try { localStorage.setItem(REVIEWER_ID_KEY, found.id); } catch { /* ignore */ }
      setLoading(false);
    })();
    return () => { active = false; };
  }, []);

  async function refreshReviewers() {
    setReviewers(await loadReviewers());
  }

  function selectReviewer(id: string) {
    const found = reviewers.find(r => r.id === id) || DEFAULT_USER;
    setUser(found);
    try { localStorage.setItem(REVIEWER_ID_KEY, found.id); } catch { /* ignore */ }
  }

  async function addReviewer(name: string, email?: string) {
    const r = await apiFetch("/reviewers", { method: "POST", body: JSON.stringify({ name, email: email || "" }) });
    const nu: AuthUser = { id: r.reviewer.id, email: r.reviewer.email || "", name: r.reviewer.name };
    setReviewers(await loadReviewers());
    setUser(nu);
    try { localStorage.setItem(REVIEWER_ID_KEY, nu.id); } catch { /* ignore */ }
  }

  function signOut() {
    selectReviewer(DEFAULT_REVIEWER_ID);
  }

  return (
    <Ctx.Provider value={{ user, loading, reviewers, addReviewer, selectReviewer, signOut, refreshReviewers }}>
      {children}
    </Ctx.Provider>
  );
}

const noopAuth: AuthCtx = {
  user: DEFAULT_USER,
  loading: false,
  reviewers: [DEFAULT_USER],
  addReviewer: async () => { throw new Error("AuthProvider missing"); },
  selectReviewer: () => { /* noop */ },
  signOut: () => { /* noop */ },
  refreshReviewers: async () => { /* noop */ },
};

export function useAuth() {
  return useContext(Ctx) ?? noopAuth;
}
