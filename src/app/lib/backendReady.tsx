import { createContext, useContext, useEffect, useState, ReactNode } from "react";

// Tracks whether the local FastAPI backend has finished starting. The launcher
// now opens the app window as soon as the cached frontend build is ready and
// lets the backend (which imports heavy deps) warm up in the background, so on a
// cold start there's a short window where the UI is up but API calls would fail.
// Components read this to show a "starting" indicator and to defer backend-only
// work (session restore, reviewer list) until the backend answers.

const Ctx = createContext<boolean>(true);

export function BackendReadyProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const r = await fetch("/api/health", { cache: "no-store" });
        if (r.ok) { if (active) setReady(true); return; }   // backend is up — stop polling
      } catch { /* backend still importing — retry */ }
      if (active) timer = setTimeout(poll, 1500);
    };
    poll();
    return () => { active = false; if (timer) clearTimeout(timer); };
  }, []);

  return <Ctx.Provider value={ready}>{children}</Ctx.Provider>;
}

export function useBackendReady() {
  return useContext(Ctx);
}
