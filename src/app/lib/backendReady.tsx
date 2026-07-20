import { createContext, useContext, useEffect, useState, ReactNode } from "react";

// Tracks the local engine's startup so the UI can show what's happening. The
// launcher opens the app window as soon as the cached frontend build is ready
// and warms the backend (heavy imports + AI model) in the background, so on a
// cold start there's a window where the UI is up but the engine is still coming
// online. Components read `ready` to defer backend-only work (session restore,
// reviewer list); the header shows `message` so the user sees incremental
// progress instead of a single static spinner.

export type EngineStatus = {
  ready: boolean;    // backend answers /api/health — screening/search/session APIs work
  message: string;   // human-readable phase, e.g. "Loading AI model…"
  done: boolean;     // engine fully up (model loaded or cloud provider) — hide the indicator
};

const INITIAL: EngineStatus = { ready: false, message: "Starting engine…", done: false };
const Ctx = createContext<EngineStatus>({ ready: true, message: "", done: true });

export function BackendReadyProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<EngineStatus>(INITIAL);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    // Once the backend answers, don't nag forever if the local model is slow to
    // load — after a grace period treat the engine as ready enough to hide the
    // indicator (the first screening call will finish loading the model).
    let readyPolls = 0;
    const READY_GRACE_POLLS = 20;   // ~30s at 1.5s cadence

    const poll = async () => {
      let next: EngineStatus;
      try {
        const r = await fetch("/api/health", { cache: "no-store" });
        if (!r.ok) throw new Error("not ok");
        const j: any = await r.json().catch(() => ({}));
        const cloud = j?.providers && Object.values(j.providers).some(Boolean);
        const models: string[] = j?.ollama?.loaded_models || [];
        const ollamaUp = !!j?.ollama?.reachable;
        readyPolls += 1;
        if (models.length > 0 || cloud) {
          next = { ready: true, message: "Engine ready", done: true };
        } else if (!ollamaUp) {
          next = { ready: true, message: "Starting AI runtime…", done: readyPolls >= READY_GRACE_POLLS };
        } else {
          next = { ready: true, message: "Loading AI model…", done: readyPolls >= READY_GRACE_POLLS };
        }
      } catch {
        next = { ready: false, message: "Starting engine…", done: false };
      }
      if (!active) return;
      setStatus(next);
      if (next.done) return;                 // fully up — stop polling
      timer = setTimeout(poll, 1500);
    };

    poll();
    return () => { active = false; if (timer) clearTimeout(timer); };
  }, []);

  return <Ctx.Provider value={status}>{children}</Ctx.Provider>;
}

// Full engine status for the startup indicator.
export function useEngineStatus(): EngineStatus {
  return useContext(Ctx);
}

// Backwards-compatible boolean: is the backend reachable? Used to gate backend-only
// work (session restore, reviewer list).
export function useBackendReady(): boolean {
  return useContext(Ctx).ready;
}
