import { useEffect, useRef, useState } from "react";
import { useStore, SESSION_STORAGE_KEY } from "../lib/store";
import { useAuth } from "../lib/auth";
import { listSessions, loadSession, saveSession, deleteSession, SessionMeta } from "../lib/sessions";
import { AIService } from "../lib/mockServices";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Label } from "./ui/label";
import { Plus, FolderOpen, Trash2, Cloud, CloudOff, Loader2, Check } from "lucide-react";
import { toast } from "sonner";

function genId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

type SyncStatus = "idle" | "saving" | "synced" | "error";

export function SessionsPanel() {
  const s = useStore();
  const { user } = useAuth();
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [busy, setBusy] = useState(false);
  const [syncStatus, setSyncStatus] = useState<SyncStatus>("idle");
  const [syncError, setSyncError] = useState<string | null>(null);
  const [lastSyncedAt, setLastSyncedAt] = useState<number | null>(null);
  // Show the "sync failed" toast at most once per session to avoid spamming.
  const errorToastShown = useRef(false);

  // Debounce auto-saves so a flurry of state changes during one analysis run
  // collapses into a single write.
  const saveTimer = useRef<number | null>(null);
  // Track which session ids we've already kicked off an LLM title for, so we
  // don't fire the title call repeatedly during the analysis.
  const titledRef = useRef<Set<string>>(new Set());

  async function refresh() {
    if (!user) { setSessions([]); return; }
    try {
      const items = await listSessions();
      setSessions(items);
      // A successful list call means the backend is reachable — clear errors.
      if (syncStatus === "error") {
        setSyncStatus("idle");
        setSyncError(null);
      }
    } catch (e: any) {
      console.error(`Failed to list sessions: ${e?.message}`);
      setSyncStatus("error");
      setSyncError(e?.message || "Sessions backend unreachable");
      if (!errorToastShown.current) {
        errorToastShown.current = true;
        toast.error(
          "Cannot reach the sessions backend. Sessions won't persist across logout until this is fixed.",
          { duration: 8000 },
        );
      }
    }
  }
  useEffect(() => { refresh(); }, [user?.id]);

  // On a fresh page load, silently restore the last active session so a browser
  // refresh keeps the user's work (and tab) in place. Runs once when auth is
  // ready, and only if nothing is already loaded — never clobbers active work.
  const restoredRef = useRef(false);
  useEffect(() => {
    if (!user || restoredRef.current) return;
    if (s.currentSessionId || s.history.length > 0) { restoredRef.current = true; return; }
    let savedId: string | null = null;
    try { savedId = localStorage.getItem(SESSION_STORAGE_KEY); } catch { /* ignore */ }
    restoredRef.current = true;
    if (!savedId) return;
    (async () => {
      try {
        const sess = await loadSession(savedId!);
        s.hydrate(sess.data);
        s.setCurrentSessionId(sess.id);
        s.setCurrentSessionTitle(sess.title);
        // Intentionally do NOT change the page — keep the tab restored from
        // storage so the refresh lands exactly where the user was.
      } catch (e: any) {
        // Stale/inaccessible session id — clear it so we don't retry forever.
        console.error("Auto-restore session failed:", e?.message);
        try { localStorage.removeItem(SESSION_STORAGE_KEY); } catch { /* ignore */ }
      }
    })();
  }, [user?.id]);

  async function onLoad(id: string) {
    setBusy(true);
    try {
      const sess = await loadSession(id);
      s.hydrate(sess.data);
      s.setCurrentSessionId(sess.id);
      s.setCurrentSessionTitle(sess.title);
      s.setPage("home");
      toast.success(`Loaded "${sess.title}"`);
    } catch (e: any) { toast.error(e.message || "Load failed"); }
    finally { setBusy(false); }
  }

  async function onDelete(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    try {
      await deleteSession(id);
      if (s.currentSessionId === id) s.reset();
      refresh();
    } catch (e: any) { toast.error(e.message || "Delete failed"); }
  }

  async function onNew() {
    // Flush any pending debounced save synchronously so the current work is
    // persisted before we wipe the store.
    if (saveTimer.current) {
      window.clearTimeout(saveTimer.current);
      saveTimer.current = null;
    }
    if (user && s.history.length > 0) {
      try {
        const id = s.currentSessionId || genId();
        const firstGoal = s.history[0]?.goal || "";
        const title =
          s.currentSessionTitle && s.currentSessionTitle !== "Untitled session"
            ? s.currentSessionTitle
            : (firstGoal.slice(0, 50) + (firstGoal.length > 50 ? "…" : "")) || "Untitled session";
        await saveSession(id, title, s.snapshot());
        await refresh();
      } catch (e) {
        console.error("Save before new session failed:", e);
      }
    }
    s.reset();
  }

  // ---------------------------------------------------------------------------
  // Auto-save: fires whenever the snapshot-relevant state changes, debounced.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!user) return;
    if (s.history.length === 0) return;

    if (saveTimer.current) window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(async () => {
      setSyncStatus("saving");
      try {
        const id = s.currentSessionId || genId();
        const firstGoal = s.history[0]?.goal || "";
        // Use a string-slice title up front so we never block on the LLM.
        const fallbackTitle = firstGoal.slice(0, 50) + (firstGoal.length > 50 ? "…" : "");
        let title = s.currentSessionTitle && s.currentSessionTitle !== "Untitled session"
          ? s.currentSessionTitle
          : fallbackTitle || "Untitled session";

        if (!s.currentSessionId) {
          s.setCurrentSessionId(id);
          s.setCurrentSessionTitle(title);
        }
        await saveSession(id, title, s.snapshot());
        setSyncStatus("synced");
        setSyncError(null);
        setLastSyncedAt(Date.now());
        refresh();

        // Upgrade the title via LLM exactly once per session, in the background.
        // Subsequent auto-saves will reuse the upgraded title via s.currentSessionTitle.
        if (!titledRef.current.has(id) && firstGoal) {
          titledRef.current.add(id);
          AIService.generateSessionTitle(firstGoal)
            .then(async (better) => {
              const clean = (better || "").trim();
              if (clean && clean !== title) {
                s.setCurrentSessionTitle(clean);
                await saveSession(id, clean, s.snapshot()).catch(() => {});
                refresh();
              }
            })
            .catch(() => { /* keep fallback title */ });
        }
      } catch (e: any) {
        console.error("Auto-save failed:", e);
        setSyncStatus("error");
        setSyncError(e?.message || "Save failed");
        if (!errorToastShown.current) {
          errorToastShown.current = true;
          toast.error(
            `Cannot save to your account: ${e?.message?.slice(0, 80) || "backend unreachable"}`,
            { duration: 8000 },
          );
        }
      }
    }, 1500);

    return () => {
      if (saveTimer.current) window.clearTimeout(saveTimer.current);
    };
    // Re-run when any major piece of the snapshot changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    user?.id,
    s.history,
    s.pico,
    s.inclusion,
    s.exclusion,
    s.query,
    s.rawPapers,
    s.uniquePapers,
    s.qualityReports,
    s.results,
    s.fullTextResults,
    s.snowballResults,
    s.snowballScreened,
    s.extractedPapers,
    s.prisma,
  ]);

  if (!user) {
    return (
      <Card className="p-3 text-xs text-muted-foreground">
        Sign in (top-right) to save and sync research sessions.
      </Card>
    );
  }

  return (
    <Card className="p-3 space-y-2">
      <div className="flex items-center justify-between">
        <Label className="block text-xs">Sessions</Label>
        <SyncIndicator status={syncStatus} lastSyncedAt={lastSyncedAt} error={syncError} />
      </div>
      <Button size="sm" variant="outline" onClick={onNew} className="w-full">
        <Plus className="size-3 mr-1" />New
      </Button>
      <div className="space-y-1 max-h-48 overflow-y-auto">
        {sessions.length === 0 && (
          <div className="text-xs text-muted-foreground">
            Sessions auto-save to your account.
          </div>
        )}
        {sessions.map(m => {
          const active = m.id === s.currentSessionId;
          return (
            <div key={m.id}
              className={`group flex items-center gap-1 text-xs rounded px-2 py-1.5 cursor-pointer ${active ? "bg-primary/10 border border-primary/30" : "hover:bg-muted"}`}
              onClick={() => onLoad(m.id)}>
              <FolderOpen className="size-3 shrink-0 text-muted-foreground" />
              <div className="flex-1 truncate">{m.title}</div>
              <button onClick={(e) => onDelete(m.id, e)}
                className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive">
                <Trash2 className="size-3" />
              </button>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function SyncIndicator({
  status,
  lastSyncedAt,
  error,
}: {
  status: SyncStatus;
  lastSyncedAt: number | null;
  error: string | null;
}) {
  if (status === "saving") {
    return (
      <span className="text-[10px] text-muted-foreground flex items-center gap-1">
        <Loader2 className="size-3 animate-spin" />Saving…
      </span>
    );
  }
  if (status === "synced") {
    const ago = lastSyncedAt ? Math.max(1, Math.round((Date.now() - lastSyncedAt) / 1000)) : null;
    return (
      <span className="text-[10px] text-muted-foreground flex items-center gap-1" title={ago ? `Synced ${ago}s ago` : "Synced"}>
        <Check className="size-3 text-primary" />Synced
      </span>
    );
  }
  if (status === "error") {
    return (
      <span
        className="text-[10px] text-destructive flex items-center gap-1"
        title={error || "Sync failed"}
      >
        <CloudOff className="size-3" />Sync failed
      </span>
    );
  }
  return (
    <span className="text-[10px] text-muted-foreground flex items-center gap-1">
      <Cloud className="size-3" />Auto-save
    </span>
  );
}
