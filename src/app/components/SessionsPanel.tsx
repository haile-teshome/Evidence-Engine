import { useEffect, useRef, useState } from "react";
import { useStore, SESSION_STORAGE_KEY } from "../lib/store";
import { useAuth } from "../lib/auth";
import { useBackendReady } from "../lib/backendReady";
import { FRESH_LAUNCH } from "../lib/launchFlags";
import { listSessions, loadSession, saveSession, deleteSession, SessionMeta } from "../lib/sessions";
import { AIService } from "../lib/mockServices";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Label } from "./ui/label";
import { Plus, FolderOpen, Trash2, Cloud, CloudOff, Loader2, Check, Pin } from "lucide-react";
import { toast } from "sonner";

function genId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

// Deterministic short summary of a research goal — strips conversational lead-ins
// ("I want to know about the relationship between …") instead of slicing the
// question verbatim. Mirrors the backend heuristic; used for the instant title
// before the LLM's nicer one arrives.
const GOAL_LEADINS = /^(?:i\s+(?:want|would\s+like|need|wish|aim|plan|hope|am\s+looking)\s+to\s+(?:know|understand|find\s+out|learn|explore|investigate|examine|study|review|assess|evaluate|research|determine|see)(?:\s+(?:about|whether|if|how|what|the))?|what\s+(?:is|are|was|were)|how\s+(?:does|do|did|can|to)|can\s+you|could\s+you|please|tell\s+me\s+about|i'?m\s+interested\s+in|(?:a\s+)?(?:study|systematic\s+review|review|analysis|investigation|meta[-\s]analysis)\s+(?:of|on|about)|the\s+(?:relationship|association|effect|effects|impact|role|link|correlation)\s+(?:between|of|on)|explore|investigate|examine|assess|evaluate|determine|understand|is\s+there\s+(?:a|an)?)\s+/i;

function summarizeGoal(goal: string): string {
  let g = (goal || "").trim().replace(/[?.!\s]+$/, "");
  for (let i = 0; i < 3; i++) {
    const stripped = g.replace(GOAL_LEADINS, "").replace(/^[\s?.!,:]+/, "").trim();
    if (stripped === g || !stripped) break;
    g = stripped;
  }
  if (!g) g = (goal || "").trim();
  let short = g.split(/\s+/).slice(0, 8).join(" ").replace(/[\s?.!,:]+$/, "");
  if (short.length > 60) short = short.slice(0, 60).replace(/\s\S*$/, "");
  if (!short) return "Untitled session";
  return short.charAt(0).toUpperCase() + short.slice(1);
}

const PINNED_KEY = "ee:pinnedSessions";
function loadPinned(): Set<string> {
  try { return new Set(JSON.parse(localStorage.getItem(PINNED_KEY) || "[]")); } catch { return new Set(); }
}

// A single-line label that stays on one line, and on hover scrolls horizontally
// to reveal the full text (marquee-on-hover), then resets on leave.
function ScrollingText({ text }: { text: string }) {
  const spanRef = useRef<HTMLSpanElement>(null);
  const [tx, setTx] = useState(0);
  return (
    <div
      className="flex-1 min-w-0 overflow-hidden"
      title={text}
      onMouseEnter={() => {
        const el = spanRef.current;
        const parent = el?.parentElement;
        if (!el || !parent) return;
        const over = el.scrollWidth - parent.clientWidth;
        setTx(over > 2 ? over : 0);
      }}
      onMouseLeave={() => setTx(0)}
    >
      <span
        ref={spanRef}
        className="inline-block whitespace-nowrap transition-transform ease-linear"
        style={{ transform: `translateX(-${tx}px)`, transitionDuration: `${Math.max(tx * 15, 0)}ms` }}
      >
        {text}
      </span>
    </div>
  );
}

type SyncStatus = "idle" | "saving" | "synced" | "error";

export function SessionsPanel() {
  const s = useStore();
  const { user } = useAuth();
  const backendReady = useBackendReady();
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [busy, setBusy] = useState(false);
  const [syncStatus, setSyncStatus] = useState<SyncStatus>("idle");
  const [syncError, setSyncError] = useState<string | null>(null);
  const [lastSyncedAt, setLastSyncedAt] = useState<number | null>(null);
  // Inline rename (double-click a session to edit its name).
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  // Pinned sessions float to the top; persisted locally per browser.
  const [pinned, setPinned] = useState<Set<string>>(loadPinned);
  // Single vs double click: delay the load so a double-click can cancel it and
  // enter edit mode instead.
  const clickTimer = useRef<number | null>(null);
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
  // Wait for the backend to finish starting before hitting it, so a cold start
  // doesn't spuriously fail (and trip the "backend unreachable" toast).
  useEffect(() => { if (backendReady) refresh(); }, [user?.id, backendReady]);

  // On a fresh page load, silently restore the last active session so a browser
  // refresh keeps the user's work (and tab) in place. Runs once when auth is
  // ready, and only if nothing is already loaded — never clobbers active work.
  const restoredRef = useRef(false);
  useEffect(() => {
    // Wait for the backend: attempting the restore before it's up would fail and
    // clear SESSION_STORAGE_KEY below, permanently losing the auto-restore.
    if (!user || !backendReady || restoredRef.current) return;
    restoredRef.current = true;
    // Fresh app launch → start new; don't auto-load the last session.
    if (FRESH_LAUNCH) return;
    let savedId: string | null = null;
    try { savedId = localStorage.getItem(SESSION_STORAGE_KEY); } catch { /* ignore */ }
    if (!savedId) return;
    // The local snapshot may already have restored this session, but localStorage
    // is capped (~5 MB) and silently drops the heaviest late-stage fields
    // (extractions, quality, snowball) when it overflows. The backend KV store has
    // no such cap, so it's the authoritative, fuller copy — re-hydrate from it on
    // open. Guard against clobbering a DIFFERENT session or unsaved local work.
    if (s.currentSessionId && s.currentSessionId !== savedId) return;
    if (!s.currentSessionId && s.history.length > 0) return;
    (async () => {
      try {
        const sess = await loadSession(savedId!);
        // Reconcile (non-destructive): if the local snapshot already restored
        // late-stage tabs, an empty field in the backend copy must not wipe them.
        s.hydrate(sess.data, false);
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
  }, [user?.id, backendReady]);

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
            : summarizeGoal(firstGoal) || "Untitled session";
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
    if (!backendReady) return;   // don't try to save before the backend is up
    if (s.history.length === 0) return;

    if (saveTimer.current) window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(async () => {
      setSyncStatus("saving");
      try {
        const id = s.currentSessionId || genId();
        const firstGoal = s.history[0]?.goal || "";
        // Use a string-slice title up front so we never block on the LLM.
        const fallbackTitle = summarizeGoal(firstGoal);
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
    backendReady,
    s.history,
    s.pico,
    s.inclusion,
    s.exclusion,
    s.query,
    s.rawPapers,
    s.uniquePapers,
    s.qualityReports,
    s.qualityArchive,
    s.results,
    s.fullTextResults,
    s.snowballResults,
    s.snowballScreened,
    s.extractedPapers,
    s.textExtractions,
    s.gradeOutcomes,
    // Search-planning outputs (per-database queries + optimization runs).
    s.perDbQueries,
    s.unifiedSearchQuery,
    s.simulation,
    s.simulationRuns,
    s.dbTestResults,
    s.agenticTrace,
    s.agenticSummary,
    s.prisma,
  ]);

  if (!user) {
    return (
      <Card className="p-3 text-xs text-muted-foreground">
        Your work is saved locally on this computer automatically.
      </Card>
    );
  }

  // Persist a new name. For the active session we save the live snapshot; for
  // others we reload their data first so the rename doesn't clobber it.
  async function commitRename() {
    const id = editingId;
    const t = editValue.trim();
    setEditingId(null);
    if (!id || !t) return;
    setSessions(prev => prev.map(x => (x.id === id ? { ...x, title: t } : x)));
    if (id === s.currentSessionId) s.setCurrentSessionTitle(t);
    try {
      if (id === s.currentSessionId) await saveSession(id, t, s.snapshot());
      else { const sess = await loadSession(id); await saveSession(id, t, sess.data); }
    } catch { refresh(); }
  }

  function handleRowClick(id: string) {
    if (clickTimer.current) return;                       // dbl-click in progress
    clickTimer.current = window.setTimeout(() => { clickTimer.current = null; onLoad(id); }, 220);
  }
  function handleRowDblClick(id: string, title: string) {
    if (clickTimer.current) { clearTimeout(clickTimer.current); clickTimer.current = null; }
    setEditingId(id);
    setEditValue(title);
  }

  function togglePin(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    setPinned(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      try { localStorage.setItem(PINNED_KEY, JSON.stringify([...next])); } catch { /* ignore */ }
      return next;
    });
  }

  // Pinned sessions first; stable sort preserves the original (recency) order otherwise.
  const orderedSessions = [...sessions].sort((a, b) => (pinned.has(a.id) ? 0 : 1) - (pinned.has(b.id) ? 0 : 1));

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
        {orderedSessions.map(m => {
          const active = m.id === s.currentSessionId;
          const editing = editingId === m.id;
          const isPinned = pinned.has(m.id);
          return (
            <div key={m.id}
              className={`group flex items-center gap-1 text-xs rounded px-2 py-1.5 ${editing ? "" : "cursor-pointer"} ${active ? "bg-primary/10 border border-primary/30" : "hover:bg-muted"}`}
              onClick={() => { if (!editing) handleRowClick(m.id); }}
              onDoubleClick={() => handleRowDblClick(m.id, m.title)}
              title={editing ? undefined : "Double-click to rename"}>
              <FolderOpen className="size-3 shrink-0 text-muted-foreground" />
              {editing ? (
                <input
                  autoFocus
                  value={editValue}
                  onClick={e => e.stopPropagation()}
                  onChange={e => setEditValue(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === "Enter") { e.preventDefault(); commitRename(); }
                    else if (e.key === "Escape") { setEditingId(null); }
                  }}
                  onBlur={commitRename}
                  className="flex-1 min-w-0 bg-transparent border-b border-primary/50 outline-none px-0.5"
                />
              ) : (
                <ScrollingText text={m.title} />
              )}
              <button onClick={(e) => togglePin(m.id, e)}
                className={`shrink-0 ${isPinned ? "text-primary" : "opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-foreground"}`}
                title={isPinned ? "Unpin" : "Pin to top"}>
                <Pin className={`size-3 ${isPinned ? "fill-primary" : ""}`} />
              </button>
              <button onClick={(e) => onDelete(m.id, e)}
                className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive shrink-0">
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
