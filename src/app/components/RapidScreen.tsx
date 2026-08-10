import { useEffect, useMemo, useState } from "react";
import { Dialog, DialogContent } from "./ui/dialog";
import { Button } from "./ui/button";
import { Check, X, SkipForward, Undo2, Sparkles, CheckCircle2, ExternalLink, ScanLine, Eye, EyeOff, Star, Search } from "lucide-react";
import { activeRank } from "../lib/activeLearning";

// One record to review, normalized so the same rapid reviewer drives both
// abstract and full-text screening (which use different decision casings).
export type RapidItem = {
  id: string;
  title: string;
  source: string;
  url: string;
  text: string;                         // abstract or full text
  aiInclude: boolean;                   // the AI's decision
  reason: string;
  override?: "include" | "exclude";     // the reviewer's current override, if any
};

// Small keycap, tinted to sit on either a solid or a subtle button.
function Kbd({ children, tone = "muted" }: { children: React.ReactNode; tone?: "muted" | "onColor" }) {
  return (
    <kbd className={`ml-1.5 text-[10px] font-medium leading-none px-1 py-0.5 rounded ${tone === "onColor" ? "bg-white/25 text-white" : "bg-muted text-muted-foreground"}`}>
      {children}
    </kbd>
  );
}

// Keyboard-driven rapid review, reused across screening stages. i/e set an
// include/exclude override, s skips, b goes back, f stars a card, / searches.
// The current position, and which cards are starred, persist per stage so the
// reviewer can close and resume where they left off.
export function RapidScreen({
  open, onClose, items, onDecide, label = "Rapid screen",
}: {
  open: boolean;
  onClose: () => void;
  items: RapidItem[];
  onDecide: (id: string, decision: "include" | "exclude") => void;
  label?: string;
}) {
  // ── Per-stage persistence (position + starred cards) ──────────────────────
  const storeKey = `rapidscreen:v1:${label}`;
  const readStore = (): { lastId?: string; flags?: string[] } => {
    try { return JSON.parse(localStorage.getItem(storeKey) || "{}"); } catch { return {}; }
  };
  const writeStore = (patch: object) => {
    try { localStorage.setItem(storeKey, JSON.stringify({ ...readStore(), ...patch })); } catch { /* ignore */ }
  };

  const [learn, setLearn] = useState(false);
  // Blinded by default: hide the AI decision + reasoning so the reviewer's own
  // judgement isn't anchored by the machine (automation bias). Reveal on demand.
  const [showAI, setShowAI] = useState(false);
  const [flags, setFlags] = useState<Set<string>>(() => new Set(readStore().flags || []));
  const [browseOpen, setBrowseOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [onlyStarred, setOnlyStarred] = useState(false);

  const al = useMemo(() => activeRank(items), [items]);
  const byId = useMemo(() => new Map(items.map(it => [it.id, it])), [items]);

  // Freeze the review order so the list doesn't reshuffle under the reviewer as
  // they label (which otherwise makes the index skip cards and get stuck at the
  // end). Recomputed only when the dialog opens or the mode changes.
  const [order, setOrder] = useState<string[]>([]);
  const [i, setI] = useState(0);
  useEffect(() => {
    if (!open) return;
    const ids = learn
      ? al.order
      : [...items].sort((a, b) => {
          const aRev = a.override ? 1 : 0, bRev = b.override ? 1 : 0;
          if (aRev !== bRev) return aRev - bRev;
          return (a.aiInclude ? 0 : 1) - (b.aiInclude ? 0 : 1);
        }).map(it => it.id);
    setOrder(ids);
    // Resume: jump to the last card viewed, else the first unreviewed one.
    const saved = readStore().lastId;
    let start = saved ? ids.indexOf(saved) : -1;
    if (start < 0) start = ids.findIndex(id => !byId.get(id)?.override);
    setI(start < 0 ? 0 : start);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, learn]);

  const queue = useMemo(
    () => order.map(id => byId.get(id)).filter(Boolean) as typeof items,
    [order, byId],
  );

  const cur = queue[i];
  const reviewed = items.filter(x => x.override).length;
  const pct = items.length ? (reviewed / items.length) * 100 : 0;
  const safeToStop = learn && al.reviewed > 0 && (al.predictedRemaining === 0 || (al.estRecall != null && al.estRecall >= 0.95));

  const next = () => setI(v => Math.min(queue.length - 1, v + 1));
  const back = () => setI(v => Math.max(0, v - 1));
  const decide = (d: "include" | "exclude") => { if (cur) { onDecide(cur.id, d); next(); } };
  const jumpTo = (idx: number) => { setI(idx); setBrowseOpen(false); setSearch(""); };
  const toggleFlag = (id?: string) => {
    if (!id) return;
    setFlags(prev => { const n = new Set(prev); n.has(id) ? n.delete(id) : n.add(id); return n; });
  };

  // Persist position + flags.
  useEffect(() => { if (open && cur) writeStore({ lastId: cur.id }); /* eslint-disable-next-line */ }, [open, cur?.id]);
  useEffect(() => { writeStore({ flags: [...flags] }); /* eslint-disable-next-line */ }, [flags]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      const typing = !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA");
      const k = e.key.toLowerCase();
      if (k === "escape") { e.preventDefault(); browseOpen ? setBrowseOpen(false) : onClose(); return; }
      if (typing || browseOpen) return;
      if (k === "i") { e.preventDefault(); decide("include"); }
      else if (k === "e") { e.preventDefault(); decide("exclude"); }
      else if (k === "s") { e.preventDefault(); next(); }
      else if (k === "b" || k === "backspace") { e.preventDefault(); back(); }
      else if (k === "f") { e.preventDefault(); toggleFlag(cur?.id); }
      else if (k === "/") { e.preventDefault(); setBrowseOpen(true); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, cur, queue, browseOpen]);

  if (!open) return null;
  const myOverride = cur?.override;
  const curFlagged = !!cur && flags.has(cur.id);

  // "Stage: detail" → a two-line title in the header.
  const [labTitle, labSub] = label.includes(":")
    ? [label.slice(0, label.indexOf(":")).trim(), label.slice(label.indexOf(":") + 1).trim()]
    : [label, ""];

  const browseList = queue
    .map((it, idx) => ({ it, idx }))
    .filter(({ it }) => !onlyStarred || flags.has(it.id))
    .filter(({ it }) => {
      const q = search.trim().toLowerCase();
      return !q || it.title.toLowerCase().includes(q) || it.text.toLowerCase().includes(q);
    });

  const dotClass = (it: RapidItem) =>
    it.override === "include" ? "bg-emerald-500" : it.override === "exclude" ? "bg-rose-500" : "bg-muted-foreground/30";

  return (
    <Dialog open={open} onOpenChange={v => !v && onClose()}>
      <DialogContent className="w-[90vw] max-w-[780px] sm:max-w-[780px] h-[88vh] flex flex-col p-0 gap-0 overflow-hidden bg-muted [&>button]:hidden">
        {/* ── Header ─────────────────────────────────────────────────────── */}
        <div className="shrink-0 bg-card border-b px-5 py-3 space-y-2.5">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="size-8 rounded-lg bg-primary/10 text-primary grid place-items-center shrink-0">
                <ScanLine className="size-4" />
              </div>
              <div className="min-w-0 leading-tight">
                <div className="text-sm font-semibold">{labTitle}</div>
                {labSub && <div className="text-[11px] text-muted-foreground capitalize">{labSub}</div>}
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <button
                onClick={() => setBrowseOpen(true)}
                title="Search and jump to any card (/)"
                className="text-[11px] px-2.5 py-1.5 rounded-full border inline-flex items-center gap-1.5 text-muted-foreground hover:bg-muted transition-colors"
              >
                <Search className="size-3" />Find
              </button>
              <button
                onClick={() => setShowAI(v => !v)}
                title={showAI ? "Hide the AI decision and reasoning while you screen" : "Reveal the AI decision and reasoning (may bias your judgement)"}
                className={`text-[11px] px-2.5 py-1.5 rounded-full border inline-flex items-center gap-1.5 transition-colors ${showAI ? "bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-950/40 dark:text-amber-300 dark:border-amber-900" : "text-muted-foreground hover:bg-muted"}`}
              >
                {showAI ? <Eye className="size-3" /> : <EyeOff className="size-3" />}AI {showAI ? "shown" : "hidden"}
              </button>
              <button
                onClick={() => setLearn(v => !v)}
                title="Rank the queue by a model that learns from your labels, and estimate recall to suggest when to stop"
                className={`text-[11px] px-2.5 py-1.5 rounded-full border inline-flex items-center gap-1.5 transition-colors ${learn ? "bg-primary text-primary-foreground border-primary shadow-sm" : "text-muted-foreground hover:bg-muted"}`}
              >
                <Sparkles className="size-3" />Active learning {learn ? "on" : "off"}
              </button>
              <button onClick={onClose} className="size-7 grid place-items-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground" title="Close (Esc)">
                <X className="size-4" />
              </button>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="h-2 flex-1 rounded-full bg-muted overflow-hidden">
              <div className="h-full rounded-full bg-gradient-to-r from-primary/80 to-primary transition-all duration-300" style={{ width: `${pct}%` }} />
            </div>
            <div className="text-xs text-muted-foreground tabular-nums shrink-0">
              <span className="font-medium text-foreground">{reviewed}</span> / {items.length} reviewed
              {queue.length ? <span className="mx-1.5 text-border">·</span> : null}
              {queue.length ? `card ${Math.min(i + 1, queue.length)} of ${queue.length}` : null}
              {flags.size ? <span className="mx-1.5 text-border">·</span> : null}
              {flags.size ? <span className="inline-flex items-center gap-0.5 text-amber-600"><Star className="size-3 fill-amber-400 text-amber-500" />{flags.size}</span> : null}
            </div>
          </div>

          {learn && (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
              <span>{al.includesFound} includes found</span>
              <span>~{al.predictedRemaining} predicted relevant remaining</span>
              {al.estRecall != null && <span>est. recall {Math.round(al.estRecall * 100)}%</span>}
              {!al.trained && <span className="text-amber-600">learning: label a few includes and excludes</span>}
              {safeToStop && <span className="inline-flex items-center gap-1 text-emerald-600 font-medium"><CheckCircle2 className="size-3.5" />safe to stop</span>}
            </div>
          )}
        </div>

        {/* ── Body: review card, or the browse/search overlay ─────────────── */}
        <div className="flex-1 min-h-0 relative">
          {!cur ? (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-muted-foreground">Nothing to review yet.</div>
          ) : (
            <div className="absolute inset-0 overflow-y-auto p-4 sm:p-6">
              <div className="mx-auto max-w-2xl rounded-xl border bg-card shadow-sm">
                <div className="p-5 sm:p-6 space-y-4">
                  {/* title + star + reviewer decision */}
                  <div className="flex items-start justify-between gap-3">
                    <h3 className="text-lg font-semibold leading-snug tracking-tight">{cur.title}</h3>
                    <div className="flex items-center gap-2 shrink-0">
                      <button
                        onClick={() => toggleFlag(cur.id)}
                        title={curFlagged ? "Unstar (f)" : "Star to revisit (f)"}
                        className={`size-8 grid place-items-center rounded-md border transition-colors ${curFlagged ? "bg-amber-50 border-amber-200 dark:bg-amber-950/40 dark:border-amber-900" : "hover:bg-muted"}`}
                      >
                        <Star className={`size-4 ${curFlagged ? "fill-amber-400 text-amber-500" : "text-muted-foreground"}`} />
                      </button>
                      {myOverride && (
                        <span className={`text-[11px] font-medium px-2 py-1 rounded-full border ${myOverride === "include" ? "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-300 dark:border-emerald-900" : "bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-950/40 dark:text-rose-300 dark:border-rose-900"}`}>
                          Your call: {myOverride}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* meta pills */}
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <span className="inline-flex items-center px-2 py-1 rounded-md bg-muted text-muted-foreground font-medium">{cur.source || "Unknown source"}</span>
                    {showAI && (
                      <span className={`inline-flex items-center px-2 py-1 rounded-md font-medium border ${cur.aiInclude ? "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-300 dark:border-emerald-900" : "bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-950/40 dark:text-rose-300 dark:border-rose-900"}`}>
                        AI suggests {cur.aiInclude ? "include" : "exclude"}
                      </span>
                    )}
                    {cur.url && (
                      <a href={cur.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-primary hover:bg-primary/10 font-medium">
                        <ExternalLink className="size-3.5" />View source
                      </a>
                    )}
                  </div>

                  {/* AI reasoning callout */}
                  {showAI && cur.reason && (
                    <div className="rounded-lg border border-border/70 bg-muted/40 p-3 pl-3.5 border-l-2 border-l-primary/50">
                      <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                        <Sparkles className="size-3" />AI reasoning
                      </div>
                      <p className="text-[13px] leading-relaxed text-foreground/80">{cur.reason}</p>
                    </div>
                  )}

                  {/* abstract / full text */}
                  <div className="pt-1">
                    <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground mb-1.5">
                      {labSub && /full/i.test(labSub) ? "Full text" : "Abstract"}
                    </div>
                    {cur.text
                      ? <p className="text-[15px] leading-relaxed text-foreground/90 whitespace-pre-wrap">{cur.text}</p>
                      : <p className="text-sm text-muted-foreground italic">No text available for this record.</p>}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Browse / search overlay */}
          {browseOpen && (
            <div className="absolute inset-0 z-20 bg-card flex flex-col">
              <div className="shrink-0 p-3 border-b flex items-center gap-2">
                <div className="relative flex-1">
                  <Search className="size-4 absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground" />
                  <input
                    autoFocus
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                    placeholder="Search titles and abstracts…"
                    className="w-full h-9 pl-8 pr-3 rounded-md border bg-background text-sm outline-none focus:ring-2 focus:ring-ring/40"
                  />
                </div>
                <button
                  onClick={() => setOnlyStarred(v => !v)}
                  title="Show starred only"
                  className={`h-9 px-3 rounded-md border text-xs inline-flex items-center gap-1.5 transition-colors ${onlyStarred ? "bg-amber-50 text-amber-800 border-amber-300 dark:bg-amber-950/40 dark:text-amber-300 dark:border-amber-900" : "text-muted-foreground hover:bg-muted"}`}
                >
                  <Star className={`size-3.5 ${onlyStarred ? "fill-amber-400 text-amber-500" : ""}`} />Starred
                </button>
                <Button variant="ghost" className="h-9" onClick={() => { setBrowseOpen(false); setSearch(""); }}>Done</Button>
              </div>
              <div className="flex-1 overflow-y-auto p-2">
                {browseList.length === 0 ? (
                  <div className="p-6 text-center text-sm text-muted-foreground">No cards match.</div>
                ) : browseList.map(({ it, idx }) => (
                  <div
                    key={it.id}
                    className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-md hover:bg-muted cursor-pointer ${idx === i ? "bg-muted" : ""}`}
                    onClick={() => jumpTo(idx)}
                  >
                    <span className={`size-2 rounded-full shrink-0 ${dotClass(it)}`} title={it.override ? `Your call: ${it.override}` : "Unreviewed"} />
                    <div className="min-w-0 flex-1">
                      <div className="text-sm leading-snug truncate">{it.title}</div>
                      <div className="text-[11px] text-muted-foreground truncate">{it.source || "Unknown source"}</div>
                    </div>
                    <button
                      onClick={e => { e.stopPropagation(); toggleFlag(it.id); }}
                      title={flags.has(it.id) ? "Unstar" : "Star to revisit"}
                      className="size-7 grid place-items-center rounded-md hover:bg-background shrink-0"
                    >
                      <Star className={`size-4 ${flags.has(it.id) ? "fill-amber-400 text-amber-500" : "text-muted-foreground/50"}`} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── Actions ────────────────────────────────────────────────────── */}
        <div className="shrink-0 bg-card border-t px-5 py-3">
          <div className="flex items-center justify-center gap-2 flex-nowrap">
            <Button onClick={() => decide("include")} disabled={!cur} className="h-10 px-3.5 bg-emerald-600 hover:bg-emerald-700 text-white shadow-sm">
              <Check className="size-4 mr-1" />Include<Kbd tone="onColor">i</Kbd>
            </Button>
            <Button onClick={() => decide("exclude")} disabled={!cur} className="h-10 px-3.5 bg-rose-600 hover:bg-rose-700 text-white shadow-sm">
              <X className="size-4 mr-1" />Exclude<Kbd tone="onColor">e</Kbd>
            </Button>
            <span className="mx-0.5 h-6 w-px bg-border shrink-0" aria-hidden="true" />
            <Button variant="ghost" onClick={() => toggleFlag(cur?.id)} disabled={!cur} className={`h-10 px-2.5 ${curFlagged ? "text-amber-600" : "text-muted-foreground"}`}>
              <Star className={`size-4 mr-1 ${curFlagged ? "fill-amber-400 text-amber-500" : ""}`} />Star<Kbd>f</Kbd>
            </Button>
            <Button variant="ghost" onClick={next} disabled={!cur} className="h-10 px-2.5 text-muted-foreground">
              <SkipForward className="size-4 mr-1" />Skip<Kbd>s</Kbd>
            </Button>
            <Button variant="ghost" onClick={back} disabled={i === 0} className="h-10 px-2.5 text-muted-foreground">
              <Undo2 className="size-4 mr-1" />Back<Kbd>b</Kbd>
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
