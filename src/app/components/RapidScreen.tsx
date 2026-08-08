import { useEffect, useMemo, useState } from "react";
import { Dialog, DialogContent } from "./ui/dialog";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Check, X, SkipForward, Undo2, Bot } from "lucide-react";

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

// Keyboard-driven rapid review, reused across screening stages. i/e set an
// include/exclude override, a accepts the AI call, s skips, b goes back.
// Unreviewed records (and AI includes) surface first.
export function RapidScreen({
  open, onClose, items, onDecide, label = "Rapid screen",
}: {
  open: boolean;
  onClose: () => void;
  items: RapidItem[];
  onDecide: (id: string, decision: "include" | "exclude") => void;
  label?: string;
}) {
  const queue = useMemo(() => {
    return [...items].sort((a, b) => {
      const aRev = a.override ? 1 : 0, bRev = b.override ? 1 : 0;
      if (aRev !== bRev) return aRev - bRev;
      return (a.aiInclude ? 0 : 1) - (b.aiInclude ? 0 : 1);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items, open]);

  const [i, setI] = useState(0);
  useEffect(() => { if (open) setI(0); }, [open]);

  const cur = queue[i];
  const reviewed = items.filter(x => x.override).length;

  const next = () => setI(v => Math.min(queue.length - 1, v + 1));
  const back = () => setI(v => Math.max(0, v - 1));
  const decide = (d: "include" | "exclude") => { if (cur) { onDecide(cur.id, d); next(); } };
  const acceptAI = () => { if (cur) { onDecide(cur.id, cur.aiInclude ? "include" : "exclude"); next(); } };

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (k === "i") { e.preventDefault(); decide("include"); }
      else if (k === "e") { e.preventDefault(); decide("exclude"); }
      else if (k === "a") { e.preventDefault(); acceptAI(); }
      else if (k === "s") { e.preventDefault(); next(); }
      else if (k === "b" || k === "backspace") { e.preventDefault(); back(); }
      else if (k === "escape") { onClose(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, cur, queue]);

  if (!open) return null;
  const myOverride = cur?.override;

  return (
    <Dialog open={open} onOpenChange={v => !v && onClose()}>
      <DialogContent className="w-[92vw] max-w-[860px] h-[88vh] flex flex-col p-0 gap-0">
        <div className="px-5 py-3 border-b space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">{label}</span>
            <span className="text-muted-foreground">{reviewed} of {items.length} reviewed · card {Math.min(i + 1, queue.length)} / {queue.length}</span>
          </div>
          <div className="h-1.5 rounded-full bg-muted overflow-hidden">
            <div className="h-full bg-primary transition-all" style={{ width: `${items.length ? (reviewed / items.length) * 100 : 0}%` }} />
          </div>
        </div>

        {!cur ? (
          <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground">Nothing to review yet.</div>
        ) : (
          <div className="flex-1 min-h-0 flex flex-col">
            <div className="px-5 py-4 border-b">
              <div className="flex items-start justify-between gap-3">
                <h3 className="font-medium leading-snug">{cur.title}</h3>
                {myOverride && <Badge variant={myOverride === "include" ? "default" : "secondary"}>you: {myOverride}</Badge>}
              </div>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-1.5 text-xs text-muted-foreground">
                <span>{cur.source}</span>
                <span className="inline-flex items-center gap-1"><Bot className="size-3" />AI: <span className={cur.aiInclude ? "text-emerald-600" : "text-rose-600"}>{cur.aiInclude ? "include" : "exclude"}</span></span>
                {cur.url && <a href={cur.url} target="_blank" rel="noreferrer" className="text-primary hover:underline">source</a>}
              </div>
              {cur.reason && <p className="text-xs text-muted-foreground mt-1.5 italic">AI reason: {cur.reason}</p>}
            </div>
            <div className="flex-1 overflow-y-auto px-5 py-4">
              <p className="text-sm leading-relaxed whitespace-pre-wrap">{cur.text || <span className="text-muted-foreground">No text.</span>}</p>
            </div>
          </div>
        )}

        {/* actions: a compact, centered pod so every control is within a single
            hand's reach; icons are color-coded and tooltipped, with a key
            legend below. The 8px gap keeps them from being fat-fingered. */}
        <div className="border-t">
          <div className="px-5 pt-3 flex items-center justify-center gap-2">
            <Button size="icon" onClick={() => decide("include")} disabled={!cur} className="h-9 w-11 bg-emerald-600 hover:bg-emerald-700" title="Include (i)"><Check className="size-4" /></Button>
            <Button size="icon" variant="destructive" onClick={() => decide("exclude")} disabled={!cur} className="h-9 w-11" title="Exclude (e)"><X className="size-4" /></Button>
            <Button size="icon" variant="outline" onClick={acceptAI} disabled={!cur} className="h-9 w-11" title="Accept the AI decision (a)"><Bot className="size-4" /></Button>
            <span className="mx-1 h-6 w-px bg-border" aria-hidden="true" />
            <Button size="icon" variant="ghost" onClick={next} disabled={!cur} className="h-9 w-11" title="Skip (s)"><SkipForward className="size-4" /></Button>
            <Button size="icon" variant="ghost" onClick={back} disabled={i === 0} className="h-9 w-11" title="Back (b)"><Undo2 className="size-4" /></Button>
          </div>
          <div className="pb-2.5 pt-2 text-center text-[11px] text-muted-foreground flex flex-wrap items-center justify-center gap-x-3 gap-y-0.5">
            <span><kbd className="px-1 rounded bg-muted">i</kbd> include</span>
            <span><kbd className="px-1 rounded bg-muted">e</kbd> exclude</span>
            <span><kbd className="px-1 rounded bg-muted">a</kbd> accept AI</span>
            <span><kbd className="px-1 rounded bg-muted">s</kbd> skip</span>
            <span><kbd className="px-1 rounded bg-muted">b</kbd> back</span>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
