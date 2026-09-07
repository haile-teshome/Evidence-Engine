import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useStore, HistoryEntry, SummaryRow, FullTextRecord } from "../lib/store";
import { Checkbox } from "../components/ui/checkbox";
import { AIService, DataAggregator, supportsTools } from "../lib/mockServices";
import type { ClarifyingQuestion, Paper } from "../lib/mockServices";
import { useStudyImport } from "../lib/useStudyImport";
import { AttachedStudies } from "../components/AttachedStudies";
import { Card } from "../components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../components/ui/dialog";
import { getPdfBlob, getPdfBlobMime, getDocHtml, docFilesReady, registerPdfBlobs } from "../lib/pdfBlobs";
import { importStudies, ACCEPTED_EXTS } from "../lib/pdfImport";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Textarea } from "../components/ui/textarea";
import { Separator } from "../components/ui/separator";
import { PicoCards } from "../components/PicoCards";
import { frameworkOf, FRAMEWORK_IDS, type FrameworkId } from "../lib/frameworks";
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel } from "../components/ui/dropdown-menu";
import { AnalysisProgress, Stage, StageId } from "../components/AnalysisProgress";
import { RefineSeeds } from "../components/RefineSeeds";
import { FormattedText } from "../lib/formattedText";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../components/ui/collapsible";
import { Sparkles, Send, ChevronDown, X, Plus, Wand2, Check, Lightbulb, Copy, Download, RotateCcw, Paperclip, Loader2, Files, Search, SlidersHorizontal, CalendarRange } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { toast } from "sonner";

// File-picker filter, derived from the importer's accepted extensions so the two
// never drift apart.
const ACCEPT_ATTR = ACCEPTED_EXTS.map(e => "." + e).join(",");

function ReferencesBySource({ refs, idPrefix }: { refs: { title: string; url: string; source: string; id: string }[]; idPrefix?: string }) {
  // The backend re-orders papers so that papers from the same source are
  // contiguous, with [N] citation markers in the summary matching this order.
  // We render the same sequence here under source headings, within each
  // heading the numbers stay continuous because the backend grouped them
  // before assigning [1]..[N].
  const numbered = refs.map((r, i) => ({ ...r, n: i + 1 }));
  const groups: { source: string; items: typeof numbered }[] = [];
  for (const r of numbered) {
    const key = (r.source || "Other").trim() || "Other";
    const existing = groups.find(g => g.source === key);
    if (existing) {
      existing.items.push(r);
    } else {
      groups.push({ source: key, items: [r] });
    }
  }
  return (
    <div className="space-y-3 text-sm">
      {groups.map(g => (
        <div key={g.source}>
          <div className="text-xs font-medium text-foreground/80 mb-1">{g.source}</div>
          <ol className="space-y-1">
            {g.items.map(r => (
              <li
                key={r.id || r.n}
                id={idPrefix ? `${idPrefix}-${r.n}` : undefined}
                className="flex gap-2 rounded px-1 -mx-1 scroll-mt-2 transition-colors duration-300"
              >
                <span className="text-muted-foreground tabular-nums shrink-0">[{r.n}]</span>
                <a
                  href={r.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline break-words flex-1 min-w-0"
                >
                  {r.title ? <FormattedText text={r.title} /> : r.url}
                </a>
              </li>
            ))}
          </ol>
        </div>
      ))}
    </div>
  );
}

function AutoTextarea({
  value,
  onChange,
  placeholder,
  className = "",
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  className?: string;
}) {
  const ref = useRef<HTMLTextAreaElement | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [value]);
  return (
    <textarea
      ref={ref}
      value={value}
      placeholder={placeholder}
      rows={1}
      onChange={e => onChange(e.target.value)}
      className={`flex w-full rounded-md border border-input bg-input-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring resize-none overflow-hidden leading-relaxed ${className}`}
    />
  );
}

function CriteriaList({
  items,
  onChange,
  placeholder,
  variant = "include",
}: {
  items: string[];
  onChange: (next: string[]) => void;
  placeholder: string;
  variant?: "include" | "exclude";
}) {
  const dotColor = variant === "include" ? "bg-primary" : "bg-destructive";
  return (
    <div className="space-y-2">
      {items.map((item, i) => (
        <div key={i} className="flex items-start gap-2 group">
          <span className={`size-1.5 rounded-full ${dotColor} mt-[14px] shrink-0`} />
          <div className="flex-1 min-w-0">
            <AutoTextarea
              value={item}
              placeholder={placeholder}
              onChange={v => {
                const next = [...items];
                next[i] = v;
                onChange(next);
              }}
            />
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity shrink-0 mt-1"
            onClick={() => onChange(items.filter((_, j) => j !== i))}
          >
            <X className="size-4" />
          </Button>
        </div>
      ))}
      <Button
        variant="outline"
        size="sm"
        onClick={() => onChange([...items, ""])}
        className="ml-3.5"
      >
        <Plus className="size-3 mr-1" />Add criterion
      </Button>
    </div>
  );
}

function QueryBlock({ label, value }: { label: string; value: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <div className="text-sm font-semibold text-foreground">{label}</div>
        <Button
          size="sm"
          variant="ghost"
          className="h-7 px-2 text-xs"
          onClick={() => {
            navigator.clipboard?.writeText(value);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }}
        >
          {copied ? <Check className="size-3.5 mr-1" /> : <Copy className="size-3.5 mr-1" />}
          {copied ? "Copied" : "Copy"}
        </Button>
      </div>
      <pre className="bg-muted rounded-md p-3 max-h-72 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-relaxed">{value}</pre>
    </div>
  );
}

// Turn inline citation markers like "[3]" or "[5, 7]" into clickable links that
// jump to the matching reference. Returns a mix of strings and link nodes.
function renderWithCitations(text: string, onCite?: (n: number) => void): React.ReactNode {
  if (!onCite) return text;
  const re = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  const out: React.ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const nums = m[1].split(/\s*,\s*/).map(x => parseInt(x, 10)).filter(n => !Number.isNaN(n));
    out.push(
      <span key={`c${key++}`} className="whitespace-nowrap">
        [{nums.map((n, i) => (
          <span key={i}>
            {i > 0 && ", "}
            <button
              type="button"
              onClick={() => onCite(n)}
              style={{ fontSize: "inherit" }}
              className="align-baseline text-primary hover:underline"
              title={`Go to reference ${n}`}
            >
              {n}
            </button>
          </span>
        ))}]
      </span>,
    );
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

// Move a citation that leads a line (e.g. "[4, 19]: claim" or "- [7]: claim") to
// the END of the claim ("claim [4, 19].") so citations follow the statement they
// support. Applied at render so summaries generated before the backend fix (and
// stored in history) still display correctly.
function moveLeadingCitation(line: string): string {
  const m = line.match(/^(\s*(?:[-*•]\s+)?)(\[\d+(?:\s*,\s*\d+)*\])\s*[:.–-]?\s+(.+)$/);
  if (!m) return line;
  const prefix = m[1];
  const cite = m[2];
  const rest = m[3].replace(/\s+$/, "");
  const last = rest.slice(-1);
  return ".!?;:".includes(last)
    ? `${prefix}${rest.slice(0, -1)} ${cite}${last}`
    : `${prefix}${rest} ${cite}`;
}

function SummaryText({ text, onCite }: { text: string; onCite?: (n: number) => void }) {
  // Split into sections on recognised headers, preserve bullets and paragraph breaks.
  const HEADERS = [
    "Research landscape overview",
    "Arguments supporting the research question",
    "Arguments against or challenging the research question",
  ];
  const lines = text.replace(/\r\n/g, "\n").split("\n").map(moveLeadingCitation);
  const sections: { heading: string | null; lines: string[] }[] = [{ heading: null, lines: [] }];
  for (const raw of lines) {
    const line = raw.trim();
    const isHeader = HEADERS.some(h => line.toLowerCase().startsWith(h.toLowerCase().slice(0, 12)));
    if (isHeader) {
      sections.push({ heading: line.replace(/:$/, ""), lines: [] });
    } else {
      sections[sections.length - 1].lines.push(raw);
    }
  }

  return (
    <div className="space-y-3 text-sm leading-relaxed text-foreground/90">
      {sections
        .filter(sec => sec.heading || sec.lines.some(l => l.trim()))
        .map((sec, idx) => {
          // Detect bullet block
          const bullets = sec.lines.filter(l => /^\s*[-*•]\s+/.test(l));
          const isBullets = bullets.length >= 2 && bullets.length >= sec.lines.filter(l => l.trim()).length * 0.6;
          return (
            <div key={idx}>
              {sec.heading && (
                <div className="text-xs uppercase tracking-wide text-muted-foreground font-medium mb-1">
                  {sec.heading}
                </div>
              )}
              {isBullets ? (
                <ul className="list-disc pl-5 space-y-1">
                  {bullets.map((b, i) => (
                    <li key={i}>{renderWithCitations(b.replace(/^\s*[-*•]\s+/, ""), onCite)}</li>
                  ))}
                </ul>
              ) : (
                sec.lines
                  .join("\n")
                  .split(/\n{2,}/)
                  .map((para, i) => (
                    <p key={i} className="mb-2 last:mb-0 whitespace-pre-wrap">
                      {renderWithCitations(para.trim(), onCite)}
                    </p>
                  ))
              )}
            </div>
          );
        })}
    </div>
  );
}

const INITIAL_STAGES: Stage[] = [
  // Four phases: frame the question (pico + formal question), build the search
  // (MeSH string + adversarial variant), retrieve (fetch + score), then scope
  // what's there (summary + refinements). PICO-derived steps are grouped with
  // framing/search rather than scattered through retrieval.
  { id: "pico", label: "Infer PICO elements", status: "pending" },
  { id: "question", label: "Draft formal research question", status: "pending" },
  { id: "query", label: "Generate MeSH search string", status: "pending" },
  { id: "adversarial", label: "Build adversarial search", status: "pending" },
  { id: "papers", label: "Fetch an initial sample of articles", status: "pending" },
  { id: "rerank", label: "Score articles for relevance", status: "pending" },
  { id: "summary", label: "Summarize the literature found", status: "pending" },
  { id: "suggestions", label: "Suggest refinements", status: "pending" },
];

// Default LEADS aggregate score threshold for the pre-summary relevance filter.
function scoreBadgeClass(score: number, threshold: number): string {
  if (score >= 0.5) return "bg-emerald-100 text-emerald-800";
  if (score >= threshold) return "bg-emerald-50 text-emerald-700";
  if (score >= threshold - 0.3) return "bg-amber-50 text-amber-700";
  return "bg-rose-50 text-rose-700";
}

// Compact −/＋ stepper with a typable field, used for per-database paper limits.
function NumberStepper({ value, onChange, min = 1, max = 2000, step = 5 }: {
  value: number; onChange: (n: number) => void; min?: number; max?: number; step?: number;
}) {
  const clamp = (n: number) => Math.max(min, Math.min(max, Number.isFinite(n) ? n : min));
  return (
    <div className="inline-flex items-center rounded-lg border bg-background overflow-hidden shrink-0">
      <button type="button" title={`−${step}`} onClick={() => onChange(clamp(value - step))}
        className="w-7 h-7 flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-muted transition-colors">
        <span className="text-sm leading-none">−</span>
      </button>
      <input type="number" min={min} max={max} value={value}
        onChange={e => onChange(clamp(parseInt(e.target.value || String(min), 10)))}
        className="w-12 h-7 bg-transparent text-center text-xs font-medium tabular-nums outline-none border-x [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none" />
      <button type="button" title={`+${step}`} onClick={() => onChange(clamp(value + step))}
        className="w-7 h-7 flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-muted transition-colors">
        <span className="text-sm leading-none">+</span>
      </button>
    </div>
  );
}

// Conversational PICO clarifier modal. Fetches one question at a time from the
// backend until all PICO elements are SR-ready, then calls onDone. Each question
// has exactly 3 specific suggestions + 1 blank fill-in.
function ClarifyingQuestionsModal({
  open,
  goal,
  framework,
  onDone,
  onSkipAll,
}: {
  open: boolean;
  goal: string;
  framework: FrameworkId;
  onDone: (answers: Record<string, string>) => void;
  onSkipAll: () => void;
}) {
  // Navigation over the asked questions so the reviewer can jump back to a
  // previous element (via the header pills) and revise its answer.
  const [nav, setNav] = useState<{ asked: ClarifyingQuestion[]; pos: number }>({ asked: [], pos: 0 });
  const [answers, setAnswers] = useState<Record<string, string>>({});
  // Selected option chips per element id, so revisiting a question restores them.
  const [partsByQ, setPartsByQ] = useState<Record<string, string[]>>({});
  const [round, setRound] = useState(0);
  const [loading, setLoading] = useState(false);
  const [freeText, setFreeText] = useState("");
  // Multiple options can be selected per PICO element (e.g. several eligible
  // populations or outcomes); they're combined into one criterion on confirm.
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const freeRef = useRef<HTMLInputElement>(null);
  // PICO element ids already asked, at most one question per element, no repeats.
  const askedRef = useRef<Set<string>>(new Set());
  const question = nav.asked[nav.pos] ?? null;

  const fetchNext = useCallback(async (current: Record<string, string>, r: number) => {
    setLoading(true);
    setFreeText("");
    setSelected(new Set());
    try {
      const result = await AIService.getClarifyNext(goal, current, r, Array.from(askedRef.current), framework);
      // Stop if done, no question, or the model circled back to an element we
      // already asked about.
      if (result.done || !result.question || askedRef.current.has(result.question.id)) {
        onDone(current);
      } else {
        askedRef.current.add(result.question.id);
        // Append to the trail and move to it, so earlier questions stay
        // reachable via the header pills.
        setNav(prev => ({ asked: [...prev.asked, result.question!], pos: prev.asked.length }));
      }
    } catch {
      onDone(current);
    } finally {
      setLoading(false);
    }
  }, [goal, framework, onDone]);

  useEffect(() => {
    if (open && goal) {
      setAnswers({});
      setNav({ asked: [], pos: 0 });
      setPartsByQ({});
      setRound(0);
      setFreeText("");
      askedRef.current = new Set();
      fetchNext({}, 0);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, goal]);

  // Restore the option chips previously chosen for a question we're returning to.
  function restoreSelection(qid: string) {
    setSelected(new Set(partsByQ[qid] || []));
    setFreeText("");
  }

  // Jump to an already-asked question (from a header pill) to revise its answer.
  function goTo(i: number) {
    if (loading || i < 0 || i >= nav.asked.length || i === nav.pos) return;
    restoreSelection(nav.asked[i].id);
    setNav(prev => ({ ...prev, pos: i }));
  }

  function pick(value: string) {
    if (!question || loading) return;
    const qid = question.id;
    const next = { ...answers, [qid]: value };
    setAnswers(next);
    if (value === "") {
      // Skipped: drop any remembered chips for this element.
      setPartsByQ(prev => { const n = { ...prev }; delete n[qid]; return n; });
    }
    // If a later question was already asked, step forward to it (revising an
    // earlier answer); otherwise ask the next one.
    if (nav.pos < nav.asked.length - 1) {
      const nextPos = nav.pos + 1;
      restoreSelection(nav.asked[nextPos].id);
      setNav(prev => ({ ...prev, pos: nextPos }));
    } else {
      const nextRound = round + 1;
      setRound(nextRound);
      fetchNext(next, nextRound);
    }
  }

  function toggle(label: string) {
    setSelected(prev => {
      const n = new Set(prev);
      if (n.has(label)) n.delete(label); else n.add(label);
      return n;
    });
  }

  // Add whatever is typed in "Other" as a selected option (chip), so custom
  // values sit alongside the suggested ones and several can be added.
  function addCustom() {
    const v = freeText.trim();
    if (!v) return;
    setSelected(prev => new Set(prev).add(v));
    setFreeText("");
    freeRef.current?.focus();
  }

  // Combine every selected option (plus any not-yet-added free-text) into one
  // criterion for this PICO element, then advance.
  function confirm() {
    if (!question || loading) return;
    const ft = freeText.trim();
    const parts = [...new Set(ft ? [...selected, ft] : [...selected])];
    if (parts.length === 0) return;
    setPartsByQ(prev => ({ ...prev, [question.id]: parts }));
    pick(parts.join("; "));
  }

  if (!open) return null;

  const showSpinner = loading || !question;

  return (
    <div className="fixed bottom-32 left-72 right-0 z-40 px-6 pointer-events-none">
      <div className="max-w-4xl mx-auto pointer-events-auto">
        <Card className="border-primary/40 shadow-xl bg-card/98 backdrop-blur overflow-hidden">

          {/* Header */}
          <div className="flex items-center justify-between gap-3 px-4 pt-3 pb-2">
            <div className="flex items-center gap-2 min-w-0">
              <Lightbulb className="size-4 text-primary shrink-0" />
              <div className="text-sm font-medium break-words">
                {showSpinner ? `Checking your ${frameworkOf(framework).label} elements…` : question!.title}
              </div>
            </div>
            {/* Frame element progress pills — click an asked one to revisit it */}
            <div className="flex items-center gap-1.5 shrink-0">
              {frameworkOf(framework).elements.map(el => {
                const f = el.id;
                const done = !!answers[f];
                const active = question?.id === f;
                const askedIdx = nav.asked.findIndex(qq => qq.id === f);
                const style = active
                  ? "bg-primary text-primary-foreground border-primary"
                  : done
                  ? "bg-emerald-100 text-emerald-700 border-emerald-300"
                  : "bg-muted text-muted-foreground border-border";
                const base = `text-[10px] font-bold px-1.5 py-0.5 rounded-sm border transition-colors ${style}`;
                if (askedIdx >= 0 && !active) {
                  return (
                    <button
                      key={f}
                      onClick={() => goTo(askedIdx)}
                      disabled={loading}
                      className={`${base} cursor-pointer hover:brightness-95 disabled:opacity-60`}
                      title={`Go back to: ${el.label}`}
                    >
                      {el.letter}
                    </button>
                  );
                }
                return (
                  <span key={f} className={base} title={el.label}>
                    {el.letter}
                  </span>
                );
              })}
              <button
                onClick={onSkipAll}
                className="ml-1 text-muted-foreground hover:text-foreground"
                aria-label="Skip all"
              >
                <X className="size-4" />
              </button>
            </div>
          </div>

          {/* Body */}
          {showSpinner ? (
            <div className="px-4 pb-4 flex items-center gap-2 text-sm text-muted-foreground">
              <span className="inline-block size-3.5 rounded-full border-2 border-primary border-t-transparent animate-spin shrink-0" />
              Analysing…
            </div>
          ) : (
            <div className="px-4 pb-2 space-y-1.5">
              <div className="text-[11px] text-muted-foreground px-1 pb-0.5">Select one or more, then Confirm.</div>
              {question!.options.slice(0, 3).map((opt, i) => {
                const on = selected.has(opt.label);
                return (
                  <button
                    key={opt.id}
                    onClick={() => toggle(opt.label)}
                    aria-pressed={on}
                    className={`w-full text-left px-3 py-2.5 rounded-md border transition-colors flex items-center gap-3 ${
                      on ? "bg-primary/10 border-primary" : "bg-card hover:bg-accent hover:border-primary/40"
                    }`}
                  >
                    <span className={`shrink-0 size-5 rounded-sm border flex items-center justify-center ${
                      on ? "bg-primary border-primary text-primary-foreground" : "bg-muted text-muted-foreground"
                    }`}>
                      {on ? <Check className="size-3.5" /> : <span className="text-[11px] tabular-nums font-mono">{i + 1}</span>}
                    </span>
                    <span className="text-sm flex-1 leading-snug">{opt.label}</span>
                  </button>
                );
              })}

              {/* Custom values added via "Other", shown as selected chips. */}
              {[...selected].filter(l => !question!.options.slice(0, 3).some(o => o.label === l)).map(label => (
                <button
                  key={label}
                  onClick={() => toggle(label)}
                  aria-pressed
                  className="w-full text-left px-3 py-2.5 rounded-md border bg-primary/10 border-primary transition-colors flex items-center gap-3"
                >
                  <span className="shrink-0 size-5 rounded-sm border bg-primary border-primary text-primary-foreground flex items-center justify-center">
                    <Check className="size-3.5" />
                  </span>
                  <span className="text-sm flex-1 leading-snug">{label}</span>
                  <X className="size-3.5 text-muted-foreground shrink-0" />
                </button>
              ))}

              {/* Blank fill-in: adds the typed value as a selectable option. */}
              <div className="flex items-center gap-2 mt-1 px-3 py-1.5 rounded-md border border-dashed bg-muted/20">
                <Wand2 className="size-3.5 text-muted-foreground shrink-0" />
                <Input
                  ref={freeRef}
                  value={freeText}
                  onChange={e => setFreeText(e.target.value)}
                  placeholder="Other: add your own…"
                  className="border-0 bg-transparent shadow-none focus-visible:ring-0 px-0 h-7 text-sm"
                  onKeyDown={e => { if (e.key === "Enter") { e.preventDefault(); addCustom(); } }}
                />
                {freeText.trim() && (
                  <Button size="sm" variant="outline" onClick={addCustom} className="rounded-full h-7 px-3 shrink-0">
                    <Plus className="size-3.5 mr-1" />Add
                  </Button>
                )}
              </div>

              {/* Confirm the combined selection for this PICO element */}
              <div className="flex items-center justify-between pt-1">
                <span className="text-[11px] text-muted-foreground">
                  {selected.size + (freeText.trim() ? 1 : 0)} selected
                </span>
                <Button
                  size="sm"
                  onClick={confirm}
                  disabled={selected.size === 0 && !freeText.trim()}
                  className="rounded-full h-7 px-4"
                >
                  Confirm
                </Button>
              </div>
            </div>
          )}

          {/* Footer */}
          <div className="flex items-center justify-end px-4 py-2 border-t bg-muted/30 gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => question && pick("")}
              disabled={showSpinner}
              className="h-7 text-muted-foreground"
            >
              Skip this
            </Button>
            <Button variant="ghost" size="sm" onClick={onSkipAll} className="h-7 text-muted-foreground">
              Skip all
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
}

function OverviewTab({ entry, idx }: { entry: HistoryEntry; idx: number }) {
  const [refsOpen, setRefsOpen] = useState(false);
  const refPrefix = `ref-${idx}`;
  const hasRefs = !!(entry.references && entry.references.length > 0);

  // Clicking a [n] citation opens the references panel and scrolls to ref n.
  const scrollToRef = (n: number) => {
    setRefsOpen(true);
    setTimeout(() => {
      const el = document.getElementById(`${refPrefix}-${n}`);
      if (!el) return;
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      el.classList.add("bg-primary/10");
      setTimeout(() => el.classList.remove("bg-primary/10"), 1600);
    }, 70);
  };

  return (
    <div className="space-y-4">
      <section>
        <div className="flex items-center gap-2 text-sm font-semibold text-foreground mb-1.5">
          <Sparkles className="size-4 text-primary" />Research Question
        </div>
        <p className="leading-snug italic border-l-2 border-primary/40 pl-3">{entry.formal_question}</p>
      </section>
      {entry.summary && (
        <>
          <Separator />
          <section>
            <div className="text-sm font-semibold text-foreground mb-2">Summary</div>
            <div className="rounded-md border bg-muted/20 p-3">
              <SummaryText text={entry.summary} onCite={hasRefs ? scrollToRef : undefined} />
            </div>
          </section>
        </>
      )}
      {hasRefs && (
        <>
          <Separator />
          <Collapsible open={refsOpen} onOpenChange={setRefsOpen}>
            <CollapsibleTrigger asChild>
              <button className="group flex items-center gap-1.5 text-sm font-semibold text-foreground hover:text-primary">
                <ChevronDown className="size-4 transition-transform group-data-[state=open]:rotate-180" />
                References ({entry.references!.length})
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-2">
              <div className="max-h-72 overflow-auto rounded-md border bg-muted/20 p-3">
                <ReferencesBySource refs={entry.references!} idPrefix={refPrefix} />
              </div>
            </CollapsibleContent>
          </Collapsible>
        </>
      )}
    </div>
  );
}

function RankRow({ d, threshold, effective, overridden, onSetOverride }: {
  d: any; threshold: number; effective: "keep" | "drop"; overridden: boolean;
  onSetOverride: (id: string, v: "keep" | "drop" | null) => void;
}) {
  const score = d.leads_score as number;
  const id = String((d.paper as any)?.id ?? "");
  return (
    <li className="flex items-start gap-2.5 text-sm py-1.5 px-2 rounded-md hover:bg-muted/50">
      <span
        className={`shrink-0 rounded-md px-2 py-0.5 text-xs font-mono font-semibold tabular-nums ${scoreBadgeClass(score, threshold)}`}
        title={d.reason}
      >
        {score >= 0 ? "+" : ""}{score.toFixed(2)}
      </span>
      <div className="min-w-0 flex-1">
        <a href={d.paper?.url || "#"} target="_blank" rel="noreferrer" className="hover:underline break-words leading-snug">
          {d.paper?.title || "(untitled)"}
        </a>
        {d.paper?.source && <span className="text-xs text-muted-foreground ml-2">[{d.paper.source}]</span>}
        {overridden && (
          <span className="ml-2 inline-flex items-center gap-1 align-middle text-[10px] font-medium text-amber-600">
            manual
            <button type="button" onClick={() => onSetOverride(id, null)} title="Reset to automatic decision" className="hover:text-foreground">
              <RotateCcw className="size-3" />
            </button>
          </span>
        )}
      </div>
      <button
        type="button"
        onClick={() => onSetOverride(id, effective === "keep" ? "drop" : "keep")}
        title={effective === "keep" ? "Exclude from the relevant set" : "Include in the relevant set"}
        className={`shrink-0 inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-medium transition-colors ${
          effective === "keep"
            ? "border-rose-200 text-rose-600 hover:bg-rose-50"
            : "border-emerald-200 text-emerald-700 hover:bg-emerald-50"
        }`}
      >
        {effective === "keep" ? <><X className="size-3" />Exclude</> : <><Check className="size-3" />Include</>}
      </button>
    </li>
  );
}

function RelevanceExplorer() {
  const s = useStore();
  const r = s.rerankResults;
  const [regen, setRegen] = useState(false);
  if (!r) return null;
  // Use the auto-cutoff the rerank actually applied (effective_floor), not the
  // legacy store threshold. The slider that used to drive it has been removed.
  const threshold = typeof r.effective_floor === "number" ? r.effective_floor : r.threshold;
  const ov = s.relevanceOverrides;
  const autoOf = (d: any): "keep" | "drop" => (d.leads_score >= threshold ? "keep" : "drop");
  const effOf = (d: any): "keep" | "drop" => ov[String((d.paper as any)?.id ?? "")] ?? autoOf(d);
  const kept = r.ranked.filter(d => effOf(d) === "keep");
  const dropped = r.ranked.filter(d => effOf(d) === "drop");
  const nOverrides = r.ranked.filter(d => {
    const id = String((d.paper as any)?.id ?? "");
    return ov[id] && ov[id] !== autoOf(d);
  }).length;
  const fmt = (n: number) => `${n >= 0 ? "+" : ""}${n.toFixed(2)}`;

  const setOverride = (id: string, v: "keep" | "drop" | null) => {
    // Keep the map minimal: only store a value that differs from the automatic
    // decision, so toggling back to the auto side clears the override.
    const item = r.ranked.find(d => String((d.paper as any)?.id ?? "") === id);
    const auto = item ? autoOf(item) : null;
    s.setRelevanceOverrides(prev => {
      const next = { ...prev };
      if (v === null || (auto && v === auto)) delete next[id]; else next[id] = v;
      return next;
    });
  };

  // Rebuild the latest run's summary + references from the current kept set,
  // so manual include/exclude choices are reflected in what the summary reads.
  async function regenerate() {
    const idx = s.history.length - 1;
    const entry = s.history[idx];
    if (!entry) return;
    setRegen(true);
    try {
      const papersForSummary = kept.map((d: any) => ({
        id: String(d.paper?.id ?? ""), source: d.paper?.source || "",
        title: d.paper?.title || "", abstract: d.paper?.abstract || "", url: d.paper?.url || "",
      }));
      const res = await AIService.generateComprehensiveSummaryWithRefs(entry.goal, papersForSummary as any);
      s.setHistory(h => h.map((e, i) => i === idx ? { ...e, summary: res.summary || e.summary, references: res.references || e.references } : e));
      toast.success(`Summary updated from ${kept.length} kept ${kept.length === 1 ? "study" : "studies"}`);
    } catch {
      toast.error("Could not update the summary");
    } finally {
      setRegen(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        <span className="inline-flex items-center gap-1 rounded-full border border-emerald-200 bg-emerald-50 text-emerald-700 px-2 py-0.5 font-medium">
          <Check className="size-3" />{kept.length} kept
        </span>
        <span className="inline-flex items-center gap-1 rounded-full border border-rose-200 bg-rose-50 text-rose-700 px-2 py-0.5 font-medium">
          <X className="size-3" />{dropped.length} dropped
        </span>
        <span className="inline-flex items-center rounded-full border bg-muted text-muted-foreground px-2 py-0.5 font-medium tabular-nums">
          threshold {fmt(threshold)}
        </span>
        {nOverrides > 0 && (
          <>
            <span className="inline-flex items-center gap-1 rounded-full border border-amber-200 bg-amber-50 text-amber-700 px-2 py-0.5 font-medium">
              {nOverrides} manual override{nOverrides === 1 ? "" : "s"}
            </span>
            <div className="ml-auto flex items-center gap-2">
              <button type="button" onClick={() => s.setRelevanceOverrides({})} className="text-muted-foreground hover:text-foreground">
                Reset all
              </button>
              <Button size="sm" className="h-7 gap-1.5" onClick={regenerate} disabled={regen}>
                {regen ? <Loader2 className="size-3.5 animate-spin" /> : <Sparkles className="size-3.5" />}Update summary
              </Button>
            </div>
          </>
        )}
      </div>
      <p className="text-[11px] text-muted-foreground -mt-2">
        Use Include / Exclude on any study to override the automatic cutoff, then Update summary to apply your changes.
      </p>

      {kept.length > 0 && (
        <section>
          <div className="text-sm font-semibold text-foreground mb-1.5">
            Kept (included in the summary)
          </div>
          <div className="rounded-md border bg-muted/20 max-h-72 overflow-auto p-1">
            <ul className="space-y-0.5">
              {kept.map((d, i) => {
                const id = String((d.paper as any)?.id ?? i);
                return <RankRow key={id} d={d} threshold={threshold} effective="keep" overridden={ov[id] === "keep" && autoOf(d) === "drop"} onSetOverride={setOverride} />;
              })}
            </ul>
          </div>
        </section>
      )}

      <Separator />

      {dropped.length === 0 ? (
        <p className="text-sm text-muted-foreground">Nothing is excluded.</p>
      ) : (
        <section>
          <div className="text-sm font-semibold text-foreground mb-1.5">
            Dropped (excluded)
          </div>
          <div className="rounded-md border bg-muted/20 max-h-72 overflow-auto p-1">
            <ul className="space-y-0.5">
              {dropped.map((d, i) => {
                const id = String((d.paper as any)?.id ?? i);
                return <RankRow key={id} d={d} threshold={threshold} effective="drop" overridden={ov[id] === "drop" && autoOf(d) === "keep"} onSetOverride={setOverride} />;
              })}
            </ul>
          </div>
        </section>
      )}
    </div>
  );
}

// Inline Markdown: **bold** as bold, and [n] citations as clickable buttons that
// open a preview of source n.
function renderInline(text: string, onCite: (n: number) => void, keyBase: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /\*\*([^*]+)\*\*|\[(\d+)\]/g;
  let last = 0;
  let k = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1] !== undefined) {
      out.push(<strong key={`${keyBase}-${k++}`} className="font-semibold text-foreground">{m[1]}</strong>);
    } else {
      const n = Number(m[2]);
      out.push(
        <button key={`${keyBase}-${k++}`} type="button" onClick={() => onCite(n)}
          style={{ fontSize: "inherit" }}
          className="align-baseline text-primary font-medium hover:underline">[{n}]</button>,
      );
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

// Render a document-Q&A answer as light Markdown: headings, bullet / numbered
// lists, and paragraphs, with inline **bold** and clickable [n] citations.
function renderAnswer(text: string, onCite: (n: number) => void): ReactNode[] {
  const lines = (text || "").replace(/\r/g, "").split("\n");
  const blocks: ReactNode[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  let bi = 0;

  const flushList = () => {
    if (!list) return;
    const items = list.items;
    const key = `l${bi++}`;
    if (list.ordered) {
      blocks.push(
        <ol key={key} className="list-decimal pl-5 space-y-1 my-1.5">
          {items.map((it, i) => <li key={i}>{renderInline(it, onCite, `${key}-${i}`)}</li>)}
        </ol>,
      );
    } else {
      blocks.push(
        <ul key={key} className="list-disc pl-5 space-y-1 my-1.5">
          {items.map((it, i) => <li key={i}>{renderInline(it, onCite, `${key}-${i}`)}</li>)}
        </ul>,
      );
    }
    list = null;
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) { flushList(); continue; }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushList();
      blocks.push(
        <p key={`h${bi++}`} className="font-semibold text-foreground mt-2 first:mt-0">
          {renderInline(heading[2], onCite, `h${bi}`)}
        </p>,
      );
      continue;
    }

    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    if (bullet) {
      if (!list || list.ordered) { flushList(); list = { ordered: false, items: [] }; }
      list.items.push(bullet[1]);
      continue;
    }

    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (numbered) {
      if (!list || !list.ordered) { flushList(); list = { ordered: true, items: [] }; }
      list.items.push(numbered[1]);
      continue;
    }

    flushList();
    blocks.push(
      <p key={`p${bi++}`} className="leading-relaxed">{renderInline(line, onCite, `p${bi}`)}</p>,
    );
  }
  flushList();
  return blocks;
}

// Rank documents by lexical overlap with the question so the most relevant sources
// lead the context we hand the doc-QA model, instead of smearing the whole library
// thin. Cheap and deterministic; a stable sort keeps original order among equally-
// (or zero-) scoring docs, so a low-signal message just falls back to corpus order.
const RANK_STOP = new Set("the a an and or of to in on for with is are was were be been being this that these those it as by at from about into over under what which who whom whose how why when where does do did can could will would should you your yours i me my we our us they them their he she his her not no yes".split(" "));
function rankByRelevance<T extends { title?: string; text?: string }>(question: string, docs: T[]): T[] {
  const terms = Array.from(new Set(
    (question || "").toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(w => w.length >= 3 && !RANK_STOP.has(w)),
  ));
  if (terms.length === 0) return docs;
  return docs
    .map((d, i) => {
      const title = (d.title || "").toLowerCase();
      const text = (d.text || "").toLowerCase();
      let score = 0;
      for (const t of terms) { if (title.includes(t)) score += 3; if (text.includes(t)) score += 1; }
      return { d, score, i };
    })
    .sort((a, b) => b.score - a.score || a.i - b.i)
    .map(x => x.d);
}

export function HomePage() {
  const s = useStore();
  const studyImport = useStudyImport();
  // Installed local (Ollama) models, so we can fall back to a tool-capable one
  // (qwen2.5) when the selected model can't drive the chat agent.
  const [localModels, setLocalModels] = useState<string[]>([]);
  useEffect(() => {
    fetch("/api/models/local").then(r => r.json()).then(d => setLocalModels(Array.isArray(d.models) ? d.models : [])).catch(() => {});
  }, []);
  const attachRef = useRef<HTMLInputElement>(null);
  const [attachOpen, setAttachOpen] = useState(false);
  const [input, setInput] = useState("");
  // Free-form / custom requests are structured by the LLM (never used verbatim).
  // Shared spinner, the inferred research question, and the draft description for
  // the Study-design "Infer with AI" boxes.
  const [inferring, setInferring] = useState(false);
  const [inferredQuestion, setInferredQuestion] = useState("");
  const [picoDraft, setPicoDraft] = useState("");
  // Pre-search review gate: before any database is queried, surface the full
  // generated plan (question, PICO, criteria, search string, databases) so the
  // reviewer can edit it and Run, or Cancel. Resolves with a fresh store snapshot
  // (reads inside the still-running handleSubmit are stale in this Context store).
  type PlanDecision = { pico: typeof s.pico; inclusion: string[]; exclusion: string[]; query: string; sources: string[]; perSourceLimits: Record<string, number>; numPerSource: number };
  const [planGate, setPlanGate] = useState<null | { question: string }>(null);
  const planResolverRef = useRef<((d: PlanDecision | null) => void) | null>(null);
  // Grounded query builder (Phase 1): MeSH-grounded, comprehensive search built
  // from the current PICO concepts, with the per-concept breakdown shown.
  const [building, setBuilding] = useState(false);
  const [builtConcepts, setBuiltConcepts] = useState<{ name: string; tiab: string[]; mesh: string[] }[]>([]);
  // Seeds for query refinement: the reviewer's uploaded studies plus any records
  // marked relevant (relevance feedback). Fed to the builder to broaden concepts.
  function seedStudies() {
    return (s.rawPapers || [])
      .filter(p => p.source === "Local PDFs" || s.seedIds.has(p.id))
      .slice(0, 20)
      .map(p => ({ id: String(p.id), title: p.title || "", abstract: p.abstract || "", source: p.source }));
  }
  // Build the comprehensive, MeSH-grounded search from the current PICO concepts
  // (and any seed studies). This is the default query builder — it runs as part of
  // the pipeline before the review gate, so the string shown is always grounded.
  // `pico`/`question` can be passed explicitly because store reads inside the
  // still-running handleSubmit closure are stale.
  async function runGroundedBuild(opts?: { pico?: typeof s.pico; question?: string; signal?: AbortSignal }) {
    setBuilding(true);
    try {
      const seeds = seedStudies();
      const pico = opts?.pico ?? s.pico;
      const question = opts?.question ?? planGate?.question ?? input ?? "";
      const r = await AIService.buildSearch(pico, question, seeds, opts?.signal);
      if (r.query) { s.setQuery(r.query); s.setUnifiedSearchQuery(r.query); }
      setBuiltConcepts(r.concepts || []);
      if (seeds.length) toast.success(`Search broadened with ${seeds.length} known ${seeds.length === 1 ? "study" : "studies"}`);
      return r;
    } catch {
      return null;
    } finally {
      setBuilding(false);
    }
  }
  // Conversational Q&A over the documents in play (retrieved / uploaded / cited
  // in explanations). Answered by the same main chat, shown as conversation turns.
  // Persisted in the store so they survive a refresh, like the search history.
  const qaTurns = s.docQa;
  const setQaTurns = s.setDocQa;
  const [previewDoc, setPreviewDoc] = useState<{ id: string; title: string; text: string; url?: string; fileUrl?: string; fileMime?: string; fileInline?: boolean; html?: string; uploaded?: boolean } | null>(null);
  const [previewMode, setPreviewMode] = useState<"doc" | "text">("doc");
  const reattachRef = useRef<HTMLInputElement>(null);
  const reattachId = useRef<string | null>(null);
  const docCorpus = useMemo(() => {
    const seen = new Set<string>();
    const out: { id: string; title: string; text: string; source: string; url?: string }[] = [];
    for (const p of (s.rawPapers || [])) {
      if (!p.id || seen.has(p.id)) continue;
      const text = (s.fullTexts[p.id]?.text || p.abstract || "").trim();
      if (!text) continue;
      seen.add(p.id);
      out.push({ id: p.id, title: p.title || "Untitled", text, source: p.source, url: p.url });
    }
    return out;
  }, [s.rawPapers, s.fullTexts]);
  const uploadedDocs = docCorpus.filter(d => d.source === "Local PDFs");
  // For the Relevance tab, list uploads straight from rawPapers so they appear the
  // instant they're attached, before full-text extraction finishes (docCorpus drops
  // text-less docs, which hid freshly uploaded PDFs).
  const uploadedRaw = (s.rawPapers || []).filter(p => p.source === "Local PDFs");

  // Structured-summary source selection: relevant pulled articles + uploaded docs.
  // Relevance-kept articles seed once; uploaded documents auto-select in real time
  // as they are attached, so the Relevance tab reflects new uploads immediately.
  const [summarySel, setSummarySel] = useState<Set<string>>(new Set());
  const [summarizing, setSummarizing] = useState(false);
  const [relSearch, setRelSearch] = useState("");
  const seededRerank = useRef(false);
  const knownSummaryDocIds = useRef<Set<string> | null>(null);
  useEffect(() => {
    const toAdd = new Set<string>();
    if (!seededRerank.current && (s.rerankResults?.kept?.length ?? 0) > 0) {
      seededRerank.current = true;
      for (const r of s.rerankResults!.kept) toAdd.add(r.paper.id);
    }
    const ids = uploadedRaw.map(p => p.id);
    if (knownSummaryDocIds.current === null) {
      for (const id of ids) toAdd.add(id);                               // existing uploads on first run
    } else {
      for (const id of ids) if (!knownSummaryDocIds.current.has(id)) toAdd.add(id);  // newly attached
    }
    knownSummaryDocIds.current = new Set(ids);
    if (toAdd.size) setSummarySel(prev => new Set([...prev, ...toAdd]));
  }, [s.rerankResults, uploadedRaw]);
  const selectedSources = summarySel;
  function toggleSource(id: string) {
    setSummarySel(prev => { const n = new Set(prev); if (n.has(id)) n.delete(id); else n.add(id); return n; });
  }

  // Freshly attached documents become the focus of the NEXT message, so "summarize
  // this" acts on what the user just attached rather than the whole corpus. We
  // detect newly-added uploads by diffing their ids; documents restored from a
  // saved session on mount are NOT treated as freshly attached.
  const [pendingDocIds, setPendingDocIds] = useState<string[]>([]);
  const knownUploadIds = useRef<Set<string> | null>(null);
  useEffect(() => {
    const cur = new Set(uploadedDocs.map(d => d.id));
    if (knownUploadIds.current === null) { knownUploadIds.current = cur; return; }  // baseline on first run
    const added = [...cur].filter(id => !knownUploadIds.current!.has(id));
    knownUploadIds.current = cur;
    if (added.length) setPendingDocIds(prev => Array.from(new Set([...prev, ...added])));
  }, [uploadedDocs]);
  // Keep only ids that still exist, resolved to their documents.
  const pendingDocs = useMemo(
    () => pendingDocIds.map(id => docCorpus.find(d => d.id === id)).filter(Boolean) as typeof docCorpus,
    [pendingDocIds, docCorpus],
  );

  const openDocPreview = async (id: string) => {
    const d = docCorpus.find(x => x.id === id);
    if (!d) return;
    await docFilesReady;   // ensure persisted previews are rehydrated after a reload
    const remotePdf = s.fullTexts[id]?.pdf_url;
    const fileUrl = getPdfBlob(id) || remotePdf || undefined;
    const fileMime = getPdfBlobMime(id) || (remotePdf ? "application/pdf" : undefined);
    const html = getDocHtml(id);
    // We can show the actual document when: it's a PDF or plain text (iframe), or
    // it's a Word doc we've rendered to HTML at import. Otherwise fall back to the
    // extracted text, with the original offered as a download.
    const fileInline = !!html || (!!fileUrl && (fileMime === "application/pdf" || (fileMime || "").startsWith("text/")));
    setPreviewMode(fileInline ? "doc" : "text");
    setPreviewDoc({ id, title: d.title, text: d.text, url: d.url, fileUrl, fileMime, fileInline, html, uploaded: d.source === "Local PDFs" });
  };

  // Re-attach an uploaded document's original file so its actual-document preview
  // works. Needed for documents added before the file was captured/persisted; the
  // picked file is bound to the SAME paper id the app already references.
  const promptReattach = (id: string) => { reattachId.current = id; reattachRef.current?.click(); };
  const onReattachPicked = async (file: File | undefined) => {
    const id = reattachId.current; reattachId.current = null;
    if (reattachRef.current) reattachRef.current.value = "";
    if (!file || !id) return;
    try {
      const { studies } = await importStudies([file]);
      const st = studies[0];
      if (!st || (!st.objectBlob && !st.html)) { toast.error("Could not read that file as a document."); return; }
      registerPdfBlobs([{ id, blob: st.objectBlob, mime: st.objectMime, html: st.html }]);
      if (st.fullText) s.setFullTexts(prev => ({ ...prev, [id]: { ...st.fullText!, paper_id: id } }));
      await openDocPreview(id);
      toast.success("Original document attached");
    } catch (e: any) {
      toast.error(e?.message || "Could not attach that file.");
    }
  };

  // A general reply for messages that need neither a search nor the documents
  // (greetings, "what can you do?", how-to). Appended as a chat turn, no sources.
  async function runChat(message: string) {
    const qq = message.trim();
    if (!qq) return;
    setInput("");
    const idx = qaTurns.length;
    setQaTurns(prev => [...prev, { question: qq, answer: "", sources: [], busy: true, ts: Date.now() }]);
    const history = qaTurns.flatMap(t => [
      { role: "user", content: t.question },
      { role: "assistant", content: t.answer },
    ]);
    try {
      const answer = await AIService.chat(qq, history);
      setQaTurns(prev => prev.map((x, i) => i === idx ? { ...x, answer: answer || "Sorry, I couldn't respond just now.", sources: [], busy: false } : x));
    } catch (e: any) {
      setQaTurns(prev => prev.map((x, i) => i === idx ? { ...x, answer: e?.message || "Sorry, that failed. Please try again.", busy: false } : x));
    }
  }

  // Build a per-source structured evidence table over the selected pulled
  // articles + uploaded documents, appended to the chat thread like a search.
  async function runStructuredSummary() {
    const rankedById = new Map((s.rerankResults?.ranked || []).map(r => [r.paper.id, r.paper]));
    const raw = [
      ...[...selectedSources].map(id => rankedById.get(id)).filter(Boolean).map((p: any) => ({ id: p.id, title: p.title, abstract: p.abstract || "", full_text: s.fullTexts[p.id]?.text || "" })),
      ...uploadedRaw.filter(p => selectedSources.has(p.id)).map(p => ({ id: p.id, title: p.title, abstract: p.abstract || "", full_text: s.fullTexts[p.id]?.text || "" })),
    ];
    const seen = new Set<string>();
    const uniq = raw.filter(x => seen.has(x.id) ? false : (seen.add(x.id), true));
    if (!uniq.length) { toast.error("Select at least one relevant article or uploaded document."); return; }
    setSummarizing(true);
    setReviewOpen(false);
    const idx = qaTurns.length;
    setQaTurns(prev => [...prev, { question: `Structured summary of ${uniq.length} source${uniq.length === 1 ? "" : "s"}`, answer: "", sources: [], busy: true, ts: Date.now() }]);
    try {
      const rows = await AIService.structuredSummary(uniq);
      const titleById = new Map(uniq.map(x => [x.id, x.title]));
      const table: SummaryRow[] = rows.map(r => ({ id: r.id, title: titleById.get(r.id) || r.id, design: r.design, population: r.population, intervention: r.intervention, comparator: r.comparator, outcomes: r.outcomes, key_finding: r.key_finding }));
      setQaTurns(prev => prev.map((x, j) => j === idx ? { ...x, table, busy: false } : x));
    } catch (e: any) {
      setQaTurns(prev => prev.map((x, j) => j === idx ? { ...x, answer: e?.message || "Structured summary failed.", busy: false } : x));
    } finally { setSummarizing(false); }
  }

  // Home-chat agent: send a light view of the library and let a tool-capable model
  // search it and read full text on demand. `note` explains a model switch, if any.
  async function answerWithAgent(question: string, model: string, note: string) {
    const qq = question.trim();
    if (!qq) return;
    setInput("");
    const raw = s.rawPapers || [];
    const library = raw.slice(0, 120).map(p => {
      const ft = s.fullTexts[p.id]?.text || "";
      return {
        id: p.id,
        title: p.title || "Untitled",
        snippet: (ft || p.abstract || "").slice(0, 1200),
        url: p.url || "",
        source: p.source || "",
        has_full_text: !!ft,
      };
    });
    // Send text the server can't re-fetch itself (uploads / no URL), capped.
    const fullTexts: Record<string, string> = {};
    for (const p of raw) {
      const ft = s.fullTexts[p.id]?.text;
      if (ft && (p.source === "Local PDFs" || !p.url)) fullTexts[p.id] = ft.slice(0, 8000);
    }
    const idx = qaTurns.length;
    setQaTurns(prev => [...prev, { question: qq, answer: "", sources: [], busy: true, status: "Searching your library…", ts: Date.now(), note: note || undefined }]);
    try {
      const history = qaTurns.filter(t => t.answer).flatMap(t => [
        { role: "user", content: t.question },
        { role: "assistant", content: t.answer },
      ]);
      const r = await AIService.agentChat(qq, history, library, fullTexts, model);
      if (r.unsupported) {
        setQaTurns(prev => prev.map((x, i) => i === idx ? { ...x, answer: "That model can't use tools. Pick a tool-capable model (Claude, GPT-4o, or a local qwen2.5 / llama3.1) and try again.", busy: false, status: undefined } : x));
        return;
      }
      setQaTurns(prev => prev.map((x, i) => i === idx ? { ...x, answer: r.answer, sources: r.documents, busy: false, status: undefined } : x));
    } catch (e: any) {
      setQaTurns(prev => prev.map((x, i) => i === idx ? { ...x, answer: e?.message || "Sorry, that failed. Please try again.", busy: false, status: undefined } : x));
    }
  }

  // Answer a question from the user's collected library, appended as a chat turn.
  // Ranks the WHOLE library (rawPapers) so a paper that only has a title/abstract
  // on record is still eligible: the most relevant ones missing text get fetched on
  // demand (reusing the acquisition engine) before we ground the answer.
  async function answerFromLibrary(question: string) {
    const qq = question.trim();
    if (!qq) return;
    setInput("");

    type Cand = { id: string; title: string; url?: string; source?: string; text: string };
    let candidates: Cand[];
    if (pendingDocs.length) {
      // Freshly attached documents stay the focus of the next message.
      candidates = pendingDocs.map(d => ({ id: d.id, title: d.title, url: d.url, source: d.source, text: d.text }));
      setPendingDocIds([]);
    } else {
      const all: Cand[] = (s.rawPapers || []).map(p => ({
        id: p.id, title: p.title, url: p.url, source: p.source,
        text: (s.fullTexts[p.id]?.text || p.abstract || ""),
      }));
      candidates = rankByRelevance(qq, all).slice(0, 12);
    }
    if (!candidates.length) { await runChat(qq); return; }

    const idx = qaTurns.length;
    setQaTurns(prev => [...prev, { question: qq, answer: "", sources: [], busy: true, ts: Date.now() }]);
    try {
      // Read full text on demand for the top relevant papers that have little or no
      // text yet and a URL to fetch from. Best-effort and capped so a slow or
      // paywalled source can't hang the reply.
      const needFetch = candidates.slice(0, 6).filter(c => (c.text || "").trim().length < 200 && (c.url || "").trim()).slice(0, 4);
      if (needFetch.length) {
        setQaTurns(prev => prev.map((x, i) => i === idx ? { ...x, status: needFetch.length === 1 ? "Reading the full text…" : `Reading the full text of ${needFetch.length} sources…` } : x));
        const fetched = await Promise.allSettled(needFetch.map(c =>
          AIService.fetchFullText({ Title: c.title, URL: c.url || "", Source: c.source || "", paper_id: c.id }).then(r => ({ c, r })),
        ));
        const updates: Record<string, FullTextRecord> = {};
        for (const f of fetched) {
          if (f.status !== "fulfilled") continue;
          const { c, r } = f.value;
          if (r.status === "found" && (r.text || "").trim()) {
            c.text = r.text!;
            updates[c.id] = { paper_id: c.id, title: c.title, url: c.url || "", source: r.source || c.source || "", status: "found", text: r.text };
          }
        }
        if (Object.keys(updates).length) s.setFullTexts(prev => ({ ...prev, ...updates }));
        setQaTurns(prev => prev.map((x, i) => i === idx ? { ...x, status: undefined } : x));
      }

      const set = candidates.filter(c => (c.text || "").trim()).slice(0, 10);
      if (!set.length) {
        setQaTurns(prev => prev.map((x, i) => i === idx ? { ...x, answer: "I have those sources on record, but couldn't retrieve any readable text (the full text may be paywalled or unavailable). Attach the PDF and I can read it directly.", busy: false, status: undefined } : x));
        return;
      }
      // Recent textual turns give follow-ups their context ("access it" -> what?).
      const history = qaTurns.filter(t => t.answer).flatMap(t => [
        { role: "user", content: t.question },
        { role: "assistant", content: t.answer },
      ]);
      const r = await AIService.askDocuments(
        qq,
        set.map(c => ({ id: c.id, title: c.title, text: c.text.slice(0, 6000) })),
        { history, totalDocuments: (s.rawPapers || []).length },
      );
      setQaTurns(prev => prev.map((x, i) => i === idx ? { ...x, answer: r.answer, sources: r.documents, busy: false, status: undefined } : x));
    } catch (e: any) {
      setQaTurns(prev => prev.map((x, i) => i === idx ? { ...x, answer: e?.message || "Sorry, that failed. Please try again.", busy: false, status: undefined } : x));
    }
  }

  const [refining, setRefining] = useState(false);
  const [refinement, setRefinement] = useState<null | {
    field: string;   // active frame element id (PICO or PCC)
    current: string;
    suggested: string;
    reason: string;
    is_clarification?: boolean;
  }>(null);

  const last = s.history[s.history.length - 1];
  const task = s.tasks["home-analysis"];
  const analyzing = task?.status === "running";

  // Clarifying-questions modal state. The modal opens before the search runs
  // when the backend returns 1-3 multiple-choice questions about underspecified
  // PICO elements. The resolver ref is wired up inside handleSubmit so the
  // async flow there can await the user's answers.
  const [clarifyOpen, setClarifyOpen] = useState(false);
  const [clarifyGoal, setClarifyGoal] = useState("");
  // Frame shown in the clarifier pills (mirrors s.framework, set just before the
  // modal opens). fwOverrideRef records when the user manually picks a frame via
  // the selector, so auto-detect on submit doesn't clobber their choice.
  const [clarifyFramework, setClarifyFramework] = useState<FrameworkId>(s.framework);
  const fwOverrideRef = useRef(false);
  // Strategy Review drawer open-state lives in the store so the header bar can
  // toggle it (the floating button was removed).
  const reviewOpen = s.reviewOpen;
  const setReviewOpen = s.setReviewOpen;
  const clarifyResolverRef = useRef<((answers: Record<string, string>) => void) | null>(null);

  function markStage(id: StageId, patch: Partial<Stage>) {
    s.updateTaskStage("home-analysis", id, patch);
  }

  async function runStage<T>(
    id: StageId,
    signal: AbortSignal,
    fn: (signal: AbortSignal) => Promise<T>,
  ): Promise<T | null> {
    if (signal.aborted) return null;
    markStage(id, { status: "running" });
    try {
      const result = await fn(signal);
      markStage(id, { status: "done" });
      return result;
    } catch (e: any) {
      if (signal.aborted || e?.name === "AbortError") {
        markStage(id, { status: "canceled" });
      } else {
        markStage(id, { status: "error", detail: e?.message?.slice(0, 80) || "Failed" });
      }
      return null;
    }
  }

  // Turn a plain-language / custom request into a structured strategy via the LLM
  // — the framework, PICO (or PCC) elements, inclusion/exclusion, the search query
  // and the research question — instead of using the raw text verbatim. Used by the
  // Study-design PICO box so a pasted custom request is always interpreted.
  async function inferStrategyFromText(text: string): Promise<boolean> {
    const t = text.trim();
    if (!t) return false;
    setInferring(true);
    try {
      let fw = s.framework;
      if (s.history.length === 0 && !fwOverrideRef.current) {
        try { fw = await AIService.detectFramework(t); s.setFramework(fw); } catch { /* keep current */ }
      }
      const prior = s.history.length > 0
        ? { p: s.pico.population, i: s.pico.intervention, c: s.pico.comparator, o: s.pico.outcome,
            concept: s.pico.concept, context: s.pico.context, inclusion: s.inclusion, exclusion: s.exclusion }
        : null;
      const analysis = await AIService.inferPicoAndQuery(t, prior, fw);
      const resolvedFw = analysis.framework || fw;
      const newPico = {
        population: analysis.p, intervention: analysis.i, comparator: analysis.c, outcome: analysis.o,
        concept: analysis.concept || "", context: analysis.context || "", framework: resolvedFw,
      };
      s.setPico(newPico);
      s.setFramework(resolvedFw);
      s.setInclusion(analysis.inclusion);
      s.setExclusion(analysis.exclusion);
      s.setQuery(analysis.query);
      s.setUnifiedSearchQuery(analysis.query);
      try { setInferredQuestion(await AIService.generateFormalQuestion(newPico, t)); } catch { /* optional */ }
      toast.success("Strategy inferred from your description");
      return true;
    } catch {
      toast.error("Couldn't infer a strategy from that. Try rephrasing.");
      return false;
    } finally {
      setInferring(false);
    }
  }

  // Regenerate the search string from the current framework elements via the LLM,
  // so the query reflects the structured strategy instead of a hand-typed string.
  async function rebuildQueryFromPico(): Promise<void> {
    const desc = frameworkOf(s.framework).elements
      .map(el => `${el.label}: ${((s.pico as Record<string, string>)[el.id] || "").trim()}`)
      .filter(line => line.split(": ")[1])
      .join(". ");
    if (!desc) { toast.error("Fill in the elements first, then rebuild the query."); return; }
    setInferring(true);
    try {
      const prior = { p: s.pico.population, i: s.pico.intervention, c: s.pico.comparator, o: s.pico.outcome,
                      concept: s.pico.concept, context: s.pico.context, inclusion: s.inclusion, exclusion: s.exclusion };
      const analysis = await AIService.inferPicoAndQuery(desc, prior, s.framework);
      s.setQuery(analysis.query);
      s.setUnifiedSearchQuery(analysis.query);
      toast.success("Search string rebuilt from your elements");
    } catch {
      toast.error("Couldn't rebuild the query.");
    } finally {
      setInferring(false);
    }
  }

  async function handleSubmit(text: string, opts: { skipClarify?: boolean } = {}) {
    const t = text.trim();
    if (!t) return;
    setInput("");

    // Route every message to the lightest tool so a command like "summarize" or a
    // general question doesn't kick off the full clarify + PICO + retrieval
    // pipeline. Only a genuine search goal runs that. Skipped for refinements
    // (skipClarify), which are always a deliberate search action.
    if (!opts.skipClarify) {
      // A freshly attached file is always answered from that file, with its
      // extracted text sent as chat context, regardless of how the intent router
      // would classify the message. "Add a file to the chat" => that file grounds
      // the reply and lifts response quality.
      if (pendingDocs.length) { await answerFromLibrary(t); return; }

      // Brand-new review with nothing collected yet: the first substantive message
      // is a research goal, so always run the infer + search pipeline. This keeps a
      // conversational-sounding goal from being routed to a verbatim chat reply
      // instead of being structured by the LLM into PICO + question + search. Short
      // greetings / how-tos still fall through to a plain chat.
      const chatty = /^(hi|hey|hello|thanks|thank you|ok(ay)?|yo|sup|help|how (do|can|does)\b|what (is|are|can) you\b)/i.test(t);
      const forceSearch = s.history.length === 0 && (s.rawPapers?.length ?? 0) === 0 && !chatty && t.length >= 10;

      if (!forceSearch) {
        let intent: "documents" | "search" | "chat" = "search";
        try {
          intent = await AIService.routeIntent(t, docCorpus.length > 0);
        } catch {
          intent = docCorpus.length > 0 && /\?\s*$/.test(t) ? "documents" : "search";
        }
        if (intent === "documents" || intent === "chat") {
          // Answer from the collected library for BOTH intents so the assistant never
          // denies access to sources the user has. Open library questions go to the
          // tool-calling agent when a capable model is available (it searches and reads
          // on demand); the no-capable-model case uses the deterministic path instead.
          const hasLibrary = (s.rawPapers?.length ?? 0) > 0;
          if (hasLibrary) {
            let agentModel = supportsTools(s.model) ? s.model : "";
            let note = "";
            if (!agentModel) {
              const qwen = localModels.find(m => /qwen2\.5/i.test(m)) || localModels.find(m => /qwen2/i.test(m)) || localModels.find(m => supportsTools(m));
              if (qwen) { agentModel = qwen; note = `${s.model} can't use tools, so I switched to ${qwen} for this answer.`; }
            }
            if (agentModel) { await answerWithAgent(t, agentModel, note); return; }
            await answerFromLibrary(t);
            return;
          }
          // No library yet — a plain conversational reply (greetings, how-to).
          await runChat(t);
          return;
        }
      }
      // Genuine search: newly-attached docs stay in the corpus but lose "focus".
      setPendingDocIds([]);
    }

    // 0. Ask the user to disambiguate underspecified PICO elements BEFORE the
    //    search runs. The system used to silently infer these, which caused
    //    catastrophic drift (e.g. "Mediterranean diet → longevity" becoming
    //    "low-carb diet → BMI in overweight 18-65"). Now the user owns the
    //    answer through a Claude-style multi-question popup. If the model
    //    returns no clarifying questions, this is a no-op.
    //
    //    Skipped when handleSubmit is called from applyRefinement: the
    //    refinement IS the clarification, no need to ask again.
    // Auto-detect the question frame (PICO vs PCC) for a fresh review, unless the
    // user has manually picked one via the selector. `activeFw` (a local) is the
    // source of truth for the rest of this submit — s.framework won't update within
    // this same closure after setFramework, so we must thread the local through.
    let activeFw: FrameworkId = s.framework;
    if (!opts.skipClarify && s.history.length === 0 && !fwOverrideRef.current) {
      try {
        activeFw = await AIService.detectFramework(t);
      } catch { /* keep current */ }
      s.setFramework(activeFw);
      setClarifyFramework(activeFw);
    }

    let clarifyAnswers: Record<string, string> = {};
    if (!opts.skipClarify) {
      try {
        clarifyAnswers = await new Promise<Record<string, string>>((resolve) => {
          setClarifyGoal(t);
          setClarifyOpen(true);
          clarifyResolverRef.current = (answers) => {
            setClarifyOpen(false);
            clarifyResolverRef.current = null;
            resolve(answers);
          };
        });
      } catch (e) {
        console.warn("[clarify] failed; proceeding without:", e);
      }
    }

    // Fold the user's clarifying answers into the goal text so PICO inference
    // sees them. We append rather than replace so the original phrasing is
    // preserved verbatim for the must-include query anchors.
    const clarifyExtras = Object.entries(clarifyAnswers)
      .map(([k, v]) => `${k}: ${v}`)
      .join("; ");
    const effectiveText = clarifyExtras ? `${t}\n\nFurther context. ${clarifyExtras}` : t;

    const submittedAt = Date.now();   // stamp the eventual history entry with submit time so it interleaves with chat turns in order
    const stageList: Stage[] = INITIAL_STAGES.map(st => ({ ...st, status: "pending" as const }));
    const { abort } = s.startTask("home-analysis", stageList);
    const signal = abort.signal;

    try {
      // 1. PICO inference. When a strategy already exists, treat this message as
      //    a REFINEMENT of the CURRENT active strategy (the one shown in the
      //    Strategy Review drawer, includes any edits / version reverts) so the
      //    operationalised detail is preserved instead of regenerated.
      const prior = s.history.length > 0
        ? { p: s.pico.population, i: s.pico.intervention, c: s.pico.comparator, o: s.pico.outcome,
            concept: s.pico.concept, context: s.pico.context,
            inclusion: s.inclusion, exclusion: s.exclusion }
        : null;
      const fw = activeFw;
      let analysis = await runStage("pico", signal, sig => AIService.inferPicoAndQuery(effectiveText, prior, fw, sig));
      if (!analysis) { s.updateTask("home-analysis", { status: signal.aborted ? "canceled" : "error" }); return; }

      const resolvedFw = analysis.framework || fw;
      let newPico = {
        population: analysis.p, intervention: analysis.i, comparator: analysis.c, outcome: analysis.o,
        concept: analysis.concept || "", context: analysis.context || "", framework: resolvedFw,
      };
      s.setPico(newPico);
      s.setFramework(resolvedFw);
      s.setInclusion(analysis.inclusion);
      s.setExclusion(analysis.exclusion);
      s.setQuery(analysis.query);
      s.setUnifiedSearchQuery(analysis.query);

      // Frame the question: formalise it from PICO before the search is shown.
      const formalQ = await runStage("question", signal, sig => AIService.generateFormalQuestion(newPico, effectiveText, sig));

      // Build the search: always run the comprehensive, MeSH-grounded builder
      // (broadened by any seed studies) rather than the basic PICO string, then
      // derive the adversarial (counter-evidence) variant as a companion.
      const built = await runGroundedBuild({ pico: newPico, question: formalQ || effectiveText, signal });
      const groundedQuery = built?.query || analysis.query;
      analysis = { ...analysis, query: groundedQuery };
      markStage("query", { status: "done", detail: groundedQuery ? groundedQuery.slice(0, 60) + "…" : undefined });
      const adv = await runStage("adversarial", signal, sig => AIService.generateAdversarialQuery(newPico, sig));

      // ── Pre-search review gate ──────────────────────────────────────────
      // Surface the full plan and let the reviewer edit anything (search string,
      // PICO, criteria, databases, limits) before ANY database is queried. The
      // modal edits the store live; on Run it resolves with a fresh snapshot,
      // which we apply here (store reads in this running closure would be stale).
      const plan = await new Promise<PlanDecision | null>(resolve => {
        planResolverRef.current = resolve;
        setPlanGate({ question: formalQ || "" });
        signal.addEventListener("abort", () => resolve(null), { once: true });
      });
      planResolverRef.current = null;
      setPlanGate(null);
      if (!plan || signal.aborted) {
        s.updateTask("home-analysis", { status: "canceled" });
        return;
      }
      newPico = {
        population: plan.pico.population, intervention: plan.pico.intervention,
        comparator: plan.pico.comparator, outcome: plan.pico.outcome,
        concept: plan.pico.concept || "", context: plan.pico.context || "",
        framework: plan.pico.framework || resolvedFw,
      };
      analysis = { ...analysis, query: plan.query, inclusion: plan.inclusion, exclusion: plan.exclusion };
      const runSources = plan.sources;
      const runLimits = plan.perSourceLimits;
      const runNum = plan.numPerSource;

      // 2. If the user uploaded their own studies, analyse THOSE (no database
      //    fetch). Otherwise fetch a wide sample so the relevance filter has room.
      // Always pull from the databases and fold in any uploaded PDFs, so the corpus
      // is pulled results PLUS the user's own files, never one at the expense of the
      // other. (Having an upload used to skip the database fetch entirely, which is
      // why searches "didn't pull" once anything was attached.) To analyse only your
      // own PDFs, uncheck the databases in the sidebar so the fetch returns nothing.
      const uploaded = (s.rawPapers || []).filter(p => p.source === "Local PDFs");
      const fetched = await runStage("papers", signal, sig =>
        DataAggregator.fetchPerSource(analysis.query, runSources, newPico, runLimits, runNum, sig, s.searchSelection, s.searchFilters)
      );
      const fetchedPapers = fetched?.papers || [];
      let papers: Paper[] = [...fetchedPapers, ...uploaded];
      if (fetched || uploaded.length) {
        const breakdown = Object.entries(fetched?.sourceCounts || {})
          .map(([k, v]) => `${k}: ${v}`)
          .join(" · ");
        const detail = `${fetchedPapers.length} article${fetchedPapers.length === 1 ? "" : "s"}`
          + (uploaded.length ? ` + ${uploaded.length} uploaded` : "")
          + (breakdown ? `: ${breakdown}` : "");
        markStage("papers", { status: "done", detail });
        s.setRawPapers(papers);
      }

      // 3. LEADS-native relevance rerank. Papers that pass the threshold get
      //    fed to the summariser; the rest are discarded so the summary stops
      //    citing tangential hits (zoonoses, etc.) just because they matched
      //    the keyword query. This uses LEADS for its trained task regardless
      //    of which model is selected in the sidebar.
      // Auto-cutoff mode: the backend picks the relevance floor from the score
      // distribution itself (gap detection + hard floor at 0.0). No user-facing
      // threshold; the rerank endpoint adapts to whether the retrieved corpus is
      // junk-heavy or clean.
      let relevantPapers = papers;
      if (papers.length > 0) {
        const reranked = await runStage("rerank", signal, async sig => {
          try {
            // Streamed so the stage shows which article is being scored.
            return await DataAggregator.rerankByRelevanceStream(
              papers,
              newPico,
              analysis.inclusion,
              analysis.exclusion,
              -1.0,        // disabled, auto mode supersedes
              undefined,   // topK
              sig,
              undefined,   // quantileKeep
              (done, total) => markStage("rerank", { status: "running", detail: `Scoring article ${done} of ${total}` }),
            );
          } catch (e: any) {
            if (sig.aborted || e?.name === "AbortError") throw e;   // genuine cancel
            // Stream unavailable (old backend / buffering proxy) → blocking call.
            return await DataAggregator.rerankByRelevance(
              papers, newPico, analysis.inclusion, analysis.exclusion, -1.0, undefined, sig,
            );
          }
        });
        if (reranked) {
          relevantPapers = reranked.kept.map(r => r.paper);
          s.setRerankResults(reranked);
          s.setRelevanceOverrides({});   // fresh rerank: clear any manual include/exclude overrides
          const floor = typeof reranked.effective_floor === "number"
            ? reranked.effective_floor.toFixed(2)
            : reranked.threshold.toFixed(2);
          markStage("rerank", {
            status: "done",
            detail: `${reranked.total_kept} of ${reranked.total_scored} kept (auto cutoff ${floor})`,
          });
        }
      } else {
        s.setRerankResults(null);
        markStage("rerank", { status: "done", detail: "no articles to score" });
      }

      // Scope what's there: summarise the retrieved literature, then suggest refinements.
      const summaryWithRefs = await runStage("summary", signal, sig => AIService.generateComprehensiveSummaryWithRefs(t, relevantPapers, sig));
      const suggs = await runStage("suggestions", signal, sig => AIService.getRefinementSuggestions(t, relevantPapers, sig));

      if (signal.aborted) {
        s.updateTask("home-analysis", { status: "canceled" });
        return;
      }

      s.setHistory(h => [...h, {
        goal: t,
        query: analysis.query,
        formal_question: formalQ || "",
        summary: summaryWithRefs?.summary || "",
        references: summaryWithRefs?.references || [],
        pico_dict: analysis,
        suggestions: suggs || [],
        inclusion: analysis.inclusion,
        exclusion: analysis.exclusion,
        adversarial_query: adv || "",
        ts: submittedAt,
      }]);
      s.updateTask("home-analysis", { status: "done" });
    } catch (e: any) {
      s.updateTask("home-analysis", { status: "error", detail: e?.message });
    }
  }

  async function suggestRefinement() {
    setRefining(true);
    setRefinement(null);
    try {
      const r = await AIService.refinePico(s.pico, last?.goal || "");
      if (r.field) {
        setRefinement(r as any);
      }
    } finally {
      setRefining(false);
    }
  }

  function applyRefinement() {
    if (!refinement?.field) return;
    // Update PICO in the store immediately so the next handleSubmit picks it
    // up. Then re-run the full analysis pipeline with the refinement folded
    // into the goal text so the search query, retrieval, rerank, and summary
    // all reflect the new constraint. We skip the clarifying-questions popup
    // because the refinement itself IS a clarification, re-prompting would
    // be circular.
    const field = refinement.field;
    const suggested = refinement.suggested;
    s.setPico(p => ({ ...p, [field]: suggested }));
    const baseGoal = (last?.goal || "").trim();
    setRefinement(null);
    if (baseGoal) {
      const augmented = `${baseGoal}\n\nFurther context. ${field}: ${suggested}`;
      // Fire-and-forget. handleSubmit manages its own task lifecycle.
      void handleSubmit(augmented, { skipClarify: true });
    }
  }

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {s.history.length === 0 && !analyzing && (
        <Alert className="flex items-center gap-2"><span className="shrink-0 text-base leading-none" role="img" aria-label="Waving hand">👋</span><AlertDescription>Welcome! Describe your research goal below to generate a strategy and see initial findings.</AlertDescription></Alert>
      )}

      {/* One chronological thread: search (history) entries and chat (Q&A) turns
          are interleaved by their creation time so a new search stays where it
          was asked instead of jumping above earlier chat turns. */}
      {[
        ...s.history.map((entry, idx) => ({ ts: entry.ts ?? 0, node: (
        <div key={`h-${idx}`} className="space-y-3">
          <div className="flex justify-end">
            <div className="bg-primary text-primary-foreground rounded-2xl rounded-tr-sm px-4 py-2 max-w-2xl">
              <span className="opacity-80 text-xs">Research Goal</span>
              <div>{entry.goal}</div>
            </div>
          </div>
          <Card className="overflow-hidden border-border/70 shadow-sm ring-1 ring-black/[0.02]">
            <Tabs defaultValue="overview" className="w-full">
              <div className="border-b border-border/60 p-2.5 flex items-center justify-between gap-2 flex-wrap">
                <TabsList>
                  {[
                    ["overview", "Overview"],
                    // Label by this run's frame: "PICO" for intervention reviews,
                    // "PCC" for scoping reviews (matches the elements shown).
                    ["pico", frameworkOf(entry.pico_dict?.framework ?? s.framework).label],
                    ["criteria", "Criteria"],
                    ["search", "Search"],
                    ...(idx === s.history.length - 1 && s.rerankResults ? [["relevance", "Relevance"]] : []),
                    ...(idx === s.history.length - 1 && (s.rawPapers?.length ?? 0) > 0 ? [["refine", "Refine"]] : []),
                  ].map(([value, label]) => (
                    <TabsTrigger key={value} value={value}>
                      {label}
                    </TabsTrigger>
                  ))}
                </TabsList>
                {(() => {
                  const pd: any = entry.pico_dict || {};
                  const ePop = pd.population ?? pd.p ?? "";
                  const eInt = pd.intervention ?? pd.i ?? "";
                  const eCmp = pd.comparator ?? pd.c ?? "";
                  const eOut = pd.outcome ?? pd.o ?? "";
                  const isActive = entry.query === s.query
                    && ePop === s.pico.population && eInt === s.pico.intervention
                    && eCmp === s.pico.comparator && eOut === s.pico.outcome;
                  return isActive ? (
                    <span className="text-xs text-emerald-600 font-medium inline-flex items-center gap-1 px-2 shrink-0">
                      <Check className="size-3.5" />Active version
                    </span>
                  ) : (
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-8 text-xs shrink-0"
                      title="Make this version's PICO, criteria, and search the active strategy"
                      onClick={() => {
                        s.setPico({ population: ePop, intervention: eInt, comparator: eCmp, outcome: eOut });
                        s.setInclusion(entry.inclusion || []);
                        s.setExclusion(entry.exclusion || []);
                        s.setQuery(entry.query || "");
                        s.setUnifiedSearchQuery(entry.query || "");
                        toast.success("Strategy restored from this version");
                      }}
                    >
                      <RotateCcw className="size-3.5 mr-1.5" />Use this version
                    </Button>
                  );
                })()}
              </div>
              <div className="p-5 min-h-[240px]">
                <TabsContent value="overview" className="mt-0">
                  <OverviewTab entry={entry} idx={idx} />
                </TabsContent>

                <TabsContent value="pico" className="mt-0">
                  <PicoCards pico={entry.pico_dict} framework={entry.pico_dict?.framework ?? s.framework} />
                </TabsContent>

                <TabsContent value="criteria" className="mt-0">
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="rounded-lg border bg-card p-3">
                      <div className="text-sm font-semibold text-emerald-700 mb-2">Include</div>
                      <ul className="space-y-1.5">
                        {entry.inclusion.map((x, i) => (
                          <li key={i} className="flex gap-2 text-sm leading-snug">
                            <span className="mt-1.5 size-1.5 rounded-full bg-emerald-500 shrink-0" />
                            <span>{x}</span>
                          </li>
                        ))}
                        {entry.inclusion.length === 0 && <li className="text-sm text-muted-foreground">None specified.</li>}
                      </ul>
                    </div>
                    <div className="rounded-lg border bg-card p-3">
                      <div className="text-sm font-semibold text-rose-700 mb-2">Exclude</div>
                      <ul className="space-y-1.5">
                        {entry.exclusion.map((x, i) => (
                          <li key={i} className="flex gap-2 text-sm leading-snug">
                            <span className="mt-1.5 size-1.5 rounded-full bg-rose-500 shrink-0" />
                            <span>{x}</span>
                          </li>
                        ))}
                        {entry.exclusion.length === 0 && <li className="text-sm text-muted-foreground">None specified.</li>}
                      </ul>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="search" className="mt-0 space-y-4">
                  <QueryBlock label="Final MeSH search string" value={entry.query} />
                  {entry.adversarial_query && (
                    <QueryBlock label="Adversarial query (sensitivity check)" value={entry.adversarial_query} />
                  )}
                </TabsContent>

                {/* Relevance-rerank explorer, only for the most recent run,
                    since rerankResults holds only the latest LEADS pass. */}
                {idx === s.history.length - 1 && s.rerankResults && (
                  <TabsContent value="relevance" className="mt-0">
                    <RelevanceExplorer />
                  </TabsContent>
                )}

                {/* Refine: after reading the summary and evidence, pick relevant
                    studies as seeds for a broadened next search. Latest run only. */}
                {idx === s.history.length - 1 && (s.rawPapers?.length ?? 0) > 0 && (
                  <TabsContent value="refine" className="mt-0">
                    <RefineSeeds
                      papers={s.rawPapers || []}
                      seedIds={s.seedIds}
                      onToggleSeed={id => s.setSeedIds(prev => { const n = new Set(prev); if (n.has(id)) n.delete(id); else n.add(id); return n; })}
                      onRefine={() => { const goal = entry.goal || input; if (goal.trim()) void handleSubmit(goal, { skipClarify: true }); }}
                    />
                  </TabsContent>
                )}

              </div>
            </Tabs>
          </Card>
        </div>
        ) })),
        // Document Q&A turns, answered by the main chat and shown inline.
        ...qaTurns.map((turn, i) => ({ ts: turn.ts ?? 0, node: (
        <div key={`qa-${i}`} className="space-y-3">
          <div className="flex justify-end">
            <div className="bg-primary text-primary-foreground rounded-2xl rounded-tr-sm px-4 py-2 max-w-2xl">{turn.question}</div>
          </div>
          <Card className="p-4">
            {turn.note && <div className="mb-2 text-xs text-muted-foreground italic">{turn.note}</div>}
            {turn.busy ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" />{turn.status || (turn.question.startsWith("Structured summary") ? "Summarizing your selected sources…" : "Reading your documents…")}</div>
            ) : turn.table ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-medium">Structured summary · {turn.table.length} source{turn.table.length === 1 ? "" : "s"}</span>
                  <div className="flex items-center gap-1">
                    <Button size="sm" variant="ghost" className="size-7 px-0 text-muted-foreground" title="Download as CSV" onClick={() => {
                      const esc = (v: string) => `"${String(v ?? "").replace(/"/g, '""')}"`;
                      const cols: (keyof SummaryRow)[] = ["design", "population", "intervention", "comparator", "outcomes", "key_finding"];
                      const lines = [["Source", "Design", "Population", "Intervention", "Comparator", "Outcomes", "Key finding"], ...turn.table!.map(r => [r.title, ...cols.map(c => String(r[c] ?? ""))])];
                      const blob = new Blob([lines.map(l => l.map(esc).join(",")).join("\n")], { type: "text/csv" });
                      const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "structured_summary.csv"; a.click(); URL.revokeObjectURL(a.href);
                    }}><Download className="size-3.5" /></Button>
                    <Button size="sm" variant="ghost" className="size-7 px-0 text-muted-foreground" title="Copy table" onClick={() => {
                      const cols: (keyof SummaryRow)[] = ["design", "population", "intervention", "comparator", "outcomes", "key_finding"];
                      const lines = [["Source", "Design", "Population", "Intervention", "Comparator", "Outcomes", "Key finding"], ...turn.table!.map(r => [r.title, ...cols.map(c => String(r[c] ?? "").replace(/[\t\n]+/g, " "))])];
                      navigator.clipboard.writeText(lines.map(l => l.join("\t")).join("\n"));
                      toast.success("Copied");
                    }}><Copy className="size-3.5" /></Button>
                  </div>
                </div>
                <div className="overflow-x-auto rounded-md border">
                  <table className="w-full text-xs border-collapse">
                    <thead>
                      <tr className="bg-muted/50 text-left">
                        {["Source", "Design", "Population", "Intervention", "Comparator", "Outcomes", "Key finding"].map(h => (
                          <th key={h} className="px-2 py-1.5 font-medium whitespace-nowrap">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {turn.table.map(r => (
                        <tr key={r.id} className="border-t align-top">
                          <td className="px-2 py-1.5 font-medium max-w-[12rem]">{r.title}</td>
                          {(["design", "population", "intervention", "comparator", "outcomes", "key_finding"] as (keyof SummaryRow)[]).map(c => (
                            <td key={c} className="px-2 py-1.5 min-w-[9rem] whitespace-pre-wrap">{r[c] || <span className="text-muted-foreground">—</span>}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="text-sm leading-relaxed space-y-1.5 text-foreground/90">
                  {renderAnswer(turn.answer, n => { const src = turn.sources.find(x => x.n === n); if (src) openDocPreview(src.id); })}
                </div>
                {(() => {
                  const cited = new Set((turn.answer.match(/\[(\d+)\]/g) || []).map(m => Number(m.replace(/[^0-9]/g, ""))));
                  const shown = cited.size ? turn.sources.filter(x => cited.has(x.n)) : [];
                  return shown.length ? (
                    <div className="text-xs text-muted-foreground pt-2 border-t flex flex-wrap items-baseline gap-x-3 gap-y-1">
                      <span className="font-medium">Sources:</span>
                      {shown.map(src => (
                        <button key={src.n} type="button" onClick={() => openDocPreview(src.id)}
                          style={{ fontSize: "inherit" }}
                          className="text-primary hover:underline text-left whitespace-normal break-words max-w-full" title="Preview this document">
                          [{src.n}] {src.title}
                        </button>
                      ))}
                    </div>
                  ) : null;
                })()}
              </div>
            )}
          </Card>
        </div>
        ) })),
      ].sort((a, b) => a.ts - b.ts).map(x => x.node)}


      {analyzing && task && (
        <AnalysisProgress
          stages={task.stages as Stage[]}
          startedAt={task.startedAt}
          onCancel={() => s.cancelTask("home-analysis")}
        />
      )}

      {/* Strategy review lives in a collapsible right-hand drawer (below), not
          inline, so the main column stays short. */}

      <div className="h-24" />

      {/* Clarifying-questions modal, opens BEFORE the search runs whenever the
          user's goal is under-specified. Pauses handleSubmit until the user
          answers or skips, then resumes with the answers folded into the
          effective goal text. */}
      <ClarifyingQuestionsModal
        open={clarifyOpen}
        goal={clarifyGoal}
        framework={clarifyFramework}
        onDone={(answers) => clarifyResolverRef.current?.(answers)}
        onSkipAll={() => clarifyResolverRef.current?.({})}
      />


      {/* Refinement popup, floats above the chat input, Claude-clarifying-question style */}
      {(refining || refinement) && s.history.length > 0 && (
        <div className="fixed bottom-20 left-72 right-0 z-30 px-6 pointer-events-none transition-all">
          <div className="max-w-4xl mx-auto pointer-events-auto">
            <Card className="p-4 border-primary/40 shadow-xl bg-card/98 backdrop-blur">
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-2.5 flex-1">
                  <Lightbulb className="size-4 text-primary mt-0.5 shrink-0" />
                  <div className="space-y-2 flex-1 min-w-0">
                    {refining && !refinement && (
                      <div className="text-sm text-muted-foreground">
                        Looking at your question for the weakest PICO element…
                      </div>
                    )}
                    {refinement && refinement.is_clarification && (
                      <>
                        <div className="text-xs uppercase tracking-wide text-muted-foreground">
                          Clarifying question · <span className="text-foreground font-medium">{refinement.field}</span>
                        </div>
                        <div className="text-sm text-foreground break-words">
                          {refinement.reason || `What ${refinement.field} should we focus on?`}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          One possible starting point:
                        </div>
                        <div className="text-sm text-foreground font-medium break-words">
                          {refinement.suggested}
                        </div>
                      </>
                    )}
                    {refinement && !refinement.is_clarification && (
                      <>
                        <div className="text-sm">
                          <span className="text-muted-foreground">Want to make your </span>
                          <span className="font-medium">{refinement.field}</span>
                          <span className="text-muted-foreground"> more specific?</span>
                        </div>
                        <div className="text-sm space-y-1">
                          <div className="text-muted-foreground line-through break-words text-xs">
                            {refinement.current || <em>empty</em>}
                          </div>
                          <div className="text-foreground font-medium break-words">
                            {refinement.suggested}
                          </div>
                        </div>
                        <div className="text-xs text-muted-foreground italic">
                          {refinement.reason}
                        </div>
                      </>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => setRefinement(null)}
                  className="text-muted-foreground hover:text-foreground shrink-0"
                  aria-label="Dismiss"
                >
                  <X className="size-4" />
                </button>
              </div>
              {refinement && (
                <div className="flex gap-2 mt-3 pl-6">
                  <Button size="sm" onClick={applyRefinement}>
                    <Check className="size-4 mr-2" />Use this
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      if (!refinement) return;
                      setInput(prev => (prev ? `${prev}, ${refinement.suggested}` : refinement.suggested));
                      setRefinement(null);
                    }}
                  >
                    Add to question
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => setRefinement(null)}>
                    Skip
                  </Button>
                </div>
              )}
            </Card>
          </div>
        </div>
      )}

      {/* ── Pre-search review gate: shows the plan up front and lets the user
          edit anything before the search actually runs. ────────────────── */}
      <Dialog open={!!planGate} onOpenChange={o => { if (!o) planResolverRef.current?.(null); }}>
        <DialogContent className="sm:max-w-2xl w-[95vw] h-[86vh] max-h-[760px] flex flex-col p-0 gap-0">
          <DialogHeader className="px-5 py-4 border-b shrink-0 text-left space-y-0.5">
            <DialogTitle className="flex items-center gap-2 text-base">
              <Search className="size-4 text-primary" />Review your search
            </DialogTitle>
            <p className="text-xs text-muted-foreground m-0">Check and edit the plan below.</p>
          </DialogHeader>
          <div className="flex-1 overflow-auto px-5 py-4 space-y-5">
            {planGate?.question && (
              <div>
                <label className="text-muted-foreground text-sm block mb-1.5">Research question</label>
                <p className="text-sm leading-snug rounded-md border bg-muted/30 p-3 m-0">{planGate.question}</p>
              </div>
            )}
            <div>
              <div className="flex items-center justify-between gap-2 mb-1.5">
                <label className="text-muted-foreground text-sm">Search string</label>
              </div>
              <Textarea value={s.query} onChange={e => { s.setQuery(e.target.value); s.setUnifiedSearchQuery(e.target.value); }} rows={6} className="font-mono text-xs" />
              <p className="text-[11px] text-muted-foreground mt-1.5">Comprehensive query built from your concepts. Edit it freely — this exact string is what runs against your selected databases.</p>
              {builtConcepts.length > 0 && (
                <div className="mt-2 rounded-md border bg-muted/20 p-2.5 divide-y divide-border/60">
                  <div className="text-[11px] font-medium text-muted-foreground pb-1.5">Concept blocks</div>
                  {builtConcepts.map((c, i) => (
                    <details key={i} className="group text-[11px] leading-snug py-1.5 first-of-type:pt-0 last:pb-0">
                      <summary className="flex items-start gap-1.5 cursor-pointer list-none select-none">
                        <ChevronDown className="size-3 mt-[3px] shrink-0 text-muted-foreground transition-transform -rotate-90 group-open:rotate-0" />
                        <span className="flex-1">
                          <span className="font-semibold text-foreground">{c.name}</span>
                          <span className="text-muted-foreground"> · {c.mesh.length} MeSH, {c.tiab.length} keyword{c.tiab.length === 1 ? "" : "s"}</span>
                        </span>
                      </summary>
                      <div className="pl-[18px] pt-1.5 space-y-1.5">
                        {c.mesh.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {c.mesh.map((m, j) => (
                              <span key={`m${j}`} className="inline-block rounded bg-primary/10 text-primary px-1">{m}</span>
                            ))}
                          </div>
                        )}
                        <div className="text-muted-foreground">{c.tiab.join(", ")}</div>
                      </div>
                    </details>
                  ))}
                </div>
              )}
            </div>
            {/* Search filters researchers normally set (publication-year window,
                extensible). Applied at fetch time to PubMed / Europe PMC. */}
            <div>
              <label className="text-muted-foreground text-sm block mb-1.5">Filters</label>
              <div className="rounded-lg border bg-muted/30 p-3 space-y-2">
                <div className="flex items-center gap-3 flex-wrap">
                  <span className="inline-flex items-center gap-1.5 text-xs font-medium">
                    <CalendarRange className="size-3.5 text-muted-foreground" />Publication year
                  </span>
                  <div className="flex items-center gap-1.5">
                    <Input
                      type="number" inputMode="numeric" placeholder="From" min={1900} max={2100}
                      value={s.searchFilters.yearFrom ?? ""}
                      onChange={e => s.setSearchFilters(f => ({ ...f, yearFrom: e.target.value ? Number(e.target.value) : null }))}
                      className="h-8 w-24 text-sm"
                    />
                    <span className="text-xs text-muted-foreground">to</span>
                    <Input
                      type="number" inputMode="numeric" placeholder="To" min={1900} max={2100}
                      value={s.searchFilters.yearTo ?? ""}
                      onChange={e => s.setSearchFilters(f => ({ ...f, yearTo: e.target.value ? Number(e.target.value) : null }))}
                      className="h-8 w-24 text-sm"
                    />
                  </div>
                  {(s.searchFilters.yearFrom || s.searchFilters.yearTo) && (
                    <button type="button" onClick={() => s.setSearchFilters({})}
                      className="text-xs text-muted-foreground hover:text-foreground transition-colors">
                      Clear
                    </button>
                  )}
                </div>
                <p className="text-[11px] text-muted-foreground m-0">Restricts PubMed and Europe PMC to this window. Leave blank for no date limit.</p>
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-muted-foreground text-sm">Databases &amp; per-database limits</label>
                <span className="text-[11px] text-muted-foreground">
                  ≈ {s.sources.reduce((sum, src) => sum + (s.perSourceLimits[src] ?? s.numPerSource), 0).toLocaleString()} papers
                </span>
              </div>
              <div className="flex items-center justify-between gap-3 rounded-lg border bg-muted/40 px-3 py-2.5 mb-2">
                <div className="min-w-0">
                  <div className="text-xs font-semibold">Default per database</div>
                  <div className="text-[11px] text-muted-foreground">Applied to every database without its own cap</div>
                </div>
                <NumberStepper value={s.numPerSource} onChange={n => s.setNumPerSource(n)} />
              </div>
              {/* Selection strategy: only bites when a database matches MORE than
                  its limit. Below the limit, every match is kept. */}
              <div className="rounded-lg border bg-muted/40 px-3 py-2.5 mb-2 space-y-2">
                <div>
                  <div className="text-xs font-semibold">When a database has more matches than its limit</div>
                  <div className="text-[11px] text-muted-foreground">If it has fewer, all of them are kept.</div>
                </div>
                <div className="inline-flex rounded-lg border bg-background p-0.5">
                  {([["relevance", "Most relevant"], ["recent", "Most recent"]] as const).map(([val, lbl]) => (
                    <button
                      key={val}
                      type="button"
                      onClick={() => s.setSearchSelection(val)}
                      className={`px-3 h-7 rounded-md text-xs font-medium transition-colors ${
                        s.searchSelection === val ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {lbl}
                    </button>
                  ))}
                </div>
              </div>
              {s.sources.length === 0 ? (
                <div className="rounded-lg border border-dashed px-3 py-4 text-center text-[11px] text-muted-foreground">
                  No databases selected. Enable databases in the left sidebar under <span className="font-medium">Active Databases</span>.
                </div>
              ) : (
                <div className="rounded-lg border divide-y overflow-hidden">
                  {s.sources.map(src => {
                    const overridden = src in s.perSourceLimits;
                    const eff = s.perSourceLimits[src] ?? s.numPerSource;
                    return (
                      <div key={src} className="flex items-center justify-between gap-3 px-3 py-2.5 hover:bg-muted/40 transition-colors">
                        <div className="flex items-center gap-2 min-w-0">
                          <span className="size-2 rounded-full bg-primary/60 shrink-0" />
                          <span className="text-sm truncate">{src}</span>
                          {overridden && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded-full shrink-0 bg-primary/10 text-primary font-medium">
                              custom
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-1.5 shrink-0">
                          <NumberStepper value={eff} onChange={n => s.setPerSourceLimits(prev => ({ ...prev, [src]: n }))} />
                          <button type="button" title="Reset to default" disabled={!overridden}
                            onClick={() => s.setPerSourceLimits(prev => { const n = { ...prev }; delete n[src]; return n; })}
                            className={`p-1 rounded-md transition-colors ${overridden ? "text-muted-foreground hover:text-foreground hover:bg-muted" : "opacity-0 pointer-events-none"}`}>
                            <RotateCcw className="size-3.5" />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
              <p className="text-[11px] text-muted-foreground mt-1.5">Add or remove databases in the left sidebar.</p>
            </div>
            <div className="space-y-2">
              <label className="text-muted-foreground text-sm block">{frameworkOf(s.framework).label} elements</label>
              {frameworkOf(s.framework).elements.map(el => (
                <div key={el.id}>
                  <label className="text-[11px] text-muted-foreground">{el.label}</label>
                  <Textarea value={(s.pico as Record<string, string>)[el.id] || ""}
                    onChange={e => s.setPico({ ...s.pico, [el.id]: e.target.value })} rows={2} />
                </div>
              ))}
            </div>
            <div className="grid gap-4">
              <div>
                <label className="text-muted-foreground text-sm block mb-2">Inclusion criteria</label>
                <CriteriaList items={s.inclusion} onChange={s.setInclusion} placeholder="e.g., randomized controlled trials" variant="include" />
              </div>
              <div>
                <label className="text-muted-foreground text-sm block mb-2">Exclusion criteria</label>
                <CriteriaList items={s.exclusion} onChange={s.setExclusion} placeholder="e.g., animal studies" variant="exclude" />
              </div>
            </div>
          </div>
          <div className="border-t px-5 py-3 flex items-center justify-end gap-2 shrink-0">
            <Button variant="ghost" onClick={() => planResolverRef.current?.(null)}>Cancel</Button>
            <Button
              disabled={!s.query.trim() || s.sources.length === 0}
              onClick={() => planResolverRef.current?.({
                pico: s.pico, inclusion: s.inclusion, exclusion: s.exclusion,
                query: s.query, sources: s.sources, perSourceLimits: s.perSourceLimits, numPerSource: s.numPerSource,
              })}>
              <Send className="size-4 mr-1.5" />Run search
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* ── Strategy Review, centered main-page modal ────────────────── */}
      <Dialog open={reviewOpen} onOpenChange={setReviewOpen}>
        <DialogContent className="sm:max-w-2xl w-[95vw] h-[82vh] max-h-[720px] flex flex-col p-0 gap-0">
            <DialogHeader className="px-5 py-4 border-b shrink-0 text-left space-y-0.5">
              <DialogTitle className="text-base">Strategy Review</DialogTitle>
              <p className="text-xs text-muted-foreground m-0">Study design, {frameworkOf(s.framework).label}, criteria, search &amp; protocol</p>
            </DialogHeader>
            <Tabs defaultValue="design" className="flex-1 flex flex-col min-h-0">
              <div className="px-5 pt-3 shrink-0">
                <TabsList className="grid grid-cols-5 w-full">
                  <TabsTrigger value="design">Study Design</TabsTrigger>
                  <TabsTrigger value="pico">{frameworkOf(s.framework).label}</TabsTrigger>
                  <TabsTrigger value="criteria">Criteria</TabsTrigger>
                  <TabsTrigger value="search">Search</TabsTrigger>
                  <TabsTrigger value="relevance">Relevance</TabsTrigger>
                </TabsList>
              </div>
              <div className="flex-1 overflow-auto p-5">
                <TabsContent value="design" className="mt-0 space-y-2.5">
                  <div className="space-y-2.5">
                    {FRAMEWORK_IDS.map(fid => {
                      const meta = frameworkOf(fid);
                      const active = s.framework === fid;
                      return (
                        <button key={fid} type="button"
                          onClick={() => { fwOverrideRef.current = true; s.setFramework(fid); setClarifyFramework(fid); }}
                          aria-pressed={active}
                          className={`w-full text-left rounded-lg border p-3.5 transition-colors ${active ? "border-primary bg-primary/5 ring-1 ring-primary" : "hover:bg-muted"}`}>
                          <div className="flex items-center gap-2">
                            <span className={`flex items-center justify-center size-5 rounded text-xs font-bold ${active ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"}`}>
                              {active ? <Check className="size-3.5" /> : meta.label[0]}
                            </span>
                            <span className="text-sm font-semibold">{meta.label}</span>
                            <span className="text-xs text-muted-foreground">{meta.reviewType}</span>
                          </div>
                          <p className="text-xs text-muted-foreground mt-1.5">{meta.blurb}</p>
                          <div className="mt-2.5 space-y-1">
                            {meta.elements.map(e => (
                              <div key={e.id} className="text-[11px] text-muted-foreground">
                                <span className="font-semibold text-foreground">{e.label}:</span> {e.desc}
                              </div>
                            ))}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </TabsContent>
                <TabsContent value="pico" className="mt-0 space-y-3">
                  {/* Structure a free-form / custom request with the LLM instead of
                      taking it verbatim: fills the elements, question and search below. */}
                  <div className="rounded-lg border bg-muted/30 p-3 space-y-2">
                    <label className="text-xs font-medium text-foreground">Describe your review in plain language</label>
                    <Textarea value={picoDraft} onChange={e => setPicoDraft(e.target.value)} rows={2}
                      placeholder={`e.g. Does intermittent fasting improve HbA1c in adults with type 2 diabetes vs continuous calorie restriction?`} />
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-[11px] text-muted-foreground">Infers the {frameworkOf(s.framework).label} elements, research question and search string.</p>
                      <Button size="sm" className="h-8 gap-1.5 shrink-0" disabled={inferring || !picoDraft.trim()}
                        onClick={async () => { const ok = await inferStrategyFromText(picoDraft); if (ok) setPicoDraft(""); }}>
                        {inferring ? <Loader2 className="size-3.5 animate-spin" /> : <Sparkles className="size-3.5" />}Infer with AI
                      </Button>
                    </div>
                  </div>
                  {inferredQuestion && (
                    <p className="text-xs italic text-muted-foreground border-l-2 border-primary/40 pl-2.5 leading-snug">
                      <span className="not-italic font-medium text-foreground">Research question: </span>{inferredQuestion}
                    </p>
                  )}
                  {frameworkOf(s.framework).elements.map(el => (
                    <div key={el.id}>
                      <label className="text-muted-foreground text-sm">{el.label}</label>
                      <Textarea value={(s.pico as Record<string, string>)[el.id] || ""}
                        onChange={e => s.setPico({ ...s.pico, [el.id]: e.target.value })} rows={2} />
                    </div>
                  ))}
                </TabsContent>
                <TabsContent value="criteria" className="mt-0 space-y-4">
                  <div>
                    <label className="text-muted-foreground text-sm block mb-2">Inclusion Criteria</label>
                    <CriteriaList items={s.inclusion} onChange={s.setInclusion} placeholder="e.g., randomized controlled trials" variant="include" />
                  </div>
                  <div>
                    <label className="text-muted-foreground text-sm block mb-2">Exclusion Criteria</label>
                    <CriteriaList items={s.exclusion} onChange={s.setExclusion} placeholder="e.g., animal studies" variant="exclude" />
                  </div>
                </TabsContent>
                <TabsContent value="search" className="mt-0 space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <label className="text-muted-foreground text-sm">Databases &amp; per-database limits</label>
                      <span className="text-[11px] text-muted-foreground">
                        Total budget ≈ {s.sources.reduce((sum, src) => sum + (s.perSourceLimits[src] ?? s.numPerSource), 0).toLocaleString()} papers
                      </span>
                    </div>
                    <p className="text-[11px] text-muted-foreground mb-2.5">How many papers to pull from each active database. Rows use the default unless you set a specific cap.</p>

                    {/* Default control */}
                    <div className="flex items-center justify-between gap-3 rounded-lg border bg-muted/40 px-3 py-2.5 mb-2">
                      <div className="min-w-0">
                        <div className="text-xs font-semibold">Default</div>
                        <div className="text-[11px] text-muted-foreground">Applied to every database without its own cap</div>
                      </div>
                      <NumberStepper value={s.numPerSource} onChange={n => s.setNumPerSource(n)} />
                    </div>

                    {/* Per-database rows */}
                    {s.sources.length === 0 ? (
                      <div className="rounded-lg border border-dashed px-3 py-4 text-center text-[11px] text-muted-foreground">
                        No databases selected. Enable databases in the left sidebar under <span className="font-medium">Active Databases</span>.
                      </div>
                    ) : (
                      <div className="rounded-lg border divide-y overflow-hidden">
                        {s.sources.map(src => {
                          const overridden = src in s.perSourceLimits;
                          const eff = s.perSourceLimits[src] ?? s.numPerSource;
                          return (
                            <div key={src} className="flex items-center justify-between gap-3 px-3 py-2.5 hover:bg-muted/40 transition-colors">
                              <div className="flex items-center gap-2 min-w-0">
                                <span className="size-2 rounded-full bg-primary/60 shrink-0" />
                                <span className="text-sm truncate">{src}</span>
                                <span className={`text-[10px] px-1.5 py-0.5 rounded-full shrink-0 ${overridden ? "bg-primary/10 text-primary font-medium" : "text-muted-foreground"}`}>
                                  {overridden ? "custom" : "default"}
                                </span>
                              </div>
                              <div className="flex items-center gap-1.5 shrink-0">
                                <NumberStepper value={eff}
                                  onChange={n => s.setPerSourceLimits(prev => ({ ...prev, [src]: n }))} />
                                <button type="button" title="Reset to default"
                                  disabled={!overridden}
                                  onClick={() => s.setPerSourceLimits(prev => { const n = { ...prev }; delete n[src]; return n; })}
                                  className={`p-1 rounded-md transition-colors ${overridden ? "text-muted-foreground hover:text-foreground hover:bg-muted" : "opacity-0 pointer-events-none"}`}>
                                  <RotateCcw className="size-3.5" />
                                </button>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                  <div>
                    <div className="flex items-center justify-between gap-2 mb-1.5">
                      <label className="text-muted-foreground text-sm">Final Search String</label>
                      <Button size="sm" variant="ghost" className="h-7 gap-1.5 text-xs px-2 shrink-0"
                        onClick={rebuildQueryFromPico} disabled={inferring}
                        title="Regenerate the query from your elements with the LLM instead of hand-editing">
                        {inferring ? <Loader2 className="size-3.5 animate-spin" /> : <Sparkles className="size-3.5" />}Rebuild with AI
                      </Button>
                    </div>
                    <Textarea value={s.query} onChange={e => { s.setQuery(e.target.value); s.setUnifiedSearchQuery(e.target.value); }} rows={7} className="font-mono text-xs" />
                  </div>
                </TabsContent>
                <TabsContent value="relevance" className="mt-0 space-y-3">
                  <p className="text-xs text-muted-foreground">
                    Pick relevant articles then generate a per-source structured summary.
                  </p>
                  <div>
                    <div className="text-xs font-semibold text-muted-foreground mb-1.5">Pulled articles ({(s.rerankResults?.ranked || []).length})</div>
                    {(s.rerankResults?.ranked || []).length === 0 ? (
                      <p className="text-xs text-muted-foreground">Run a search first to score articles for relevance.</p>
                    ) : (
                      <div className="rounded-md border overflow-hidden">
                        <div className="flex items-center gap-1.5 px-2 border-b bg-muted/30">
                          <Search className="size-3.5 text-muted-foreground shrink-0" />
                          <input value={relSearch} onChange={e => setRelSearch(e.target.value)} placeholder="Search articles…"
                            className="flex-1 bg-transparent py-2 text-xs outline-none placeholder:text-muted-foreground min-w-0" />
                          {relSearch && (
                            <button type="button" onClick={() => setRelSearch("")} className="text-muted-foreground hover:text-foreground shrink-0" title="Clear search"><X className="size-3.5" /></button>
                          )}
                        </div>
                        <div className="max-h-[45vh] overflow-auto divide-y">
                          {(() => {
                            const shown = (s.rerankResults?.ranked || []).filter(r => !relSearch.trim() || (r.paper.title || "").toLowerCase().includes(relSearch.trim().toLowerCase()));
                            return shown.length === 0 ? (
                              <p className="text-xs text-muted-foreground p-3">No articles match “{relSearch}”.</p>
                            ) : shown.map(r => (
                              <label key={r.paper.id} className="flex items-start gap-2 px-2 py-1.5 hover:bg-muted/40 cursor-pointer">
                                <Checkbox checked={selectedSources.has(r.paper.id)} onCheckedChange={() => toggleSource(r.paper.id)} className="mt-0.5" />
                                <span className="flex-1 min-w-0">
                                  <span className="block text-xs line-clamp-2 leading-snug">{r.paper.title}</span>
                                  <span className="block text-[10px] text-muted-foreground">relevance {r.leads_score.toFixed(2)}</span>
                                </span>
                              </label>
                            ));
                          })()}
                        </div>
                      </div>
                    )}
                  </div>
                  {uploadedRaw.length > 0 && (
                    <div className="border-t pt-3">
                      <div className="text-xs font-semibold text-muted-foreground mb-1.5">Uploaded documents ({uploadedRaw.length})</div>
                      <div className="max-h-[45vh] overflow-auto rounded-md border divide-y">
                        {uploadedRaw.map(p => (
                          <label key={p.id} className="flex items-start gap-2 px-2 py-1.5 hover:bg-muted/40 cursor-pointer">
                            <Checkbox checked={selectedSources.has(p.id)} onCheckedChange={() => toggleSource(p.id)} className="mt-0.5" />
                            <span className="flex-1 min-w-0 text-xs line-clamp-2 leading-snug">{p.title}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  )}
                  <Button className="w-full" onClick={runStructuredSummary} disabled={summarizing || selectedSources.size === 0}>
                    {summarizing ? <><Loader2 className="size-4 mr-2 animate-spin" />Summarizing…</> : <><Sparkles className="size-4 mr-2" />Structured summary ({selectedSources.size})</>}
                  </Button>
                </TabsContent>
              </div>
            </Tabs>
        </DialogContent>
      </Dialog>

      {/* Chat input, fixed to bottom, matching content width */}
      <div className="fixed bottom-0 left-72 right-0 z-30 px-6 py-4 pointer-events-none transition-all">
        <div className="max-w-4xl mx-auto pointer-events-auto">
          <input ref={attachRef} type="file" multiple
            accept={ACCEPT_ATTR} className="hidden"
            onChange={e => { if (e.target.files?.length) studyImport.importFiles(Array.from(e.target.files)); e.currentTarget.value = ""; }} />
          {/* Freshly-attached documents that will be included with the next message. */}
          {pendingDocs.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-2 px-1">
              {pendingDocs.map(d => (
                <span key={d.id} className="inline-flex items-center gap-1.5 max-w-[16rem] bg-card border border-border shadow-sm rounded-full pl-2.5 pr-1 py-1 text-xs" title={`${d.title} — included in your next message`}>
                  <Paperclip className="size-3 shrink-0 text-primary" />
                  <span className="truncate text-foreground/90">{d.title}</span>
                  <button type="button" onClick={() => setPendingDocIds(prev => prev.filter(x => x !== d.id))}
                    className="shrink-0 rounded-full hover:bg-muted p-0.5 text-muted-foreground hover:text-foreground" title="Don't include this in the next message">
                    <X className="size-3" />
                  </button>
                </span>
              ))}
            </div>
          )}
          {/* Chat composer: input on top, controls in a row at the bottom. The
              Study-design & strategy opener lives inline in the composer (bottom
              left), alongside attach and deep-scan, rather than as a bar above. */}
          <form onSubmit={(e) => { e.preventDefault(); handleSubmit(input); }}
            className="bg-card/95 backdrop-blur border rounded-2xl shadow-lg px-3 pt-2.5 pb-2">
            <Input value={input} onChange={e => setInput(e.target.value)}
              placeholder="Ask a question or refine your research goal..."
              className="w-full border-0 bg-transparent shadow-none focus-visible:ring-0 px-1 mb-2" />
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-1.5 min-w-0">
                <Button type="button" size="icon" variant="ghost" className="rounded-full shrink-0 size-8"
                  onClick={() => attachRef.current?.click()} disabled={studyImport.busy}
                  title="Attach files (PDF, Word, Excel/CSV, RIS/BibTeX) to review">
                  {studyImport.busy ? <Loader2 className="size-4 animate-spin" /> : <Plus className="size-4" />}
                </Button>
                {studyImport.uploadedCount > 0 && (
                  <Button type="button" size="sm" variant="ghost" className="rounded-full shrink-0 gap-1.5 px-2.5 h-8"
                    onClick={() => setAttachOpen(true)} title="View & preview attached studies">
                    <Paperclip className="size-3.5" />{studyImport.uploadedCount}
                  </Button>
                )}
                <button type="button" onClick={() => setReviewOpen(true)}
                  title="Study design, frame, eligibility criteria, search string & protocol"
                  className="inline-flex items-center gap-1.5 px-2.5 h-8 rounded-full border bg-card text-xs font-medium text-foreground hover:bg-muted transition-colors shrink-0">
                  <SlidersHorizontal className="size-3.5" />
                  <span className="truncate">{s.history.length > 0 ? "Study design & strategy" : "Study design"}</span>
                  <span className="ml-0.5 rounded-full bg-primary/10 text-primary px-1.5 py-0.5 text-[10px] font-semibold shrink-0">{frameworkOf(s.framework).label}</span>
                </button>
              </div>
              <div className="flex items-center gap-1.5 shrink-0">
                <Button type="submit" size="sm" disabled={analyzing || !input.trim()} className="rounded-full h-8"><Send className="size-4 mr-1.5" />Send</Button>
              </div>
            </div>
          </form>
        </div>
      </div>

      <AttachedStudies open={attachOpen} onOpenChange={setAttachOpen} studyImport={studyImport} />

      {/* Preview of a cited source document — the actual file (PDF / text) when it
          can be rendered inline, otherwise the extracted text with a download. */}
      <Dialog open={!!previewDoc} onOpenChange={o => { if (!o) setPreviewDoc(null); }}>
        <DialogContent className="sm:max-w-6xl w-[95vw] h-[92vh] flex flex-col p-5 gap-3">
          <DialogHeader className="pr-8 shrink-0">
            <DialogTitle className="text-base leading-snug">{previewDoc?.title}</DialogTitle>
          </DialogHeader>
          <div className="flex items-center justify-between gap-2 shrink-0">
            {previewDoc?.fileInline ? (
              <div className="inline-flex rounded-md border p-0.5 text-xs">
                {(["doc", "text"] as const).map(mode => (
                  <button key={mode} type="button" onClick={() => setPreviewMode(mode)}
                    className={`px-2.5 h-6 rounded font-medium transition-colors ${previewMode === mode ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"}`}>
                    {mode === "doc" ? "Document" : "Text"}
                  </button>
                ))}
              </div>
            ) : (
              <span className="text-xs text-muted-foreground">
                {previewDoc?.fileUrl ? "Extracted text — original can't render inline" : "Extracted text"}
              </span>
            )}
            <div className="flex items-center gap-3 shrink-0">
              {/* No original captured for this uploaded doc (added before it was
                  stored) — let the user re-attach it to see the real document. */}
              {previewDoc?.uploaded && !previewDoc?.fileUrl && !previewDoc?.html && (
                <button type="button" onClick={() => previewDoc && promptReattach(previewDoc.id)}
                  className="text-xs text-primary hover:underline font-medium">Show original document ↥</button>
              )}
              {previewDoc?.fileUrl && (
                <a href={previewDoc.fileUrl} download={previewDoc.title} className="text-xs text-primary hover:underline">Download original ↧</a>
              )}
              {previewDoc?.url && /^https?:\/\//.test(previewDoc.url) && (
                <a href={previewDoc.url} target="_blank" rel="noreferrer" className="text-xs text-primary hover:underline">Open source ↗</a>
              )}
            </div>
          </div>
          {previewDoc?.fileInline && previewMode === "doc" ? (
            previewDoc.html ? (
              <div className="flex-1 overflow-auto rounded-md border bg-white">
                <div className="doc-preview mx-auto max-w-3xl px-10 py-8 text-[15px] leading-relaxed text-neutral-800"
                  dangerouslySetInnerHTML={{ __html: previewDoc.html }} />
              </div>
            ) : (
              <iframe title="Document preview" src={previewDoc.fileUrl} className="flex-1 w-full rounded-md border bg-white" />
            )
          ) : (
            <div className="flex-1 min-h-0 flex flex-col gap-2">
              {previewDoc?.uploaded && !previewDoc?.fileUrl && !previewDoc?.html && (
                <div className="shrink-0 text-xs text-muted-foreground bg-muted/50 border rounded-md px-3 py-2">
                  The original file for this document was not stored (it was added before file preview was available).
                  <button type="button" onClick={() => previewDoc && promptReattach(previewDoc.id)}
                    className="ml-1 text-primary hover:underline font-medium">Show original document</button> to view it as the real file.
                </div>
              )}
              <div className="overflow-auto text-sm whitespace-pre-wrap leading-relaxed text-foreground/90 flex-1 rounded-md border p-4">
                {previewDoc?.text}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Hidden picker used to re-attach an uploaded doc's original file. */}
      <input ref={reattachRef} type="file" accept={ACCEPT_ATTR} className="hidden"
        onChange={e => onReattachPicked(e.target.files?.[0])} />
    </div>
  );
}
