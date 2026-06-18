import { useCallback, useEffect, useRef, useState } from "react";
import { useStore, HistoryEntry } from "../lib/store";
import { AIService, DataAggregator } from "../lib/mockServices";
import type { ClarifyingQuestion } from "../lib/mockServices";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Textarea } from "../components/ui/textarea";
import { Separator } from "../components/ui/separator";
import { PicoCards } from "../components/PicoCards";
import { AnalysisProgress, Stage, StageId } from "../components/AnalysisProgress";
import { FormattedText } from "../lib/formattedText";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../components/ui/collapsible";
import { Sparkles, Send, ChevronDown, X, Plus, Wand2, Check, Lightbulb, SlidersHorizontal, Copy, RotateCcw } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { toast } from "sonner";

function ReferencesBySource({ refs, idPrefix }: { refs: { title: string; url: string; source: string; id: string }[]; idPrefix?: string }) {
  // The backend re-orders papers so that papers from the same source are
  // contiguous, with [N] citation markers in the summary matching this order.
  // We render the same sequence here under source headings — within each
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
      <pre className="bg-muted rounded-md p-3 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-relaxed">{value}</pre>
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
              className="text-primary font-medium hover:underline"
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

function SummaryText({ text, onCite }: { text: string; onCite?: (n: number) => void }) {
  // Split into sections on recognised headers, preserve bullets and paragraph breaks.
  const HEADERS = [
    "Research landscape overview",
    "Arguments supporting the research question",
    "Arguments against or challenging the research question",
  ];
  const lines = text.replace(/\r\n/g, "\n").split("\n");
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
  { id: "pico", label: "Infer PICO framework", status: "pending" },
  { id: "query", label: "Generate MeSH search string", status: "pending" },
  { id: "papers", label: "Fetch an initial sample of papers", status: "pending" },
  { id: "rerank", label: "Score papers for relevance (LEADS)", status: "pending" },
  { id: "question", label: "Draft formal research question", status: "pending" },
  { id: "summary", label: "Summarize the literature found", status: "pending" },
  { id: "suggestions", label: "Suggest refinements", status: "pending" },
  { id: "adversarial", label: "Build adversarial search", status: "pending" },
];

// Default LEADS aggregate score threshold for the pre-summary relevance filter.
function scoreBadgeClass(score: number, threshold: number): string {
  if (score >= 0.5) return "bg-emerald-100 text-emerald-800";
  if (score >= threshold) return "bg-emerald-50 text-emerald-700";
  if (score >= threshold - 0.3) return "bg-amber-50 text-amber-700";
  return "bg-rose-50 text-rose-700";
}

const PICO_FIELDS = ["population", "intervention", "comparator", "outcome"] as const;
const PICO_LABEL: Record<string, string> = { population: "P", intervention: "I", comparator: "C", outcome: "O" };

// Conversational PICO clarifier modal. Fetches one question at a time from the
// backend until all PICO elements are SR-ready, then calls onDone. Each question
// has exactly 3 specific suggestions + 1 blank fill-in.
function ClarifyingQuestionsModal({
  open,
  goal,
  onDone,
  onSkipAll,
}: {
  open: boolean;
  goal: string;
  onDone: (answers: Record<string, string>) => void;
  onSkipAll: () => void;
}) {
  const [question, setQuestion] = useState<ClarifyingQuestion | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [round, setRound] = useState(0);
  const [loading, setLoading] = useState(false);
  const [freeText, setFreeText] = useState("");
  const freeRef = useRef<HTMLInputElement>(null);
  // PICO element ids already asked — at most one question per element, no repeats.
  const askedRef = useRef<Set<string>>(new Set());

  const fetchNext = useCallback(async (current: Record<string, string>, r: number) => {
    setLoading(true);
    setFreeText("");
    try {
      const result = await AIService.getClarifyNext(goal, current, r, Array.from(askedRef.current));
      // Stop if done, no question, or the model circled back to an element we
      // already asked about.
      if (result.done || !result.question || askedRef.current.has(result.question.id)) {
        onDone(current);
      } else {
        askedRef.current.add(result.question.id);
        setQuestion(result.question);
      }
    } catch {
      onDone(current);
    } finally {
      setLoading(false);
    }
  }, [goal, onDone]);

  useEffect(() => {
    if (open && goal) {
      setAnswers({});
      setQuestion(null);
      setRound(0);
      setFreeText("");
      askedRef.current = new Set();
      fetchNext({}, 0);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, goal]);

  function pick(value: string) {
    if (!question || loading) return;
    const next = { ...answers, [question.id]: value };
    const nextRound = round + 1;
    setAnswers(next);
    setRound(nextRound);
    fetchNext(next, nextRound);
  }

  function submitFreeText() {
    const v = freeText.trim();
    if (v) pick(v);
  }

  if (!open) return null;

  const showSpinner = loading || !question;

  return (
    <div className="fixed bottom-20 left-72 right-0 z-40 px-6 pointer-events-none">
      <div className="max-w-4xl mx-auto pointer-events-auto">
        <Card className="border-primary/40 shadow-xl bg-card/98 backdrop-blur overflow-hidden">

          {/* Header */}
          <div className="flex items-center justify-between gap-3 px-4 pt-3 pb-2">
            <div className="flex items-center gap-2 min-w-0">
              <Lightbulb className="size-4 text-primary shrink-0" />
              <div className="text-sm font-medium break-words">
                {showSpinner ? "Checking your PICO elements…" : question!.title}
              </div>
            </div>
            {/* PICO field progress pills */}
            <div className="flex items-center gap-1.5 shrink-0">
              {PICO_FIELDS.map(f => {
                const done = !!answers[f];
                const active = !done && question?.id === f;
                return (
                  <span
                    key={f}
                    className={`text-[10px] font-bold px-1.5 py-0.5 rounded-sm border transition-colors ${
                      done
                        ? "bg-emerald-100 text-emerald-700 border-emerald-300"
                        : active
                        ? "bg-primary text-primary-foreground border-primary"
                        : "bg-muted text-muted-foreground border-border"
                    }`}
                    title={f}
                  >
                    {PICO_LABEL[f]}
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
              {question!.options.slice(0, 3).map((opt, i) => (
                <button
                  key={opt.id}
                  onClick={() => pick(opt.label)}
                  className="w-full text-left px-3 py-2.5 rounded-md border bg-card hover:bg-accent hover:border-primary/40 transition-colors flex items-center gap-3"
                >
                  <span className="shrink-0 size-5 rounded-sm border bg-muted text-muted-foreground text-[11px] flex items-center justify-center tabular-nums font-mono">
                    {i + 1}
                  </span>
                  <span className="text-sm flex-1 leading-snug">{opt.label}</span>
                </button>
              ))}

              {/* Blank fill-in */}
              <div className="flex items-center gap-2 mt-1 px-3 py-1.5 rounded-md border border-dashed bg-muted/20">
                <Wand2 className="size-3.5 text-muted-foreground shrink-0" />
                <Input
                  ref={freeRef}
                  value={freeText}
                  onChange={e => setFreeText(e.target.value)}
                  placeholder="Other: describe your own…"
                  className="border-0 bg-transparent shadow-none focus-visible:ring-0 px-0 h-7 text-sm"
                  onKeyDown={e => { if (e.key === "Enter") { e.preventDefault(); submitFreeText(); } }}
                />
                {freeText.trim() && (
                  <Button size="sm" onClick={submitFreeText} className="rounded-full h-7 px-3 shrink-0">
                    Use this
                  </Button>
                )}
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

function RankRow({ d, threshold }: { d: any; threshold: number }) {
  const score = d.leads_score as number;
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
      </div>
    </li>
  );
}

function RelevanceExplorer() {
  const s = useStore();
  const r = s.rerankResults;
  if (!r) return null;
  // Use the auto-cutoff the rerank actually applied (effective_floor), not the
  // legacy store threshold — the slider that used to drive it has been removed.
  const threshold = typeof r.effective_floor === "number" ? r.effective_floor : r.threshold;
  const kept = r.ranked.filter(x => x.leads_score >= threshold);
  const dropped = r.ranked.filter(x => x.leads_score < threshold);
  const fmt = (n: number) => `${n >= 0 ? "+" : ""}${n.toFixed(2)}`;

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
      </div>

      {kept.length > 0 && (
        <section>
          <div className="text-sm font-semibold text-foreground mb-1.5">
            Kept — above threshold (used for the summary)
          </div>
          <div className="rounded-md border bg-muted/20 max-h-72 overflow-auto p-1">
            <ul className="space-y-0.5">
              {kept.map((d, i) => <RankRow key={(d.paper as any).id || i} d={d} threshold={threshold} />)}
            </ul>
          </div>
        </section>
      )}

      <Separator />

      {dropped.length === 0 ? (
        <p className="text-sm text-muted-foreground">No papers were dropped at this threshold.</p>
      ) : (
        <section>
          <div className="text-sm font-semibold text-foreground mb-1.5">
            Dropped — below threshold
          </div>
          <div className="rounded-md border bg-muted/20 max-h-72 overflow-auto p-1">
            <ul className="space-y-0.5">
              {dropped.map((d, i) => <RankRow key={(d.paper as any).id || i} d={d} threshold={threshold} />)}
            </ul>
          </div>
        </section>
      )}
    </div>
  );
}

export function HomePage() {
  const s = useStore();
  const [input, setInput] = useState("");
  const [refining, setRefining] = useState(false);
  const [refinement, setRefinement] = useState<null | {
    field: "population" | "intervention" | "comparator" | "outcome";
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
  const [reviewOpen, setReviewOpen] = useState(false);
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

  async function handleSubmit(text: string, opts: { skipClarify?: boolean } = {}) {
    const t = text.trim();
    if (!t) return;
    setInput("");

    // 0. Ask the user to disambiguate underspecified PICO elements BEFORE the
    //    search runs. The system used to silently infer these, which caused
    //    catastrophic drift (e.g. "Mediterranean diet → longevity" becoming
    //    "low-carb diet → BMI in overweight 18-65"). Now the user owns the
    //    answer through a Claude-style multi-question popup. If the model
    //    returns no clarifying questions, this is a no-op.
    //
    //    Skipped when handleSubmit is called from applyRefinement — the
    //    refinement IS the clarification, no need to ask again.
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
    const effectiveText = clarifyExtras ? `${t}\n\nFurther context — ${clarifyExtras}` : t;

    const { abort } = s.startTask("home-analysis", INITIAL_STAGES.map(st => ({ ...st, status: "pending" as const })));
    const signal = abort.signal;

    try {
      // 1. PICO inference. When a strategy already exists, treat this message as
      //    a REFINEMENT of the CURRENT active strategy (the one shown in the
      //    Strategy Review drawer — includes any edits / version reverts) so the
      //    operationalised detail is preserved instead of regenerated.
      const prior = s.history.length > 0
        ? { p: s.pico.population, i: s.pico.intervention, c: s.pico.comparator, o: s.pico.outcome,
            inclusion: s.inclusion, exclusion: s.exclusion }
        : null;
      const analysis = await runStage("pico", signal, sig => AIService.inferPicoAndQuery(effectiveText, prior, sig));
      if (!analysis) { s.updateTask("home-analysis", { status: signal.aborted ? "canceled" : "error" }); return; }

      const newPico = { population: analysis.p, intervention: analysis.i, comparator: analysis.c, outcome: analysis.o };
      s.setPico(newPico);
      s.setInclusion(analysis.inclusion);
      s.setExclusion(analysis.exclusion);
      s.setQuery(analysis.query);
      s.setUnifiedSearchQuery(analysis.query);
      markStage("query", { status: "done", detail: analysis.query ? analysis.query.slice(0, 60) + "…" : undefined });

      // 2. Fetch a wider sample of papers so the relevance filter has room to pick from.
      const fetched = await runStage("papers", signal, sig =>
        DataAggregator.fetchAll(analysis.query, s.sources, newPico, undefined, sig, s.elsevierToken, s.ezproxyConnected)
      );
      const papers = fetched?.papers || [];
      if (fetched) {
        const breakdown = Object.entries(fetched.sourceCounts || {})
          .map(([k, v]) => `${k}: ${v}`)
          .join(" · ");
        markStage("papers", { status: "done", detail: `${papers.length} papers — ${breakdown}` });
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
        const reranked = await runStage("rerank", signal, sig =>
          DataAggregator.rerankByRelevance(
            papers,
            newPico,
            analysis.inclusion,
            analysis.exclusion,
            -1.0,        // disabled — auto mode supersedes
            undefined,
            sig,
          )
        );
        if (reranked) {
          relevantPapers = reranked.kept.map(r => r.paper);
          s.setRerankResults(reranked);
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
        markStage("rerank", { status: "done", detail: "no papers to score" });
      }

      const formalQ = await runStage("question", signal, sig => AIService.generateFormalQuestion(newPico, sig));
      const summaryWithRefs = await runStage("summary", signal, sig => AIService.generateComprehensiveSummaryWithRefs(t, relevantPapers, sig));
      const suggs = await runStage("suggestions", signal, sig => AIService.getRefinementSuggestions(t, relevantPapers, sig));
      const adv = await runStage("adversarial", signal, sig => AIService.generateAdversarialQuery(newPico, sig));

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
    // because the refinement itself IS a clarification — re-prompting would
    // be circular.
    const field = refinement.field;
    const suggested = refinement.suggested;
    s.setPico(p => ({ ...p, [field]: suggested }));
    const baseGoal = (last?.goal || "").trim();
    setRefinement(null);
    if (baseGoal) {
      const augmented = `${baseGoal}\n\nFurther context — ${field}: ${suggested}`;
      // Fire-and-forget — handleSubmit manages its own task lifecycle.
      void handleSubmit(augmented, { skipClarify: true });
    }
  }

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {s.history.length === 0 && !analyzing && (
        <Alert><AlertDescription>👋 Welcome! Describe your research goal below to generate a strategy and see initial findings.</AlertDescription></Alert>
      )}

      {s.history.map((entry, idx) => (
        <div key={idx} className="space-y-3">
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
                    ["pico", "PICO"],
                    ["criteria", "Criteria"],
                    ["search", "Search"],
                    ...(idx === s.history.length - 1 && s.rerankResults ? [["relevance", "Relevance"]] : []),
                  ].map(([value, label]) => (
                    <TabsTrigger
                      key={value}
                      value={value}
                      className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-sm"
                    >
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
              <div className="p-5">
                <TabsContent value="overview" className="mt-0">
                  <OverviewTab entry={entry} idx={idx} />
                </TabsContent>

                <TabsContent value="pico" className="mt-0">
                  <PicoCards pico={entry.pico_dict} />
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

                {/* Relevance-rerank explorer — only for the most recent run,
                    since rerankResults holds only the latest LEADS pass. */}
                {idx === s.history.length - 1 && s.rerankResults && (
                  <TabsContent value="relevance" className="mt-0">
                    <RelevanceExplorer />
                  </TabsContent>
                )}
              </div>
            </Tabs>
          </Card>
        </div>
      ))}

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

      {/* Clarifying-questions modal — opens BEFORE the search runs whenever the
          user's goal is under-specified. Pauses handleSubmit until the user
          answers or skips, then resumes with the answers folded into the
          effective goal text. */}
      <ClarifyingQuestionsModal
        open={clarifyOpen}
        goal={clarifyGoal}
        onDone={(answers) => clarifyResolverRef.current?.(answers)}
        onSkipAll={() => clarifyResolverRef.current?.({})}
      />

      {/* Refinement popup — floats above the chat input, Claude-clarifying-question style */}
      {(refining || refinement) && s.history.length > 0 && (
        <div className={`fixed bottom-20 left-72 z-30 px-6 pointer-events-none transition-all ${reviewOpen ? "right-[400px]" : "right-0"}`}>
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
                      setInput(prev => (prev ? `${prev} — ${refinement.suggested}` : refinement.suggested));
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

      {/* ── Strategy Review — collapsible right-hand drawer ────────────────── */}
      {s.history.length > 0 && (
        <>
          {/* Floating opener — clearly visible, clear of the scrollbar + chat bar */}
          {!reviewOpen && (
            <button
              onClick={() => setReviewOpen(true)}
              className="fixed right-5 bottom-28 z-40 inline-flex items-center gap-2 rounded-full bg-primary text-primary-foreground shadow-lg px-4 py-2.5 text-sm font-medium hover:opacity-95 transition-opacity"
              title="Edit PICO, criteria & search string"
            >
              <SlidersHorizontal className="size-4" />Strategy Review
            </button>
          )}
          {/* Drawer panel */}
          <div
            className={`fixed top-0 right-0 h-screen w-[400px] max-w-[92vw] bg-card border-l shadow-2xl z-40 flex flex-col transition-transform duration-200 ${reviewOpen ? "translate-x-0" : "translate-x-full"}`}
            aria-hidden={!reviewOpen}
          >
            <div className="flex items-center justify-between px-4 py-3 border-b shrink-0">
              <div>
                <h2 className="m-0 text-base">Strategy Review</h2>
                <span className="text-xs text-muted-foreground">Edit PICO, criteria &amp; search string</span>
              </div>
              <button onClick={() => setReviewOpen(false)} className="text-muted-foreground hover:text-foreground" title="Close">
                <X className="size-4" />
              </button>
            </div>
            <Tabs defaultValue="pico" className="flex-1 flex flex-col min-h-0">
              <div className="px-4 pt-3 shrink-0">
                <TabsList className="grid grid-cols-3 w-full">
                  <TabsTrigger value="pico">PICO</TabsTrigger>
                  <TabsTrigger value="criteria">Criteria</TabsTrigger>
                  <TabsTrigger value="search">Search</TabsTrigger>
                </TabsList>
              </div>
              <div className="flex-1 overflow-auto p-4">
                <TabsContent value="pico" className="mt-0 space-y-3">
                  {(["population", "intervention", "comparator", "outcome"] as const).map(f => (
                    <div key={f}>
                      <label className="text-muted-foreground text-sm capitalize">{f}</label>
                      <Textarea value={s.pico[f]} onChange={e => s.setPico({ ...s.pico, [f]: e.target.value })} rows={2} />
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
                <TabsContent value="search" className="mt-0 space-y-2">
                  <label className="text-muted-foreground text-sm">Final Search String</label>
                  <Textarea value={s.query} onChange={e => { s.setQuery(e.target.value); s.setUnifiedSearchQuery(e.target.value); }} rows={8} className="font-mono text-xs" />
                  <p className="text-xs text-muted-foreground">Edits here also update the per-database queries on the Planning page.</p>
                </TabsContent>
              </div>
            </Tabs>
          </div>
        </>
      )}

      {/* Chat input — fixed to bottom, matching content width */}
      <div className={`fixed bottom-0 left-72 z-30 px-6 py-4 pointer-events-none transition-all ${reviewOpen ? "right-[400px]" : "right-0"}`}>
        <form onSubmit={(e) => { e.preventDefault(); handleSubmit(input); }}
          className="max-w-4xl mx-auto flex gap-2 items-center bg-card/95 backdrop-blur border rounded-full shadow-lg pl-5 pr-2 py-2 pointer-events-auto">
          <Input value={input} onChange={e => setInput(e.target.value)}
            placeholder="Ask a question or refine your research goal..."
            className="flex-1 border-0 bg-transparent shadow-none focus-visible:ring-0 px-0" />
          {s.history.length > 0 && (
            <Button
              type="button"
              size="sm"
              variant="ghost"
              onClick={suggestRefinement}
              disabled={refining || analyzing}
              className="rounded-full"
              title="Get a clarifying question to refine your search"
            >
              <Wand2 className="size-4 mr-1.5" />{refining ? "…" : "Refine"}
            </Button>
          )}
          <Button type="submit" disabled={analyzing || !input.trim()} className="rounded-full"><Send className="size-4 mr-2" />Send</Button>
        </form>
      </div>
    </div>
  );
}
