import { useEffect, useRef, useState } from "react";
import { useStore } from "../lib/store";
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
import { Sparkles, Send, ChevronDown, X, Plus, Wand2, Check, Lightbulb } from "lucide-react";

function ReferencesBySource({ refs }: { refs: { title: string; url: string; source: string; id: string }[] }) {
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
              <li key={r.id || r.n} className="flex gap-2">
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

function SummaryText({ text }: { text: string }) {
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
                    <li key={i}>{b.replace(/^\s*[-*•]\s+/, "")}</li>
                  ))}
                </ul>
              ) : (
                sec.lines
                  .join("\n")
                  .split(/\n{2,}/)
                  .map((para, i) => (
                    <p key={i} className="mb-2 last:mb-0 whitespace-pre-wrap">
                      {para.trim()}
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

// Claude-style multi-question popup, anchored directly above the chat bar
// (not a centered Dialog). Walks the user through 1-3 clarifying questions one
// at a time with numbered option chips, a "something else" free-text row, and
// Back / Skip controls. Returns a flat answers map.
function ClarifyingQuestionsModal({
  open,
  questions,
  onCommit,
  onSkipAll,
}: {
  open: boolean;
  questions: ClarifyingQuestion[];
  onCommit: (answers: Record<string, string>) => void;
  onSkipAll: () => void;
}) {
  const [idx, setIdx] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [freeText, setFreeText] = useState("");

  useEffect(() => {
    if (open) {
      setIdx(0);
      setAnswers({});
      setFreeText("");
    }
  }, [open]);

  if (!open || questions.length === 0) return null;
  const q = questions[idx];
  const isLast = idx >= questions.length - 1;

  function pick(value: string) {
    const next = { ...answers, [q.id]: value };
    setAnswers(next);
    setFreeText("");
    if (isLast) {
      onCommit(next);
    } else {
      setIdx(idx + 1);
    }
  }

  function submitFreeText() {
    const v = freeText.trim();
    if (!v) return;
    pick(v);
  }

  function skipQ() {
    if (isLast) {
      onCommit(answers);
    } else {
      setIdx(idx + 1);
      setFreeText("");
    }
  }

  return (
    <div className="fixed bottom-20 left-72 right-0 z-40 px-6 pointer-events-none">
      <div className="max-w-4xl mx-auto pointer-events-auto">
        <Card className="border-primary/40 shadow-xl bg-card/98 backdrop-blur overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between gap-3 px-4 pt-3 pb-2">
            <div className="flex items-center gap-2 min-w-0">
              <Lightbulb className="size-4 text-primary shrink-0" />
              <div className="text-sm font-medium break-words">{q.title}</div>
            </div>
            <div className="flex items-center gap-3 shrink-0">
              {questions.length > 1 && (
                <div className="text-xs text-muted-foreground tabular-nums">
                  {idx + 1} of {questions.length}
                </div>
              )}
              <button
                onClick={onSkipAll}
                className="text-muted-foreground hover:text-foreground"
                aria-label="Skip all"
              >
                <X className="size-4" />
              </button>
            </div>
          </div>

          {/* Options */}
          <div className="px-4 pb-2 space-y-1.5">
            {q.options.map((opt, i) => (
              <button
                key={opt.id}
                onClick={() => pick(opt.label)}
                className="w-full text-left px-3 py-2 rounded-md border bg-card hover:bg-accent hover:border-primary/30 transition-colors flex items-center gap-3"
              >
                <span className="shrink-0 size-5 rounded-sm border bg-muted text-muted-foreground text-[11px] flex items-center justify-center tabular-nums">
                  {i + 1}
                </span>
                <span className="text-sm flex-1">{opt.label}</span>
              </button>
            ))}

            {/* Free-text "Something else" row */}
            <div className="flex items-center gap-2 mt-1 px-3 py-1.5 rounded-md border border-dashed">
              <Wand2 className="size-3.5 text-muted-foreground shrink-0" />
              <Input
                value={freeText}
                onChange={(e) => setFreeText(e.target.value)}
                placeholder="Something else…"
                className="border-0 bg-transparent shadow-none focus-visible:ring-0 px-0 h-7 text-sm"
                onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); submitFreeText(); } }}
              />
              {freeText.trim() && (
                <Button size="sm" onClick={submitFreeText} className="rounded-full h-7 px-3">
                  {isLast ? "Done" : "Next"}
                </Button>
              )}
            </div>
          </div>

          {/* Footer controls */}
          <div className="flex items-center justify-between px-4 py-2 border-t bg-muted/30">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIdx(Math.max(0, idx - 1))}
              disabled={idx === 0}
              className="h-7"
            >
              Back
            </Button>
            <div className="flex gap-1">
              <Button variant="ghost" size="sm" onClick={skipQ} className="h-7">
                Skip
              </Button>
              <Button variant="ghost" size="sm" onClick={onSkipAll} className="h-7">
                Skip all
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </div>
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

  return (
    <Collapsible>
      <CollapsibleTrigger asChild>
        <Button variant="outline" size="sm" className="w-full justify-between mt-3">
          <span>Relevance rerank · {kept.length} kept / {dropped.length} dropped</span>
          <ChevronDown className="size-4" />
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent className="pt-4 space-y-4">
        {kept.length > 0 && (
          <section>
            <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
              Kept — above relevance threshold (used for summary)
            </div>
            <ul className="space-y-1.5">
              {kept.map((d, i) => (
                <li key={(d.paper as any).id || i} className="flex items-start gap-2 text-sm">
                  <span
                    className={`shrink-0 rounded px-2 py-0.5 text-xs font-mono tabular-nums ${scoreBadgeClass(d.leads_score, threshold)}`}
                    title={d.reason}
                  >
                    {d.leads_score >= 0 ? "+" : ""}{d.leads_score.toFixed(2)}
                  </span>
                  <div className="min-w-0 flex-1">
                    <a
                      href={(d.paper as any).url || "#"}
                      target="_blank"
                      rel="noreferrer"
                      className="hover:underline break-words"
                    >
                      {(d.paper as any).title || "(untitled)"}
                    </a>
                    {(d.paper as any).source && (
                      <span className="text-xs text-muted-foreground ml-2">[{(d.paper as any).source}]</span>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          </section>
        )}

        {dropped.length === 0 && (
          <p className="text-sm text-muted-foreground">No papers were dropped at this threshold.</p>
        )}

        {dropped.length > 0 && (
          <section>
            <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
              Dropped — below relevance threshold
            </div>
            <ul className="space-y-1.5">
              {dropped.map((d, i) => (
                <li key={(d.paper as any).id || i} className="flex items-start gap-2 text-sm">
                  <span
                    className={`shrink-0 rounded px-2 py-0.5 text-xs font-mono tabular-nums ${scoreBadgeClass(d.leads_score, threshold)}`}
                    title={d.reason}
                  >
                    {d.leads_score >= 0 ? "+" : ""}{d.leads_score.toFixed(2)}
                  </span>
                  <div className="min-w-0 flex-1">
                    <a
                      href={(d.paper as any).url || "#"}
                      target="_blank"
                      rel="noreferrer"
                      className="hover:underline break-words"
                    >
                      {(d.paper as any).title || "(untitled)"}
                    </a>
                    {(d.paper as any).source && (
                      <span className="text-xs text-muted-foreground ml-2">[{(d.paper as any).source}]</span>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          </section>
        )}
      </CollapsibleContent>
    </Collapsible>
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
  const [clarifyQuestions, setClarifyQuestions] = useState<ClarifyingQuestion[]>([]);
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
        const qs = await AIService.getClarifyingQuestions(t);
        if (qs.length > 0) {
          clarifyAnswers = await new Promise<Record<string, string>>((resolve) => {
            setClarifyQuestions(qs);
            setClarifyOpen(true);
            clarifyResolverRef.current = (answers) => {
              setClarifyOpen(false);
              clarifyResolverRef.current = null;
              resolve(answers);
            };
          });
        }
      } catch (e) {
        console.warn("[clarify] questions endpoint failed; proceeding without:", e);
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
      // 1. PICO inference
      const analysis = await runStage("pico", signal, sig => AIService.inferPicoAndQuery(effectiveText, sig));
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
        DataAggregator.fetchAll(analysis.query, s.sources, newPico, undefined, sig)
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
            <div className="bg-gradient-to-br from-primary/5 via-card to-card border-b border-border/60 px-5 py-4">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground mb-1">
                <Sparkles className="size-3.5 text-primary" />Research Question
              </div>
              <p className="leading-snug italic">{entry.formal_question}</p>
            </div>
            <div className="p-5 space-y-5">
              {entry.summary && (
                <section>
                  <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">Summary</div>
                  <SummaryText text={entry.summary} />
                </section>
              )}
              {entry.references && entry.references.length > 0 && (
                <section>
                  <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">References</div>
                  <ReferencesBySource refs={entry.references} />
                </section>
              )}
              <section>
                <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">PICO Framework</div>
                <PicoCards pico={entry.pico_dict} />
              </section>

            <Collapsible>
              <CollapsibleTrigger asChild>
                <Button variant="outline" size="sm" className="w-full justify-between">
                  <span>Strategy: Criteria & Search String</span><ChevronDown className="size-4" />
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="pt-4 space-y-3">
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <div className="font-medium mb-2">Include Criteria</div>
                    <ul className="list-disc pl-5 space-y-1">
                      {entry.inclusion.map((x, i) => <li key={i}>{x}</li>)}
                    </ul>
                  </div>
                  <div>
                    <div className="font-medium mb-2">Exclude Criteria</div>
                    <ul className="list-disc pl-5 space-y-1">
                      {entry.exclusion.map((x, i) => <li key={i}>{x}</li>)}
                    </ul>
                  </div>
                </div>
                <Separator />
                <div>
                  <div className="font-medium mb-2">Final MeSH Search String</div>
                  <pre className="bg-muted rounded-md p-3 overflow-auto whitespace-pre-wrap font-mono text-xs">{entry.query}</pre>
                </div>
                {entry.adversarial_query && (
                  <div>
                    <div className="font-medium mb-2">Adversarial Query (for sensitivity check)</div>
                    <pre className="bg-muted rounded-md p-3 overflow-auto whitespace-pre-wrap font-mono text-xs">{entry.adversarial_query}</pre>
                  </div>
                )}
              </CollapsibleContent>
            </Collapsible>

            {/* Relevance-rerank explorer — only for the most recent run,
                since rerankResults holds only the latest LEADS pass. */}
            {idx === s.history.length - 1 && s.rerankResults && (
              <RelevanceExplorer />
            )}
            </div>
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

      {/* Strategy review */}
      {s.history.length > 0 && (
        <Card className="p-5 space-y-5">
          <h2>Strategy Review</h2>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="text-muted-foreground text-sm">Population</label>
              <Textarea value={s.pico.population} onChange={e => s.setPico({ ...s.pico, population: e.target.value })} rows={2} />
            </div>
            <div>
              <label className="text-muted-foreground text-sm">Intervention</label>
              <Textarea value={s.pico.intervention} onChange={e => s.setPico({ ...s.pico, intervention: e.target.value })} rows={2} />
            </div>
            <div>
              <label className="text-muted-foreground text-sm">Comparator</label>
              <Textarea value={s.pico.comparator} onChange={e => s.setPico({ ...s.pico, comparator: e.target.value })} rows={2} />
            </div>
            <div>
              <label className="text-muted-foreground text-sm">Outcome</label>
              <Textarea value={s.pico.outcome} onChange={e => s.setPico({ ...s.pico, outcome: e.target.value })} rows={2} />
            </div>
          </div>

          <Separator />

          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <label className="text-muted-foreground text-sm block mb-2">Inclusion Criteria</label>
              <CriteriaList
                items={s.inclusion}
                onChange={s.setInclusion}
                placeholder="e.g., randomized controlled trials"
                variant="include"
              />
            </div>
            <div>
              <label className="text-muted-foreground text-sm block mb-2">Exclusion Criteria</label>
              <CriteriaList
                items={s.exclusion}
                onChange={s.setExclusion}
                placeholder="e.g., animal studies"
                variant="exclude"
              />
            </div>
          </div>

          <Separator />

          <div>
            <label className="text-muted-foreground text-sm">Final Search String</label>
            <Textarea value={s.query} onChange={e => { s.setQuery(e.target.value); s.setUnifiedSearchQuery(e.target.value); }} rows={3} className="font-mono" />
          </div>
        </Card>
      )}

      <div className="h-24" />

      {/* Clarifying-questions modal — opens BEFORE the search runs whenever the
          user's goal is under-specified. Pauses handleSubmit until the user
          answers or skips, then resumes with the answers folded into the
          effective goal text. */}
      <ClarifyingQuestionsModal
        open={clarifyOpen}
        questions={clarifyQuestions}
        onCommit={(answers) => clarifyResolverRef.current?.(answers)}
        onSkipAll={() => clarifyResolverRef.current?.({})}
      />

      {/* Refinement popup — floats above the chat input, Claude-clarifying-question style */}
      {(refining || refinement) && s.history.length > 0 && (
        <div className="fixed bottom-20 left-72 right-0 z-30 px-6 pointer-events-none">
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

      {/* Chat input — fixed to bottom, matching content width */}
      <div className="fixed bottom-0 left-72 right-0 z-30 px-6 py-4 pointer-events-none">
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
