import { useEffect, useRef, useState } from "react";
import { useStore } from "../lib/store";
import { AIService, DataAggregator } from "../lib/mockServices";
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
import { Sparkles, Send, ChevronDown, X, Plus, Wand2, Check } from "lucide-react";

function ReferencesBySource({ refs }: { refs: { title: string; url: string; source: string; id: string }[] }) {
  // Preserve global numbering ([1], [2], ...) so they line up with inline citations in the summary.
  const numbered = refs.map((r, i) => ({ ...r, n: i + 1 }));
  const groups = new Map<string, typeof numbered>();
  for (const r of numbered) {
    const key = r.source || "Unknown";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(r);
  }
  return (
    <div className="space-y-3">
      {Array.from(groups.entries()).map(([source, list]) => (
        <div key={source}>
          <div className="text-xs font-medium text-foreground/80 mb-1">{source}</div>
          <ol className="space-y-1 text-sm">
            {list.map(r => (
              <li key={r.id || r.n} className="flex gap-2">
                <span className="text-muted-foreground tabular-nums">[{r.n}]</span>
                <a
                  href={r.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline break-words flex-1"
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
  { id: "question", label: "Draft formal research question", status: "pending" },
  { id: "summary", label: "Summarize the literature found", status: "pending" },
  { id: "suggestions", label: "Suggest refinements", status: "pending" },
  { id: "adversarial", label: "Build adversarial search", status: "pending" },
];

export function HomePage() {
  const s = useStore();
  const [input, setInput] = useState("");
  const [refining, setRefining] = useState(false);
  const [refinement, setRefinement] = useState<null | {
    field: "population" | "intervention" | "comparator" | "outcome";
    current: string;
    suggested: string;
    reason: string;
  }>(null);

  const last = s.history[s.history.length - 1];
  const task = s.tasks["home-analysis"];
  const analyzing = task?.status === "running";

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

  async function handleSubmit(text: string) {
    const t = text.trim();
    if (!t) return;
    setInput("");

    const { abort } = s.startTask("home-analysis", INITIAL_STAGES.map(st => ({ ...st, status: "pending" as const })));
    const signal = abort.signal;

    try {
      // 1. PICO inference
      const analysis = await runStage("pico", signal, sig => AIService.inferPicoAndQuery(t, sig));
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
        DataAggregator.fetchAll(analysis.query, s.sources, newPico, 20, sig)
      );
      const papers = fetched?.papers || [];
      if (fetched) {
        const breakdown = Object.entries(fetched.sourceCounts || {})
          .map(([k, v]) => `${k}: ${v}`)
          .join(" · ");
        markStage("papers", { status: "done", detail: `${papers.length} papers — ${breakdown}` });
        s.setRawPapers(papers);
      }

      const formalQ = await runStage("question", signal, sig => AIService.generateFormalQuestion(newPico, sig));
      const summaryWithRefs = await runStage("summary", signal, sig => AIService.generateComprehensiveSummaryWithRefs(t, papers, sig));
      const suggs = await runStage("suggestions", signal, sig => AIService.getRefinementSuggestions(t, papers, sig));
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
    s.setPico(p => ({ ...p, [refinement.field]: refinement.suggested }));
    setRefinement(null);
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

      {/* Targeted PICO refinement */}
      {s.history.length > 0 && !s.results && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium">Refine your PICO</div>
              <div className="text-xs text-muted-foreground">Get one targeted suggestion for the weakest element.</div>
            </div>
            <Button size="sm" onClick={suggestRefinement} disabled={refining}>
              <Wand2 className="size-4 mr-2" />{refining ? "Analyzing…" : "Suggest a refinement"}
            </Button>
          </div>

          {refinement && (
            <Card className="p-4 space-y-3 border-primary/30">
              <div className="flex items-start justify-between gap-3">
                <div className="space-y-2 flex-1">
                  <div className="text-xs uppercase tracking-wide text-muted-foreground">
                    Suggested change · <span className="text-foreground font-medium">{refinement.field}</span>
                  </div>
                  <div className="text-sm">
                    <div className="text-muted-foreground line-through break-words">{refinement.current || <em>empty</em>}</div>
                    <div className="text-foreground font-medium break-words mt-1">{refinement.suggested}</div>
                  </div>
                  <div className="text-xs text-muted-foreground italic">{refinement.reason}</div>
                </div>
                <button onClick={() => setRefinement(null)} className="text-muted-foreground hover:text-foreground">
                  <X className="size-4" />
                </button>
              </div>
              <div className="flex gap-2">
                <Button size="sm" onClick={applyRefinement}>
                  <Check className="size-4 mr-2" />Apply
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setRefinement(null)}>Dismiss</Button>
              </div>
            </Card>
          )}
        </div>
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

      {/* Chat input — fixed to bottom, matching content width */}
      <div className="fixed bottom-0 left-72 right-0 z-30 px-6 py-4 pointer-events-none">
        <form onSubmit={(e) => { e.preventDefault(); handleSubmit(input); }}
          className="max-w-4xl mx-auto flex gap-2 items-center bg-card/95 backdrop-blur border rounded-full shadow-lg pl-5 pr-2 py-2 pointer-events-auto">
          <Input value={input} onChange={e => setInput(e.target.value)}
            placeholder="Ask a question or refine your research goal..."
            className="flex-1 border-0 bg-transparent shadow-none focus-visible:ring-0 px-0" />
          <Button type="submit" disabled={analyzing || !input.trim()} className="rounded-full"><Send className="size-4 mr-2" />Send</Button>
        </form>
      </div>
    </div>
  );
}
