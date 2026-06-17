import { useState, useEffect } from "react";
import { useStore, SimulationRun } from "../lib/store";
import { AIService, DataAggregator } from "../lib/mockServices";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Textarea } from "../components/ui/textarea";
import { Badge } from "../components/ui/badge";
import { Separator } from "../components/ui/separator";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../components/ui/collapsible";
import { ChevronDown, Bot, Play, X, History, GitCompare, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { QueryDiff } from "../components/QueryDiff";
import { AnalysisProgress, Stage as ProgStage } from "../components/AnalysisProgress";

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmtTime(ts: number) {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function deltaClass(d: number) {
  if (d > 0) return "text-emerald-600";
  if (d < 0) return "text-rose-600";
  return "text-muted-foreground";
}

function deltaLabel(d: number) {
  return d === 0 ? "—" : `${d > 0 ? "+" : ""}${d.toLocaleString()}`;
}

// ── Comparison panel ──────────────────────────────────────────────────────────

function ComparePanel({ a, b, onClose }: { a: SimulationRun; b: SimulationRun; onClose: () => void }) {
  const allSources = Array.from(new Set([...Object.keys(a.counts), ...Object.keys(b.counts)]));

  return (
    <Card className="p-4 space-y-4 border-primary/40">
      <div className="flex items-center justify-between">
        <h3 className="font-medium flex items-center gap-2">
          <GitCompare className="size-4 text-primary" />
          Comparing {a.label} vs {b.label}
        </h3>
        <Button variant="ghost" size="sm" onClick={onClose}><X className="size-4" /></Button>
      </div>

      {/* Count table */}
      <div className="divide-y rounded-md border text-sm">
        <div className="grid grid-cols-[140px_1fr_1fr_90px] px-3 py-2 text-xs uppercase tracking-wide text-muted-foreground bg-muted/40">
          <div>Database</div>
          <div className="text-right">{a.label}</div>
          <div className="text-right">{b.label}</div>
          <div className="text-right">Δ</div>
        </div>
        {allSources.map(src => {
          const ca = a.counts[src] ?? 0;
          const cb = b.counts[src] ?? 0;
          const d = cb - ca;
          return (
            <div key={src} className="grid grid-cols-[140px_1fr_1fr_90px] px-3 py-2.5 gap-2">
              <div className="font-medium">{src}</div>
              <div className="text-right tabular-nums text-muted-foreground">{ca.toLocaleString()}</div>
              <div className="text-right tabular-nums">{cb.toLocaleString()}</div>
              <div className={`text-right tabular-nums font-medium ${deltaClass(d)}`}>{deltaLabel(d)}</div>
            </div>
          );
        })}
        {/* Total row */}
        <div className="grid grid-cols-[140px_1fr_1fr_90px] px-3 py-2.5 gap-2 font-semibold bg-muted/20">
          <div>Total</div>
          <div className="text-right tabular-nums text-muted-foreground">{a.totalYield.toLocaleString()}</div>
          <div className="text-right tabular-nums">{b.totalYield.toLocaleString()}</div>
          <div className={`text-right tabular-nums ${deltaClass(b.totalYield - a.totalYield)}`}>
            {deltaLabel(b.totalYield - a.totalYield)}
          </div>
        </div>
      </div>

      {/* Query diffs per source */}
      {allSources.map(src => {
        const qa = a.perDbQueries[src] || a.unifiedQuery;
        const qb = b.perDbQueries[src] || b.unifiedQuery;
        if (qa === qb) return null;
        return (
          <div key={src} className="space-y-1">
            <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">{src} — query diff</div>
            <QueryDiff previous={qa} current={qb} />
          </div>
        );
      })}

      {/* Unified query diff (if per-db queries are same as unified) */}
      {a.unifiedQuery !== b.unifiedQuery && allSources.every(src => {
        const qa = a.perDbQueries[src] || a.unifiedQuery;
        const qb = b.perDbQueries[src] || b.unifiedQuery;
        return qa === (a.perDbQueries[src] || a.unifiedQuery) && qb === (b.perDbQueries[src] || b.unifiedQuery);
      }) && (
        <div className="space-y-1">
          <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Unified query diff</div>
          <QueryDiff previous={a.unifiedQuery} current={b.unifiedQuery} />
        </div>
      )}
    </Card>
  );
}

// ── History panel ─────────────────────────────────────────────────────────────

function HistoryPanel({
  runs,
  compareA,
  compareB,
  onSelectCompare,
  onClear,
}: {
  runs: SimulationRun[];
  compareA: string | null;
  compareB: string | null;
  onSelectCompare: (id: string) => void;
  onClear: () => void;
}) {
  if (runs.length === 0) return null;

  return (
    <Card className="p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-medium flex items-center gap-2">
          <History className="size-4 text-primary" />
          Run History
          <span className="text-xs text-muted-foreground font-normal">({runs.length} run{runs.length !== 1 ? "s" : ""})</span>
        </h3>
        <div className="flex items-center gap-2">
          {(compareA || compareB) && (
            <span className="text-xs text-primary">
              {compareA && compareB ? "Ready — scroll up to compare" : "Select one more run to compare"}
            </span>
          )}
          <Button variant="ghost" size="sm" onClick={onClear} title="Clear history">
            <Trash2 className="size-4 text-muted-foreground" />
          </Button>
        </div>
      </div>

      <div className="text-xs text-muted-foreground">
        Click <strong>Compare</strong> on two runs to see a side-by-side diff.
      </div>

      <div className="divide-y rounded-md border text-sm">
        <div className="grid grid-cols-[80px_80px_1fr_auto] px-3 py-2 text-xs uppercase tracking-wide text-muted-foreground bg-muted/40">
          <div>Run</div>
          <div>Time</div>
          <div>Query preview / total</div>
          <div></div>
        </div>
        {[...runs].reverse().map(run => {
          const isA = compareA === run.id;
          const isB = compareB === run.id;
          const selected = isA || isB;
          return (
            <div
              key={run.id}
              className={`grid grid-cols-[80px_80px_1fr_auto] px-3 py-2.5 gap-2 items-center transition-colors ${selected ? "bg-primary/5" : "hover:bg-muted/30"}`}
            >
              <div className="flex items-center gap-1.5">
                <span className="font-medium">{run.label}</span>
                {run.source === "ai-optimize" && (
                  <Badge variant="outline" className="text-[9px] px-1 py-0">AI</Badge>
                )}
              </div>
              <div className="text-muted-foreground tabular-nums">{fmtTime(run.timestamp)}</div>
              <div className="min-w-0">
                <span className="font-medium text-primary tabular-nums">{run.totalYield.toLocaleString()}</span>
                <span className="text-muted-foreground mx-1.5">·</span>
                <span className="text-muted-foreground truncate font-mono text-[11px]">
                  {run.unifiedQuery.slice(0, 80)}{run.unifiedQuery.length > 80 ? "…" : ""}
                </span>
              </div>
              <Button
                size="sm"
                variant={selected ? "default" : "outline"}
                className="h-7 text-xs px-2 shrink-0"
                onClick={() => onSelectCompare(run.id)}
              >
                {isA ? "A" : isB ? "B" : "Compare"}
              </Button>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function SimulationPage() {
  const s = useStore();
  const apiSources = s.sources.filter(x => x !== "Local PDFs");
  const optTask = s.tasks["ai-optimize"];
  const optRunning = optTask?.status === "running";
  const [testing, setTesting] = useState<string | null>(null);

  // Compare state — tracks which two run IDs are selected (A then B).
  const [compareA, setCompareA] = useState<string | null>(null);
  const [compareB, setCompareB] = useState<string | null>(null);

  useEffect(() => {
    s.setPerDbQueries(prev => {
      const next = { ...prev };
      apiSources.forEach(src => { if (!next[src]) next[src] = s.unifiedSearchQuery || s.query; });
      return next;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.sources.join(","), s.unifiedSearchQuery]);

  if (s.history.length === 0) {
    return <Alert><AlertDescription>Start by defining a research goal on the Home page.</AlertDescription></Alert>;
  }

  const updateUnified = (v: string) => {
    s.setUnifiedSearchQuery(v);
    s.setSimulation(null);
  };

  function saveRun(counts: Record<string, number>, source: "manual" | "ai-optimize") {
    s.addSimulationRun({
      unifiedQuery: s.unifiedSearchQuery,
      perDbQueries: { ...s.perDbQueries },
      counts,
      totalYield: Object.values(counts).reduce((a, b) => a + b, 0),
      source,
    });
  }

  function handleSelectCompare(id: string) {
    if (compareA === id) { setCompareA(null); return; }
    if (compareB === id) { setCompareB(null); return; }
    if (!compareA) { setCompareA(id); return; }
    if (!compareB) { setCompareB(id); return; }
    // Both already set — replace B (cycle through)
    setCompareB(id);
  }

  async function testDb(src: string) {
    setTesting(src);
    try {
      const q = s.perDbQueries[src] || s.unifiedSearchQuery;
      const yieldRes = await DataAggregator.simulateYield(q, [src], undefined, s.elsevierToken, s.ezproxyConnected);
      const { papers } = await DataAggregator.fetchAll(q, [src], s.pico, 10, undefined, s.elsevierToken, s.ezproxyConnected);
      s.setDbTestResults(prev => ({
        ...(prev || {}),
        [src]: { query: q, total_found: yieldRes[src] || 0, papers: papers.slice(0, 10).map(p => ({ title: p.title, url: p.url })) },
      }));
      toast.success(`${src}: ${yieldRes[src]} papers`);
    } finally { setTesting(null); }
  }

  async function runSimulation() {
    toast.info("Calculating yields...");
    const out: Record<string, number> = {};
    for (const src of apiSources) {
      const q = (s.perDbQueries[src] || s.unifiedSearchQuery || "").trim();
      if (!q) { out[src] = 0; continue; }
      const r = await DataAggregator.simulateYield(q, [src], undefined, s.elsevierToken, s.ezproxyConnected);
      out[src] = r[src] || 0;
    }
    s.setSimulation(out);
    saveRun(out, "manual");
  }

  async function runAiOptimize() {
    const stages: ProgStage[] = apiSources.map(src => ({
      id: src,
      label: src,
      status: "pending" as const,
    }));
    const { abort, taskId } = s.startTask("ai-optimize", stages);
    const signal = abort.signal;

    try {
      const result = await AIService.agenticOptimizePerSource(
        s.unifiedSearchQuery, s.pico, apiSources,
        (iter, _total, source, count, relevance, reasoning) => {
          s.updateTaskStage("ai-optimize", source, {
            status: "running",
            detail: `iter ${iter} · ${count.toLocaleString()} papers · rel ${relevance.toFixed(2)}`,
          });
          s.appendTaskLog(
            "ai-optimize",
            `Iter ${iter} — ${source}: ${count.toLocaleString()} papers, relevance ${relevance.toFixed(2)} — ${reasoning || ""}`,
          );
        },
        signal,
        taskId,
      );
      if (signal.aborted) {
        s.updateTask("ai-optimize", { status: "canceled" });
        toast.info("AI Optimize canceled");
        return;
      }
      const last = result.trace[result.trace.length - 1];
      Object.entries(last.sources).forEach(([k, v]: any) => {
        s.updateTaskStage("ai-optimize", k, {
          status: "done",
          detail: `${(v.count as number).toLocaleString()} papers · rel ${(v.relevance_score as number).toFixed(2)}`,
        });
      });
      s.setAgenticTrace(result.trace);
      s.setAgenticSummary({
        iterations_run: result.iterations_run,
        total_papers_found: result.total_papers_found,
        best_relevance: result.best_relevance,
      });
      s.setPerDbQueries(prev => ({ ...prev, ...result.per_source_queries }));
      const counts: Record<string, number> = {};
      Object.entries(last.sources).forEach(([k, v]: any) => { counts[k] = v.count; });
      s.setSimulation(counts);
      saveRun(counts, "ai-optimize");
      s.updateTask("ai-optimize", { status: "done" });
      toast.success(`Optimization complete (${result.iterations_run} iterations)`);
    } catch (e: any) {
      if (signal.aborted || e?.name === "AbortError") {
        s.updateTask("ai-optimize", { status: "canceled" });
        toast.info("AI Optimize canceled");
      } else {
        s.updateTask("ai-optimize", { status: "error", detail: e?.message });
        toast.error(`AI Optimize failed: ${e?.message || "unknown error"}`);
      }
    }
  }

  const totalYield = s.simulation ? Object.values(s.simulation).reduce((a, b) => a + b, 0) : 0;

  const runA = compareA ? s.simulationRuns.find(r => r.id === compareA) : null;
  const runB = compareB ? s.simulationRuns.find(r => r.id === compareB) : null;

  return (
    <div className="space-y-4">
      <Card className="p-4 space-y-3">
        <div>
          <label className="text-sm text-muted-foreground">Search all databases with:</label>
          <Textarea value={s.unifiedSearchQuery} onChange={e => updateUnified(e.target.value)} rows={3} className="font-mono" />
        </div>
        <Separator />
        <div className="text-sm text-muted-foreground">Customize search strings for individual databases:</div>
        {apiSources.map(src => (
          <Collapsible key={src}>
            <CollapsibleTrigger asChild>
              <Button variant="outline" className="w-full justify-between">
                <span className="font-medium">{src}</span>
                {s.dbTestResults?.[src] && <Badge variant="secondary">{s.dbTestResults[src].total_found} papers</Badge>}
                <ChevronDown className="size-4" />
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-3 space-y-3 px-2">
              <div className="grid grid-cols-[1fr_auto] gap-2">
                <Textarea value={s.perDbQueries[src] || ""} onChange={e => s.setPerDbQueries(p => ({ ...p, [src]: e.target.value }))} rows={3} className="font-mono text-xs" />
                <div className="flex flex-col gap-2">
                  {s.dbTestResults?.[src] && <div className="text-center font-bold text-primary">{s.dbTestResults[src].total_found}<br /><span className="text-xs font-normal">papers</span></div>}
                  <Button size="sm" onClick={() => testDb(src)} disabled={testing === src}>{testing === src ? "..." : "Test"}</Button>
                </div>
              </div>
              {s.dbTestResults?.[src]?.papers?.length ? (
                <div className="space-y-1 text-sm">
                  <div className="font-medium">Top 10 results:</div>
                  {s.dbTestResults[src].papers.map((p, i) => (
                    <div key={i} className="border-l-2 border-muted pl-3 py-1">
                      <div>{i + 1}. {p.title}</div>
                      <a href={p.url} target="_blank" rel="noreferrer" className="text-xs text-primary hover:underline">{p.url}</a>
                    </div>
                  ))}
                </div>
              ) : null}
            </CollapsibleContent>
          </Collapsible>
        ))}

        <div className="grid grid-cols-3 gap-2 pt-2">
          <Button variant="outline" onClick={runAiOptimize} disabled={optRunning}>
            <Bot className="size-4 mr-2" />AI Optimize Per Source
          </Button>
          <Button onClick={runSimulation} disabled={optRunning}>
            <Play className="size-4 mr-2" />Run Simulation
          </Button>
          <Button variant="ghost" onClick={() => { s.setSimulation(null); s.setDbTestResults(null); s.setAgenticTrace(null); s.setAgenticSummary(null); }}>
            <X className="size-4 mr-2" />Clear Results
          </Button>
        </div>

        {optTask && optTask.status === "running" && (
          <div className="space-y-2">
            <AnalysisProgress
              stages={optTask.stages as ProgStage[]}
              startedAt={optTask.startedAt}
              title="AI optimizing per source"
              onCancel={() => s.cancelTask("ai-optimize")}
            />
            <div className="max-h-44 overflow-auto text-xs space-y-1 bg-muted/30 rounded p-2 font-mono">
              {optTask.log.map((m, i) => <div key={i}>{m}</div>)}
            </div>
          </div>
        )}
      </Card>

      {s.simulation && (
        <Card className="p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-medium">Simulation Results</h3>
            <div className="text-sm">
              Total: <span className="font-bold text-primary">{totalYield.toLocaleString()}</span> papers
              <span className="text-xs text-muted-foreground ml-2">(abstract screening downloads up to {s.numPerSource}/source then deduplicates)</span>
            </div>
          </div>
          <div className="divide-y rounded-md border">
            <div className="grid grid-cols-[140px_120px_1fr] px-3 py-2 text-xs uppercase tracking-wide text-muted-foreground bg-muted/40">
              <div>Database</div>
              <div className="text-right">Paper Count</div>
              <div>Query</div>
            </div>
            {Object.entries(s.simulation).map(([k, v]) => (
              <div key={k} className="grid grid-cols-[140px_120px_1fr] px-3 py-3 gap-2 text-sm">
                <div className="font-medium">{k}</div>
                <div className="text-right tabular-nums">{v.toLocaleString()}</div>
                <pre className="font-mono text-xs whitespace-pre-wrap break-words bg-muted/40 rounded p-2 leading-relaxed">
                  {s.perDbQueries[k] || s.unifiedSearchQuery || ""}
                </pre>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Comparison panel — shown when two runs are selected */}
      {runA && runB && (
        <ComparePanel
          a={runA}
          b={runB}
          onClose={() => { setCompareA(null); setCompareB(null); }}
        />
      )}

      {/* History */}
      <HistoryPanel
        runs={s.simulationRuns}
        compareA={compareA}
        compareB={compareB}
        onSelectCompare={handleSelectCompare}
        onClear={() => { s.clearSimulationRuns(); setCompareA(null); setCompareB(null); }}
      />

      {s.agenticTrace && s.agenticSummary && (
        <Card className="p-4 space-y-3">
          <h3 className="font-medium">Agentic Optimization Trace</h3>
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-muted/30 rounded p-3 text-center">
              <div className="text-2xl font-bold">{s.agenticSummary.iterations_run}</div>
              <div className="text-xs text-muted-foreground">Iterations</div>
            </div>
            <div className="bg-muted/30 rounded p-3 text-center">
              <div className="text-2xl font-bold">{s.agenticSummary.total_papers_found.toLocaleString()}</div>
              <div className="text-xs text-muted-foreground">Total Papers</div>
            </div>
            <div className="bg-muted/30 rounded p-3 text-center">
              <div className="text-2xl font-bold">{s.agenticSummary.best_relevance.toFixed(2)}</div>
              <div className="text-xs text-muted-foreground">Best Relevance</div>
            </div>
          </div>

          {Object.keys(groupTraceByDb(s.agenticTrace)).map(db => {
            const hist = groupTraceByDb(s.agenticTrace!)[db];
            const final = hist[hist.length - 1];
            return (
              <Collapsible key={db}>
                <CollapsibleTrigger asChild>
                  <Button variant="outline" className="w-full justify-between">
                    <span>{db}: {final.count} papers · relevance {final.relevance.toFixed(2)} ({final.quality}) · {hist.length} iterations</span>
                    <ChevronDown className="size-4" />
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent className="pt-3 space-y-3">
                  {hist.map((step, i) => {
                    const prev = i > 0 ? hist[i - 1] : null;
                    const deltaCount = prev ? step.count - prev.count : 0;
                    const deltaRel = prev ? step.relevance - prev.relevance : 0;
                    return (
                      <div key={i} className="border-l-2 border-primary pl-3 space-y-2 pb-2">
                        <div className="text-sm font-medium flex flex-wrap items-baseline gap-2">
                          <span>{i === hist.length - 1 ? "✅" : `${i + 1}.`} Iteration {step.iteration}</span>
                          <span className="text-muted-foreground font-normal">·</span>
                          <span>{step.count.toLocaleString()} papers</span>
                          {prev && (
                            <span className={`text-xs ${deltaCount >= 0 ? "text-emerald-600" : "text-rose-600"}`}>
                              ({deltaCount >= 0 ? "+" : ""}{deltaCount.toLocaleString()})
                            </span>
                          )}
                          <span className="text-muted-foreground font-normal">·</span>
                          <span>relevance {step.relevance.toFixed(2)}</span>
                          {prev && (
                            <span className={`text-xs ${deltaRel >= 0 ? "text-emerald-600" : "text-rose-600"}`}>
                              ({deltaRel >= 0 ? "+" : ""}{deltaRel.toFixed(2)})
                            </span>
                          )}
                          <span className="text-muted-foreground font-normal">·</span>
                          <span className="text-muted-foreground font-normal">{step.quality}</span>
                        </div>
                        {step.reasoning && (
                          <div className="text-xs text-muted-foreground italic">{step.reasoning}</div>
                        )}
                        <QueryDiff previous={prev?.query || ""} current={step.query} />
                      </div>
                    );
                  })}
                </CollapsibleContent>
              </Collapsible>
            );
          })}
        </Card>
      )}
    </div>
  );
}

function groupTraceByDb(trace: any[]): Record<string, { iteration: number; count: number; relevance: number; quality: string; query: string; reasoning: string }[]> {
  const out: Record<string, any[]> = {};
  for (const item of trace) {
    for (const [src, data] of Object.entries<any>(item.sources)) {
      if (!out[src]) out[src] = [];
      out[src].push({
        iteration: item.iteration,
        count: data.count,
        relevance: data.relevance_score,
        quality: data.quality_rating,
        query: data.query,
        reasoning: data.iteration_reasoning,
      });
    }
  }
  return out;
}
