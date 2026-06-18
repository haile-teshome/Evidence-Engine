import { useState, useEffect } from "react";
import { useStore, SimulationRun } from "../lib/store";
import { AIService, DataAggregator } from "../lib/mockServices";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Textarea } from "../components/ui/textarea";
import { Badge } from "../components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../components/ui/collapsible";
import { ChevronDown, Bot, Play, X, History, GitCompare, Trash2, Database, Layers, FlaskConical } from "lucide-react";
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
  const qOf = (run: SimulationRun, src: string) => run.perDbQueries[src] || run.unifiedQuery;
  const changedSources = allSources.filter(src => qOf(a, src) !== qOf(b, src));
  const unifiedChanged = a.unifiedQuery !== b.unifiedQuery;
  const anyQueryChange = changedSources.length > 0 || unifiedChanged;

  return (
    <Card className="p-4 space-y-4 border-primary/40">
      <div className="flex items-center justify-between">
        <h3 className="font-medium flex items-center gap-2">
          <GitCompare className="size-4 text-primary" />
          Comparing {a.label} vs {b.label}
        </h3>
        <Button variant="ghost" size="sm" onClick={onClose}><X className="size-4" /></Button>
      </div>

      {/* Yield by database */}
      <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Yield by database</div>
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

      {/* Query changes */}
      <div className="space-y-3">
        <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Query changes <span className="text-muted-foreground/60">({a.label} → {b.label})</span>
        </div>
        {!anyQueryChange ? (
          <div className="text-xs text-muted-foreground">No query differences between these runs.</div>
        ) : (
          <>
            {unifiedChanged && (
              <div className="space-y-1">
                <div className="text-[11px] font-medium text-muted-foreground">Base query</div>
                <QueryDiff previous={a.unifiedQuery} current={b.unifiedQuery} />
              </div>
            )}
            {changedSources.map(src => (
              <div key={src} className="space-y-1">
                <div className="text-[11px] font-medium text-muted-foreground">{src}</div>
                <QueryDiff previous={qOf(a, src)} current={qOf(b, src)} />
              </div>
            ))}
          </>
        )}
      </div>
    </Card>
  );
}

// ── History panel ─────────────────────────────────────────────────────────────

function HistoryColumn({
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
  return (
    <Card className="w-72 shrink-0 p-0 overflow-hidden flex flex-col">
      <div className="px-3 py-2 border-b flex items-center justify-between">
        <span className="text-xs uppercase tracking-wide text-muted-foreground flex items-center gap-1.5">
          <History className="size-3.5" />Run history
          {runs.length > 0 && <span className="text-muted-foreground/70">({runs.length})</span>}
        </span>
        {runs.length > 0 && (
          <Button variant="ghost" size="sm" className="h-7 px-2 text-xs" onClick={onClear} title="Clear history">
            <Trash2 className="size-3.5" />
          </Button>
        )}
      </div>

      {runs.length === 0 ? (
        <div className="flex-1 flex items-center justify-center p-4 text-xs text-muted-foreground text-center">
          Runs appear here after you Run Planning or AI Optimize. Pick two to compare.
        </div>
      ) : (
        <div className="overflow-auto flex-1">
          {[...runs].reverse().map(run => {
            const isA = compareA === run.id;
            const isB = compareB === run.id;
            const selected = isA || isB;
            return (
              <div
                key={run.id}
                className={`px-3 py-2.5 border-b transition-colors ${selected ? "bg-primary/5" : "hover:bg-muted/30"}`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-medium flex items-center gap-1.5">
                    {run.label}
                    {run.source === "ai-optimize" && <Badge variant="outline" className="text-[9px] px-1 py-0">AI</Badge>}
                  </span>
                  <span className="text-[10px] text-muted-foreground tabular-nums">{fmtTime(run.timestamp)}</span>
                </div>
                <div className="mt-0.5">
                  <span className="font-semibold text-primary tabular-nums">{run.totalYield.toLocaleString()}</span>
                  <span className="text-xs text-muted-foreground"> papers</span>
                </div>
                <div className="font-mono text-[10px] text-muted-foreground truncate mt-1">
                  {run.unifiedQuery.slice(0, 60)}{run.unifiedQuery.length > 60 ? "…" : ""}
                </div>
                <Button
                  size="sm"
                  variant={selected ? "default" : "outline"}
                  className="h-6 text-xs px-2 mt-1.5 w-full"
                  onClick={() => onSelectCompare(run.id)}
                >
                  {isA ? "Selected A" : isB ? "Selected B" : "Compare"}
                </Button>
              </div>
            );
          })}
        </div>
      )}

      {(compareA || compareB) && (
        <div className="px-3 py-2 border-t text-[11px] text-primary">
          {compareA && compareB ? "Comparison shown below." : "Select one more run to compare."}
        </div>
      )}
    </Card>
  );
}

// ── Compact count pill ──────────────────────────────────────────────────────
function Pill({
  icon: Icon, children, tone = "default", title,
}: {
  icon: React.ComponentType<{ className?: string }>;
  children: React.ReactNode;
  tone?: "default" | "green";
  title?: string;
}) {
  const cls = tone === "green"
    ? "bg-emerald-50 text-emerald-700 border-emerald-200"
    : "bg-muted text-muted-foreground border-transparent";
  return (
    <span title={title} className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${cls}`}>
      <Icon className="size-3" />{children}
    </span>
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
  // Two-pane workspace: which database's query is open on the right.
  const [selectedDb, setSelectedDb] = useState<string | null>(null);
  const [showTrace, setShowTrace] = useState(false);

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
      const { papers } = await DataAggregator.fetchAll(q, [src], s.pico, 25, undefined, s.elsevierToken, s.ezproxyConnected);
      s.setDbTestResults(prev => ({
        ...(prev || {}),
        [src]: { query: q, total_found: yieldRes[src] || 0, papers: papers.slice(0, 25).map(p => ({ title: p.title, url: p.url })) },
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

  // ── Two-pane workspace derived values ──────────────────────────────────────
  const activeDb = selectedDb && apiSources.includes(selectedDb) ? selectedDb : (apiSources[0] ?? null);
  const yieldOf = (src: string): number | null =>
    s.simulation?.[src] ?? s.dbTestResults?.[src]?.total_found ?? null;
  // Bars show each database's contribution as a share of the combined yield.
  const sumYields = apiSources.reduce((acc, src) => acc + (yieldOf(src) ?? 0), 0);
  const shareOf = (y: number | null): number => (sumYields > 0 && y ? y / sumYields : 0);

  return (
    <div className="space-y-3">
      {/* ── Header: base query + actions + pills ─────────────────────────── */}
      <Card className="p-3 space-y-3">
        <div className="flex items-start gap-3 flex-wrap">
          <div className="flex-1 min-w-[280px]">
            <label className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Base query · all databases</label>
            <Textarea value={s.unifiedSearchQuery} onChange={e => updateUnified(e.target.value)} rows={2} className="font-mono text-sm mt-1" />
          </div>
          <div className="flex gap-2 pt-5 flex-wrap">
            <Button variant="outline" onClick={runAiOptimize} disabled={optRunning}>
              <Bot className="size-4 mr-2" />AI Optimize
            </Button>
            <Button onClick={runSimulation} disabled={optRunning}>
              <Play className="size-4 mr-2" />Run Planning
            </Button>
            <Button variant="ghost" onClick={() => { s.setSimulation(null); s.setDbTestResults(null); s.setAgenticTrace(null); s.setAgenticSummary(null); }}>
              <X className="size-4 mr-2" />Clear
            </Button>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <Pill icon={Database}>{apiSources.length} database{apiSources.length === 1 ? "" : "s"}</Pill>
          {s.simulation && <Pill icon={Layers} tone="green" title={`Up to ${s.numPerSource}/source is downloaded then deduplicated`}>{totalYield.toLocaleString()} total yield</Pill>}
          {s.simulationRuns.length > 0 && <Pill icon={History}>{s.simulationRuns.length} run{s.simulationRuns.length === 1 ? "" : "s"}</Pill>}
        </div>
        {optTask && optTask.status === "running" && (
          <div className="space-y-2">
            <AnalysisProgress
              stages={optTask.stages as ProgStage[]}
              startedAt={optTask.startedAt}
              title="AI optimizing per source"
              onCancel={() => s.cancelTask("ai-optimize")}
            />
            <div className="max-h-40 overflow-auto text-xs space-y-1 bg-muted/30 rounded p-2 font-mono">
              {optTask.log.map((m, i) => <div key={i}>{m}</div>)}
            </div>
          </div>
        )}
      </Card>

      {/* ── Two-pane: database list (left) + query editor (right) ─────────── */}
      <div className="flex gap-4 h-[calc(100vh-20rem)] min-h-[26rem]">
        {/* LEFT: databases with yield bars */}
        <Card className="w-64 shrink-0 p-0 overflow-hidden flex flex-col">
          <div className="px-3 py-2 border-b flex items-center justify-between">
            <span className="text-xs uppercase tracking-wide text-muted-foreground">Databases</span>
            {s.simulation && <span className="text-xs font-semibold text-primary tabular-nums">{totalYield.toLocaleString()}</span>}
          </div>
          <div className="overflow-auto flex-1">
            {apiSources.map(src => {
              const y = yieldOf(src);
              const active = src === activeDb;
              const share = shareOf(y);
              const custom = (s.perDbQueries[src] ?? "").trim() !== "" && s.perDbQueries[src] !== s.unifiedSearchQuery;
              return (
                <button
                  key={src}
                  onClick={() => setSelectedDb(src)}
                  className={`w-full text-left px-3 py-3 border-b transition-colors ${active ? "bg-primary/10 border-l-2 border-l-primary" : "border-l-2 border-l-transparent hover:bg-muted/50"}`}
                >
                  <div className="flex items-center justify-between gap-2 mb-1.5">
                    <span className="text-sm font-medium truncate flex items-center gap-1.5">
                      {src}
                      {custom && <Badge variant="outline" className="text-[9px] px-1 py-0">custom</Badge>}
                    </span>
                    <span className="text-sm font-semibold tabular-nums shrink-0">{y === null ? "—" : y.toLocaleString()}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-2 flex-1 rounded-full bg-muted overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${active ? "bg-primary" : "bg-primary/55"}`}
                        style={{ width: `${y ? Math.max(3, share * 100) : 0}%` }}
                      />
                    </div>
                    <span className="text-[10px] text-muted-foreground tabular-nums w-9 text-right shrink-0">
                      {y === null ? "" : `${(share * 100).toFixed(0)}%`}
                    </span>
                  </div>
                </button>
              );
            })}
            {apiSources.length === 0 && (
              <div className="p-4 text-sm text-muted-foreground">No API databases selected. Pick sources on the Home page.</div>
            )}
          </div>
        </Card>

        {/* RIGHT: selected database's query editor */}
        <Card className="flex-1 min-w-0 p-0 overflow-hidden flex flex-col">
          {!activeDb ? (
            <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">Select a database on the left.</div>
          ) : (
            <>
              <div className="border-b p-3 flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="font-medium leading-tight">{activeDb}</div>
                  {yieldOf(activeDb) === null ? (
                    <div className="text-xs text-muted-foreground mt-0.5">Not tested yet</div>
                  ) : (
                    <div className="flex items-baseline gap-1.5 mt-0.5">
                      <span className="text-2xl font-bold text-primary tabular-nums leading-none">{yieldOf(activeDb)!.toLocaleString()}</span>
                      <span className="text-xs text-muted-foreground">papers · {(shareOf(yieldOf(activeDb)) * 100).toFixed(0)}% of total</span>
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <Button
                    size="sm" variant="ghost" className="text-xs h-8"
                    onClick={() => s.setPerDbQueries(p => ({ ...p, [activeDb]: s.unifiedSearchQuery }))}
                    title="Reset this database's query to the base query"
                  >
                    Reset to base
                  </Button>
                  <Button size="sm" onClick={() => testDb(activeDb)} disabled={testing === activeDb}>
                    <FlaskConical className="size-4 mr-1.5" />{testing === activeDb ? "Testing…" : "Test"}
                  </Button>
                </div>
              </div>
              <div className="flex-1 overflow-auto p-3 space-y-3">
                <div>
                  <label className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Query for {activeDb}</label>
                  <Textarea
                    value={s.perDbQueries[activeDb] ?? ""}
                    onChange={e => s.setPerDbQueries(p => ({ ...p, [activeDb]: e.target.value }))}
                    rows={6}
                    className="font-mono text-xs mt-1"
                  />
                </div>
                {s.dbTestResults?.[activeDb]?.papers?.length ? (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                        Sample results
                      </span>
                      <span className="text-[11px] text-muted-foreground">
                        showing {s.dbTestResults[activeDb].papers.length} of {(yieldOf(activeDb) ?? 0).toLocaleString()}
                      </span>
                    </div>
                    <div className="space-y-1.5">
                      {s.dbTestResults[activeDb].papers.map((p, i) => (
                        <div key={i} className="rounded-md border p-2.5 hover:border-primary/40 transition-colors">
                          <div className="flex gap-2">
                            <span className="text-xs text-muted-foreground tabular-nums shrink-0 pt-0.5 w-5 text-right">{i + 1}</span>
                            <div className="min-w-0">
                              <div className="text-sm leading-snug">{p.title}</div>
                              {p.url && (
                                <a href={p.url} target="_blank" rel="noreferrer" className="text-xs text-primary hover:underline break-all">{p.url}</a>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-xs text-muted-foreground">
                    Edit the query and click <strong>Test</strong> to preview the yield and a sample of results for {activeDb}.
                  </div>
                )}
              </div>
            </>
          )}
        </Card>

        {/* RIGHT-MOST: run history column */}
        <HistoryColumn
          runs={s.simulationRuns}
          compareA={compareA}
          compareB={compareB}
          onSelectCompare={handleSelectCompare}
          onClear={() => { s.clearSimulationRuns(); setCompareA(null); setCompareB(null); }}
        />
      </div>

      {/* Comparison panel — shown when two runs are selected */}
      {runA && runB && (
        <ComparePanel
          a={runA}
          b={runB}
          onClose={() => { setCompareA(null); setCompareB(null); }}
        />
      )}

      {/* Agentic trace (tucked away, collapsed by default) */}
      {s.agenticTrace && s.agenticSummary && (
        <Collapsible open={showTrace} onOpenChange={setShowTrace}>
          <CollapsibleTrigger asChild>
            <Button variant="outline" className="w-full justify-between">
              <span className="flex items-center gap-2">
                <Bot className="size-4 text-primary" />
                Optimization trace · {s.agenticSummary.iterations_run} iterations · best relevance {s.agenticSummary.best_relevance.toFixed(2)}
              </span>
              <ChevronDown className={`size-4 transition-transform ${showTrace ? "rotate-180" : ""}`} />
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="pt-3">
        <Card className="p-4 space-y-3">
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
          </CollapsibleContent>
        </Collapsible>
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
