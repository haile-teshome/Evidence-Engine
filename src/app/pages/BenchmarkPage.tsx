import { useEffect, useMemo, useState } from "react";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";
import { RefreshCw, Copy, ChevronUp, ChevronDown } from "lucide-react";
import { toast } from "sonner";

type RunSummary = {
  id: string;
  datasets: string[];
  architectures: string[];
  model_tiers: string[];
  cell_count: number;
};

type MetricRow = {
  dataset: string;
  architecture: string;
  model_tier: string;
  model_name: string;
  n: number;
  tp: number; fp: number; tn: number; fn: number;
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1: number;
  mcc: number;
  wss_at_95: number;
  avg_llm_calls_per_paper: number;
  avg_seconds_per_paper: number;
  total_seconds: number;
};

type PredictionRow = {
  dataset: string;
  architecture: string;
  model_tier: string;
  model_name: string;
  paper_id: string;
  title: string;
  label: number | null;
  prediction: number;
  llm_calls: number;
  wall_time_s: number;
  reasoning: string;
};

type SortKey = keyof MetricRow;

const METRIC_COLUMNS: { key: SortKey; label: string; format?: (v: any) => string; highlight?: "max" | "min" }[] = [
  { key: "architecture", label: "Architecture" },
  { key: "model_tier", label: "Model" },
  { key: "f1", label: "F1", format: v => v?.toFixed(3) ?? "—", highlight: "max" },
  { key: "recall", label: "Recall", format: v => v?.toFixed(3) ?? "—", highlight: "max" },
  { key: "specificity", label: "Specificity", format: v => v?.toFixed(3) ?? "—", highlight: "max" },
  { key: "precision", label: "Precision", format: v => v?.toFixed(3) ?? "—", highlight: "max" },
  { key: "mcc", label: "MCC", format: v => v?.toFixed(3) ?? "—", highlight: "max" },
  { key: "accuracy", label: "Accuracy", format: v => v?.toFixed(3) ?? "—", highlight: "max" },
  { key: "wss_at_95", label: "WSS@95", format: v => v?.toFixed(3) ?? "—", highlight: "max" },
  { key: "avg_llm_calls_per_paper", label: "LLM/paper", format: v => Number(v).toFixed(2), highlight: "min" },
  { key: "avg_seconds_per_paper", label: "s/paper", format: v => Number(v).toFixed(1), highlight: "min" },
];

export function BenchmarkPage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [predictions, setPredictions] = useState<PredictionRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [datasetFilter, setDatasetFilter] = useState<string>("all");
  const [sortKey, setSortKey] = useState<SortKey>("f1");
  const [sortDir, setSortDir] = useState<"desc" | "asc">("desc");

  async function refreshRuns() {
    try {
      setLoading(true);
      const r = await fetch("/api/benchmark/runs").then(r => r.json());
      const list: RunSummary[] = r.runs || [];
      setRuns(list);
      if (list.length > 0 && !runId) setRunId(list[0].id);
    } catch (e: any) {
      toast.error(`Could not load benchmark runs: ${e?.message || e}`);
    } finally {
      setLoading(false);
    }
  }

  async function loadRun(id: string) {
    setLoading(true);
    try {
      const r = await fetch(`/api/benchmark/runs/${encodeURIComponent(id)}`).then(r => r.json());
      setMetrics(r.metrics || []);
      setPredictions(r.predictions || []);
      setDatasetFilter("all");
    } catch (e: any) {
      toast.error(`Could not load run ${id}: ${e?.message || e}`);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { refreshRuns(); }, []);
  useEffect(() => { if (runId) loadRun(runId); }, [runId]);

  const datasets = useMemo(() => {
    const set = new Set(metrics.map(m => m.dataset));
    return Array.from(set).sort();
  }, [metrics]);

  const filtered = useMemo(() => {
    return datasetFilter === "all" ? metrics : metrics.filter(m => m.dataset === datasetFilter);
  }, [metrics, datasetFilter]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      const av = (a as any)[sortKey];
      const bv = (b as any)[sortKey];
      if (typeof av === "number" && typeof bv === "number") {
        return sortDir === "desc" ? bv - av : av - bv;
      }
      const sa = String(av ?? "");
      const sb = String(bv ?? "");
      return sortDir === "desc" ? sb.localeCompare(sa) : sa.localeCompare(sb);
    });
    return arr;
  }, [filtered, sortKey, sortDir]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir(d => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  }

  // Bests per column for highlighting
  const bests = useMemo(() => {
    const out: Record<string, number> = {};
    for (const col of METRIC_COLUMNS) {
      if (!col.highlight) continue;
      const vals = filtered.map(r => Number((r as any)[col.key])).filter(v => Number.isFinite(v));
      if (!vals.length) continue;
      out[col.key as string] = col.highlight === "max" ? Math.max(...vals) : Math.min(...vals);
    }
    return out;
  }, [filtered]);

  // Chart data: F1 by architecture, one bar per model_tier
  const chartData = useMemo(() => {
    const byArch = new Map<string, Record<string, number | string>>();
    for (const r of filtered) {
      const row = byArch.get(r.architecture) || { architecture: r.architecture };
      row[r.model_tier] = Number(r.f1);
      byArch.set(r.architecture, row);
    }
    return Array.from(byArch.values()).sort((a, b) =>
      String(a.architecture).localeCompare(String(b.architecture)),
    );
  }, [filtered]);

  const tiersInData = useMemo(() => {
    return Array.from(new Set(filtered.map(r => r.model_tier))).sort();
  }, [filtered]);

  const cliExample = useMemo(() => {
    const dsList = (runs?.[0]?.datasets || ["sample"]).join(" ");
    return `cd benchmark
pip install -r requirements.txt
python run_benchmark.py --datasets ${dsList} --architectures all --models small medium`;
  }, [runs]);

  return (
    <div className="space-y-4">
      <Alert>
        <AlertDescription>
          Browse comparison reports produced by the benchmark CLI. To create a new run, see the CLI snippet below.
        </AlertDescription>
      </Alert>

      <Card className="p-4 space-y-3">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-2 flex-wrap">
            <label className="text-sm text-muted-foreground">Run:</label>
            <Select value={runId || ""} onValueChange={setRunId}>
              <SelectTrigger className="w-64"><SelectValue placeholder="Select a run…" /></SelectTrigger>
              <SelectContent>
                {(runs || []).map(r => (
                  <SelectItem key={r.id} value={r.id}>
                    {r.id} · {r.cell_count} cells
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {datasets.length > 1 && (
              <>
                <label className="text-sm text-muted-foreground ml-2">Dataset:</label>
                <Select value={datasetFilter} onValueChange={setDatasetFilter}>
                  <SelectTrigger className="w-44"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All datasets</SelectItem>
                    {datasets.map(d => <SelectItem key={d} value={d}>{d}</SelectItem>)}
                  </SelectContent>
                </Select>
              </>
            )}
          </div>
          <Button variant="outline" size="sm" onClick={refreshRuns} disabled={loading}>
            <RefreshCw className={`size-3 mr-1 ${loading ? "animate-spin" : ""}`} />Refresh
          </Button>
        </div>

        {runs && runs.length === 0 && (
          <Alert>
            <AlertDescription>
              No benchmark runs yet. Use the CLI snippet below to produce one — it writes to <code>benchmark/reports/</code>.
            </AlertDescription>
          </Alert>
        )}

        <CliSnippet command={cliExample} />
      </Card>

      {filtered.length > 0 && (
        <Tabs defaultValue="table">
          <TabsList>
            <TabsTrigger value="table">Comparison table</TabsTrigger>
            <TabsTrigger value="chart">F1 by architecture</TabsTrigger>
            <TabsTrigger value="predictions">Per-paper predictions</TabsTrigger>
          </TabsList>

          <TabsContent value="table">
            <Card className="p-3 overflow-auto">
              <table className="w-full text-sm border-collapse">
                <thead className="bg-muted">
                  <tr className="text-left">
                    {METRIC_COLUMNS.map(col => (
                      <th
                        key={col.key as string}
                        onClick={() => toggleSort(col.key)}
                        className="px-3 py-2 border-b cursor-pointer hover:bg-muted/80 select-none whitespace-nowrap"
                      >
                        <span className="inline-flex items-center gap-1">
                          {col.label}
                          {sortKey === col.key && (sortDir === "desc" ? <ChevronDown className="size-3" /> : <ChevronUp className="size-3" />)}
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((row, idx) => (
                    <tr key={idx} className="border-b last:border-b-0 hover:bg-muted/20">
                      {METRIC_COLUMNS.map(col => {
                        const v = (row as any)[col.key];
                        const isBest = col.highlight && bests[col.key as string] !== undefined &&
                          Number(v) === bests[col.key as string];
                        return (
                          <td key={col.key as string}
                              className={`px-3 py-2 ${isBest ? "font-semibold text-primary" : ""} ${typeof v === "number" ? "tabular-nums" : ""}`}>
                            {col.format ? col.format(v) : String(v ?? "—")}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </TabsContent>

          <TabsContent value="chart">
            <Card className="p-4">
              <div style={{ width: "100%", height: 360 }}>
                <ResponsiveContainer>
                  <BarChart data={chartData} margin={{ top: 10, right: 24, bottom: 30, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted))" />
                    <XAxis dataKey="architecture" angle={-15} textAnchor="end" interval={0} height={60} />
                    <YAxis domain={[0, 1]} tickFormatter={v => v.toFixed(1)} />
                    <Tooltip />
                    <Legend />
                    {tiersInData.map((t, i) => (
                      <Bar key={t} dataKey={t} fill={["#0d9488", "#7c3aed", "#f59e0b", "#0284c7"][i % 4]} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                Higher is better. Bars grouped by architecture; one bar per model tier.
              </p>
            </Card>
          </TabsContent>

          <TabsContent value="predictions">
            <Card className="p-3 overflow-auto max-h-[70vh]">
              <PredictionsTable rows={predictions} datasetFilter={datasetFilter} />
            </Card>
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}

function CliSnippet({ command }: { command: string }) {
  return (
    <div className="bg-muted/40 rounded-md p-3 relative">
      <button
        onClick={() => { navigator.clipboard.writeText(command); toast.success("Copied"); }}
        className="absolute top-2 right-2 text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1"
      >
        <Copy className="size-3" />Copy
      </button>
      <pre className="text-xs font-mono whitespace-pre-wrap break-words pr-12">{command}</pre>
    </div>
  );
}

function PredictionsTable({ rows, datasetFilter }: { rows: PredictionRow[]; datasetFilter: string }) {
  const [archFilter, setArchFilter] = useState<string>("all");
  const [tierFilter, setTierFilter] = useState<string>("all");
  const [showOnly, setShowOnly] = useState<"all" | "fp" | "fn" | "correct">("all");

  const filtered = useMemo(() => {
    return rows.filter(r => {
      if (datasetFilter !== "all" && r.dataset !== datasetFilter) return false;
      if (archFilter !== "all" && r.architecture !== archFilter) return false;
      if (tierFilter !== "all" && r.model_tier !== tierFilter) return false;
      if (showOnly === "fp" && !(r.prediction === 1 && r.label === 0)) return false;
      if (showOnly === "fn" && !(r.prediction === 0 && r.label === 1)) return false;
      if (showOnly === "correct" && r.prediction !== r.label) return false;
      return true;
    });
  }, [rows, datasetFilter, archFilter, tierFilter, showOnly]);

  const archs = useMemo(() => Array.from(new Set(rows.map(r => r.architecture))).sort(), [rows]);
  const tiers = useMemo(() => Array.from(new Set(rows.map(r => r.model_tier))).sort(), [rows]);

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Select value={archFilter} onValueChange={setArchFilter}>
          <SelectTrigger className="w-48 h-8"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All architectures</SelectItem>
            {archs.map(a => <SelectItem key={a} value={a}>{a}</SelectItem>)}
          </SelectContent>
        </Select>
        <Select value={tierFilter} onValueChange={setTierFilter}>
          <SelectTrigger className="w-44 h-8"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All model tiers</SelectItem>
            {tiers.map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}
          </SelectContent>
        </Select>
        <Select value={showOnly} onValueChange={(v: any) => setShowOnly(v)}>
          <SelectTrigger className="w-44 h-8"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All papers</SelectItem>
            <SelectItem value="correct">Correct only</SelectItem>
            <SelectItem value="fp">False positives</SelectItem>
            <SelectItem value="fn">False negatives</SelectItem>
          </SelectContent>
        </Select>
        <Badge variant="outline">{filtered.length} of {rows.length}</Badge>
      </div>

      <table className="w-full text-xs border-collapse">
        <thead className="bg-muted sticky top-0">
          <tr className="text-left">
            <th className="px-2 py-1 border-b">Dataset</th>
            <th className="px-2 py-1 border-b">Arch</th>
            <th className="px-2 py-1 border-b">Model</th>
            <th className="px-2 py-1 border-b">Paper</th>
            <th className="px-2 py-1 border-b">Label</th>
            <th className="px-2 py-1 border-b">Pred</th>
            <th className="px-2 py-1 border-b">Result</th>
            <th className="px-2 py-1 border-b">Reason</th>
          </tr>
        </thead>
        <tbody>
          {filtered.slice(0, 500).map((r, i) => {
            const correct = r.prediction === r.label;
            const tag = r.label === null ? "—" :
              correct ? "✓" :
              r.prediction === 1 ? "FP" : "FN";
            const tagColor = correct ? "text-emerald-700" : "text-rose-700";
            return (
              <tr key={i} className="border-b last:border-b-0 align-top hover:bg-muted/20">
                <td className="px-2 py-1">{r.dataset}</td>
                <td className="px-2 py-1">{r.architecture}</td>
                <td className="px-2 py-1">{r.model_tier}</td>
                <td className="px-2 py-1 max-w-[320px] break-words">{r.title}</td>
                <td className="px-2 py-1 tabular-nums">{r.label}</td>
                <td className="px-2 py-1 tabular-nums">{r.prediction}</td>
                <td className={`px-2 py-1 font-medium ${tagColor}`}>{tag}</td>
                <td className="px-2 py-1 max-w-[420px] break-words text-muted-foreground">{r.reasoning}</td>
              </tr>
            );
          })}
          {filtered.length > 500 && (
            <tr><td colSpan={8} className="px-2 py-2 text-muted-foreground text-center">
              … {filtered.length - 500} more rows (filter to narrow)
            </td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
