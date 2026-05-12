import { useMemo, useState } from "react";
import { useStore, TextExtractionResult } from "../lib/store";
import { AIService } from "../lib/mockServices";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Textarea } from "../components/ui/textarea";
import { Badge } from "../components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../components/ui/collapsible";
import { ScanText, Sparkles, ChevronDown, AlertTriangle, Download } from "lucide-react";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";

const PRESETS = [
  "What was the primary outcome and its effect size?",
  "What was the sample size and participant demographics?",
  "List all reported adverse events with frequencies.",
  "Extract dose, frequency, and duration of the intervention.",
  "What were the inclusion and exclusion criteria for participants?",
];

export function TextExtractionPage() {
  const s = useStore();
  const [query, setQuery] = useState(PRESETS[0]);
  const task = s.tasks["text-extract"];
  const running = task?.status === "running";

  const acquired = useMemo(() => Object.values(s.fullTexts).filter(r => r.status === "found" && r.text), [s.fullTexts]);
  const missing = useMemo(() => Object.values(s.fullTexts).filter(r => r.status === "missing"), [s.fullTexts]);

  if (!s.results) return <Alert><AlertDescription>Complete Abstract Screening first.</AlertDescription></Alert>;
  if (acquired.length === 0) return <Alert><AlertDescription>No full texts acquired yet — visit Full-Text Acquisition to fetch them first.</AlertDescription></Alert>;

  async function run() {
    if (!query.trim()) { toast.error("Enter a question or instruction first."); return; }
    const { abort } = s.startTask("text-extract", [{ id: "te", label: "Extracting from text", status: "running" }]);
    s.updateTask("text-extract", { progress: { done: 0, total: acquired.length } });
    const signal = abort.signal;
    try {
      const out: TextExtractionResult[] = [];
      for (let i = 0; i < acquired.length; i++) {
        if (signal.aborted) break;
        const r = acquired[i];
        s.updateTask("text-extract", {
          progress: { done: i, total: acquired.length, label: r.title.slice(0, 80) },
          detail: r.title.slice(0, 80),
        });
        try {
          const ext = await AIService.extractFromText(r.text || "", query, signal);
          out.push({ paper_id: r.paper_id, title: r.title, query, ...ext });
        } catch (e: any) {
          if (signal.aborted) break;
          console.error(`text-extract ${i + 1} failed:`, e?.message);
        }
        s.updateTask("text-extract", { progress: { done: i + 1, total: acquired.length } });
      }
      s.setTextExtractions(out);
      if (signal.aborted) {
        s.updateTask("text-extract", { status: "canceled" });
        toast.info(`Canceled — ${out.length} of ${acquired.length} processed`);
      } else {
        s.updateTask("text-extract", { status: "done" });
        toast.success(`Extracted from ${out.length} papers`);
      }
    } catch (e: any) {
      s.updateTask("text-extract", { status: "error", detail: e?.message });
    }
  }

  function exportJson() {
    const blob = new Blob([JSON.stringify({ query, results: s.textExtractions }, null, 2)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "text_extractions.json"; a.click(); URL.revokeObjectURL(a.href);
  }

  const totalSpans = s.textExtractions.reduce((a, r) => a + r.spans.length, 0);
  const totalValues = s.textExtractions.reduce((a, r) => a + r.values.length, 0);
  const noEvidence = s.textExtractions.filter(r => r.spans.length === 0).length;

  return (
    <div className="space-y-4">
      {missing.length > 0 && (
        <Alert>
          <AlertDescription>
            <AlertTriangle className="size-4 inline mr-1 text-amber-600" />
            <strong>{missing.length}</strong> included paper{missing.length > 1 ? "s have" : " has"} no full text and will be skipped. Acquire or upload them on the Full-Text Acquisition tab.
          </AlertDescription>
        </Alert>
      )}

      <Card className="p-4 space-y-3">
        <h3 className="font-medium flex items-center gap-2"><ScanText className="size-4 text-primary" />Ask in natural language</h3>
        <Textarea value={query} onChange={e => setQuery(e.target.value)} rows={3}
          placeholder="e.g. What was the primary outcome and effect size?" />
        <div className="flex flex-wrap gap-1.5">
          {PRESETS.map((p, i) => (
            <Button key={i} size="sm" variant="outline" className="h-7 text-xs" onClick={() => setQuery(p)}>{p}</Button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={run} disabled={running}>
            <Sparkles className="size-4 mr-2" />{running ? "Extracting..." : `Extract from ${acquired.length} papers`}
          </Button>
          {s.textExtractions.length > 0 && (
            <Button variant="outline" onClick={exportJson}><Download className="size-4 mr-2" />Export JSON</Button>
          )}
        </div>
      </Card>

      {task && task.status === "running" && (
        <TaskProgressCard
          task={task}
          title="Text extraction"
          onCancel={() => s.cancelTask("text-extract")}
        />
      )}

      {s.textExtractions.length > 0 && (
        <>
          <div className="grid grid-cols-4 gap-3">
            <Stat label="Papers" value={s.textExtractions.length} />
            <Stat label="Highlighted passages" value={totalSpans} variant="success" />
            <Stat label="Extracted values" value={totalValues} variant="success" />
            <Stat label="No evidence" value={noEvidence} variant={noEvidence > 0 ? "warn" : undefined} />
          </div>

          <div className="space-y-2">
            {s.textExtractions.map((r) => {
              const rec = s.fullTexts[r.paper_id];
              const text = rec?.text || "";
              return (
                <Collapsible key={r.paper_id} defaultOpen={r.spans.length > 0}>
                  <Card>
                    <CollapsibleTrigger asChild>
                      <button className="w-full text-left p-4 flex items-center gap-3 hover:bg-muted/30">
                        <div className="flex-1 min-w-0">
                          <div className="truncate">{r.title}</div>
                          <div className="text-xs text-muted-foreground">{r.summary}</div>
                        </div>
                        <Badge variant={r.spans.length > 0 ? "default" : "secondary"}>{r.spans.length} hits</Badge>
                        <Badge variant="outline">{r.values.length} values</Badge>
                        <ChevronDown className="size-4 text-muted-foreground shrink-0" />
                      </button>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <div className="px-4 pb-4 space-y-3">
                        {r.spans.length === 0 ? (
                          <div className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded p-3">
                            <AlertTriangle className="size-4 inline mr-1" />
                            No relevant passages found for this query. The information may not be in this paper.
                          </div>
                        ) : (
                          <div className="text-sm leading-relaxed bg-muted/30 rounded p-3 max-h-72 overflow-auto whitespace-pre-wrap">
                            {renderHighlighted(text, r.spans)}
                          </div>
                        )}

                        {r.values.length > 0 && (
                          <div>
                            <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">Extracted values</div>
                            <div className="grid md:grid-cols-2 gap-2">
                              {r.values.map((v, i) => (
                                <div key={i} className="border rounded p-2 text-sm bg-card">
                                  <div className="flex items-center gap-2">
                                    <Badge variant="outline" className="text-[10px]">{v.field}</Badge>
                                    <span className="font-mono">{v.value}</span>
                                  </div>
                                  {v.quote && <div className="text-xs italic text-muted-foreground mt-1">"…{v.quote}…"</div>}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </CollapsibleContent>
                  </Card>
                </Collapsible>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

function renderHighlighted(text: string, spans: { start: number; end: number; label?: string }[]) {
  if (!text) return <span className="text-muted-foreground">No text available.</span>;
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  const parts: React.ReactNode[] = [];
  let cursor = 0;
  sorted.forEach((s, i) => {
    if (s.start > cursor) parts.push(<span key={`p${i}`}>{text.slice(cursor, s.start)}</span>);
    parts.push(<mark key={`m${i}`} className="bg-yellow-200 dark:bg-yellow-900/50 rounded px-0.5">{text.slice(s.start, s.end)}</mark>);
    cursor = s.end;
  });
  if (cursor < text.length) parts.push(<span key="tail">{text.slice(cursor)}</span>);
  return parts;
}

function Stat({ label, value, variant }: { label: string; value: any; variant?: "success" | "warn" }) {
  const cls = variant === "success" ? "text-green-700" : variant === "warn" ? "text-amber-700" : "text-foreground";
  return (
    <Card className="p-3 text-center">
      <div className={`text-2xl font-bold ${cls}`}>{value}</div>
      <div className="text-xs text-muted-foreground">{label}</div>
    </Card>
  );
}
