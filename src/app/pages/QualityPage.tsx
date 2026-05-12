import { useStore } from "../lib/store";
import { DataAggregator, Deduplicator, QualityService, QualityReport } from "../lib/mockServices";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Checkbox } from "../components/ui/checkbox";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../components/ui/collapsible";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../components/ui/table";
import { ChevronDown, Copy, ShieldCheck, AlertTriangle, ArrowRight, Search } from "lucide-react";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";

export function QualityPage() {
  const s = useStore();
  const task = s.tasks["quality-assess"];
  const running = task?.status === "running";

  if (s.history.length === 0) {
    return <Alert><AlertDescription>Define a research goal on the Home page first.</AlertDescription></Alert>;
  }

  async function runFetchAndAssess() {
    if (!s.query) { toast.error("Define a query on the Home page first."); return; }
    const { abort } = s.startTask("quality-assess", [{ id: "qa", label: "Quality assessment", status: "running" }]);
    s.updateTask("quality-assess", { detail: "Fetching papers from databases…" });
    const signal = abort.signal;
    try {
      const { papers: all } = await DataAggregator.fetchAll(s.query, s.sources, s.pico, s.numPerSource, signal);
      if (signal.aborted) { s.updateTask("quality-assess", { status: "canceled" }); return; }
      s.setRawPapers(all);

      s.updateTask("quality-assess", { detail: "Deduplicating…" });
      const { unique, duplicates } = Deduplicator.run(all);
      s.setUniquePapers(unique);
      s.setDuplicatesCount(duplicates.length);

      const reports: QualityReport[] = [];
      s.updateTask("quality-assess", { progress: { done: 0, total: unique.length } });
      for (let i = 0; i < unique.length; i++) {
        if (signal.aborted) break;
        s.updateTask("quality-assess", {
          progress: { done: i, total: unique.length, label: unique[i].title.slice(0, 80) },
          detail: unique[i].title.slice(0, 80),
        });
        try {
          reports.push(await QualityService.assessPaper(unique[i], signal));
        } catch (e: any) {
          if (signal.aborted) break;
          console.error(`quality-assess ${i + 1} failed:`, e?.message);
        }
        s.updateTask("quality-assess", { progress: { done: i + 1, total: unique.length } });
      }
      s.setQualityReports(reports);
      const auto = new Set(reports.filter(r => r.issues.some(i => i.severity === "high")).map(r => r.paper_id));
      s.setExcludedByQuality(auto);
      if (signal.aborted) {
        s.updateTask("quality-assess", { status: "canceled" });
        toast.info(`Canceled — ${reports.length} of ${unique.length} assessed`);
      } else {
        s.updateTask("quality-assess", { status: "done" });
        toast.success(`Assessed ${reports.length} unique papers (${duplicates.length} duplicates removed).`);
      }
    } catch (e: any) {
      s.updateTask("quality-assess", { status: "error", detail: e?.message });
    }
  }

  function toggleExclude(id: string) {
    s.setExcludedByQuality(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  function proceedToScreening() {
    if (!s.qualityReports || !s.uniquePapers) return;
    const kept = s.uniquePapers.filter(p => !s.excludedByQuality.has(p.id));
    if (kept.length === 0) { toast.error("All papers are currently excluded — adjust your selections."); return; }
    s.setPage("abstract");
    toast.info(`${kept.length} papers will be carried forward to abstract screening.`);
  }

  const reports = s.qualityReports;
  const totals = reports ? {
    excellent: reports.filter(r => r.rating === "Excellent").length,
    good: reports.filter(r => r.rating === "Good").length,
    fair: reports.filter(r => r.rating === "Fair").length,
    poor: reports.filter(r => r.rating === "Poor").length,
  } : null;
  const kept = reports ? reports.filter(r => !s.excludedByQuality.has(r.paper_id)).length : 0;

  return (
    <div className="space-y-4">
      {!reports && (
        <>
          <Alert><AlertDescription>Pull papers from your selected databases, deduplicate them, and run a quality assessment to surface issues before screening.</AlertDescription></Alert>
          {task && task.status === "running" && (
            <TaskProgressCard
              task={task}
              title="Quality assessment"
              onCancel={() => s.cancelTask("quality-assess")}
            />
          )}
          <Button onClick={runFetchAndAssess} disabled={running} size="lg" className="w-full">
            <Search className="size-4 mr-2" />{running ? "Working..." : "Fetch, Deduplicate & Assess Quality"}
          </Button>
        </>
      )}

      {reports && totals && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
            <Stat label="Fetched" value={s.rawPapers?.length ?? 0} icon={<Search className="size-4" />} />
            <Stat label="Duplicates" value={s.duplicatesCount} icon={<Copy className="size-4" />} />
            <Stat label="Unique" value={s.uniquePapers?.length ?? 0} />
            <Stat label="Excellent" value={totals.excellent} variant="success" />
            <Stat label="Fair / Poor" value={totals.fair + totals.poor} variant="warn" />
            <Stat label="Carrying Forward" value={kept} variant="info" />
          </div>

          <Card className="p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium">Quality Assessment</h3>
              <div className="text-xs text-muted-foreground">High-severity issues are pre-excluded; uncheck to override.</div>
            </div>
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-12">Keep</TableHead>
                    <TableHead>Paper</TableHead>
                    <TableHead>Source</TableHead>
                    <TableHead>Score</TableHead>
                    <TableHead>Issues</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {reports.map(r => {
                    const excluded = s.excludedByQuality.has(r.paper_id);
                    return (
                      <TableRow key={r.paper_id} className={excluded ? "opacity-50" : ""}>
                        <TableCell>
                          <Checkbox checked={!excluded} onCheckedChange={() => toggleExclude(r.paper_id)} />
                        </TableCell>
                        <TableCell className="max-w-md">
                          <Collapsible>
                            <CollapsibleTrigger asChild>
                              <button className="text-left w-full hover:underline flex items-start gap-1">
                                <ChevronDown className="size-4 mt-0.5 shrink-0" />
                                <span>{r.title}</span>
                              </button>
                            </CollapsibleTrigger>
                            <CollapsibleContent className="pt-2 space-y-3">
                              <a href={r.url} target="_blank" rel="noreferrer" className="text-xs text-primary hover:underline">{r.url}</a>
                              <div className="text-sm leading-relaxed bg-muted/30 rounded p-3">
                                {r.highlightedAbstract.map((seg, i) =>
                                  seg.flagged
                                    ? <mark key={i} title={seg.reason} className="bg-yellow-200 dark:bg-yellow-900/50 px-0.5 rounded">{seg.text}</mark>
                                    : <span key={i}>{seg.text}</span>
                                )}
                              </div>
                              {r.issues.length > 0 && (
                                <div className="space-y-1">
                                  <div className="text-xs font-medium text-muted-foreground">Detected Issues</div>
                                  {r.issues.map((iss, i) => (
                                    <div key={i} className="flex items-start gap-2 text-sm">
                                      <SeverityDot severity={iss.severity} />
                                      <div>
                                        <span className="font-medium">{iss.category}</span>
                                        <span className="text-muted-foreground"> — {iss.message}</span>
                                        {iss.evidence && <div className="text-xs italic text-muted-foreground mt-0.5">"{iss.evidence}…"</div>}
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </CollapsibleContent>
                          </Collapsible>
                        </TableCell>
                        <TableCell><Badge variant="outline">{r.source}</Badge></TableCell>
                        <TableCell><ScoreBadge score={r.score} rating={r.rating} /></TableCell>
                        <TableCell>
                          <div className="flex flex-wrap gap-1">
                            {r.issues.length === 0 && <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200"><ShieldCheck className="size-3 mr-1" />Clean</Badge>}
                            {r.issues.map((iss, i) => (
                              <Badge key={i} variant="outline" title={iss.message}
                                className={iss.severity === "high" ? "bg-red-50 text-red-700 border-red-200" : iss.severity === "medium" ? "bg-amber-50 text-amber-700 border-amber-200" : "bg-slate-50 text-slate-700 border-slate-200"}>
                                {iss.severity === "high" && <AlertTriangle className="size-3 mr-1" />}
                                {iss.category}
                              </Badge>
                            ))}
                          </div>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </Card>

          <div className="grid grid-cols-2 gap-2">
            <Button variant="outline" onClick={runFetchAndAssess} disabled={running}>Re-run Assessment</Button>
            <Button onClick={proceedToScreening}><ArrowRight className="size-4 mr-2" />Proceed to Abstract Screening ({kept})</Button>
          </div>
        </>
      )}
    </div>
  );
}

function Stat({ label, value, variant, icon }: { label: string; value: any; variant?: "success" | "warn" | "info"; icon?: React.ReactNode }) {
  const cls = variant === "success" ? "text-green-700" : variant === "warn" ? "text-amber-700" : variant === "info" ? "text-primary" : "text-foreground";
  return (
    <Card className="p-3 text-center">
      <div className={`text-2xl font-bold ${cls} flex items-center justify-center gap-1`}>{icon}{value}</div>
      <div className="text-xs text-muted-foreground">{label}</div>
    </Card>
  );
}

function ScoreBadge({ score, rating }: { score: number; rating: string }) {
  const cls = rating === "Excellent" ? "bg-green-100 text-green-800" :
    rating === "Good" ? "bg-emerald-100 text-emerald-800" :
    rating === "Fair" ? "bg-amber-100 text-amber-800" : "bg-red-100 text-red-800";
  return <span className={`px-2 py-0.5 rounded text-xs font-medium ${cls}`}>{score} · {rating}</span>;
}

function SeverityDot({ severity }: { severity: "high" | "medium" | "low" }) {
  const c = severity === "high" ? "bg-red-500" : severity === "medium" ? "bg-amber-500" : "bg-slate-400";
  return <span className={`inline-block size-2 rounded-full mt-1.5 shrink-0 ${c}`} />;
}
