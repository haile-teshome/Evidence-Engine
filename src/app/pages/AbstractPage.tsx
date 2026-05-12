import { useStore } from "../lib/store";
import { AIService, formatDuration, ScreenResult } from "../lib/mockServices";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Popover, PopoverContent, PopoverTrigger } from "../components/ui/popover";
import { Search, Check, X as XIcon } from "lucide-react";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";

export function AbstractPage() {
  const s = useStore();
  const task = s.tasks["abstract-screen"];
  const running = task?.status === "running";

  async function runSearch() {
    if (!s.uniquePapers) { toast.error("Run Quality Assessment first to fetch and deduplicate papers."); return; }
    const queue = s.uniquePapers.filter(p => !s.excludedByQuality.has(p.id));
    if (queue.length === 0) { toast.error("All papers were excluded in quality assessment."); return; }

    const { abort } = s.startTask("abstract-screen", [
      { id: "screen", label: "Screening papers", status: "running" },
    ]);
    s.updateTask("abstract-screen", { progress: { done: 0, total: queue.length } });
    const signal = abort.signal;
    const start = Date.now();
    try {
      const screened: ScreenResult[] = [];
      const reasons: Record<string, number> = {};
      for (let i = 0; i < queue.length; i++) {
        if (signal.aborted) break;
        s.updateTask("abstract-screen", {
          progress: { done: i, total: queue.length, label: queue[i].title.slice(0, 80) },
          detail: queue[i].title.slice(0, 80),
        });
        try {
          const r = await AIService.screenPaperMultiAgent(queue[i], s.pico, s.inclusion, s.exclusion, signal);
          screened.push(r);
          if (r.Decision === "EXCLUDE") {
            const k = r.Reason.slice(0, 50);
            reasons[k] = (reasons[k] || 0) + 1;
          }
        } catch (e: any) {
          if (signal.aborted) break;
          console.error(`abstract-screen ${i + 1} failed:`, e?.message);
        }
        s.updateTask("abstract-screen", { progress: { done: i + 1, total: queue.length } });
      }

      if (signal.aborted) {
        s.updateTask("abstract-screen", { status: "canceled" });
        toast.info(`Canceled — ${screened.length} of ${queue.length} screened`);
      }

      // Always persist whatever was screened so the user keeps partial work.
      const totalExcluded = Object.values(reasons).reduce((a, b) => a + b, 0);
      const sourceCounts: Record<string, number> = {};
      (s.rawPapers || []).forEach(p => { sourceCounts[p.source] = (sourceCounts[p.source] || 0) + 1; });
      s.setPrisma(p => ({
        ...p,
        identified: s.rawPapers?.length || queue.length,
        source_counts: sourceCounts,
        duplicates_removed: s.duplicatesCount,
        screened: screened.length,
        excluded_total: totalExcluded,
        exclusion_breakdown: reasons,
        included_final: screened.length - totalExcluded,
      }));
      s.setResults(screened);
      s.setScreeningDuration((Date.now() - start) / 1000);

      if (!signal.aborted) {
        s.updateTask("abstract-screen", { status: "done" });
        toast.success(`Screened ${screened.length} papers in ${formatDuration((Date.now() - start) / 1000)}`);
      }
    } catch (e: any) {
      s.updateTask("abstract-screen", { status: "error", detail: e?.message });
    }
  }

  const r = s.results;
  const passed = r ? r.filter(x => x.Decision === "INCLUDE") : [];
  const excluded = r ? r.filter(x => x.Decision === "EXCLUDE") : [];

  return (
    <div className="space-y-4">
      {!r && !s.uniquePapers && <Alert><AlertDescription>Run Quality Assessment first — fetched and deduplicated papers will then be ready for screening here.</AlertDescription></Alert>}
      {!r && s.uniquePapers && (
        <Alert><AlertDescription>{s.uniquePapers.length - s.excludedByQuality.size} of {s.uniquePapers.length} unique papers passed quality assessment and are ready for multi-agent screening.</AlertDescription></Alert>
      )}

      {r && (
        <>
          <div className="grid grid-cols-4 gap-3">
            <StatBox label="Papers Screened" value={r.length} />
            <StatBox label="Included" value={passed.length} variant="success" />
            <StatBox label="Excluded" value={excluded.length} variant="danger" />
            <StatBox label="Time" value={formatDuration(s.screeningDuration)} />
          </div>

          <Card className="p-4">
            <h3 className="font-medium mb-3">Screening Results</h3>
            <div className="rounded-md border max-h-[600px] overflow-auto">
              <table className="w-full text-sm border-collapse">
                <thead className="bg-muted sticky top-0 z-30">
                  <tr className="text-left">
                    <th className="px-3 py-2 sticky left-0 bg-muted z-40 border-b border-r min-w-[110px]">Decision</th>
                    <th className="px-3 py-2 sticky left-[110px] bg-muted z-40 border-b border-r min-w-[320px] max-w-[320px]">Title</th>
                    <th className="px-3 py-2 border-b whitespace-nowrap">Source</th>
                    <th className="px-3 py-2 border-b whitespace-nowrap">Criteria met</th>
                    <th className="px-3 py-2 border-b min-w-[380px]">Reasoning</th>
                  </tr>
                </thead>
                <tbody>
                  {r.map(row => {
                    const traceEntries = Object.entries(row.Agent_Trace || {});
                    const passes = traceEntries.filter(([_, v]) => v.vote === "PASS").length;
                    const total = traceEntries.length;
                    return (
                      <tr key={row.paper_id} className="border-b last:border-b-0 align-top group/row">
                        <td className="px-3 py-2 sticky left-0 z-20 border-r bg-card group-hover/row:bg-muted/30">
                          <DecisionCell value={row.Decision} />
                        </td>
                        <td className="px-3 py-2 sticky left-[110px] z-20 border-r min-w-[320px] max-w-[320px] bg-card group-hover/row:bg-muted/30">
                          <a href={row.URL} target="_blank" rel="noreferrer" className="hover:underline break-words">
                            {row.Title}
                          </a>
                        </td>
                        <td className="px-3 py-2"><Badge variant="outline">{row.Source}</Badge></td>
                        <td className="px-3 py-2 whitespace-nowrap">
                          {total > 0 ? (
                            <Popover>
                              <PopoverTrigger asChild>
                                <button className="text-xs underline-offset-2 hover:underline tabular-nums">
                                  {passes} / {total}
                                </button>
                              </PopoverTrigger>
                              <PopoverContent className="w-96 text-xs space-y-2 max-h-96 overflow-auto">
                                {traceEntries.map(([crit, v]) => (
                                  <div key={crit} className="border-b last:border-b-0 pb-2 last:pb-0">
                                    <div className="flex items-start gap-1.5">
                                      {v.vote === "PASS" ? (
                                        <Check className="size-3 text-emerald-600 shrink-0 mt-0.5" />
                                      ) : v.vote === "FAIL" ? (
                                        <XIcon className="size-3 text-rose-600 shrink-0 mt-0.5" />
                                      ) : (
                                        <span className="size-3 shrink-0 mt-0.5 text-muted-foreground">·</span>
                                      )}
                                      <div className="font-medium leading-snug">{crit}</div>
                                    </div>
                                    {v.evidence && (
                                      <blockquote className="border-l-2 border-primary/30 pl-2 italic text-muted-foreground mt-1">
                                        {v.evidence}
                                      </blockquote>
                                    )}
                                  </div>
                                ))}
                              </PopoverContent>
                            </Popover>
                          ) : (
                            <span className="text-xs text-muted-foreground">—</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-foreground/90 min-w-[380px] group-hover/row:bg-muted/30">{row.Reason}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {passed.length > 0 && (
              <Alert className="mt-3"><AlertDescription>{passed.length} papers passed abstract screening — head to Full-Text Evidence to continue.</AlertDescription></Alert>
            )}
          </Card>
        </>
      )}

      {task && task.status === "running" && (
        <TaskProgressCard
          task={task}
          title="Multi-agent abstract screening"
          onCancel={() => s.cancelTask("abstract-screen")}
        />
      )}

      {s.uniquePapers && (
        <Button onClick={runSearch} disabled={running} size="lg" className="w-full">
          <Search className="size-4 mr-2" />{running ? "Screening..." : "Run Multi-Agent Screening"}
        </Button>
      )}
    </div>
  );
}

function StatBox({ label, value, variant }: { label: string; value: any; variant?: "success" | "danger" }) {
  const cls = variant === "success" ? "text-green-700" : variant === "danger" ? "text-red-700" : "text-foreground";
  return (
    <Card className="p-4 text-center">
      <div className={`text-2xl font-bold ${cls}`}>{value}</div>
      <div className="text-xs text-muted-foreground">{label}</div>
    </Card>
  );
}

function DecisionCell({ value }: { value: string }) {
  const inc = value.toUpperCase().includes("INCLUDE");
  return <span className={`px-2 py-1 rounded font-medium text-xs ${inc ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>{inc ? "✅ Include" : "❌ Exclude"}</span>;
}

