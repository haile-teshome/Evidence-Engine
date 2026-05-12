import { useStore } from "../lib/store";
import { AIService, formatDuration, FullTextResult } from "../lib/mockServices";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Popover, PopoverContent, PopoverTrigger } from "../components/ui/popover";
import { FlaskConical, Check, Minus, X as XIcon } from "lucide-react";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";

export function FullTextPage() {
  const s = useStore();
  const task = s.tasks["full-text-screen"];
  const running = task?.status === "running";

  if (!s.results) return <Alert><AlertDescription>Complete Abstract Screening first.</AlertDescription></Alert>;

  const passed = s.results.filter(r => r.Decision === "INCLUDE");
  if (passed.length === 0) return <Alert><AlertDescription>No papers passed abstract screening — adjust your criteria and rerun.</AlertDescription></Alert>;

  async function run() {
    const { abort } = s.startTask("full-text-screen", [{ id: "ft", label: "Full-text screening", status: "running" }]);
    s.updateTask("full-text-screen", { progress: { done: 0, total: passed.length } });
    const signal = abort.signal;
    const start = Date.now();
    try {
      const out: FullTextResult[] = [];
      const ftReasons: Record<string, number> = {};
      for (let i = 0; i < passed.length; i++) {
        if (signal.aborted) break;
        s.updateTask("full-text-screen", {
          progress: { done: i, total: passed.length, label: passed[i].Title.slice(0, 80) },
          detail: passed[i].Title.slice(0, 80),
        });
        try {
          const r = await AIService.screenFullTextMultiAgent(
            { paper_id: passed[i].paper_id, Title: passed[i].Title, URL: passed[i].URL, Source: passed[i].Source, Abstract: passed[i].Abstract },
            s.inclusion, s.exclusion,
            s.fullTexts[passed[i].paper_id]?.text,
            signal,
            s.pico,
          );
          out.push(r);
          if (r.Decision === "Exclude") {
            const k = r.Reason.split(" ").slice(0, 4).join(" ");
            ftReasons[k] = (ftReasons[k] || 0) + 1;
          }
        } catch (e: any) {
          if (signal.aborted) break;
          console.error(`full-text-screen ${i + 1} failed:`, e?.message);
        }
        s.updateTask("full-text-screen", { progress: { done: i + 1, total: passed.length } });
      }
      s.setFullTextResults(out);
      s.setFtDuration((Date.now() - start) / 1000);
      s.setPrisma(p => ({ ...p, ft_exclusion_breakdown: ftReasons, included_final: out.filter(x => x.Decision === "Include").length }));
      if (signal.aborted) {
        s.updateTask("full-text-screen", { status: "canceled" });
        toast.info(`Canceled — ${out.length} of ${passed.length} screened`);
      } else {
        s.updateTask("full-text-screen", { status: "done" });
        toast.success(`Full-text screening complete in ${formatDuration((Date.now() - start) / 1000)}`);
      }
    } catch (e: any) {
      s.updateTask("full-text-screen", { status: "error", detail: e?.message });
    }
  }

  const ft = s.fullTextResults;
  const allCriteria = [...s.inclusion, ...s.exclusion];

  return (
    <div className="space-y-4">
      {!ft && (
        <>
          <Alert><AlertDescription>{passed.length} papers ready for full-text analysis.</AlertDescription></Alert>
          {task && task.status === "running" && (
            <TaskProgressCard
              task={task}
              title="Full-text screening"
              onCancel={() => s.cancelTask("full-text-screen")}
            />
          )}
          <Button onClick={run} disabled={running} size="lg" className="w-full">
            <FlaskConical className="size-4 mr-2" />{running ? "Analyzing..." : "Begin Full-Text Screening"}
          </Button>
        </>
      )}

      {ft && (
        <Card className="p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-medium">Full-Text Results</h3>
            <div className="flex gap-2">
              <Badge variant="secondary">{ft.length} papers</Badge>
              <Badge className="bg-green-600">{ft.filter(x => x.Decision === "Include").length} included</Badge>
              <Badge variant="destructive">{ft.filter(x => x.Decision === "Exclude").length} excluded</Badge>
              <Badge variant="outline">{formatDuration(s.ftDuration)}</Badge>
            </div>
          </div>
          <div className="rounded-md border max-h-[600px] overflow-auto">
            <table className="w-full text-sm border-collapse">
              <thead className="bg-muted sticky top-0 z-30">
                <tr className="text-left">
                  <th className="px-3 py-2 sticky left-0 bg-muted z-40 border-b border-r min-w-[110px]">Decision</th>
                  <th className="px-3 py-2 sticky left-[110px] bg-muted z-40 border-b border-r min-w-[320px] max-w-[320px]">Title</th>
                  <th className="px-3 py-2 border-b whitespace-nowrap border-l">Population</th>
                  <th className="px-3 py-2 border-b whitespace-nowrap">Intervention</th>
                  <th className="px-3 py-2 border-b whitespace-nowrap">Comparator</th>
                  <th className="px-3 py-2 border-b whitespace-nowrap border-r">Outcome</th>
                  {allCriteria.map(c => (
                    <th key={c} className="px-3 py-2 border-b whitespace-nowrap max-w-[140px] truncate" title={c}>
                      {c.length > 22 ? c.slice(0, 22) + "…" : c}
                    </th>
                  ))}
                  <th className="px-3 py-2 border-b min-w-[320px]">Reason</th>
                </tr>
              </thead>
              <tbody>
                {ft.map(row => (
                  <tr key={row.paper_id} className="border-b last:border-b-0 align-top group/row">
                    <td className="px-3 py-2 sticky left-0 z-20 border-r whitespace-nowrap bg-card group-hover/row:bg-muted/30">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${row.Decision === "Include" ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
                        {row.Decision}
                      </span>
                    </td>
                    <td className="px-3 py-2 sticky left-[110px] z-20 border-r min-w-[320px] max-w-[320px] bg-card group-hover/row:bg-muted/30">
                      <a href={row.URL} target="_blank" rel="noreferrer" className="hover:underline break-words">
                        {row.Title}
                      </a>
                    </td>
                    {(["population", "intervention", "comparator", "outcome"] as const).map((k, idx) => {
                      const pe = row.picoEvidence?.[k];
                      const cls = idx === 0 ? "border-l" : idx === 3 ? "border-r" : "";
                      if (!pe || !pe.value) {
                        return <td key={k} className={`px-3 py-2 ${cls}`}><span className="text-xs text-muted-foreground">—</span></td>;
                      }
                      return (
                        <td key={k} className={`px-3 py-2 ${cls}`}>
                          <Popover>
                            <PopoverTrigger asChild>
                              <button>
                                <PicoBadge match={pe.match} />
                              </button>
                            </PopoverTrigger>
                            <PopoverContent className="w-80 text-xs space-y-2">
                              <div>
                                <div className="font-medium mb-1 capitalize">{k}</div>
                                <div className="text-muted-foreground">{pe.value}</div>
                              </div>
                              <div>
                                <div className="font-medium mb-1">Evidence from text</div>
                                {pe.evidence ? (
                                  <blockquote className="border-l-2 border-primary/40 pl-2 italic text-muted-foreground whitespace-pre-wrap">
                                    {pe.evidence}
                                  </blockquote>
                                ) : (
                                  <div className="text-muted-foreground italic">No matching passage found in the abstract or full text.</div>
                                )}
                              </div>
                            </PopoverContent>
                          </Popover>
                        </td>
                      );
                    })}
                    {allCriteria.map(c => {
                      const ev = row.criteriaEvidence?.[c];
                      const badge = (
                        <span className={`px-2 py-0.5 rounded text-xs font-medium cursor-pointer ${row.criteriaEval[c] === "INCLUDE" ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
                          {row.criteriaEval[c] || "N/A"}
                        </span>
                      );
                      return (
                        <td key={c} className="px-3 py-2">
                          {ev ? (
                            <Popover>
                              <PopoverTrigger asChild>{badge}</PopoverTrigger>
                              <PopoverContent className="w-80 text-xs space-y-2">
                                <div>
                                  <div className="font-medium mb-1">Criterion</div>
                                  <div className="text-muted-foreground">{c}</div>
                                </div>
                                <div>
                                  <div className="font-medium mb-1">Evidence</div>
                                  <blockquote className="border-l-2 border-primary/40 pl-2 italic text-muted-foreground whitespace-pre-wrap">{ev.evidence || "No matching passage found in source text."}</blockquote>
                                </div>
                                <div>
                                  <div className="font-medium mb-1">Reasoning</div>
                                  <div className="text-muted-foreground">{ev.reasoning}</div>
                                </div>
                              </PopoverContent>
                            </Popover>
                          ) : badge}
                        </td>
                      );
                    })}
                    <td className="px-3 py-2 text-foreground/90 min-w-[320px]">{row.Reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

function PicoBadge({ match }: { match: "yes" | "partial" | "no" }) {
  if (match === "yes") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-emerald-100 text-emerald-800">
        <Check className="size-3" />Met
      </span>
    );
  }
  if (match === "partial") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-800">
        <Minus className="size-3" />Partial
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-rose-100 text-rose-800">
      <XIcon className="size-3" />None
    </span>
  );
}
