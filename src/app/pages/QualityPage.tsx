import { useMemo, useState } from "react";
import { useStore } from "../lib/store";
import {
  DataAggregator, Deduplicator, QualityService,
  QualityReport, QualityOverride, RoBDomain, RoBJudgment,
} from "../lib/mockServices";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Checkbox } from "../components/ui/checkbox";
import { Popover, PopoverContent, PopoverTrigger } from "../components/ui/popover";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Textarea } from "../components/ui/textarea";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import {
  Copy, ShieldCheck, AlertTriangle, ArrowRight, Search,
  Pencil, History, RotateCcw,
} from "lucide-react";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";

const JUDGMENT_OPTIONS: RoBJudgment[] = [
  "Low", "Some Concerns", "High", "No information", "Not applicable",
];

// ---- judgment styling -----------------------------------------------------

function judgmentClass(j: RoBJudgment): string {
  switch (j) {
    case "Low": return "bg-emerald-100 text-emerald-800 border-emerald-200";
    case "Some Concerns": return "bg-amber-100 text-amber-800 border-amber-200";
    case "High": return "bg-rose-100 text-rose-800 border-rose-200";
    case "No information": return "bg-slate-100 text-slate-700 border-slate-200";
    case "Not applicable": return "bg-slate-50 text-slate-500 border-slate-200";
    default: return "bg-slate-50 text-slate-500 border-slate-200";
  }
}

function judgmentDotClass(j: RoBJudgment): string {
  switch (j) {
    case "Low": return "bg-emerald-500";
    case "Some Concerns": return "bg-amber-500";
    case "High": return "bg-rose-500";
    case "No information": return "bg-slate-300";
    case "Not applicable": return "bg-slate-200";
    default: return "bg-slate-200";
  }
}

function shortDomainLabel(name: string): string {
  return name
    .replace(/^Bias (arising from|due to|in measurement of|in selection of|in classification of) /i, "")
    .replace(/^Bias /i, "")
    .replace(/^the /, "");
}

// ---- override resolution --------------------------------------------------

/** Most-recent override for a (paper, domain) pair, or undefined if none. */
function latestOverride(
  paperId: string,
  domainId: string,
  overrides: QualityOverride[],
): QualityOverride | undefined {
  let last: QualityOverride | undefined;
  for (const o of overrides) {
    if (o.paper_id === paperId && o.domain_id === domainId) last = o;
  }
  return last;
}

/** Effective judgment for a domain — the override if present, else the AI's. */
function effectiveJudgment(
  paperId: string,
  domain: RoBDomain,
  overrides: QualityOverride[],
): RoBJudgment {
  const o = latestOverride(paperId, domain.id, overrides);
  return o ? o.new_judgment : domain.judgment;
}

/** Recompute the overall judgment using EFFECTIVE per-domain judgments. */
function recomputeOverall(
  report: QualityReport,
  overrides: QualityOverride[],
): { judgment: RoBJudgment; rationale: string } {
  const effective = report.domains.map(d => effectiveJudgment(report.paper_id, d, overrides));
  if (effective.length === 0) return { judgment: "No information", rationale: "No domains assessed." };
  if (effective.every(j => j === "No information"))
    return { judgment: "No information", rationale: "All domains lacked sufficient information for an appraisal." };
  if (effective.some(j => j === "High")) {
    const n = effective.filter(j => j === "High").length;
    return { judgment: "High", rationale: `${n} domain(s) judged High risk of bias.` };
  }
  if (effective.some(j => j === "Some Concerns")) {
    const n = effective.filter(j => j === "Some Concerns").length;
    return { judgment: "Some Concerns", rationale: `${n} domain(s) raised some concerns.` };
  }
  if (effective.every(j => j === "Low"))
    return { judgment: "Low", rationale: "All domains judged Low risk of bias." };
  return { judgment: "Some Concerns", rationale: "Mixed domain judgments without any High risk." };
}

// ---- page -----------------------------------------------------------------

export function QualityPage() {
  const s = useStore();
  const task = s.tasks["quality-assess"];
  const running = task?.status === "running";

  const [excludeRule, setExcludeRule] = useState<"none" | "any_high" | "two_or_more_high">("none");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [q, setQ] = useState("");

  if (s.history.length === 0) {
    return <Alert><AlertDescription>Define a research goal on the Home page first.</AlertDescription></Alert>;
  }

  async function runFetchAndAssess() {
    if (!s.query) { toast.error("Define a query on the Home page first."); return; }
    const { abort } = s.startTask("quality-assess", [{ id: "qa", label: "Quality assessment", status: "running" }]);
    s.updateTask("quality-assess", { detail: "Fetching papers from databases…" });
    const signal = abort.signal;
    try {
      const { papers: all } = await DataAggregator.fetchAll(s.query, s.sources, s.pico, s.numPerSource, signal, s.elsevierToken, s.ezproxyConnected);
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
      s.setExcludedByQuality(new Set());
      // Fresh assessment invalidates previous overrides — those referred to
      // domain judgments from the old run.
      s.setQualityOverrides([]);
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

  function applyExcludeRule(rule: "none" | "any_high" | "two_or_more_high") {
    setExcludeRule(rule);
    if (!s.qualityReports) return;
    if (rule === "none") {
      s.setExcludedByQuality(new Set());
      return;
    }
    const next = new Set<string>();
    for (const r of s.qualityReports) {
      // Use EFFECTIVE judgments (post-override) when counting High domains.
      const highCount = r.domains.filter(
        d => effectiveJudgment(r.paper_id, d, s.qualityOverrides) === "High",
      ).length;
      if (rule === "any_high" && highCount >= 1) next.add(r.paper_id);
      if (rule === "two_or_more_high" && highCount >= 2) next.add(r.paper_id);
    }
    s.setExcludedByQuality(next);
  }

  function proceedToScreening() {
    if (!s.qualityReports || !s.uniquePapers) return;
    const kept = s.uniquePapers.filter(p => !s.excludedByQuality.has(p.id));
    if (kept.length === 0) { toast.error("All papers are currently excluded — adjust your selections."); return; }
    s.setPage("abstract");
    toast.info(`${kept.length} papers will be carried forward to abstract screening.`);
  }

  const reports = s.qualityReports;
  const overrides = s.qualityOverrides;

  // Summary stats use effective overall judgments.
  const summaryCounts = useMemo(() => {
    if (!reports) return null;
    let low = 0, some = 0, high = 0, no_info = 0;
    for (const r of reports) {
      const j = recomputeOverall(r, overrides).judgment;
      if (j === "Low") low++;
      else if (j === "Some Concerns") some++;
      else if (j === "High") high++;
      else if (j === "No information") no_info++;
    }
    return { low, some, high, no_info };
  }, [reports, overrides]);

  const kept = reports ? reports.filter(r => !s.excludedByQuality.has(r.paper_id)).length : 0;

  const filtered = reports
    ? (q.trim() ? reports.filter(r => r.title.toLowerCase().includes(q.toLowerCase())) : reports)
    : [];
  const selected = reports ? (reports.find(r => r.paper_id === selectedId) ?? reports[0]) : null;

  return (
    <div className="space-y-4">
      {!reports && (
        <>
          <Alert>
            <AlertDescription>
              Risk-of-bias appraisal per study design (RoB 2, ROBINS-I, JBI, AMSTAR 2). Reviewer overrides are audit-logged.
            </AlertDescription>
          </Alert>
          {task && task.status === "running" && (
            <TaskProgressCard
              task={task}
              title="Quality assessment"
              onCancel={() => s.cancelTask("quality-assess")}
            />
          )}
          <Button onClick={runFetchAndAssess} disabled={running} size="lg" className="w-full">
            <Search className="size-4 mr-2" />{running ? "Working..." : "Fetch, Deduplicate & Appraise Risk of Bias"}
          </Button>
        </>
      )}

      {reports && summaryCounts && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
            <Stat label="Fetched" value={s.rawPapers?.length ?? 0} icon={<Search className="size-4" />} />
            <Stat label="Duplicates" value={s.duplicatesCount} icon={<Copy className="size-4" />} />
            <Stat label="Unique" value={s.uniquePapers?.length ?? 0} />
            <Stat label="Low RoB" value={summaryCounts.low} variant="success" />
            <Stat label="Some / High" value={summaryCounts.some + summaryCounts.high} variant="warn" />
            <Stat label="Carrying Forward" value={kept} variant="info" />
          </div>

          <Card className="p-4">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
              <div>
                <div className="text-sm font-medium">Exclusion rule</div>
                <div className="text-xs text-muted-foreground">
                  How aggressively to exclude papers based on RoB. Reviewer can override on individual rows below.
                  Uses your edited judgments where applicable.
                </div>
              </div>
              <div className="flex gap-1 flex-wrap">
                {([
                  ["none", "Keep all"],
                  ["any_high", "Exclude if any High"],
                  ["two_or_more_high", "Exclude if ≥ 2 High"],
                ] as const).map(([id, label]) => (
                  <Button
                    key={id}
                    size="sm"
                    variant={excludeRule === id ? "default" : "outline"}
                    onClick={() => applyExcludeRule(id)}
                  >
                    {label}
                  </Button>
                ))}
              </div>
            </div>
            {overrides.length > 0 && (
              <div className="mt-3 pt-3 border-t flex items-center justify-between text-xs">
                <span className="text-muted-foreground">
                  {overrides.length} reviewer override{overrides.length === 1 ? "" : "s"} logged.
                </span>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    s.clearQualityOverrides();
                    toast.info("Cleared all reviewer overrides.");
                  }}
                >
                  <RotateCcw className="size-3 mr-1" /> Revert all to AI judgments
                </Button>
              </div>
            )}
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium">Risk-of-bias appraisal</h3>
              <div className="text-xs text-muted-foreground">
                Select a paper, then click a judgment chip to override it.
              </div>
            </div>
            {/* ── Two-pane: paper list (left) + selected appraisal (right) ───── */}
            <div className="flex gap-4 h-[calc(100vh-26rem)] min-h-[26rem]">
              {/* LEFT: searchable paper list */}
              <div className="w-80 shrink-0 rounded-md border overflow-hidden flex flex-col">
                <div className="p-2 border-b">
                  <div className="relative">
                    <Search className="size-3.5 absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground" />
                    <Input
                      value={q}
                      onChange={e => setQ(e.target.value)}
                      placeholder={`Filter ${reports.length} papers…`}
                      className="pl-7 h-8 text-sm"
                    />
                  </div>
                </div>
                <div className="overflow-auto flex-1">
                  {filtered.map(r => {
                    const active = r.paper_id === selected?.paper_id;
                    const excluded = s.excludedByQuality.has(r.paper_id);
                    const ov = recomputeOverall(r, overrides);
                    return (
                      <button
                        key={r.paper_id}
                        onClick={() => setSelectedId(r.paper_id)}
                        className={`w-full text-left px-3 py-2.5 border-b hover:bg-muted/50 transition-colors ${active ? "bg-primary/10 border-l-2 border-l-primary" : "border-l-2 border-l-transparent"} ${excluded ? "opacity-50" : ""}`}
                      >
                        <div className="flex items-center gap-1.5 mb-1">
                          <span
                            title={`Overall: ${ov.judgment}`}
                            className={`inline-block size-2.5 rounded-full ${judgmentDotClass(ov.judgment)}`}
                          />
                          <div className="flex gap-0.5 flex-wrap">
                            {r.domains.map(d => {
                              const j = effectiveJudgment(r.paper_id, d, overrides);
                              const overridden = !!latestOverride(r.paper_id, d.id, overrides);
                              return (
                                <span
                                  key={d.id}
                                  title={`${d.name}: ${j}${overridden ? " (reviewer-edited)" : ""}`}
                                  className={`inline-block size-2 ${overridden ? "rounded-full ring-1 ring-foreground/40" : "rounded-sm"} ${judgmentDotClass(j)}`}
                                />
                              );
                            })}
                          </div>
                        </div>
                        <div className="text-sm leading-snug line-clamp-2 max-h-[2.75em] overflow-hidden">{r.title}</div>
                      </button>
                    );
                  })}
                  {filtered.length === 0 && (
                    <div className="p-4 text-sm text-muted-foreground">No papers match “{q}”.</div>
                  )}
                </div>
              </div>

              {/* RIGHT: selected paper appraisal */}
              <div className="flex-1 min-w-0 rounded-md border overflow-hidden flex flex-col">
                {!selected ? (
                  <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
                    Select a paper on the left.
                  </div>
                ) : (
                  <PaperDetail
                    key={selected.paper_id}
                    report={selected}
                    overrides={overrides}
                    excluded={s.excludedByQuality.has(selected.paper_id)}
                    paperOverrides={overrides.filter(o => o.paper_id === selected.paper_id)}
                    onToggleExclude={() => toggleExclude(selected.paper_id)}
                    onOverride={(o) => {
                      s.addQualityOverride(o);
                      toast.success(`Override saved for ${o.domain_id}`);
                      applyExcludeRule(excludeRule);
                    }}
                    onRevertDomain={(domainId) => {
                      s.setQualityOverrides(
                        overrides.filter(o => !(o.paper_id === selected.paper_id && o.domain_id === domainId)),
                      );
                      toast.info(`Reverted ${domainId} to AI judgment.`);
                      applyExcludeRule(excludeRule);
                    }}
                  />
                )}
              </div>
            </div>
          </Card>

          <div className="grid grid-cols-2 gap-2">
            <Button variant="outline" onClick={runFetchAndAssess} disabled={running}>Re-run Appraisal</Button>
            <Button onClick={proceedToScreening}>
              <ArrowRight className="size-4 mr-2" />Proceed to Abstract Screening ({kept})
            </Button>
          </div>
        </>
      )}
    </div>
  );
}

// ---- helpers / sub-components --------------------------------------------

function PaperDetail({
  report,
  excluded,
  overrides,
  paperOverrides,
  onToggleExclude,
  onOverride,
  onRevertDomain,
}: {
  report: QualityReport;
  excluded: boolean;
  overrides: QualityOverride[];
  paperOverrides: QualityOverride[];
  onToggleExclude: () => void;
  onOverride: (o: QualityOverride) => void;
  onRevertDomain: (domainId: string) => void;
}) {
  const overall = recomputeOverall(report, overrides);

  return (
    <>
      <div className="border-b p-4 space-y-3">
        <div className="flex items-start gap-3">
          <label
            className="flex items-center gap-1.5 text-xs shrink-0 pt-0.5 cursor-pointer"
            title={excluded ? "Excluded by quality — check to carry forward" : "Carrying forward to screening"}
          >
            <Checkbox checked={!excluded} onCheckedChange={onToggleExclude} />
            <span className="text-muted-foreground">Keep</span>
          </label>
          <div className="flex-1 min-w-0">
            <div className="font-medium leading-snug">{report.title}</div>
            <div className="flex flex-wrap items-center gap-2 text-xs mt-2">
              <OverallBadge judgment={overall.judgment} />
              <Badge variant="outline">{report.study_design || "Other"}</Badge>
              <Badge variant="outline">{report.rubric}</Badge>
              <Badge variant="outline">{report.source}</Badge>
              {!report.used_full_text && (
                <Badge variant="outline" className="bg-amber-50 text-amber-700 border-amber-200">
                  Abstract only
                </Badge>
              )}
              {report.url && (
                <a href={report.url} target="_blank" rel="noreferrer" className="text-primary hover:underline">
                  source
                </a>
              )}
            </div>
          </div>
        </div>
        <div className="text-xs text-muted-foreground italic">
          Overall: <span className="not-italic font-medium">{overall.judgment}</span>
          {" — "}{overall.rationale}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-3">
        <DomainList report={report} overrides={overrides} onOverride={onOverride} />
        {paperOverrides.length > 0 && (
          <AuditLogPanel
            paperOverrides={paperOverrides}
            report={report}
            onRevert={onRevertDomain}
          />
        )}
      </div>
    </>
  );
}

function Stat({ label, value, variant, icon }: {
  label: string; value: any; variant?: "success" | "warn" | "info"; icon?: React.ReactNode;
}) {
  const cls = variant === "success" ? "text-green-700"
    : variant === "warn" ? "text-amber-700"
    : variant === "info" ? "text-primary"
    : "text-foreground";
  return (
    <Card className="p-3 text-center">
      <div className={`text-2xl font-bold ${cls} flex items-center justify-center gap-1`}>{icon}{value}</div>
      <div className="text-xs text-muted-foreground">{label}</div>
    </Card>
  );
}

function OverallBadge({ judgment }: { judgment: RoBJudgment }) {
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium border ${judgmentClass(judgment)}`}>
      {judgment === "High" && <AlertTriangle className="inline size-3 mr-1" />}
      {judgment === "Low" && <ShieldCheck className="inline size-3 mr-1" />}
      {judgment}
    </span>
  );
}

function DomainList({
  report,
  overrides,
  onOverride,
}: {
  report: QualityReport;
  overrides: QualityOverride[];
  onOverride: (o: QualityOverride) => void;
}) {
  return (
    <div className="space-y-2">
      {report.domains.map((d) => {
        const override = latestOverride(report.paper_id, d.id, overrides);
        const effective: RoBJudgment = override ? override.new_judgment : d.judgment;
        return (
          <div key={d.id} className="rounded border bg-card p-3">
            <div className="flex items-start justify-between gap-3">
              <div className="flex-1">
                <div className="text-sm font-medium">{d.name}</div>
                <div className="text-sm text-muted-foreground mt-1">{d.rationale}</div>
              </div>
              <OverridePopover
                paperId={report.paper_id}
                domain={d}
                currentJudgment={effective}
                aiJudgment={d.judgment}
                onSave={onOverride}
              />
            </div>
            {override && (
              <div className="mt-2 text-[11px] text-muted-foreground flex items-center gap-2">
                <Pencil className="size-3" />
                Reviewer-edited from{" "}
                <span className="font-medium">{override.original_judgment}</span>
                {" — "}
                <span className="italic">"{override.reason}"</span>
                {" — "}
                {new Date(override.timestamp).toLocaleString()}
              </div>
            )}
            {d.supporting_quote && (
              <div className="mt-2 text-xs italic border-l-2 border-muted-foreground/30 pl-2 text-foreground/80 break-words">
                "{d.supporting_quote}"
                {d.section && (
                  <span className="not-italic text-muted-foreground ml-2">— {d.section}</span>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function OverridePopover({
  paperId,
  domain,
  currentJudgment,
  aiJudgment,
  onSave,
}: {
  paperId: string;
  domain: RoBDomain;
  currentJudgment: RoBJudgment;
  aiJudgment: RoBJudgment;
  onSave: (o: QualityOverride) => void;
}) {
  const [open, setOpen] = useState(false);
  const [nextJudgment, setNextJudgment] = useState<RoBJudgment>(currentJudgment);
  const [reason, setReason] = useState("");

  function handleSave() {
    if (nextJudgment === currentJudgment) {
      toast.info("Pick a different judgment to record an override.");
      return;
    }
    if (!reason.trim()) {
      toast.error("Provide a brief reason for the override.");
      return;
    }
    onSave({
      paper_id: paperId,
      domain_id: domain.id,
      original_judgment: aiJudgment,
      new_judgment: nextJudgment,
      reason: reason.trim(),
      reviewer: "",            // populated when auth is configured
      timestamp: new Date().toISOString(),
    });
    setReason("");
    setOpen(false);
  }

  return (
    <Popover open={open} onOpenChange={(v) => {
      setOpen(v);
      if (v) {
        setNextJudgment(currentJudgment);
        setReason("");
      }
    }}>
      <PopoverTrigger asChild>
        <button
          className={`shrink-0 inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium border ${judgmentClass(currentJudgment)} hover:ring-1 hover:ring-foreground/30`}
          title="Click to override this judgment"
        >
          {currentJudgment}
          <Pencil className="size-3 opacity-60" />
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-80" align="end">
        <div className="space-y-3">
          <div className="text-xs uppercase tracking-wide text-muted-foreground">
            Override judgment
          </div>
          <div className="text-sm font-medium break-words">{domain.name}</div>
          <div className="text-xs text-muted-foreground">
            AI judgment: <span className="font-medium text-foreground">{aiJudgment}</span>
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">New judgment</Label>
            <Select value={nextJudgment} onValueChange={(v) => setNextJudgment(v as RoBJudgment)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {JUDGMENT_OPTIONS.map(j => (
                  <SelectItem key={j} value={j}>{j}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">Reason</Label>
            <Textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="Why does this domain warrant a different judgment?"
              rows={3}
            />
          </div>
          <div className="flex justify-end gap-2 pt-1">
            <Button size="sm" variant="ghost" onClick={() => setOpen(false)}>Cancel</Button>
            <Button size="sm" onClick={handleSave}>Save override</Button>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}

function AuditLogPanel({
  paperOverrides,
  report,
  onRevert,
}: {
  paperOverrides: QualityOverride[];
  report: QualityReport;
  onRevert: (domainId: string) => void;
}) {
  // Aggregate one row per (domain) showing the most recent change. Earlier
  // changes for the same domain are listed underneath in muted text.
  const byDomain = new Map<string, QualityOverride[]>();
  for (const o of paperOverrides) {
    const arr = byDomain.get(o.domain_id) || [];
    arr.push(o);
    byDomain.set(o.domain_id, arr);
  }
  // Preserve domain order from the report.
  const ordered = report.domains
    .map(d => ({ domain: d, history: byDomain.get(d.id) || [] }))
    .filter(x => x.history.length > 0);

  if (ordered.length === 0) return null;

  return (
    <div className="rounded border bg-muted/20 p-3 space-y-2">
      <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
        <History className="size-3" /> Audit log — {paperOverrides.length} change{paperOverrides.length === 1 ? "" : "s"}
      </div>
      {ordered.map(({ domain, history }) => {
        const latest = history[history.length - 1];
        const earlier = history.slice(0, -1);
        return (
          <div key={domain.id} className="text-xs space-y-1">
            <div className="flex items-start justify-between gap-3">
              <div className="flex-1 min-w-0">
                <div className="font-medium">{domain.name}</div>
                <div className="text-muted-foreground">
                  <span className="line-through">{latest.original_judgment}</span>
                  {" → "}
                  <span className="text-foreground font-medium">{latest.new_judgment}</span>
                  {"  ·  "}
                  <span className="italic">"{latest.reason}"</span>
                </div>
                <div className="text-muted-foreground">
                  {new Date(latest.timestamp).toLocaleString()}
                  {latest.reviewer && ` · ${latest.reviewer}`}
                </div>
              </div>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => onRevert(domain.id)}
                className="shrink-0 h-7"
              >
                <RotateCcw className="size-3 mr-1" /> Revert
              </Button>
            </div>
            {earlier.length > 0 && (
              <details className="text-muted-foreground/80 pl-2">
                <summary className="cursor-pointer text-[11px]">{earlier.length} earlier change{earlier.length === 1 ? "" : "s"}</summary>
                <ul className="mt-1 space-y-0.5 pl-2 border-l border-muted-foreground/20">
                  {earlier.map((o, i) => (
                    <li key={i}>
                      <span className="line-through">{o.original_judgment}</span>
                      {" → "}
                      <span>{o.new_judgment}</span>
                      {"  ·  "}
                      <span className="italic">"{o.reason}"</span>
                      {" · "}
                      {new Date(o.timestamp).toLocaleString()}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        );
      })}
    </div>
  );
}

