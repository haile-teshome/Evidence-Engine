import { useEffect, useMemo, useState } from "react";
import { useStore } from "../lib/store";
import {
  QualityService, Paper,
  QualityReport, QualityOverride, RoBDomain, RoBJudgment, Instrument,
} from "../lib/mockServices";
import { effectiveAbstractDecision, effectiveFullTextDecision } from "../lib/exclusionBucketing";
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
  Pencil, History, RotateCcw, FileDown, Plus, Trash2, Loader2,
} from "lucide-react";
import {
  gradeCertainty, GRADE_DOWNGRADE_DOMAINS, GRADE_UPGRADE_FACTORS, GradeOutcome,
} from "../lib/apiClient";
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

// Generic severity tier for ANY instrument's ordinal judgment (case-insensitive),
// so the same colours work across RoB 2 / ROBINS-I / QUADAS-2 / PROBAST.
const _SEV: Record<string, number> = {
  "low": 0,
  "some concerns": 1, "unclear": 1,
  "moderate": 2,
  "high": 3, "serious": 3,
  "critical": 4,
};
function sevRank(j: string): number {
  const v = _SEV[(j || "").toLowerCase()];
  return v === undefined ? -1 : v;   // -1 = no information / not applicable
}
function judgmentDotClass(j: RoBJudgment): string {
  const r = sevRank(j);
  if (r < 0) return "bg-slate-300";
  if (r === 0) return "bg-emerald-500";
  if (r <= 2) return "bg-amber-500";
  if (r === 3) return "bg-rose-500";
  return "bg-rose-700";              // critical
}

// Colour for a signalling-question answer (Y/PY/PN/N/NI) — a neutral indicator
// of the answer itself, independent of per-signal polarity.
function answerClass(a: string): string {
  const v = (a || "").toUpperCase();
  if (v === "Y" || v === "PY") return "bg-emerald-100 text-emerald-700 border-emerald-200";
  if (v === "N" || v === "PN") return "bg-amber-100 text-amber-700 border-amber-200";
  return "bg-slate-100 text-slate-600 border-slate-200";
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

/** Overall judgment. With NO reviewer overrides this is the backend's precise
 *  instrument roll-up; once a domain is overridden we recompute generically as
 *  the worst (highest-severity) effective domain judgment. */
function recomputeOverall(
  report: QualityReport,
  overrides: QualityOverride[],
): { judgment: RoBJudgment; rationale: string } {
  const hasOverride = report.domains.some(d => latestOverride(report.paper_id, d.id, overrides));
  if (!hasOverride) {
    return { judgment: report.overall_judgment, rationale: report.overall_rationale };
  }
  const eff = report.domains.map(d => effectiveJudgment(report.paper_id, d, overrides));
  const rated = eff.filter(j => sevRank(j) >= 0);
  if (rated.length === 0) return { judgment: "No information", rationale: "All domains lacked sufficient information." };
  let worst = rated[0];
  for (const j of rated) if (sevRank(j) > sevRank(worst)) worst = j;
  const n = rated.filter(j => j === worst).length;
  return { judgment: worst, rationale: `${n} domain(s) judged ${worst} (after reviewer edits).` };
}

// ---- page -----------------------------------------------------------------

export function QualityPage() {
  const s = useStore();
  const task = s.tasks["quality-assess"];
  const running = task?.status === "running";

  const [excludeRule, setExcludeRule] = useState<"none" | "any_high" | "two_or_more_high">("none");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [q, setQ] = useState("");
  // Available appraisal instruments (for the "choose framework" selector). The
  // backend routes one by study design, but the reviewer can override per paper.
  const [instruments, setInstruments] = useState<Instrument[]>([]);
  const [reassessingId, setReassessingId] = useState<string | null>(null);
  // Global framework applied to every paper on Re-run ("__auto" = match by design).
  const [bulkInstrument, setBulkInstrument] = useState<string>("__auto");
  useEffect(() => {
    let alive = true;
    QualityService.listInstruments()
      .then(list => { if (alive) setInstruments(list); })
      .catch(() => { /* selector just won't render options; auto-routing still works */ });
    return () => { alive = false; };
  }, []);

  // Archive `outgoing` (keep at most one saved entry per (paper, instrument)) so a
  // reviewer never loses a prior appraisal when switching frameworks.
  function archiveReport(outgoing: QualityReport | undefined, incomingInstrumentId?: string) {
    if (!outgoing || !outgoing.instrument_id) return;
    if (outgoing.instrument_id === incomingInstrumentId) return;   // same tool re-run — nothing to keep
    s.setQualityArchive(prev => [
      outgoing,
      ...prev.filter(a => !(a.paper_id === outgoing.paper_id
        && (a.instrument_id === outgoing.instrument_id || a.instrument_id === incomingInstrumentId))),
    ]);
  }

  // Re-appraise ONE paper with a reviewer-chosen instrument (framework override).
  // The previous appraisal is saved (not discarded), so both remain available.
  async function reassessOne(report: QualityReport, instrumentId: string) {
    const paper: Paper = { id: report.paper_id, source: report.source, title: report.title, abstract: report.abstract, url: report.url };
    setReassessingId(report.paper_id);
    try {
      const fullText = s.fullTexts[report.paper_id]?.text || undefined;
      const fresh = await QualityService.assessPaper(paper, undefined, {
        fullText, instrumentId, studyDesign: report.study_design,
      });
      fresh.assessed_at = new Date().toISOString();
      archiveReport(report, fresh.instrument_id);
      s.setQualityReports((s.qualityReports || []).map(r => r.paper_id === report.paper_id ? fresh : r));
      // A new instrument means new domains — old overrides for this paper no longer apply.
      s.setQualityOverrides(s.qualityOverrides.filter(o => o.paper_id !== report.paper_id));
      applyExcludeRule(excludeRule);
      const inst = instruments.find(i => i.id === instrumentId);
      toast.success(`Re-appraised with ${inst?.short_name || instrumentId}.`);
    } catch (e: any) {
      toast.error(e?.message || "Re-appraisal failed");
    } finally {
      setReassessingId(null);
    }
  }

  // Make a saved appraisal the active one (swapping the current active into the archive).
  function activateArchived(archived: QualityReport) {
    const pid = archived.paper_id;
    const current = (s.qualityReports || []).find(r => r.paper_id === pid);
    // Remove the one we're activating; archive the outgoing current in its place.
    s.setQualityArchive(prev => {
      const withoutTarget = prev.filter(a => !(a.paper_id === pid && a.instrument_id === archived.instrument_id));
      if (current && current.instrument_id && current.instrument_id !== archived.instrument_id) {
        return [current, ...withoutTarget.filter(a => !(a.paper_id === pid && a.instrument_id === current.instrument_id))];
      }
      return withoutTarget;
    });
    s.setQualityReports((s.qualityReports || []).map(r => r.paper_id === pid ? archived : r));
    s.setQualityOverrides(s.qualityOverrides.filter(o => o.paper_id !== pid));
    applyExcludeRule(excludeRule);
    toast.success(`Switched to ${archived.instrument || archived.rubric} appraisal.`);
  }

  function deleteArchived(archived: QualityReport) {
    s.setQualityArchive(prev => prev.filter(a => !(a.paper_id === archived.paper_id && a.instrument_id === archived.instrument_id)));
    toast.info("Saved appraisal removed.");
  }

  // Papers to appraise: the ones that PASSED screening. Risk-of-bias is only
  // meaningful for studies you're actually including — prefer full-text includes,
  // otherwise fall back to abstract includes.
  function includedPapers(): { papers: Paper[]; stage: "full-text" | "abstract" } {
    const toPaper = (r: any): Paper => ({ id: r.paper_id, source: r.Source, title: r.Title, abstract: r.Abstract, url: r.URL });
    if (s.fullTextResults) {
      const inc = s.fullTextResults.filter(r => effectiveFullTextDecision(r, s.fullTextOverrides) === "Include");
      if (inc.length) return { papers: inc.map(toPaper), stage: "full-text" };
    }
    if (s.results) {
      const inc = s.results.filter(r => effectiveAbstractDecision(r, s.abstractOverrides) === "INCLUDE");
      if (inc.length) return { papers: inc.map(toPaper), stage: "abstract" };
    }
    return { papers: [], stage: "abstract" };
  }

  // Appraise all included papers. With `forceInstrumentId` every paper is assessed
  // with that one tool (a global framework override); otherwise each is auto-routed
  // by study design. Any prior appraisal replaced by a *different* instrument is
  // saved to the archive so results are never lost.
  async function runAssess(forceInstrumentId?: string) {
    const { papers, stage } = includedPapers();
    if (papers.length === 0) {
      toast.error("No included papers yet — run Abstract or Full-Text screening first.");
      return;
    }
    const priorById = new Map((s.qualityReports || []).map(r => [r.paper_id, r]));
    const forcedName = forceInstrumentId ? (instruments.find(i => i.id === forceInstrumentId)?.short_name || forceInstrumentId) : "";
    const { abort } = s.startTask("quality-assess", [{ id: "qa", label: "Quality assessment", status: "running" }]);
    const signal = abort.signal;
    try {
      const reports: QualityReport[] = [];
      s.updateTask("quality-assess", { progress: { done: 0, total: papers.length }, detail: `Appraising ${papers.length} included papers${forcedName ? ` with ${forcedName}` : ""}…` });
      for (let i = 0; i < papers.length; i++) {
        if (signal.aborted) break;
        const p = papers[i];
        s.updateTask("quality-assess", {
          progress: { done: i, total: papers.length, label: p.title.slice(0, 80) },
          detail: p.title.slice(0, 80),
        });
        try {
          // Full text (when acquired) yields far better risk-of-bias judgments
          // than the abstract alone — the backend uses it when provided.
          const fullText = s.fullTexts[p.id]?.text || undefined;
          const fresh = await QualityService.assessPaper(p, signal, {
            fullText,
            instrumentId: forceInstrumentId,
            studyDesign: priorById.get(p.id)?.study_design,
          });
          fresh.assessed_at = new Date().toISOString();
          reports.push(fresh);
        } catch (e: any) {
          if (signal.aborted) break;
          console.error(`quality-assess ${i + 1} failed:`, e?.message);
        }
        s.updateTask("quality-assess", { progress: { done: i + 1, total: papers.length } });
      }
      // Save each outgoing appraisal that a different instrument is replacing.
      if (priorById.size > 0) {
        s.setQualityArchive(prev => {
          let next = [...prev];
          for (const fresh of reports) {
            const prior = priorById.get(fresh.paper_id);
            if (prior && prior.instrument_id && fresh.instrument_id && prior.instrument_id !== fresh.instrument_id) {
              next = next.filter(a => !(a.paper_id === prior.paper_id && (a.instrument_id === prior.instrument_id || a.instrument_id === fresh.instrument_id)));
              next = [prior, ...next];
            }
          }
          return next;
        });
      }
      s.setQualityReports(reports);
      s.setExcludedByQuality(new Set());
      // Fresh assessment invalidates previous overrides.
      s.setQualityOverrides([]);
      if (signal.aborted) {
        s.updateTask("quality-assess", { status: "canceled" });
        toast.info(`Canceled — ${reports.length} of ${papers.length} assessed`);
      } else {
        s.updateTask("quality-assess", { status: "done" });
        const withFT = papers.filter(p => s.fullTexts[p.id]?.text).length;
        toast.success(`Appraised ${reports.length} included papers${forcedName ? ` with ${forcedName}` : ` (${stage})`}${withFT ? ` — ${withFT} using full text` : ""}.`);
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
    // Tier by severity so every instrument's scale rolls into the same buckets.
    let low = 0, some = 0, high = 0, no_info = 0;
    for (const r of reports) {
      const rank = sevRank(recomputeOverall(r, overrides).judgment);
      if (rank < 0) no_info++;
      else if (rank === 0) low++;
      else if (rank <= 2) some++;    // some concerns / unclear / moderate
      else high++;                    // high / serious / critical
    }
    return { low, some, high, no_info };
  }, [reports, overrides]);

  // All hooks are declared above; only now is it safe to bail out early.
  if (s.history.length === 0) {
    return <Alert><AlertDescription>Define a research goal on the Home page first.</AlertDescription></Alert>;
  }

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
              Risk-of-bias appraisal of your <strong>included</strong> papers, with a rubric matched to each study
              design (RoB 2, ROBINS-I, JBI, AMSTAR 2). Uses the acquired full text where available for more reliable
              judgments; reviewer overrides are audit-logged.
            </AlertDescription>
          </Alert>
          {(() => {
            const { papers, stage } = includedPapers();
            const withFT = papers.filter(p => s.fullTexts[p.id]?.text).length;
            return papers.length === 0 ? (
              <Alert><AlertDescription>No included papers yet — run Abstract or Full-Text screening first, then come back to appraise risk of bias.</AlertDescription></Alert>
            ) : (
              <div className="text-xs text-muted-foreground px-1">
                {papers.length} included paper{papers.length === 1 ? "" : "s"} ({stage} screening){withFT ? ` · ${withFT} with full text` : " · no full text acquired yet — abstract-only appraisal"}
              </div>
            );
          })()}
          {task && task.status === "running" && (
            <TaskProgressCard
              task={task}
              title="Quality assessment"
              onCancel={() => s.cancelTask("quality-assess")}
            />
          )}
          <Button onClick={() => runAssess()} disabled={running || includedPapers().papers.length === 0} size="lg" className="w-full">
            <ShieldCheck className="size-4 mr-2" />{running ? "Appraising…" : "Appraise Risk of Bias on Included Papers"}
          </Button>
        </>
      )}

      {reports && summaryCounts && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
            <Stat label="Appraised" value={reports.length} icon={<ShieldCheck className="size-4" />} />
            <Stat label="Low RoB" value={summaryCounts.low} variant="success" />
            <Stat label="Some concerns" value={summaryCounts.some} variant="warn" />
            <Stat label="High RoB" value={summaryCounts.high} variant="warn" icon={<AlertTriangle className="size-4" />} />
            <Stat label="No information" value={summaryCounts.no_info} />
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

            {/* Global framework: apply one tool to every paper, or auto-match by design. */}
            <div className="mt-3 pt-3 border-t flex flex-col md:flex-row md:items-center justify-between gap-2">
              <div>
                <div className="text-sm font-medium">Appraisal framework</div>
                <div className="text-xs text-muted-foreground">
                  Re-run all included papers with one instrument, or auto-match each to its study design.
                  Prior appraisals are saved per paper.
                </div>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <Select value={bulkInstrument} onValueChange={setBulkInstrument} disabled={running || instruments.length === 0}>
                  <SelectTrigger className="h-9 w-[240px] text-xs">
                    <SelectValue placeholder="Auto — match by design" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__auto">Auto — match each paper's design</SelectItem>
                    {["internal_validity", "reporting", "certainty"]
                      .filter(ax => instruments.some(i => i.axis === ax))
                      .map(ax => (
                        <div key={ax}>
                          <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">{AXIS_LABEL[ax]}</div>
                          {instruments.filter(i => i.axis === ax).map(i => (
                            <SelectItem key={i.id} value={i.id}>
                              {i.short_name}{i.legacy ? " (legacy)" : ""}
                            </SelectItem>
                          ))}
                        </div>
                      ))}
                  </SelectContent>
                </Select>
                <Button
                  size="sm"
                  onClick={() => runAssess(bulkInstrument !== "__auto" ? bulkInstrument : undefined)}
                  disabled={running}
                >
                  <RotateCcw className="size-4 mr-2" />{running ? "Re-running…" : "Re-run"}
                </Button>
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
                    instruments={instruments}
                    reassessing={reassessingId === selected.paper_id}
                    archived={s.qualityArchive.filter(a => a.paper_id === selected.paper_id)}
                    onReassess={(instrumentId) => reassessOne(selected, instrumentId)}
                    onActivateArchived={activateArchived}
                    onDeleteArchived={deleteArchived}
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

          <RobvisSummary reports={reports} overrides={overrides} />

          <GradePanel />

          <div className="grid grid-cols-3 gap-2">
            <Button variant="outline" onClick={() => runAssess()} disabled={running}>Re-run Appraisal</Button>
            <Button variant="outline" onClick={() => exportAssessments(reports, overrides)}>
              <FileDown className="size-4 mr-2" />Export (CSV + JSON)
            </Button>
            <Button variant="outline" onClick={() => s.setPage("prisma")}>
              <ArrowRight className="size-4 mr-2" />Continue to Diagramming
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
  instruments,
  reassessing,
  archived,
  onReassess,
  onActivateArchived,
  onDeleteArchived,
  onToggleExclude,
  onOverride,
  onRevertDomain,
}: {
  report: QualityReport;
  excluded: boolean;
  overrides: QualityOverride[];
  paperOverrides: QualityOverride[];
  instruments: Instrument[];
  reassessing: boolean;
  archived: QualityReport[];
  onReassess: (instrumentId: string) => void;
  onActivateArchived: (r: QualityReport) => void;
  onDeleteArchived: (r: QualityReport) => void;
  onToggleExclude: () => void;
  onOverride: (o: QualityOverride) => void;
  onRevertDomain: (domainId: string) => void;
}) {
  const overall = recomputeOverall(report, overrides);
  const currentInstrumentId = report.instrument_id
    || instruments.find(i => i.short_name === report.rubric)?.id
    || "";

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
              <FrameworkSelector
                instruments={instruments}
                currentId={currentInstrumentId}
                fallbackLabel={report.rubric}
                reassessing={reassessing}
                onChange={onReassess}
              />
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
        {archived.length > 0 && (
          <SavedAppraisals
            active={report}
            archived={archived}
            onActivate={onActivateArchived}
            onDelete={onDeleteArchived}
          />
        )}
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

// Lets the reviewer override the auto-routed appraisal framework for one paper.
// The backend still defaults by study design; this re-runs with a chosen tool.
const AXIS_LABEL: Record<string, string> = {
  internal_validity: "Risk of bias / internal validity",
  reporting: "Reporting completeness",
  certainty: "Certainty of evidence",
};

function FrameworkSelector({
  instruments,
  currentId,
  fallbackLabel,
  reassessing,
  onChange,
}: {
  instruments: Instrument[];
  currentId: string;
  fallbackLabel: string;
  reassessing: boolean;
  onChange: (instrumentId: string) => void;
}) {
  // No instrument list yet (backend not reached) → show the plain badge.
  if (instruments.length === 0) {
    return <Badge variant="outline">{fallbackLabel}</Badge>;
  }
  const axes = ["internal_validity", "reporting", "certainty"].filter(
    ax => instruments.some(i => i.axis === ax),
  );
  return (
    <div className="inline-flex items-center gap-1">
      <Select value={currentId} onValueChange={(v) => { if (v && v !== currentId) onChange(v); }} disabled={reassessing}>
        <SelectTrigger className="h-6 px-2 py-0 text-xs w-auto gap-1 border-dashed" title="Appraisal framework — reviewer can override the auto-routed choice">
          {reassessing ? <Loader2 className="size-3 animate-spin" /> : null}
          <SelectValue placeholder={fallbackLabel} />
        </SelectTrigger>
        <SelectContent>
          {axes.map(ax => (
            <div key={ax}>
              <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">{AXIS_LABEL[ax]}</div>
              {instruments.filter(i => i.axis === ax).map(i => (
                <SelectItem key={i.id} value={i.id} disabled={i.scaffold}>
                  {i.short_name}{i.scaffold ? " (coming soon)" : i.legacy ? " (legacy)" : ""}
                </SelectItem>
              ))}
            </div>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

// Prior appraisals kept when a paper is re-appraised under a different framework.
// The active one drives the analysis; saved ones can be viewed (made active) or removed.
function SavedAppraisals({
  active,
  archived,
  onActivate,
  onDelete,
}: {
  active: QualityReport;
  archived: QualityReport[];
  onActivate: (r: QualityReport) => void;
  onDelete: (r: QualityReport) => void;
}) {
  const fmt = (iso?: string) => {
    if (!iso) return "";
    try { return new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }
    catch { return ""; }
  };
  return (
    <div className="rounded-md border bg-muted/40 p-2 space-y-1.5">
      <div className="text-[11px] font-medium text-muted-foreground">Saved appraisals ({archived.length + 1})</div>
      <div className="flex flex-wrap gap-1.5">
        <span className="inline-flex items-center gap-1.5 rounded border border-primary/40 bg-primary/10 px-2 py-1 text-xs">
          <span className="font-medium">{active.instrument || active.rubric}</span>
          <Badge variant="outline" className={`${judgmentDotClass(active.overall_judgment)} h-4 px-1 text-[10px]`}>{active.overall_judgment}</Badge>
          <span className="text-[10px] text-muted-foreground">active</span>
        </span>
        {archived.map(a => (
          <span key={`${a.paper_id}:${a.instrument_id}`}
            className="group inline-flex items-center gap-1.5 rounded border px-2 py-1 text-xs hover:bg-background">
            <button className="font-medium hover:underline" title="Make this the active appraisal" onClick={() => onActivate(a)}>
              {a.instrument || a.rubric}
            </button>
            <Badge variant="outline" className={`${judgmentDotClass(a.overall_judgment)} h-4 px-1 text-[10px]`}>{a.overall_judgment}</Badge>
            {a.assessed_at && <span className="text-[10px] text-muted-foreground">{fmt(a.assessed_at)}</span>}
            <button className="text-muted-foreground hover:text-destructive" title="Remove this saved appraisal" onClick={() => onDelete(a)}>
              <Trash2 className="size-3" />
            </button>
          </span>
        ))}
      </div>
    </div>
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
            {d.signals && d.signals.length > 0 && (
              <details className="mt-2 group">
                <summary className="text-[11px] text-muted-foreground cursor-pointer select-none hover:text-foreground">
                  {d.signals.length} signalling question{d.signals.length === 1 ? "" : "s"} → judgment
                </summary>
                <div className="mt-1.5 space-y-1.5">
                  {d.signals.map(sig => (
                    <div key={sig.id} className="text-xs">
                      <div className="flex items-start gap-1.5">
                        <span className={`shrink-0 mt-0.5 px-1 py-px rounded border text-[10px] font-mono font-semibold ${answerClass(sig.answer)}`} title={sig.rationale}>
                          {sig.answer}
                        </span>
                        <span className="text-foreground/80 leading-snug"><span className="font-mono text-muted-foreground">{sig.id}</span> {sig.text}</span>
                      </div>
                      {sig.quote && (
                        <div className="italic text-muted-foreground pl-6 mt-0.5 break-words">
                          "{sig.quote}"{sig.section && <span className="ml-1 not-italic">— {sig.section}</span>}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </details>
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


// ---------------------------------------------------------------------------
// robvis-style traffic-light summary, generated from the stored per-domain
// judgments. Papers are grouped by instrument (each tool has its own domains).
// ---------------------------------------------------------------------------
function RobvisSummary({ reports, overrides }: { reports: QualityReport[]; overrides: QualityOverride[] }) {
  const byInstrument = new Map<string, QualityReport[]>();
  for (const r of reports) {
    const key = r.instrument || r.rubric || "Appraisal";
    (byInstrument.get(key) ?? byInstrument.set(key, []).get(key)!).push(r);
  }
  return (
    <Card className="p-4 space-y-4">
      <div className="flex items-center gap-2">
        <ShieldCheck className="size-4 text-primary" />
        <h3 className="font-medium">Risk-of-bias summary (traffic light)</h3>
      </div>
      {[...byInstrument.entries()].map(([inst, rs]) => {
        const domains = rs[0]?.domains ?? [];
        return (
          <div key={inst} className="space-y-1.5 overflow-x-auto">
            <div className="text-xs font-medium text-muted-foreground">{inst} · {rs.length} paper{rs.length === 1 ? "" : "s"}</div>
            <table className="text-xs border-separate border-spacing-0.5">
              <thead>
                <tr>
                  <th className="text-left font-normal text-muted-foreground pr-2 max-w-[16rem]">Study</th>
                  {domains.map((d, i) => (
                    <th key={d.id} className="px-1 font-normal text-muted-foreground" title={d.name}>D{i + 1}</th>
                  ))}
                  <th className="px-1 font-normal text-muted-foreground">Overall</th>
                </tr>
              </thead>
              <tbody>
                {rs.map(r => {
                  const overall = recomputeOverall(r, overrides).judgment;
                  return (
                    <tr key={r.paper_id}>
                      <td className="pr-2 truncate max-w-[16rem] align-middle" title={r.title}>{r.title}</td>
                      {r.domains.map(d => {
                        const j = effectiveJudgment(r.paper_id, d, overrides);
                        return (
                          <td key={d.id} className="text-center">
                            <span title={`${d.name}: ${j}`} className={`inline-block size-4 rounded-full ${judgmentDotClass(j)}`} />
                          </td>
                        );
                      })}
                      <td className="text-center">
                        <span title={`Overall: ${overall}`} className={`inline-block size-4 rounded-full ring-1 ring-foreground/20 ${judgmentDotClass(overall)}`} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        );
      })}
      <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground pt-1 border-t">
        {[["Low", "bg-emerald-500"], ["Some concerns / Unclear / Moderate", "bg-amber-500"], ["High / Serious", "bg-rose-500"], ["Critical", "bg-rose-700"], ["No information", "bg-slate-300"]].map(([l, c]) => (
          <span key={l} className="flex items-center gap-1"><span className={`inline-block size-2.5 rounded-full ${c}`} />{l}</span>
        ))}
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// GRADE — certainty of the body of evidence, per user-defined OUTCOME. This axis
// is outcome-level (not study-level); certainty rolls up deterministically from
// the downgrade/upgrade selections (mirrors Backend /api/grade).
// ---------------------------------------------------------------------------
const GRADE_DOMAIN_LABELS: Record<string, string> = {
  risk_of_bias: "Risk of bias", inconsistency: "Inconsistency", indirectness: "Indirectness",
  imprecision: "Imprecision", publication_bias: "Publication bias",
};
const GRADE_UPGRADE_LABELS: Record<string, string> = {
  large_effect: "Large effect", dose_response: "Dose-response", plausible_confounding: "Plausible confounding",
};
function gradeCertaintyClass(c: string): string {
  return c === "High" ? "bg-emerald-100 text-emerald-700 border-emerald-200"
    : c === "Moderate" ? "bg-lime-100 text-lime-700 border-lime-200"
    : c === "Low" ? "bg-amber-100 text-amber-700 border-amber-200"
    : "bg-rose-100 text-rose-700 border-rose-200";
}

function GradePanel() {
  const s = useStore();
  const outcomes = s.gradeOutcomes;
  const [name, setName] = useState("");

  function add() {
    const o = name.trim();
    if (!o) return;
    s.setGradeOutcomes(prev => [...prev, {
      id: Date.now().toString(36), outcome: o, starting: "randomized",
      downgrades: {}, upgrades: {},
    }]);
    setName("");
  }
  function update(id: string, patch: Partial<GradeOutcome>) {
    s.setGradeOutcomes(prev => prev.map(o => o.id === id ? { ...o, ...patch } : o));
  }

  return (
    <Card className="p-4 space-y-3">
      <div className="flex items-center gap-2">
        <ShieldCheck className="size-4 text-primary" />
        <h3 className="font-medium">Certainty of evidence — GRADE (per outcome)</h3>
        <span className="text-xs text-muted-foreground">Outcome-level, separate from study risk of bias.</span>
      </div>
      <div className="flex gap-2">
        <Input value={name} onChange={e => setName(e.target.value)} placeholder="Add an outcome, e.g. All-cause mortality"
          onKeyDown={e => { if (e.key === "Enter") { e.preventDefault(); add(); } }} className="h-8 text-sm" />
        <Button size="sm" onClick={add} disabled={!name.trim()}><Plus className="size-3.5 mr-1" />Add outcome</Button>
      </div>
      {outcomes.length === 0 && <div className="text-xs text-muted-foreground">No outcomes yet — add the outcomes you'll rate.</div>}
      <div className="space-y-3">
        {outcomes.map(o => {
          const certainty = gradeCertainty(o);
          return (
            <div key={o.id} className="rounded border p-3 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="font-medium text-sm">{o.outcome}</div>
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-0.5 rounded border text-xs font-medium ${gradeCertaintyClass(certainty)}`}>{certainty} certainty</span>
                  <button onClick={() => s.setGradeOutcomes(prev => prev.filter(x => x.id !== o.id))} title="Remove outcome" className="text-muted-foreground hover:text-rose-600"><Trash2 className="size-3.5" /></button>
                </div>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="text-muted-foreground">Starting certainty:</span>
                {(["randomized", "observational"] as const).map(st => (
                  <Button key={st} size="sm" variant={o.starting === st ? "default" : "outline"} className="h-6 px-2 text-xs"
                    onClick={() => update(o.id, { starting: st })}>{st === "randomized" ? "Randomized (High)" : "Observational (Low)"}</Button>
                ))}
              </div>
              <div className="grid md:grid-cols-2 gap-x-6 gap-y-1">
                <div>
                  <div className="text-[11px] font-medium text-muted-foreground mb-1">Downgrade</div>
                  {GRADE_DOWNGRADE_DOMAINS.map(k => (
                    <div key={k} className="flex items-center justify-between text-xs py-0.5">
                      <span>{GRADE_DOMAIN_LABELS[k]}</span>
                      <div className="flex gap-0.5">
                        {[0, -1, -2].map(v => (
                          <button key={v} onClick={() => update(o.id, { downgrades: { ...o.downgrades, [k]: v } })}
                            className={`w-8 h-6 rounded border text-[11px] ${(o.downgrades[k] || 0) === v ? "bg-primary text-primary-foreground border-primary" : "bg-card hover:bg-muted"}`}>
                            {v === 0 ? "0" : v}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
                {o.starting === "observational" && (
                  <div>
                    <div className="text-[11px] font-medium text-muted-foreground mb-1">Upgrade (observational)</div>
                    {GRADE_UPGRADE_FACTORS.map(k => (
                      <div key={k} className="flex items-center justify-between text-xs py-0.5">
                        <span>{GRADE_UPGRADE_LABELS[k]}</span>
                        <div className="flex gap-0.5">
                          {[0, 1, 2].map(v => (
                            <button key={v} onClick={() => update(o.id, { upgrades: { ...o.upgrades, [k]: v } })}
                              className={`w-8 h-6 rounded border text-[11px] ${(o.upgrades[k] || 0) === v ? "bg-primary text-primary-foreground border-primary" : "bg-card hover:bg-muted"}`}>
                              {v === 0 ? "0" : `+${v}`}
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

// ---- export ---------------------------------------------------------------
function _dl(name: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}
function exportAssessments(reports: QualityReport[], overrides: QualityOverride[]) {
  // JSON: full assessments incl. signalling answers + overrides.
  _dl("quality_assessments.json", JSON.stringify({ reports, overrides }, null, 2), "application/json");
  // CSV: one row per (paper, domain) with the effective judgment.
  const esc = (x: any) => `"${String(x ?? "").replace(/"/g, '""')}"`;
  const rows = [["paper_id", "title", "instrument", "study_design", "domain", "judgment", "overridden", "supporting_quote", "section", "overall"].join(",")];
  for (const r of reports) {
    const overall = recomputeOverall(r, overrides).judgment;
    for (const d of r.domains) {
      const j = effectiveJudgment(r.paper_id, d, overrides);
      const overridden = !!latestOverride(r.paper_id, d.id, overrides);
      rows.push([r.paper_id, r.title, r.instrument || r.rubric, r.study_design, d.name, j, overridden, d.supporting_quote, d.section, overall].map(esc).join(","));
    }
  }
  _dl("quality_assessments.csv", rows.join("\n"), "text/csv");
  toast.success(`Exported ${reports.length} appraisals (CSV + JSON).`);
}
