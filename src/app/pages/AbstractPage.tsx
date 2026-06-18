import { useState } from "react";
import { useStore } from "../lib/store";
import { AIService, DataAggregator, Deduplicator, formatDuration, Paper, ScreenResult, PicoVote, PicoFieldAssessment } from "../lib/mockServices";
import { categoriseAbstractExclusion, effectiveAbstractDecision } from "../lib/exclusionBucketing";
import { ProjectScreeningBar, recordProjectDecision } from "../components/ProjectScreeningBar";
import { Checkbox } from "../components/ui/checkbox";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Popover, PopoverContent, PopoverTrigger } from "../components/ui/popover";
import { Search, Download, Minus } from "lucide-react";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";

// ---- styling helpers -------------------------------------------------------

function picoVoteClass(v: PicoVote): string {
  switch (v) {
    case "PASS":    return "bg-emerald-100 text-emerald-800 border-emerald-200";
    case "PARTIAL": return "bg-amber-100 text-amber-800 border-amber-200";
    case "FAIL":    return "bg-rose-100 text-rose-800 border-rose-200";
    case "NA":      return "bg-slate-100 text-slate-600 border-slate-200";
    default:        return "bg-slate-100 text-slate-600 border-slate-200";
  }
}

function picoVoteShort(v: PicoVote): string {
  // Compact label so the column cells stay narrow. NA is intentionally not
  // mapped here — NA cells render as an empty placeholder via PicoCell, not
  // as a chip.
  switch (v) {
    case "PASS":    return "✓";
    case "PARTIAL": return "~";
    case "FAIL":    return "✗";
    default:        return "";
  }
}

function picoVoteFullLabel(v: PicoVote): string {
  switch (v) {
    case "PASS":    return "Match";
    case "PARTIAL": return "Partial";
    case "FAIL":    return "Mismatch";
    default:        return "Not assessed";
  }
}

// ---- page ------------------------------------------------------------------

export function AbstractPage() {
  const s = useStore();
  const task = s.tasks["abstract-screen"];
  const running = task?.status === "running";

  async function runSearch() {
    if (!s.query) { toast.error("Define a research goal on the Home page first."); return; }

    const { abort } = s.startTask("abstract-screen", [
      { id: "fetch",  label: "Fetching papers",  status: "pending" },
      { id: "screen", label: "Screening papers", status: "pending" },
    ]);
    const signal = abort.signal;
    const start = Date.now();

    // Decide where the paper queue comes from:
    //   1. If Quality Assessment has already produced uniquePapers, use those
    //      (and honour any QA exclusions).
    //   2. Otherwise fetch + deduplicate inline, so screening doesn't require
    //      the QA step.
    let queue: Paper[] = [];
    try {
      if (s.uniquePapers) {
        queue = s.uniquePapers.filter(p => !s.excludedByQuality.has(p.id));
        s.updateTask("abstract-screen", {
          stages: [
            { id: "fetch", label: "Using existing paper set from Quality Assessment", status: "done" },
            { id: "screen", label: "Screening papers", status: "running" },
          ],
        });
        if (queue.length === 0) {
          s.updateTask("abstract-screen", { status: "error", detail: "All papers were excluded in Quality Assessment." });
          toast.error("All papers were excluded in quality assessment.");
          return;
        }
      } else {
        s.updateTask("abstract-screen", {
          detail: "Fetching papers from databases…",
          stages: [
            { id: "fetch", label: "Fetching papers", status: "running" },
            { id: "screen", label: "Screening papers", status: "pending" },
          ],
        });
        const { papers: all } = await DataAggregator.fetchAll(s.query, s.sources, s.pico, s.numPerSource, signal, s.elsevierToken, s.ezproxyConnected);
        if (signal.aborted) { s.updateTask("abstract-screen", { status: "canceled" }); return; }
        const { unique, duplicates } = Deduplicator.run(all);
        // Persist so a later QA run can reuse this set, and so the PRISMA flow
        // has accurate counts even when the user skipped QA.
        s.setRawPapers(all);
        s.setUniquePapers(unique);
        s.setDuplicatesCount(duplicates.length);
        queue = unique;
        if (queue.length === 0) {
          s.updateTask("abstract-screen", { status: "error", detail: "No papers retrieved for this query." });
          toast.error("No papers retrieved for this query — broaden it on the Home page.");
          return;
        }
        s.updateTask("abstract-screen", {
          stages: [
            { id: "fetch", label: `Fetched ${all.length} papers (${unique.length} unique)`, status: "done" },
            { id: "screen", label: "Screening papers", status: "running" },
          ],
        });
      }
    } catch (e: any) {
      if (signal.aborted) { s.updateTask("abstract-screen", { status: "canceled" }); return; }
      s.updateTask("abstract-screen", { status: "error", detail: e?.message || "Fetch failed" });
      return;
    }

    s.updateTask("abstract-screen", { progress: { done: 0, total: queue.length } });
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
            const bucket = categoriseAbstractExclusion(r, s.inclusion, s.exclusion);
            reasons[bucket] = (reasons[bucket] || 0) + 1;
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
  // Stats reflect EFFECTIVE decisions (reviewer overrides applied) so a
  // reviewer flipping the Keep checkbox immediately updates the included /
  // excluded counts.
  const passed = r ? r.filter(x => effectiveAbstractDecision(x, s.abstractOverrides) === "INCLUDE") : [];
  const excluded = r ? r.filter(x => effectiveAbstractDecision(x, s.abstractOverrides) === "EXCLUDE") : [];
  const overrideCount = r ? r.filter(x => x.Decision !== effectiveAbstractDecision(x, s.abstractOverrides)).length : 0;

  const [projectRefresh, setProjectRefresh] = useState(0);

  return (
    <div className="space-y-4">
      {s.currentProjectId && r && (
        <ProjectScreeningBar
          stage="abstract"
          papers={r.map(x => ({ id: x.paper_id }))}
          refreshSignal={projectRefresh}
        />
      )}
      {!r && !s.uniquePapers && !s.query && (
        <Alert><AlertDescription>Define a research goal on the Home page first.</AlertDescription></Alert>
      )}
      {!r && !s.uniquePapers && s.query && (
        <Alert>
          <AlertDescription>
            Ready to screen against your PICO. Papers will be fetched and deduplicated on demand — Quality Assessment is optional and can still be run beforehand from the sidebar.
          </AlertDescription>
        </Alert>
      )}
      {!r && s.uniquePapers && (
        <Alert>
          <AlertDescription>
            {s.uniquePapers.length - s.excludedByQuality.size} of {s.uniquePapers.length} unique papers ready for screening{s.excludedByQuality.size > 0 ? ` (${s.excludedByQuality.size} excluded in Quality Assessment)` : ""}.
          </AlertDescription>
        </Alert>
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
            <div className="flex items-center justify-between mb-3 gap-3">
              <h3 className="font-medium">Screening Results</h3>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => downloadAbstractXlsx(r, s.abstractOverrides)}
                className="h-7 px-2 shrink-0"
                title="Download results as Excel (XLSX)"
              >
                <Download className="size-4" />
              </Button>
            </div>
            {overrideCount > 0 && (
              <div className="text-xs text-muted-foreground mb-2">
                {overrideCount} reviewer override{overrideCount === 1 ? "" : "s"} active —
                Decision column reflects the reviewer's choice.
              </div>
            )}
            <div className="rounded-md border max-h-[600px] overflow-auto">
              <table className="w-full text-sm border-collapse">
                <thead className="bg-muted sticky top-0 z-30">
                  <tr className="text-left">
                    <th className="px-3 py-2 sticky left-0 bg-muted z-40 border-b border-r min-w-[60px] text-center" title="Reviewer override — check to keep, uncheck to drop">Keep</th>
                    <th className="px-3 py-2 sticky left-[60px] bg-muted z-40 border-b border-r min-w-[120px]">Decision</th>
                    <th className="px-3 py-2 sticky left-[180px] bg-muted z-40 border-b border-r min-w-[260px] max-w-[260px] shadow-[6px_0_8px_-6px_rgba(0,0,0,0.22)]">Title</th>
                    <th className="px-3 py-2 border-b whitespace-nowrap">Source</th>
                    <th className="px-3 py-2 border-b text-center">P</th>
                    <th className="px-3 py-2 border-b text-center">I</th>
                    <th className="px-3 py-2 border-b text-center">C</th>
                    <th className="px-3 py-2 border-b text-center">O</th>
                    <th className="px-3 py-2 border-b min-w-[380px]">Reasoning</th>
                  </tr>
                </thead>
                <tbody>
                  {r.map(row => {
                    const pa = row.Pico_Assessment;
                    const eff = effectiveAbstractDecision(row, s.abstractOverrides);
                    const isOverridden = eff !== row.Decision;
                    const keep = eff === "INCLUDE";
                    return (
                      <tr key={row.paper_id} className="border-b last:border-b-0 align-top bg-card hover:bg-muted">
                        <td className="px-3 py-2 sticky left-0 z-20 border-r bg-inherit text-center">
                          <Checkbox
                            checked={keep}
                            onCheckedChange={(v) => {
                              const wantKeep = v === true;
                              // Local override (always — keeps in-memory state in sync).
                              if ((row.Decision === "INCLUDE") === wantKeep) {
                                s.clearAbstractOverride(row.paper_id);
                              } else {
                                s.setAbstractOverride(row.paper_id, wantKeep ? "INCLUDE" : "EXCLUDE");
                              }
                              // Project mode: persist as a per-reviewer decision.
                              // We record an explicit user decision (which doubles
                              // as an override if it differs from the AI call) so
                              // the multi-reviewer audit trail captures every
                              // human action.
                              if (s.currentProjectId) {
                                const aiDec = row.Decision === "INCLUDE" ? "include" : "exclude";
                                recordProjectDecision(s.currentProjectId, {
                                  paper_id: row.paper_id,
                                  stage: "abstract",
                                  decision: wantKeep ? "include" : "exclude",
                                  reason: row.Reason || "",
                                  per_pico_verdict: row.Pico_Assessment || null,
                                  ai_decision: aiDec,
                                  is_override: aiDec !== (wantKeep ? "include" : "exclude"),
                                }).then(() => setProjectRefresh(n => n + 1))
                                  .catch((e: any) => console.error("Project decision write failed:", e?.message));
                              }
                            }}
                            aria-label="Keep this paper"
                          />
                        </td>
                        <td className="px-3 py-2 sticky left-[60px] z-20 border-r bg-inherit">
                          <DecisionCell value={eff} overridden={isOverridden} aiValue={row.Decision} />
                        </td>
                        <td className="px-3 py-2 sticky left-[180px] z-20 border-r min-w-[260px] max-w-[260px] bg-inherit shadow-[6px_0_8px_-6px_rgba(0,0,0,0.18)]">
                          <a href={row.URL} target="_blank" rel="noreferrer" className="hover:underline break-words">
                            {row.Title}
                          </a>
                        </td>
                        <td className="px-3 py-2 bg-inherit"><Badge variant="outline">{row.Source}</Badge></td>
                        <td className="px-3 py-2 text-center bg-inherit">
                          <PicoCell label="Population" field={pa?.population} criterion={s.pico.population} />
                        </td>
                        <td className="px-3 py-2 text-center bg-inherit">
                          <PicoCell label="Intervention" field={pa?.intervention} criterion={s.pico.intervention} />
                        </td>
                        <td className="px-3 py-2 text-center bg-inherit">
                          <PicoCell label="Comparator" field={pa?.comparator} criterion={s.pico.comparator} />
                        </td>
                        <td className="px-3 py-2 text-center bg-inherit">
                          <PicoCell label="Outcome" field={pa?.outcome} criterion={s.pico.outcome} />
                        </td>
                        <td className="px-3 py-2 text-foreground/90 min-w-[380px] bg-inherit">
                          {row.Reason}
                        </td>
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
          title="Abstract screening with per-PICO appraisal"
          onCancel={() => s.cancelTask("abstract-screen")}
        />
      )}

      {s.query && (
        <Button onClick={runSearch} disabled={running} size="lg" className="w-full">
          <Search className="size-4 mr-2" />{running ? "Screening..." : "Run Abstract Screening"}
        </Button>
      )}
    </div>
  );
}

// ---- per-PICO chip with evidence popover ----------------------------------
//
// NA ("not assessed") renders as a neutral slate chip with a minus glyph and a
// popover that explains *why* — either the PICO element was left blank in the
// frame, or the abstract lacked the information needed to judge it. This reads
// as an intentional state rather than a missing/incomplete value.

function PicoCell({ label, field, criterion }: { label: string; field?: PicoFieldAssessment; criterion?: string }) {
  const vote: PicoVote = field?.vote ?? "NA";
  const isNA = vote === "NA";
  const naReason = (field?.reasoning || "").trim()
    || ((criterion || "").trim()
      ? `The abstract did not give enough information to judge the ${label.toLowerCase()}.`
      : `No ${label.toLowerCase()} was specified in your PICO frame, so there is nothing to assess this paper against. Add one on the Home page to enable this check.`);
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          className={`inline-flex items-center justify-center size-6 rounded text-xs font-semibold border ${picoVoteClass(vote)} hover:ring-1 hover:ring-foreground/30`}
          title={`${label}: ${picoVoteFullLabel(vote)}`}
        >
          {isNA ? <Minus className="size-3.5" /> : picoVoteShort(vote)}
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-96 text-xs space-y-2" align="end">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">{label}</span>
          <span className={`px-2 py-0.5 rounded text-xs font-medium border ${picoVoteClass(vote)}`}>
            {picoVoteFullLabel(vote)}
          </span>
        </div>
        {isNA ? (
          <div className="text-foreground/90">{naReason}</div>
        ) : (
          <>
            {field?.reasoning && <div className="text-foreground/90">{field.reasoning}</div>}
            {field?.evidence && (
              <blockquote className="border-l-2 border-primary/30 pl-2 italic text-muted-foreground break-words">
                "{field.evidence}"
              </blockquote>
            )}
          </>
        )}
      </PopoverContent>
    </Popover>
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

function DecisionCell({ value, overridden, aiValue }: { value: string; overridden?: boolean; aiValue?: string }) {
  const inc = value.toUpperCase().includes("INCLUDE");
  return (
    <div className="space-y-1">
      <span className={`inline-block px-2 py-1 rounded font-medium text-xs ${inc ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
        {inc ? "✅ Include" : "❌ Exclude"}
      </span>
      {overridden && aiValue && (
        <div className="text-[10px] text-muted-foreground leading-tight">
          AI: {aiValue.toUpperCase().includes("INCLUDE") ? "Include" : "Exclude"} · reviewer-edited
        </div>
      )}
    </div>
  );
}

// ---- XLSX export ----------------------------------------------------------
//
// Produces a formatted Excel workbook (not a CSV) so reviewers can open the
// file straight into Excel / Google Sheets / Numbers with colour-coded
// decision cells, wrapped text in long quote columns, and frozen headers.

import ExcelJS from "exceljs";

const PICO_VOTE_FILL: Record<string, string> = {
  PASS:    "C8F2D4",   // green-100
  PARTIAL: "FEEBC8",   // amber-100
  FAIL:    "FED7DA",   // rose-100
  NA:      "F1F5F9",   // slate-100
};

const PICO_VOTE_TEXT: Record<string, string> = {
  PASS:    "065F46",
  PARTIAL: "92400E",
  FAIL:    "9F1239",
  NA:      "475569",
};

async function downloadAbstractXlsx(
  rows: ScreenResult[],
  overrides: Record<string, "INCLUDE" | "EXCLUDE"> = {},
) {
  const wb = new ExcelJS.Workbook();
  wb.creator = "Evidence Engine";
  wb.created = new Date();

  const ws = wb.addWorksheet("Abstract screening", {
    views: [{ state: "frozen", xSplit: 2, ySplit: 1 }],
  });

  ws.columns = [
    { header: "Decision",      key: "decision",   width: 12 },
    { header: "AI Decision",   key: "ai_decision", width: 12 },
    { header: "Reviewer Edit", key: "rev_edit",   width: 14 },
    { header: "Title",         key: "title",      width: 48 },
    { header: "Source",        key: "source",     width: 14 },
    { header: "URL",           key: "url",        width: 38 },
    { header: "Reasoning",     key: "reasoning",  width: 60 },
    { header: "P · vote",      key: "p_vote",     width: 12 },
    { header: "P · quote",     key: "p_quote",    width: 48 },
    { header: "P · reasoning", key: "p_reason",   width: 40 },
    { header: "I · vote",      key: "i_vote",     width: 12 },
    { header: "I · quote",     key: "i_quote",    width: 48 },
    { header: "I · reasoning", key: "i_reason",   width: 40 },
    { header: "C · vote",      key: "c_vote",     width: 12 },
    { header: "C · quote",     key: "c_quote",    width: 48 },
    { header: "C · reasoning", key: "c_reason",   width: 40 },
    { header: "O · vote",      key: "o_vote",     width: 12 },
    { header: "O · quote",     key: "o_quote",    width: 48 },
    { header: "O · reasoning", key: "o_reason",   width: 40 },
  ];

  // Header styling — dark band with white bold text.
  const header = ws.getRow(1);
  header.height = 26;
  header.eachCell((cell) => {
    cell.font = { bold: true, color: { argb: "FFFFFFFF" }, size: 11 };
    cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF1F2937" } };
    cell.alignment = { vertical: "middle", horizontal: "left", wrapText: true };
    cell.border = { bottom: { style: "thin", color: { argb: "FF374151" } } };
  });

  for (const r of rows) {
    const pa = r.Pico_Assessment;
    const overrideVal = overrides[r.paper_id];
    const effective = overrideVal ?? r.Decision;
    const row = ws.addRow({
      decision:    effective,
      ai_decision: r.Decision,
      rev_edit:    overrideVal && overrideVal !== r.Decision ? "yes" : "",
      title:    r.Title,
      source:   r.Source,
      url:      r.URL,
      reasoning: r.Reason,
      p_vote:   pa?.population.vote   ?? "",
      p_quote:  pa?.population.evidence ?? "",
      p_reason: pa?.population.reasoning ?? "",
      i_vote:   pa?.intervention.vote   ?? "",
      i_quote:  pa?.intervention.evidence ?? "",
      i_reason: pa?.intervention.reasoning ?? "",
      c_vote:   pa?.comparator.vote   ?? "",
      c_quote:  pa?.comparator.evidence ?? "",
      c_reason: pa?.comparator.reasoning ?? "",
      o_vote:   pa?.outcome.vote   ?? "",
      o_quote:  pa?.outcome.evidence ?? "",
      o_reason: pa?.outcome.reasoning ?? "",
    });

    // Default to top-aligned wrapped text everywhere — quotes are often long.
    row.eachCell((cell) => {
      cell.alignment = { vertical: "top", wrapText: true };
      cell.font = { size: 10 };
    });

    // Colour-coded Decision cell — uses the EFFECTIVE decision. The AI's
    // original verdict stays available in the adjacent "AI Decision" column
    // for audit purposes.
    const decCell = row.getCell("decision");
    decCell.fill = {
      type: "pattern", pattern: "solid",
      fgColor: { argb: effective === "INCLUDE" ? "FFDCFCE7" : "FFFEE2E2" },
    };
    decCell.font = {
      size: 10, bold: true,
      color: { argb: effective === "INCLUDE" ? "FF065F46" : "FF991B1B" },
    };
    if (overrideVal && overrideVal !== r.Decision) {
      decCell.border = {
        top: { style: "thin", color: { argb: "FFD97706" } },
        bottom: { style: "thin", color: { argb: "FFD97706" } },
        left: { style: "thin", color: { argb: "FFD97706" } },
        right: { style: "thin", color: { argb: "FFD97706" } },
      };
    }
    decCell.alignment = { vertical: "top", horizontal: "center", wrapText: true };

    // Colour the four vote cells by judgment.
    for (const key of ["p_vote", "i_vote", "c_vote", "o_vote"]) {
      const cell = row.getCell(key);
      const v = String(cell.value || "").toUpperCase();
      if (v && PICO_VOTE_FILL[v]) {
        cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF" + PICO_VOTE_FILL[v] } };
        cell.font = { size: 10, bold: true, color: { argb: "FF" + PICO_VOTE_TEXT[v] } };
        cell.alignment = { vertical: "middle", horizontal: "center", wrapText: false };
      }
    }

    // URL as a hyperlink in the URL cell.
    if (r.URL) {
      const urlCell = row.getCell("url");
      urlCell.value = { text: r.URL, hyperlink: r.URL };
      urlCell.font = { size: 10, color: { argb: "FF1D4ED8" }, underline: true };
    }

    // Reasonable minimum row height — let Excel auto-grow when needed.
    row.height = 60;
  }

  // Auto-filter on the header row so reviewers can sort/filter immediately.
  ws.autoFilter = {
    from: { row: 1, column: 1 },
    to: { row: 1, column: ws.columnCount },
  };

  const buf = await wb.xlsx.writeBuffer();
  const blob = new Blob([buf], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  a.href = url;
  a.download = `abstract-screening-${stamp}.xlsx`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
