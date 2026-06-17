import { useStore } from "../lib/store";
import { AIService, formatDuration, FullTextResult } from "../lib/mockServices";
import {
  categoriseFullTextExclusion,
  effectiveAbstractDecision,
  effectiveFullTextDecision,
} from "../lib/exclusionBucketing";
import { Checkbox } from "../components/ui/checkbox";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Popover, PopoverContent, PopoverTrigger } from "../components/ui/popover";
import { FlaskConical, Check, Minus, X as XIcon, Download } from "lucide-react";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";

export function FullTextPage() {
  const s = useStore();
  const task = s.tasks["full-text-screen"];
  const running = task?.status === "running";

  if (!s.results) return <Alert><AlertDescription>Complete Abstract Screening first.</AlertDescription></Alert>;

  // The queue uses EFFECTIVE abstract decisions, so a paper the reviewer
  // rescued at the abstract stage (by checking its Keep box) now joins the
  // full-text queue, and one they dropped is removed.
  const passed = s.results.filter(r => effectiveAbstractDecision(r, s.abstractOverrides) === "INCLUDE");
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
            const bucket = categoriseFullTextExclusion(r, s.inclusion, s.exclusion);
            ftReasons[bucket] = (ftReasons[bucket] || 0) + 1;
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
          {(() => {
            const includedEff = ft.filter(x => effectiveFullTextDecision(x, s.fullTextOverrides) === "Include").length;
            const excludedEff = ft.filter(x => effectiveFullTextDecision(x, s.fullTextOverrides) === "Exclude").length;
            const overrideCount = ft.filter(x => effectiveFullTextDecision(x, s.fullTextOverrides) !== x.Decision).length;
            return (
              <>
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-medium">Full-Text Results</h3>
                  <div className="flex gap-2 items-center">
                    <Badge variant="secondary">{ft.length} papers</Badge>
                    <Badge className="bg-green-600">{includedEff} included</Badge>
                    <Badge variant="destructive">{excludedEff} excluded</Badge>
                    <Badge variant="outline">{formatDuration(s.ftDuration)}</Badge>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => downloadFullTextXlsx(ft, allCriteria, s.fullTextOverrides)}
                      className="h-7 px-2"
                      title="Download results as Excel (XLSX)"
                    >
                      <Download className="size-4" />
                    </Button>
                  </div>
                </div>
                {overrideCount > 0 && (
                  <div className="text-xs text-muted-foreground mb-2">
                    {overrideCount} reviewer override{overrideCount === 1 ? "" : "s"} active — Decision column reflects the reviewer's choice.
                  </div>
                )}
              </>
            );
          })()}
          <div className="rounded-md border max-h-[600px] overflow-auto">
            <table className="w-full text-sm border-collapse">
              <thead className="bg-muted sticky top-0 z-30">
                <tr className="text-left">
                  <th className="px-3 py-2 sticky left-0 bg-muted z-40 border-b border-r min-w-[60px] text-center" title="Reviewer override — check to keep, uncheck to drop">Keep</th>
                  <th className="px-3 py-2 sticky left-[60px] bg-muted z-40 border-b border-r min-w-[120px]">Decision</th>
                  <th className="px-3 py-2 sticky left-[180px] bg-muted z-40 border-b border-r min-w-[300px] max-w-[300px]">Title</th>
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
                {ft.map(row => {
                  const eff = effectiveFullTextDecision(row, s.fullTextOverrides);
                  const isOverridden = eff !== row.Decision;
                  const keep = eff === "Include";
                  return (
                  <tr key={row.paper_id} className="border-b last:border-b-0 align-top group/row">
                    <td className="px-3 py-2 sticky left-0 z-20 border-r bg-card group-hover/row:bg-muted text-center">
                      <Checkbox
                        checked={keep}
                        onCheckedChange={(v) => {
                          const wantKeep = v === true;
                          if ((row.Decision === "Include") === wantKeep) {
                            s.clearFullTextOverride(row.paper_id);
                          } else {
                            s.setFullTextOverride(row.paper_id, wantKeep ? "Include" : "Exclude");
                          }
                        }}
                        aria-label="Keep this paper"
                      />
                    </td>
                    <td className="px-3 py-2 sticky left-[60px] z-20 border-r whitespace-nowrap bg-card group-hover/row:bg-muted">
                      <div className="space-y-1">
                        <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${eff === "Include" ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
                          {eff}
                        </span>
                        {isOverridden && (
                          <div className="text-[10px] text-muted-foreground leading-tight">
                            AI: {row.Decision} · reviewer-edited
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="px-3 py-2 sticky left-[180px] z-20 border-r min-w-[300px] max-w-[300px] bg-card group-hover/row:bg-muted">
                      <a href={row.URL} target="_blank" rel="noreferrer" className="hover:underline break-words">
                        {row.Title}
                      </a>
                    </td>
                    {(["population", "intervention", "comparator", "outcome"] as const).map((k, idx) => {
                      const pe = row.picoEvidence?.[k];
                      const cls = idx === 0 ? "border-l" : idx === 3 ? "border-r" : "";
                      if (!pe || !pe.value) {
                        const naReason = (s.pico[k] || "").trim()
                          ? `The full text did not provide enough information to judge the ${k}.`
                          : `No ${k} was specified in your PICO frame, so there is nothing to assess this paper against. Add one on the Home page to enable this check.`;
                        return (
                          <td key={k} className={`px-3 py-2 ${cls}`}>
                            <Popover>
                              <PopoverTrigger asChild>
                                <button>
                                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-slate-100 text-slate-600">
                                    <Minus className="size-3" />n/a
                                  </span>
                                </button>
                              </PopoverTrigger>
                              <PopoverContent className="w-80 text-xs space-y-2">
                                <div>
                                  <div className="font-medium mb-1 capitalize">{k}</div>
                                  <div className="text-muted-foreground"><span className="font-medium text-foreground/80">Not assessed.</span> {naReason}</div>
                                </div>
                              </PopoverContent>
                            </Popover>
                          </td>
                        );
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
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

// ---- XLSX export ----------------------------------------------------------

import ExcelJS from "exceljs";

const PICO_MATCH_FILL: Record<string, string> = {
  yes:     "FFDCFCE7",
  partial: "FFFEF3C7",
  no:      "FFFEE2E2",
};
const PICO_MATCH_TEXT: Record<string, string> = {
  yes:     "FF065F46",
  partial: "FF92400E",
  no:      "FF9F1239",
};

async function downloadFullTextXlsx(
  rows: FullTextResult[],
  criteria: string[],
  overrides: Record<string, "Include" | "Exclude"> = {},
) {
  const wb = new ExcelJS.Workbook();
  wb.creator = "Evidence Engine";
  wb.created = new Date();

  const ws = wb.addWorksheet("Full-text screening", {
    views: [{ state: "frozen", xSplit: 2, ySplit: 1 }],
  });

  // Static columns + one column per criterion.
  const baseCols = [
    { header: "Decision",      key: "decision",  width: 12 },
    { header: "AI Decision",   key: "ai_decision", width: 12 },
    { header: "Reviewer Edit", key: "rev_edit",  width: 14 },
    { header: "Title",         key: "title",     width: 48 },
    { header: "Source",        key: "source",    width: 14 },
    { header: "URL",           key: "url",       width: 38 },
    { header: "Reason",        key: "reason",    width: 60 },
    { header: "P · match",     key: "p_match",   width: 12 },
    { header: "P · value",     key: "p_value",   width: 28 },
    { header: "P · evidence",  key: "p_evidence", width: 48 },
    { header: "I · match",     key: "i_match",   width: 12 },
    { header: "I · value",     key: "i_value",   width: 28 },
    { header: "I · evidence",  key: "i_evidence", width: 48 },
    { header: "C · match",     key: "c_match",   width: 12 },
    { header: "C · value",     key: "c_value",   width: 28 },
    { header: "C · evidence",  key: "c_evidence", width: 48 },
    { header: "O · match",     key: "o_match",   width: 12 },
    { header: "O · value",     key: "o_value",   width: 28 },
    { header: "O · evidence",  key: "o_evidence", width: 48 },
    { header: "Inclusion met", key: "inc",       width: 14 },
    { header: "Excl. violations", key: "exc",    width: 16 },
  ];
  const critCols = criteria.map((c, i) => ({
    header: c,
    key: `crit_${i}`,
    width: 36,
  }));
  ws.columns = [...baseCols, ...critCols];

  const header = ws.getRow(1);
  header.height = 28;
  header.eachCell((cell) => {
    cell.font = { bold: true, color: { argb: "FFFFFFFF" }, size: 11 };
    cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF1F2937" } };
    cell.alignment = { vertical: "middle", horizontal: "left", wrapText: true };
    cell.border = { bottom: { style: "thin", color: { argb: "FF374151" } } };
  });

  for (const r of rows) {
    const pe = (r.picoEvidence || {}) as any;
    const overrideVal = overrides[r.paper_id];
    const effective = overrideVal ?? r.Decision;
    const rowData: Record<string, any> = {
      decision:    effective,
      ai_decision: r.Decision,
      rev_edit:    overrideVal && overrideVal !== r.Decision ? "yes" : "",
      title:    r.Title,
      source:   r.Source,
      url:      r.URL,
      reason:   r.Reason,
      p_match:  pe.population?.match || "",
      p_value:  pe.population?.value || "",
      p_evidence: pe.population?.evidence || "",
      i_match:  pe.intervention?.match || "",
      i_value:  pe.intervention?.value || "",
      i_evidence: pe.intervention?.evidence || "",
      c_match:  pe.comparator?.match || "",
      c_value:  pe.comparator?.value || "",
      c_evidence: pe.comparator?.evidence || "",
      o_match:  pe.outcome?.match || "",
      o_value:  pe.outcome?.value || "",
      o_evidence: pe.outcome?.evidence || "",
      inc:      r.inclusion_score,
      exc:      r.exclusion_violations,
    };
    criteria.forEach((c, i) => {
      const vote = r.criteriaEval?.[c] ?? "";
      const ev = r.criteriaEvidence?.[c];
      // Use newlines so wrapText puts the vote, the quote and the reasoning
      // on separate visual lines inside a single cell.
      const parts: string[] = [vote];
      if (ev?.evidence) parts.push(`"${ev.evidence}"`);
      if (ev?.reasoning) parts.push(ev.reasoning);
      rowData[`crit_${i}`] = parts.join("\n\n");
    });

    const row = ws.addRow(rowData);
    row.eachCell((cell) => {
      cell.alignment = { vertical: "top", wrapText: true };
      cell.font = { size: 10 };
    });

    // Decision cell colour reflects the EFFECTIVE decision; AI Decision column
    // separately preserves the AI's original verdict for the audit trail.
    const decCell = row.getCell("decision");
    decCell.fill = {
      type: "pattern", pattern: "solid",
      fgColor: { argb: effective === "Include" ? "FFDCFCE7" : "FFFEE2E2" },
    };
    decCell.font = {
      size: 10, bold: true,
      color: { argb: effective === "Include" ? "FF065F46" : "FF991B1B" },
    };
    decCell.alignment = { vertical: "top", horizontal: "center", wrapText: true };
    if (overrideVal && overrideVal !== r.Decision) {
      // Faint amber border on overridden Decision cells so reviewers can spot
      // their own edits at a glance.
      decCell.border = {
        top: { style: "thin", color: { argb: "FFD97706" } },
        bottom: { style: "thin", color: { argb: "FFD97706" } },
        left: { style: "thin", color: { argb: "FFD97706" } },
        right: { style: "thin", color: { argb: "FFD97706" } },
      };
    }

    // Colour the four PICO match cells.
    for (const key of ["p_match", "i_match", "c_match", "o_match"]) {
      const cell = row.getCell(key);
      const v = String(cell.value || "").toLowerCase();
      if (v && PICO_MATCH_FILL[v]) {
        cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: PICO_MATCH_FILL[v] } };
        cell.font = { size: 10, bold: true, color: { argb: PICO_MATCH_TEXT[v] } };
        cell.alignment = { vertical: "middle", horizontal: "center", wrapText: false };
      }
    }

    // Colour each criterion cell by the first line (INCLUDE / EXCLUDE).
    criteria.forEach((_, i) => {
      const cell = row.getCell(`crit_${i}`);
      const firstLine = String(cell.value || "").split("\n", 1)[0].toUpperCase();
      if (firstLine === "INCLUDE") {
        cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFECFCCB" } };
      } else if (firstLine === "EXCLUDE") {
        cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFFEE2E2" } };
      }
    });

    // URL hyperlink.
    if (r.URL) {
      const urlCell = row.getCell("url");
      urlCell.value = { text: r.URL, hyperlink: r.URL };
      urlCell.font = { size: 10, color: { argb: "FF1D4ED8" }, underline: true };
    }

    row.height = 78;
  }

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
  a.download = `fulltext-screening-${stamp}.xlsx`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
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
