import { useMemo } from "react";
import { useStore } from "../lib/store";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Badge } from "./ui/badge";
import { ListChecks, Download } from "lucide-react";
import { toast } from "sonner";

// The PRISMA 2020 checklist (27 items with sub-items). Journals require it at
// submission; most inputs already live in the app, so we auto-mark what we can
// and flag the rest for the author to complete.
type Item = { id: string; section: string; num: string; text: string };
const ITEMS: Item[] = [
  { id: "1", section: "Title", num: "1", text: "Identify the report as a systematic review." },
  { id: "2", section: "Abstract", num: "2", text: "See the PRISMA 2020 for Abstracts checklist." },
  { id: "3", section: "Introduction", num: "3", text: "Describe the rationale for the review in the context of existing knowledge." },
  { id: "4", section: "Introduction", num: "4", text: "Provide an explicit statement of the objective(s) or question(s) the review addresses." },
  { id: "5", section: "Methods", num: "5", text: "Specify the inclusion and exclusion criteria and how studies were grouped for the syntheses." },
  { id: "6", section: "Methods", num: "6", text: "Specify all databases, registers, websites, etc. searched, and the date each was last searched." },
  { id: "7", section: "Methods", num: "7", text: "Present the full search strategies for all databases, including any filters and limits." },
  { id: "8", section: "Methods", num: "8", text: "Specify the methods to decide whether a study met the inclusion criteria (reviewers, independence, automation)." },
  { id: "9", section: "Methods", num: "9", text: "Specify the methods used to collect data from reports (reviewers, independence, automation)." },
  { id: "10a", section: "Methods", num: "10a", text: "List and define all outcomes for which data were sought." },
  { id: "10b", section: "Methods", num: "10b", text: "List and define all other variables for which data were sought." },
  { id: "11", section: "Methods", num: "11", text: "Specify the methods used to assess risk of bias, including the tool(s) and independence." },
  { id: "12", section: "Methods", num: "12", text: "Specify for each outcome the effect measure(s) used in the synthesis or presentation." },
  { id: "13a", section: "Methods", num: "13a", text: "Describe the process used to decide which studies were eligible for each synthesis." },
  { id: "13b", section: "Methods", num: "13b", text: "Describe any methods required to prepare the data for presentation or synthesis." },
  { id: "13c", section: "Methods", num: "13c", text: "Describe any methods used to tabulate or visually display results." },
  { id: "13d", section: "Methods", num: "13d", text: "Describe any methods used to synthesize results and the rationale (e.g. meta-analysis model)." },
  { id: "13e", section: "Methods", num: "13e", text: "Describe any methods used to explore heterogeneity (subgroup analysis, meta-regression)." },
  { id: "13f", section: "Methods", num: "13f", text: "Describe any sensitivity analyses conducted." },
  { id: "14", section: "Methods", num: "14", text: "Describe methods to assess risk of bias due to missing results (reporting bias)." },
  { id: "15", section: "Methods", num: "15", text: "Describe methods used to assess certainty in the body of evidence (e.g. GRADE)." },
  { id: "16a", section: "Results", num: "16a", text: "Describe the results of the search and selection (numbers, ideally a flow diagram)." },
  { id: "16b", section: "Results", num: "16b", text: "Cite studies that appeared to meet criteria but were excluded, and give reasons." },
  { id: "17", section: "Results", num: "17", text: "Cite each included study and present its characteristics." },
  { id: "18", section: "Results", num: "18", text: "Present risk-of-bias assessments for each included study." },
  { id: "19", section: "Results", num: "19", text: "For each study, present summary statistics and effect estimates." },
  { id: "20a", section: "Results", num: "20a", text: "For each synthesis, summarize the characteristics and risk of bias of contributing studies." },
  { id: "20b", section: "Results", num: "20b", text: "Present results of syntheses; for meta-analysis, the summary estimate, precision, and heterogeneity." },
  { id: "20c", section: "Results", num: "20c", text: "Present results of investigations of possible causes of heterogeneity." },
  { id: "20d", section: "Results", num: "20d", text: "Present results of sensitivity analyses." },
  { id: "21", section: "Results", num: "21", text: "Present assessments of risk of bias due to missing results." },
  { id: "22", section: "Results", num: "22", text: "Present assessments of certainty for each outcome assessed." },
  { id: "23a", section: "Discussion", num: "23a", text: "Provide a general interpretation of the results in the context of other evidence." },
  { id: "23b", section: "Discussion", num: "23b", text: "Discuss any limitations of the evidence included in the review." },
  { id: "23c", section: "Discussion", num: "23c", text: "Discuss any limitations of the review processes used." },
  { id: "23d", section: "Discussion", num: "23d", text: "Discuss implications for practice, policy, and future research." },
  { id: "24a", section: "Other", num: "24a", text: "Provide registration information, including the register name and number, or state not registered." },
  { id: "24b", section: "Other", num: "24b", text: "Indicate where the review protocol can be accessed, or state it was not prepared." },
  { id: "24c", section: "Other", num: "24c", text: "Describe and explain any amendments to the registration or protocol." },
  { id: "25", section: "Other", num: "25", text: "Describe sources of financial or non-financial support and the role of funders." },
  { id: "26", section: "Other", num: "26", text: "Declare any competing interests of the review authors." },
  { id: "27", section: "Other", num: "27", text: "Report which of data, code, and other materials are publicly available and where." },
];

const STATUSES = ["reported", "partial", "not-reported", "na"] as const;
type Status = typeof STATUSES[number];
const STATUS_LABEL: Record<Status, string> = { reported: "Reported", partial: "Partial", "not-reported": "Not reported", na: "N/A" };
function statusClass(st: Status): string {
  return st === "reported" ? "bg-emerald-100 text-emerald-700 border-emerald-200"
    : st === "partial" ? "bg-amber-100 text-amber-700 border-amber-200"
    : st === "na" ? "bg-slate-100 text-slate-500 border-slate-200"
    : "bg-rose-50 text-rose-600 border-rose-200";
}

function dl(name: string, content: string, mime: string) {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

export function PrismaChecklist() {
  const s = useStore();

  // Auto-suggest a status + location for items whose inputs the app already holds.
  const auto = useMemo(() => {
    const a: Record<string, { status: Status; note: string }> = {};
    const set = (id: string, status: Status, note: string) => { a[id] = { status, note }; };
    if (s.inclusion.length || s.exclusion.length) set("5", "reported", `${s.inclusion.length} inclusion, ${s.exclusion.length} exclusion criteria defined.`);
    if (s.searchLog.length || s.sources.length) {
      const dbs = s.sources.join(", ") || (s.searchLog.at(-1)?.rows.map(r => r.source).join(", ") ?? "");
      const last = s.searchLog.at(-1)?.ranAt ? new Date(s.searchLog.at(-1)!.ranAt).toLocaleDateString() : "";
      set("6", "reported", `Databases: ${dbs}.${last ? ` Last searched ${last}.` : ""} See search log.`);
    }
    if (s.searchLog.length || Object.keys(s.perDbQueries).length) set("7", "reported", "Full per-database search strings recorded in the search log / appendix.");
    if (s.results?.length) set("8", "reported", `${s.currentProjectMode && s.currentProjectMode !== "single" ? "Dual independent" : "AI-assisted"} title/abstract and full-text screening.`);
    if (s.extractedPapers?.length) set("9", "reported", "Structured data extraction from included reports.");
    if (s.qualityReports?.length) set("11", "reported", `Risk of bias assessed (${s.qualityReports[0]?.instrument || "instrument recorded"}).`);
    if (s.gradeOutcomes.length) { set("15", "reported", "Certainty assessed with GRADE per outcome."); set("22", "reported", `${s.gradeOutcomes.length} outcome(s) rated with GRADE.`); }
    if (s.metaRun) { set("12", "reported", "Effect measure(s) specified for the synthesis."); set("13d", "reported", "Meta-analysis performed (fixed and random effects)."); set("20b", "reported", "Pooled estimate with precision and heterogeneity reported."); }
    if (s.prisma?.identified) set("16a", "reported", `Identified ${s.prisma.identified}; included ${s.prisma.included_final}. See PRISMA flow diagram.`);
    if (s.prisma?.exclusion_breakdown && Object.keys(s.prisma.exclusion_breakdown).length) set("16b", "reported", "Full-text exclusions recorded with reasons.");
    if (s.extractedPapers?.length || s.results?.length) set("17", "partial", "Included-study characteristics available from extraction.");
    if (s.qualityReports?.length) set("18", "reported", "Per-study risk-of-bias assessments recorded.");
    if (s.protocol) {
      if (s.protocol.registered && s.protocol.registrationId) set("24a", "reported", `Registered: ${s.protocol.registrationId}.`);
      else set("24a", "partial", "Protocol drafted; registration pending.");
      set("24b", "reported", "Protocol available (drafted in the app).");
      if (s.protocolDeviations.length) set("24c", "reported", `${s.protocolDeviations.length} deviation(s) logged.`);
    }
    return a;
  }, [s.inclusion, s.exclusion, s.searchLog, s.sources, s.perDbQueries, s.results, s.extractedPapers, s.qualityReports, s.gradeOutcomes, s.metaRun, s.prisma, s.protocol, s.protocolDeviations, s.currentProjectMode]);

  const effective = (id: string): { status: Status; note: string; fromAuto: boolean } => {
    const saved = s.prismaChecklist[id];
    if (saved) return { status: (saved.status as Status) || "not-reported", note: saved.note || "", fromAuto: false };
    if (auto[id]) return { ...auto[id], fromAuto: true };
    return { status: "not-reported", note: "", fromAuto: false };
  };

  const setItem = (id: string, patch: Partial<{ status: Status; note: string }>) => {
    const cur = effective(id);
    s.setPrismaChecklist(prev => ({ ...prev, [id]: { status: patch.status ?? cur.status, note: patch.note ?? cur.note } }));
  };

  const reportedCount = ITEMS.filter(it => effective(it.id).status === "reported").length;

  function exportCsv() {
    const q = (x: any) => `"${String(x ?? "").replace(/"/g, '""')}"`;
    const head = ["Section", "Item", "Checklist item", "Status", "Location / notes"];
    const rows = ITEMS.map(it => { const e = effective(it.id); return [it.section, it.num, it.text, STATUS_LABEL[e.status], e.note]; });
    dl("prisma-2020-checklist.csv", [head, ...rows].map(r => r.map(q).join(",")).join("\r\n"), "text/csv");
  }
  function exportHtml() {
    const esc = (x: any) => String(x ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const rows = ITEMS.map(it => { const e = effective(it.id); return `<tr><td>${esc(it.section)}</td><td>${esc(it.num)}</td><td>${esc(it.text)}</td><td>${STATUS_LABEL[e.status]}</td><td>${esc(e.note)}</td></tr>`; }).join("");
    dl("prisma-2020-checklist.html", `<!doctype html><html><head><meta charset="utf-8"><title>PRISMA 2020 checklist</title>
<style>body{font-family:system-ui,sans-serif;margin:2rem;color:#0f172a}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #cbd5e1;padding:6px 9px;text-align:left;vertical-align:top}th{background:#f1f5f9}</style>
</head><body><h1>PRISMA 2020 checklist</h1><table><thead><tr><th>Section</th><th>Item</th><th>Checklist item</th><th>Status</th><th>Location / notes</th></tr></thead><tbody>${rows}</tbody></table></body></html>`, "text/html");
  }

  let lastSection = "";

  return (
    <Card className="p-4 space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 font-medium"><ListChecks className="size-4 text-primary" />PRISMA 2020 checklist</div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary">{reportedCount} / {ITEMS.length} reported</Badge>
          <Button size="sm" variant="outline" onClick={exportCsv}><Download className="size-3.5 mr-1.5" />CSV</Button>
          <Button size="sm" variant="outline" onClick={exportHtml}><Download className="size-3.5 mr-1.5" />HTML</Button>
        </div>
      </div>
      <p className="text-xs text-muted-foreground">Items are pre-marked from your review data where possible; confirm each and add a manuscript location. Amber "auto" tags show a suggestion you have not confirmed yet.</p>
      <div className="rounded-md border overflow-hidden">
        <table className="w-full text-xs">
          <tbody>
            {ITEMS.map(it => {
              const e = effective(it.id);
              const header = it.section !== lastSection ? (lastSection = it.section) : null;
              return (
                <tr key={it.id} className="border-b last:border-0 align-top">
                  <td className="p-0" colSpan={3}>
                    {header && <div className="bg-muted/50 px-3 py-1 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">{header}</div>}
                    <div className="px-3 py-2 flex gap-3">
                      <span className="w-9 shrink-0 font-mono text-muted-foreground">{it.num}</span>
                      <span className="flex-1 min-w-0 space-y-1.5">
                        <span className="block">{it.text}</span>
                        <span className="flex flex-wrap items-center gap-1.5">
                          {STATUSES.map(st => (
                            <button key={st} onClick={() => setItem(it.id, { status: st })}
                              className={`px-2 py-0.5 rounded border text-[11px] ${e.status === st ? statusClass(st) : "bg-card hover:bg-muted text-muted-foreground border-border"}`}>
                              {STATUS_LABEL[st]}
                            </button>
                          ))}
                          {e.fromAuto && <span className="text-[10px] text-amber-600">auto</span>}
                        </span>
                        <Input value={e.note} onChange={ev => setItem(it.id, { note: ev.target.value })}
                          placeholder="Location in manuscript / notes" className="h-7 text-xs" />
                      </span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
