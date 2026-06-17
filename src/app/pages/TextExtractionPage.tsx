import { useMemo, useRef, useState } from "react";
import { useStore, TextExtractionResult, TextEvidenceItem } from "../lib/store";
import { AIService } from "../lib/mockServices";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Textarea } from "../components/ui/textarea";
import { Input } from "../components/ui/input";
import { Badge } from "../components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../components/ui/dialog";
import {
  ScanText, Sparkles, Search, AlertTriangle, Download,
  MapPin, Quote as QuoteIcon, FileSpreadsheet, Maximize2,
} from "lucide-react";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";
import ExcelJS from "exceljs";

const PRESETS = [
  "What was the primary outcome and its effect size?",
  "What was the sample size and participant demographics?",
  "List all reported adverse events with frequencies.",
  "Extract dose, frequency, and duration of the intervention.",
  "What were the inclusion and exclusion criteria for participants?",
];

// Section → badge palette. Falls back to slate for unrecognised labels.
const SECTION_STYLE: Record<string, string> = {
  Abstract:     "bg-violet-100 text-violet-800 border-violet-200",
  Background:   "bg-blue-100 text-blue-800 border-blue-200",
  Introduction: "bg-blue-100 text-blue-800 border-blue-200",
  Methods:      "bg-sky-100 text-sky-800 border-sky-200",
  Results:      "bg-emerald-100 text-emerald-800 border-emerald-200",
  Discussion:   "bg-amber-100 text-amber-800 border-amber-200",
  Conclusion:   "bg-amber-100 text-amber-800 border-amber-200",
  Limitations:  "bg-orange-100 text-orange-800 border-orange-200",
  References:   "bg-slate-100 text-slate-700 border-slate-200",
};
function sectionBadgeClass(section?: string): string {
  if (!section) return "bg-slate-100 text-slate-700 border-slate-200";
  return SECTION_STYLE[section] || "bg-slate-100 text-slate-700 border-slate-200";
}

export function TextExtractionPage() {
  const s = useStore();
  const [query, setQuery] = useState(PRESETS[0]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [listQ, setListQ] = useState("");
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
    // 8 concurrent requests saturates SGLang's continuous-batching scheduler;
    // on Ollama (single-threaded) it still helps by hiding HTTP round-trip overhead.
    const BATCH = 8;
    try {
      const out: TextExtractionResult[] = [];
      let done = 0;
      for (let i = 0; i < acquired.length; i += BATCH) {
        if (signal.aborted) break;
        const batch = acquired.slice(i, i + BATCH);
        s.updateTask("text-extract", {
          progress: { done, total: acquired.length, label: batch[0].title.slice(0, 80) },
          detail: `Batch ${Math.floor(i / BATCH) + 1} / ${Math.ceil(acquired.length / BATCH)}`,
        });
        const settled = await Promise.allSettled(
          batch.map(r => AIService.extractFromText(r.text || "", query, signal)),
        );
        settled.forEach((result, j) => {
          const r = batch[j];
          if (result.status === "fulfilled") {
            const ext = result.value;
            out.push({
              paper_id: r.paper_id,
              title: r.title,
              query,
              answer: ext.answer,
              summary: ext.summary,
              evidence: ext.evidence || [],
              spans: ext.spans || [],
              values: ext.values || [],
            });
          } else if (!signal.aborted) {
            console.error(`text-extract ${i + j + 1} failed:`, result.reason?.message);
          }
        });
        done += batch.length;
        s.updateTask("text-extract", { progress: { done, total: acquired.length } });
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

  const totalEvidence = s.textExtractions.reduce(
    (a, r) => a + (r.evidence?.length ?? r.spans.length),
    0,
  );
  const totalValues = s.textExtractions.reduce((a, r) => a + r.values.length, 0);
  const noEvidence = s.textExtractions.filter(
    r => (r.evidence?.length ?? r.spans.length) === 0,
  ).length;

  const filtered = listQ.trim()
    ? s.textExtractions.filter(r => r.title.toLowerCase().includes(listQ.toLowerCase()))
    : s.textExtractions;
  const selected = s.textExtractions.find(r => r.paper_id === selectedId) ?? s.textExtractions[0] ?? null;

  return (
    <div className="space-y-3">
      {/* ── Compact header: question + run + export, with inline counts ─────── */}
      <Card className="p-3 space-y-2">
        <div className="flex items-start justify-between gap-2">
          <h3 className="font-medium flex items-center gap-2 text-sm shrink-0"><ScanText className="size-4 text-primary" />Ask in natural language</h3>
          <div className="flex items-center gap-2 flex-wrap justify-end">
            <Button onClick={run} disabled={running} size="sm">
              <Sparkles className="size-4 mr-2" />{running ? "Extracting..." : `Extract from ${acquired.length} papers`}
            </Button>
            {s.textExtractions.length > 0 && (
              <>
                <Button size="sm" variant="outline" onClick={() => downloadTextExtractionXlsx(s.textExtractions, query)}>
                  <FileSpreadsheet className="size-4 mr-2" />Export Excel
                </Button>
                <Button size="sm" variant="outline" onClick={exportJson}>
                  <Download className="size-4 mr-2" />Export JSON
                </Button>
              </>
            )}
          </div>
        </div>
        <Textarea value={query} onChange={e => setQuery(e.target.value)} rows={2}
          placeholder="e.g. What was the primary outcome and effect size?" />
        <div className="flex flex-wrap gap-1.5">
          {PRESETS.map((p, i) => (
            <Button key={i} size="sm" variant="outline" className="h-7 text-xs" onClick={() => setQuery(p)}>{p}</Button>
          ))}
        </div>
        <div className="text-xs text-muted-foreground flex flex-wrap gap-x-3 gap-y-1">
          {missing.length > 0 && (
            <span className="text-amber-700">
              <AlertTriangle className="size-3 inline mr-1" />{missing.length} skipped (no full text)
            </span>
          )}
          {s.textExtractions.length > 0 && (
            <span>
              {s.textExtractions.length} papers · {totalEvidence} evidence quotes · {totalValues} values
              {noEvidence > 0 && <span className="text-amber-700"> · {noEvidence} with no evidence</span>}
            </span>
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
          {/* ── Two-pane: paper list (left) + selected extraction (right) ─── */}
          <div className="flex gap-4 h-[calc(100vh-17rem)] min-h-[28rem]">
            {/* LEFT: searchable paper list */}
            <Card className="w-80 shrink-0 p-0 overflow-hidden flex flex-col">
              <div className="p-2 border-b">
                <div className="relative">
                  <Search className="size-3.5 absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={listQ}
                    onChange={e => setListQ(e.target.value)}
                    placeholder={`Filter ${s.textExtractions.length} papers…`}
                    className="pl-7 h-8 text-sm"
                  />
                </div>
              </div>
              <div className="overflow-auto flex-1">
                {filtered.map((r) => {
                  const active = r.paper_id === selected?.paper_id;
                  const evCount = r.evidence?.length ?? r.spans.length;
                  return (
                    <button
                      key={r.paper_id}
                      onClick={() => setSelectedId(r.paper_id)}
                      className={`w-full text-left px-3 py-2.5 border-b hover:bg-muted/50 transition-colors ${active ? "bg-primary/10 border-l-2 border-l-primary" : "border-l-2 border-l-transparent"}`}
                    >
                      <div className="flex items-center gap-1.5 mb-1">
                        <Badge variant={evCount > 0 ? "default" : "secondary"} className="text-[10px]">{evCount} quotes</Badge>
                        <Badge variant="outline" className="text-[10px]">{r.values.length} values</Badge>
                      </div>
                      <div className="text-sm line-clamp-2 leading-snug">{r.title}</div>
                    </button>
                  );
                })}
                {filtered.length === 0 && (
                  <div className="p-4 text-sm text-muted-foreground">No papers match “{listQ}”.</div>
                )}
              </div>
            </Card>

            {/* RIGHT: selected paper's extraction */}
            <Card className="flex-1 min-w-0 p-0 overflow-hidden flex flex-col">
              {!selected ? (
                <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
                  Select a paper on the left.
                </div>
              ) : (
                <PaperExtractionDetail
                  key={selected.paper_id}
                  result={selected}
                  fullText={s.fullTexts[selected.paper_id]?.text || ""}
                />
              )}
            </Card>
          </div>
        </>
      )}
    </div>
  );
}

function PaperExtractionDetail({ result, fullText }: { result: TextExtractionResult; fullText: string }) {
  // Evidence may live on the new `evidence[]` field OR fall back to the
  // legacy `spans[]`. Normalise into TextEvidenceItem so the renderer below
  // doesn't have to branch.
  const evidence: TextEvidenceItem[] = result.evidence?.length
    ? result.evidence
    : result.spans.map((sp, i) => ({
        quote: fullText.slice(sp.start, sp.end),
        section: sp.label,
        start: sp.start,
        end: sp.end,
        why: `Match ${i + 1}`,
      }));

  const fullTextRef = useRef<HTMLDivElement | null>(null);
  const maxTextRef = useRef<HTMLDivElement | null>(null);
  const [activeSpan, setActiveSpan] = useState<[number, number] | null>(null);
  const [maxOpen, setMaxOpen] = useState(false);

  // Scroll to a span inside whichever viewer is currently visible (inline, or
  // the maximized dialog).
  function scrollToSpan(ref: React.RefObject<HTMLDivElement | null>, start: number, end: number) {
    setActiveSpan([start, end]);
    // Defer scroll until React renders the highlight class on the new active span.
    requestAnimationFrame(() => {
      const el = ref.current?.querySelector<HTMLElement>(`[data-span-start="${start}"]`);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }
  function locate(start: number, end: number) {
    scrollToSpan(maxOpen ? maxTextRef : fullTextRef, start, end);
  }

  return (
    <>
      <div className="border-b p-4">
        <div className="font-medium leading-snug">{result.title}</div>
        <div className="flex items-center gap-2 mt-2">
          <Badge variant={evidence.length > 0 ? "default" : "secondary"}>{evidence.length} quotes</Badge>
          <Badge variant="outline">{result.values.length} values</Badge>
        </div>
      </div>
      <div className="flex-1 overflow-auto">
        <div className="p-4 space-y-4">
            {/* ANSWER ------------------------------------------------------ */}
            {(result.answer || result.summary) && (
              <div className="bg-primary/5 border border-primary/30 rounded-md p-3">
                <div className="text-xs font-medium uppercase tracking-wide text-primary mb-1">Answer</div>
                <div className="text-sm leading-relaxed text-foreground/90">
                  {result.answer || result.summary}
                </div>
              </div>
            )}

            {evidence.length === 0 ? (
              <div className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded p-3">
                <AlertTriangle className="size-4 inline mr-1" />
                No relevant passages found for this query. The information may not be in this paper.
              </div>
            ) : (
              <>
                {/* EVIDENCE CARDS -------------------------------------------- */}
                <div>
                  <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
                    Evidence quotes ({evidence.length})
                  </div>
                  <div className="space-y-2">
                    {evidence.map((e, i) => (
                      <div key={i} className="border rounded-md p-3 bg-card hover:border-primary/40 transition-colors">
                        <div className="flex items-start justify-between gap-3 mb-2">
                          <div className="flex items-center gap-2 text-xs">
                            <span className="text-muted-foreground tabular-nums">[{i + 1}]</span>
                            {e.section && (
                              <Badge variant="outline" className={`text-[10px] font-medium border ${sectionBadgeClass(e.section)}`}>
                                {e.section}
                              </Badge>
                            )}
                            <span className="text-muted-foreground tabular-nums">char {e.start}-{e.end}</span>
                          </div>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => locate(e.start, e.end)}
                            className="h-6 px-2 text-xs"
                            title="Scroll to this passage in the full text below"
                          >
                            <MapPin className="size-3 mr-1" /> Locate
                          </Button>
                        </div>
                        <blockquote className="text-sm border-l-2 border-primary/40 pl-3 italic text-foreground/90 break-words">
                          <QuoteIcon className="size-3 inline mr-1 text-muted-foreground" />
                          {e.quote}
                        </blockquote>
                        {e.why && (
                          <div className="text-xs text-muted-foreground mt-2">{e.why}</div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* EXTRACTED VALUES ------------------------------------------ */}
                {result.values.length > 0 && (
                  <div>
                    <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
                      Extracted values ({result.values.length})
                    </div>
                    <div className="grid md:grid-cols-2 gap-2">
                      {result.values.map((v, i) => (
                        <div
                          key={i}
                          className={`border rounded p-2 text-sm bg-card transition-colors ${
                            v.start !== undefined ? "hover:border-primary/40 cursor-pointer" : ""
                          }`}
                          onClick={() => {
                            if (v.start !== undefined && v.end !== undefined) {
                              locate(v.start, v.end);
                            }
                          }}
                          title={v.start !== undefined ? "Click to locate in text" : undefined}
                        >
                          <div className="flex items-center gap-2 flex-wrap">
                            <Badge variant="outline" className="text-[10px]">{v.field}</Badge>
                            {v.section && (
                              <Badge variant="outline" className={`text-[10px] font-medium border ${sectionBadgeClass(v.section)}`}>
                                {v.section}
                              </Badge>
                            )}
                            <span className="font-mono text-sm">{v.value}</span>
                          </div>
                          {v.quote && (
                            <div className="text-xs italic text-muted-foreground mt-1">"…{v.quote}…"</div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* FULL-TEXT VIEWER WITH LINE NUMBERS ------------------------ */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-xs uppercase tracking-wide text-muted-foreground">
                      Full text · click any "Locate" button above to jump
                    </div>
                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={() => setMaxOpen(true)}
                      className="size-7 text-muted-foreground hover:text-foreground"
                      title="Maximize full text"
                    >
                      <Maximize2 className="size-4" />
                    </Button>
                  </div>
                  <div
                    ref={fullTextRef}
                    className="rounded-md border bg-muted/20 text-xs font-mono leading-relaxed max-h-96 overflow-auto"
                  >
                    <FullTextViewer text={fullText} evidence={evidence} activeSpan={activeSpan} />
                  </div>
                </div>
              </>
            )}
        </div>
      </div>

      {/* Full-screen full-text viewer. */}
      <Dialog open={maxOpen} onOpenChange={setMaxOpen}>
        <DialogContent className="max-w-[99vw] w-[99vw] h-[97vh] sm:max-w-[99vw] flex flex-col p-3 gap-2">
          <DialogHeader className="pr-8">
            <DialogTitle className="text-base leading-snug">
              Full text · <span className="font-normal text-muted-foreground">{result.title}</span>
            </DialogTitle>
          </DialogHeader>
          {evidence.length > 0 && (
            <div className="flex flex-wrap gap-1.5 shrink-0">
              {evidence.map((e, i) => (
                <Button
                  key={i}
                  size="sm"
                  variant="outline"
                  className="h-6 px-2 text-xs"
                  onClick={() => scrollToSpan(maxTextRef, e.start, e.end)}
                  title={e.quote}
                >
                  <MapPin className="size-3 mr-1" />[{i + 1}]{e.section ? ` ${e.section}` : ""}
                </Button>
              ))}
            </div>
          )}
          <div
            ref={maxTextRef}
            className="flex-1 min-h-0 overflow-auto rounded-md border bg-muted/20 text-sm font-mono leading-relaxed"
          >
            <FullTextViewer text={fullText} evidence={evidence} activeSpan={activeSpan} />
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

/** Renders the full text with line numbers, highlighting every evidence span.
 *  Each span carries `data-span-start={start}` so the parent can scrollIntoView.
 *  Lines that contain at least one highlighted span are nudged a bit so the
 *  reader can see the line number in the gutter. */
function FullTextViewer({
  text, evidence, activeSpan,
}: {
  text: string;
  evidence: TextEvidenceItem[];
  activeSpan: [number, number] | null;
}) {
  if (!text) return <div className="p-3 text-muted-foreground">No full text available.</div>;

  // Sort spans by start offset for the cursor-based renderer.
  const spans = [...evidence].sort((a, b) => a.start - b.start);

  // Split into lines and remember each line's character range so we can
  // overlay highlights line-by-line.
  const lines: { text: string; start: number; end: number; idx: number }[] = [];
  let lineStart = 0;
  text.split(/(\n)/).forEach((part) => {
    if (part === "\n") {
      // Treat the newline as part of the preceding line for offset purposes.
      const last = lines[lines.length - 1];
      if (last) last.end += 1;
      lineStart += 1;
      return;
    }
    if (!part && lines.length > 0) return;
    lines.push({ text: part, start: lineStart, end: lineStart + part.length, idx: lines.length + 1 });
    lineStart += part.length;
  });

  return (
    <table className="w-full border-collapse">
      <tbody>
        {lines.map((ln) => {
          // Restrict spans that overlap this line.
          const lineSpans = spans.filter(s => s.start < ln.end && s.end > ln.start);
          return (
            <tr key={ln.idx} className="align-top">
              <td className="select-none text-right pr-2 pl-2 py-0.5 text-muted-foreground/60 bg-muted/30 border-r tabular-nums w-10">
                {ln.idx}
              </td>
              <td className="py-0.5 px-3 whitespace-pre-wrap break-words">
                {lineSpans.length === 0
                  ? ln.text
                  : renderLineWithHighlights(ln.text, ln.start, lineSpans, activeSpan)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function renderLineWithHighlights(
  lineText: string,
  lineStart: number,
  spans: TextEvidenceItem[],
  activeSpan: [number, number] | null,
): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  let cursor = 0;
  // Local offsets within this line, clipped to the line bounds.
  const sorted = spans
    .map(s => ({
      ...s,
      localStart: Math.max(0, s.start - lineStart),
      localEnd:   Math.max(0, s.end   - lineStart),
    }))
    .sort((a, b) => a.localStart - b.localStart);

  sorted.forEach((s, i) => {
    if (s.localStart > cursor) {
      parts.push(<span key={`pre${i}`}>{lineText.slice(cursor, s.localStart)}</span>);
    }
    const isActive = activeSpan && s.start === activeSpan[0] && s.end === activeSpan[1];
    parts.push(
      <mark
        key={`mark${i}`}
        data-span-start={s.start}
        className={
          isActive
            ? "bg-amber-300 ring-2 ring-amber-500 rounded px-0.5 dark:bg-amber-700/60"
            : "bg-yellow-200 dark:bg-yellow-900/50 rounded px-0.5"
        }
        title={s.section ? `${s.section} — chars ${s.start}-${s.end}` : `chars ${s.start}-${s.end}`}
      >
        {lineText.slice(s.localStart, Math.min(lineText.length, s.localEnd))}
      </mark>
    );
    cursor = Math.min(lineText.length, s.localEnd);
  });
  if (cursor < lineText.length) {
    parts.push(<span key="tail">{lineText.slice(cursor)}</span>);
  }
  return parts;
}


// ---- XLSX export ----------------------------------------------------------
//
// Produces a formatted workbook with four kinds of sheets:
//   1. "Summary"      — one row per paper with link to its per-paper sheet
//   2. "All values"   — flat list of every extracted value across papers
//   3. "All evidence" — flat list of every evidence quote across papers
//   4. one sheet per paper with Answer / Evidence / Values stacked
// Section badges in the workbook share the same colour palette as the on-
// screen badges so the export reads like a continuation of the UI.

const SECTION_FILL: Record<string, string> = {
  Abstract:     "FFEDE9FE",    // violet-100
  Background:   "FFDBEAFE",    // blue-100
  Introduction: "FFDBEAFE",
  Methods:      "FFE0F2FE",    // sky-100
  Results:      "FFDCFCE7",    // emerald-100
  Discussion:   "FFFEF3C7",    // amber-100
  Conclusion:   "FFFEF3C7",
  Limitations:  "FFFFEDD5",    // orange-100
  References:   "FFF1F5F9",    // slate-100
};
const SECTION_TEXT: Record<string, string> = {
  Abstract:     "FF5B21B6",
  Background:   "FF1E40AF",
  Introduction: "FF1E40AF",
  Methods:      "FF075985",
  Results:      "FF065F46",
  Discussion:   "FF92400E",
  Conclusion:   "FF92400E",
  Limitations:  "FF9A3412",
  References:   "FF475569",
};

function _styleSectionCell(cell: ExcelJS.Cell, section?: string) {
  if (!section) {
    cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFF1F5F9" } };
    cell.font = { size: 10, color: { argb: "FF475569" } };
    return;
  }
  const fg = SECTION_FILL[section] || "FFF1F5F9";
  const tx = SECTION_TEXT[section] || "FF475569";
  cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: fg } };
  cell.font = { size: 10, bold: true, color: { argb: tx } };
  cell.alignment = { vertical: "middle", horizontal: "center", wrapText: false };
}

function _styleHeader(row: ExcelJS.Row) {
  row.height = 26;
  row.eachCell((cell) => {
    cell.font = { size: 11, bold: true, color: { argb: "FFFFFFFF" } };
    cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF1F2937" } };
    cell.alignment = { vertical: "middle", horizontal: "left", wrapText: true };
    cell.border = { bottom: { style: "thin", color: { argb: "FF374151" } } };
  });
}

function _safeSheetName(name: string, taken: Set<string>): string {
  let base = (name || "Paper").replace(/[:\\/?*[\]]/g, " ").trim().slice(0, 28) || "Paper";
  let candidate = base;
  let i = 2;
  while (taken.has(candidate.toLowerCase())) {
    const suffix = ` (${i})`;
    candidate = base.slice(0, 31 - suffix.length) + suffix;
    i += 1;
  }
  taken.add(candidate.toLowerCase());
  return candidate;
}

async function downloadTextExtractionXlsx(results: TextExtractionResult[], query: string) {
  if (!results || results.length === 0) {
    toast.error("Nothing to export — run extraction first.");
    return;
  }

  const wb = new ExcelJS.Workbook();
  wb.creator = "Evidence Engine";
  wb.created = new Date();

  const taken = new Set<string>(["summary", "all values", "all evidence"]);
  const sheetForPaper = new Map<string, string>();

  // ---- 1. Summary sheet --------------------------------------------------
  const summary = wb.addWorksheet("Summary", {
    views: [{ state: "frozen", ySplit: 1 }],
  });
  summary.columns = [
    { header: "Paper",            key: "title",    width: 60 },
    { header: "Query",            key: "query",    width: 50 },
    { header: "Answer",           key: "answer",   width: 70 },
    { header: "Evidence quotes",  key: "n_ev",     width: 16 },
    { header: "Extracted values", key: "n_val",    width: 16 },
    { header: "Worksheet",        key: "link",     width: 24 },
  ];
  _styleHeader(summary.getRow(1));

  results.forEach((r) => {
    const sheetName = _safeSheetName(r.title, taken);
    sheetForPaper.set(r.paper_id, sheetName);
    const evCount = r.evidence?.length ?? r.spans.length;
    const row = summary.addRow({
      title:  r.title,
      query:  r.query,
      answer: r.answer || r.summary,
      n_ev:   evCount,
      n_val:  r.values.length,
      link:   sheetName,
    });
    row.eachCell((cell) => {
      cell.alignment = { vertical: "top", wrapText: true };
      cell.font = { size: 10 };
    });
    const linkCell = row.getCell("link");
    linkCell.value = { text: `Open → ${sheetName}`, hyperlink: `#'${sheetName}'!A1` } as any;
    linkCell.font = { size: 10, color: { argb: "FF1D4ED8" }, underline: true };

    // Tint Evidence + Values counts.
    const evCell = row.getCell("n_ev");
    evCell.alignment = { vertical: "top", horizontal: "center" };
    evCell.fill = {
      type: "pattern", pattern: "solid",
      fgColor: { argb: evCount > 0 ? "FFDCFCE7" : "FFFEE2E2" },
    };
    evCell.font = {
      size: 10, bold: true,
      color: { argb: evCount > 0 ? "FF065F46" : "FF991B1B" },
    };
    const valCell = row.getCell("n_val");
    valCell.alignment = { vertical: "top", horizontal: "center" };
    valCell.fill = {
      type: "pattern", pattern: "solid",
      fgColor: { argb: r.values.length > 0 ? "FFDCFCE7" : "FFFEE2E2" },
    };
    valCell.font = {
      size: 10, bold: true,
      color: { argb: r.values.length > 0 ? "FF065F46" : "FF991B1B" },
    };

    row.height = 60;
  });
  summary.autoFilter = { from: { row: 1, column: 1 }, to: { row: 1, column: 6 } };

  // Query banner at the very top so reviewers can see what was asked.
  summary.insertRow(1, [`Question asked across all papers:  ${query}`]);
  summary.mergeCells(1, 1, 1, 6);
  const banner = summary.getCell(1, 1);
  banner.font = { size: 11, italic: true, color: { argb: "FF374151" } };
  banner.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFF1F5F9" } };
  banner.alignment = { vertical: "middle", horizontal: "left", wrapText: true };
  summary.getRow(1).height = 28;
  // Re-style header row 2 after insertion.
  _styleHeader(summary.getRow(2));
  summary.views = [{ state: "frozen", ySplit: 2 }];
  summary.autoFilter = { from: { row: 2, column: 1 }, to: { row: 2, column: 6 } };

  // ---- 2. "All values" — flat list ---------------------------------------
  const allValues = wb.addWorksheet("All values", {
    views: [{ state: "frozen", ySplit: 1 }],
  });
  allValues.columns = [
    { header: "Paper",   key: "paper",   width: 50 },
    { header: "Field",   key: "field",   width: 22 },
    { header: "Value",   key: "value",   width: 28 },
    { header: "Section", key: "section", width: 14 },
    { header: "Quote",   key: "quote",   width: 60 },
  ];
  _styleHeader(allValues.getRow(1));
  for (const r of results) {
    for (const v of r.values) {
      const row = allValues.addRow({
        paper:   r.title,
        field:   v.field,
        value:   v.value,
        section: v.section || "",
        quote:   v.quote || "",
      });
      row.eachCell((c) => {
        c.alignment = { vertical: "top", wrapText: true };
        c.font = { size: 10 };
      });
      row.getCell("value").font = { size: 10, name: "Menlo, Consolas, monospace" };
      _styleSectionCell(row.getCell("section"), v.section);
    }
  }
  allValues.autoFilter = { from: { row: 1, column: 1 }, to: { row: 1, column: 5 } };

  // ---- 3. "All evidence" — flat list -------------------------------------
  const allEv = wb.addWorksheet("All evidence", {
    views: [{ state: "frozen", ySplit: 1 }],
  });
  allEv.columns = [
    { header: "Paper",         key: "paper",   width: 50 },
    { header: "Section",       key: "section", width: 14 },
    { header: "Char range",    key: "range",   width: 14 },
    { header: "Quote",         key: "quote",   width: 80 },
    { header: "Why it matters", key: "why",    width: 50 },
  ];
  _styleHeader(allEv.getRow(1));
  for (const r of results) {
    const ev = r.evidence?.length ? r.evidence : r.spans.map((sp, i) => ({
      quote: "(legacy span)",
      section: sp.label,
      start: sp.start,
      end: sp.end,
      why: `Match ${i + 1}`,
    } as TextEvidenceItem));
    for (const e of ev) {
      const row = allEv.addRow({
        paper:   r.title,
        section: e.section || "",
        range:   `${e.start}-${e.end}`,
        quote:   e.quote,
        why:     e.why || "",
      });
      row.eachCell((c) => {
        c.alignment = { vertical: "top", wrapText: true };
        c.font = { size: 10 };
      });
      _styleSectionCell(row.getCell("section"), e.section);
    }
  }
  allEv.autoFilter = { from: { row: 1, column: 1 }, to: { row: 1, column: 5 } };

  // ---- 4. Per-paper sheets ----------------------------------------------
  for (const r of results) {
    const sheetName = sheetForPaper.get(r.paper_id)!;
    const ws = wb.addWorksheet(sheetName, { views: [{ state: "frozen", ySplit: 1 }] });

    // Banner with paper title
    const titleRow = ws.addRow([r.title]);
    titleRow.height = 30;
    ws.mergeCells(titleRow.number, 1, titleRow.number, 5);
    const titleCell = titleRow.getCell(1);
    titleCell.font = { size: 13, bold: true, color: { argb: "FFFFFFFF" } };
    titleCell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF1F2937" } };
    titleCell.alignment = { vertical: "middle", horizontal: "left", wrapText: true };

    // Query
    const qRow = ws.addRow([`Question: ${r.query}`]);
    ws.mergeCells(qRow.number, 1, qRow.number, 5);
    const qCell = qRow.getCell(1);
    qCell.font = { size: 10, italic: true, color: { argb: "FF374151" } };
    qCell.alignment = { vertical: "middle", horizontal: "left", wrapText: true };

    // Back-to-summary
    const backRow = ws.addRow(["← Back to Summary"]);
    ws.mergeCells(backRow.number, 1, backRow.number, 5);
    const back = backRow.getCell(1);
    back.value = { text: "← Back to Summary", hyperlink: "#Summary!A1" } as any;
    back.font = { size: 10, color: { argb: "FF1D4ED8" }, underline: true };

    ws.addRow([]);

    // Answer panel
    const ansHeader = ws.addRow(["Answer"]);
    ws.mergeCells(ansHeader.number, 1, ansHeader.number, 5);
    const ah = ansHeader.getCell(1);
    ah.font = { size: 11, bold: true, color: { argb: "FFFFFFFF" } };
    ah.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF7C3AED" } };
    ah.alignment = { vertical: "middle", horizontal: "left" };
    const ansBody = ws.addRow([r.answer || r.summary || "(no answer)"]);
    ws.mergeCells(ansBody.number, 1, ansBody.number, 5);
    const ab = ansBody.getCell(1);
    ab.alignment = { vertical: "top", wrapText: true };
    ab.font = { size: 10 };
    ansBody.height = 60;

    ws.addRow([]);

    // Evidence section
    const ev = r.evidence?.length ? r.evidence : r.spans.map((sp, i) => ({
      quote: "(legacy span)",
      section: sp.label,
      start: sp.start,
      end: sp.end,
      why: `Match ${i + 1}`,
    } as TextEvidenceItem));

    if (ev.length > 0) {
      const evHeaderRow = ws.addRow(["#", "Section", "Char range", "Quote", "Why it matters"]);
      _styleHeader(evHeaderRow);
      ws.getColumn(1).width = 4;
      ws.getColumn(2).width = 14;
      ws.getColumn(3).width = 14;
      ws.getColumn(4).width = 70;
      ws.getColumn(5).width = 46;
      ev.forEach((e, i) => {
        const r = ws.addRow([i + 1, e.section || "", `${e.start}-${e.end}`, e.quote, e.why || ""]);
        r.eachCell((c) => {
          c.alignment = { vertical: "top", wrapText: true };
          c.font = { size: 10 };
        });
        r.getCell(1).alignment = { vertical: "top", horizontal: "center" };
        _styleSectionCell(r.getCell(2), e.section);
        r.height = 56;
      });
    } else {
      const noEv = ws.addRow(["No evidence quotes found."]);
      ws.mergeCells(noEv.number, 1, noEv.number, 5);
      noEv.getCell(1).font = { size: 10, italic: true, color: { argb: "FF6B7280" } };
    }

    ws.addRow([]);
    ws.addRow([]);

    // Values section
    if (r.values.length > 0) {
      const valHeaderRow = ws.addRow(["#", "Field", "Value", "Section", "Quote"]);
      _styleHeader(valHeaderRow);
      r.values.forEach((v, i) => {
        const row = ws.addRow([i + 1, v.field, v.value, v.section || "", v.quote || ""]);
        row.eachCell((c) => {
          c.alignment = { vertical: "top", wrapText: true };
          c.font = { size: 10 };
        });
        row.getCell(1).alignment = { vertical: "top", horizontal: "center" };
        row.getCell(3).font = { size: 10, name: "Menlo, Consolas, monospace" };
        _styleSectionCell(row.getCell(4), v.section);
        row.height = 36;
      });
    } else {
      const noVals = ws.addRow(["No extracted values."]);
      ws.mergeCells(noVals.number, 1, noVals.number, 5);
      noVals.getCell(1).font = { size: 10, italic: true, color: { argb: "FF6B7280" } };
    }
  }

  const buf = await wb.xlsx.writeBuffer();
  const blob = new Blob([buf], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  a.href = url;
  a.download = `text-extraction-${stamp}.xlsx`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  const evCount = results.reduce((acc, r) => acc + (r.evidence?.length ?? r.spans.length), 0);
  const valCount = results.reduce((acc, r) => acc + r.values.length, 0);
  toast.success(`Exported ${results.length} papers (${evCount} quotes, ${valCount} values) to XLSX.`);
}
