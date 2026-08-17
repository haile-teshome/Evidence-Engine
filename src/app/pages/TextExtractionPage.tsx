import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "motion/react";
import { useStore, TextExtractionResult, TextEvidenceItem } from "../lib/store";
import { AIService } from "../lib/mockServices";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Textarea } from "../components/ui/textarea";
import { Input } from "../components/ui/input";
import { Badge } from "../components/ui/badge";
import { Checkbox } from "../components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../components/ui/dialog";
import {
  ScanText, Sparkles, Search, AlertTriangle, Download,
  MapPin, Quote as QuoteIcon, FileSpreadsheet, Maximize2, ChevronDown, ChevronRight,
  FileText, ListChecks, X, Plus, Trash2, Table2, Loader2, FormInput, Upload, RotateCcw,
} from "lucide-react";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";
import { ControlPane, InlineStat, PaneDivider } from "../components/ControlPane";
import { getPdfBlob } from "../lib/pdfBlobs";
import ExcelJS from "exceljs";

const PRESETS = [
  "What was the primary outcome and its effect size?",
  "What was the sample size and participant demographics?",
  "List all reported adverse events with frequencies.",
  "Extract dose, frequency, and duration of the intervention.",
  "What were the inclusion and exclusion criteria for participants?",
];

// A structured extraction form: named fields the AI fills for each chosen
// article, producing a study-characteristics table.
type FormField = { id: string; label: string; type: string; options: string[]; description?: string };
const slug = (x: string) => (x || "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "") || "field";
const DEFAULT_FORM_FIELDS: FormField[] = [
  { id: "design", label: "Study design", type: "text", options: [], description: "the study design, e.g. randomized controlled trial, cohort, case-control, or cross-sectional survey" },
  { id: "sample_size", label: "Sample size", type: "number", options: [], description: "the total number of participants analysed (N); check the abstract, methods, or a baseline/flow table" },
  { id: "population", label: "Population", type: "text", options: [], description: "who was studied: age range, condition, and setting" },
  { id: "intervention", label: "Intervention", type: "text", options: [], description: "the intervention or exposure studied; write 'none (observational)' if the study has no intervention" },
  { id: "outcome", label: "Primary outcome", type: "text", options: [], description: "the primary outcome measure or main variable assessed" },
  { id: "result", label: "Main result / effect size", type: "text", options: [], description: "the main quantitative finding WITH its value: a correlation coefficient (r), beta, odds/risk/hazard ratio, or mean difference, including its confidence interval or p-value if reported" },
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

// Compact count pill used in the header summary line.
function Pill({
  icon: Icon, children, tone = "default", title,
}: {
  icon: React.ComponentType<{ className?: string }>;
  children: React.ReactNode;
  tone?: "default" | "green" | "amber";
  title?: string;
}) {
  const cls = tone === "green" ? "bg-emerald-50 text-emerald-700 border-emerald-200"
    : tone === "amber" ? "bg-amber-50 text-amber-700 border-amber-200"
    : "bg-muted text-muted-foreground border-transparent";
  return (
    <span title={title} className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${cls}`}>
      <Icon className="size-3" />{children}
    </span>
  );
}

export function TextExtractionPage() {
  const s = useStore();
  const [query, setQuery] = useState(PRESETS[0]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [listQ, setListQ] = useState("");
  // The "Ask in natural language" panel collapses once results exist, so the
  // two-pane gets the screen. Starts open only when nothing's been run yet.
  const [askOpen, setAskOpen] = useState(s.textExtractions.length === 0);
  const task = s.tasks["text-extract"];
  const running = task?.status === "running";

  const acquired = useMemo(() => Object.values(s.fullTexts).filter(r => r.status === "found" && r.text), [s.fullTexts]);
  const missing = useMemo(() => Object.values(s.fullTexts).filter(r => r.status === "missing"), [s.fullTexts]);

  // Structured extraction form: fields (persisted), which articles to run on,
  // and the resulting matrix.
  const [fields, setFields] = useState<FormField[]>(() => {
    try { const v = JSON.parse(localStorage.getItem("ee:extract-form") || "null"); if (Array.isArray(v) && v.length) return v; } catch { /* ignore */ }
    return DEFAULT_FORM_FIELDS;
  });
  useEffect(() => { try { localStorage.setItem("ee:extract-form", JSON.stringify(fields)); } catch { /* ignore */ } }, [fields]);
  const [mode, setMode] = useState<"ask" | "form">("ask");
  const [formCollapsed, setFormCollapsed] = useState(false);
  const [resultsCollapsed, setResultsCollapsed] = useState(false);
  const [openHints, setOpenHints] = useState<Set<string>>(new Set());
  const [deselected, setDeselected] = useState<Set<string>>(new Set());
  const [formBusy, setFormBusy] = useState(false);
  const [formProgress, setFormProgress] = useState<{ done: number; total: number }>({ done: 0, total: 0 });
  const [formCurrent, setFormCurrent] = useState<string>("");
  const [formRows, setFormRows] = useState<{ paper_id: string; title: string; values: Record<string, any> }[]>([]);
  const formFileRef = useRef<HTMLInputElement>(null);

  if (!s.results) return <Alert><AlertDescription>Complete Abstract Screening first.</AlertDescription></Alert>;
  if (acquired.length === 0) return <Alert><AlertDescription>No full texts acquired yet. Fetch them on Full-Text Acquisition first.</AlertDescription></Alert>;

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
        toast.info(`Canceled: ${out.length} of ${acquired.length} processed`);
      } else {
        s.updateTask("text-extract", { status: "done" });
        toast.success(`Extracted from ${out.length} articles`);
        setAskOpen(false);   // collapse the question panel to free up the screen
      }
    } catch (e: any) {
      s.updateTask("text-extract", { status: "error", detail: e?.message });
    }
  }

  function exportJson() {
    const blob = new Blob([JSON.stringify({ query, results: s.textExtractions }, null, 2)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "text_extractions.json"; a.click(); URL.revokeObjectURL(a.href);
  }

  // ── Structured extraction form ─────────────────────────────────────────────
  const addField = () => setFields(f => [...f, { id: `f_${Date.now().toString(36)}`, label: "New field", type: "text", options: [] }]);
  const patchField = (i: number, p: Partial<FormField>) => setFields(f => f.map((x, idx) => idx === i ? { ...x, ...p } : x));
  const removeField = (i: number) => setFields(f => f.filter((_, idx) => idx !== i));

  const targets = acquired.filter(p => !deselected.has(p.paper_id));
  const allSelected = targets.length === acquired.length && acquired.length > 0;
  const toggleTarget = (id: string) => setDeselected(prev => { const n = new Set(prev); n.has(id) ? n.delete(id) : n.add(id); return n; });
  const toggleAllTargets = () => setDeselected(allSelected ? new Set(acquired.map(p => p.paper_id)) : new Set());

  async function runForm() {
    if (!fields.length) { toast.error("Add at least one field to the form."); return; }
    if (!targets.length) { toast.error("Select at least one article to extract from."); return; }
    setFormBusy(true);
    setFormProgress({ done: 0, total: targets.length });
    setFormCurrent(targets[0]?.title || "");
    const rows: { paper_id: string; title: string; values: Record<string, any> }[] = [];
    let done = 0;
    // Local (Ollama) serves one request at a time, so run in series — it also
    // makes the live label exact. Cloud / GPU-served models batch well, so fan
    // out to saturate them.
    const CONCURRENCY = /^(claude|gpt|gemini)/i.test(s.model) ? 8 : 1;
    const spec = fields.map(f => ({ id: f.id, label: f.label, type: f.type, options: f.options, description: f.description }));
    try {
      for (let i = 0; i < targets.length; i += CONCURRENCY) {
        const batch = targets.slice(i, i + CONCURRENCY);
        await Promise.all(batch.map(async p => {
          setFormCurrent(p.title || "Untitled");   // in series this is exact; concurrent shows the batch
          try {
            // Feed extracted tables too — numeric results and effect sizes live
            // in tables, not the prose. Big lever for sample-size/effect fields.
            const tables = ((p as any).tables || [])
              .map((t: any) => [t.caption, ...(t.rows || []).map((r: any[]) => r.join(" | "))].join("\n"))
              .join("\n\n");
            const values = await AIService.extractFields(p.text || "", spec, p.title || "", undefined, tables);
            rows.push({ paper_id: p.paper_id, title: p.title || "Untitled", values });
            setFormRows([...rows]);
          } catch { /* skip failed article */ }
          finally { done += 1; setFormProgress({ done, total: targets.length }); }
        }));
      }
      toast.success(`Extracted the form from ${rows.length} article${rows.length === 1 ? "" : "s"}`);
      if (rows.length) setFormCollapsed(true);   // keep the builder collapsed after a run
    } catch (e: any) {
      toast.error(e?.message || "Form extraction failed");
    } finally { setFormBusy(false); }
  }

  async function exportFormXlsx() {
    if (!formRows.length) { toast.error("Run the form first."); return; }
    const thin = { style: "thin" as const, color: { argb: "FFE2E8F0" } };
    const border = { top: thin, left: thin, bottom: thin, right: thin };
    const wb = new ExcelJS.Workbook();
    wb.creator = "Evidence Engine";

    // Unique, Excel-safe sheet names (<=31 chars, no []:*?/\).
    const used = new Set<string>();
    const names = formRows.map((r, i) => {
      const base = (r.title || `Article ${i + 1}`).replace(/[[\]*?/\\:]/g, " ").replace(/\s+/g, " ").trim().slice(0, 28) || `Article ${i + 1}`;
      let name = base, n = 2;
      while (used.has(name.toLowerCase())) name = `${base.slice(0, 25)} ${n++}`;
      used.add(name.toLowerCase());
      return name;
    });

    // ── Summary sheet: articles × fields ──
    const sum = wb.addWorksheet("Summary", { views: [{ state: "frozen", xSplit: 1, ySplit: 1 }] });
    sum.columns = [{ key: "article", width: 46 }, ...fields.map(f => ({ key: f.id, width: 28 }))];
    const head = sum.insertRow(1, ["Article", ...fields.map(f => f.label)]);
    head.height = 22;
    head.eachCell(c => {
      c.font = { bold: true, color: { argb: "FFFFFFFF" }, size: 11 };
      c.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF1F2937" } };
      c.alignment = { vertical: "middle", wrapText: true };
      c.border = border;
    });
    formRows.forEach((r, i) => {
      const row = sum.addRow([r.title || "Untitled", ...fields.map(f => String(r.values[f.id] ?? "").trim())]);
      row.eachCell(c => { c.alignment = { vertical: "top", wrapText: true }; c.font = { size: 10 }; c.border = border; });
      const a = row.getCell(1);
      a.value = { text: r.title || "Untitled", hyperlink: `#'${names[i]}'!A1` };
      a.font = { size: 10, color: { argb: "FF1D4ED8" }, underline: true };
      if (i % 2) row.eachCell(c => { c.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFF8FAFC" } }; });
    });
    sum.autoFilter = { from: { row: 1, column: 1 }, to: { row: 1, column: fields.length + 1 } };

    // ── One sheet per article: Field | Value ──
    formRows.forEach((r, i) => {
      const ws = wb.addWorksheet(names[i], { views: [{ state: "frozen", ySplit: 3 }] });
      ws.columns = [{ width: 28 }, { width: 90 }];
      ws.mergeCells("A1:B1");
      const t = ws.getCell("A1");
      t.value = r.title || "Untitled";
      t.font = { size: 13, bold: true, color: { argb: "FFFFFFFF" } };
      t.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF1F2937" } };
      t.alignment = { wrapText: true, vertical: "middle" };
      ws.getRow(1).height = 30;
      ws.mergeCells("A2:B2");
      const back = ws.getCell("A2");
      back.value = { text: "← Back to Summary", hyperlink: "#Summary!A1" };
      back.font = { size: 10, color: { argb: "FF1D4ED8" }, underline: true };
      const hr = ws.getRow(3);
      hr.getCell(1).value = "Field"; hr.getCell(2).value = "Value";
      hr.eachCell(c => {
        c.font = { bold: true, color: { argb: "FFFFFFFF" }, size: 11 };
        c.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF7C3AED" } };
        c.border = border;
      });
      fields.forEach((f, j) => {
        const row = ws.addRow([f.label, String(r.values[f.id] ?? "").trim() || "—"]);
        row.getCell(1).font = { bold: true, size: 10 };
        row.getCell(1).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFF1F5F9" } };
        row.getCell(2).font = { size: 10 };
        row.eachCell(c => { c.alignment = { vertical: "top", wrapText: true }; c.border = border; });
        if (j % 2) row.getCell(2).fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFF8FAFC" } };
      });
    });

    const buf = await wb.xlsx.writeBuffer();
    const blob = new Blob([buf], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "extraction-form.xlsx"; a.click(); URL.revokeObjectURL(a.href);
    toast.success("Exported Excel workbook");
  }

  function exportFormCsv() {
    if (!formRows.length) { toast.error("Run the form first."); return; }
    const q = (x: any) => `"${String(x ?? "").replace(/"/g, '""')}"`;
    const head = ["Article", ...fields.map(f => f.label)];
    const rows = formRows.map(r => [r.title, ...fields.map(f => r.values[f.id] ?? "")]);
    const csv = [head, ...rows].map(r => r.map(q).join(",")).join("\r\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "extraction-form.csv"; a.click(); URL.revokeObjectURL(a.href);
  }

  // Import an existing extraction form. Accepts JSON (an array of fields or
  // labels), or a spreadsheet / CSV whose column headers are the fields.
  async function importForm(file: File) {
    try {
      const name = file.name.toLowerCase();
      let next: FormField[] | null = null;
      if (name.endsWith(".json")) {
        const j = JSON.parse(await file.text());
        const arr: any[] = Array.isArray(j) ? j : (Array.isArray(j?.fields) ? j.fields : []);
        next = arr.map((x, i) => typeof x === "string"
          ? { id: `${slug(x)}_${i}`, label: x, type: "text", options: [] }
          : { id: `${slug(x.label || x.name || "field")}_${i}`, label: x.label || x.name || `Field ${i + 1}`, type: x.type || "text", options: Array.isArray(x.options) ? x.options : [] });
      } else {
        let labels: string[] = [];
        if (name.endsWith(".xlsx") || name.endsWith(".xls")) {
          const wb = new ExcelJS.Workbook();
          await wb.xlsx.load(await file.arrayBuffer());
          wb.worksheets[0]?.getRow(1).eachCell(c => { const v = String(c.value ?? "").trim(); if (v) labels.push(v); });
        } else {
          const text = await file.text();
          const firstLine = text.split(/\r?\n/).find(l => l.trim()) || "";
          const sep = firstLine.includes("\t") ? "\t" : ",";
          labels = firstLine.split(sep).map(x => x.replace(/^"|"$/g, "").trim()).filter(Boolean);
        }
        // Drop a leading row-label column ("Article"/"Study"/etc.) if present.
        labels = labels.filter(l => !/^(article|articles|title|study|studies|paper|papers|id|#)$/i.test(l));
        next = labels.map((l, i) => ({ id: `${slug(l)}_${i}`, label: l, type: "text", options: [] }));
      }
      if (!next || !next.length) { toast.error("No fields found in that file."); return; }
      setFields(next);
      setFormRows([]);
      toast.success(`Imported ${next.length} field${next.length === 1 ? "" : "s"} from ${file.name}`);
    } catch {
      toast.error("Could not read that form file. Use JSON, CSV, or an Excel file whose headers are the fields.");
    }
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
      {/* Mode toggle: free-text question vs a structured extraction form. */}
      <div className="inline-flex rounded-lg border bg-muted/40 p-0.5 text-sm">
        <button onClick={() => setMode("ask")} className={`relative inline-flex items-center gap-1.5 rounded-md px-3 h-8 font-medium transition-colors ${mode === "ask" ? "text-foreground" : "text-muted-foreground hover:text-foreground"}`}>
          {mode === "ask" && <motion.span layoutId="textextract-mode" transition={{ type: "spring", stiffness: 440, damping: 36, mass: 0.7 }} className="absolute inset-0 rounded-md bg-card shadow-sm" />}
          <span className="relative z-10 inline-flex items-center gap-1.5"><ScanText className="size-3.5" />Ask</span>
        </button>
        <button onClick={() => setMode("form")} className={`relative inline-flex items-center gap-1.5 rounded-md px-3 h-8 font-medium transition-colors ${mode === "form" ? "text-foreground" : "text-muted-foreground hover:text-foreground"}`}>
          {mode === "form" && <motion.span layoutId="textextract-mode" transition={{ type: "spring", stiffness: 440, damping: 36, mass: 0.7 }} className="absolute inset-0 rounded-md bg-card shadow-sm" />}
          <span className="relative z-10 inline-flex items-center gap-1.5"><FormInput className="size-3.5" />Form</span>
        </button>
      </div>

      {mode === "ask" && (<>
      {/* ── Control pane: counts + run + export ──────────────────────────── */}
      <ControlPane
        stats={s.textExtractions.length > 0 ? <>
          <InlineStat icon={FileText} value={s.textExtractions.length} label="Articles" />
          <PaneDivider />
          <InlineStat icon={QuoteIcon} tone="success" value={totalEvidence} label="Quotes" />
          <InlineStat icon={ListChecks} tone="success" value={totalValues} label="Values" />
          {noEvidence > 0 && <InlineStat icon={AlertTriangle} tone="amber" value={noEvidence} label="No evidence" />}
          {missing.length > 0 && <InlineStat icon={AlertTriangle} tone="amber" value={missing.length} label="Skipped" />}
        </> : (
          <InlineStat icon={FileText} value={acquired.length} label="Ready to extract" />
        )}
        actions={<>
          {running ? (
            <Button size="sm" variant="destructive" className="h-8 shadow-sm" onClick={() => s.cancelTask("text-extract")} title="Stop extraction">
              <X className="size-3.5 mr-1.5" />Cancel
            </Button>
          ) : (
            <Button size="sm" className="h-8 shadow-sm" onClick={run}>
              <Sparkles className="size-3.5 mr-1.5" />Extract
            </Button>
          )}
          {s.textExtractions.length > 0 && (
            <>
              <Button size="sm" variant="outline" className="h-8" onClick={() => downloadTextExtractionXlsx(s.textExtractions, query)}>
                <FileSpreadsheet className="size-3.5 mr-1.5" />Excel
              </Button>
              <Button size="sm" variant="outline" className="h-8" onClick={exportJson}>
                <Download className="size-3.5 mr-1.5" />JSON
              </Button>
            </>
          )}
        </>}
      />

      {/* Ask-in-natural-language editor. */}
      <Card className="p-3 space-y-2">
        <button
          onClick={() => setAskOpen(o => !o)}
          className="flex items-center gap-2 text-sm font-medium w-full text-left min-w-0 hover:text-primary"
          title={askOpen ? "Collapse question" : "Edit question"}
        >
          <ChevronDown className={`size-4 shrink-0 transition-transform ${askOpen ? "" : "-rotate-90"}`} />
          <ScanText className="size-4 text-primary shrink-0" />
          {askOpen ? "Ask in natural language" : <span className="truncate text-muted-foreground font-normal">{query}</span>}
        </button>
        {askOpen && (
          <>
            <Textarea value={query} onChange={e => setQuery(e.target.value)} rows={2}
              placeholder="e.g. What was the primary outcome and effect size?" />
            <div className="flex flex-wrap gap-1.5">
              {PRESETS.map((p, i) => (
                <Button key={i} size="sm" variant="outline" className="h-7 text-xs" onClick={() => setQuery(p)}>{p}</Button>
              ))}
            </div>
          </>
        )}
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
                    placeholder={`Filter ${s.textExtractions.length} articles…`}
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
                      className={`block w-full text-left px-3 py-2.5 border-b hover:bg-muted/50 transition-colors ${active ? "bg-primary/10 border-l-2 border-l-primary" : "border-l-2 border-l-transparent"}`}
                    >
                      <div className="flex items-center gap-1.5 mb-1">
                        <Badge variant={evCount > 0 ? "default" : "secondary"} className="text-[10px]">{evCount} quotes</Badge>
                        <Badge variant="outline" className="text-[10px]">{r.values.length} values</Badge>
                      </div>
                      <div className="text-sm leading-snug line-clamp-2 break-words">{(r.title || "").replace(/\s+/g, " ").trim() || "Untitled"}</div>
                    </button>
                  );
                })}
                {filtered.length === 0 && (
                  <div className="p-4 text-sm text-muted-foreground">No articles match “{listQ}”.</div>
                )}
              </div>
            </Card>

            {/* RIGHT: selected paper's extraction */}
            <Card className="flex-1 min-w-0 p-0 overflow-hidden flex flex-col">
              {!selected ? (
                <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
                  Select an article on the left.
                </div>
              ) : (
                <PaperExtractionDetail
                  key={selected.paper_id}
                  result={selected}
                  fullText={s.fullTexts[selected.paper_id]?.text || ""}
                  pdfUrl={getPdfBlob(selected.paper_id) || s.fullTexts[selected.paper_id]?.pdf_url || ""}
                />
              )}
            </Card>
          </div>
        </>
      )}
      </>)}

      {mode === "form" && (
        <>
          <input type="file" ref={formFileRef} accept=".json,.csv,.tsv,.txt,.xlsx,.xls" className="hidden"
            onChange={e => { const f = e.target.files?.[0]; if (f) importForm(f); e.currentTarget.value = ""; }} />

          {/* Controls live in their own pane, above the form box. */}
          {!formCollapsed && !formBusy && (
            <ControlPane
              stats={<>
                <InlineStat icon={ListChecks} value={fields.length} label="Fields" />
                <PaneDivider />
                <InlineStat icon={FileText} value={targets.length} label="Selected" hint={`of ${acquired.length}`} tone="success" />
              </>}
              actions={<>
                <Button size="sm" variant="outline" className="h-8" onClick={toggleAllTargets}>
                  {allSelected ? <><X className="size-3.5 mr-1.5" />Deselect all</> : <><ListChecks className="size-3.5 mr-1.5" />Select all</>}
                </Button>
                <Button size="sm" variant="outline" className="h-8" onClick={() => formFileRef.current?.click()} title="Import fields from JSON, CSV, or a spreadsheet whose headers are the fields">
                  <Upload className="size-3.5 mr-1.5" />Import form
                </Button>
                <Button size="sm" variant="outline" className="h-8" onClick={() => { setFields(DEFAULT_FORM_FIELDS); setFormRows([]); }} title="Restore the default fields">
                  <RotateCcw className="size-3.5 mr-1.5" />Reset
                </Button>
                <span className="mx-0.5 h-6 w-px bg-border" aria-hidden="true" />
                <Button size="sm" className="h-8 shadow-sm" onClick={runForm} disabled={!targets.length || !fields.length}>
                  <Table2 className="size-4 mr-1.5" />Extract from {targets.length} article{targets.length === 1 ? "" : "s"}
                </Button>
              </>}
            />
          )}
          {/* Collapsible form builder. Auto-collapses after a run so the results
              take the screen; expand the header to edit fields / selection. */}
          <Card className="p-3 space-y-2">
            <div className="flex items-center justify-between gap-2 flex-wrap">
              <button onClick={() => setFormCollapsed(c => !c)} className="flex items-center gap-2 text-sm font-medium min-w-0 hover:text-primary">
                {formCollapsed ? <ChevronRight className="size-4 shrink-0" /> : <ChevronDown className="size-4 shrink-0" />}
                <FormInput className="size-4 text-primary shrink-0" />Extraction form
              </button>
            </div>

            {formBusy && (
              <div className="space-y-1.5 pt-0.5">
                <div className="h-2 rounded-full bg-muted overflow-hidden">
                  <div className="h-full rounded-full bg-gradient-to-r from-primary/70 to-primary transition-[width] duration-500 ease-out" style={{ width: `${formProgress.total ? (formProgress.done / formProgress.total) * 100 : 0}%` }} />
                </div>
                <div className="flex items-center gap-2 text-xs text-muted-foreground min-w-0">
                  <span className="inline-flex items-center gap-1.5 min-w-0 flex-1">
                    <Loader2 className="size-3 animate-spin shrink-0" />
                    <span className="truncate">{formCurrent ? <>Extracting <span className="text-foreground/70">{formCurrent}</span></> : "Extracting…"}</span>
                  </span>
                  <span className="tabular-nums shrink-0">{formProgress.done} / {formProgress.total}</span>
                </div>
              </div>
            )}

            {!formCollapsed && !formBusy && (
            <div className="space-y-4 pt-1">
            <div className="grid lg:grid-cols-2 gap-4 items-start">
              {/* Fields */}
              <div className="space-y-2">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Fields</div>
                <div className="space-y-2">
                  {fields.map((f, i) => (
                    <div key={f.id} className="rounded-lg border bg-card p-2 space-y-1.5 shadow-sm transition-colors hover:border-primary/30">
                      <div className="flex items-center gap-2">
                        <Input className="h-8 flex-1 font-medium" value={f.label} onChange={e => patchField(i, { label: e.target.value })} placeholder="Field label (e.g. Sample size)" />
                        <Select value={f.type} onValueChange={v => patchField(i, { type: v })}>
                          <SelectTrigger className="h-8 w-24 text-xs"><SelectValue /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="text">Text</SelectItem>
                            <SelectItem value="number">Number</SelectItem>
                            <SelectItem value="category">Category</SelectItem>
                            <SelectItem value="date">Date</SelectItem>
                          </SelectContent>
                        </Select>
                        <button onClick={() => removeField(i)} title="Remove field" className="text-muted-foreground hover:text-destructive shrink-0 p-1 rounded-md hover:bg-destructive/10 transition-colors"><Trash2 className="size-3.5" /></button>
                      </div>
                      {(f.description || openHints.has(f.id)) ? (
                        <div className="relative">
                          <Search className="size-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground/50 pointer-events-none" />
                          <Input autoFocus={openHints.has(f.id) && !f.description} className="h-7 pl-8 pr-7 text-xs bg-muted/40 border-dashed placeholder:text-muted-foreground/60 placeholder:italic" value={f.description || ""}
                            onChange={e => patchField(i, { description: e.target.value })}
                            title="Guide the extractor: synonyms, units, the section it appears in, or how it's phrased"
                            placeholder="Synonyms, units, section, phrasing…" />
                          <button onClick={() => { patchField(i, { description: "" }); setOpenHints(s => { const n = new Set(s); n.delete(f.id); return n; }); }}
                            title="Remove hint" className="absolute right-1.5 top-1/2 -translate-y-1/2 text-muted-foreground/50 hover:text-destructive"><X className="size-3" /></button>
                        </div>
                      ) : (
                        <button onClick={() => setOpenHints(s => new Set(s).add(f.id))}
                          className="inline-flex items-center gap-1 text-[11px] text-muted-foreground/60 hover:text-primary transition-colors">
                          <Plus className="size-3" />Add extraction hint
                        </button>
                      )}
                      {f.type === "category" && (
                        <div className="relative">
                          <ListChecks className="size-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground/50 pointer-events-none" />
                          <Input className="h-7 pl-8 text-xs" value={(f.options || []).join(", ")}
                            onChange={e => patchField(i, { options: e.target.value.split(",").map(x => x.trim()).filter(Boolean) })}
                            placeholder="Allowed values: option, option, option" />
                        </div>
                      )}
                    </div>
                  ))}
                  <Button size="sm" variant="outline" className="h-7" onClick={addField}><Plus className="size-3.5 mr-1.5" />Add field</Button>
                </div>
              </div>

              {/* Articles */}
              <div className="space-y-2 flex flex-col">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Articles ({targets.length} of {acquired.length})</div>
                <div className="rounded-md border flex-1 min-h-[15.5rem] overflow-auto divide-y">
                  {acquired.map(p => (
                    <label key={p.paper_id} className="flex items-start gap-2.5 px-2.5 py-1.5 text-xs cursor-pointer hover:bg-muted/40">
                      <Checkbox checked={!deselected.has(p.paper_id)} onCheckedChange={() => toggleTarget(p.paper_id)} className="shrink-0 mt-0.5" />
                      <span className="flex-1 min-w-0 leading-snug break-words" title={(p.title || "").replace(/\s+/g, " ").trim() || "Untitled"}>{(p.title || "").replace(/\s+/g, " ").trim() || "Untitled"}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
            </div>
            )}
          </Card>

          {/* Results matrix */}
          {formRows.length > 0 && (
            <Card className="p-0 overflow-hidden">
              <div className="flex items-center justify-between gap-2 px-4 py-2.5 border-b bg-gradient-to-b from-muted/40 to-transparent">
                <button onClick={() => setResultsCollapsed(c => !c)} className="flex items-center gap-2 text-sm font-medium min-w-0 hover:text-primary" title={resultsCollapsed ? "Expand results" : "Collapse results"}>
                  {resultsCollapsed ? <ChevronRight className="size-4 shrink-0" /> : <ChevronDown className="size-4 shrink-0" />}
                  <span>Results <span className="text-xs text-muted-foreground font-normal">· {formRows.length} article{formRows.length === 1 ? "" : "s"} × {fields.length} field{fields.length === 1 ? "" : "s"}</span></span>
                </button>
                <div className="flex items-center gap-2">
                  <Button size="sm" className="h-8 shadow-sm" onClick={exportFormXlsx}><FileSpreadsheet className="size-3.5 mr-1.5" />Excel</Button>
                  <Button size="sm" variant="outline" className="h-8" onClick={exportFormCsv}><Download className="size-3.5 mr-1.5" />CSV</Button>
                </div>
              </div>
              {!resultsCollapsed && (
              <div className="overflow-auto max-h-[30rem]">
                <table className="w-full text-xs border-collapse">
                  <thead className="bg-muted sticky top-0 z-10 [&_th]:text-[11px] [&_th]:font-semibold [&_th]:uppercase [&_th]:tracking-wider [&_th]:text-muted-foreground">
                    <tr>
                      <th className="px-3 py-2 text-left sticky left-0 bg-muted border-b border-r min-w-[220px] max-w-[220px] z-20">Article</th>
                      {fields.map(f => <th key={f.id} className="px-3 py-2 text-left border-b whitespace-nowrap">{f.label}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {formRows.map(r => (
                      <tr key={r.paper_id} className="border-b border-border/60 last:border-0 align-top hover:bg-muted transition-colors bg-card">
                        <td className="px-3 py-2 sticky left-0 bg-inherit border-r min-w-[220px] max-w-[220px]"><div className="line-clamp-2 font-medium">{r.title}</div></td>
                        {fields.map(f => <td key={f.id} className="px-3 py-2 whitespace-pre-wrap break-words min-w-[130px] text-foreground/90">{String(r.values[f.id] ?? "").trim() || <span className="text-muted-foreground/40">—</span>}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              )}
            </Card>
          )}
        </>
      )}
    </div>
  );
}

function PaperExtractionDetail({ result, fullText, pdfUrl = "" }: { result: TextExtractionResult; fullText: string; pdfUrl?: string }) {
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
  // Default to the PDF when a paper is opened; "Locate" flips to the text view.
  const [viewMode, setViewMode] = useState<"pdf" | "text">(pdfUrl ? "pdf" : "text");

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
    // Char-offset jumping only works in the extracted-text view, so switch to it.
    if (viewMode !== "text") setViewMode("text");
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
                No relevant passages found for this query. The information may not be in this article.
              </div>
            ) : (
              <>
                <div className={result.values.length > 0 ? "grid lg:grid-cols-2 gap-4 items-start" : ""}>
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
                    <div className="grid grid-cols-1 gap-2">
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
                </div>

                {/* SOURCE VIEWER: PDF (default) or extracted text ----------- */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="inline-flex rounded-md border bg-muted/40 p-0.5 text-xs">
                      <button
                        onClick={() => pdfUrl && setViewMode("pdf")}
                        disabled={!pdfUrl}
                        className={`px-2.5 py-1 rounded transition-colors ${viewMode === "pdf" ? "bg-background shadow-sm font-medium" : "text-muted-foreground"} ${!pdfUrl ? "opacity-40 cursor-not-allowed" : ""}`}
                        title={pdfUrl ? "View the source PDF" : "No PDF available for this article"}
                      >
                        PDF
                      </button>
                      <button
                        onClick={() => setViewMode("text")}
                        className={`px-2.5 py-1 rounded transition-colors ${viewMode === "text" ? "bg-background shadow-sm font-medium" : "text-muted-foreground"}`}
                        title="View the extracted text with jump-to-passage"
                      >
                        Extracted text
                      </button>
                    </div>
                    {viewMode === "text" && (
                      <Button
                        size="icon"
                        variant="ghost"
                        onClick={() => setMaxOpen(true)}
                        className="size-7 text-muted-foreground hover:text-foreground"
                        title="Maximize full text"
                      >
                        <Maximize2 className="size-4" />
                      </Button>
                    )}
                  </div>
                  {viewMode === "pdf" && pdfUrl ? (
                    <iframe src={pdfUrl} title="Source PDF" className="w-full h-96 rounded-md border bg-muted/20" />
                  ) : (
                    <div
                      ref={fullTextRef}
                      className="rounded-md border bg-muted/20 text-xs font-mono leading-relaxed max-h-96 overflow-auto"
                    >
                      <FullTextViewer text={fullText} evidence={evidence} activeSpan={activeSpan} />
                    </div>
                  )}
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
        title={s.section ? `${s.section}, chars ${s.start}-${s.end}` : `chars ${s.start}-${s.end}`}
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
//   1. "Summary"      : one row per paper with link to its per-paper sheet
//   2. "All values"   : flat list of every extracted value across papers
//   3. "All evidence" : flat list of every evidence quote across papers
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
    toast.error("Nothing to export. Run extraction first.");
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

  // ---- 2. "All values": flat list ---------------------------------------
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

  // ---- 3. "All evidence": flat list -------------------------------------
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
  toast.success(`Exported ${results.length} articles (${evCount} quotes, ${valCount} values) to XLSX.`);
}
