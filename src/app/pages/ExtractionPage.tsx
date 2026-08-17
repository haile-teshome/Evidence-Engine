import { useState } from "react";
import { useStore } from "../lib/store";
import { AIService } from "../lib/mockServices";
import { effectiveAbstractDecision } from "../lib/exclusionBucketing";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Input } from "../components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../components/ui/dialog";
import { Search, Download, Table2, ExternalLink, Maximize2, FileText, AlertTriangle, X, Rows3, Columns3, Copy, Check } from "lucide-react";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";
import { ControlPane, InlineStat, PaneDivider } from "../components/ControlPane";
import { ExtractionWorkspace } from "../components/ExtractionWorkspace";
import ExcelJS from "exceljs";

type Format = "CSV Export" | "JSON Export" | "Excel Export";

export function ExtractionPage() {
  const s = useStore();
  const [format, setFormat] = useState<Format>("Excel Export");
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [q, setQ] = useState("");
  const task = s.tasks["table-extract"];
  const running = task?.status === "running";

  // In a project, dual data extraction is available regardless of the
  // single-user table-extraction state below.
  const projectBlock = s.currentProjectId
    ? <ExtractionWorkspace projectId={s.currentProjectId} role={s.currentProjectRole} />
    : null;

  if (!s.results) return <div className="space-y-3">{projectBlock}{!projectBlock && <Alert><AlertDescription>Complete Abstract Screening first to unlock Table Extraction.</AlertDescription></Alert>}</div>;

  // Honour reviewer overrides from abstract screening.
  const passed = s.results.filter(r => effectiveAbstractDecision(r, s.abstractOverrides) === "INCLUDE");
  if (passed.length === 0) return <div className="space-y-3">{projectBlock}{!projectBlock && <Alert><AlertDescription>No included articles available for extraction.</AlertDescription></Alert>}</div>;

  async function extract() {
    const { abort } = s.startTask("table-extract", [{ id: "tbl", label: "Extracting tables", status: "running" }]);
    s.updateTask("table-extract", { progress: { done: 0, total: passed.length } });
    const signal = abort.signal;
    try {
      const out: { Paper_Title: string; Paper_URL: string; Source: string; Extracted_Tables: any[] }[] = [];
      for (let i = 0; i < passed.length; i++) {
        if (signal.aborted) break;
        const p = passed[i];
        s.updateTask("table-extract", {
          progress: { done: i, total: passed.length, label: p.Title.slice(0, 80) },
          detail: p.Title.slice(0, 80),
        });
        try {
          const tables = await AIService.extractTables({
            Title: p.Title,
            URL: p.URL,
            Source: p.Source,
            paper_id: p.paper_id,
            Abstract: p.Abstract,
            // Use the cached full text if Full-Text Acquisition has fetched it.
            full_text: s.fullTexts[p.paper_id]?.text || "",
          }, signal);
          out.push({ Paper_Title: p.Title, Paper_URL: p.URL, Source: p.Source, Extracted_Tables: tables });
        } catch (e: any) {
          if (signal.aborted) break;
          console.error(`table-extract ${i + 1} failed:`, e?.message);
        }
        s.updateTask("table-extract", { progress: { done: i + 1, total: passed.length } });
      }
      s.setExtractedPapers(out);
      if (signal.aborted) {
        s.updateTask("table-extract", { status: "canceled" });
        toast.info(`Canceled: ${out.length} of ${passed.length} processed`);
      } else {
        s.updateTask("table-extract", { status: "done" });
        toast.success(`Tables extracted from ${out.length} articles`);
      }
    } catch (e: any) {
      s.updateTask("table-extract", { status: "error", detail: e?.message });
    }
  }

  function exportAll() {
    if (!s.extractedPapers) return;
    if (format === "JSON Export") {
      const json = JSON.stringify(s.extractedPapers, null, 2);
      download("all_extracted_tables.json", json, "application/json");
    } else if (format === "CSV Export") {
      const rows = s.extractedPapers.map(p => `"${p.Paper_Title}","${p.Source}",${p.Extracted_Tables.length}`).join("\n");
      download("all_extracted_tables_summary.csv", `Paper,Source,Tables\n${rows}`, "text/csv");
    } else {
      // Excel Export: formatted XLSX with one worksheet per paper, all
      // tables stacked inside that worksheet.
      void downloadExtractedTablesXlsx(s.extractedPapers);
    }
  }

  const ep = s.extractedPapers;
  const totalPapers = ep?.length || 0;
  const withTables = ep?.filter(p => p.Extracted_Tables.length > 0).length || 0;
  const withoutTables = totalPapers - withTables;

  // Keep original indices so the left list can filter without losing the
  // mapping back into s.extractedPapers.
  const filtered = ep
    ? ep.map((p, idx) => ({ p, idx })).filter(
        ({ p }) => !q.trim() || p.Paper_Title.toLowerCase().includes(q.toLowerCase()),
      )
    : [];
  const selected = ep ? (ep[selectedIdx] ?? ep[0]) : null;

  return (
    <div className="space-y-3">
      {projectBlock}
      {/* ── Control pane: counts + format + run + export ─────────────────── */}
      <ControlPane
        stats={<>
          <InlineStat icon={FileText} value={passed.length} label="Included" />
          {ep && <><PaneDivider /><InlineStat icon={Table2} tone="success" value={withTables} label="With tables" /></>}
          {ep && withoutTables > 0 && <InlineStat icon={AlertTriangle} tone="amber" value={withoutTables} label="No tables" />}
        </>}
        actions={<>
          <div className="flex items-center gap-1.5">
            <label className="text-xs text-muted-foreground">Format</label>
            <Select value={format} onValueChange={v => setFormat(v as Format)}>
              <SelectTrigger className="h-8 w-28"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="CSV Export">CSV</SelectItem>
                <SelectItem value="JSON Export">JSON</SelectItem>
                <SelectItem value="Excel Export">Excel</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {running ? (
            <Button size="sm" variant="destructive" className="h-8 shadow-sm" onClick={() => s.cancelTask("table-extract")} title="Stop extraction">
              <X className="size-3.5 mr-1.5" />Cancel
            </Button>
          ) : (
            <Button size="sm" className="h-8 shadow-sm" onClick={extract}>
              <Table2 className="size-3.5 mr-1.5" />Start extraction
            </Button>
          )}
          {ep && (
            <Button size="sm" variant="outline" className="h-8" onClick={exportAll}>
              <Download className="size-3.5 mr-1.5" />Export ({format.split(" ")[0]})
            </Button>
          )}
        </>}
      />

      {task && task.status === "running" && (
        <TaskProgressCard
          task={task}
          title="Table extraction"
          onCancel={() => s.cancelTask("table-extract")}
        />
      )}

      {ep && (
        <>
          {/* ── Two-pane: paper list (left) + selected tables (right) ──────── */}
          <div className="flex gap-4 h-[calc(100vh-15rem)] min-h-[28rem]">
            {/* LEFT: searchable paper list */}
            <Card className="w-80 shrink-0 p-0 overflow-hidden flex flex-col">
              <div className="p-2 border-b">
                <div className="relative">
                  <Search className="size-3.5 absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={q}
                    onChange={e => setQ(e.target.value)}
                    placeholder={`Filter ${totalPapers} articles…`}
                    className="pl-7 h-8 text-sm"
                  />
                </div>
              </div>
              <div className="overflow-auto flex-1">
                {filtered.map(({ p, idx }) => {
                  const active = idx === selectedIdx;
                  return (
                    <button
                      key={idx}
                      onClick={() => setSelectedIdx(idx)}
                      className={`block w-full text-left px-3 py-2.5 border-b hover:bg-muted/50 transition-colors ${active ? "bg-primary/10 border-l-2 border-l-primary" : "border-l-2 border-l-transparent"}`}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <Badge variant={p.Extracted_Tables.length ? "default" : "secondary"} className="text-[10px]">
                          {p.Extracted_Tables.length} table{p.Extracted_Tables.length === 1 ? "" : "s"}
                        </Badge>
                        <span className="text-[10px] text-muted-foreground">{p.Source}</span>
                      </div>
                      <div className="text-sm leading-snug line-clamp-2 break-words">{(p.Paper_Title || "").replace(/\s+/g, " ").trim() || "Untitled"}</div>
                    </button>
                  );
                })}
                {filtered.length === 0 && (
                  <div className="p-4 text-sm text-muted-foreground">No articles match “{q}”.</div>
                )}
              </div>
            </Card>

            {/* RIGHT: selected paper's tables */}
            <Card className="flex-1 min-w-0 p-0 overflow-hidden flex flex-col">
              {!selected ? (
                <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
                  Select an article on the left.
                </div>
              ) : (
                <PaperTablesDetail key={selectedIdx} paper={selected} format={format} />
              )}
            </Card>
          </div>
        </>
      )}
    </div>
  );
}

function GridTable({ data }: { data: string[][] }) {
  if (!data.length) return null;
  const [head, ...body] = data;
  const cols = Math.max(head?.length || 0, ...body.map(r => r.length));
  const isNum = (v: string) => {
    const t = String(v ?? "").trim();
    return t !== "" && !isNaN(Number(t.replace(/[,%$\s]/g, "")));
  };
  const cell = (v: string) => (v == null || String(v).trim() === "" ? "" : v);
  return (
    <table className="w-full text-xs border-collapse">
      <thead className="sticky top-0 z-10">
        <tr>
          {Array.from({ length: cols }).map((_, j) => (
            <th key={j} className="bg-muted text-left font-semibold text-foreground px-3 py-2 border-b border-r last:border-r-0 whitespace-nowrap">
              {cell(head?.[j] ?? "")}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {body.map((row, ri) => (
          <tr key={ri} className={ri % 2 ? "bg-muted/25" : "bg-card"}>
            {Array.from({ length: cols }).map((_, ci) => {
              const v = cell(row[ci] ?? "");
              return (
                <td key={ci} className={`px-3 py-1.5 border-b border-r last:border-r-0 align-top break-words ${isNum(v) ? "text-right tabular-nums whitespace-nowrap" : "whitespace-pre-wrap"}`}>
                  {v || <span className="text-muted-foreground/40">—</span>}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function PaperTablesDetail({ paper, format }: { paper: ExtractedPaper; format: Format }) {
  // Index of the table currently shown full-screen, or null when none.
  const [maxIdx, setMaxIdx] = useState<number | null>(null);
  const maxTable = maxIdx !== null ? paper.Extracted_Tables[maxIdx] : null;
  const fmtLabel = format.split(" ")[0];

  const totalRows = paper.Extracted_Tables.reduce((a, t) => a + t.data.length, 0);
  const maxCols = Math.max(0, ...paper.Extracted_Tables.map(t => t.data[0]?.length || 0));

  return (
    <>
      <div className="border-b p-4 space-y-3.5 bg-gradient-to-b from-muted/40 to-transparent">
        {paper.Paper_URL ? (
          <a href={paper.Paper_URL} target="_blank" rel="noreferrer"
             className="group block font-medium leading-snug hover:text-primary transition-colors" title="Open the full article">
            {paper.Paper_Title}
            <ExternalLink className="inline size-3.5 ml-1.5 align-[-1px] opacity-40 group-hover:opacity-90 transition-opacity" />
          </a>
        ) : (
          <div className="font-medium leading-snug">{paper.Paper_Title}</div>
        )}
        <div className="flex items-stretch gap-4 sm:gap-6 flex-wrap">
          <InlineStat icon={FileText} value={paper.Source || "—"} label="Source" />
          <PaneDivider />
          <InlineStat icon={Table2} tone="success" value={paper.Extracted_Tables.length} label="Tables" />
          <InlineStat icon={Rows3} value={totalRows.toLocaleString()} label="Rows" tone="neutral" />
          <InlineStat icon={Columns3} value={maxCols} label="Columns" tone="neutral" />
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4">
        {paper.Extracted_Tables.length > 0 ? (
          <Tabs defaultValue="0">
            <div className="relative">
              <div className="overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
                <TabsList className="w-max gap-0.5">
                  {paper.Extracted_Tables.map((t, i) => (
                    <TabsTrigger key={i} value={String(i)} className="whitespace-nowrap data-[state=active]:shadow-sm transition-shadow">{t.type} {i + 1}</TabsTrigger>
                  ))}
                </TabsList>
              </div>
              {/* Fade hint that there are more tabs to scroll to. */}
              <div className="pointer-events-none absolute right-0 top-0 bottom-0 w-10 bg-gradient-to-l from-card to-transparent" />
            </div>
            {paper.Extracted_Tables.map((t, i) => (
              <TabsContent key={i} value={String(i)} className="space-y-3 mt-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">{t.type} · {t.data.length} row{t.data.length === 1 ? "" : "s"}</span>
                  <div className="flex items-center gap-1">
                    <CopyTableButton data={t.data} />
                    <Button size="sm" variant="ghost" onClick={() => exportTable(paper, t, i, format)} className="h-7 px-2 text-muted-foreground" title={`Export ${fmtLabel}`}>
                      <Download className="size-4 mr-1.5" />Export
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => setMaxIdx(i)} className="h-7 px-2 text-muted-foreground" title="Maximize table">
                      <Maximize2 className="size-4 mr-1.5" />Expand
                    </Button>
                  </div>
                </div>
                <div className="rounded-lg border max-h-[28rem] overflow-auto shadow-sm">
                  <GridTable data={t.data} />
                </div>
                {t.caption && <div className="text-xs italic text-muted-foreground leading-relaxed">{t.caption}</div>}
              </TabsContent>
            ))}
          </Tabs>
        ) : <Alert><AlertDescription>No tables found in this article. Click the title above to open it and check for tables in figures or the full text.</AlertDescription></Alert>}
      </div>

      {/* Full-screen view of a single table. */}
      <Dialog open={maxIdx !== null} onOpenChange={(o) => { if (!o) setMaxIdx(null); }}>
        <DialogContent className="max-w-[99vw] w-[99vw] h-[97vh] sm:max-w-[99vw] flex flex-col p-3 gap-2">
          {maxTable && (
            <>
              <DialogHeader className="pr-8">
                <DialogTitle className="text-base leading-snug">
                  {maxTable.type} {(maxIdx ?? 0) + 1} · <span className="font-normal text-muted-foreground">{paper.Paper_Title}</span>
                </DialogTitle>
              </DialogHeader>
              <div className="flex-1 min-h-0 overflow-auto rounded-md border">
                <GridTable data={maxTable.data} />
              </div>
              {maxTable.caption && <div className="text-xs italic text-muted-foreground shrink-0">{maxTable.caption}</div>}
              <div className="shrink-0 flex items-center gap-2">
                <Button size="sm" variant="outline" onClick={() => exportTable(paper, maxTable, maxIdx ?? 0, format)}>
                  <Download className="size-4 mr-2" />Export {fmtLabel}
                </Button>
                <CopyTableButton data={maxTable.data} className="border rounded-md" />
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
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
// Copy a table to the clipboard as TSV (tab-separated), which pastes as real
// cells into Excel / Google Sheets / Word. Shows a brief check-mark on success.
function CopyTableButton({ data, className = "" }: { data: string[][]; className?: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    const tsv = data.map(row => row.map(c => String(c ?? "").replace(/\t/g, " ").replace(/\r?\n/g, " ")).join("\t")).join("\n");
    try {
      await navigator.clipboard.writeText(tsv);
      setCopied(true);
      toast.success("Table copied to clipboard");
      setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error("Copy failed");
    }
  };
  return (
    <Button size="sm" variant="ghost" onClick={copy} className={`h-7 px-2 text-muted-foreground ${className}`} title="Copy table to clipboard">
      {copied ? <Check className="size-4 mr-1.5 text-emerald-600" /> : <Copy className="size-4 mr-1.5" />}{copied ? "Copied" : "Copy"}
    </Button>
  );
}
function exportTableCsv(paper: { Paper_Title: string }, t: { data: string[][] }, i: number) {
  const csv = t.data.map(row => row.map(c => `"${String(c).replace(/"/g, '""')}"`).join(",")).join("\n");
  download(`table_${i + 1}_${paper.Paper_Title.slice(0, 30).replace(/\s/g, "_")}.csv`, csv, "text/csv");
}

// Export a single table honouring the currently-selected output format, mirroring
// how exportAll() routes each format: JSON → .json, CSV → .csv, Excel → a
// one-paper, one-table styled workbook (reusing the bulk XLSX path).
function exportTable(paper: ExtractedPaper, t: ExtractedPaper["Extracted_Tables"][number], i: number, format: Format) {
  const slug = paper.Paper_Title.slice(0, 30).replace(/\s/g, "_");
  if (format === "JSON Export") {
    download(`table_${i + 1}_${slug}.json`, JSON.stringify(t, null, 2), "application/json");
  } else if (format === "CSV Export") {
    exportTableCsv(paper, t, i);
  } else {
    void downloadExtractedTablesXlsx([{ ...paper, Extracted_Tables: [t] }]);
  }
}
function download(name: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

// ---- XLSX export ----------------------------------------------------------
//
// One worksheet per paper. Inside each worksheet, the tables are stacked
// vertically: a banded title row, then the column headers (dark band),
// then the data rows alternating zebra-striping for readability. A summary
// worksheet at the front lists every paper, table count, and links into the
// individual worksheets.

type ExtractedPaper = {
  Paper_Title: string;
  Paper_URL: string;
  Source: string;
  Extracted_Tables: { title: string; type?: string; data: string[][]; caption?: string }[];
};

const TABLE_TYPE_FILL: Record<string, string> = {
  "Demographics":           "FFE0E7FF",   // indigo-100
  "Statistical Results":    "FFDCFCE7",   // green-100
  "Adverse Events":         "FFFEE2E2",   // red-100
  "Outcomes":               "FFFEF3C7",   // amber-100
  "General":                "FFF1F5F9",   // slate-100
  "All":                    "FFF1F5F9",
};
const TABLE_TYPE_TEXT: Record<string, string> = {
  "Demographics":           "FF3730A3",
  "Statistical Results":    "FF065F46",
  "Adverse Events":         "FF9F1239",
  "Outcomes":               "FF92400E",
  "General":                "FF475569",
  "All":                    "FF475569",
};

function _safeSheetName(name: string, taken: Set<string>): string {
  // Excel sheet names: ≤ 31 chars, no `:\/?*[]`.
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

async function downloadExtractedTablesXlsx(papers: ExtractedPaper[]) {
  if (!papers || papers.length === 0) {
    toast.error("Nothing to export. Run table extraction first.");
    return;
  }

  const wb = new ExcelJS.Workbook();
  wb.creator = "Evidence Engine";
  wb.created = new Date();

  // ---- Summary worksheet --------------------------------------------------
  const summary = wb.addWorksheet("Summary", {
    views: [{ state: "frozen", ySplit: 1 }],
  });
  summary.columns = [
    { header: "Paper",          key: "title",  width: 60 },
    { header: "Source",         key: "source", width: 14 },
    { header: "Tables",         key: "count",  width: 10 },
    { header: "URL",            key: "url",    width: 50 },
    { header: "Worksheet link", key: "link",   width: 24 },
  ];
  styleHeaderRow(summary.getRow(1));

  const takenSheetNames = new Set<string>(["summary"]);
  const sheetMap = new Map<number, string>();   // paper index → sheet name

  papers.forEach((p, idx) => {
    const sheetName = _safeSheetName(p.Paper_Title, takenSheetNames);
    sheetMap.set(idx, sheetName);
    const row = summary.addRow({
      title:  p.Paper_Title,
      source: p.Source,
      count:  p.Extracted_Tables.length,
      url:    p.Paper_URL,
      link:   sheetName,
    });
    row.eachCell((cell) => {
      cell.alignment = { vertical: "top", wrapText: true };
      cell.font = { size: 10 };
    });
    if (p.Paper_URL) {
      const urlCell = row.getCell("url");
      urlCell.value = { text: p.Paper_URL, hyperlink: p.Paper_URL };
      urlCell.font = { size: 10, color: { argb: "FF1D4ED8" }, underline: true };
    }
    // Worksheet-link cell: clickable internal hyperlink to the paper's tab.
    const linkCell = row.getCell("link");
    linkCell.value = {
      text: `Open → ${sheetName}`,
      hyperlink: `#'${sheetName}'!A1`,
    } as any;
    linkCell.font = { size: 10, color: { argb: "FF1D4ED8" }, underline: true };

    // Tint the Tables-count cell green when > 0, red when 0.
    const cntCell = row.getCell("count");
    cntCell.alignment = { vertical: "top", horizontal: "center" };
    cntCell.fill = {
      type: "pattern", pattern: "solid",
      fgColor: { argb: p.Extracted_Tables.length > 0 ? "FFDCFCE7" : "FFFEE2E2" },
    };
    cntCell.font = {
      size: 10, bold: true,
      color: { argb: p.Extracted_Tables.length > 0 ? "FF065F46" : "FF991B1B" },
    };
    row.height = 36;
  });

  summary.autoFilter = { from: { row: 1, column: 1 }, to: { row: 1, column: 5 } };

  // ---- One worksheet per paper -------------------------------------------
  papers.forEach((p, idx) => {
    const sheetName = sheetMap.get(idx)!;
    const ws = wb.addWorksheet(sheetName, {
      views: [{ state: "frozen", ySplit: 1 }],
    });

    // Top banner: paper title, source, link back to summary, total tables.
    const banner = ws.addRow([p.Paper_Title]);
    banner.height = 30;
    ws.mergeCells(banner.number, 1, banner.number, 6);
    const b = banner.getCell(1);
    b.font = { size: 13, bold: true, color: { argb: "FFFFFFFF" } };
    b.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF1F2937" } };
    b.alignment = { vertical: "middle", horizontal: "left", wrapText: true };

    const meta = ws.addRow([`${p.Source}  ·  ${p.Extracted_Tables.length} table${p.Extracted_Tables.length === 1 ? "" : "s"}`]);
    ws.mergeCells(meta.number, 1, meta.number, 6);
    const m = meta.getCell(1);
    m.font = { size: 10, italic: true, color: { argb: "FF6B7280" } };
    m.alignment = { vertical: "middle", horizontal: "left" };

    if (p.Paper_URL) {
      const linkRow = ws.addRow([p.Paper_URL]);
      ws.mergeCells(linkRow.number, 1, linkRow.number, 6);
      const lc = linkRow.getCell(1);
      lc.value = { text: p.Paper_URL, hyperlink: p.Paper_URL };
      lc.font = { size: 10, color: { argb: "FF1D4ED8" }, underline: true };
      lc.alignment = { vertical: "middle", horizontal: "left" };
    }

    const backRow = ws.addRow(["← Back to Summary"]);
    ws.mergeCells(backRow.number, 1, backRow.number, 6);
    const back = backRow.getCell(1);
    back.value = { text: "← Back to Summary", hyperlink: "#Summary!A1" } as any;
    back.font = { size: 10, color: { argb: "FF1D4ED8" }, underline: true };
    back.alignment = { vertical: "middle", horizontal: "left" };

    ws.addRow([]);   // gap

    if (p.Extracted_Tables.length === 0) {
      const noRow = ws.addRow(["No tables extracted from this paper."]);
      ws.mergeCells(noRow.number, 1, noRow.number, 6);
      noRow.getCell(1).font = { size: 10, italic: true, color: { argb: "FF6B7280" } };
      // Reasonable default widths for the empty case.
      for (let c = 1; c <= 6; c++) ws.getColumn(c).width = 22;
      return;
    }

    // Compute max column count across tables so we can set sensible widths.
    let maxCols = 1;
    for (const t of p.Extracted_Tables) {
      for (const row of t.data) {
        if (row.length > maxCols) maxCols = row.length;
      }
    }
    for (let c = 1; c <= Math.max(maxCols, 4); c++) ws.getColumn(c).width = 24;

    p.Extracted_Tables.forEach((t, ti) => {
      const typeKey = t.type || "All";
      const fill = TABLE_TYPE_FILL[typeKey] || TABLE_TYPE_FILL["General"];
      const textColor = TABLE_TYPE_TEXT[typeKey] || TABLE_TYPE_TEXT["General"];

      // Table title row with type chip
      const titleRow = ws.addRow([`${t.title || `Table ${ti + 1}`}  ·  ${typeKey}`]);
      ws.mergeCells(titleRow.number, 1, titleRow.number, Math.max(maxCols, 1));
      const tc = titleRow.getCell(1);
      tc.font = { size: 11, bold: true, color: { argb: textColor } };
      tc.fill = { type: "pattern", pattern: "solid", fgColor: { argb: fill } };
      tc.alignment = { vertical: "middle", horizontal: "left", wrapText: true };
      titleRow.height = 22;

      // Optional caption row
      if (t.caption) {
        const capRow = ws.addRow([t.caption]);
        ws.mergeCells(capRow.number, 1, capRow.number, Math.max(maxCols, 1));
        const cc = capRow.getCell(1);
        cc.font = { size: 9, italic: true, color: { argb: "FF6B7280" } };
        cc.alignment = { vertical: "top", horizontal: "left", wrapText: true };
        capRow.height = 24;
      }

      // Header + body rows
      const rows = t.data;
      if (rows.length > 0) {
        const headerRow = ws.addRow(rows[0]);
        headerRow.eachCell((cell) => {
          cell.font = { size: 10, bold: true, color: { argb: "FFFFFFFF" } };
          cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF374151" } };
          cell.alignment = { vertical: "middle", horizontal: "left", wrapText: true };
          cell.border = { bottom: { style: "thin", color: { argb: "FF1F2937" } } };
        });
        headerRow.height = 22;

        for (let r = 1; r < rows.length; r++) {
          const dataRow = ws.addRow(rows[r]);
          const zebra = r % 2 === 1;
          dataRow.eachCell((cell) => {
            cell.font = { size: 10 };
            cell.alignment = { vertical: "top", horizontal: "left", wrapText: true };
            cell.border = { bottom: { style: "hair", color: { argb: "FFE5E7EB" } } };
            if (zebra) {
              cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFF9FAFB" } };
            }
          });
        }
      }

      // Spacer row between tables
      ws.addRow([]);
      ws.addRow([]);
    });
  });

  const buf = await wb.xlsx.writeBuffer();
  const blob = new Blob([buf], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  a.href = url;
  a.download = `extracted-tables-${stamp}.xlsx`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  toast.success(`Exported ${papers.length} articles (${papers.reduce((acc, p) => acc + p.Extracted_Tables.length, 0)} tables) to XLSX.`);
}

function styleHeaderRow(row: ExcelJS.Row) {
  row.height = 26;
  row.eachCell((cell) => {
    cell.font = { size: 11, bold: true, color: { argb: "FFFFFFFF" } };
    cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF1F2937" } };
    cell.alignment = { vertical: "middle", horizontal: "left", wrapText: true };
    cell.border = { bottom: { style: "thin", color: { argb: "FF374151" } } };
  });
}
