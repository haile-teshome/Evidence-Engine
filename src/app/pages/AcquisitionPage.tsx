import { useState } from "react";
import { useStore, FullTextRecord } from "../lib/store";
import { AIService } from "../lib/mockServices";
import { effectiveAbstractDecision } from "../lib/exclusionBucketing";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Input } from "../components/ui/input";
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle,
} from "../components/ui/dialog";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "../components/ui/select";
import {
  FileDown, CheckCircle2, AlertTriangle, ExternalLink, RefreshCcw, Upload, FileUp, FileText, X, Search,
} from "lucide-react";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";

const NONE = "__none__"; // Radix Select cannot use an empty-string value

// ── File text extraction (PDF via pdfjs, else plain text) ────────────────────
async function extractFileText(file: File): Promise<string> {
  let text = "";
  if (file.name.toLowerCase().endsWith(".pdf") || file.type === "application/pdf") {
    const buf = await file.arrayBuffer();
    const pdfjs: any = await import("pdfjs-dist");
    if (pdfjs.GlobalWorkerOptions) pdfjs.GlobalWorkerOptions.workerSrc = "";
    const doc = await pdfjs.getDocument({ data: buf, disableWorker: true, isEvalSupported: false }).promise;
    const pages: string[] = [];
    for (let i = 1; i <= doc.numPages; i++) {
      const page = await doc.getPage(i);
      const content = await page.getTextContent();
      pages.push(content.items.map((it: any) => ("str" in it ? it.str : "")).join(" "));
    }
    text = pages.join("\n\n");
  } else {
    text = await file.text();
  }
  if (!text.trim()) throw new Error("Could not extract any text from the file.");
  return text;
}

// ── Filename → study matching for bulk upload ────────────────────────────────
const _norm = (s: string) => s.toLowerCase().replace(/\.[a-z0-9]+$/i, "").replace(/[^a-z0-9]+/g, " ").trim();
const _tokens = (s: string) => new Set(_norm(s).split(" ").filter(w => w.length > 2));

function matchScore(fileName: string, study: { Title: string; paper_id: string }): number {
  // Exact id / PMID / DOI-digits embedded in the filename → certain match
  const idDigits = (study.paper_id.match(/\d{5,}/) || [])[0];
  if (idDigits && fileName.replace(/[^0-9]/g, "").includes(idDigits)) return 1;
  const ft = _tokens(fileName);
  const tt = _tokens(study.Title);
  if (ft.size === 0 || tt.size === 0) return 0;
  let common = 0;
  ft.forEach(t => { if (tt.has(t)) common++; });
  return common / Math.min(ft.size, tt.size);
}

type BulkRow = { file: File; matchId: string; score: number };

export function AcquisitionPage() {
  const s = useStore();
  const task = s.tasks["fulltext-fetch"];
  const running = task?.status === "running";

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [bulkOpen, setBulkOpen] = useState(false);
  const [bulkRows, setBulkRows] = useState<BulkRow[]>([]);
  const [bulkBusy, setBulkBusy] = useState(false);

  if (!s.results) return <Alert><AlertDescription>Complete Abstract Screening first to unlock full-text acquisition.</AlertDescription></Alert>;
  // Honour reviewer overrides — papers the user marked Keep at abstract
  // screening get their full text fetched here even if the AI excluded them.
  const included = s.results.filter(r => effectiveAbstractDecision(r, s.abstractOverrides) === "INCLUDE");
  if (included.length === 0) return <Alert><AlertDescription>No included papers from screening yet.</AlertDescription></Alert>;

  async function runFetch(onlyMissing = false) {
    const queue = onlyMissing ? included.filter(p => s.fullTexts[p.paper_id]?.status !== "found") : included;
    if (queue.length === 0) { toast.info("Nothing to fetch."); return; }

    const { abort } = s.startTask("fulltext-fetch", [{ id: "fetch", label: "Fetching full texts", status: "running" }]);
    s.updateTask("fulltext-fetch", { progress: { done: 0, total: queue.length } });
    const signal = abort.signal;

    try {
      let found = 0;
      for (let i = 0; i < queue.length; i++) {
        if (signal.aborted) break;
        const p = queue[i];
        s.updateTask("fulltext-fetch", {
          progress: { done: i, total: queue.length, label: p.Title.slice(0, 80) },
          detail: p.Title.slice(0, 80),
        });
        s.setFullTexts(prev => ({ ...prev, [p.paper_id]: { paper_id: p.paper_id, title: p.Title, url: p.URL, source: p.Source, status: "pending" } }));
        try {
          const res = await AIService.fetchFullText({ Title: p.Title, URL: p.URL, Source: p.Source }, signal);
          s.setFullTexts(prev => ({
            ...prev,
            [p.paper_id]: {
              paper_id: p.paper_id, title: p.Title, url: p.URL, source: p.Source,
              status: res.status, text: res.text, reason: res.reason,
              retrieved_via: res.source,
            },
          }));
          if (res.status === "found") found++;
        } catch (e: any) {
          if (signal.aborted) break;
          console.error(`fulltext-fetch ${i + 1} failed:`, e?.message);
        }
        s.updateTask("fulltext-fetch", { progress: { done: i + 1, total: queue.length } });
      }
      if (signal.aborted) {
        s.updateTask("fulltext-fetch", { status: "canceled" });
        toast.info("Full-text fetch canceled");
      } else {
        s.updateTask("fulltext-fetch", { status: "done" });
        toast.success(`Acquired ${found}/${queue.length} full texts`);
      }
    } catch (e: any) {
      s.updateTask("fulltext-fetch", { status: "error", detail: e?.message });
    }
  }

  function storeText(p: { paper_id: string; Title: string; URL: string; Source: string }, text: string) {
    s.setFullTexts(prev => ({
      ...prev,
      [p.paper_id]: { paper_id: p.paper_id, title: p.Title, url: p.URL, source: p.Source, status: "found", text },
    }));
  }

  async function uploadPdfFor(p: typeof included[number], file: File) {
    try {
      const text = await extractFileText(file);
      storeText(p, text);
      setSelectedId(p.paper_id);
      toast.success(`Loaded ${file.name} (${text.length.toLocaleString()} chars)`);
    } catch (e: any) {
      console.error(`Upload failed for ${p.Title}: ${e?.message}`);
      toast.error(e?.message || "Could not read file");
    }
  }

  // ── Bulk upload + matching ────────────────────────────────────────────────
  function onBulkFiles(files: File[]) {
    const rows: BulkRow[] = files.map(file => {
      let best = { id: "", score: 0 };
      for (const p of included) {
        const sc = matchScore(file.name, p);
        if (sc > best.score) best = { id: p.paper_id, score: sc };
      }
      return { file, matchId: best.score >= 0.34 ? best.id : "", score: best.score };
    });
    setBulkRows(prev => [...prev, ...rows]);
  }

  async function applyBulk() {
    setBulkBusy(true);
    let ok = 0, fail = 0;
    for (const row of bulkRows) {
      if (!row.matchId) continue;
      const p = included.find(x => x.paper_id === row.matchId);
      if (!p) continue;
      try {
        const text = await extractFileText(row.file);
        storeText(p, text);
        ok++;
      } catch { fail++; }
    }
    setBulkBusy(false);
    setBulkOpen(false);
    setBulkRows([]);
    toast.success(`Matched ${ok} file${ok === 1 ? "" : "s"}` + (fail ? `, ${fail} failed` : ""));
  }

  const records: FullTextRecord[] = included.map(p => s.fullTexts[p.paper_id] || { paper_id: p.paper_id, title: p.Title, url: p.URL, source: p.Source, status: "pending" as const });
  const found = records.filter(r => r.status === "found").length;
  const missing = records.filter(r => r.status === "missing").length;
  const pending = records.filter(r => r.status === "pending").length;

  const selected = records.find(r => r.paper_id === selectedId) ?? records[0];
  const selectedPaper = included.find(p => p.paper_id === selected?.paper_id);
  const filtered = q.trim()
    ? records.filter(r => r.title.toLowerCase().includes(q.toLowerCase()))
    : records;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-4 gap-3">
        <Stat label="Included" value={included.length} />
        <Stat label="Acquired" value={found} variant="success" />
        <Stat label="Missing" value={missing} variant="warn" />
        <Stat label="Not yet fetched" value={pending} />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" onClick={() => runFetch(false)} disabled={running}>
          <FileDown className="size-4 mr-2" />{running ? "Fetching…" : "Fetch all full texts"}
        </Button>
        {missing > 0 && (
          <Button size="sm" variant="outline" onClick={() => runFetch(true)} disabled={running}>
            <RefreshCcw className="size-4 mr-2" />Retry missing ({missing})
          </Button>
        )}
        <div className="flex-1" />
        <Button size="sm" variant="outline" onClick={() => setBulkOpen(true)}>
          <Upload className="size-4 mr-2" />Bulk upload &amp; match
        </Button>
      </div>

      {task && task.status === "running" && (
        <TaskProgressCard task={task} title="Acquiring full texts" onCancel={() => s.cancelTask("fulltext-fetch")} />
      )}

      {/* ── Two-pane: study list (left) + full text / upload (right) ───────── */}
      <div className="flex gap-4 h-[calc(100vh-19rem)] min-h-[28rem]">
        {/* LEFT: study list */}
        <Card className="w-80 shrink-0 p-0 overflow-hidden flex flex-col">
          <div className="p-2 border-b">
            <div className="relative">
              <Search className="size-3.5 absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground" />
              <Input value={q} onChange={e => setQ(e.target.value)} placeholder={`Filter ${included.length} studies…`} className="pl-7 h-8 text-sm" />
            </div>
          </div>
          <div className="overflow-auto flex-1">
            {filtered.map(r => {
              const active = r.paper_id === selected?.paper_id;
              return (
                <button
                  key={r.paper_id}
                  onClick={() => setSelectedId(r.paper_id)}
                  className={`w-full text-left px-3 py-2.5 border-b hover:bg-muted/50 transition-colors ${active ? "bg-primary/10 border-l-2 border-l-primary" : "border-l-2 border-l-transparent"}`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <StatusDot status={r.status} />
                    <Badge variant="outline" className="text-[10px]">{r.source}</Badge>
                  </div>
                  <div className="text-sm line-clamp-2 leading-snug">{r.title}</div>
                </button>
              );
            })}
            {filtered.length === 0 && (
              <div className="p-4 text-sm text-muted-foreground">No studies match “{q}”.</div>
            )}
          </div>
        </Card>

        {/* RIGHT: detail (full text or upload) */}
        <Card className="flex-1 min-w-0 p-0 overflow-hidden flex flex-col">
          {!selected ? (
            <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">Select a study on the left.</div>
          ) : (
            <>
              <div className="border-b p-4 space-y-3">
                <div className="flex items-start gap-3">
                  <StatusBadge status={selected.status} />
                  <div className="flex-1 min-w-0">
                    <div className="font-medium leading-snug">{selected.title}</div>
                    <div className="text-xs text-muted-foreground flex items-center gap-2 flex-wrap mt-1">
                      <Badge variant="outline" className="text-[10px]">{selected.source}</Badge>
                      {selected.status === "found" && selected.retrieved_via && (
                        <Badge variant="outline" className="text-[10px] bg-emerald-50 text-emerald-700 border-emerald-200">via {selected.retrieved_via}</Badge>
                      )}
                      {selected.status === "found" && selected.text && <span>{selected.text.length.toLocaleString()} chars</span>}
                    </div>
                  </div>
                </div>
                {(selected.url || (selected.status === "found" && selectedPaper)) && (
                  <div className="flex items-center gap-2 flex-wrap">
                    {selected.url && (
                      <a href={selected.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-sm text-primary hover:underline">
                        <ExternalLink className="size-3" />Open source
                      </a>
                    )}
                    {selected.status === "found" && selectedPaper && (
                      <label className="inline-flex items-center gap-1 text-xs cursor-pointer border rounded px-2 py-1 hover:bg-muted ml-auto">
                        <input type="file" accept=".pdf,.txt,.md" className="hidden"
                          onChange={e => { const f = e.target.files?.[0]; if (f) uploadPdfFor(selectedPaper, f); }} />
                        <FileUp className="size-3" />Replace file
                      </label>
                    )}
                  </div>
                )}
              </div>

              <div className="flex-1 overflow-auto">
                {selected.status === "found" && selected.text ? (
                  <pre className="text-xs whitespace-pre-wrap font-mono p-4 leading-relaxed">{selected.text}</pre>
                ) : selectedPaper ? (
                  <DropZone
                    reason={selected.status === "missing" ? (selected.reason || "Source did not return retrievable content.") : "Not fetched yet — click “Fetch all full texts”, or upload the file directly."}
                    warn={selected.status === "missing"}
                    onFile={f => uploadPdfFor(selectedPaper, f)}
                  />
                ) : null}
              </div>
            </>
          )}
        </Card>
      </div>

      {/* ── Bulk upload + match dialog ─────────────────────────────────────── */}
      <Dialog open={bulkOpen} onOpenChange={(o) => { setBulkOpen(o); if (!o) setBulkRows([]); }}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Bulk upload &amp; match</DialogTitle>
            <DialogDescription>
              Upload several PDFs at once. Each file is auto-matched to a study by its filename; review the matches and fix any that are wrong before applying.
            </DialogDescription>
          </DialogHeader>

          <label
            onDragOver={e => e.preventDefault()}
            onDrop={e => { e.preventDefault(); if (e.dataTransfer.files?.length) onBulkFiles(Array.from(e.dataTransfer.files)); }}
            className="flex flex-col items-center justify-center gap-1.5 border-2 border-dashed rounded-lg p-6 text-center cursor-pointer hover:bg-muted/30"
          >
            <input type="file" multiple accept=".pdf,.txt,.md" className="hidden"
              onChange={e => { if (e.target.files?.length) onBulkFiles(Array.from(e.target.files)); e.currentTarget.value = ""; }} />
            <FileUp className="size-6 text-muted-foreground" />
            <div className="text-sm font-medium">Drop files here or click to select</div>
            <div className="text-xs text-muted-foreground">PDF, TXT, or MD — multiple files supported</div>
          </label>

          {bulkRows.length > 0 && (
            <div className="space-y-2 max-h-[42vh] overflow-auto pr-1">
              {bulkRows.map((row, i) => (
                <div key={i} className="flex items-center gap-2 border rounded-md p-2">
                  <FileText className="size-4 text-muted-foreground shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm truncate">{row.file.name}</div>
                    <MatchHint matched={!!row.matchId} score={row.score} />
                  </div>
                  <Select
                    value={row.matchId || NONE}
                    onValueChange={v => setBulkRows(rows => rows.map((r, j) => j === i ? { ...r, matchId: v === NONE ? "" : v } : r))}
                  >
                    <SelectTrigger className="w-64 h-8 text-xs"><SelectValue placeholder="— choose study —" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value={NONE}>— unmatched —</SelectItem>
                      {included.map(p => (
                        <SelectItem key={p.paper_id} value={p.paper_id} className="text-xs">{p.Title.slice(0, 80)}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <button className="shrink-0 text-muted-foreground hover:text-foreground" onClick={() => setBulkRows(rows => rows.filter((_, j) => j !== i))}>
                    <X className="size-4" />
                  </button>
                </div>
              ))}
            </div>
          )}

          <DialogFooter>
            {bulkRows.length > 0 && <Button variant="ghost" onClick={() => setBulkRows([])} disabled={bulkBusy}>Clear</Button>}
            <Button onClick={applyBulk} disabled={bulkBusy || !bulkRows.some(r => r.matchId)}>
              {bulkBusy ? "Matching…" : `Apply ${bulkRows.filter(r => r.matchId).length} match${bulkRows.filter(r => r.matchId).length === 1 ? "" : "es"}`}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function DropZone({ reason, warn, onFile }: { reason?: string; warn?: boolean; onFile: (f: File) => void }) {
  return (
    <label
      onDragOver={e => e.preventDefault()}
      onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f) onFile(f); }}
      className="m-4 flex flex-col items-center justify-center gap-2 border-2 border-dashed rounded-lg p-10 text-center cursor-pointer hover:bg-muted/30 min-h-[12rem]"
    >
      <input type="file" accept=".pdf,.txt,.md" className="hidden"
        onChange={e => { const f = e.target.files?.[0]; if (f) onFile(f); }} />
      <FileUp className="size-8 text-muted-foreground" />
      <div className="text-sm font-medium">Drop a PDF or text file here, or click to browse</div>
      {reason && (
        <div className={`text-xs max-w-md ${warn ? "text-amber-700" : "text-muted-foreground"}`}>
          {warn && <AlertTriangle className="size-3 inline mr-1" />}{reason}
        </div>
      )}
    </label>
  );
}

function MatchHint({ matched, score }: { matched: boolean; score: number }) {
  if (!matched) return <div className="text-[11px] text-amber-700">No confident match — choose a study</div>;
  const strong = score >= 0.6;
  return (
    <div className={`text-[11px] ${strong ? "text-emerald-700" : "text-muted-foreground"}`}>
      {strong ? "Strong match" : "Likely match"} · {Math.round(score * 100)}%
    </div>
  );
}

function StatusDot({ status }: { status: FullTextRecord["status"] }) {
  const cls = status === "found" ? "bg-green-500" : status === "missing" ? "bg-amber-500" : "bg-muted-foreground/40";
  return <span className={`size-2 rounded-full shrink-0 ${cls}`} />;
}

function StatusBadge({ status }: { status: FullTextRecord["status"] }) {
  if (status === "found") return <span className="inline-flex items-center gap-1 text-green-700 bg-green-50 border border-green-200 rounded px-2 py-0.5 text-xs shrink-0"><CheckCircle2 className="size-3" />Acquired</span>;
  if (status === "missing") return <span className="inline-flex items-center gap-1 text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-0.5 text-xs shrink-0"><AlertTriangle className="size-3" />Missing</span>;
  return <span className="inline-flex items-center gap-1 text-muted-foreground bg-muted border rounded px-2 py-0.5 text-xs shrink-0">Pending</span>;
}

function Stat({ label, value, variant }: { label: string; value: any; variant?: "success" | "warn" }) {
  const cls = variant === "success" ? "text-green-700" : variant === "warn" ? "text-amber-700" : "text-foreground";
  return (
    <Card className="p-3 text-center">
      <div className={`text-2xl font-bold ${cls}`}>{value}</div>
      <div className="text-xs text-muted-foreground">{label}</div>
    </Card>
  );
}
