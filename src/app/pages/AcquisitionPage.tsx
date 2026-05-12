import { useStore, FullTextRecord } from "../lib/store";
import { AIService } from "../lib/mockServices";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../components/ui/collapsible";
import { FileDown, CheckCircle2, AlertTriangle, ExternalLink, ChevronDown, RefreshCcw } from "lucide-react";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";

export function AcquisitionPage() {
  const s = useStore();
  const task = s.tasks["fulltext-fetch"];
  const running = task?.status === "running";

  if (!s.results) return <Alert><AlertDescription>Complete Abstract Screening first to unlock full-text acquisition.</AlertDescription></Alert>;
  const included = s.results.filter(r => r.Decision === "INCLUDE");
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

  async function uploadPdfFor(p: typeof included[number], file: File) {
    try {
      let text = "";
      if (file.name.toLowerCase().endsWith(".pdf") || file.type === "application/pdf") {
        const buf = await file.arrayBuffer();
        const pdfjs: any = await import("pdfjs-dist");
        // Disable the worker to avoid extra asset wiring in this environment
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
      s.setFullTexts(prev => ({
        ...prev,
        [p.paper_id]: { paper_id: p.paper_id, title: p.Title, url: p.URL, source: p.Source, status: "found", text },
      }));
      toast.success(`Loaded ${file.name} (${text.length.toLocaleString()} chars)`);
    } catch (e: any) {
      console.error(`PDF upload failed for ${p.Title}: ${e?.message}`);
      toast.error(e?.message || "Could not read file");
    }
  }

  const records: FullTextRecord[] = included.map(p => s.fullTexts[p.paper_id] || { paper_id: p.paper_id, title: p.Title, url: p.URL, source: p.Source, status: "pending" as const });
  const found = records.filter(r => r.status === "found").length;
  const missing = records.filter(r => r.status === "missing").length;
  const pending = records.filter(r => r.status === "pending").length;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-4 gap-3">
        <Stat label="Included" value={included.length} />
        <Stat label="Acquired" value={found} variant="success" />
        <Stat label="Missing" value={missing} variant="warn" />
        <Stat label="Not yet fetched" value={pending} />
      </div>

      <Card className="p-4 flex flex-wrap gap-2 items-center">
        <Button onClick={() => runFetch(false)} disabled={running}>
          <FileDown className="size-4 mr-2" />{running ? "Fetching..." : "Fetch all full texts"}
        </Button>
        {missing > 0 && (
          <Button variant="outline" onClick={() => runFetch(true)} disabled={running}>
            <RefreshCcw className="size-4 mr-2" />Retry missing ({missing})
          </Button>
        )}
      </Card>

      {task && task.status === "running" && (
        <TaskProgressCard
          task={task}
          title="Acquiring full texts"
          onCancel={() => s.cancelTask("fulltext-fetch")}
        />
      )}

      {missing > 0 && (
        <Alert>
          <AlertDescription>
            <strong>{missing} paper{missing > 1 ? "s" : ""} could not be retrieved.</strong> Common reasons: paywalls, no PDF available, or broken links. You can upload a PDF/text file manually for any missing item below.
          </AlertDescription>
        </Alert>
      )}

      <div className="space-y-2">
        {records.map(r => (
          <Collapsible key={r.paper_id}>
            <Card className={r.status === "missing" ? "border-amber-300" : ""}>
              <div className="w-full flex items-center gap-3 p-4">
                <StatusBadge status={r.status} />
                <CollapsibleTrigger asChild>
                  <button className="flex-1 min-w-0 text-left hover:opacity-80">
                    <div className="truncate">{r.title}</div>
                    <div className="text-xs text-muted-foreground flex items-center gap-2">
                      <Badge variant="outline" className="text-[10px]">{r.source}</Badge>
                      {r.status === "missing" && r.reason && <span className="text-amber-700">{r.reason}</span>}
                      {r.status === "found" && r.text && <span>{r.text.length.toLocaleString()} chars</span>}
                    </div>
                  </button>
                </CollapsibleTrigger>
                {r.status === "missing" && (
                  <label className="inline-flex items-center gap-1 text-xs cursor-pointer bg-primary text-primary-foreground rounded px-2 py-1 hover:opacity-90 shrink-0">
                    <input type="file" accept=".pdf,.txt,.md" className="hidden"
                      onChange={(e) => { const f = e.target.files?.[0]; if (f) uploadPdfFor(included.find(p => p.paper_id === r.paper_id)!, f); }} />
                    Upload PDF
                  </label>
                )}
                <CollapsibleTrigger asChild>
                  <button className="shrink-0"><ChevronDown className="size-4 text-muted-foreground" /></button>
                </CollapsibleTrigger>
              </div>
              <CollapsibleContent>
                <div className="px-4 pb-4 space-y-3">
                  <div className="flex flex-wrap gap-2">
                    {r.url && <a href={r.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-sm text-primary hover:underline"><ExternalLink className="size-3" />Open source</a>}
                    <label className="inline-flex items-center gap-1 text-sm cursor-pointer text-primary hover:underline">
                      <input type="file" accept=".txt,.pdf,.md" className="hidden"
                        onChange={(e) => { const f = e.target.files?.[0]; if (f) uploadPdfFor(included.find(p => p.paper_id === r.paper_id)!, f); }} />
                      Upload file
                    </label>
                  </div>
                  {r.status === "found" && r.text && (
                    <pre className="text-xs whitespace-pre-wrap bg-muted/40 rounded p-3 max-h-64 overflow-auto font-mono">{r.text.slice(0, 4000)}{r.text.length > 4000 ? "\n\n…" : ""}</pre>
                  )}
                  {r.status === "missing" && (
                    <div className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded p-3">
                      <AlertTriangle className="size-4 inline mr-1" />
                      Full text unavailable. {r.reason || "Source did not return retrievable content."} Upload a PDF or text file above to provide it manually.
                    </div>
                  )}
                  {r.status === "pending" && (
                    <div className="text-sm text-muted-foreground">Not fetched yet — click "Fetch all full texts" above.</div>
                  )}
                </div>
              </CollapsibleContent>
            </Card>
          </Collapsible>
        ))}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: FullTextRecord["status"] }) {
  if (status === "found") return <span className="inline-flex items-center gap-1 text-green-700 bg-green-50 border border-green-200 rounded px-2 py-0.5 text-xs"><CheckCircle2 className="size-3" />Acquired</span>;
  if (status === "missing") return <span className="inline-flex items-center gap-1 text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-0.5 text-xs"><AlertTriangle className="size-3" />Missing</span>;
  return <span className="inline-flex items-center gap-1 text-muted-foreground bg-muted border rounded px-2 py-0.5 text-xs">Pending</span>;
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
