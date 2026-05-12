import { useState } from "react";
import { useStore } from "../lib/store";
import { AIService } from "../lib/mockServices";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../components/ui/collapsible";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../components/ui/table";
import { ChevronDown, Download, Table2, ExternalLink } from "lucide-react";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";

type Format = "DataFrame" | "CSV Export" | "JSON Export" | "Excel Export";

export function ExtractionPage() {
  const s = useStore();
  const [format, setFormat] = useState<Format>("DataFrame");
  const task = s.tasks["table-extract"];
  const running = task?.status === "running";

  if (!s.results) return <Alert><AlertDescription>Complete Abstract Screening first to unlock Table Extraction.</AlertDescription></Alert>;

  const passed = s.results.filter(r => r.Decision === "INCLUDE");
  if (passed.length === 0) return <Alert><AlertDescription>No included papers available for extraction.</AlertDescription></Alert>;

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
          const tables = await AIService.extractTables({ Title: p.Title, URL: p.URL, Source: p.Source }, signal);
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
        toast.info(`Canceled — ${out.length} of ${passed.length} processed`);
      } else {
        s.updateTask("table-extract", { status: "done" });
        toast.success(`Tables extracted from ${out.length} papers`);
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
      toast.info("DataFrame view shown above; per-table CSV available within each table.");
    }
  }

  const ep = s.extractedPapers;
  const totalPapers = ep?.length || 0;
  const withTables = ep?.filter(p => p.Extracted_Tables.length > 0).length || 0;
  const withoutTables = totalPapers - withTables;

  return (
    <div className="space-y-4">
      <Alert><AlertDescription>{passed.length} included papers available for table extraction.</AlertDescription></Alert>

      <Card className="p-4 space-y-3">
        <h3 className="font-medium">Extraction Settings</h3>
        <div className="grid grid-cols-[1fr_auto] gap-3 items-end">
          <div>
            <label className="text-sm text-muted-foreground">Output Format</label>
            <Select value={format} onValueChange={v => setFormat(v as Format)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="DataFrame">DataFrame</SelectItem>
                <SelectItem value="CSV Export">CSV Export</SelectItem>
                <SelectItem value="JSON Export">JSON Export</SelectItem>
                <SelectItem value="Excel Export">Excel Export</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <Button onClick={extract} disabled={running}><Table2 className="size-4 mr-2" />{running ? "Extracting..." : "Start Table Extraction"}</Button>
        </div>
      </Card>

      {task && task.status === "running" && (
        <TaskProgressCard
          task={task}
          title="Table extraction"
          onCancel={() => s.cancelTask("table-extract")}
        />
      )}

      {ep && (
        <>
          {withoutTables > 0 && (
            <Alert>
              <AlertDescription>
                <strong>⚠️ {withoutTables} of {totalPapers} papers</strong> had no extractable tables — typically paywalled / closed access, no tabular content, or PDF-only.
              </AlertDescription>
            </Alert>
          )}

          {ep.map((paper, idx) => (
            <Collapsible key={idx}>
              <CollapsibleTrigger asChild>
                <Button variant="outline" className="w-full justify-between text-left h-auto py-3">
                  <span className="truncate flex-1">{paper.Paper_Title}</span>
                  <Badge variant={paper.Extracted_Tables.length ? "default" : "secondary"}>{paper.Extracted_Tables.length} tables</Badge>
                  <ChevronDown className="size-4 ml-2" />
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="pt-3">
                <Card className="p-4 space-y-3">
                  <div className="grid grid-cols-4 gap-3 text-center">
                    <Stat label="Source" value={paper.Source} />
                    <Stat label="Tables" value={paper.Extracted_Tables.length} />
                    <Stat label="Total Rows" value={paper.Extracted_Tables.reduce((a, t) => a + t.data.length, 0)} />
                    <Stat label="Max Columns" value={Math.max(0, ...paper.Extracted_Tables.map(t => t.data[0]?.length || 0))} />
                  </div>

                  {paper.Extracted_Tables.length > 0 ? (
                    <Tabs defaultValue="0">
                      <TabsList>
                        {paper.Extracted_Tables.map((t, i) => (
                          <TabsTrigger key={i} value={String(i)}>{t.type} {i + 1}</TabsTrigger>
                        ))}
                      </TabsList>
                      {paper.Extracted_Tables.map((t, i) => (
                        <TabsContent key={i} value={String(i)} className="space-y-3">
                          <div className="rounded-md border max-h-96 overflow-auto">
                            <Table>
                              <TableHeader><TableRow>{t.data[0]?.map((h, j) => <TableHead key={j}>{h}</TableHead>)}</TableRow></TableHeader>
                              <TableBody>
                                {t.data.slice(1).map((row, ri) => (
                                  <TableRow key={ri}>{row.map((c, ci) => <TableCell key={ci}>{c}</TableCell>)}</TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </div>
                          {t.caption && <div className="text-xs italic text-muted-foreground">{t.caption}</div>}
                          <div className="grid grid-cols-2 gap-2">
                            <Button size="sm" variant="outline" onClick={() => exportTableCsv(paper, t, i)}><Download className="size-4 mr-2" />Export CSV</Button>
                            {paper.Paper_URL && <a href={paper.Paper_URL} target="_blank" rel="noreferrer" className="inline-flex items-center justify-center gap-2 text-sm bg-primary text-primary-foreground rounded-md py-2 hover:opacity-90"><ExternalLink className="size-4" />View Full Paper</a>}
                          </div>
                        </TabsContent>
                      ))}
                    </Tabs>
                  ) : <Alert><AlertDescription>No tables found in this paper.</AlertDescription></Alert>}
                </Card>
              </CollapsibleContent>
            </Collapsible>
          ))}

          <Button onClick={exportAll} className="w-full"><Download className="size-4 mr-2" />Export All Tables ({format.split(" ")[0]})</Button>
        </>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: any }) {
  return <div className="bg-muted/30 rounded p-2"><div className="font-bold">{value}</div><div className="text-xs text-muted-foreground">{label}</div></div>;
}
function exportTableCsv(paper: { Paper_Title: string }, t: { data: string[][] }, i: number) {
  const csv = t.data.map(row => row.map(c => `"${String(c).replace(/"/g, '""')}"`).join(",")).join("\n");
  download(`table_${i + 1}_${paper.Paper_Title.slice(0, 30).replace(/\s/g, "_")}.csv`, csv, "text/csv");
}
function download(name: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}
