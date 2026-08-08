import { useStore } from "../lib/store";
import { SearchLogEntry } from "../lib/apiClient";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Download, ClipboardList, Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";

function dl(name: string, content: string, mime: string) {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

// PRISMA-S search audit log: a timestamped record of each search run, with the
// exact query and number of records retrieved per database. Entries are
// captured automatically when screening runs, and can also be recorded manually
// from the current planned search.
export function SearchLog() {
  const s = useStore();
  const log = s.searchLog;

  function recordCurrent() {
    const rows = (s.sources || []).map(src => ({
      source: src,
      query: s.perDbQueries[src] || s.unifiedSearchQuery || s.query || "",
      count: s.dbTestResults?.[src]?.total_found ?? 0,
    })).filter(r => r.query);
    if (!rows.length) { toast.error("No per-database queries yet. Design the search first."); return; }
    s.setSearchLog(prev => [...prev, { id: Date.now().toString(36), ranAt: new Date().toISOString(), label: "Planned search", rows }]);
    toast.success("Recorded current search");
  }

  function csv() {
    if (!log.length) { toast.error("No searches logged yet."); return; }
    const q = (x: any) => `"${String(x ?? "").replace(/"/g, '""')}"`;
    const head = ["Date searched", "Database", "Query", "Records retrieved"];
    const rows: string[][] = [];
    for (const e of log) for (const r of e.rows) rows.push([e.ranAt, r.source, r.query, String(r.count)]);
    dl("search-log-prisma-s.csv", [head, ...rows].map(r => r.map(q).join(",")).join("\r\n"), "text/csv");
  }

  function html() {
    if (!log.length) { toast.error("No searches logged yet."); return; }
    const e = (x: any) => String(x ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const blocks = log.map(en => {
      const rows = en.rows.map(r => `<tr><td>${e(r.source)}</td><td><code>${e(r.query)}</code></td><td>${r.count}</td></tr>`).join("");
      return `<h2>${e(en.label || "Search")} — ${e(en.ranAt)}</h2>
<table><thead><tr><th>Database</th><th>Query as run</th><th>Records</th></tr></thead><tbody>${rows}</tbody></table>`;
    }).join("");
    dl("search-log-prisma-s.html", `<!doctype html><html><head><meta charset="utf-8"><title>Search log</title>
<style>body{font-family:system-ui,sans-serif;margin:2rem;color:#0f172a}h1{font-size:1.3rem}h2{font-size:1rem;margin-top:1.4rem}table{border-collapse:collapse;width:100%;font-size:13px;margin:.5rem 0}th,td{border:1px solid #cbd5e1;padding:6px 9px;text-align:left;vertical-align:top}th{background:#f1f5f9}code{font-size:12px;white-space:pre-wrap}</style>
</head><body><h1>Search log (PRISMA-S)</h1>${blocks}</body></html>`, "text/html");
  }

  return (
    <Card className="p-3 space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 text-sm font-medium"><ClipboardList className="size-4 text-primary" />Search log (PRISMA-S)</div>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={recordCurrent}><Plus className="size-3.5 mr-1.5" />Record current search</Button>
          <Button size="sm" variant="outline" onClick={csv} disabled={!log.length}><Download className="size-3.5 mr-1.5" />CSV</Button>
          <Button size="sm" variant="outline" onClick={html} disabled={!log.length}><Download className="size-3.5 mr-1.5" />HTML</Button>
        </div>
      </div>
      <p className="text-xs text-muted-foreground">
        Each database, the exact query as run, the date, and the number of records retrieved. Captured automatically when you screen, so the search is reproducible.
      </p>
      {log.length === 0 ? (
        <div className="text-xs text-muted-foreground py-3">No searches logged yet. Design a search and screen, or record the current one.</div>
      ) : (
        <div className="space-y-3">
          {[...log].reverse().map((en: SearchLogEntry) => (
            <div key={en.id} className="rounded-md border p-2.5 space-y-1.5">
              <div className="flex items-center justify-between">
                <div className="text-xs font-medium">{en.label || "Search"} <span className="text-muted-foreground font-normal">· {new Date(en.ranAt).toLocaleString()}</span></div>
                <div className="flex items-center gap-2">
                  <Badge variant="secondary">{en.rows.reduce((a, r) => a + r.count, 0)} records</Badge>
                  <button onClick={() => s.setSearchLog(prev => prev.filter(x => x.id !== en.id))} className="text-muted-foreground hover:text-destructive"><Trash2 className="size-3.5" /></button>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead><tr className="text-left text-muted-foreground border-b"><th className="py-1 pr-3 font-medium">Database</th><th className="py-1 pr-3 font-medium">Query</th><th className="py-1 font-medium text-right">Records</th></tr></thead>
                  <tbody>
                    {en.rows.map((r, i) => (
                      <tr key={i} className="border-b last:border-0 align-top">
                        <td className="py-1 pr-3 whitespace-nowrap">{r.source}</td>
                        <td className="py-1 pr-3 font-mono text-[11px] break-all">{r.query || <span className="text-muted-foreground italic">not recorded</span>}</td>
                        <td className="py-1 text-right tabular-nums">{r.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
