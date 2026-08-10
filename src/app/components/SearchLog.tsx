import { useState } from "react";
import { useStore } from "../lib/store";
import { SearchLogEntry } from "../lib/apiClient";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Download, ClipboardList, Plus, Trash2, X } from "lucide-react";
import { toast } from "sonner";

function dl(name: string, content: string, mime: string) {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

const DEFAULT_LOG = "Log 1";
const groupOf = (e: SearchLogEntry) => e.group || DEFAULT_LOG;

// PRISMA-S search log. Entries are grouped into named logs shown as tabs, so a
// reviewer can keep several separate logs (e.g. per search round or database
// set). Entries are captured automatically when screening runs, and can also be
// saved manually from the current planned search.
export function SearchLog({ bare = false }: { bare?: boolean }) {
  const s = useStore();
  const log = s.searchLog;
  const Wrap: any = bare ? "div" : Card;

  const activeSources = (s.sources || []).filter(x => x !== "Local PDFs");

  // Log tabs: derived from entries, plus any empty logs created in this session.
  const [extraLogs, setExtraLogs] = useState<string[]>([]);
  const [active, setActive] = useState<string>("");
  const [editing, setEditing] = useState<string | null>(null);
  const [editName, setEditName] = useState("");

  const entryGroups = Array.from(new Set(log.map(groupOf)));
  const logNames = Array.from(new Set([...(entryGroups.length ? entryGroups : [DEFAULT_LOG]), ...extraLogs]));
  const activeLog = logNames.includes(active) ? active : logNames[0];
  const activeEntries = log.filter(e => groupOf(e) === activeLog);

  function addLog() {
    let n = 1; let name = `Log ${logNames.length + 1}`;
    while (logNames.includes(name)) name = `Log ${logNames.length + 1 + n++}`;
    setExtraLogs(prev => [...prev, name]);
    setActive(name);
  }
  function deleteLog(name: string) {
    s.setSearchLog(prev => prev.filter(e => groupOf(e) !== name));
    setExtraLogs(prev => prev.filter(x => x !== name));
    if (active === name) setActive("");
  }
  function commitRename(oldName: string) {
    const newName = editName.trim();
    setEditing(null);
    if (!newName || newName === oldName || logNames.includes(newName)) return;
    s.setSearchLog(prev => prev.map(e => (groupOf(e) === oldName ? { ...e, group: newName } : e)));
    setExtraLogs(prev => prev.map(x => (x === oldName ? newName : x)));
    if (active === oldName || activeLog === oldName) setActive(newName);
  }

  // Save one database's current query into the active log.
  function recordOne(src: string) {
    const query = s.perDbQueries[src] || s.unifiedSearchQuery || s.query || "";
    if (!query) { toast.error(`No query for ${src} yet. Design the search first.`); return; }
    const rows = [{ source: src, query, count: s.dbTestResults?.[src]?.total_found ?? 0 }];
    s.setSearchLog(prev => [...prev, { id: Date.now().toString(36), ranAt: new Date().toISOString(), label: src, group: activeLog, rows }]);
    toast.success(`Saved ${src} query to ${activeLog}`);
  }
  // Save every database's current query as one entry in the active log.
  function recordCurrent() {
    const rows = activeSources.map(src => ({
      source: src,
      query: s.perDbQueries[src] || s.unifiedSearchQuery || s.query || "",
      count: s.dbTestResults?.[src]?.total_found ?? 0,
    })).filter(r => r.query);
    if (!rows.length) { toast.error("No per-database queries yet. Design the search first."); return; }
    s.setSearchLog(prev => [...prev, { id: Date.now().toString(36), ranAt: new Date().toISOString(), label: "All databases", group: activeLog, rows }]);
    toast.success(`Saved all database queries to ${activeLog}`);
  }

  function csv() {
    if (!activeEntries.length) { toast.error("Nothing saved in this log yet."); return; }
    const q = (x: any) => `"${String(x ?? "").replace(/"/g, '""')}"`;
    const head = ["Date searched", "Database", "Query", "Records retrieved"];
    const rows: string[][] = [];
    for (const e of activeEntries) for (const r of e.rows) rows.push([e.ranAt, r.source, r.query, String(r.count)]);
    dl(`search-log-${activeLog.replace(/\s+/g, "-").toLowerCase()}.csv`, [head, ...rows].map(r => r.map(q).join(",")).join("\r\n"), "text/csv");
  }
  function html() {
    if (!activeEntries.length) { toast.error("Nothing saved in this log yet."); return; }
    const e = (x: any) => String(x ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const blocks = activeEntries.map(en => {
      const rows = en.rows.map(r => `<tr><td>${e(r.source)}</td><td><code>${e(r.query)}</code></td><td>${r.count}</td></tr>`).join("");
      return `<h2>${e(en.label || "Search")} — ${e(en.ranAt)}</h2>
<table><thead><tr><th>Database</th><th>Query as run</th><th>Records</th></tr></thead><tbody>${rows}</tbody></table>`;
    }).join("");
    dl(`search-log-${activeLog.replace(/\s+/g, "-").toLowerCase()}.html`, `<!doctype html><html><head><meta charset="utf-8"><title>${e(activeLog)}</title>
<style>body{font-family:system-ui,sans-serif;margin:2rem;color:#0f172a}h1{font-size:1.3rem}h2{font-size:1rem;margin-top:1.4rem}table{border-collapse:collapse;width:100%;font-size:13px;margin:.5rem 0}th,td{border:1px solid #cbd5e1;padding:6px 9px;text-align:left;vertical-align:top}th{background:#f1f5f9}code{font-size:12px;white-space:pre-wrap}</style>
</head><body><h1>${e(activeLog)} (PRISMA-S)</h1>${blocks}</body></html>`, "text/html");
  }

  return (
    <Wrap className={bare ? "space-y-3" : "p-3 space-y-3"}>
      {!bare && <div className="flex items-center gap-1.5 text-sm font-medium"><ClipboardList className="size-4 text-primary" />Search log</div>}
      <p className="text-xs text-muted-foreground">
        A reproducible record of each search: the database, the exact query, the date, and how many records it returned. Keep separate logs as tabs.
      </p>

      {/* Log tabs */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Logs</span>
        {logNames.map(name => {
          const count = log.filter(en => groupOf(en) === name).length;
          const isActive = name === activeLog;
          if (editing === name) {
            return (
              <input key={name} autoFocus value={editName}
                onChange={e => setEditName(e.target.value)}
                onBlur={() => commitRename(name)}
                onKeyDown={e => { if (e.key === "Enter") commitRename(name); if (e.key === "Escape") setEditing(null); }}
                className="h-7 w-28 rounded-md border px-2 text-xs outline-none focus:ring-2 focus:ring-ring/40" />
            );
          }
          return (
            <button key={name} onClick={() => setActive(name)} onDoubleClick={() => { setEditing(name); setEditName(name); }}
              title="Click to open · double-click to rename"
              className={`group/tab inline-flex items-center gap-1.5 rounded-md h-7 px-2.5 text-xs border transition-colors ${isActive ? "bg-primary text-primary-foreground border-primary" : "hover:bg-muted"}`}>
              {name}{count > 0 && <span className={`tabular-nums ${isActive ? "opacity-80" : "text-muted-foreground"}`}>{count}</span>}
              {logNames.length > 1 && (
                <span onClick={e => { e.stopPropagation(); deleteLog(name); }}
                  className={`grid place-items-center rounded ${isActive ? "hover:bg-white/20" : "hover:bg-muted-foreground/20"} opacity-0 group-hover/tab:opacity-70`}>
                  <X className="size-3" />
                </span>
              )}
            </button>
          );
        })}
        <button onClick={addLog} title="New log"
          className="inline-flex items-center rounded-md h-7 px-2 text-xs border border-dashed text-muted-foreground hover:bg-muted transition-colors">
          <Plus className="size-3.5" />
        </button>
      </div>

      {/* Save a database's current query into the active log — pick one, or all. */}
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-xs text-muted-foreground mr-0.5">Save to {activeLog}:</span>
        {activeSources.map(src => {
          const q = s.perDbQueries[src] || s.unifiedSearchQuery || s.query || "";
          const count = s.dbTestResults?.[src]?.total_found;
          return (
            <button key={src} onClick={() => recordOne(src)} disabled={!q}
              title={q ? `Save the ${src} query` : "No query for this database yet"}
              className="inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs hover:bg-muted disabled:opacity-40 transition-colors">
              {src}{count != null && <span className="text-muted-foreground tabular-nums">· {count.toLocaleString()}</span>}<Plus className="size-3" />
            </button>
          );
        })}
        {activeSources.length > 1 && (
          <button onClick={recordCurrent}
            className="inline-flex items-center gap-1 rounded-full border border-primary/40 px-2.5 py-1 text-xs font-medium text-primary hover:bg-primary/10 transition-colors">
            <Plus className="size-3" />All
          </button>
        )}
      </div>

      {/* Active log's entries + export */}
      <div className="flex items-center justify-between gap-2 pt-1 border-t">
        <span className="text-xs font-medium text-muted-foreground pt-2">{activeEntries.length ? `${activeEntries.length} saved in ${activeLog}` : "Nothing saved yet"}</span>
        <div className="flex items-center gap-1.5 pt-2">
          <Button size="sm" variant="outline" className="h-7" onClick={csv} disabled={!activeEntries.length}><Download className="size-3.5 mr-1.5" />CSV</Button>
          <Button size="sm" variant="outline" className="h-7" onClick={html} disabled={!activeEntries.length}><Download className="size-3.5 mr-1.5" />HTML</Button>
        </div>
      </div>

      {activeEntries.length > 0 && (
        <div className="space-y-3">
          {[...activeEntries].reverse().map((en: SearchLogEntry) => (
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
    </Wrap>
  );
}
