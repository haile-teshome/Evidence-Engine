import { useEffect, useState } from "react";
import { useStore } from "../lib/store";
import { AIService, ScreenResult } from "../lib/mockServices";
import { effectiveFullTextDecision } from "../lib/exclusionBucketing";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Checkbox } from "../components/ui/checkbox";
import { Input } from "../components/ui/input";
import {
  Network, Plus, Minus, Trash2, Search, FileText, ArrowDownLeft, ArrowUpRight,
  CheckCircle2, XCircle, ExternalLink, Layers, ChevronDown, ChevronsDownUp, ChevronsUpDown,
} from "lucide-react";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";

type SnowballType = "Both" | "Backward (References)" | "Forward (Cited by)";

const DIRECTIONS: { value: SnowballType; label: string }[] = [
  { value: "Both", label: "Both" },
  { value: "Backward (References)", label: "Backward" },
  { value: "Forward (Cited by)", label: "Forward" },
];

export function SnowballPage() {
  const s = useStore();
  const [type, setType] = useState<SnowballType>("Both");
  const [maxCit, setMaxCit] = useState(20);
  // null = "All seeds"; otherwise the seed paper title whose citations are shown.
  const [selectedSeed, setSelectedSeed] = useState<string | null>(null);
  const [q, setQ] = useState("");
  // Citation ids the reviewer has chosen to carry into the main results.
  const [chosen, setChosen] = useState<Set<string>>(new Set());
  const fetchTask = s.tasks["snowball"];
  const screenTask = s.tasks["snowball-screen"];
  const running = fetchTask?.status === "running";
  const screening = screenTask?.status === "running";

  // Seed the selection from the AI's INCLUDE verdicts whenever a fresh
  // screening run lands; the reviewer can then check/uncheck from there.
  useEffect(() => {
    const sc = s.snowballScreened, rs = s.snowballResults;
    if (!sc || !rs) return;
    const dmap = new Map<string, string>();
    sc.forEach(r => {
      if (r.paper_id) dmap.set(r.paper_id, r.Decision);
      if (r.Title) dmap.set(r.Title.toLowerCase().trim(), r.Decision);
    });
    const init = new Set<string>();
    rs.forEach(p => {
      const d = dmap.get(p.id) || dmap.get((p.title || "").toLowerCase().trim());
      if (d === "INCLUDE") init.add(p.id);
    });
    setChosen(init);
  }, [s.snowballScreened]);

  function toggleChosen(id: string) {
    setChosen(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  // Which citation abstracts are expanded. Default to all-expanded whenever a
  // fresh result set lands; the reviewer can collapse individually or all at once.
  const [absOpen, setAbsOpen] = useState<Set<string>>(new Set());
  useEffect(() => {
    if (s.snowballResults) setAbsOpen(new Set(s.snowballResults.map((p: any) => p.id)));
  }, [s.snowballResults]);
  function toggleAbs(id: string) {
    setAbsOpen(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  if (!s.fullTextResults) {
    return <Alert><AlertDescription>Complete Full-Text Evidence screening first to unlock Citation Snowballing.</AlertDescription></Alert>;
  }
  // Honour reviewer overrides — papers the user kept by checking the
  // full-text Keep box should seed snowballing even if the AI excluded them.
  const seeds = s.fullTextResults.filter(r => effectiveFullTextDecision(r, s.fullTextOverrides) === "Include");
  if (seeds.length === 0) return <Alert><AlertDescription>No papers passed full-text screening. Cannot perform snowballing.</AlertDescription></Alert>;

  async function start() {
    const { abort } = s.startTask("snowball", [{ id: "snow", label: "Fetching citations", status: "running" }]);
    s.updateTask("snowball", { progress: { done: 0, total: seeds.length } });
    const signal = abort.signal;
    try {
      const all: any[] = [];
      for (let i = 0; i < seeds.length; i++) {
        if (signal.aborted) break;
        const seed = seeds[i];
        s.updateTask("snowball", {
          progress: { done: i, total: seeds.length, label: seed.Title.slice(0, 80) },
          detail: seed.Title.slice(0, 80),
        });
        try {
          const cits = await AIService.fetchCitations(seed.Title, type, maxCit, s.sources, signal);
          cits.forEach(c => { c.seed_paper_title = seed.Title; });
          all.push(...cits);
        } catch (e: any) {
          if (signal.aborted) break;
          console.error(`snowball ${i + 1} failed:`, e?.message);
        }
        s.updateTask("snowball", { progress: { done: i + 1, total: seeds.length } });
      }
      const seenT = new Set<string>();
      const unique = all.filter(p => seenT.has((p.title || "").toLowerCase()) ? false : (seenT.add((p.title || "").toLowerCase()), true));
      s.setSnowballResults(unique);
      if (signal.aborted) {
        s.updateTask("snowball", { status: "canceled" });
        toast.info(`Canceled — ${unique.length} unique citations gathered`);
      } else {
        s.updateTask("snowball", { status: "done" });
        toast.success(`Found ${unique.length} unique papers via snowballing`);
      }
    } catch (e: any) {
      s.updateTask("snowball", { status: "error", detail: e?.message });
    }
  }

  async function screen() {
    if (!s.snowballResults) return;
    const { abort } = s.startTask("snowball-screen", [{ id: "scr", label: "Screening snowballed", status: "running" }]);
    s.updateTask("snowball-screen", { progress: { done: 0, total: s.snowballResults.length } });
    const signal = abort.signal;
    try {
      const out: ScreenResult[] = [];
      for (let i = 0; i < s.snowballResults.length; i++) {
        if (signal.aborted) break;
        const p = s.snowballResults[i];
        s.updateTask("snowball-screen", {
          progress: { done: i, total: s.snowballResults.length, label: (p.title || "").slice(0, 80) },
          detail: (p.title || "").slice(0, 80),
        });
        try {
          const r = await AIService.screenPaperMultiAgent(
            { id: p.id, source: p.source, title: p.title, abstract: p.abstract, url: p.url },
            s.pico, s.inclusion, s.exclusion, signal,
          );
          out.push(r);
        } catch (e: any) {
          if (signal.aborted) break;
          console.error(`snowball-screen ${i + 1} failed:`, e?.message);
        }
        s.updateTask("snowball-screen", { progress: { done: i + 1, total: s.snowballResults.length } });
      }
      s.setSnowballScreened(out);
      if (signal.aborted) {
        s.updateTask("snowball-screen", { status: "canceled" });
        toast.info(`Canceled — ${out.length} of ${s.snowballResults.length} screened`);
      } else {
        s.updateTask("snowball-screen", { status: "done" });
        toast.success(`Screened ${out.length} snowballed papers`);
      }
    } catch (e: any) {
      s.updateTask("snowball-screen", { status: "error", detail: e?.message });
    }
  }

  function addToMain() {
    if (!s.snowballResults || !s.results) return;
    // Map any AI verdicts by id and title so we can recover the full ScreenResult
    // for a chosen citation regardless of which key the row used.
    const byKey = new Map<string, ScreenResult>();
    (s.snowballScreened || []).forEach(r => {
      if (r.paper_id) byKey.set(r.paper_id, r);
      if (r.Title) byKey.set(r.Title.toLowerCase().trim(), r);
    });
    const toAdd: ScreenResult[] = [];
    for (const p of s.snowballResults) {
      if (!chosen.has(p.id)) continue;
      const r = byKey.get(p.id) || byKey.get((p.title || "").toLowerCase().trim());
      // The reviewer chose it, so it carries forward as an INCLUDE — even if the
      // AI excluded it, or it was never AI-screened at all.
      if (r) {
        toAdd.push({ ...r, Decision: "INCLUDE" });
      } else {
        toAdd.push({
          paper_id: p.id, Source: p.source, Title: p.title, URL: p.url, Abstract: p.abstract || "",
          Decision: "INCLUDE", Reason: "Manually re-included from citation snowballing.",
          Agent_Trace: {},
        });
      }
    }
    if (toAdd.length === 0) { toast.error("No papers selected to add."); return; }
    s.setResults([...s.results, ...toAdd]);
    s.setPrisma(p => ({ ...p, identified: p.identified + (s.snowballResults?.length || 0), included_final: p.included_final + toAdd.length }));
    toast.success(`Added ${toAdd.length} paper${toAdd.length === 1 ? "" : "s"} to main results`);
  }
  function clearAll() {
    s.setSnowballResults(null); s.setSnowballScreened(null);
  }

  const results = s.snowballResults;
  const screened = s.snowballScreened;

  // Decision lookup keyed by both id and normalised title, so a citation row
  // can show its INCLUDE/EXCLUDE verdict regardless of which the backend echoes.
  const decisionByKey = new Map<string, ScreenResult>();
  (screened || []).forEach(r => {
    if (r.paper_id) decisionByKey.set(r.paper_id, r);
    if (r.Title) decisionByKey.set(r.Title.toLowerCase().trim(), r);
  });
  const verdict = (p: any): ScreenResult | undefined =>
    decisionByKey.get(p.id) || decisionByKey.get((p.title || "").toLowerCase().trim());

  const backCount = results?.filter(p => p.citation_type === "backward").length ?? 0;
  const fwdCount = results?.filter(p => p.citation_type === "forward").length ?? 0;
  const includedCount = screened?.filter(r => r.Decision === "INCLUDE").length ?? 0;
  const excludedCount = (screened?.length ?? 0) - includedCount;

  // Group citations by their seed paper, preserving first-seen order.
  const seedGroups: { seed: string; items: any[] }[] = [];
  if (results) {
    const idx = new Map<string, number>();
    for (const p of results) {
      const seed = p.seed_paper_title || "(unknown seed)";
      if (!idx.has(seed)) { idx.set(seed, seedGroups.length); seedGroups.push({ seed, items: [] }); }
      seedGroups[idx.get(seed)!].items.push(p);
    }
  }

  const shown = (results || [])
    .filter(p => selectedSeed === null || p.seed_paper_title === selectedSeed)
    .filter(p => !q.trim() || (p.title || "").toLowerCase().includes(q.toLowerCase()));

  return (
    <div className="space-y-3">
      {/* ── Compact header: stats + controls + run ─────────────────────────── */}
      <Card className="p-3">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="mr-auto min-w-0">
            <div className="flex flex-wrap items-center gap-1.5">
              <Pill icon={FileText} title="Seed papers that passed full-text screening">{seeds.length} seeds</Pill>
              {results && <Pill icon={Network}>{results.length} found</Pill>}
              {results && <Pill icon={ArrowDownLeft} title="Backward — references cited by the seeds">{backCount} backward</Pill>}
              {results && <Pill icon={ArrowUpRight} title="Forward — papers that cite the seeds">{fwdCount} forward</Pill>}
              {screened && <Pill icon={CheckCircle2} tone="green" title="AI marked INCLUDE">{includedCount} AI-included</Pill>}
              {screened && <Pill icon={XCircle} tone="amber" title="AI marked EXCLUDE">{excludedCount} AI-excluded</Pill>}
              {results && chosen.size > 0 && <Pill icon={CheckCircle2} tone="green" title="Selected to carry into main results">{chosen.size} selected</Pill>}
            </div>
          </div>
          <div className="flex items-center gap-2.5 h-10 rounded-lg border bg-muted/30 px-3">
            <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Direction</span>
            <div className="inline-flex h-8 rounded-md border bg-background p-0.5">
              {DIRECTIONS.map(d => (
                <button
                  key={d.value}
                  onClick={() => setType(d.value)}
                  className={`px-2.5 text-sm rounded-[5px] transition-colors ${type === d.value ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
                >
                  {d.label}
                </button>
              ))}
            </div>
            <div className="h-5 w-px bg-border mx-1" />
            <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Max per paper</span>
            <div className="inline-flex h-8 items-center rounded-md border bg-background">
              <button
                onClick={() => setMaxCit(m => Math.max(5, m - 5))}
                className="px-2 h-full text-muted-foreground hover:text-foreground disabled:opacity-40"
                disabled={maxCit <= 5}
                aria-label="Decrease"
              >
                <Minus className="size-3.5" />
              </button>
              <input
                type="number"
                value={maxCit}
                min={1}
                max={500}
                onChange={e => {
                  const n = parseInt(e.target.value, 10);
                  setMaxCit(Number.isNaN(n) ? 1 : Math.min(500, Math.max(1, n)));
                }}
                className="w-10 h-full text-center text-sm font-semibold tabular-nums bg-transparent border-x outline-none [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
              />
              <button
                onClick={() => setMaxCit(m => Math.min(500, m + 5))}
                className="px-2 h-full text-muted-foreground hover:text-foreground disabled:opacity-40"
                disabled={maxCit >= 500}
                aria-label="Increase"
              >
                <Plus className="size-3.5" />
              </button>
            </div>
          </div>
          <Button onClick={start} disabled={running} className="h-10">
            <Network className="size-4 mr-2" />{running ? "Fetching..." : results ? "Re-run" : "Start Snowballing"}
          </Button>
        </div>
      </Card>

      {fetchTask && fetchTask.status === "running" && (
        <TaskProgressCard task={fetchTask} title="Fetching citations" onCancel={() => s.cancelTask("snowball")} />
      )}
      {screenTask && screenTask.status === "running" && (
        <TaskProgressCard task={screenTask} title="Screening snowballed papers" onCancel={() => s.cancelTask("snowball-screen")} />
      )}

      {results && (
        <>
          {/* ── Two-pane: seed papers (left) + citations they surfaced (right) ─ */}
          <div className="flex gap-4 h-[calc(100vh-16rem)] min-h-[28rem]">
            {/* LEFT: seed papers */}
            <Card className="w-80 shrink-0 p-0 overflow-hidden flex flex-col">
              <div className="px-3 py-2 border-b text-xs uppercase tracking-wide text-muted-foreground">
                Seed papers
              </div>
              <div className="overflow-auto flex-1">
                <button
                  onClick={() => setSelectedSeed(null)}
                  className={`w-full text-left px-3 py-2.5 border-b hover:bg-muted/50 transition-colors ${selectedSeed === null ? "bg-primary/10 border-l-2 border-l-primary" : "border-l-2 border-l-transparent"}`}
                >
                  <div className="flex items-center gap-2">
                    <Layers className="size-4 text-muted-foreground shrink-0" />
                    <span className="text-sm font-medium">All citations</span>
                    <Badge variant="secondary" className="ml-auto text-[10px]">{results.length}</Badge>
                  </div>
                </button>
                {seedGroups.map(({ seed, items }) => {
                  const active = selectedSeed === seed;
                  const b = items.filter(p => p.citation_type === "backward").length;
                  const f = items.filter(p => p.citation_type === "forward").length;
                  return (
                    <button
                      key={seed}
                      onClick={() => setSelectedSeed(seed)}
                      className={`w-full text-left px-3 py-2.5 border-b hover:bg-muted/50 transition-colors ${active ? "bg-primary/10 border-l-2 border-l-primary" : "border-l-2 border-l-transparent"}`}
                    >
                      <div className="text-sm leading-snug line-clamp-2 max-h-[2.75em] overflow-hidden mb-1">{seed}</div>
                      <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                        <Badge variant="outline" className="text-[10px]">{items.length}</Badge>
                        {b > 0 && <span className="flex items-center gap-0.5"><ArrowDownLeft className="size-3" />{b}</span>}
                        {f > 0 && <span className="flex items-center gap-0.5"><ArrowUpRight className="size-3" />{f}</span>}
                      </div>
                    </button>
                  );
                })}
              </div>
            </Card>

            {/* RIGHT: citations surfaced by the selected seed */}
            <Card className="flex-1 min-w-0 p-0 overflow-hidden flex flex-col">
              <div className="p-2 border-b flex items-center gap-3">
                <label className="flex items-center gap-1.5 text-xs text-muted-foreground shrink-0 cursor-pointer pl-1" title="Select / deselect all shown">
                  <Checkbox
                    checked={shown.length > 0 && shown.every(p => chosen.has(p.id))}
                    onCheckedChange={(c) => setChosen(prev => {
                      const next = new Set(prev);
                      shown.forEach(p => { if (c) next.add(p.id); else next.delete(p.id); });
                      return next;
                    })}
                  />
                  Select all
                </label>
                <div className="relative flex-1">
                  <Search className="size-3.5 absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground" />
                  <Input value={q} onChange={e => setQ(e.target.value)} placeholder={`Filter ${shown.length} citations…`} className="pl-7 h-8 text-sm" />
                </div>
                {(() => {
                  const withAbs = shown.filter(p => p.abstract);
                  const allOpen = withAbs.length > 0 && withAbs.every(p => absOpen.has(p.id));
                  return (
                    <Button
                      size="sm"
                      variant="ghost"
                      disabled={withAbs.length === 0}
                      onClick={() => setAbsOpen(prev => {
                        const next = new Set(prev);
                        withAbs.forEach(p => { if (allOpen) next.delete(p.id); else next.add(p.id); });
                        return next;
                      })}
                      className="h-8 text-xs shrink-0 gap-1"
                      title={allOpen ? "Collapse all abstracts" : "Expand all abstracts"}
                    >
                      {allOpen ? <ChevronsDownUp className="size-3.5" /> : <ChevronsUpDown className="size-3.5" />}
                      {allOpen ? "Collapse all" : "Expand all"}
                    </Button>
                  );
                })()}
              </div>
              <div className="overflow-auto flex-1 divide-y">
                {shown.map((p, i) => {
                  const v = verdict(p);
                  const picked = chosen.has(p.id);
                  return (
                    <div key={p.id || i} className={`px-3 py-2.5 ${picked ? "bg-primary/5" : ""}`}>
                      <div className="flex items-start gap-2">
                        <Checkbox
                          checked={picked}
                          onCheckedChange={() => toggleChosen(p.id)}
                          className="mt-0.5 shrink-0"
                          title={picked ? "Selected — will carry into main results" : "Click to include in main results"}
                        />
                        <Badge variant="outline" className="text-[10px] shrink-0 mt-0.5 gap-1">
                          {p.citation_type === "backward" ? <ArrowDownLeft className="size-3" /> : <ArrowUpRight className="size-3" />}
                          {p.citation_type}
                        </Badge>
                        <div className="flex-1 min-w-0">
                          <a href={p.url} target="_blank" rel="noreferrer" className="text-sm leading-snug hover:underline inline-flex items-start gap-1">
                            <span>{p.title}</span>
                            {p.url && <ExternalLink className="size-3 mt-0.5 shrink-0 text-muted-foreground" />}
                          </a>
                          <div className="flex items-center gap-2 mt-1 text-[10px] text-muted-foreground">
                            <span>{p.source}</span>
                            {selectedSeed === null && p.seed_paper_title && (
                              <span className="truncate">· via {p.seed_paper_title}</span>
                            )}
                            {!p.abstract && <span className="text-amber-600">· no abstract</span>}
                          </div>
                          {p.abstract && (() => {
                            const open = absOpen.has(p.id);
                            return (
                              <div className="mt-1.5">
                                <button
                                  onClick={() => toggleAbs(p.id)}
                                  className="inline-flex items-center gap-1 text-[11px] font-medium text-muted-foreground hover:text-foreground"
                                >
                                  <ChevronDown className={`size-3 transition-transform ${open ? "" : "-rotate-90"}`} />
                                  Abstract
                                </button>
                                {open && (
                                  <div className="mt-1 rounded-md border bg-muted/30 p-2.5 text-xs leading-relaxed text-foreground/80 max-h-60 overflow-auto whitespace-pre-wrap">
                                    {p.abstract}
                                  </div>
                                )}
                              </div>
                            );
                          })()}
                          {v && (
                            <div className="text-xs text-foreground/70 mt-1.5 line-clamp-2"><span className="font-medium">Screening:</span> {v.Reason}</div>
                          )}
                        </div>
                        {v && (
                          <Badge
                            className={`text-[10px] shrink-0 gap-1 ${v.Decision === "INCLUDE" ? "bg-emerald-50 text-emerald-700 border-emerald-200" : "bg-amber-50 text-amber-700 border-amber-200"}`}
                            variant="outline"
                          >
                            {v.Decision === "INCLUDE" ? <CheckCircle2 className="size-3" /> : <XCircle className="size-3" />}
                            {v.Decision === "INCLUDE" ? "Include" : "Exclude"}
                          </Badge>
                        )}
                      </div>
                    </div>
                  );
                })}
                {shown.length === 0 && (
                  <div className="p-4 text-sm text-muted-foreground">No citations match “{q}”.</div>
                )}
              </div>
            </Card>
          </div>

          {/* ── Footer actions ────────────────────────────────────────────── */}
          <div className="grid grid-cols-3 gap-2">
            <Button variant="outline" onClick={screen} disabled={screening}>
              <Network className="size-4 mr-2" />{screening ? "Screening..." : screened ? "Re-screen with AI" : "AI-screen for me"}
            </Button>
            <Button onClick={addToMain} disabled={chosen.size === 0}>
              <Plus className="size-4 mr-2" />Add {chosen.size} to Main Results
            </Button>
            <Button variant="outline" onClick={clearAll}><Trash2 className="size-4 mr-2" />Clear Results</Button>
          </div>
        </>
      )}
    </div>
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
