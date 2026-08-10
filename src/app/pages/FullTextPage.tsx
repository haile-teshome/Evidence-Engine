import { Fragment, useState, useMemo, useEffect } from "react";
import { useStore } from "../lib/store";
import { AIService, formatDuration, FullTextResult } from "../lib/mockServices";
import { RapidScreen } from "../components/RapidScreen";
import {
  categoriseFullTextExclusion,
  effectiveAbstractDecision,
  effectiveFullTextDecision,
} from "../lib/exclusionBucketing";
import { Checkbox } from "../components/ui/checkbox";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { PopoverContent, PopoverTrigger, PopoverClose, ScrollAwarePopover } from "../components/ui/popover";
import { FlaskConical, Check, Minus, X as XIcon, Download, Zap, FileSearch, ChevronRight, ChevronDown, Maximize2, Minimize2, FileText, CheckCircle2, XCircle, Clock, Sparkles, GripVertical } from "lucide-react";
import { ControlPane, InlineStat, PaneDivider } from "../components/ControlPane";
import { EmptyState } from "../components/EmptyState";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";

// PICO match → pill styling for the expanded evidence boxes.
const matchTone = (m?: string) =>
  m === "yes" ? "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-300 dark:border-emerald-900"
    : m === "partial" ? "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/40 dark:text-amber-300 dark:border-amber-900"
      : "bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-950/40 dark:text-rose-300 dark:border-rose-900";
const matchLabel = (m?: string) => m === "yes" ? "Match" : m === "partial" ? "Partial" : "No match";

export function FullTextPage() {
  const s = useStore();
  const task = s.tasks["full-text-screen"];
  const running = task?.status === "running";
  const [rapidOpen, setRapidOpen] = useState(false);
  const [maxOpen, setMaxOpen] = useState(false);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const toggleRow = (id: string) => setExpanded(prev => {
    const n = new Set(prev);
    n.has(id) ? n.delete(id) : n.add(id);
    return n;
  });

  // Reorderable (non-frozen) columns: PICO + each criterion + Reason. Keep/
  // Decision/Title stay pinned. Order persists and reconciles with the current
  // criteria set (new criteria append, removed ones drop out). These hooks run
  // before any early return so hook order stays stable.
  const movableIds = ["population", "intervention", "comparator", "outcome",
    ...s.inclusion.map(c => `c:${c}`), ...s.exclusion.map(c => `c:${c}`), "reason"];
  const [rawColOrder, setRawColOrder] = useState<string[]>(() => {
    try { const saved = JSON.parse(localStorage.getItem("ee:fulltext-col-order") || "null"); return Array.isArray(saved) ? saved : []; } catch { return []; }
  });
  const colOrder = useMemo(() => {
    const kept = rawColOrder.filter(id => movableIds.includes(id));
    const added = movableIds.filter(id => !kept.includes(id));
    return [...kept, ...added];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rawColOrder, movableIds.join("|")]);
  useEffect(() => { try { localStorage.setItem("ee:fulltext-col-order", JSON.stringify(colOrder)); } catch { /* ignore */ } }, [colOrder]);
  const [dragCol, setDragCol] = useState<string | null>(null);
  const [overCol, setOverCol] = useState<string | null>(null);
  const moveCol = (from: string, to: string) => setRawColOrder(() => {
    const a = [...colOrder]; const fi = a.indexOf(from), ti = a.indexOf(to);
    if (fi < 0 || ti < 0 || fi === ti) return colOrder;
    a.splice(fi, 1); a.splice(ti, 0, from); return a;
  });

  if (!s.results) return <EmptyState icon={FileSearch} title="No screening results yet" description="Run abstract screening first to unlock per-criterion full-text evaluation." action={{ label: "Go to Abstract Screening", onClick: () => s.setPage("abstract"), icon: FileSearch }} />;

  // The queue uses EFFECTIVE abstract decisions, so a paper the reviewer
  // rescued at the abstract stage (by checking its Keep box) now joins the
  // full-text queue, and one they dropped is removed.
  const passed = s.results.filter(r => effectiveAbstractDecision(r, s.abstractOverrides) === "INCLUDE");
  if (passed.length === 0) return <EmptyState icon={FileSearch} title="No articles passed abstract screening" description="Adjust your inclusion criteria and re-run abstract screening." action={{ label: "Back to Abstract Screening", onClick: () => s.setPage("abstract"), icon: FileSearch }} />;

  // Screen `targets`. When `append`, keep the existing results and add the new
  // rows (used to screen papers added after the first run, e.g. via snowball);
  // otherwise replace the whole table.
  async function run(targets: typeof passed = passed, append = false) {
    const list = targets;
    if (list.length === 0) { toast.info("Nothing to screen."); return; }
    const { abort } = s.startTask("full-text-screen", [{ id: "ft", label: "Full-text screening", status: "running" }]);
    s.updateTask("full-text-screen", { progress: { done: 0, total: list.length } });
    const signal = abort.signal;
    const start = Date.now();
    try {
      const out: FullTextResult[] = [];
      for (let i = 0; i < list.length; i++) {
        if (signal.aborted) break;
        s.updateTask("full-text-screen", {
          progress: { done: i, total: list.length, label: list[i].Title.slice(0, 80) },
          detail: list[i].Title.slice(0, 80),
        });
        try {
          const r = await AIService.screenFullTextMultiAgent(
            { paper_id: list[i].paper_id, Title: list[i].Title, URL: list[i].URL, Source: list[i].Source, Abstract: list[i].Abstract },
            s.inclusion, s.exclusion,
            s.fullTexts[list[i].paper_id]?.text,
            signal,
            s.pico,
          );
          out.push(r);
        } catch (e: any) {
          if (signal.aborted) break;
          console.error(`full-text-screen ${i + 1} failed:`, e?.message);
        }
        s.updateTask("full-text-screen", { progress: { done: i + 1, total: list.length } });
      }
      // Merge with existing results when appending; otherwise replace. Dedupe by
      // paper_id so a re-screen of an existing paper overwrites the old row.
      const prior = append && s.fullTextResults ? s.fullTextResults : [];
      const byId = new Map(prior.map(r => [r.paper_id, r]));
      out.forEach(r => byId.set(r.paper_id, r));
      const combined = Array.from(byId.values());
      s.setFullTextResults(combined);
      s.setFtDuration((Date.now() - start) / 1000);
      const ftReasons: Record<string, number> = {};
      for (const r of combined) {
        if (r.Decision === "Exclude") {
          const bucket = categoriseFullTextExclusion(r, s.inclusion, s.exclusion);
          ftReasons[bucket] = (ftReasons[bucket] || 0) + 1;
        }
      }
      s.setPrisma(p => ({ ...p, ft_exclusion_breakdown: ftReasons, included_final: combined.filter(x => x.Decision === "Include").length }));
      if (signal.aborted) {
        s.updateTask("full-text-screen", { status: "canceled" });
        toast.info(`Canceled: ${out.length} of ${list.length} screened`);
      } else {
        s.updateTask("full-text-screen", { status: "done" });
        toast.success(`Full-text screening complete in ${formatDuration((Date.now() - start) / 1000)}`);
      }
    } catch (e: any) {
      s.updateTask("full-text-screen", { status: "error", detail: e?.message });
    }
  }

  const ft = s.fullTextResults;
  const allCriteria = [...s.inclusion, ...s.exclusion];
  // Papers in the queue that haven't been screened at full text yet (e.g. added
  // later via snowball). Offer to screen just these without redoing the rest.
  const newPapers = ft ? passed.filter(p => !ft.some(r => r.paper_id === p.paper_id)) : [];

  const CAP = (x: string) => x.charAt(0).toUpperCase() + x.slice(1);
  const movableCols = colOrder.map(id => {
    if (id === "reason") return { id, label: "Reason", title: "" };
    if (id.startsWith("c:")) { const c = id.slice(2); return { id, label: c.length > 22 ? c.slice(0, 22) + "…" : c, title: c }; }
    return { id, label: CAP(id), title: "" };
  });

  // Render one movable body cell for a given column id, reproducing the PICO /
  // criterion / reason cell so the columns can be reordered freely.
  function renderMovableCell(row: FullTextResult, id: string) {
    if (id === "reason") {
      return <td key={id} className="px-3 py-2 text-foreground/90 min-w-[320px] max-w-[480px] bg-inherit"><div className="line-clamp-4 leading-relaxed">{row.Reason}</div></td>;
    }
    if (id.startsWith("c:")) {
      const c = id.slice(2);
      const ev = row.criteriaEvidence?.[c];
      const badge = (
        <span className={`px-2 py-0.5 rounded text-xs font-medium cursor-pointer ${row.criteriaEval[c] === "INCLUDE" ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
          {row.criteriaEval[c] || "N/A"}
        </span>
      );
      return (
        <td key={id} className="px-3 py-2 bg-inherit">
          {ev ? (
            <ScrollAwarePopover>
              <PopoverTrigger asChild>{badge}</PopoverTrigger>
              <PopoverContent className="w-80 text-xs space-y-2 relative pr-7">
                <PopoverClose className="absolute right-2 top-2 size-5 grid place-items-center rounded text-muted-foreground hover:bg-muted hover:text-foreground" title="Close"><XIcon className="size-3.5" /></PopoverClose>
                <div><div className="font-medium mb-1">Criterion</div><div className="text-muted-foreground">{c}</div></div>
                <div><div className="font-medium mb-1">Evidence</div><blockquote className="border-l-2 border-primary/40 pl-2 italic text-muted-foreground whitespace-pre-wrap">{ev.evidence || "No matching passage found in source text."}</blockquote></div>
                <div><div className="font-medium mb-1">Reasoning</div><div className="text-muted-foreground">{ev.reasoning}</div></div>
              </PopoverContent>
            </ScrollAwarePopover>
          ) : badge}
        </td>
      );
    }
    // PICO column
    const k = id as "population" | "intervention" | "comparator" | "outcome";
    const pe = row.picoEvidence?.[k];
    if (!pe || !pe.value) {
      const naReason = (s.pico[k] || "").trim()
        ? `The full text did not provide enough information to judge the ${k}.`
        : `No ${k} was specified in your PICO frame, so there is nothing to assess this article against. Add one on the Home page to enable this check.`;
      return (
        <td key={id} className="px-3 py-2 bg-inherit">
          <ScrollAwarePopover>
            <PopoverTrigger asChild>
              <button><span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-slate-100 text-slate-600"><Minus className="size-3" />n/a</span></button>
            </PopoverTrigger>
            <PopoverContent className="w-80 text-xs space-y-2 relative pr-7">
              <PopoverClose className="absolute right-2 top-2 size-5 grid place-items-center rounded text-muted-foreground hover:bg-muted hover:text-foreground" title="Close"><XIcon className="size-3.5" /></PopoverClose>
              <div><div className="font-medium mb-1 capitalize">{k}</div><div className="text-muted-foreground"><span className="font-medium text-foreground/80">Not assessed.</span> {naReason}</div></div>
            </PopoverContent>
          </ScrollAwarePopover>
        </td>
      );
    }
    return (
      <td key={id} className="px-3 py-2 bg-inherit">
        <ScrollAwarePopover>
          <PopoverTrigger asChild><button><PicoBadge match={pe.match} /></button></PopoverTrigger>
          <PopoverContent className="w-80 text-xs space-y-2 relative pr-7">
            <PopoverClose className="absolute right-2 top-2 size-5 grid place-items-center rounded text-muted-foreground hover:bg-muted hover:text-foreground" title="Close"><XIcon className="size-3.5" /></PopoverClose>
            <div><div className="font-medium mb-1 capitalize">{k}</div><div className="text-muted-foreground">{pe.value}</div></div>
            <div>
              <div className="font-medium mb-1">Evidence from text</div>
              {pe.evidence ? (
                <blockquote className="border-l-2 border-primary/40 pl-2 italic text-muted-foreground whitespace-pre-wrap">{pe.evidence}</blockquote>
              ) : (
                <div className="text-muted-foreground italic">No matching passage found in the abstract or full text.</div>
              )}
            </div>
          </PopoverContent>
        </ScrollAwarePopover>
      </td>
    );
  }

  return (
    <div className="space-y-4">
      {!ft && (
        <>
          <Alert><AlertDescription>{passed.length} articles ready for full-text analysis.</AlertDescription></Alert>
          {task && task.status === "running" && (
            <TaskProgressCard
              task={task}
              title="Full-text screening"
              onCancel={() => s.cancelTask("full-text-screen")}
            />
          )}
          <Button onClick={() => run()} disabled={running} size="lg" className="w-full">
            <FlaskConical className="size-4 mr-2" />{running ? "Analyzing..." : "Begin Full-Text Screening"}
          </Button>
        </>
      )}

      {ft && (() => {
            const includedEff = ft.filter(x => effectiveFullTextDecision(x, s.fullTextOverrides) === "Include").length;
            const excludedEff = ft.filter(x => effectiveFullTextDecision(x, s.fullTextOverrides) === "Exclude").length;
            return (
              <>
                <ControlPane
                  stats={<>
                    <InlineStat icon={FileText} value={ft.length} label="Articles" />
                    <PaneDivider />
                    <InlineStat icon={CheckCircle2} value={includedEff} label="Included" tone="success" hint={ft.length ? `${Math.round((includedEff / ft.length) * 100)}%` : undefined} />
                    <InlineStat icon={XCircle} value={excludedEff} label="Excluded" tone="danger" hint={ft.length ? `${Math.round((excludedEff / ft.length) * 100)}%` : undefined} />
                    <PaneDivider />
                    <InlineStat icon={Clock} value={formatDuration(s.ftDuration)} label="Time" tone="neutral" />
                  </>}
                  actions={<>
                    {running ? (
                      <Button size="sm" variant="destructive" className="h-8 shadow-sm" onClick={() => s.cancelTask("full-text-screen")} title="Stop screening (keeps the current results)">
                        <XIcon className="size-3.5 mr-1.5" />Cancel
                      </Button>
                    ) : (
                      <>
                        {newPapers.length > 0 && (
                          <Button size="sm" className="h-8 shadow-sm" onClick={() => run(newPapers, true)}>
                            <FlaskConical className="size-3.5 mr-1.5" />Screen {newPapers.length} new
                          </Button>
                        )}
                        <Button size="sm" variant="outline" className="h-8" onClick={() => run(passed, false)}>
                          <FlaskConical className="size-3.5 mr-1.5" />Re-run all
                        </Button>
                      </>
                    )}
                    <span className="mx-0.5 h-6 w-px bg-border" aria-hidden="true" />
                    <Button size="sm" onClick={() => setRapidOpen(true)} className="h-8 shadow-sm bg-amber-400 hover:bg-amber-500 text-amber-950 border-amber-400 dark:bg-amber-500 dark:hover:bg-amber-400 dark:text-amber-950" title="Review records fast with the keyboard">
                      <Zap className="size-3.5 mr-1.5" />Rapid review
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => setMaxOpen(true)} className="h-8 px-2" title="Expand the table to full screen">
                      <Maximize2 className="size-4" />
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => downloadFullTextXlsx(ft, allCriteria, s.fullTextOverrides)} className="h-8 px-2" title="Download results as Excel (XLSX)">
                      <Download className="size-4" />
                    </Button>
                  </>}
                />
                <RapidScreen
                  open={rapidOpen}
                  onClose={() => setRapidOpen(false)}
                  label="Rapid review: full text"
                  items={ft.map(x => ({
                    id: x.paper_id, title: x.Title, source: x.Source, url: x.URL,
                    text: s.fullTexts[x.paper_id]?.text || x.Abstract,
                    aiInclude: x.Decision === "Include", reason: x.Reason,
                    override: s.fullTextOverrides[x.paper_id]
                      ? (s.fullTextOverrides[x.paper_id] === "Include" ? "include" : "exclude")
                      : undefined,
                  }))}
                  onDecide={(id, d) => s.setFullTextOverride(id, d === "include" ? "Include" : "Exclude")}
                />
                {task && task.status === "running" && (
                  <div>
                    <TaskProgressCard
                      task={task}
                      title="Full-text screening"
                      onCancel={() => s.cancelTask("full-text-screen")}
                    />
                  </div>
                )}
                <Card className="p-4">
                <div className={maxOpen ? "fixed inset-0 z-50 bg-background p-4 flex flex-col gap-2" : ""}>
            {maxOpen && (
              <div className="flex items-center justify-between shrink-0">
                <h3 className="font-medium">Full-Text Results ({ft.length})</h3>
                <Button size="sm" variant="outline" onClick={() => setMaxOpen(false)}><Minimize2 className="size-4 mr-1.5" />Close</Button>
              </div>
            )}
            <div className={`rounded-lg border overflow-auto ${maxOpen ? "flex-1 min-h-0" : "max-h-[600px]"}`}>
            <table className="w-full text-sm border-collapse">
              <thead className="bg-muted sticky top-0 z-30 [&_th]:text-[11px] [&_th]:font-semibold [&_th]:uppercase [&_th]:tracking-wider [&_th]:text-muted-foreground [&_th]:py-2.5">
                <tr className="text-left">
                  <th className="px-3 py-2 sticky left-0 bg-muted z-40 border-b border-r min-w-[60px] text-center" title="Reviewer override: check to keep, uncheck to drop">Keep</th>
                  <th className="px-3 py-2 sticky left-[60px] bg-muted z-40 border-b border-r min-w-[120px]">Decision</th>
                  <th className="px-3 py-2 sticky left-[180px] bg-muted z-40 border-b border-r min-w-[300px] max-w-[300px] shadow-[6px_0_8px_-6px_rgba(0,0,0,0.22)]">Title</th>
                  {movableCols.map(col => (
                    <th
                      key={col.id}
                      draggable
                      onDragStart={() => setDragCol(col.id)}
                      onDragEnter={() => setOverCol(col.id)}
                      onDragOver={e => e.preventDefault()}
                      onDrop={e => { e.preventDefault(); if (dragCol) moveCol(dragCol, col.id); setDragCol(null); setOverCol(null); }}
                      onDragEnd={() => { setDragCol(null); setOverCol(null); }}
                      title={col.title ? `${col.title} — drag to reorder` : "Drag to reorder column"}
                      className={`group/th px-3 border-b whitespace-nowrap max-w-[160px] cursor-grab active:cursor-grabbing select-none transition-colors ${col.id === "reason" ? "min-w-[320px]" : ""} ${dragCol === col.id ? "opacity-40" : ""} ${overCol === col.id && dragCol && dragCol !== col.id ? "bg-primary/10" : ""}`}
                    >
                      <span className="inline-flex items-center gap-1 truncate">
                        <GripVertical className="size-3 shrink-0 opacity-0 group-hover/th:opacity-40 transition-opacity" />{col.label}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ft.map(row => {
                  const eff = effectiveFullTextDecision(row, s.fullTextOverrides);
                  const isOverridden = eff !== row.Decision;
                  const keep = eff === "Include";
                  const isOpen = expanded.has(row.paper_id);
                  return (
                  <Fragment key={row.paper_id}>
                  <tr className="border-b border-border/60 last:border-b-0 align-top bg-card hover:bg-muted transition-colors">
                    <td className="px-3 py-2 sticky left-0 z-20 border-r bg-inherit text-center">
                      <Checkbox
                        checked={keep}
                        onCheckedChange={(v) => {
                          const wantKeep = v === true;
                          if ((row.Decision === "Include") === wantKeep) {
                            s.clearFullTextOverride(row.paper_id);
                          } else {
                            s.setFullTextOverride(row.paper_id, wantKeep ? "Include" : "Exclude");
                          }
                        }}
                        aria-label="Keep this article"
                      />
                    </td>
                    <td className="px-3 py-2 sticky left-[60px] z-20 border-r whitespace-nowrap bg-inherit">
                      <div className="space-y-1">
                        <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${eff === "Include" ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
                          {eff}
                        </span>
                        {isOverridden && (
                          <div className="text-[10px] text-muted-foreground leading-tight">
                            AI: {row.Decision} · reviewer-edited
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="px-3 py-2 sticky left-[180px] z-20 border-r min-w-[300px] max-w-[300px] bg-inherit shadow-[6px_0_8px_-6px_rgba(0,0,0,0.18)]">
                      <div className="flex items-start gap-1.5">
                        <button onClick={() => toggleRow(row.paper_id)} className="mt-0.5 shrink-0 text-muted-foreground hover:text-foreground" title={isOpen ? "Hide details" : "Show abstract and full text"} aria-expanded={isOpen}>
                          {isOpen ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
                        </button>
                        <a href={row.URL} target="_blank" rel="noreferrer" className="hover:underline break-words">
                          {row.Title}
                        </a>
                      </div>
                    </td>
                    {movableCols.map(col => renderMovableCell(row, col.id))}
                  </tr>
                  {isOpen && (
                    <tr className="border-b border-border/60 bg-muted/30">
                      <td colSpan={8 + allCriteria.length} className="p-0">
                        <div className="sticky left-0 w-[min(920px,92vw)] p-4 space-y-4">
                          <div className="rounded-lg border bg-card p-3.5 shadow-sm">
                            <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">Abstract</div>
                            <p className="text-sm leading-relaxed whitespace-pre-wrap text-foreground/90">{row.Abstract || <span className="text-muted-foreground italic">No abstract available.</span>}</p>
                          </div>
                          {row.Reason && (
                            <div className="rounded-lg border border-primary/30 bg-primary/[0.04] p-3.5 shadow-sm">
                              <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-primary mb-1.5">
                                <Sparkles className="size-3" />Reasoning
                              </div>
                              <p className="text-sm leading-relaxed text-foreground/90">{row.Reason}</p>
                            </div>
                          )}

                          {/* PICO evidence — each element in its own box */}
                          {(["population", "intervention", "comparator", "outcome"] as const).some(k => row.picoEvidence?.[k]?.value) && (
                            <div>
                              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">PICO evidence</div>
                              <div className="grid md:grid-cols-2 gap-2.5">
                                {(["population", "intervention", "comparator", "outcome"] as const).map(k => {
                                  const pe = row.picoEvidence?.[k];
                                  if (!pe || !pe.value) return null;
                                  return (
                                    <div key={k} className="rounded-lg border bg-card p-3 shadow-sm space-y-1.5">
                                      <div className="flex items-center justify-between gap-2">
                                        <span className="text-sm font-medium capitalize">{k}</span>
                                        <span className={`px-2 py-0.5 rounded-full text-[11px] font-medium border ${matchTone(pe.match)}`}>{matchLabel(pe.match)}</span>
                                      </div>
                                      <p className="text-xs text-foreground/80">{pe.value}</p>
                                      {pe.evidence && <blockquote className="text-xs italic text-foreground/70 border-l-2 border-primary/30 pl-2 break-words">“{pe.evidence}”</blockquote>}
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          )}

                          {/* Per-criterion evidence — each criterion in its own box */}
                          {allCriteria.some(c => row.criteriaEvidence?.[c] || row.criteriaEval?.[c]) && (
                            <div>
                              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">Criteria evidence</div>
                              <div className="grid md:grid-cols-2 gap-2.5">
                                {allCriteria.map(c => {
                                  const ev = row.criteriaEvidence?.[c];
                                  const verdict = row.criteriaEval?.[c];
                                  if (!ev && !verdict) return null;
                                  const inc = verdict === "INCLUDE";
                                  return (
                                    <div key={c} className="rounded-lg border bg-card p-3 shadow-sm space-y-1.5">
                                      <div className="flex items-start justify-between gap-2">
                                        <span className="text-sm font-medium leading-snug">{c}</span>
                                        <span className={`shrink-0 px-2 py-0.5 rounded-full text-[11px] font-medium border ${inc ? "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-300 dark:border-emerald-900" : "bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-950/40 dark:text-rose-300 dark:border-rose-900"}`}>{verdict || "N/A"}</span>
                                      </div>
                                      {ev?.evidence && <blockquote className="text-xs italic text-foreground/70 border-l-2 border-primary/30 pl-2 break-words">“{ev.evidence}”</blockquote>}
                                      {ev?.reasoning && <p className="text-xs text-muted-foreground leading-relaxed">{ev.reasoning}</p>}
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          )}
                        </div>
                      </td>
                    </tr>
                  )}
                  </Fragment>
                  );
                })}
              </tbody>
            </table>
            </div>
          </div>
                </Card>
              </>
            );
          })()}
    </div>
  );
}

// ---- XLSX export ----------------------------------------------------------

import ExcelJS from "exceljs";

const PICO_MATCH_FILL: Record<string, string> = {
  yes:     "FFDCFCE7",
  partial: "FFFEF3C7",
  no:      "FFFEE2E2",
};
const PICO_MATCH_TEXT: Record<string, string> = {
  yes:     "FF065F46",
  partial: "FF92400E",
  no:      "FF9F1239",
};

async function downloadFullTextXlsx(
  rows: FullTextResult[],
  criteria: string[],
  overrides: Record<string, "Include" | "Exclude"> = {},
) {
  const wb = new ExcelJS.Workbook();
  wb.creator = "Evidence Engine";
  wb.created = new Date();

  const ws = wb.addWorksheet("Full-text screening", {
    views: [{ state: "frozen", xSplit: 2, ySplit: 1 }],
  });

  // Static columns + one column per criterion.
  const baseCols = [
    { header: "Decision",      key: "decision",  width: 12 },
    { header: "AI Decision",   key: "ai_decision", width: 12 },
    { header: "Reviewer Edit", key: "rev_edit",  width: 14 },
    { header: "Title",         key: "title",     width: 48 },
    { header: "Source",        key: "source",    width: 14 },
    { header: "URL",           key: "url",       width: 38 },
    { header: "Reason",        key: "reason",    width: 60 },
    { header: "P · match",     key: "p_match",   width: 12 },
    { header: "P · value",     key: "p_value",   width: 28 },
    { header: "P · evidence",  key: "p_evidence", width: 48 },
    { header: "I · match",     key: "i_match",   width: 12 },
    { header: "I · value",     key: "i_value",   width: 28 },
    { header: "I · evidence",  key: "i_evidence", width: 48 },
    { header: "C · match",     key: "c_match",   width: 12 },
    { header: "C · value",     key: "c_value",   width: 28 },
    { header: "C · evidence",  key: "c_evidence", width: 48 },
    { header: "O · match",     key: "o_match",   width: 12 },
    { header: "O · value",     key: "o_value",   width: 28 },
    { header: "O · evidence",  key: "o_evidence", width: 48 },
    { header: "Inclusion met", key: "inc",       width: 14 },
    { header: "Excl. violations", key: "exc",    width: 16 },
  ];
  const critCols = criteria.map((c, i) => ({
    header: c,
    key: `crit_${i}`,
    width: 36,
  }));
  ws.columns = [...baseCols, ...critCols];

  const header = ws.getRow(1);
  header.height = 28;
  header.eachCell((cell) => {
    cell.font = { bold: true, color: { argb: "FFFFFFFF" }, size: 11 };
    cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF1F2937" } };
    cell.alignment = { vertical: "middle", horizontal: "left", wrapText: true };
    cell.border = { bottom: { style: "thin", color: { argb: "FF374151" } } };
  });

  for (const r of rows) {
    const pe = (r.picoEvidence || {}) as any;
    const overrideVal = overrides[r.paper_id];
    const effective = overrideVal ?? r.Decision;
    const rowData: Record<string, any> = {
      decision:    effective,
      ai_decision: r.Decision,
      rev_edit:    overrideVal && overrideVal !== r.Decision ? "yes" : "",
      title:    r.Title,
      source:   r.Source,
      url:      r.URL,
      reason:   r.Reason,
      p_match:  pe.population?.match || "",
      p_value:  pe.population?.value || "",
      p_evidence: pe.population?.evidence || "",
      i_match:  pe.intervention?.match || "",
      i_value:  pe.intervention?.value || "",
      i_evidence: pe.intervention?.evidence || "",
      c_match:  pe.comparator?.match || "",
      c_value:  pe.comparator?.value || "",
      c_evidence: pe.comparator?.evidence || "",
      o_match:  pe.outcome?.match || "",
      o_value:  pe.outcome?.value || "",
      o_evidence: pe.outcome?.evidence || "",
      inc:      r.inclusion_score,
      exc:      r.exclusion_violations,
    };
    criteria.forEach((c, i) => {
      const vote = r.criteriaEval?.[c] ?? "";
      const ev = r.criteriaEvidence?.[c];
      // Use newlines so wrapText puts the vote, the quote and the reasoning
      // on separate visual lines inside a single cell.
      const parts: string[] = [vote];
      if (ev?.evidence) parts.push(`"${ev.evidence}"`);
      if (ev?.reasoning) parts.push(ev.reasoning);
      rowData[`crit_${i}`] = parts.join("\n\n");
    });

    const row = ws.addRow(rowData);
    row.eachCell((cell) => {
      cell.alignment = { vertical: "top", wrapText: true };
      cell.font = { size: 10 };
    });

    // Decision cell colour reflects the EFFECTIVE decision; AI Decision column
    // separately preserves the AI's original verdict for the audit trail.
    const decCell = row.getCell("decision");
    decCell.fill = {
      type: "pattern", pattern: "solid",
      fgColor: { argb: effective === "Include" ? "FFDCFCE7" : "FFFEE2E2" },
    };
    decCell.font = {
      size: 10, bold: true,
      color: { argb: effective === "Include" ? "FF065F46" : "FF991B1B" },
    };
    decCell.alignment = { vertical: "top", horizontal: "center", wrapText: true };
    if (overrideVal && overrideVal !== r.Decision) {
      // Faint amber border on overridden Decision cells so reviewers can spot
      // their own edits at a glance.
      decCell.border = {
        top: { style: "thin", color: { argb: "FFD97706" } },
        bottom: { style: "thin", color: { argb: "FFD97706" } },
        left: { style: "thin", color: { argb: "FFD97706" } },
        right: { style: "thin", color: { argb: "FFD97706" } },
      };
    }

    // Colour the four PICO match cells.
    for (const key of ["p_match", "i_match", "c_match", "o_match"]) {
      const cell = row.getCell(key);
      const v = String(cell.value || "").toLowerCase();
      if (v && PICO_MATCH_FILL[v]) {
        cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: PICO_MATCH_FILL[v] } };
        cell.font = { size: 10, bold: true, color: { argb: PICO_MATCH_TEXT[v] } };
        cell.alignment = { vertical: "middle", horizontal: "center", wrapText: false };
      }
    }

    // Colour each criterion cell by the first line (INCLUDE / EXCLUDE).
    criteria.forEach((_, i) => {
      const cell = row.getCell(`crit_${i}`);
      const firstLine = String(cell.value || "").split("\n", 1)[0].toUpperCase();
      if (firstLine === "INCLUDE") {
        cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFECFCCB" } };
      } else if (firstLine === "EXCLUDE") {
        cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFFEE2E2" } };
      }
    });

    // URL hyperlink.
    if (r.URL) {
      const urlCell = row.getCell("url");
      urlCell.value = { text: r.URL, hyperlink: r.URL };
      urlCell.font = { size: 10, color: { argb: "FF1D4ED8" }, underline: true };
    }

    row.height = 78;
  }

  ws.autoFilter = {
    from: { row: 1, column: 1 },
    to: { row: 1, column: ws.columnCount },
  };

  const buf = await wb.xlsx.writeBuffer();
  const blob = new Blob([buf], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  a.href = url;
  a.download = `fulltext-screening-${stamp}.xlsx`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function PicoBadge({ match }: { match: "yes" | "partial" | "no" }) {
  if (match === "yes") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-emerald-100 text-emerald-800">
        <Check className="size-3" />Met
      </span>
    );
  }
  if (match === "partial") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-800">
        <Minus className="size-3" />Partial
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-rose-100 text-rose-800">
      <XIcon className="size-3" />None
    </span>
  );
}
