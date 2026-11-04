import { Fragment, useState, useMemo, useEffect } from "react";
import { usePagedRows } from "../lib/usePagedRows";
import { ShowMore } from "../components/ShowMore";
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
import { FlaskConical, Check, Minus, X as XIcon, Download, Zap, FileSearch, ChevronRight, ChevronDown, Maximize2, Minimize2, FileText, CheckCircle2, XCircle, Clock, Sparkles, GripVertical, Plus, Search, CircleDashed } from "lucide-react";
import { ControlPane, InlineStat, PaneDivider } from "../components/ControlPane";
import { EmptyState } from "../components/EmptyState";
import { CriteriaColumns } from "../components/CriteriaColumns";
import {
  DropdownMenu, DropdownMenuTrigger, DropdownMenuContent,
  DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator,
} from "../components/ui/dropdown-menu";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from "../components/ui/alert-dialog";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";
import { frameworkOf, labelFor, type FrameworkId } from "../lib/frameworks";

/** What this run will actually read, and what it will judge against.
 *
 *  The full-text counterpart of the abstract stage's preview. The question it
 *  answers is specific to this stage: a record whose full text was never
 *  acquired gets screened on its abstract, and calling that a full-text
 *  decision misstates the review. That has to be visible BEFORE the run, not
 *  discovered in the results table afterwards. */
export function FullTextPreview({
  queue, fullTexts, framework, pico, inclusion, exclusion, onAcquire,
}: {
  queue: { paper_id: string; Title: string }[];
  fullTexts: Record<string, { status?: string; source?: string; text?: string }>;
  framework: FrameworkId;
  pico: Record<string, string>;
  inclusion: string[];
  exclusion: string[];
  onAcquire?: () => void;
}) {
  if (queue.length === 0) return null;

  // Group by what each record will be read from, so "full text" is a claim the
  // reviewer can check rather than take on trust.
  const bySource = new Map<string, number>();
  let withText = 0;
  for (const p of queue) {
    const ft = fullTexts[p.paper_id];
    if (ft?.status === "found" && (ft.text || "").trim()) {
      withText++;
      const src = ft.source || "Acquired";
      bySource.set(src, (bySource.get(src) || 0) + 1);
    }
  }
  const abstractOnly = queue.length - withText;

  return (
    <Card className="p-0 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b bg-muted/10">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Evidence per article</span>
        <span className="text-[11px] text-muted-foreground">
          {withText.toLocaleString()} of {queue.length.toLocaleString()} have full text
        </span>
      </div>

      <div className="divide-y">
        {[...bySource.entries()].sort((a, b) => b[1] - a[1]).map(([src, n]) => (
          <div key={src} className="flex items-center justify-between gap-2 px-3 py-2 text-xs">
            <span className="min-w-0 truncate font-medium">{src}</span>
            <span className="tabular-nums text-muted-foreground">{n.toLocaleString()}</span>
          </div>
        ))}
        {abstractOnly > 0 && (
          <div className="px-3 py-2">
            <div className="flex items-center justify-between gap-2 text-xs">
              <span className="font-medium">Abstract only</span>
              <span className="tabular-nums font-medium text-amber-600">{abstractOnly.toLocaleString()}</span>
            </div>
            <div className="mt-1 text-[11px] text-amber-600">
              No full text was acquired for {abstractOnly === 1 ? "this article" : `these ${abstractOnly.toLocaleString()} articles`},
              so {abstractOnly === 1 ? "it" : "they"} will be judged on the abstract alone. Acquire the full
              text first if the decision depends on the methods or results.
              {/* Explicit size and weight on the button: a bare <button> does
                  not inherit the surrounding text style, so it would render a
                  size up and heavier than the sentence it sits in. */}
              {onAcquire && (
                <button onClick={onAcquire} className="ml-1 text-[11px] font-normal underline hover:no-underline">
                  Go to Full-Text Acquisition
                </button>
              )}
            </div>
          </div>
        )}
      </div>

      <CriteriaColumns
        framework={framework} pico={pico} inclusion={inclusion} exclusion={exclusion}
        emptyHint="No criteria set. Define the question on the Home page first, or full-text screening has nothing to judge each article against."
      />
    </Card>
  );
}

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
  const [confirmNew, setConfirmNew] = useState(false);
  const [maxOpen, setMaxOpen] = useState(false);
  const [ftQuery, setFtQuery] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const toggleRow = (id: string) => setExpanded(prev => {
    const n = new Set(prev);
    n.has(id) ? n.delete(id) : n.add(id);
    return n;
  });

  // Reorderable (non-frozen) columns: the active framework's elements + each
  // criterion + Reason. Keep/Decision/Title stay pinned. Order persists and
  // reconciles with the current element/criteria set (new ones append, removed
  // ones drop out). These hooks run before any early return so hook order stays
  // stable.
  const fw = frameworkOf(s.framework);
  const frameIds = fw.elements.map(e => e.id);
  const movableIds = [...frameIds,
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
  // Client-side row filter for the results table (title / reason / decision), then
  // chunked rendering: every full-text row used to mount on each tab open. Declared
  // here, ahead of the early returns below, so the hook order stays fixed.
  const ftQ = ftQuery.trim().toLowerCase();
  const visibleFt = useMemo(() => {
    const all = s.fullTextResults;
    if (!all) return [];
    return ftQ
      ? all.filter(row => `${row.Title} ${row.Reason} ${row.Decision} ${effectiveFullTextDecision(row, s.fullTextOverrides)}`.toLowerCase().includes(ftQ))
      : all;
  }, [s.fullTextResults, ftQ, s.fullTextOverrides]);
  const pagedFt = usePagedRows(visibleFt);
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
  // Only show the empty state when there is genuinely nothing to display. If a
  // full-text pass already ran, keep showing it even when the CURRENT abstract
  // run includes 0 papers (e.g. after restoring a different screening pass), so
  // completed full-text work isn't hidden by swapping abstract runs.
  if (passed.length === 0 && !s.fullTextResults) return <EmptyState icon={FileSearch} title="No articles passed abstract screening" description="The current screening run included 0 articles. Adjust your inclusion criteria and re-run abstract screening, or restore a run that has included articles." action={{ label: "Back to Abstract Screening", onClick: () => s.setPage("abstract"), icon: FileSearch }} />;

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
      // On cancel, merge this run's partial results into the ACTUAL existing set
      // (by paper) so a full re-screen never loses prior rows; on success use
      // `combined` (append or full replace, per mode).
      let finalSet = combined;
      if (signal.aborted) {
        const keep = new Map((s.fullTextResults ?? []).map(r => [r.paper_id, r]));
        out.forEach(r => keep.set(r.paper_id, r));
        finalSet = Array.from(keep.values());
      }
      s.setFullTextResults(finalSet);
      s.setFtDuration((Date.now() - start) / 1000);
      const ftReasons: Record<string, number> = {};
      for (const r of finalSet) {
        if (r.Decision === "Exclude") {
          const bucket = categoriseFullTextExclusion(r, s.inclusion, s.exclusion);
          ftReasons[bucket] = (ftReasons[bucket] || 0) + 1;
        }
      }
      s.setPrisma(p => ({ ...p, ft_exclusion_breakdown: ftReasons, included_final: finalSet.filter(x => x.Decision === "Include").length }));
      if (signal.aborted) {
        s.updateTask("full-text-screen", { status: "canceled" });
        toast.info(`Canceled: ${out.length} screened this run, ${finalSet.length} kept in total.`);
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
  // How much of the queue will actually be read at full length. A record with
  // no acquired text is screened on its abstract, which is a different claim.
  const ftReady = passed.filter(p => {
    const rec = s.fullTexts[p.paper_id];
    return rec?.status === "found" && (rec.text || "").trim().length > 0;
  }).length;
  const ftMissing = passed.length - ftReady;

  const movableCols = colOrder.map(id => {
    if (id === "reason") return { id, label: "Reason", title: "" };
    if (id.startsWith("c:")) { const c = id.slice(2); return { id, label: c, title: c }; }
    return { id, label: labelFor(s.framework, id), title: "" };
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
    // Frame-element column (PICO population/intervention/comparator/outcome or
    // PCC population/concept/context), keyed by element id.
    const pe = row.picoEvidence?.[id];
    if (!pe || !pe.value) {
      const naReason = (s.pico[id] || "").trim()
        ? `The full text did not provide enough information to judge the ${id}.`
        : `No ${id} was specified in your ${fw.label} frame, so there is nothing to assess this article against. Add one on the Home page to enable this check.`;
      return (
        <td key={id} className="px-3 py-2 bg-inherit">
          <ScrollAwarePopover>
            <PopoverTrigger asChild>
              <button><span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-slate-100 text-slate-600"><Minus className="size-3" />n/a</span></button>
            </PopoverTrigger>
            <PopoverContent className="w-80 text-xs space-y-2 relative pr-7">
              <PopoverClose className="absolute right-2 top-2 size-5 grid place-items-center rounded text-muted-foreground hover:bg-muted hover:text-foreground" title="Close"><XIcon className="size-3.5" /></PopoverClose>
              <div><div className="font-medium mb-1 capitalize">{id}</div><div className="text-muted-foreground"><span className="font-medium text-foreground/80">Not assessed.</span> {naReason}</div></div>
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
            <div><div className="font-medium mb-1 capitalize">{id}</div><div className="text-muted-foreground">{pe.value}</div></div>
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
          {/* Counts and the run control in one bar, the shape every other
              working tab uses. This was an alert restating a number and a
              full-width button, with nothing said about what would be read. */}
          {!running && (
            <ControlPane
              stats={<>
                <InlineStat icon={FileText} value={passed.length} label="Ready to screen" />
                <PaneDivider />
                <InlineStat icon={CheckCircle2} tone="success" value={ftReady} label="Full text" />
                {ftMissing > 0 && (
                  <InlineStat icon={CircleDashed} tone="amber" value={ftMissing} label="Abstract only" />
                )}
              </>}
              actions={<>
                {ftMissing > 0 && (
                  <Button size="sm" variant="outline" className="h-8 shadow-sm" onClick={() => s.setPage("acquisition")}
                          title="Collect the missing full texts before judging them">
                    <FileSearch className="size-3.5 mr-1.5" />Acquire full texts
                  </Button>
                )}
                <Button size="sm" className="h-8 shadow-sm" onClick={() => run()}>
                  <FlaskConical className="size-3.5 mr-1.5" />Run full-text screening
                </Button>
              </>}
            />
          )}
          {task && task.status === "running" && (
            <TaskProgressCard
              task={task}
              title="Full-text screening"
              onCancel={() => s.cancelTask("full-text-screen")}
            />
          )}
          {!running && (
            <FullTextPreview
              queue={passed}
              fullTexts={s.fullTexts}
              framework={s.framework}
              pico={s.pico}
              inclusion={s.inclusion}
              exclusion={s.exclusion}
              onAcquire={() => s.setPage("acquisition")}
            />
          )}
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
                    <InlineStat icon={CircleDashed} value={newPapers.length} label="Pending" tone={newPapers.length > 0 ? "amber" : "neutral"} />
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
                        <Button size="sm" className="h-8 shadow-sm"
                          onClick={() => setConfirmNew(true)}
                          disabled={newPapers.length === 0}
                          title={newPapers.length === 0 ? "No newly included articles to screen" : `Screen ${newPapers.length} newly included article${newPapers.length === 1 ? "" : "s"}`}>
                          <Plus className="size-3.5 mr-1.5" />Screen new
                        </Button>
                        <Button size="sm" variant="outline" className="h-8" onClick={() => run(passed, false)}>
                          <FlaskConical className="size-3.5 mr-1.5" />Re-run all
                        </Button>
                      </>
                    )}
                    <span className="mx-0.5 h-6 w-px bg-border" aria-hidden="true" />
                    <Button size="sm" onClick={() => setRapidOpen(true)} className="h-8 shadow-sm bg-amber-400 hover:bg-amber-500 text-amber-950 border-amber-400 dark:bg-amber-500 dark:hover:bg-amber-400 dark:text-amber-950" title="Review records fast with the keyboard">
                      <Zap className="size-3.5 mr-1.5" />Rapid screen
                    </Button>
                  </>}
                />
                <AlertDialog open={confirmNew} onOpenChange={setConfirmNew}>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>Screen {newPapers.length} new article{newPapers.length === 1 ? "" : "s"}?</AlertDialogTitle>
                      <AlertDialogDescription>
                        Only newly included articles not yet full-text screened will be reviewed. Existing decisions and overrides are kept.
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>Cancel</AlertDialogCancel>
                      <AlertDialogAction onClick={() => run(newPapers, true)}>Screen {newPapers.length}</AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
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
            {maxOpen ? (
              <div className="flex items-center gap-3 shrink-0">
                <h3 className="font-medium shrink-0">Full-Text Results ({ft.length})</h3>
                <div className="relative flex-1 min-w-0">
                  <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground pointer-events-none" />
                  <input value={ftQuery} onChange={e => setFtQuery(e.target.value)} placeholder="Filter results…"
                    className="w-full h-8 pl-8 pr-7 rounded-md border bg-background text-xs outline-none focus-visible:ring-1 focus-visible:ring-ring placeholder:text-muted-foreground" />
                  {ftQuery && <button type="button" onClick={() => setFtQuery("")} className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground" title="Clear filter"><XIcon className="size-3.5" /></button>}
                </div>
                <div className="ml-auto flex items-center gap-1 shrink-0">
                  {ftQuery && <span className="text-[11px] text-muted-foreground tabular-nums">{visibleFt.length}/{ft.length}</span>}
<ExportMenu
                    rows={ft} criteria={allCriteria} overrides={s.fullTextOverrides}
                    framework={s.framework}
                    meta={{ framework: s.framework, pico: s.pico, inclusion: s.inclusion, exclusion: s.exclusion }}
                    className="h-8 px-2"
                  />
                  <Button size="sm" variant="ghost" onClick={() => setMaxOpen(false)}
                    className="h-8 px-2" title="Exit full screen" aria-label="Exit full screen">
                    <Minimize2 className="size-4" />
                  </Button>
                </div>
              </div>
            ) : (
              /* Table toolbar: filter on the left (fills the header), tools pinned top-right. */
              <div className="flex items-center gap-2 mb-1.5">
                <div className="relative flex-1 min-w-0">
                  <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground pointer-events-none" />
                  <input value={ftQuery} onChange={e => setFtQuery(e.target.value)} placeholder="Filter results by title, reason, or decision…"
                    className="w-full h-8 pl-8 pr-7 rounded-md border bg-background text-xs outline-none focus-visible:ring-1 focus-visible:ring-ring placeholder:text-muted-foreground" />
                  {ftQuery && <button type="button" onClick={() => setFtQuery("")} className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground" title="Clear filter"><XIcon className="size-3.5" /></button>}
                </div>
                <div className="ml-auto flex items-center gap-1 shrink-0">
                  {ftQuery && <span className="text-[11px] text-muted-foreground tabular-nums">{visibleFt.length}/{ft.length}</span>}
                  <Button size="sm" variant="ghost" onClick={() => setMaxOpen(true)} className="h-7 px-2 text-muted-foreground hover:text-foreground" title="Expand the table to full screen">
                    <Maximize2 className="size-4" />
                  </Button>
<ExportMenu
                    rows={ft} criteria={allCriteria} overrides={s.fullTextOverrides}
                    framework={s.framework}
                    meta={{ framework: s.framework, pico: s.pico, inclusion: s.inclusion, exclusion: s.exclusion }}
                    className="h-7 px-2 text-muted-foreground hover:text-foreground"
                  />
                </div>
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
                      title={col.title || undefined}
                      className={`group/th px-3 border-b cursor-grab active:cursor-grabbing select-none transition-colors ${col.id === "reason" ? "min-w-[320px]" : "w-[190px] min-w-[190px] max-w-[190px]"} ${dragCol === col.id ? "opacity-40" : ""} ${overCol === col.id && dragCol && dragCol !== col.id ? "bg-primary/10" : ""}`}
                    >
                      <span className="flex items-center gap-1 min-w-0">
                        <GripVertical className="size-3 shrink-0 opacity-0 group-hover/th:opacity-40 transition-opacity" />
                        <span className="truncate">{col.label}</span>
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ftQuery && visibleFt.length === 0 && (
                  <tr><td colSpan={3 + movableCols.length} className="px-3 py-10 text-center text-sm text-muted-foreground">No results match "{ftQuery}".</td></tr>
                )}
                {pagedFt.visible.map(row => {
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
                      <td colSpan={3 + movableCols.length} className="p-0">
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

                          {/* Frame-element evidence — each element in its own box */}
                          {fw.elements.some(el => row.picoEvidence?.[el.id]?.value) && (
                            <div>
                              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">{fw.label} evidence</div>
                              <div className="grid md:grid-cols-2 gap-2.5">
                                {fw.elements.map(el => {
                                  const pe = row.picoEvidence?.[el.id];
                                  if (!pe || !pe.value) return null;
                                  return (
                                    <div key={el.id} className="rounded-lg border bg-card p-3 shadow-sm space-y-1.5">
                                      <div className="flex items-center justify-between gap-2">
                                        <span className="text-sm font-medium">{el.label}</span>
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
                {pagedFt.hasMore && (
                  <tr><td colSpan={3 + movableCols.length} className="p-0">
                    <ShowMore shown={pagedFt.shown} total={pagedFt.total} step={pagedFt.step} onMore={pagedFt.showMore} onAll={pagedFt.showAll} />
                  </td></tr>
                )}
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

// ---- Export ---------------------------------------------------------------
//
// Layout and values only. Everything about how the workbook LOOKS lives in
// lib/exportStyle, shared with abstract screening so the two cannot drift.

import ExcelJS from "exceljs";
import {
  XL, ExportCol, tidyEvidence, quoteOnce, rowHeightFor, writeGroupHeader,
  styleDataRow, paintVerdict, markReviewerEdit, addSummarySheet,
  saveBlob, saveCsv, stamp,
} from "../lib/exportStyle";

export { tidyEvidence, rowHeightFor };

const PICO_MATCH_FILL: Record<string, string> = {
  yes:     XL.greenTint,
  partial: XL.amberTint,
  no:      XL.redTint,
};
const PICO_MATCH_TEXT: Record<string, string> = {
  yes:     XL.green,
  partial: XL.amber,
  no:      XL.red,
};

/** The column layout, shared by both formats so the two exports cannot drift. */
export function exportColumns(criteria: string[], framework: FrameworkId): ExportCol[] {
  const els = frameworkOf(framework).elements;
  return [
    { header: "Decision",         key: "decision",    width: 12, group: "Decision" },
    { header: "AI decision",      key: "ai_decision", width: 12, group: "Decision" },
    { header: "Reviewer edit",    key: "rev_edit",    width: 13, group: "Decision" },
    { header: "Title",            key: "title",       width: 52, group: "Article" },
    { header: "Source",           key: "source",      width: 14, group: "Article" },
    { header: "URL",              key: "url",         width: 34, group: "Article" },
    { header: "Reason",           key: "reason",      width: 60, group: "Assessment" },
    { header: "Inclusion met",    key: "inc",         width: 13, group: "Assessment" },
    { header: "Excl. violations", key: "exc",         width: 15, group: "Assessment" },
    ...els.flatMap(el => {
      const h = framework === "pico" ? el.letter : el.label;
      const group = el.label;
      return [
        { header: `${h} · match`,    key: `${el.id}_match`,    width: 11, group },
        { header: `${h} · value`,    key: `${el.id}_value`,    width: 30, group },
        { header: `${h} · evidence`, key: `${el.id}_evidence`, width: 52, group },
      ];
    }),
    ...criteria.map((c, i) => ({ header: c, key: `crit_${i}`, width: 40, group: "Criteria" })),
  ];
}

/** One row's worth of values, keyed by column. Shared by XLSX and CSV. */
export function exportRow(
  r: FullTextResult,
  criteria: string[],
  overrides: Record<string, "Include" | "Exclude">,
  framework: FrameworkId,
): Record<string, any> {
  const els = frameworkOf(framework).elements;
  const pe = (r.picoEvidence || {}) as any;
  const overrideVal = overrides[r.paper_id];
  const effective = overrideVal ?? r.Decision;
  const out: Record<string, any> = {
    decision:    effective,
    ai_decision: r.Decision,
    rev_edit:    overrideVal && overrideVal !== r.Decision ? "yes" : "",
    title:       r.Title,
    source:      r.Source,
    url:         r.URL,
    reason:      tidyEvidence(r.Reason || ""),
    inc:         r.inclusion_score,
    exc:         r.exclusion_violations,
  };
  for (const el of els) {
    const cell = pe[el.id];
    out[`${el.id}_match`]    = cell?.match || "";
    out[`${el.id}_value`]    = tidyEvidence(cell?.value || "");
    out[`${el.id}_evidence`] = tidyEvidence(cell?.evidence || "");
  }
  criteria.forEach((c, i) => {
    const vote = r.criteriaEval?.[c] ?? "";
    const ev = r.criteriaEvidence?.[c];
    // Newlines so wrapText puts the vote, the quote and the reasoning on
    // separate visual lines inside one cell.
    const parts = [vote, quoteOnce(ev?.evidence || ""), tidyEvidence(ev?.reasoning || "")];
    out[`crit_${i}`] = parts.filter(Boolean).join("\n\n");
  });
  return out;
}

type ExportMeta = {
  framework: FrameworkId;
  pico: Record<string, string>;
  inclusion: string[];
  exclusion: string[];
};

export async function downloadFullTextXlsx(
  rows: FullTextResult[],
  criteria: string[],
  overrides: Record<string, "Include" | "Exclude"> = {},
  framework: FrameworkId = "pico",
  meta?: ExportMeta,
) {
  const cols = exportColumns(criteria, framework);
  const wb = new ExcelJS.Workbook();
  wb.creator = "Evidence Engine";
  wb.created = new Date();

  if (meta) {
    const included = rows.filter(r => (overrides[r.paper_id] ?? r.Decision) === "Include").length;
    const edited = rows.filter(r => overrides[r.paper_id] && overrides[r.paper_id] !== r.Decision).length;
    const els = frameworkOf(meta.framework).elements.filter(el => (meta.pico?.[el.id] || "").trim());
    addSummarySheet(wb, "Full-text screening", [
      { heading: "Result", pairs: [
        ["Articles screened", String(rows.length)],
        ["Included", String(included)],
        ["Excluded", String(rows.length - included)],
        ["Reviewer edits to AI decisions", String(edited)],
      ] },
      { heading: "Review question",
        pairs: els.length ? els.map(el => [el.label, meta.pico[el.id]] as [string, string])
                          : [["Not set", ""]] },
      { heading: "Inclusion criteria",
        pairs: meta.inclusion.length ? meta.inclusion.map((c, i) => [String(i + 1), c] as [string, string])
                                     : [["None set", ""]] },
      { heading: "Exclusion criteria",
        pairs: meta.exclusion.length ? meta.exclusion.map((c, i) => [String(i + 1), c] as [string, string])
                                     : [["None set", ""]] },
    ]);
  }

  const ws = wb.addWorksheet("Full-text screening", {
    // Freeze the decision block and both header rows, so the verdict stays
    // beside the evidence however far right you scroll.
    views: [{ state: "frozen", xSplit: 3, ySplit: 2 }],
  });
  ws.columns = cols.map(c => ({ key: c.key, width: c.width }));
  writeGroupHeader(ws, cols);

  rows.forEach((r, idx) => {
    const values = exportRow(r, criteria, overrides, framework);
    const row = ws.addRow(values);
    styleDataRow(row, cols, idx % 2 === 1);

    const overrideVal = overrides[r.paper_id];
    const effective = overrideVal ?? r.Decision;
    // Every column in the Decision group carries its verdict's colour, so the
    // reviewer's call and the model's can be compared without reading them.
    paintVerdict(row.getCell("decision"), effective === "Include", true);
    paintVerdict(row.getCell("ai_decision"), r.Decision === "Include", false);
    if (overrideVal && overrideVal !== r.Decision) {
      markReviewerEdit(row.getCell("decision"), row.getCell("rev_edit"));
    }

    for (const el of frameworkOf(framework).elements) {
      const cell = row.getCell(`${el.id}_match`);
      const v = String(cell.value || "").toLowerCase();
      if (v && PICO_MATCH_FILL[v]) {
        cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: PICO_MATCH_FILL[v] } };
        cell.font = { size: 10, bold: true, color: { argb: PICO_MATCH_TEXT[v] } };
        cell.alignment = { vertical: "top", horizontal: "center", wrapText: false };
      }
    }

    criteria.forEach((_, ci) => {
      const cell = row.getCell(`crit_${ci}`);
      const firstLine = String(cell.value || "").split("\n", 1)[0].toUpperCase();
      if (firstLine === "INCLUDE") {
        cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: XL.greenTint } };
      } else if (firstLine === "EXCLUDE") {
        cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: XL.redTint } };
      }
    });

    if (r.URL) {
      const urlCell = row.getCell("url");
      urlCell.value = { text: r.URL, hyperlink: r.URL };
      urlCell.font = { size: 10, color: { argb: XL.blue }, underline: true };
    }

    // Tall enough for the longest cell in the row, rather than a fixed height.
    row.height = rowHeightFor(values, cols);
  });

  ws.autoFilter = { from: { row: 2, column: 1 }, to: { row: 2, column: cols.length } };

  const buf = await wb.xlsx.writeBuffer();
  saveBlob(
    new Blob([buf], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" }),
    `fulltext-screening-${stamp()}.xlsx`,
  );
}

/** The same table as plain CSV, for R, Python, or a reviewer without Excel. */
export function downloadFullTextCsv(
  rows: FullTextResult[],
  criteria: string[],
  overrides: Record<string, "Include" | "Exclude"> = {},
  framework: FrameworkId = "pico",
) {
  const cols = exportColumns(criteria, framework);
  saveCsv(cols, rows.map(r => exportRow(r, criteria, overrides, framework)),
          `fulltext-screening-${stamp()}.csv`);
}

/** One download control, two formats. Which one you want depends on where the
 *  results are going, so the choice belongs in the menu rather than as two
 *  buttons competing for the same slot in a crowded toolbar. */
function ExportMenu({
  rows, criteria, overrides, framework, meta, className,
}: {
  rows: FullTextResult[];
  criteria: string[];
  overrides: Record<string, "Include" | "Exclude">;
  framework: FrameworkId;
  meta: ExportMeta;
  className?: string;
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button size="sm" variant="ghost" className={className} title="Download the results">
          <Download className="size-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-72">
        <DropdownMenuLabel className="text-xs font-normal text-muted-foreground">
          {rows.length} {rows.length === 1 ? "article" : "articles"}
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onClick={() => downloadFullTextXlsx(rows, criteria, overrides, framework, meta)}
          className="flex-col items-start gap-0.5 py-2">
          <span className="font-medium">Excel</span>
          <span className="text-xs text-muted-foreground">
            Formatted, with a summary sheet of the question and criteria
          </span>
        </DropdownMenuItem>
        <DropdownMenuItem
          onClick={() => downloadFullTextCsv(rows, criteria, overrides, framework)}
          className="flex-col items-start gap-0.5 py-2">
          <span className="font-medium">CSV</span>
          <span className="text-xs text-muted-foreground">
            Plain text, for R, Python or another tool
          </span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
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
