import { Fragment, useState, useEffect, useMemo } from "react";
import { useStore, resolveInUseRun, ScreeningPlan } from "../lib/store";
import { AIService, DataAggregator, Deduplicator, formatDuration, Paper, ScreenResult, PicoVote, PicoFieldAssessment } from "../lib/mockServices";
import { frameworkOf, labelFor, type FrameworkId } from "../lib/frameworks";
import { categoriseAbstractExclusion, effectiveAbstractDecision } from "../lib/exclusionBucketing";
import { ProjectScreeningBar, recordProjectDecision } from "../components/ProjectScreeningBar";
import { Checkbox } from "../components/ui/checkbox";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { PopoverContent, PopoverTrigger, PopoverClose, ScrollAwarePopover } from "../components/ui/popover";
import { Search, Download, Minus, Zap, ChevronRight, ChevronDown, Maximize2, Minimize2, Plus, RotateCcw, History, X, Check, FileText, CheckCircle2, XCircle, Clock, GripVertical, Sparkles, Pencil, CircleDashed } from "lucide-react";
import { EmptyState } from "../components/EmptyState";
import { CriteriaColumns } from "../components/CriteriaColumns";
import {
  DropdownMenu, DropdownMenuTrigger, DropdownMenuContent,
  DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator,
} from "../components/ui/dropdown-menu";
import { RapidScreen } from "../components/RapidScreen";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";
import { ControlPane, InlineStat, PaneDivider } from "../components/ControlPane";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from "../components/ui/alert-dialog";

// ---- styling helpers -------------------------------------------------------

function picoVoteClass(v: PicoVote): string {
  switch (v) {
    case "PASS":    return "bg-emerald-100 text-emerald-800 border-emerald-200";
    case "PARTIAL": return "bg-amber-100 text-amber-800 border-amber-200";
    case "FAIL":    return "bg-rose-100 text-rose-800 border-rose-200";
    case "NA":      return "bg-slate-100 text-slate-600 border-slate-200";
    default:        return "bg-slate-100 text-slate-600 border-slate-200";
  }
}

function picoVoteShort(v: PicoVote): string {
  // Compact label so the column cells stay narrow. NA is intentionally not
  // mapped here. NA cells render as an empty placeholder via PicoCell, not
  // as a chip.
  switch (v) {
    case "PASS":    return "✓";
    case "PARTIAL": return "~";
    case "FAIL":    return "✗";
    default:        return "";
  }
}

function picoVoteFullLabel(v: PicoVote): string {
  switch (v) {
    case "PASS":    return "Match";
    case "PARTIAL": return "Partial";
    case "FAIL":    return "Mismatch";
    default:        return "Not assessed";
  }
}

// ---- page ------------------------------------------------------------------


/** What is about to be screened, shown before the run starts.
 *
 *  Screening is the expensive, irreversible-feeling step: one model call per
 *  article against criteria that were mostly inferred rather than typed. The
 *  count alone ("34 of 34 unique articles ready") did not help anyone check
 *  anything, because the interesting question is not how many but WHICH
 *  articles and against WHAT. Both are visible here while it is still free to
 *  go back and fix the query or the criteria.
 */
export function ScreeningPreview({
  papers, excludedIds, framework, pico, inclusion, exclusion,
  plan, activeQuery = "", countedQuery = null,
}: {
  papers: Paper[];
  excludedIds: Set<string>;
  framework: FrameworkId;
  pico: Record<string, string>;
  inclusion: string[];
  exclusion: string[];
  /** The Planning run this corpus came from, when there was one. */
  plan?: ScreeningPlan | null;
  activeQuery?: string;
  /** The query the planned counts were measured against, when known. */
  countedQuery?: string | null;
}) {
  const toScreen = papers.filter(p => !excludedIds.has(p.id));

  if (toScreen.length === 0) return null;

  return (
    <Card className="p-0 overflow-hidden">
      {/* Where the records came from, and where they went: planned vs
          retrieved per source, reconciled to the unique count. A
          planned/retrieved mismatch is the signal that the curation did not
          track over -- a source erroring out, or a query the target database
          interpreted differently. It belongs here rather than in a separate
          card because it answers the same question as the rest of this panel:
          is this the set I meant to screen? */}
      {plan && (
        <div>

          <div className="flex items-center justify-between px-4 py-2 border-b bg-muted/10">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Retrieval per database</span>
            <span className="text-[11px] text-muted-foreground">
              {plan.runLabel ? `${plan.runLabel} · ` : ""}retrieved {new Date(plan.ranAt).toLocaleString()}
            </span>
          </div>
          <div className="divide-y">
            {plan.sources.map(src => {
              const planned = plan.counts?.[src];
              const got = plan.retrieved?.[src] ?? 0;
              // Only a MATERIAL shortfall is worth a warning. Any strict
              // inequality fired on 1,379 vs 1,378, which is noise.
              const short = planned != null && planned > 0 && got < planned * 0.9 && planned - got >= 5;
              const st = plan.sourceStatus?.[src];
              const q = plan.perDbQueries[src] || "";
              // Compare the strings that were ACTUALLY used, per source. The
              // count is measured per source and the retrieval runs per source,
              // so a unified-query comparison could not see a per-database edit:
              // it passed silently and the shortfall got blamed on the database.
              // Plans recorded before this fall back to the unified comparison.
              const counted = plan.countedPerDb?.[src];
              const stalePlan = counted !== undefined
                ? !!counted && !!q && counted !== q
                : !!countedQuery && !!activeQuery && countedQuery !== activeQuery;
              // "Capped" means the source had MORE than we asked for. Asking for
              // a source's full planned yield and receiving it is a complete
              // result, not a truncated one; reporting that as a cap told the
              // reviewer to raise a limit that was never binding.
              const unplanned = planned == null || planned <= 0;
              const truncatedHere = !!st?.capped && (unplanned || st.requested < planned);
              return (
                <div key={src} className="px-3 py-2">
                  <div className="flex items-center justify-between gap-2 text-xs">
                    <span className="font-medium">{src}</span>
                    <span className="tabular-nums text-muted-foreground">
                      planned {planned != null ? planned.toLocaleString() : "—"} · retrieved{" "}
                      <span className={short ? "text-amber-600 font-medium" : ""}>{got.toLocaleString()}</span>
                    </span>
                  </div>
                  {/* State what happened. The app performed this call and knows
                      the outcome, so offering the reviewer a list of things that
                      might have gone wrong was a failure to report rather than a
                      genuine ambiguity. Ordered by how actionable each case is. */}
                  {st?.status === "error" ? (
                    <div className="mt-1 text-[11px] text-rose-600">
                      {src} failed after {st.ms.toLocaleString()} ms: {st.error || st.error_type || "unknown error"}.
                      Nothing was retrieved from this source. Retry retrieval to try again.
                    </div>
                  ) : truncatedHere ? (
                    <div className="mt-1 text-[11px] text-amber-600">
                      Stopped at {st!.requested.toLocaleString()} records, and this source had more.
                      {unplanned
                        ? " It was never counted on the Planning tab, so there was no yield to retrieve against. Re-run Planning to count it."
                        : " Re-run Planning to refresh its yield."}
                    </div>
                  ) : short && stalePlan ? (
                    <div className="mt-1 text-[11px] text-amber-600">
                      The planned count for {src} was measured on a different query from the one
                      this search ran, so the two numbers do not describe the same search. Re-run
                      Planning, then retry retrieval.
                    </div>
                  ) : short ? (
                    <div className="mt-1 text-[11px] text-amber-600">
                      {src} returned {got.toLocaleString()} of the {planned?.toLocaleString()} planned
                      {st ? ` in ${st.ms.toLocaleString()} ms` : ""}, without erroring, on the same
                      query the count was taken on. Its index has changed since Planning ran.
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        </div>
      )}

      <CriteriaColumns
        framework={framework} pico={pico} inclusion={inclusion} exclusion={exclusion}
        emptyHint="No criteria set. Define the question on the Home page first, or screening has nothing to judge each article against."
      />
    </Card>
  );
}

export function AbstractPage() {
  const s = useStore();
  // Screening retrieves with the planning run that is IN USE, never with the
  // query editor's current text. Planning measures a count per source and
  // screening retrieves per source, so unless both read the same string a
  // planned-vs-retrieved gap says nothing about the database: it is an artefact
  // of the app comparing two different searches. Defaults to the most recent
  // run; the reviewer can pin an older one on the Planning tab.
  const planRun = resolveInUseRun(s.simulationRuns, s.inUseRunId);
  const planQueries = planRun ? planRun.perDbQueries : s.perDbQueries;
  const planCounts = planRun ? planRun.counts : s.simulation;
  const planBase = planRun?.unifiedQuery || s.unifiedSearchQuery || s.query || "";
  /** The exact string each source is searched with, frozen alongside its count. */
  const queryFor = (src: string) => planQueries[src] || planBase;

  /** One plan record for every retrieval path, so the provenance of a screened
   *  corpus is written the same way whether it came from AI screening, manual
   *  screening, or a retry. */
  const buildPlan = (
    planSources: string[],
    sourceCounts: Record<string, number>,
    sourceStatus: Record<string, any>,
  ) => ({
    sources: planSources,
    perDbQueries: Object.fromEntries(planSources.map(src => [src, queryFor(src)])),
    counts: planCounts ? { ...planCounts } : null,
    countedQuery: planRun?.unifiedQuery ?? s.simulationQuery ?? null,
    // The counts and the retrieval now come from one run, so this records the
    // per-source string both were measured on rather than a unified query that
    // may never have been sent to anything.
    countedPerDb: Object.fromEntries(planSources.map(src => [src, queryFor(src)])),
    runId: planRun?.id ?? null,
    runLabel: planRun?.label ?? null,
    retrieved: { ...sourceCounts },
    sourceStatus: { ...sourceStatus },
    ranAt: new Date().toISOString(),
  });

  const task = s.tasks["abstract-screen"];
  const running = task?.status === "running";
  // Retrieval is its own task so a retry never masquerades as a screening run.
  const fetchTask = s.tasks["retrieval"];
  const fetching = fetchTask?.status === "running";

  // mode "full" re-screens the whole corpus and replaces the results; mode
  // "incremental" only screens papers not already in results (e.g. records added
  // later from citation snowballing), leaving existing decisions + overrides
  // untouched. Either way the pass is archived so prior runs can be restored.
  const [retrying, setRetrying] = useState(false);

  // What the pane counts. "Ready to screen" is the number that will actually
  // cost a model call, so Quality-Assessment exclusions come off it.
  const readyToScreen = (s.uniquePapers || []).filter(p => !s.excludedByQuality.has(p.id)).length;
  const totalRetrieved = s.rawPapers?.length ?? s.uniquePapers?.length ?? 0;

  async function runSearch(mode: "full" | "incremental" = "full") {
    if (!s.query) { toast.error("Define a research goal on the Home page first."); return; }

    const { abort } = s.startTask("abstract-screen", [
      { id: "fetch",  label: "Fetching articles",  status: "pending" },
      { id: "screen", label: "Screening articles", status: "pending" },
    ]);
    const signal = abort.signal;
    const start = Date.now();

    // Decide where the paper queue comes from:
    //   1. If Quality Assessment has already produced uniquePapers, use those
    //      (and honour any QA exclusions).
    //   2. Otherwise fetch + deduplicate inline, so screening doesn't require
    //      the QA step.
    let queue: Paper[] = [];

    // The cached corpus is only reusable while it still reflects the CURRENT plan.
    const planDrift = planDriftForCorpus();
    const corpusIsCurrent = planDrift.length === 0;
    if (!corpusIsCurrent) {
      toast.info(`Planning changed (${planDrift.join("; ")}). Re-fetching so screening matches the current plan.`);
    }

    try {
      // Reuse the existing deduplicated set only if it still has something to
      // screen AND still matches the current plan. If it's empty or every paper
      // was excluded by the Quality step, fall through and re-fetch with the
      // in-use planning run's query instead of dead-ending (an empty array is
      // truthy, which caused the misleading "all excluded" error).
      if (corpusIsCurrent && s.uniquePapers && s.uniquePapers.some(p => !s.excludedByQuality.has(p.id))) {
        queue = s.uniquePapers.filter(p => !s.excludedByQuality.has(p.id));
        s.updateTask("abstract-screen", {
          stages: [
            { id: "fetch", label: "Using existing article set from Quality Assessment", status: "done" },
            { id: "screen", label: "Screening articles", status: "running" },
          ],
        });
        if (queue.length === 0) {
          s.updateTask("abstract-screen", { status: "error", detail: "All articles were excluded in Quality Assessment." });
          toast.error("All articles were excluded in quality assessment.");
          return;
        }
      } else {
        s.updateTask("abstract-screen", {
          detail: "Fetching articles from databases…",
          stages: [
            { id: "fetch", label: "Fetching articles", status: "running" },
            { id: "screen", label: "Screening articles", status: "pending" },
          ],
        });
        // Search exactly the sources the Planning tab curated. Planning works on
        // `sources` minus "Local PDFs" (it has no query to simulate), so screening
        // must apply the same filter or it would fetch a source that was never
        // planned or counted.
        const planSources = s.sources.filter(x => x !== "Local PDFs");

        // A missing simulation is not a harmless default: fetchForScreening then
        // falls back to a fixed per-source budget and the curated yields are
        // silently ignored. Stop and say so instead.
        if (!planCounts || !Object.keys(planCounts).length) {
          s.updateTask("abstract-screen", {
            status: "error",
            detail: "No planning yields — run the simulation on the Planning tab first.",
          });
          toast.error(
            "Run the simulation on the Planning tab first, so screening uses your curated per-database queries and yields.",
          );
          return;
        }
        const unplanned = planSources.filter(src => !(src in (planCounts || {})));
        if (unplanned.length) {
          toast.warning(
            `${unplanned.join(", ")} ${unplanned.length === 1 ? "was" : "were"} never simulated on the Planning tab. ` +
            `Re-run the simulation so ${unplanned.length === 1 ? "its" : "their"} yield is curated too.`,
          );
        }

        // Pull the FULL planning corpus: every source's own query, up to its
        // planning yield, so all found papers are screened, not a small sample.
        const { papers: all, truncated, sourceCounts, sourceStatus } = await DataAggregator.fetchForScreening(
          planSources, planQueries, planBase, planCounts, s.pico, { signal },
        );
        if (signal.aborted) { s.updateTask("abstract-screen", { status: "canceled" }); return; }

        // Record which plan produced this corpus. PRISMA has to report the search
        // per database, and without this the link between the planning run and
        // the screened set exists nowhere.
        s.setScreeningPlan(buildPlan(planSources, sourceCounts, sourceStatus));
        if (truncated.length > 0) {
          toast.info(`Large result set: capped ${truncated.join(", ")} for screening.`);
        }
        // Screening is uncapped, so a broad query can legitimately queue tens of
        // thousands of records. Say so up front rather than letting the reviewer
        // discover it from a progress bar that barely moves.
        if (all.length > 1000) {
          toast.info(
            `Screening ${all.length.toLocaleString()} articles. This is a large run — ` +
            `progress is shown above and you can cancel at any time without losing prior results.`,
          );
        }
        const { unique, duplicates } = Deduplicator.run(all);
        // Persist so a later QA run can reuse this set, and so the PRISMA flow
        // has accurate counts even when the user skipped QA.
        s.setRawPapers(all);
        s.setUniquePapers(unique);
        s.setDuplicatesCount(duplicates.length);
        queue = unique;
        if (queue.length === 0) {
          s.updateTask("abstract-screen", { status: "error", detail: "No articles retrieved for this query." });
          toast.error("No articles retrieved for this query. Broaden it on the Home page.");
          return;
        }
        s.updateTask("abstract-screen", {
          stages: [
            { id: "fetch", label: `Fetched ${all.length} articles (${unique.length} unique)`, status: "done" },
            { id: "screen", label: "Screening articles", status: "running" },
          ],
        });
      }
    } catch (e: any) {
      if (signal.aborted) { s.updateTask("abstract-screen", { status: "canceled" }); return; }
      s.updateTask("abstract-screen", { status: "error", detail: e?.message || "Fetch failed" });
      return;
    }

    // Incremental: screen only papers not already in the existing results.
    const existing = s.results || [];
    const incremental = mode === "incremental" && existing.length > 0;
    if (incremental) {
      const seen = new Set(existing.map(x => x.paper_id));
      queue = queue.filter(p => !seen.has(p.id));
      if (queue.length === 0) {
        s.updateTask("abstract-screen", { status: "done" });
        toast.info("No new articles to screen. Everything in the corpus is already screened.");
        return;
      }
    }

    s.updateTask("abstract-screen", { progress: { done: 0, total: queue.length } });
    try {
      const reasons: Record<string, number> = {};
      // Screen with bounded concurrency, much faster on large sets. Cloud models
      // handle many parallel calls; a local Ollama serves fewer at once, so keep
      // the pool small there to avoid thrashing.
      const CONCURRENCY = /^(claude|gpt|gemini)/i.test(s.model) ? 8 : 3;
      const slots: (ScreenResult | undefined)[] = new Array(queue.length);
      let done = 0, next = 0;
      // Ground each screening decision in the in-use planning run's search
      // strategy (the query the reviewer optimised and marked "In use"), so
      // screening stays consistent with what the review is scoped to find.
      const planQuery = s.unifiedSearchQuery || Object.values(s.perDbQueries || {}).filter(Boolean).join(" | ") || "";
      const planContext = planQuery
        ? `Search strategy for this review (the in-use planning run) — keep screening consistent with its scope:\n${planQuery}`.slice(0, 4000)
        : "";
      const worker = async () => {
        while (!signal.aborted) {
          const i = next++;
          if (i >= queue.length) return;
          s.updateTask("abstract-screen", {
            progress: { done, total: queue.length, label: queue[i].title.slice(0, 80) },
            detail: queue[i].title.slice(0, 80),
          });
          try {
            const r = await AIService.screenPaperMultiAgent(queue[i], s.pico, s.inclusion, s.exclusion, signal, planContext);
            slots[i] = r;
            if (r.Decision === "EXCLUDE") {
              const bucket = categoriseAbstractExclusion(r, s.inclusion, s.exclusion);
              reasons[bucket] = (reasons[bucket] || 0) + 1;
            }
          } catch (e: any) {
            if (signal.aborted) return;
            console.error(`abstract-screen ${i + 1} failed:`, e?.message);
          }
          done += 1;
          s.updateTask("abstract-screen", { progress: { done, total: queue.length } });
        }
      };
      await Promise.all(Array.from({ length: Math.min(CONCURRENCY, queue.length) }, worker));
      const screened: ScreenResult[] = slots.filter((x): x is ScreenResult => Boolean(x));

      // On cancel, discard this partial run and keep whatever was already
      // loaded (the most recent completed screening), rather than overwriting it
      // with a half-finished set.
      if (signal.aborted) {
        s.updateTask("abstract-screen", { status: "canceled" });
        toast.info(existing.length
          ? `Canceled after ${screened.length}. Kept your previous screening (${existing.length}).`
          : `Canceled: ${screened.length} of ${queue.length} screened.`);
        return;
      }

      // Merge: incremental appends the newly screened records to the existing
      // set; full replaces it. Reviewer overrides live in a separate map keyed
      // by paper_id, so they are preserved across either path.
      const finalResults: ScreenResult[] = incremental ? [...existing, ...screened] : screened;

      // Recompute the exclusion breakdown over the FULL set so PRISMA stays
      // correct in incremental mode (where `reasons` only covers this pass).
      const finalReasons: Record<string, number> = {};
      finalResults.filter(x => x.Decision === "EXCLUDE").forEach(x => {
        const bucket = categoriseAbstractExclusion(x, s.inclusion, s.exclusion);
        finalReasons[bucket] = (finalReasons[bucket] || 0) + 1;
      });
      const totalExcluded = Object.values(finalReasons).reduce((a, b) => a + b, 0);
      const sourceCounts: Record<string, number> = {};
      (s.rawPapers || []).forEach(p => { sourceCounts[p.source] = (sourceCounts[p.source] || 0) + 1; });
      // PRISMA-S: record this search run (per-database query + records retrieved),
      // deduped so re-screening the same corpus does not add a duplicate entry.
      const logRows = Object.keys(sourceCounts).map(src => ({
        source: src,
        query: s.perDbQueries[src] || s.unifiedSearchQuery || s.query || "",
        count: sourceCounts[src],
      }));
      if (logRows.length) {
        const sig = logRows.map(r => `${r.source}|${r.query}|${r.count}`).sort().join("~");
        s.setSearchLog(prev => {
          const last = prev[prev.length - 1];
          const lastSig = last ? last.rows.map(r => `${r.source}|${r.query}|${r.count}`).sort().join("~") : "";
          if (sig === lastSig) return prev;
          return [...prev, { id: Date.now().toString(36), ranAt: new Date().toISOString(), rows: logRows }];
        });
      }
      s.setPrisma(p => ({
        ...p,
        identified: s.rawPapers?.length || finalResults.length,
        source_counts: sourceCounts,
        duplicates_removed: s.duplicatesCount,
        screened: finalResults.length,
        excluded_total: totalExcluded,
        exclusion_breakdown: finalReasons,
        included_final: finalResults.length - totalExcluded,
      }));
      s.setResults(finalResults);
      const durationSec = (Date.now() - start) / 1000;
      s.setScreeningDuration(durationSec);

      // Archive this pass so it can be compared or restored later.
      if (!signal.aborted) {
        s.setScreeningArchive(prev => [...prev, {
          id: Date.now().toString(36),
          ranAt: new Date().toISOString(),
          mode: incremental ? "incremental" : "full",
          model: s.model,
          screenedNow: screened.length,
          totalAfter: finalResults.length,
          included: finalResults.length - totalExcluded,
          excluded: totalExcluded,
          durationSec,
          results: finalResults,
        }]);
        s.updateTask("abstract-screen", { status: "done" });
        toast.success(`Screened ${screened.length} article${screened.length === 1 ? "" : "s"}${incremental ? " (new)" : ""} in ${formatDuration(durationSec)}`);
      }
    } catch (e: any) {
      s.updateTask("abstract-screen", { status: "error", detail: e?.message });
    }
  }

  // Manual screening without AI: build the corpus (fetching it if needed), seed
  // every record as not-yet-included, then open the rapid reviewer so AI is
  // never required to use the platform. Reviewer include/exclude decisions carry
  // downstream exactly like AI ones.

  // Is the cached corpus still the one the CURRENT plan describes? Removing a
  // database or editing a per-database query makes uniquePapers stale, and
  // screening it would contradict the search actually documented. Returns the
  // list of drifts (empty = still current).
  function planDriftForCorpus(): string[] {
    if (!s.uniquePapers?.length) return [];
    const now = s.sources.filter(x => x !== "Local PDFs");
    const drift: string[] = [];
    const plan = s.screeningPlan;
    if (plan) {
      const gone = plan.sources.filter(src => !now.includes(src));
      const added = now.filter(src => !plan.sources.includes(src));
      if (gone.length) drift.push(`${gone.join(", ")} removed`);
      if (added.length) drift.push(`${added.join(", ")} added`);
      const edited = now.filter(
        src => plan.perDbQueries[src] !== undefined && plan.perDbQueries[src] !== queryFor(src),
      );
      if (edited.length) drift.push(`${edited.join(", ")} query changed`);
      // Switching which planning run is in use changes every query at once.
      if (planRun && plan.runId && plan.runId !== planRun.id && !edited.length) {
        drift.push(`planning run changed to ${planRun.label}`);
      }
    } else {
      const orphans = Array.from(
        new Set(s.uniquePapers.map(p => p.source).filter(src => src && !now.includes(src))),
      );
      if (orphans.length) drift.push(`${orphans.join(", ")} no longer in the plan`);
    }
    return drift;
  }

  /** Re-fetch the corpus from the selected databases and record the plan.
   *
   *  Shared by the manual path and the Retry control. Retrying matters most
   *  when a source under-delivered: a transient API failure looks identical to
   *  a genuinely small result set, and the only way to tell them apart is to
   *  ask again. Screening is NOT run here, so a retry costs database calls
   *  rather than a model call per article. */
  async function refetchCorpus(): Promise<boolean> {
    if (!s.query) { toast.error("Define a research goal on the Home page first."); return false; }
    const planSources = s.sources.filter(x => x !== "Local PDFs");
    // A retrieval-only task id, NOT "abstract-screen": this path makes database
    // calls and no model calls, so labelling it as a screening run would hide
    // the control pane behind a progress card that claims articles are being
    // judged. The screening task stays whatever it was.
    const { abort } = s.startTask("retrieval", [
      { id: "fetch", label: "Retrieving from databases", status: "running" },
    ]);
    s.updateTask("retrieval", {
      detail: `Querying ${planSources.length} ${planSources.length === 1 ? "database" : "databases"}: ${planSources.join(", ")}`,
    });
    try {
      const { papers: all, sourceCounts, sourceStatus } = await DataAggregator.fetchForScreening(
        planSources, planQueries, planBase, planCounts, s.pico, { signal: abort.signal },
      );
      if (abort.signal.aborted) { s.updateTask("retrieval", { status: "canceled" }); return false; }
      // Retry is exactly when the per-source diagnostics matter, so the new
      // plan carries them rather than dropping back to "no status recorded".
      s.setScreeningPlan(buildPlan(planSources, sourceCounts, sourceStatus));
      const { unique, duplicates } = Deduplicator.run(all);
      s.setRawPapers(all); s.setUniquePapers(unique); s.setDuplicatesCount(duplicates.length);
      s.updateTask("retrieval", { status: "done" });
      return true;
    } catch (e: any) {
      s.updateTask("retrieval", { status: "error", detail: e?.message || "Fetch failed" });
      toast.error(`Search failed: ${e?.message?.slice(0, 100) || "unknown error"}`);
      return false;
    }
  }

  async function retrySearch() {
    setRetrying(true);
    try {
      const before = s.uniquePapers?.length ?? 0;
      if (await refetchCorpus()) {
        const after = s.uniquePapers?.length ?? 0;
        toast.success(after === before
          ? `Search re-run: ${after.toLocaleString()} unique articles, unchanged.`
          : `Search re-run: ${after.toLocaleString()} unique articles.`);
      }
    } finally {
      setRetrying(false);
    }
  }

  async function startManual() {
    const manualDrift = planDriftForCorpus();
    if (manualDrift.length) {
      toast.info(`Planning changed (${manualDrift.join("; ")}). Re-fetching so screening matches the current plan.`);
    }
    let corpus = manualDrift.length ? [] : (s.uniquePapers || []).filter(p => !s.excludedByQuality.has(p.id));
    if (!corpus.length) {
      if (!s.query) { toast.error("Define a research goal on the Home page first."); return; }
      const { abort } = s.startTask("abstract-screen", [{ id: "fetch", label: "Fetching papers", status: "running" }]);
      try {
        // Same source filter and provenance as the AI path, so a manually
        // screened corpus is traceable to its planning run too.
        const planSources = s.sources.filter(x => x !== "Local PDFs");
        const { papers: all, sourceCounts, sourceStatus } = await DataAggregator.fetchForScreening(
          planSources, planQueries, planBase, planCounts, s.pico, { signal: abort.signal },
        );
        if (abort.signal.aborted) { s.updateTask("abstract-screen", { status: "canceled" }); return; }
        s.setScreeningPlan(buildPlan(planSources, sourceCounts, sourceStatus));
        const { unique, duplicates } = Deduplicator.run(all);
        s.setRawPapers(all); s.setUniquePapers(unique); s.setDuplicatesCount(duplicates.length);
        corpus = unique.filter(p => !s.excludedByQuality.has(p.id));
        s.updateTask("abstract-screen", { status: "done" });
      } catch (e: any) {
        s.updateTask("abstract-screen", { status: "error", detail: e?.message || "Fetch failed" });
        return;
      }
    }
    if (!corpus.length) { toast.error("No papers available to screen."); return; }
    const results: ScreenResult[] = corpus.map(p => ({
      paper_id: p.id, Source: p.source, Title: p.title, URL: p.url, Abstract: p.abstract || "",
      Decision: "EXCLUDE", Reason: "", Agent_Trace: {},
    }));
    s.setResults(results);
    setRapidOpen(true);
    toast.info(`${results.length} articles ready. Include the relevant ones.`);
  }

  const r = s.results;
  // Stats reflect EFFECTIVE decisions (reviewer overrides applied) so a
  // reviewer flipping the Keep checkbox immediately updates the included /
  // excluded counts.
  const passed = r ? r.filter(x => effectiveAbstractDecision(x, s.abstractOverrides) === "INCLUDE") : [];
  const excluded = r ? r.filter(x => effectiveAbstractDecision(x, s.abstractOverrides) === "EXCLUDE") : [];

  // Papers in the corpus not yet screened (drives the "Screen new papers"
  // incremental action). Snowballed records already sit in results, so they are
  // correctly excluded from this count.
  const existingIds = new Set((r || []).map(x => x.paper_id));
  const availableCorpus = (s.uniquePapers || []).filter(p => !s.excludedByQuality.has(p.id));
  const newToScreen = availableCorpus.filter(p => !existingIds.has(p.id)).length;

  const [projectRefresh, setProjectRefresh] = useState(0);
  const [rapidOpen, setRapidOpen] = useState(false);
  const [confirmNew, setConfirmNew] = useState(false);
  const [maxOpen, setMaxOpen] = useState(false);
  const [tableQuery, setTableQuery] = useState("");
  const [histOpen, setHistOpen] = useState(false);
  const [histQuery, setHistQuery] = useState("");
  const [editingLabelId, setEditingLabelId] = useState<string | null>(null);
  const [labelDraft, setLabelDraft] = useState("");
  const startEditLabel = (id: string, current?: string) => { setEditingLabelId(id); setLabelDraft(current || ""); };
  const saveLabel = (id: string) => {
    const v = labelDraft.trim();
    s.setScreeningArchive(prev => prev.map(p => p.id === id ? { ...p, label: v || undefined } : p));
    setEditingLabelId(null);
  };

  // Non-frozen columns the reviewer can drag to reorder, driven by the ACTIVE
  // framework. PICO keys its element columns by their single letter (p/i/c/o) so
  // previously-saved layouts still load; other frames (e.g. PCC, whose two "C"
  // letters would collide) key by element id. Keep/Decision/Title stay pinned.
  const framework = s.framework;
  const fw = frameworkOf(framework);
  const ABS_MOVABLE_COLS: { id: string; label: string; align?: "center"; elementId?: string }[] = [
    { id: "source", label: "Source" },
    ...fw.elements.map(el => ({
      id: framework === "pico" ? el.letter.toLowerCase() : el.id,
      label: framework === "pico" ? el.letter : el.label,
      align: "center" as const,
      elementId: el.id,
    })),
    { id: "reasoning", label: "Reasoning" },
  ];

  // The order persists per browser and reconciles with the active framework's
  // column set: element columns from a different frame drop out, new ones append.
  const DEFAULT_COL_ORDER = ABS_MOVABLE_COLS.map(c => c.id);
  const [rawColOrder, setRawColOrder] = useState<string[]>(() => {
    try { const saved = JSON.parse(localStorage.getItem("ee:abstract-col-order") || "null"); return Array.isArray(saved) ? saved : []; } catch { return []; }
  });
  const colOrder = useMemo(() => {
    const kept = rawColOrder.filter(id => DEFAULT_COL_ORDER.includes(id));
    const added = DEFAULT_COL_ORDER.filter(id => !kept.includes(id));
    return [...kept, ...added];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rawColOrder, DEFAULT_COL_ORDER.join("|")]);
  useEffect(() => { try { localStorage.setItem("ee:abstract-col-order", JSON.stringify(colOrder)); } catch { /* ignore */ } }, [colOrder]);
  const [dragCol, setDragCol] = useState<string | null>(null);
  const [overCol, setOverCol] = useState<string | null>(null);
  const orderedCols = colOrder.map(id => ABS_MOVABLE_COLS.find(c => c.id === id)!).filter(Boolean);
  const moveCol = (from: string, to: string) => setRawColOrder(() => {
    const a = [...colOrder]; const fi = a.indexOf(from); const ti = a.indexOf(to);
    if (fi < 0 || ti < 0 || fi === ti) return colOrder;
    a.splice(fi, 1); a.splice(ti, 0, from); return a;
  });
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const toggleRow = (id: string) => setExpanded(prev => {
    const n = new Set(prev);
    n.has(id) ? n.delete(id) : n.add(id);
    return n;
  });

  // Client-side row filter for the results table (title / reason / decision).
  const tableQ = tableQuery.trim().toLowerCase();
  const visibleRows = r
    ? (tableQ
        ? r.filter(row => `${row.Title} ${row.Reason} ${row.Decision} ${effectiveAbstractDecision(row, s.abstractOverrides)}`.toLowerCase().includes(tableQ))
        : r)
    : [];

  return (
    <div className="space-y-4">
      {s.currentProjectId && r && (
        <ProjectScreeningBar
          stage="abstract"
          papers={r.map(x => ({ id: x.paper_id }))}
          refreshSignal={projectRefresh}
        />
      )}
      {!r && !s.uniquePapers && !s.query && (
        <EmptyState icon={Search} title="No research goal yet"
          description="Define your research question and search strategy on the Home page, then come back to screen titles and abstracts."
          action={{ label: "Go to Home", onClick: () => s.setPage("home"), icon: Search }} />
      )}
      {!r && !s.uniquePapers && s.query && (
        <Alert>
          <AlertDescription>
            Ready to screen against your PICO. Articles are fetched and deduplicated on demand. Quality Assessment is optional and can be run first from the sidebar.
          </AlertDescription>
        </Alert>
      )}


      {r && (
        <>
          {/* Control pane: counts + all screening actions in one place. */}
          <ControlPane
            stats={<>
              <InlineStat icon={FileText} value={r.length} label="Screened" />
              <InlineStat icon={CircleDashed} value={newToScreen} label="Unscreened" tone={newToScreen > 0 ? "amber" : "neutral"} />
              <PaneDivider />
              <InlineStat icon={CheckCircle2} value={passed.length} label="Included" tone="success" hint={r.length ? `${Math.round((passed.length / r.length) * 100)}%` : undefined} />
              <InlineStat icon={XCircle} value={excluded.length} label="Excluded" tone="danger" hint={r.length ? `${Math.round((excluded.length / r.length) * 100)}%` : undefined} />
              <PaneDivider />
              <InlineStat icon={Clock} value={formatDuration(s.screeningDuration)} label="Time" tone="neutral" />
            </>}
            actions={<>
              {running ? (
                <Button size="sm" variant="destructive" className="h-8 shadow-sm" onClick={() => s.cancelTask("abstract-screen")} title="Stop screening (keeps the current results)">
                  <X className="size-3.5 mr-1.5" />Cancel
                </Button>
              ) : (
                <>
                  <Button size="sm" className="h-8 shadow-sm"
                    onClick={() => setConfirmNew(true)}
                    disabled={newToScreen === 0}
                    title={newToScreen === 0 ? "No unscreened articles in the corpus" : `Screen ${newToScreen} article${newToScreen === 1 ? "" : "s"} not already screened, keeping existing decisions and overrides`}>
                    <Plus className="size-3.5 mr-1.5" />Screen new
                  </Button>
                  <Button size="sm" variant="outline" className="h-8" onClick={() => runSearch("full")}
                    title="Re-screen every article from scratch. The current screening is archived first, so you can restore it.">
                    <RotateCcw className="size-3.5 mr-1.5" />Re-screen all
                  </Button>
                </>
              )}
              <span className="mx-0.5 h-6 w-px bg-border" aria-hidden="true" />
              <Button size="sm" onClick={() => setRapidOpen(true)} className="h-8 shadow-sm bg-amber-400 hover:bg-amber-500 text-amber-950 border-amber-400 dark:bg-amber-500 dark:hover:bg-amber-400 dark:text-amber-950" title="Review records fast with the keyboard">
                <Zap className="size-3.5 mr-1.5" />Rapid screen
              </Button>
            </>}
          />

          {/* Live progress sits directly under the control pane while a re-run is
              in flight, so the running state reads top-down (stats → progress). */}
          {running && task && (
            <TaskProgressCard
              task={task}
              title="Abstract screening with per-PICO appraisal"
              onCancel={() => s.cancelTask("abstract-screen")}
            />
          )}

          {/* Run history: deliberately understated — a muted text toggle rather
              than a full card, collapsed by default so it stays out of the way. */}
          {s.screeningArchive.length > 0 && (
            <div>
              <button className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors" onClick={() => setHistOpen(o => !o)} title="Show previous screening passes">
                {histOpen ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
                <History className="size-3.5" />Run history
                <span className="opacity-70">({s.screeningArchive.length})</span>
              </button>
              {histOpen && (() => {
                // Systematic run number = chronological order (Run 1 = first pass).
                const passes = s.screeningArchive.map((p, i) => ({ ...p, runNo: i + 1 })).reverse();   // newest first
                // The pass currently loaded into the table is the one whose results
                // array IS s.results (restore sets that reference). Falls back to the
                // newest pass when no match (e.g. after a reload re-hydrates arrays).
                const activeId = passes.find(p => p.results === s.results)?.id ?? passes[0]?.id;
                const q = histQuery.trim().toLowerCase();
                const filtered = q
                  ? passes.filter(p => [
                      `run ${p.runNo}`, p.label || "", p.mode, p.model || "", new Date(p.ranAt).toLocaleString(),
                      `${p.screenedNow} screened`, `${p.totalAfter} total`,
                      `${p.included} included`, `${p.excluded} excluded`,
                    ].join(" ").toLowerCase().includes(q))
                  : passes;
                return (
                  <Card className="mt-2 p-0 overflow-hidden shadow-sm ring-1 ring-black/[0.02]">
                    {/* Search bar */}
                    <div className="flex items-center gap-2 px-3 py-2 border-b bg-muted/30">
                      <Search className="size-3.5 text-muted-foreground shrink-0" />
                      <input
                        value={histQuery}
                        onChange={e => setHistQuery(e.target.value)}
                        placeholder="Search passes by date, mode, or model…"
                        className="flex-1 bg-transparent text-xs outline-none placeholder:text-muted-foreground"
                      />
                      <span className="text-[11px] text-muted-foreground tabular-nums shrink-0">
                        {filtered.length}/{passes.length}
                      </span>
                      {histQuery && (
                        <button type="button" onClick={() => setHistQuery("")} className="text-muted-foreground hover:text-foreground shrink-0" title="Clear search">
                          <X className="size-3.5" />
                        </button>
                      )}
                    </div>
                    {/* Scrollable list */}
                    {/* Cap the visible list at ~4 passes; the rest scroll. */}
                    <div className="max-h-56 overflow-y-auto divide-y divide-border/60">
                      {filtered.length === 0 ? (
                        <div className="px-3 py-8 text-center text-xs text-muted-foreground">No passes match "{histQuery}".</div>
                      ) : filtered.map((pass) => {
                        const isActive = pass.id === activeId;
                        return (
                          <div key={pass.id} className="flex items-center gap-3 px-3 py-2.5 hover:bg-muted/40 transition-colors">
                            <span className={`shrink-0 w-16 text-center px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide ${pass.mode === "incremental" ? "bg-sky-100 text-sky-700 dark:bg-sky-950/40 dark:text-sky-300" : "bg-muted text-muted-foreground"}`}>
                              {pass.mode === "incremental" ? "Incr." : "Full"}
                            </span>
                            <div className="min-w-0 flex-1">
                              {/* Line 1: run name + editable nickname — one uniform size */}
                              <div className="flex items-center gap-1.5 min-w-0 text-xs leading-tight">
                                <span className="font-semibold text-foreground shrink-0">Run {pass.runNo}</span>
                                {editingLabelId === pass.id ? (
                                  <input
                                    autoFocus
                                    value={labelDraft}
                                    onChange={e => setLabelDraft(e.target.value)}
                                    onKeyDown={e => { if (e.key === "Enter") e.currentTarget.blur(); }}
                                    onBlur={() => saveLabel(pass.id)}
                                    placeholder="Add a nickname…"
                                    maxLength={40}
                                    className="flex-1 min-w-0 bg-transparent border-b border-primary/50 outline-none text-xs font-semibold text-primary placeholder:font-normal placeholder:text-muted-foreground/70"
                                  />
                                ) : pass.label ? (
                                  <button type="button" className="group inline-flex items-center gap-1 min-w-0 text-xs" onClick={() => startEditLabel(pass.id, pass.label)} title="Rename this run">
                                    <span className="shrink-0 text-muted-foreground/50">·</span>
                                    <span className="truncate font-semibold text-primary">{pass.label}</span>
                                    <Pencil className="size-3 shrink-0 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                                  </button>
                                ) : (
                                  <button type="button" className="inline-flex items-center gap-1 text-xs text-muted-foreground/70 hover:text-foreground shrink-0 transition-colors" onClick={() => startEditLabel(pass.id, "")} title="Add a nickname">
                                    <Pencil className="size-3" />Add name
                                  </button>
                                )}
                              </div>
                              {/* Line 2: when it ran, how long, and the model — muted, smaller */}
                              <div className="mt-1 flex items-center gap-1.5 text-[11px] text-muted-foreground">
                                <span className="shrink-0 tabular-nums">{new Date(pass.ranAt).toLocaleString()}</span>
                                <span className="shrink-0 text-muted-foreground/40">·</span>
                                <Clock className="size-3 shrink-0" />
                                <span className="tabular-nums shrink-0">{formatDuration(pass.durationSec)}</span>
                                {pass.model && <><span className="shrink-0 text-muted-foreground/40">·</span><span className="truncate">{pass.model}</span></>}
                              </div>
                            </div>
                            <div className="ml-auto flex items-center gap-3 shrink-0">
                              <div className="hidden sm:flex items-center gap-1.5 text-[11px] tabular-nums">
                                <span className="text-muted-foreground">{pass.screenedNow} screened</span>
                                <span className="text-muted-foreground/40">·</span>
                                <span className="font-medium text-emerald-600">{pass.included} incl</span>
                                <span className="text-muted-foreground/40">·</span>
                                <span className="font-medium text-rose-600">{pass.excluded} excl</span>
                              </div>
                              {isActive ? (
                                <span className="inline-flex items-center gap-1 h-6 px-2 text-[11px] font-medium text-primary rounded-full bg-primary/10">
                                  <CheckCircle2 className="size-3" />Current
                                </span>
                              ) : (
                                <Button size="sm" variant="outline" className="h-6 px-2 text-[11px] gap-1"
                                  onClick={() => {
                                    s.setResults(pass.results);
                                    toast.success(`Restored Run ${pass.runNo}${pass.label ? ` (“${pass.label}”)` : ""} · ${pass.included} included`);
                                  }}
                                  title="Load this archived pass into the table (reviewer overrides are kept)">
                                  <RotateCcw className="size-3" />Restore
                                </Button>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </Card>
                );
              })()}
            </div>
          )}

          <AlertDialog open={confirmNew} onOpenChange={setConfirmNew}>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Screen {newToScreen} new article{newToScreen === 1 ? "" : "s"}?</AlertDialogTitle>
                <AlertDialogDescription>
                  Only articles not already screened will be reviewed. Existing decisions and your overrides are kept.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={() => runSearch("incremental")}>Screen {newToScreen}</AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>

          <RapidScreen
            open={rapidOpen}
            onClose={() => setRapidOpen(false)}
            label="Rapid screen: abstracts"
            hideAI={r.every(x => !x.Reason)}
            items={r.map(x => ({
              id: x.paper_id, title: x.Title, source: x.Source, url: x.URL, text: x.Abstract,
              aiInclude: x.Decision === "INCLUDE", reason: x.Reason,
              override: s.abstractOverrides[x.paper_id]
                ? (s.abstractOverrides[x.paper_id] === "INCLUDE" ? "include" : "exclude")
                : undefined,
            }))}
            onDecide={(id, d) => s.setAbstractOverride(id, d === "include" ? "INCLUDE" : "EXCLUDE")}
          />

          <Card className="p-4">
            <div className={maxOpen ? "fixed inset-0 z-50 bg-background p-4 flex flex-col gap-2" : ""}>
              {maxOpen ? (
                <div className="flex items-center gap-3 shrink-0">
                  <h3 className="font-medium shrink-0">Screening Results ({r.length})</h3>
                  <div className="relative flex-1 min-w-0">
                    <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground pointer-events-none" />
                    <input value={tableQuery} onChange={e => setTableQuery(e.target.value)} placeholder="Filter results…"
                      className="w-full h-8 pl-8 pr-7 rounded-md border bg-background text-xs outline-none focus-visible:ring-1 focus-visible:ring-ring placeholder:text-muted-foreground" />
                    {tableQuery && <button type="button" onClick={() => setTableQuery("")} className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground" title="Clear filter"><X className="size-3.5" /></button>}
                  </div>
                  <div className="ml-auto flex items-center gap-1 shrink-0">
                    {tableQuery && <span className="text-[11px] text-muted-foreground tabular-nums">{visibleRows.length}/{r.length}</span>}
<AbstractExportMenu
                      rows={r} overrides={s.abstractOverrides} framework={s.framework}
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
                    <input value={tableQuery} onChange={e => setTableQuery(e.target.value)} placeholder="Filter results by title, reason, or decision…"
                      className="w-full h-8 pl-8 pr-7 rounded-md border bg-background text-xs outline-none focus-visible:ring-1 focus-visible:ring-ring placeholder:text-muted-foreground" />
                    {tableQuery && <button type="button" onClick={() => setTableQuery("")} className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground" title="Clear filter"><X className="size-3.5" /></button>}
                  </div>
                  <div className="ml-auto flex items-center gap-1 shrink-0">
                    {tableQuery && <span className="text-[11px] text-muted-foreground tabular-nums">{visibleRows.length}/{r.length}</span>}
                    <Button size="sm" variant="ghost" onClick={() => setMaxOpen(true)} className="h-7 px-2 text-muted-foreground hover:text-foreground" title="Expand the table to full screen">
                      <Maximize2 className="size-4" />
                    </Button>
<AbstractExportMenu
                      rows={r} overrides={s.abstractOverrides} framework={s.framework}
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
                    <th className="px-3 py-2 sticky left-0 bg-muted z-40 border-b border-r min-w-[60px] text-center" title="Check to keep, uncheck to drop">Keep</th>
                    <th className="px-3 py-2 sticky left-[60px] bg-muted z-40 border-b border-r min-w-[120px]">Decision</th>
                    <th className="px-3 py-2 sticky left-[180px] bg-muted z-40 border-b border-r min-w-[260px] max-w-[260px] shadow-[6px_0_8px_-6px_rgba(0,0,0,0.22)]">Title</th>
                    {orderedCols.map(col => (
                      <th
                        key={col.id}
                        draggable
                        onDragStart={() => setDragCol(col.id)}
                        onDragEnter={() => setOverCol(col.id)}
                        onDragOver={e => e.preventDefault()}
                        onDrop={e => { e.preventDefault(); if (dragCol) moveCol(dragCol, col.id); setDragCol(null); setOverCol(null); }}
                        onDragEnd={() => { setDragCol(null); setOverCol(null); }}
                        className={`group/th px-3 border-b cursor-grab active:cursor-grabbing select-none transition-colors ${col.align === "center" ? "text-center" : "whitespace-nowrap"} ${col.id === "reasoning" ? "min-w-[380px]" : ""} ${dragCol === col.id ? "opacity-40" : ""} ${overCol === col.id && dragCol && dragCol !== col.id ? "bg-primary/10" : ""}`}
                      >
                        <span className={`inline-flex items-center gap-1 ${col.align === "center" ? "justify-center" : ""}`}>
                          <GripVertical className="size-3 opacity-0 group-hover/th:opacity-40 transition-opacity" />{col.label}
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tableQuery && visibleRows.length === 0 && (
                    <tr><td colSpan={3 + orderedCols.length} className="px-3 py-10 text-center text-sm text-muted-foreground">No results match "{tableQuery}".</td></tr>
                  )}
                  {visibleRows.map(row => {
                    const pa = row.Pico_Assessment;
                    const eff = effectiveAbstractDecision(row, s.abstractOverrides);
                    const isOverridden = eff !== row.Decision;
                    const keep = eff === "INCLUDE";
                    const isOpen = expanded.has(row.paper_id);
                    return (
                      <Fragment key={row.paper_id}>
                      <tr className="border-b border-border/60 last:border-b-0 align-top bg-card hover:bg-muted transition-colors">
                        <td className="px-3 py-2 sticky left-0 z-20 border-r bg-inherit text-center">
                          <Checkbox
                            checked={keep}
                            onCheckedChange={(v) => {
                              const wantKeep = v === true;
                              // Local override (always, keeps in-memory state in sync).
                              if ((row.Decision === "INCLUDE") === wantKeep) {
                                s.clearAbstractOverride(row.paper_id);
                              } else {
                                s.setAbstractOverride(row.paper_id, wantKeep ? "INCLUDE" : "EXCLUDE");
                              }
                              // Project mode: persist as a per-reviewer decision.
                              // We record an explicit user decision (which doubles
                              // as an override if it differs from the AI call) so
                              // the multi-reviewer audit trail captures every
                              // human action.
                              if (s.currentProjectId) {
                                const aiDec = row.Decision === "INCLUDE" ? "include" : "exclude";
                                recordProjectDecision(s.currentProjectId, {
                                  paper_id: row.paper_id,
                                  stage: "abstract",
                                  decision: wantKeep ? "include" : "exclude",
                                  reason: row.Reason || "",
                                  per_pico_verdict: row.Pico_Assessment || null,
                                  ai_decision: aiDec,
                                  is_override: aiDec !== (wantKeep ? "include" : "exclude"),
                                }).then(() => setProjectRefresh(n => n + 1))
                                  .catch((e: any) => console.error("Project decision write failed:", e?.message));
                              }
                            }}
                            aria-label="Keep this article"
                          />
                        </td>
                        <td className="px-3 py-2 sticky left-[60px] z-20 border-r bg-inherit">
                          <DecisionCell value={eff} overridden={isOverridden} aiValue={row.Decision} />
                        </td>
                        <td className="px-3 py-2 sticky left-[180px] z-20 border-r min-w-[260px] max-w-[260px] bg-inherit shadow-[6px_0_8px_-6px_rgba(0,0,0,0.18)]">
                          <div className="flex items-start gap-1.5">
                            <button onClick={() => toggleRow(row.paper_id)} className="mt-0.5 shrink-0 text-muted-foreground hover:text-foreground" title={isOpen ? "Hide details" : "Show abstract and PICO detail"} aria-expanded={isOpen}>
                              {isOpen ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
                            </button>
                            <a href={row.URL} target="_blank" rel="noreferrer" className="hover:underline break-words">
                              {row.Title}
                            </a>
                          </div>
                        </td>
                        {orderedCols.map(col => {
                          if (col.id === "source") return <td key={col.id} className="px-3 py-2 bg-inherit"><Badge variant="outline">{row.Source}</Badge></td>;
                          if (col.id === "reasoning") return <td key={col.id} className="px-3 py-2 text-foreground/90 min-w-[380px] max-w-[560px] bg-inherit"><div className="line-clamp-4 leading-relaxed">{row.Reason}</div></td>;
                          // Frame-element column (PICO P/I/C/O or PCC Population/Concept/Context).
                          const elId = col.elementId!;
                          return <td key={col.id} className="px-3 py-2 text-center bg-inherit"><PicoCell label={labelFor(framework, elId)} field={pa?.[elId]} criterion={s.pico[elId]} /></td>;
                        })}
                      </tr>
                      {isOpen && (
                        <tr className="border-b border-border/60 bg-muted/30">
                          <td colSpan={3 + orderedCols.length} className="p-0">
                            <div className="sticky left-0 w-[min(920px,92vw)] p-4 space-y-4">
                              {/* Abstract */}
                              <div className="rounded-lg border bg-card p-3.5 shadow-sm">
                                <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">Abstract</div>
                                <p className="text-sm leading-relaxed whitespace-pre-wrap text-foreground/90">{row.Abstract || <span className="text-muted-foreground italic">No abstract available.</span>}</p>
                              </div>
                              {/* Per-element assessment for the active framework */}
                              {pa && (
                                <div>
                                  <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">Per-{fw.label} assessment</div>
                                  <div className="grid md:grid-cols-2 gap-2.5">
                                    {fw.elements.map(el => {
                                      const f = pa?.[el.id];
                                      const v = f?.vote ?? "NA";
                                      return (
                                        <div key={el.id} className="rounded-lg border bg-card p-3 shadow-sm space-y-1.5">
                                          <div className="flex items-center justify-between gap-2">
                                            <span className="text-sm font-medium">{el.label}</span>
                                            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium border ${picoVoteClass(v)}`}>
                                              {v === "NA" && <Minus className="size-3" />}{picoVoteFullLabel(v)}
                                            </span>
                                          </div>
                                          {f?.reasoning && <p className="text-xs text-muted-foreground leading-relaxed">{f.reasoning}</p>}
                                          {f?.evidence && <blockquote className="text-xs italic text-foreground/70 border-l-2 border-primary/30 pl-2 break-words">“{f.evidence}”</blockquote>}
                                        </div>
                                      );
                                    })}
                                  </div>
                                </div>
                              )}
                              {/* Reasoning — below the PICO assessment, colour-accented */}
                              {row.Reason && (
                                <div className="rounded-lg border border-primary/30 bg-primary/[0.04] p-3.5 shadow-sm">
                                  <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-primary mb-1.5">
                                    <Sparkles className="size-3" />Reasoning
                                  </div>
                                  <p className="text-sm leading-relaxed text-foreground/90">{row.Reason}</p>
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
      )}

      {/* First run (no results yet): no control pane exists, so the progress card
          shows here. Once results exist it renders under the control pane above. */}
      {!r && task && task.status === "running" && (
        <TaskProgressCard
          task={task}
          title="Abstract screening with per-PICO appraisal"
          onCancel={() => s.cancelTask("abstract-screen")}
        />
      )}

      {/* When results exist, the run/cancel actions live in the results-card
          toolbar. Only the first-run controls appear here. */}
      {s.query && !r && running && (
        <Button onClick={() => s.cancelTask("abstract-screen")} size="lg" variant="destructive" className="w-full">
          <X className="size-4 mr-2" />Cancel screening
        </Button>
      )}

      {/* Counts and every action in one bar, the same shape the other working
          tabs use. The run controls used to float free at the bottom of the
          page, below a card that already described what they would do. */}
      {s.query && !r && !running && (
        <ControlPane
          stats={<>
            <InlineStat icon={FileText} value={readyToScreen} label="Ready to screen" />
            {s.uniquePapers && <>
              <PaneDivider />
              <InlineStat icon={Search} value={totalRetrieved} label="Retrieved" />
              {s.duplicatesCount > 0 && (
                <InlineStat icon={Minus} tone="neutral" value={s.duplicatesCount} label="Duplicates" />
              )}
              {s.excludedByQuality.size > 0 && (
                <InlineStat icon={XCircle} tone="amber" value={s.excludedByQuality.size} label="Excluded" />
              )}
            </>}
          </>}
          actions={<>
            {s.uniquePapers && (
              <Button size="sm" variant="outline" className="h-8 shadow-sm" onClick={retrySearch} disabled={retrying}
                      title="Query the databases again. Costs database calls, not model calls.">
                <RotateCcw className={`size-3.5 mr-1.5 ${retrying ? "animate-spin" : ""}`} />
                {retrying ? "Retrying…" : "Retry retrieval"}
              </Button>
            )}
            {/* The same amber as Rapid screen in the results pane below. Both
                are the human-review action, both carry the Zap icon, and giving
                them one colour makes "review it yourself" recognisable wherever
                it appears rather than a different button each time. */}
            <Button size="sm" onClick={startManual}
                    className="h-8 shadow-sm bg-amber-400 hover:bg-amber-500 text-amber-950 border-amber-400 dark:bg-amber-500 dark:hover:bg-amber-400 dark:text-amber-950"
                    title="Fetch the articles and review them yourself, no AI required">
              <Zap className="size-3.5 mr-1.5" />Screen manually
            </Button>
            <Button size="sm" className="h-8 shadow-sm" onClick={() => runSearch("full")}>
              <Search className="size-3.5 mr-1.5" />Run AI screening
            </Button>
          </>}
        />
      )}

      {/* Retrieval progress. The control pane above stays mounted throughout,
          so the button that started this is still where it was, and the counts
          update in place when the fetch lands. */}
      {fetching && (
        <TaskProgressCard
          task={fetchTask}
          title="Retrieving from databases"
          onCancel={() => s.cancelTask("retrieval")}
        />
      )}

      {s.query && !r && !running && s.uniquePapers && (
        <ScreeningPreview
          papers={s.uniquePapers}
          excludedIds={s.excludedByQuality}
          framework={s.framework}
          pico={s.pico}
          inclusion={s.inclusion}
          exclusion={s.exclusion}
          plan={s.screeningPlan}
          activeQuery={s.unifiedSearchQuery || s.query}
          countedQuery={s.screeningPlan?.countedQuery ?? s.simulationQuery}
        />
      )}
    </div>
  );
}

// ---- per-PICO chip with evidence popover ----------------------------------
//
// NA ("not assessed") renders as a neutral slate chip with a minus glyph and a
// popover that explains *why*: either the PICO element was left blank in the
// frame, or the abstract lacked the information needed to judge it. This reads
// as an intentional state rather than a missing/incomplete value.

function PicoCell({ label, field, criterion }: { label: string; field?: PicoFieldAssessment; criterion?: string }) {
  const vote: PicoVote = field?.vote ?? "NA";
  const isNA = vote === "NA";
  const naReason = (field?.reasoning || "").trim()
    || ((criterion || "").trim()
      ? `The abstract did not give enough information to judge the ${label.toLowerCase()}.`
      : `No ${label.toLowerCase()} in your PICO frame, so there's nothing to assess against. Add one on the Home page to enable this check.`);
  return (
    <ScrollAwarePopover>
      <PopoverTrigger asChild>
        <button
          className={`inline-flex items-center justify-center size-6 rounded text-xs font-semibold border ${picoVoteClass(vote)} hover:ring-1 hover:ring-foreground/30`}
          title={`${label}: ${picoVoteFullLabel(vote)}`}
        >
          {isNA ? <Minus className="size-3.5" /> : picoVoteShort(vote)}
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-96 text-xs space-y-2" align="end">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">{label}</span>
          <div className="flex items-center gap-1.5">
            <span className={`px-2 py-0.5 rounded text-xs font-medium border ${picoVoteClass(vote)}`}>
              {picoVoteFullLabel(vote)}
            </span>
            <PopoverClose className="size-5 grid place-items-center rounded text-muted-foreground hover:bg-muted hover:text-foreground" title="Close">
              <X className="size-3.5" />
            </PopoverClose>
          </div>
        </div>
        {isNA ? (
          <div className="text-foreground/90">{naReason}</div>
        ) : (
          <>
            {field?.reasoning && <div className="text-foreground/90">{field.reasoning}</div>}
            {field?.evidence && (
              <blockquote className="border-l-2 border-primary/30 pl-2 italic text-muted-foreground break-words">
                "{field.evidence}"
              </blockquote>
            )}
          </>
        )}
      </PopoverContent>
    </ScrollAwarePopover>
  );
}


function DecisionCell({ value, overridden, aiValue }: { value: string; overridden?: boolean; aiValue?: string }) {
  // A result restored from an older session, or written by an interrupted run,
  // can be missing Decision entirely. An unguarded read here took the whole
  // screening table down rather than showing one incomplete row.
  const inc = String(value ?? "").toUpperCase().includes("INCLUDE");
  return (
    <div className="space-y-1">
      <span className={`inline-flex items-center gap-1 px-2 py-1 rounded font-medium text-xs ${inc ? "bg-green-100 text-green-800 dark:bg-emerald-950/50 dark:text-emerald-300" : "bg-red-100 text-red-800 dark:bg-rose-950/50 dark:text-rose-300"}`}>
        {inc ? <Check className="size-3.5" /> : <X className="size-3.5" />}{inc ? "Include" : "Exclude"}
      </span>
      {overridden && aiValue && (
        <div className="text-[10px] text-muted-foreground leading-tight">
          AI: {String(aiValue ?? "").toUpperCase().includes("INCLUDE") ? "Include" : "Exclude"} · reviewer-edited
        </div>
      )}
    </div>
  );
}

// ---- XLSX export ----------------------------------------------------------
//
// Produces a formatted Excel workbook (not a CSV) so reviewers can open the
// file straight into Excel / Google Sheets / Numbers with colour-coded
// decision cells, wrapped text in long quote columns, and frozen headers.

import ExcelJS from "exceljs";
import {
  XL, ExportCol, tidyEvidence, quoteOnce, rowHeightFor, writeGroupHeader,
  styleDataRow, paintVerdict, markReviewerEdit, addSummarySheet,
  saveBlob, saveCsv, stamp,
} from "../lib/exportStyle";

// Abstract screening votes are PASS / PARTIAL / FAIL / NA rather than the
// full-text yes / partial / no, but they mean the same thing and now wear the
// same colours.
const PICO_VOTE_FILL: Record<string, string> = {
  PASS:    XL.greenTint,
  PARTIAL: XL.amberTint,
  FAIL:    XL.redTint,
  NA:      XL.slateTint,
};
const PICO_VOTE_TEXT: Record<string, string> = {
  PASS:    XL.green,
  PARTIAL: XL.amber,
  FAIL:    XL.red,
  NA:      XL.slate,
};

export type AbstractExportMeta = {
  framework: FrameworkId;
  pico: Record<string, string>;
  inclusion: string[];
  exclusion: string[];
};

export function abstractColumns(framework: FrameworkId): ExportCol[] {
  const els = frameworkOf(framework).elements;
  return [
    { header: "Decision",      key: "decision",    width: 12, group: "Decision" },
    { header: "AI decision",   key: "ai_decision", width: 12, group: "Decision" },
    { header: "Reviewer edit", key: "rev_edit",    width: 13, group: "Decision" },
    { header: "Title",         key: "title",       width: 52, group: "Article" },
    { header: "Source",        key: "source",      width: 14, group: "Article" },
    { header: "URL",           key: "url",         width: 34, group: "Article" },
    { header: "Reasoning",     key: "reasoning",   width: 60, group: "Assessment" },
    ...els.flatMap(el => {
      const h = framework === "pico" ? el.letter : el.label;
      const group = el.label;
      return [
        { header: `${h} · vote`,      key: `${el.id}_vote`,   width: 11, group },
        { header: `${h} · quote`,     key: `${el.id}_quote`,  width: 52, group },
        { header: `${h} · reasoning`, key: `${el.id}_reason`, width: 44, group },
      ];
    }),
  ];
}

export function abstractRow(
  r: ScreenResult,
  overrides: Record<string, "INCLUDE" | "EXCLUDE">,
  framework: FrameworkId,
): Record<string, any> {
  const els = frameworkOf(framework).elements;
  const pa = (r.Pico_Assessment || {}) as any;
  const overrideVal = overrides[r.paper_id];
  const effective = overrideVal ?? r.Decision;
  const out: Record<string, any> = {
    decision:    effective,
    ai_decision: r.Decision,
    rev_edit:    overrideVal && overrideVal !== r.Decision ? "yes" : "",
    title:       r.Title,
    source:      r.Source,
    url:         r.URL,
    reasoning:   tidyEvidence(r.Reason || ""),
  };
  for (const el of els) {
    const f = pa?.[el.id];
    out[`${el.id}_vote`]   = f?.vote ?? "";
    out[`${el.id}_quote`]  = quoteOnce(f?.evidence ?? "");
    out[`${el.id}_reason`] = tidyEvidence(f?.reasoning ?? "");
  }
  return out;
}

export async function downloadAbstractXlsx(
  rows: ScreenResult[],
  overrides: Record<string, "INCLUDE" | "EXCLUDE"> = {},
  framework: FrameworkId = "pico",
  meta?: AbstractExportMeta,
) {
  const cols = abstractColumns(framework);
  const wb = new ExcelJS.Workbook();
  wb.creator = "Evidence Engine";
  wb.created = new Date();

  if (meta) {
    const included = rows.filter(r => (overrides[r.paper_id] ?? r.Decision) === "INCLUDE").length;
    const edited = rows.filter(r => overrides[r.paper_id] && overrides[r.paper_id] !== r.Decision).length;
    const els = frameworkOf(meta.framework).elements.filter(el => (meta.pico?.[el.id] || "").trim());
    addSummarySheet(wb, "Abstract screening", [
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

  const ws = wb.addWorksheet("Abstract screening", {
    views: [{ state: "frozen", xSplit: 3, ySplit: 2 }],
  });
  ws.columns = cols.map(c => ({ key: c.key, width: c.width }));
  writeGroupHeader(ws, cols);

  rows.forEach((r, idx) => {
    const values = abstractRow(r, overrides, framework);
    const row = ws.addRow(values);
    styleDataRow(row, cols, idx % 2 === 1);

    const overrideVal = overrides[r.paper_id];
    const effective = overrideVal ?? r.Decision;
    // Every column in the Decision group carries its verdict's colour, so the
    // reviewer's call and the model's can be compared without reading them.
    paintVerdict(row.getCell("decision"), effective === "INCLUDE", true);
    paintVerdict(row.getCell("ai_decision"), r.Decision === "INCLUDE", false);
    if (overrideVal && overrideVal !== r.Decision) {
      markReviewerEdit(row.getCell("decision"), row.getCell("rev_edit"));
    }

    for (const el of frameworkOf(framework).elements) {
      const cell = row.getCell(`${el.id}_vote`);
      const v = String(cell.value || "").toUpperCase();
      if (v && PICO_VOTE_FILL[v]) {
        cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: PICO_VOTE_FILL[v] } };
        cell.font = { size: 10, bold: true, color: { argb: PICO_VOTE_TEXT[v] } };
        cell.alignment = { vertical: "top", horizontal: "center", wrapText: false };
      }
    }

    if (r.URL) {
      const urlCell = row.getCell("url");
      urlCell.value = { text: r.URL, hyperlink: r.URL };
      urlCell.font = { size: 10, color: { argb: XL.blue }, underline: true };
    }

    row.height = rowHeightFor(values, cols);
  });

  ws.autoFilter = { from: { row: 2, column: 1 }, to: { row: 2, column: cols.length } };

  const buf = await wb.xlsx.writeBuffer();
  saveBlob(
    new Blob([buf], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" }),
    `abstract-screening-${stamp()}.xlsx`,
  );
}

/** One download control, two formats, the same shape full-text screening uses. */
function AbstractExportMenu({
  rows, overrides, framework, meta, className,
}: {
  rows: ScreenResult[];
  overrides: Record<string, "INCLUDE" | "EXCLUDE">;
  framework: FrameworkId;
  meta: AbstractExportMeta;
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
          onClick={() => downloadAbstractXlsx(rows, overrides, framework, meta)}
          className="flex-col items-start gap-0.5 py-2">
          <span className="font-medium">Excel</span>
          <span className="text-xs text-muted-foreground">
            Formatted, with a summary sheet of the question and criteria
          </span>
        </DropdownMenuItem>
        <DropdownMenuItem
          onClick={() => downloadAbstractCsv(rows, overrides, framework)}
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

export function downloadAbstractCsv(
  rows: ScreenResult[],
  overrides: Record<string, "INCLUDE" | "EXCLUDE"> = {},
  framework: FrameworkId = "pico",
) {
  const cols = abstractColumns(framework);
  saveCsv(cols, rows.map(r => abstractRow(r, overrides, framework)),
          `abstract-screening-${stamp()}.csv`);
}
