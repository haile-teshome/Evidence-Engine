import { useEffect, useMemo, useState } from "react";
import { usePagedRows } from "../lib/usePagedRows";
import { ShowMore } from "../components/ShowMore";
import { motion } from "motion/react";
import { useStore, type SnowballSort } from "../lib/store";
import { AIService, ScreenResult, Paper, DataAggregator } from "../lib/mockServices";
import { effectiveFullTextDecision } from "../lib/exclusionBucketing";
import { Card } from "../components/ui/card";
import { Popover, PopoverTrigger, PopoverContent } from "../components/ui/popover";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Checkbox } from "../components/ui/checkbox";
import { Input } from "../components/ui/input";
import { Network, Plus, Minus, Trash2, Search, Sparkles, FileText, ArrowDownLeft, ArrowUpRight, CheckCircle2, XCircle, ExternalLink, Layers, ChevronDown, ChevronsDownUp, ChevronsUpDown, FlaskConical, SlidersHorizontal, Check } from "lucide-react";
import { EmptyState } from "../components/EmptyState";
import { toast } from "sonner";
import { TaskProgressCard } from "../components/TaskProgressCard";
import { ControlPane, InlineStat, PaneDivider } from "../components/ControlPane";
import { ImportStudies } from "../components/ImportStudies";

type SnowballType = "Both" | "Backward (References)" | "Forward (Cited by)";

const DIRECTIONS: { value: SnowballType; label: string }[] = [
  { value: "Both", label: "Both" },
  { value: "Backward (References)", label: "Backward" },
  { value: "Forward (Cited by)", label: "Forward" },
];

// Seeds normalise to this shape whether they come from the full-text includes or
// (as a fallback when none passed) the whole uploaded/searched corpus.
type SnowballSeed = { paper_id: string; Title: string; Abstract: string; URL: string };

/** Short forms for the trigger button, where space is tight. */
const DIRECTION_SHORT: Record<SnowballType, string> = {
  "Both": "Both",
  "Backward (References)": "Backward",
  "Forward (Cited by)": "Forward",
};

const SORT_SHORT: Record<SnowballSort, string> = {
  relevance: "closest first",
  cited: "most cited",
  recent: "most recent",
  source: "source order",
};

const SORT_OPTIONS: { value: SnowballSort; label: string; help: string }[] = [
  { value: "relevance", label: "Closest to your question", help: "Ranked against your PICO, once across every seed" },
  { value: "cited", label: "Most cited", help: "Highest citation count first" },
  { value: "recent", label: "Most recent", help: "Newest publication year first" },
  { value: "source", label: "Source order", help: "Whatever order the database returned" },
];

/** One row per title. Snowballing reaches the same paper from many seeds. */
function dedupeByTitle(rows: any[]): any[] {
  const seen = new Set<string>();
  return rows.filter(p => {
    const k = (p.title || "").toLowerCase();
    if (!k || seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

/** Which citations survive the per-seed cap.
 *
 *  "source" preserves what the API returned, which is what the app used to do
 *  implicitly; the others state a rule the write-up can name. Ties and missing
 *  values fall back to the source order rather than shuffling, so a run is
 *  reproducible.
 */
export function sortCitations(rows: any[], sort: SnowballSort, scores?: Map<string, number> | null): any[] {
  if (sort === "source") return rows;
  const key = (p: any): number => {
    if (sort === "relevance") return scores?.get((p.title || "").toLowerCase()) ?? -Infinity;
    if (sort === "cited") return typeof p.cited_by_count === "number" ? p.cited_by_count : -Infinity;
    return typeof p.year === "number" ? p.year : -Infinity;      // "recent"
  };
  // A stable sort on the index keeps equal keys in the order the source gave.
  return rows
    .map((p, i) => ({ p, i, k: key(p) }))
    .sort((a, b) => (b.k - a.k) || (a.i - b.i))
    .map(x => x.p);
}

export function SnowballPage() {
  const s = useStore();
  const [type, setType] = useState<SnowballType>("Both");
  const [maxCit, setMaxCit] = useState(20);
  const [sort, setSort] = useState<SnowballSort>("relevance");
  // null = "All seeds"; otherwise the seed paper title whose citations are shown.
  const [selectedSeed, setSelectedSeed] = useState<string | null>(null);
  const [q, setQ] = useState("");
  // Citation ids the reviewer has chosen, read from the store so the selection
  // persists across tab switches instead of resetting on remount.
  const chosen = new Set(s.snowballChosen ?? []);
  const fetchTask = s.tasks["snowball"];
  const running = fetchTask?.status === "running";

  // Seed the selection from the AI's INCLUDE verdicts ONCE, when a fresh
  // screening run lands and nothing has been selected yet (snowballChosen is
  // null). After that the reviewer's checks/unchecks are authoritative and are
  // never overwritten on remount or re-render.
  useEffect(() => {
    const sc = s.snowballScreened, rs = s.snowballResults;
    if (!sc || !rs) return;
    if (s.snowballChosen != null) return;   // already seeded / has a manual selection
    const dmap = new Map<string, string>();
    sc.forEach(r => {
      if (r.paper_id) dmap.set(r.paper_id, r.Decision);
      if (r.Title) dmap.set(r.Title.toLowerCase().trim(), r.Decision);
    });
    const init: string[] = [];
    rs.forEach(p => {
      const d = dmap.get(p.id) || dmap.get((p.title || "").toLowerCase().trim());
      if (d === "INCLUDE") init.push(p.id);
    });
    s.setSnowballChosen(init);
  }, [s.snowballScreened, s.snowballResults, s.snowballChosen]);

  function toggleChosen(id: string) {
    s.setSnowballChosen(prev => {
      const next = new Set(prev ?? []);
      if (next.has(id)) next.delete(id); else next.add(id);
      return [...next];
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

  // Seeds are the studies KEPT at full-text evidence: AI-included plus any the
  // reviewer kept by overriding the decision. The kept studies are the only
  // seeds (no fallback to the wider uploaded corpus).
  const seeds: SnowballSeed[] = (s.fullTextResults ?? [])
    .filter(r => effectiveFullTextDecision(r, s.fullTextOverrides) === "Include")
    .map(r => ({ paper_id: r.paper_id, Title: r.Title, Abstract: r.Abstract, URL: r.URL }));

  // Above the bail-out, because a hook cannot run conditionally: starting a
  // new session empties the seeds under a mounted page, and skipping these on
  // that render is what React reports as "rendered fewer hooks than expected".
  const results = s.snowballResults;
  const shown = useMemo(() => (results || [])
    .filter(p => selectedSeed === null || p.seed_paper_title === selectedSeed)
    .filter(p => !q.trim() || (p.title || "").toLowerCase().includes(q.toLowerCase())), [results, selectedSeed, q]);
  const pagedShown = usePagedRows(shown);

  if (seeds.length === 0) {
    return <EmptyState icon={FlaskConical} title="No kept full-text studies yet" description="Keep at least one study at Full-Text Screening (include it, or check its Keep box). Snowballing expands from the kept studies." action={{ label: "Go to Full-Text Screening", onClick: () => s.setPage("fulltext"), icon: FlaskConical }}
      extra={<ImportStudies stage="citation-snowball" label="Add seed studies" />} />;
  }

  async function start() {
    const { abort } = s.startTask("snowball", [{ id: "snow", label: "Fetching citations", status: "running" }]);
    s.updateTask("snowball", { progress: { done: 0, total: seeds.length } });
    // Record the seed set this fetch runs over, so a later change to the included
    // set (e.g. re-running full-text) can flag the cached results as stale.
    s.setSnowballSeeds(seeds.map(x => x.Title));
    const signal = abort.signal;
    try {
      // Gather the WHOLE reference / citation list per seed, then rank and
      // truncate here. Letting each API decide by returning page one meant
      // keeping the papers printed first in a bibliography, which skew to
      // introduction and background rather than to the closest work.
      const perSeed: { title: string; rows: any[]; available: number }[] = [];
      let ceiling = 0;
      for (let i = 0; i < seeds.length; i++) {
        if (signal.aborted) break;
        const seed = seeds[i];
        s.updateTask("snowball", {
          progress: { done: i, total: seeds.length, label: seed.Title.slice(0, 80) },
          detail: seed.Title.slice(0, 80),
        });
        try {
          const haul = await AIService.fetchCitations(seed.Title, type, maxCit, s.sources, signal);
          haul.citations.forEach(c => { c.seed_paper_title = seed.Title; });
          perSeed.push({ title: seed.Title, rows: haul.citations, available: haul.totalAvailable });
          ceiling = haul.ceiling || ceiling;
        } catch (e: any) {
          if (signal.aborted) break;
          console.error(`snowball ${i + 1} failed:`, e?.message);
          perSeed.push({ title: seed.Title, rows: [], available: 0 });
        }
        s.updateTask("snowball", { progress: { done: i + 1, total: seeds.length } });
      }

      // Relevance is the only ordering that needs the model, and it runs ONCE
      // over the pooled set rather than per seed: the same paper deserves the
      // same score whichever seed reached it, and one pass is one model call
      // instead of ninety-five.
      let scores: Map<string, number> | null = null;
      if (sort === "relevance" && !signal.aborted) {
        const pool = dedupeByTitle(perSeed.flatMap(x => x.rows));
        if (pool.length) {
          // Ranking is its own phase and a long one, so it gets its own bar
          // rather than leaving the fetch bar sitting full. The fetch counted
          // seeds; this counts citations, which the estimate treats as a new
          // run because the count drops back to zero.
          s.updateTask("snowball", {
            detail: `Ranking ${pool.length.toLocaleString()} citations against your question…`,
            stages: [{ id: "snow", label: "Ranking by relevance", status: "running" }],
            progress: { done: 0, total: pool.length },
          });
          const rows = pool.map(p => ({
            id: p.id, title: p.title, abstract: p.abstract || "", source: p.source, url: p.url,
          })) as any;
          // The stream reports every article; a pool of thousands would re-render
          // the page as fast as the scores land. Five updates a second is more
          // than the eye needs, and the estimate reads the same either way: each
          // report carries the time it covers, however many articles that is.
          let reportedAt = 0;
          const report = (done: number, total: number) => {
            const at = Date.now();
            if (done < total && at - reportedAt < 200) return;
            reportedAt = at;
            s.updateTask("snowball", { progress: { done, total } });
          };
          try {
            let res;
            try {
              // Streamed, so the reviewer sees how far the ranking has got and
              // how long is left instead of a bar that stopped moving.
              res = await DataAggregator.rerankByRelevanceStream(
                rows, s.pico, s.inclusion, s.exclusion, -1, undefined, signal, undefined,
                report, "citation-snowball",
              );
            } catch (e: any) {
              if (signal.aborted || e?.name === "AbortError") throw e;   // a genuine cancel
              // No stream (older backend, or a proxy that buffers) → the
              // blocking call. Nothing to count, so the bar goes away rather
              // than freezing at zero.
              s.updateTask("snowball", { progress: { done: 0, total: 0 } });
              res = await DataAggregator.rerankByRelevance(
                rows, s.pico, s.inclusion, s.exclusion, -1, undefined, signal, undefined, "citation-snowball",
              );
            }
            scores = new Map(res.ranked.map(r => [(r.paper.title || "").toLowerCase(), r.leads_score]));
          } catch (e: any) {
            // A ranking failure must not lose the citations themselves.
            if (!signal.aborted) {
              console.error("snowball ranking failed:", e?.message);
              toast.warning("Could not rank by relevance; kept the source order instead.");
            }
          }
        }
      }

      const kept = perSeed.map(seed => ({
        ...seed,
        rows: sortCitations(seed.rows, sort, scores).slice(0, maxCit),
      }));
      const all = kept.flatMap(x => x.rows);
      const unique = dedupeByTitle(all);

      // Write down what was found against what was taken. Without this the
      // search cannot be reproduced from what the app records.
      s.setSnowballPlan({
        sort, maxPerSeed: maxCit, direction: type, sources: [...s.sources],
        seeds: kept.map((k, i) => ({
          title: k.title, available: perSeed[i].available,
          fetched: perSeed[i].rows.length, kept: k.rows.length,
        })),
        ceiling, ranAt: new Date().toISOString(),
      });
      if (signal.aborted) {
        // Save partial progress: merge the citations gathered this run into the
        // existing set (dedup by title) and keep the current selection.
        const seen = new Set<string>();
        const merged = [...(s.snowballResults ?? []), ...unique].filter(p =>
          seen.has((p.title || "").toLowerCase()) ? false : (seen.add((p.title || "").toLowerCase()), true));
        s.setSnowballResults(merged);
        s.updateTask("snowball", { status: "canceled" });
        toast.info(`Canceled: ${unique.length} gathered this run, ${merged.length} kept in total.`);
        return;
      }
      s.setSnowballResults(unique);
      s.setSnowballChosen(null);   // fresh citations → start the selection over
      s.updateTask("snowball", { status: "done" });
      const totalAvailable = perSeed.reduce((a, x) => a + x.available, 0);
      const dropped = totalAvailable - all.length;
      toast.success(
        dropped > 0
          ? `${unique.length.toLocaleString()} unique articles kept, out of ${totalAvailable.toLocaleString()} found`
          : `Found ${unique.length.toLocaleString()} unique articles via snowballing`,
      );
    } catch (e: any) {
      s.updateTask("snowball", { status: "error", detail: e?.message });
    }
  }

  // "Find similar": acts on the study selected in the left pane and pulls its
  // topical neighbours — OpenAlex related_works ranked by the local embedder.
  // Separate from the bulk citation snowball; it explores a single paper and
  // reuses the same results list + selection + add-to-queue below.
  async function startSimilar() {
    // The selected left-pane seed is a paper we snowballed from. Match it by
    // normalised title across all full-text results, then the whole corpus, so a
    // changed include set or a minor title difference doesn't break find-similar.
    const norm = (t: string) => (t || "").toLowerCase().trim();
    const target = norm(selectedSeed || "");
    const ft = target ? (s.fullTextResults ?? []).find(r => norm(r.Title) === target) : undefined;
    const raw = target && !ft ? (s.rawPapers ?? []).find(p => norm(p.title) === target) : undefined;
    const seed = ft ? { Title: ft.Title, Abstract: ft.Abstract || "" }
      : raw ? { Title: raw.title, Abstract: raw.abstract || "" }
      : null;
    if (!seed) { toast.error("Select a study in the left panel to find similar articles for."); return; }
    const { abort } = s.startTask("snowball", [{ id: "sim", label: "Finding similar articles", status: "running" }]);
    s.setSnowballSeeds(seeds.map(x => x.Title));
    const signal = abort.signal;
    try {
      const sims = await AIService.fetchSimilar(
        [{ title: seed.Title, abstract: seed.Abstract, doi: "" }],
        maxCit, signal);
      sims.forEach(c => { c.seed_paper_title = seed.Title; });
      // Add the similar papers to the existing snowball results (dedup by title)
      // rather than replacing them, so citation-snowball documents are never lost.
      // Existing screening/selection is preserved for the same reason.
      const existing = s.snowballResults ?? [];
      const seen = new Set(existing.map((p: any) => (p.title || "").toLowerCase()));
      const added = sims.filter(p => { const k = (p.title || "").toLowerCase(); return seen.has(k) ? false : (seen.add(k), true); });
      s.setSnowballResults([...existing, ...added]);
      if (signal.aborted) {
        s.updateTask("snowball", { status: "canceled" });
        toast.info(`Canceled. Added ${added.length} similar paper${added.length === 1 ? "" : "s"}.`);
      } else {
        s.updateTask("snowball", { status: "done" });
        toast.success(added.length
          ? `Added ${added.length} paper${added.length === 1 ? "" : "s"} similar to “${seed.Title.slice(0, 44)}…”`
          : "No new similar papers found for that study.");
      }
    } catch (e: any) {
      if (!signal.aborted) {
        s.updateTask("snowball", { status: "error", detail: e?.message });
        // The backend says which of the two it is: a refused request names the
        // fix, and anything else is ours to look at.
        toast.error(e?.message || "Find similar failed.");
      }
    }
  }

  function addToMain() {
    if (!s.snowballResults) return;
    const chosenSet = new Set(s.snowballChosen ?? []);
    // Don't re-add papers already in the corpus or already screened.
    const existingIds = new Set([
      ...((s.uniquePapers || []).map(p => p.id)),
      ...((s.results || []).map(r => r.paper_id)),
    ]);
    const toAdd: Paper[] = [];
    for (const p of s.snowballResults) {
      if (!chosenSet.has(p.id) || existingIds.has(p.id)) continue;
      toAdd.push({
        id: p.id, source: p.source || "Citation snowballing", title: p.title || "",
        abstract: p.abstract || "", url: p.url || "", year: p.year, authors: p.authors,
      });
    }
    if (toAdd.length === 0) { toast.error("No new articles to add — they are already in the corpus."); return; }
    // Add as UNSCREENED corpus candidates (not forced includes): they join the
    // abstract-screening "Screen new" queue, so every snowballed paper still goes
    // through screening unless the reviewer includes it by hand on that tab.
    s.setUniquePapers([...(s.uniquePapers || []), ...toAdd]);
    s.setRawPapers([...(s.rawPapers || []), ...toAdd]);
    s.setPrisma(p => ({ ...p, identified: p.identified + toAdd.length }));
    toast.success(`Added ${toAdd.length} article${toAdd.length === 1 ? "" : "s"} to the screening queue. Run “Screen new” on Abstract Screening to screen them.`);
  }
  function clearAll() {
    s.setSnowballResults(null); s.setSnowballScreened(null); s.setSnowballChosen(null);
  }

  // Screening lives on Abstract Screening now, where the whole corpus is
  // screened in one pass. Verdicts recorded by an earlier session are still
  // shown, and still seed the selection, but this page no longer makes them.
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

  const hasResults = Boolean(results && results.length > 0);
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

  // Flag when the current seed set differs from the one the visible citations
  // were fetched from (after re-running full-text, or switching source), so the
  // stale results prompt a re-run instead of silently showing old seeds.
  const seedsChanged = Boolean(results) && Array.isArray(s.snowballSeeds) && (() => {
    const now = new Set(seeds.map(x => x.Title));
    if (now.size !== s.snowballSeeds!.length) return true;
    return s.snowballSeeds!.some(t => !now.has(t));
  })();

  return (
    <div className="space-y-3">
      {/* ── Compact header: stats + controls + run ─────────────────────────── */}
      <ControlPane
        stats={<>
          <InlineStat icon={FileText} value={seeds.length} label="Seeds" />
          {hasResults && <>
            <PaneDivider />
            {/* The split between the two directions is a breakdown of what was
                found, so it reads under that number rather than as two tiles of
                its own. "Selected" went for the same reason: the button on the
                right already says how many are selected. */}
            <InlineStat
              icon={Network}
              value={results.length}
              label="Found"
              hint={`${backCount.toLocaleString()} back · ${fwdCount.toLocaleString()} fwd`}
            />
          </>}
          {screened && <>
            <PaneDivider />
            <InlineStat icon={CheckCircle2} tone="success" value={includedCount} label="AI included" />
            <InlineStat icon={XCircle} tone="amber" value={excludedCount} label="AI excluded" />
          </>}
        </>}
        actions={<>
          <ImportStudies stage="citation-snowball" label="Add seeds" />
          <PaneSeparator />

          {/* Direction, rule and cap are one question asked three ways: what to
              fetch. They live behind one control so the row stays a single line
              at the width this app runs at. Direction was the last of them
              still sitting out here, in a labelled box no other pane uses. */}
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" size="sm" className="h-8 shadow-sm font-normal"
                      title="Which citations to follow, how many to keep per seed, and which ones">
                <SlidersHorizontal className="size-3.5 mr-1.5" />
                {DIRECTION_SHORT[type]} · {maxCit} per seed
                <span className="text-muted-foreground ml-1.5">· {SORT_SHORT[sort]}</span>
                <ChevronDown className="size-3.5 ml-1 opacity-60" />
              </Button>
            </PopoverTrigger>
            <PopoverContent align="end" className="w-80 p-0">
              <div className="px-3.5 py-3 border-b">
                <div className="text-xs font-semibold">What to fetch</div>
                <p className="mt-0.5 text-[11px] text-muted-foreground">
                  Every reference and citation is fetched. This decides which are kept
                  when a seed has more than the cap, and the run records which rule was used.
                </p>
              </div>
              <div className="flex items-center justify-between gap-2 border-b px-3.5 py-2.5">
                <span className="text-xs font-medium">Direction</span>
                <div className="inline-flex rounded-md border p-0.5"
                     title="Backward follows the references a seed cites; forward finds papers that cite it">
                  {DIRECTIONS.map(d => (
                    <button
                      key={d.value}
                      onClick={() => setType(d.value)}
                      className={`relative h-6 rounded px-2 text-[11px] font-medium transition-colors ${
                        type === d.value ? "text-primary-foreground" : "text-muted-foreground hover:bg-muted"}`}
                    >
                      {type === d.value && <motion.span layoutId="snowball-direction" transition={{ type: "spring", stiffness: 440, damping: 36, mass: 0.7 }} className="absolute inset-0 rounded bg-primary" />}
                      <span className="relative z-10">{d.label}</span>
                    </button>
                  ))}
                </div>
              </div>
              <div className="p-1.5">
                {SORT_OPTIONS.map(o => (
                  <button
                    key={o.value}
                    onClick={() => setSort(o.value)}
                    className={`w-full text-left rounded-md px-2.5 py-2 transition-colors ${
                      sort === o.value ? "bg-primary/10" : "hover:bg-muted"}`}
                  >
                    <div className="flex items-center gap-2">
                      {sort === o.value
                        ? <Check className="size-3.5 text-primary shrink-0" />
                        : <span className="size-3.5 shrink-0" />}
                      <span className="text-xs font-medium">{o.label}</span>
                    </div>
                    <div className="pl-5.5 text-[11px] text-muted-foreground">{o.help}</div>
                  </button>
                ))}
              </div>
              <div className="flex items-center justify-between gap-2 border-t px-3.5 py-2.5">
                <span className="text-xs font-medium">Maximum per seed</span>
                <div className="inline-flex h-7 items-center rounded-md border bg-background">
                  <button onClick={() => setMaxCit(m => Math.max(5, m - 5))}
                          className="px-1.5 h-full text-muted-foreground hover:text-foreground disabled:opacity-40"
                          disabled={maxCit <= 5} aria-label="Decrease">
                    <Minus className="size-3.5" />
                  </button>
                  <input
                    type="number" value={maxCit} min={1} max={500}
                    onChange={e => {
                      const n = parseInt(e.target.value, 10);
                      setMaxCit(Number.isNaN(n) ? 1 : Math.min(500, Math.max(1, n)));
                    }}
                    className="w-10 h-full text-center text-sm font-semibold tabular-nums bg-transparent border-x outline-none [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                  />
                  <button onClick={() => setMaxCit(m => Math.min(500, m + 5))}
                          className="px-1.5 h-full text-muted-foreground hover:text-foreground disabled:opacity-40"
                          disabled={maxCit >= 500} aria-label="Increase">
                    <Plus className="size-3.5" />
                  </button>
                </div>
              </div>
            </PopoverContent>
          </Popover>
          {running ? (
            <Button size="sm" variant="destructive" className="h-8 shadow-sm" onClick={() => s.cancelTask("snowball")}>
              <XCircle className="size-3.5 mr-1.5" />Cancel
            </Button>
          ) : (
            // Filled until there are citations, because fetching them is the
            // only thing to do here. Once they exist the primary act is sending
            // them on, so the run button steps back and lets that one lead.
            <Button size="sm" variant={hasResults ? "outline" : "default"} className="h-8 shadow-sm" onClick={start}>
              <Network className="size-3.5 mr-1.5" />{hasResults ? "Re-run" : "Start snowballing"}
            </Button>
          )}

          {/* What to do with the citations, beside the controls that fetched
              them. These used to sit under the list, which on a run of a few
              thousand meant scrolling past every citation to reach them. */}
          {hasResults && <>
            <PaneSeparator />
            <Button size="sm" className="h-8 shadow-sm" onClick={addToMain} disabled={chosen.size === 0}
                    title="Add the selected citations to the corpus as unscreened. They join the Abstract Screening “Screen new” queue rather than being auto-included.">
              <Plus className="size-3.5 mr-1.5" />Add {chosen.size.toLocaleString()} to queue
            </Button>
          </>}
        </>}
      />

      {fetchTask && fetchTask.status === "running" && (
        // The run has two phases and the card titles itself after whichever is
        // in flight, so it cannot say "Fetching citations" over a line that
        // says the citations are being ranked.
        <TaskProgressCard
          task={fetchTask}
          title={fetchTask.stages[0]?.label || "Fetching citations"}
          onCancel={() => s.cancelTask("snowball")}
        />
      )}

      {!running && !hasResults && (
        <Card className="p-10 text-center">
          <Network className="size-8 mx-auto mb-3 text-muted-foreground/40" />
          <p className="text-sm text-muted-foreground max-w-md mx-auto">
            No citations yet. Click <span className="font-medium text-foreground">Start snowballing</span> above to expand your {seeds.length} kept full-text stud{seeds.length === 1 ? "y" : "ies"} by their references and citations.
          </p>
        </Card>
      )}
      {hasResults && (
        <>
          {seedsChanged && (
            <Alert>
              <AlertDescription className="flex items-center justify-between gap-3">
                <span className="text-sm">
                  Your kept full-text studies changed since these citations were fetched (now {seeds.length} kept stud{seeds.length === 1 ? "y" : "ies"}). Re-run to refresh.
                </span>
                <Button size="sm" className="h-8 shrink-0" onClick={start} disabled={running}>
                  <Network className="size-3.5 mr-1.5" />Re-run
                </Button>
              </AlertDescription>
            </Alert>
          )}
          {/* ── Two-pane: seed papers (left) + citations they surfaced (right) ─ */}
          <div className="flex gap-4 h-[calc(100vh-16rem)] min-h-[28rem]">
            {/* LEFT: seed papers */}
            <Card className="w-80 shrink-0 p-0 overflow-hidden flex flex-col">
              <div className="px-3 py-2 border-b text-xs uppercase tracking-wide text-muted-foreground">
                Seed articles
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
                    onCheckedChange={(c) => s.setSnowballChosen(prev => {
                      const next = new Set(prev ?? []);
                      shown.forEach(p => { if (c) next.add(p.id); else next.delete(p.id); });
                      return [...next];
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
                {/* Find similar: acts on the study selected in the left pane —
                    pulls its topical neighbours (OpenAlex related_works, ranked
                    by the local embedder), separate from citation chaining.
                    Always shown; greyed until a specific study is selected. */}
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8 text-xs shrink-0 gap-1"
                  onClick={startSimilar}
                  disabled={running || selectedSeed === null}
                  title={selectedSeed === null ? "Select a study on the left, then find similar papers" : "Find papers semantically similar to the selected study"}
                >
                  <Sparkles className="size-3.5" />Find similar
                </Button>
                {/* Clearing acts on this list, so it sits over this list rather
                    than in the control pane, where it was the one control that
                    had nothing to do with fetching. */}
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-8 px-2 shrink-0 text-muted-foreground"
                  onClick={clearAll}
                  title="Clear results: discard every citation fetched in this run"
                >
                  <Trash2 className="size-4" />
                </Button>
              </div>
              <div className="overflow-auto flex-1 divide-y">
                {pagedShown.visible.map((p, i) => {
                  const v = verdict(p);
                  const picked = chosen.has(p.id);
                  return (
                    <div key={p.id || i} className={`px-3 py-2.5 ${picked ? "bg-primary/5" : ""}`}>
                      <div className="flex items-start gap-2">
                        <Checkbox
                          checked={picked}
                          onCheckedChange={() => toggleChosen(p.id)}
                          className="mt-0.5 shrink-0"
                          title={picked ? "Selected: will carry into main results" : "Click to include in main results"}
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
                <ShowMore shown={pagedShown.shown} total={pagedShown.total} step={pagedShown.step} onMore={pagedShown.showMore} onAll={pagedShown.showAll} />
                {shown.length === 0 && (
                  <div className="p-4 text-sm text-muted-foreground">No citations match “{q}”.</div>
                )}
              </div>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}

/** The rule between groups of actions, as the other stages draw it. */
function PaneSeparator() {
  return <span className="mx-0.5 h-6 w-px bg-border" aria-hidden="true" />;
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
