import { useMemo, useState } from "react";
import { type Paper } from "../lib/apiClient";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { EmptyState } from "./EmptyState";
import { Check, Sparkles, Search, ChevronDown, X } from "lucide-react";

// Refine tab: the reviewer reads the summary and evidence (Overview / Relevance),
// then marks the studies that best match the question. Those picks become seeds:
// the next search learns their vocabulary + MeSH to surface more like them.
// The top is a filter bar (title text, source, year) to narrow the captured set
// before marking; filtering never changes what is marked.
export function RefineSeeds({
  papers, seedIds, onToggleSeed, onRefine,
}: {
  papers: Paper[];
  seedIds: Set<string>;
  onToggleSeed: (id: string) => void;
  onRefine: () => void;
}) {
  const [q, setQ] = useState("");
  const [src, setSrc] = useState("all");

  const sources = useMemo(
    () => Array.from(new Set(papers.map(p => p.source).filter(Boolean))).sort(),
    [papers],
  );

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return papers.filter(p => {
      if (needle && !(p.title || "").toLowerCase().includes(needle)) return false;
      if (src !== "all" && p.source !== src) return false;
      return true;
    });
  }, [papers, q, src]);

  if (!papers.length) {
    return (
      <EmptyState
        icon={Search}
        title="Nothing to refine yet"
        description="Run a search first. Then mark the studies that best match your question to refine it even further."
      />
    );
  }

  const seeded = papers.filter(p => seedIds.has(p.id));
  const seedCount = seeded.length;
  const clearSeeds = () => seeded.forEach(p => onToggleSeed(p.id));
  const hasFilter = !!(q.trim() || src !== "all");

  return (
    <div className="space-y-2.5">
      <div className="space-y-2">
      {/* Control bar: one search pill (title filter + source), primary Refine CTA */}
      <div className="flex items-center gap-2">
        <div className="flex-1 min-w-0 flex items-center gap-1.5 h-9 rounded-full border bg-muted/30 px-3 focus-within:border-primary/50 focus-within:bg-background transition-colors">
          <Search className="size-4 text-muted-foreground shrink-0" />
          <Input
            value={q}
            onChange={e => setQ(e.target.value)}
            placeholder="Filter studies by title…"
            className="border-0 bg-transparent shadow-none focus-visible:ring-0 h-8 px-0 text-sm"
          />
          {sources.length > 1 && (
            <>
              <span className="h-4 w-px bg-border shrink-0" />
              <div className="relative shrink-0">
                <select
                  value={src}
                  onChange={e => setSrc(e.target.value)}
                  className="appearance-none bg-transparent h-8 pl-1.5 pr-6 text-sm text-foreground focus:outline-none cursor-pointer max-w-[160px]"
                  title="Filter by source"
                >
                  <option value="all">All sources</option>
                  {sources.map(sc => <option key={sc} value={sc}>{sc}</option>)}
                </select>
                <ChevronDown className="size-3.5 text-muted-foreground absolute right-1 top-1/2 -translate-y-1/2 pointer-events-none" />
              </div>
            </>
          )}
        </div>
        {seedCount > 0 && (
          <Button
            size="sm"
            onClick={clearSeeds}
            className="h-9 gap-1.5 shrink-0 rounded-full px-3 border-transparent bg-rose-600 text-white hover:bg-rose-700"
          >
            <X className="size-3.5" />Clear
          </Button>
        )}
        <Button size="sm" className="h-9 gap-1.5 shrink-0 rounded-full px-4" onClick={onRefine} disabled={!seedCount}>
          <Sparkles className="size-3.5" />Refine
        </Button>
      </div>

      {/* Status row: marked-count chip on the left, result count on the right */}
      <div className="flex items-center justify-between gap-2 px-0.5 text-xs">
        <span
          className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-medium ${
            seedCount > 0
              ? "border-emerald-200 bg-emerald-50 text-emerald-700"
              : "border-border bg-muted/40 text-muted-foreground"
          }`}
        >
          <Check className="size-3" /><span className="tabular-nums">{seedCount}</span> marked relevant
        </span>
        <span className="text-muted-foreground shrink-0">
          {hasFilter ? (
            <>Showing <span className="tabular-nums font-medium text-foreground/70">{filtered.length}</span> of <span className="tabular-nums">{papers.length}</span></>
          ) : (
            <><span className="tabular-nums font-medium text-foreground/70">{papers.length}</span> studies</>
          )}
        </span>
      </div>
      </div>

      {/* Study list — whole row toggles */}
      {filtered.length === 0 ? (
        <div className="rounded-xl border border-dashed px-3 py-8 text-center text-xs text-muted-foreground">
          No studies match these filters.
        </div>
      ) : (
        <div className="rounded-xl border divide-y overflow-hidden max-h-[440px] overflow-y-auto">
          {filtered.map(p => {
            const on = seedIds.has(p.id);
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => onToggleSeed(p.id)}
                aria-pressed={on}
                className={`w-full text-left flex items-start gap-2.5 px-3 py-2.5 transition-colors ${
                  on ? "bg-primary/5" : "hover:bg-muted/40"
                }`}
              >
                <span
                  className={`shrink-0 mt-0.5 grid place-items-center size-5 rounded-md border transition-colors ${
                    on ? "bg-primary text-primary-foreground border-primary" : "border-muted-foreground/30 text-transparent"
                  }`}
                >
                  <Check className="size-3.5" />
                </span>
                <span className="min-w-0">
                  <span className={`block text-sm leading-snug line-clamp-2 ${on ? "font-medium text-foreground" : ""}`}>
                    {p.title || "(untitled)"}
                  </span>
                  <span className="block text-[11px] text-muted-foreground mt-0.5">
                    {p.source}{p.year ? ` · ${p.year}` : ""}{p.authors ? ` · ${p.authors}` : ""}
                  </span>
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
