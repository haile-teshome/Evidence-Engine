import { useEffect, useMemo, useState, type ReactNode } from "react";
import { usePagedRows } from "../lib/usePagedRows";
import { ShowMore } from "./ShowMore";
import type { Paper } from "../lib/mockServices";
import { cn } from "./ui/utils";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Badge } from "./ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { EmptyState } from "./EmptyState";
import {
  Search, X, Plus, Download, FileDown, Tag, ExternalLink, Library,
  ArrowDownWideNarrow, BarChart3, Filter, Layers, Calendar, type LucideIcon,
} from "lucide-react";

// A self-contained workspace over the CAPTURED (retrieved) papers: explore /
// filter / visualise them, and tag / annotate / export the current selection.
// Tags and notes live only in this component's state and are mirrored to
// localStorage so they survive a reload; nothing is written to the backend.

const STORAGE_KEY = "ee:results-workspace:annotations";

type SortKey = "year-desc" | "year-asc" | "title";

type Annotation = { tags: string[]; note: string };

// Parse whatever is in localStorage back into a well-typed annotation map,
// dropping anything malformed rather than trusting the stored shape.
function coerceAnnotations(raw: unknown): Record<string, Annotation> {
  const out: Record<string, Annotation> = {};
  if (!raw || typeof raw !== "object") return out;
  for (const [id, v] of Object.entries(raw as Record<string, unknown>)) {
    if (!v || typeof v !== "object") continue;
    const obj = v as { tags?: unknown; note?: unknown };
    const tags = Array.isArray(obj.tags)
      ? obj.tags.filter((t): t is string => typeof t === "string")
      : [];
    const note = typeof obj.note === "string" ? obj.note : "";
    if (tags.length === 0 && note === "") continue;
    out[id] = { tags, note };
  }
  return out;
}

const sourceKey = (p: Paper): string => (p.source || "Unknown").trim() || "Unknown";

// ── CSV / RIS builders ──────────────────────────────────────────────────────

function csvCell(v: string): string {
  return /[",\n\r]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
}

function buildCsv(rows: Paper[], annotations: Record<string, Annotation>): string {
  const header = ["id", "title", "authors", "year", "source", "url", "abstract", "tags", "note"];
  const lines = [header.join(",")];
  for (const p of rows) {
    const a = annotations[p.id];
    lines.push([
      p.id ?? "",
      p.title ?? "",
      p.authors ?? "",
      typeof p.year === "number" ? String(p.year) : "",
      p.source ?? "",
      p.url ?? "",
      p.abstract ?? "",
      (a?.tags ?? []).join("; "),
      a?.note ?? "",
    ].map(csvCell).join(","));
  }
  return lines.join("\r\n");
}

function buildRis(rows: Paper[], annotations: Record<string, Annotation>): string {
  const blocks: string[] = [];
  for (const p of rows) {
    const a = annotations[p.id];
    const lines: string[] = ["TY  - JOUR"];
    if (p.title) lines.push(`TI  - ${p.title}`);
    if (p.abstract) lines.push(`AB  - ${p.abstract.replace(/\r?\n/g, " ")}`);
    if (p.authors) {
      // The `authors` field is a single string; ";" is the only safe delimiter
      // (splitting on "," would mangle "Last, First"), so fall back to one AU line.
      const authors = p.authors.includes(";")
        ? p.authors.split(";").map(x => x.trim()).filter(Boolean)
        : [p.authors.trim()];
      for (const au of authors) lines.push(`AU  - ${au}`);
    }
    if (typeof p.year === "number") lines.push(`PY  - ${p.year}`);
    if (p.url) lines.push(`UR  - ${p.url}`);
    for (const t of a?.tags ?? []) lines.push(`KW  - ${t}`);
    if (a?.note) lines.push(`N1  - ${a.note.replace(/\r?\n/g, " ")}`);
    lines.push("ER  - ");
    blocks.push(lines.join("\r\n"));
  }
  return blocks.join("\r\n\r\n") + "\r\n";
}

function triggerDownload(filename: string, content: string, mime: string): void {
  try {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch {
    /* download unavailable (e.g. sandboxed) — nothing else to do */
  }
}

// ── small presentational helpers ────────────────────────────────────────────

function SectionLabel({ icon: Icon, children }: { icon: LucideIcon; children: ReactNode }) {
  return (
    <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
      <Icon className="size-3.5" />
      {children}
    </div>
  );
}

// ── component ────────────────────────────────────────────────────────────────

export function ResultsWorkspace({ papers }: { papers: Paper[] }) {
  const [search, setSearch] = useState("");
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [yearMin, setYearMin] = useState("");
  const [yearMax, setYearMax] = useState("");
  const [sort, setSort] = useState<SortKey>("year-desc");
  const [activeTag, setActiveTag] = useState<string | null>(null);
  const [tagDraft, setTagDraft] = useState<Record<string, string>>({});

  const [annotations, setAnnotations] = useState<Record<string, Annotation>>(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? coerceAnnotations(JSON.parse(raw)) : {};
    } catch {
      return {};
    }
  });

  // Mirror tags/notes to localStorage so they survive a reload.
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
    } catch {
      /* localStorage unavailable (private mode, quota) — keep working in memory */
    }
  }, [annotations]);

  // Distinct sources across ALL passed-in papers (so a filtered-out source can
  // still be toggled back on), with their totals.
  const allSources = useMemo(() => {
    const counts = new Map<string, number>();
    for (const p of papers) counts.set(sourceKey(p), (counts.get(sourceKey(p)) ?? 0) + 1);
    return [...counts.entries()]
      .map(([source, count]) => ({ source, count }))
      .sort((a, b) => b.count - a.count || a.source.localeCompare(b.source));
  }, [papers]);

  const yearBounds = useMemo(() => {
    const ys = papers.map(p => p.year).filter((y): y is number => typeof y === "number");
    return ys.length ? { min: Math.min(...ys), max: Math.max(...ys) } : { min: 0, max: 0 };
  }, [papers]);

  const allTags = useMemo(() => {
    const set = new Set<string>();
    for (const a of Object.values(annotations)) for (const t of a.tags) set.add(t);
    return [...set].sort((a, b) => a.localeCompare(b));
  }, [annotations]);

  // Filter + sort the passed-in papers into the current working set.
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    const min = yearMin.trim() === "" ? null : Number(yearMin);
    const max = yearMax.trim() === "" ? null : Number(yearMax);
    const rows = papers.filter(p => {
      if (q) {
        const hay = `${p.title ?? ""} ${p.abstract ?? ""} ${p.authors ?? ""}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      if (selectedSources.length && !selectedSources.includes(sourceKey(p))) return false;
      if (min != null && Number.isFinite(min) && (typeof p.year !== "number" || p.year < min)) return false;
      if (max != null && Number.isFinite(max) && (typeof p.year !== "number" || p.year > max)) return false;
      if (activeTag && !(annotations[p.id]?.tags ?? []).includes(activeTag)) return false;
      return true;
    });
    return [...rows].sort((a, b) => {
      if (sort === "title") return (a.title ?? "").localeCompare(b.title ?? "");
      const aHas = typeof a.year === "number";
      const bHas = typeof b.year === "number";
      if (!aHas && !bHas) return (a.title ?? "").localeCompare(b.title ?? "");
      if (!aHas) return 1; // undefined years sort last in both directions
      if (!bHas) return -1;
      const ay = a.year ?? 0;
      const by = b.year ?? 0;
      return sort === "year-desc" ? by - ay : ay - by;
    });
  }, [papers, search, selectedSources, yearMin, yearMax, activeTag, annotations, sort]);
  const paged = usePagedRows(filtered);

  // Per-year histogram over the current working set.
  const yearHistogram = useMemo(() => {
    const counts = new Map<number, number>();
    for (const p of filtered) {
      if (typeof p.year === "number" && Number.isFinite(p.year)) {
        counts.set(p.year, (counts.get(p.year) ?? 0) + 1);
      }
    }
    if (counts.size === 0) return [] as { year: number; count: number }[];
    const years = [...counts.keys()];
    const min = Math.min(...years);
    const max = Math.max(...years);
    // Fill gap years so the bars read as a real timeline; but for a pathological
    // span, fall back to the years that actually have data.
    if (max - min > 100) {
      return years.sort((a, b) => a - b).map(year => ({ year, count: counts.get(year) ?? 0 }));
    }
    const out: { year: number; count: number }[] = [];
    for (let y = min; y <= max; y++) out.push({ year: y, count: counts.get(y) ?? 0 });
    return out;
  }, [filtered]);
  const maxYearCount = Math.max(1, ...yearHistogram.map(b => b.count));
  const yearLabelStep = yearHistogram.length > 24 ? Math.ceil(yearHistogram.length / 12) : 1;

  // Per-source breakdown over the current working set.
  const sourceBreakdown = useMemo(() => {
    const counts = new Map<string, number>();
    for (const p of filtered) counts.set(sourceKey(p), (counts.get(sourceKey(p)) ?? 0) + 1);
    return [...counts.entries()]
      .map(([source, count]) => ({ source, count }))
      .sort((a, b) => b.count - a.count || a.source.localeCompare(b.source));
  }, [filtered]);
  const maxSourceCount = Math.max(1, ...sourceBreakdown.map(s => s.count));

  // ── annotation mutations ──
  function toggleSource(source: string) {
    setSelectedSources(prev =>
      prev.includes(source) ? prev.filter(s => s !== source) : [...prev, source],
    );
  }
  function updateNote(id: string, note: string) {
    setAnnotations(prev => {
      const cur = prev[id] ?? { tags: [], note: "" };
      return { ...prev, [id]: { ...cur, note } };
    });
  }
  function addTag(id: string, raw: string) {
    const tag = raw.trim();
    if (!tag) return;
    setAnnotations(prev => {
      const cur = prev[id] ?? { tags: [], note: "" };
      if (cur.tags.includes(tag)) return prev;
      return { ...prev, [id]: { ...cur, tags: [...cur.tags, tag] } };
    });
  }
  function removeTag(id: string, tag: string) {
    setAnnotations(prev => {
      const cur = prev[id];
      if (!cur) return prev;
      const next = cur.tags.filter(t => t !== tag);
      if (activeTag === tag && !next.length) setActiveTag(null);
      return { ...prev, [id]: { ...cur, tags: next } };
    });
  }

  function exportCsv() {
    triggerDownload("captured-papers.csv", buildCsv(filtered, annotations), "text/csv;charset=utf-8");
  }
  function exportRis() {
    triggerDownload("captured-papers.ris", buildRis(filtered, annotations), "application/x-research-info-systems");
  }

  const hasActiveFilter =
    search.trim() !== "" || selectedSources.length > 0 || yearMin !== "" || yearMax !== "" || activeTag != null;

  if (papers.length === 0) {
    return (
      <EmptyState
        icon={Library}
        title="No captured papers yet"
        description="Once records are retrieved from your searches, they will show up here to explore, filter, tag, and export."
      />
    );
  }

  return (
    <div className="space-y-4">
      {/* ── toolbar: search / sources / years / tags / export ── */}
      <Card className="p-4 space-y-4">
        <div className="flex flex-col sm:flex-row gap-3 sm:items-center">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground pointer-events-none" />
            <Input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search title, abstract, or authors..."
              className="pl-9"
            />
          </div>
          <Select value={sort} onValueChange={v => setSort(v as SortKey)}>
            <SelectTrigger className="w-full sm:w-52">
              <div className="flex items-center gap-2">
                <ArrowDownWideNarrow className="size-4 text-muted-foreground" />
                <SelectValue />
              </div>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="year-desc">Year (newest first)</SelectItem>
              <SelectItem value="year-asc">Year (oldest first)</SelectItem>
              <SelectItem value="title">Title (A to Z)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <SectionLabel icon={Filter}>Sources</SectionLabel>
          <div className="flex flex-wrap gap-1.5">
            {allSources.map(({ source, count }) => {
              const active = selectedSources.includes(source);
              return (
                <button
                  key={source}
                  type="button"
                  onClick={() => toggleSource(source)}
                  className={cn(
                    "inline-flex items-center gap-1.5 h-7 px-2.5 rounded-full text-xs font-medium border transition-colors",
                    active
                      ? "bg-primary text-primary-foreground border-transparent"
                      : "bg-muted/40 text-muted-foreground border-border hover:bg-muted hover:text-foreground",
                  )}
                >
                  {source}
                  <span className={cn("tabular-nums", active ? "text-primary-foreground/80" : "text-muted-foreground/70")}>
                    {count}
                  </span>
                </button>
              );
            })}
            {selectedSources.length > 0 && (
              <button
                type="button"
                onClick={() => setSelectedSources([])}
                className="inline-flex items-center gap-1 h-7 px-2 rounded-full text-xs text-muted-foreground hover:text-foreground"
              >
                <X className="size-3" /> Clear
              </button>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
          <div className="space-y-1.5">
            <SectionLabel icon={Calendar}>Year range</SectionLabel>
            <div className="flex items-center gap-2">
              <Input
                type="number"
                inputMode="numeric"
                value={yearMin}
                onChange={e => setYearMin(e.target.value)}
                placeholder={yearBounds.min ? String(yearBounds.min) : "from"}
                className="w-24 h-8"
                aria-label="Minimum year"
              />
              <span className="text-sm text-muted-foreground">to</span>
              <Input
                type="number"
                inputMode="numeric"
                value={yearMax}
                onChange={e => setYearMax(e.target.value)}
                placeholder={yearBounds.max ? String(yearBounds.max) : "to"}
                className="w-24 h-8"
                aria-label="Maximum year"
              />
            </div>
          </div>

          {allTags.length > 0 && (
            <div className="space-y-1.5">
              <SectionLabel icon={Tag}>Filter by tag</SectionLabel>
              <div className="flex flex-wrap gap-1.5">
                {allTags.map(t => {
                  const active = activeTag === t;
                  return (
                    <button
                      key={t}
                      type="button"
                      onClick={() => setActiveTag(active ? null : t)}
                      className={cn(
                        "inline-flex items-center gap-1 h-7 px-2.5 rounded-full text-xs font-medium border transition-colors",
                        active
                          ? "bg-primary/15 text-primary border-primary/30"
                          : "bg-muted/40 text-muted-foreground border-border hover:bg-muted hover:text-foreground",
                      )}
                    >
                      {t}
                      {active && <X className="size-3" />}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 border-t pt-3">
          <div className="text-sm text-muted-foreground">
            Showing <span className="font-medium text-foreground tabular-nums">{filtered.length}</span> of{" "}
            <span className="tabular-nums">{papers.length}</span> {papers.length === 1 ? "paper" : "papers"}
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" onClick={exportCsv} disabled={filtered.length === 0}>
              <Download className="size-3.5" /> CSV
            </Button>
            <Button size="sm" variant="outline" onClick={exportRis} disabled={filtered.length === 0}>
              <FileDown className="size-3.5" /> RIS
            </Button>
          </div>
        </div>
      </Card>

      {/* ── inline visualisations (no chart library) ── */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="p-4 space-y-3">
          <SectionLabel icon={BarChart3}>Publications by year</SectionLabel>
          {yearHistogram.length === 0 ? (
            <p className="text-sm text-muted-foreground py-6 text-center">No year data in this selection.</p>
          ) : (
            <div className="overflow-x-auto">
              <div className="flex items-end gap-1 h-28 min-w-full border-b">
                {yearHistogram.map(b => (
                  <div key={b.year} className="flex-1 min-w-[16px] h-full flex items-end" title={`${b.year}: ${b.count}`}>
                    <div
                      className="w-full rounded-t bg-primary/70 hover:bg-primary transition-colors"
                      style={{ height: `${Math.max(b.count === 0 ? 0 : 4, (b.count / maxYearCount) * 100)}%` }}
                    />
                  </div>
                ))}
              </div>
              <div className="flex gap-1 mt-1 min-w-full">
                {yearHistogram.map((b, i) => (
                  <div key={b.year} className="flex-1 min-w-[16px] text-center text-[9px] text-muted-foreground tabular-nums">
                    {i % yearLabelStep === 0 ? b.year : ""}
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>

        <Card className="p-4 space-y-3">
          <SectionLabel icon={Layers}>Records by source</SectionLabel>
          {sourceBreakdown.length === 0 ? (
            <p className="text-sm text-muted-foreground py-6 text-center">Nothing to show.</p>
          ) : (
            <div className="space-y-2">
              {sourceBreakdown.map(({ source, count }) => (
                <div key={source} className="flex items-center gap-3 text-sm">
                  <div className="w-28 shrink-0 truncate text-muted-foreground" title={source}>{source}</div>
                  <div className="flex-1 h-2.5 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-primary/70"
                      style={{ width: `${(count / maxSourceCount) * 100}%` }}
                    />
                  </div>
                  <div className="w-8 shrink-0 text-right tabular-nums text-muted-foreground">{count}</div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* ── paper list with tags + note per paper ── */}
      {filtered.length === 0 ? (
        <div className="rounded-xl border border-dashed bg-muted/20 px-6 py-10 text-center text-sm text-muted-foreground">
          No papers match your filters.{" "}
          {hasActiveFilter && (
            <button
              type="button"
              className="text-primary hover:underline"
              onClick={() => {
                setSearch("");
                setSelectedSources([]);
                setYearMin("");
                setYearMax("");
                setActiveTag(null);
              }}
            >
              Reset filters
            </button>
          )}
        </div>
      ) : (
        <div className="space-y-2">
          {paged.visible.map(p => {
            const ann = annotations[p.id] ?? { tags: [], note: "" };
            return (
              <Card key={p.id} className="p-4 space-y-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 space-y-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <Badge variant="secondary" className="shrink-0">{sourceKey(p)}</Badge>
                      {typeof p.year === "number" && (
                        <span className="text-xs text-muted-foreground tabular-nums">{p.year}</span>
                      )}
                    </div>
                    <h3 className="font-medium leading-snug">{p.title || "Untitled"}</h3>
                    {p.authors && <p className="text-xs text-muted-foreground line-clamp-1">{p.authors}</p>}
                  </div>
                  {p.url && (
                    <a
                      href={p.url}
                      target="_blank"
                      rel="noreferrer"
                      className="shrink-0 inline-flex items-center gap-1 text-xs text-primary hover:underline"
                    >
                      <ExternalLink className="size-3.5" /> Open
                    </a>
                  )}
                </div>

                {p.abstract && <p className="text-sm text-muted-foreground line-clamp-3">{p.abstract}</p>}

                <div className="flex flex-wrap items-center gap-1.5">
                  {ann.tags.map(t => (
                    <span
                      key={t}
                      className="inline-flex items-center gap-1 h-6 pl-2.5 pr-1 text-[11px] font-medium text-primary rounded-full bg-primary/10"
                    >
                      {t}
                      <button
                        type="button"
                        onClick={() => removeTag(p.id, t)}
                        className="grid place-items-center size-4 rounded-full hover:bg-primary/20"
                        aria-label={`Remove tag ${t}`}
                      >
                        <X className="size-3" />
                      </button>
                    </span>
                  ))}
                  <span className="inline-flex items-center h-6 rounded-full border border-dashed pl-1.5 pr-1 focus-within:border-ring">
                    <Plus className="size-3 text-muted-foreground shrink-0" />
                    <input
                      value={tagDraft[p.id] ?? ""}
                      onChange={e => setTagDraft(prev => ({ ...prev, [p.id]: e.target.value }))}
                      onKeyDown={e => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          addTag(p.id, tagDraft[p.id] ?? "");
                          setTagDraft(prev => ({ ...prev, [p.id]: "" }));
                        }
                      }}
                      placeholder="Add tag"
                      className="h-full w-24 bg-transparent px-1 text-[11px] outline-none placeholder:text-muted-foreground"
                      aria-label="Add a tag"
                    />
                  </span>
                </div>

                <Textarea
                  value={ann.note}
                  onChange={e => updateNote(p.id, e.target.value)}
                  placeholder="Add a note..."
                  className="h-16 text-sm"
                />
              </Card>
            );
          })}
          <ShowMore shown={paged.shown} total={paged.total} step={paged.step} onMore={paged.showMore} onAll={paged.showAll} />
        </div>
      )}
    </div>
  );
}
