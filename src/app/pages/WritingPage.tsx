import { useMemo, useState, useEffect } from "react";
import { useStore } from "../lib/store";
import { Card } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Separator } from "../components/ui/separator";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../components/ui/collapsible";
import { Textarea } from "../components/ui/textarea";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../components/ui/table";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Download, Copy, ChevronDown, Loader2, BookOpen, FileText, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { ScreenResult, FullTextResult } from "../lib/apiClient";

// ── Types ─────────────────────────────────────────────────────────────────────

type PaperMeta = {
  paper_id: string;
  title: string;
  url: string;
  source: string;
  abstract: string;
  authors?: string;
  year?: number;
  doi?: string;
  journal?: string;
  volume?: string;
  issue?: string;
  pages?: string;
};

type CrossRefWork = {
  title?: string[];
  author?: { family: string; given?: string }[];
  published?: { "date-parts": number[][] };
  "container-title"?: string[];
  volume?: string;
  issue?: string;
  page?: string;
  DOI?: string;
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function extractDoi(url: string): string | null {
  const m = url?.match(/doi\.org\/(10\.[^\s?#"']+)/i);
  return m ? m[1] : null;
}

function makeCiteKey(meta: PaperMeta): string {
  const authorPart = meta.authors
    ? meta.authors.split(/[,;]/)[0].trim().split(" ").pop()?.toLowerCase() ?? "unknown"
    : meta.source.toLowerCase().replace(/\s/g, "");
  const yearPart = meta.year ?? "nd";
  const titleWord = meta.title.split(/\s+/).find(w => w.length > 3)?.toLowerCase().replace(/\W/g, "") ?? "paper";
  return `${authorPart}${yearPart}_${titleWord}`;
}

function toBibTeXEntry(meta: PaperMeta, idx: number): string {
  const key = makeCiteKey(meta);
  const esc = (s?: string) => (s ?? "").replace(/[{}"\\]/g, "");
  const lines: string[] = [`@article{${key},`];
  lines.push(`  title     = {${esc(meta.title)}},`);
  if (meta.authors) lines.push(`  author    = {${esc(meta.authors)}},`);
  if (meta.year)    lines.push(`  year      = {${meta.year}},`);
  if (meta.journal) lines.push(`  journal   = {${esc(meta.journal)}},`);
  if (meta.volume)  lines.push(`  volume    = {${meta.volume}},`);
  if (meta.issue)   lines.push(`  number    = {${meta.issue}},`);
  if (meta.pages)   lines.push(`  pages     = {${meta.pages}},`);
  if (meta.doi)     lines.push(`  doi       = {${meta.doi}},`);
  lines.push(`  url       = {${meta.url}},`);
  if (meta.abstract) lines.push(`  abstract  = {${esc(meta.abstract.slice(0, 400))}...},`);
  lines.push(`  note      = {[${idx}] Source: ${meta.source}},`);
  lines.push("}");
  return lines.join("\n");
}

function toRisEntry(meta: PaperMeta): string {
  const lines: string[] = ["TY  - JOUR"];
  lines.push(`TI  - ${meta.title}`);
  if (meta.authors) {
    meta.authors.split(/[;]/).forEach(a => lines.push(`AU  - ${a.trim()}`));
  }
  if (meta.year)    lines.push(`PY  - ${meta.year}`);
  if (meta.journal) lines.push(`JO  - ${meta.journal}`);
  if (meta.volume)  lines.push(`VL  - ${meta.volume}`);
  if (meta.issue)   lines.push(`IS  - ${meta.issue}`);
  if (meta.pages)   lines.push(`SP  - ${meta.pages}`);
  if (meta.doi)     lines.push(`DO  - ${meta.doi}`);
  lines.push(`UR  - ${meta.url}`);
  if (meta.abstract) lines.push(`AB  - ${meta.abstract.slice(0, 600)}`);
  lines.push(`N1  - Source: ${meta.source}`);
  lines.push("ER  - ");
  return lines.join("\n");
}

function downloadFile(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ── Multi-source metadata enrichment ─────────────────────────────────────────

async function fetchEuropePmc(pmid: string): Promise<Partial<PaperMeta>> {
  try {
    const r = await fetch(
      `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:${pmid}+AND+SRC:MED&format=json&resultType=core`
    );
    if (!r.ok) return {};
    const d = await r.json();
    const rec = d?.resultList?.result?.[0];
    if (!rec) return {};
    const authors = (rec.authorList?.author ?? [])
      .map((a: any) => a.fullName ?? `${a.lastName ?? ""} ${a.initials ?? ""}`.trim())
      .filter(Boolean)
      .join("; ");
    return {
      authors: authors || undefined,
      year: rec.pubYear ? Number(rec.pubYear) : undefined,
      journal: rec.journalInfo?.journal?.title ?? rec.journalTitle ?? undefined,
      volume: rec.journalInfo?.volume ?? undefined,
      issue: rec.journalInfo?.issue ?? undefined,
      pages: rec.pageInfo ?? undefined,
      doi: rec.doi ?? undefined,
    };
  } catch { return {}; }
}

async function fetchOpenAlex(id: string): Promise<Partial<PaperMeta>> {
  try {
    const workId = id.startsWith("W") ? id : `W${id}`;
    const r = await fetch(
      `https://api.openalex.org/works/${workId}?select=title,authorships,publication_year,primary_location,doi,biblio`
    );
    if (!r.ok) return {};
    const d = await r.json();
    const authors = (d.authorships ?? [])
      .map((a: any) => a.author?.display_name)
      .filter(Boolean)
      .join("; ");
    const rawDoi = d.doi?.replace("https://doi.org/", "") ?? undefined;
    return {
      authors: authors || undefined,
      year: d.publication_year ?? undefined,
      journal: d.primary_location?.source?.display_name ?? undefined,
      volume: d.biblio?.volume ?? undefined,
      issue: d.biblio?.issue ?? undefined,
      pages: d.biblio?.first_page
        ? `${d.biblio.first_page}${d.biblio.last_page ? "–" + d.biblio.last_page : ""}`
        : undefined,
      doi: rawDoi,
    };
  } catch { return {}; }
}

async function fetchCrossRef(doi: string): Promise<Partial<PaperMeta>> {
  try {
    const r = await fetch(`https://api.crossref.org/works/${encodeURIComponent(doi)}`);
    if (!r.ok) return {};
    const { message }: { message: CrossRefWork } = await r.json();
    const authors = message.author
      ?.map(a => `${a.family}${a.given ? ", " + a.given : ""}`)
      .join("; ");
    return {
      authors: authors || undefined,
      year: message.published?.["date-parts"]?.[0]?.[0],
      journal: message["container-title"]?.[0],
      volume: message.volume,
      issue: message.issue,
      pages: message.page,
    };
  } catch { return {}; }
}

async function fetchMetadata(p: PaperMeta): Promise<Partial<PaperMeta>> {
  const src = p.source.toLowerCase();

  // Europe PMC — id is a PMID (pure digits) or EPM internal ID
  if (src.includes("europe pmc") || src.includes("europepmc") || src.includes("pubmed")) {
    const pmid = p.paper_id.replace(/\D/g, "");
    if (pmid) {
      const data = await fetchEuropePmc(pmid);
      if (data.authors || data.year) return data;
    }
  }

  // OpenAlex / Semantic Scholar — id starts with W
  if (/^W\d+/.test(p.paper_id) || src.includes("openalex") || src.includes("semantic")) {
    const data = await fetchOpenAlex(p.paper_id);
    if (data.authors || data.year) return data;
  }

  // DOI-based fallback (CrossRef)
  const doi = p.doi ?? extractDoi(p.url);
  if (doi) {
    const data = await fetchCrossRef(doi);
    if (data.authors || data.year) return { ...data, doi };
  }

  return {};
}

// ── Source normalisation ──────────────────────────────────────────────────────

function fromScreenResult(r: ScreenResult): PaperMeta {
  const doi = extractDoi(r.URL) ?? undefined;
  return { paper_id: r.paper_id, title: r.Title, url: r.URL, source: r.Source, abstract: r.Abstract, doi };
}

function fromFullTextResult(r: FullTextResult): PaperMeta {
  const doi = extractDoi(r.URL) ?? undefined;
  return { paper_id: r.paper_id, title: r.Title, url: r.URL, source: r.Source, abstract: r.Abstract, doi };
}

// ── Component ─────────────────────────────────────────────────────────────────

export function WritingPage() {
  const s = useStore();

  // Walk back through the pipeline to find the best available paper set.
  const { includedPapers, stage } = useMemo(() => {
    // Stage 1 — full-text screening decisions
    const ftIncluded = s.fullTextResults?.filter(r => r.Decision === "Include") ?? [];
    if (ftIncluded.length)
      return { includedPapers: ftIncluded.map(fromFullTextResult), stage: "full-text screening" };

    // Stage 2 — abstract screening decisions (with reviewer overrides)
    const abIncluded = (s.results ?? [])
      .filter(r => (s.abstractOverrides[r.paper_id] ?? r.Decision) === "INCLUDE");
    if (abIncluded.length)
      return { includedPapers: abIncluded.map(fromScreenResult), stage: "abstract screening" };

    // Stage 3 — LEADS relevance rerank kept set
    const reranked = s.rerankResults?.kept ?? [];
    if (reranked.length)
      return {
        includedPapers: reranked.map(item => ({
          paper_id: item.paper.id,
          title: item.paper.title,
          url: item.paper.url,
          source: item.paper.source,
          abstract: item.paper.abstract,
          year: item.paper.year,
          authors: item.paper.authors,
          doi: extractDoi(item.paper.url) ?? undefined,
        } as PaperMeta)),
        stage: "relevance rerank",
      };

    // Stage 4 — deduplicated retrieval pool
    const unique = s.uniquePapers ?? [];
    if (unique.length)
      return {
        includedPapers: unique.map(p => ({
          paper_id: p.id,
          title: p.title,
          url: p.url,
          source: p.source,
          abstract: p.abstract,
          year: p.year,
          authors: p.authors,
          doi: extractDoi(p.url) ?? undefined,
        } as PaperMeta)),
        stage: "retrieval (all)",
      };

    // Stage 5 — raw retrieved papers
    const raw = s.rawPapers ?? [];
    return {
      includedPapers: raw.map(p => ({
        paper_id: p.id,
        title: p.title,
        url: p.url,
        source: p.source,
        abstract: p.abstract,
        year: p.year,
        authors: p.authors,
        doi: extractDoi(p.url) ?? undefined,
      } as PaperMeta)),
      stage: "retrieval (all)",
    };
  }, [s.fullTextResults, s.results, s.abstractOverrides, s.rerankResults, s.uniquePapers, s.rawPapers]);

  const [enriched, setEnriched] = useState<Record<string, Partial<PaperMeta>>>({});
  const [enriching, setEnriching] = useState(false);
  const [summary, setSummary] = useState("");
  const [generating, setGenerating] = useState(false);

  const merged = useMemo<PaperMeta[]>(
    () => includedPapers.map(p => ({ ...p, ...(enriched[p.paper_id] ?? {}) })),
    [includedPapers, enriched],
  );

  // ── Enrichment ──────────────────────────────────────────────────────────────

  async function enrichAll(papers = includedPapers) {
    setEnriching(true);
    const todo = papers.filter(p => !enriched[p.paper_id]);
    let done = 0;
    for (const p of todo) {
      const data = await fetchMetadata(p);
      if (Object.keys(data).length) {
        setEnriched(prev => ({ ...prev, [p.paper_id]: data }));
        done++;
      }
    }
    if (done) toast.success(`Enriched ${done} of ${todo.length} papers`);
    setEnriching(false);
  }

  // Auto-enrich whenever the paper list changes (on mount and after screening)
  useEffect(() => {
    if (includedPapers.length > 0) enrichAll(includedPapers);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [includedPapers.map(p => p.paper_id).join(",")]);

  // ── Exports ─────────────────────────────────────────────────────────────────

  function allBibTeX(): string {
    return merged.map((p, i) => toBibTeXEntry(p, i + 1)).join("\n\n");
  }

  function allRis(): string {
    return merged.map(toRisEntry).join("\n\n");
  }

  function downloadBib() {
    downloadFile(allBibTeX(), "references.bib", "application/x-bibtex");
    toast.success("Downloaded references.bib");
  }

  function downloadRis() {
    downloadFile(allRis(), "references.ris", "application/x-research-info-systems");
    toast.success("Downloaded references.ris — import into Zotero via File → Import");
  }

  function copyBib() {
    navigator.clipboard.writeText(allBibTeX());
    toast.success("BibTeX copied to clipboard");
  }

  // ── Methods summary ──────────────────────────────────────────────────────────

  async function generateSummary() {
    setGenerating(true);
    setSummary("");
    try {
      // Collect per-database result counts from simulation or PRISMA counts
      const dbCounts: Record<string, number> = s.simulation
        ?? Object.fromEntries(s.sources.map(src => [src, 0]));

      const searchPayload = {
        // Search configuration
        databases: s.sources,
        unified_query: s.unifiedSearchQuery || s.query,
        per_db_queries: s.perDbQueries ?? {},
        search_date: new Date().toISOString().split("T")[0],
        // Result funnel
        db_counts: dbCounts,
        total_identified: s.rawPapers?.length ?? Object.values(dbCounts).reduce((a, b) => a + b, 0),
        duplicates_removed: s.duplicatesCount ?? 0,
        after_dedup: s.uniquePapers?.length ?? null,
        screened_abstracts: s.results?.length ?? null,
        included_abstracts: merged.length,
        fulltext_assessed: s.fullTextResults?.length ?? null,
        included_final: s.fullTextResults?.filter(r => r.Decision === "Include").length ?? null,
        // PICO and criteria
        pico: s.pico,
        inclusion_criteria: s.inclusion,
        exclusion_criteria: s.exclusion,
        goal: s.history[s.history.length - 1]?.query ?? "",
        model: s.model,
      };

      const r = await fetch("/api/writing/summary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(searchPayload),
      });
      if (!r.ok) throw new Error(await r.text());
      const { summary: text } = await r.json();
      setSummary(text);
    } catch (e: any) {
      toast.error(`Summary failed: ${e.message}`);
    } finally {
      setGenerating(false);
    }
  }

  // ── Render ───────────────────────────────────────────────────────────────────

  if (!includedPapers.length) {
    return (
      <Alert>
        <AlertDescription>
          No papers found. Run a search on the Home page first.
        </AlertDescription>
      </Alert>
    );
  }

  const hasDois = merged.some(p => p.doi);
  const enrichedCount = Object.keys(enriched).length;

  return (
    <div className="space-y-4">

      {/* ── Paper pool ── */}
      <Card className="p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-semibold">Included Papers</h2>
            <p className="text-xs text-muted-foreground mt-0.5">
              {merged.length} papers · {stage}
            </p>
          </div>
          <div className="flex gap-2">
            {enriching
              ? <span className="flex items-center gap-1.5 text-xs text-muted-foreground px-2">
                  <Loader2 className="size-3.5 animate-spin" />Fetching metadata…
                </span>
              : <Button variant="outline" size="sm" onClick={() => enrichAll()}>
                  <RefreshCw className="size-3.5 mr-1.5" />Re-fetch metadata
                </Button>
            }
            <Button variant="outline" size="sm" onClick={copyBib}>
              <Copy className="size-3.5 mr-1.5" />BibTeX
            </Button>
            <Button variant="outline" size="sm" onClick={downloadBib}>
              <Download className="size-3.5 mr-1.5" />.bib
            </Button>
            <Button size="sm" onClick={downloadRis}>
              <Download className="size-3.5 mr-1.5" />.ris (Zotero)
            </Button>
          </div>
        </div>

        <div className="rounded-md border overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-6">#</TableHead>
                <TableHead>Title</TableHead>
                <TableHead className="w-28">Source</TableHead>
                <TableHead className="w-16">Year</TableHead>
                <TableHead className="w-48">Authors</TableHead>
                <TableHead className="w-24">DOI</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {merged.map((p, i) => (
                <TableRow key={p.paper_id}>
                  <TableCell className="text-muted-foreground text-xs">{i + 1}</TableCell>
                  <TableCell>
                    <a href={p.url} target="_blank" rel="noreferrer"
                      className="text-sm font-medium hover:underline text-primary line-clamp-2">
                      {p.title}
                    </a>
                  </TableCell>
                  <TableCell>
                    <Badge variant="secondary" className="text-xs">{p.source}</Badge>
                  </TableCell>
                  <TableCell className="text-sm tabular-nums">{p.year ?? "—"}</TableCell>
                  <TableCell className="text-xs text-muted-foreground truncate max-w-[12rem]">
                    {p.authors ?? <span className="italic">not fetched</span>}
                  </TableCell>
                  <TableCell>
                    {p.doi
                      ? <a href={`https://doi.org/${p.doi}`} target="_blank" rel="noreferrer"
                          className="text-xs text-primary hover:underline truncate block max-w-[6rem]">
                          {p.doi}
                        </a>
                      : <span className="text-xs text-muted-foreground">—</span>}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>

        {hasDois && enrichedCount === 0 && (
          <p className="text-xs text-muted-foreground">
            {merged.filter(p => p.doi).length} papers have DOIs — click <strong>Enrich metadata</strong> to fetch authors, journal, volume, and pages from CrossRef.
          </p>
        )}
        {enrichedCount > 0 && (
          <p className="text-xs text-emerald-600">{enrichedCount} papers enriched from CrossRef.</p>
        )}
      </Card>

      {/* ── Methods summary ── */}
      <Card className="p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-semibold">Methods Summary</h2>
            <p className="text-xs text-muted-foreground mt-0.5">
              PRISMA-compliant search strategy paragraph — databases, queries, dates, screening funnel
            </p>
          </div>
          <Button onClick={generateSummary} disabled={generating}>
            {generating
              ? <><Loader2 className="size-4 mr-2 animate-spin" />Generating…</>
              : <><BookOpen className="size-4 mr-2" />Generate with AI</>}
          </Button>
        </div>

        {summary && (
          <>
            <Separator />
            <Textarea
              value={summary}
              onChange={e => setSummary(e.target.value)}
              rows={12}
              className="font-serif text-sm leading-relaxed"
            />
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={() => { navigator.clipboard.writeText(summary); toast.success("Copied"); }}>
                <Copy className="size-3.5 mr-1.5" />Copy
              </Button>
              <Button variant="outline" size="sm"
                onClick={() => downloadFile(summary, "methods_summary.txt", "text/plain")}>
                <Download className="size-3.5 mr-1.5" />Download
              </Button>
            </div>
          </>
        )}
      </Card>

      {/* ── Individual citation entries ── */}
      <Card className="p-4 space-y-2">
        <div className="flex items-center justify-between mb-1">
          <h2 className="font-semibold">Citation Library</h2>
          <Button variant="ghost" size="sm" onClick={copyBib}>
            <Copy className="size-3.5 mr-1.5" />Copy all BibTeX
          </Button>
        </div>
        {merged.map((p, i) => (
          <Collapsible key={p.paper_id}>
            <CollapsibleTrigger asChild>
              <Button variant="outline" className="w-full justify-between text-left h-auto py-2">
                <span className="flex items-center gap-2 min-w-0">
                  <FileText className="size-3.5 shrink-0 text-muted-foreground" />
                  <span className="truncate text-sm">[{i + 1}] {p.title}</span>
                </span>
                <ChevronDown className="size-4 shrink-0 ml-2" />
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="mt-1 rounded-md bg-muted/40 p-3 space-y-2">
                <pre className="text-xs font-mono whitespace-pre-wrap break-words leading-relaxed">
                  {toBibTeXEntry(p, i + 1)}
                </pre>
                <div className="flex gap-2">
                  <Button size="sm" variant="ghost" className="h-7 text-xs"
                    onClick={() => { navigator.clipboard.writeText(toBibTeXEntry(p, i + 1)); toast.success("Copied"); }}>
                    <Copy className="size-3 mr-1" />Copy BibTeX
                  </Button>
                  <Button size="sm" variant="ghost" className="h-7 text-xs"
                    onClick={() => { navigator.clipboard.writeText(toRisEntry(p)); toast.success("Copied RIS"); }}>
                    <Copy className="size-3 mr-1" />Copy RIS
                  </Button>
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>
        ))}
      </Card>

    </div>
  );
}
