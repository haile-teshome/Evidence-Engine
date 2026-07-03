import { Fragment, useMemo, useState, useEffect, type ReactNode } from "react";
import { useStore } from "../lib/store";
import { Card } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Separator } from "../components/ui/separator";
import { Textarea } from "../components/ui/textarea";
import { Input } from "../components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../components/ui/table";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Download, Copy, Loader2, BookOpen, RefreshCw, Search, ExternalLink, Layers, CheckCircle2 } from "lucide-react";
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

// ── Methods appendix rendering (bold **labels**, paragraphs) ──────────────────

function renderInlineBold(text: string): ReactNode {
  // Split on **bold** spans and render the bold parts as <strong>.
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    const m = part.match(/^\*\*([^*]+)\*\*$/);
    return m ? <strong key={i}>{m[1]}</strong> : <Fragment key={i}>{part}</Fragment>;
  });
}

function renderMethodsAppendix(text: string): ReactNode {
  const blocks = text.trim().split(/\n{2,}/).filter(Boolean);
  const paras = blocks.length > 1 ? blocks : text.trim().split(/\n/).filter(Boolean);
  return paras.map((p, i) => (
    <p key={i} className="text-sm leading-relaxed">{renderInlineBold(p.trim())}</p>
  ));
}

// ── Plain-text citation styles ────────────────────────────────────────────────

function authorList(meta: PaperMeta): string[] {
  return (meta.authors || "").split(/\s*;\s*/).map(a => a.trim()).filter(Boolean);
}

function citationAPA(m: PaperMeta): string {
  const a = authorList(m);
  const auth = a.length ? (a.length > 1 ? a.slice(0, -1).join(", ") + ", & " + a[a.length - 1] : a[0]) : (m.source || "Anonymous");
  const yr = m.year ? ` (${m.year}).` : " (n.d.).";
  const jrnl = m.journal
    ? ` ${m.journal}${m.volume ? `, ${m.volume}` : ""}${m.issue ? `(${m.issue})` : ""}${m.pages ? `, ${m.pages}` : ""}.`
    : "";
  const link = m.doi ? ` https://doi.org/${m.doi}` : (m.url ? ` ${m.url}` : "");
  return `${auth}${yr} ${m.title}.${jrnl}${link}`.replace(/\s+/g, " ").trim();
}

function citationMLA(m: PaperMeta): string {
  const a = authorList(m);
  const auth = a.length ? a.join(", ") + "." : "";
  const jrnl = m.journal
    ? ` ${m.journal}${m.volume ? `, vol. ${m.volume}` : ""}${m.issue ? `, no. ${m.issue}` : ""}`
    : "";
  const yr = m.year ? `, ${m.year}` : "";
  const pg = m.pages ? `, pp. ${m.pages}` : "";
  const link = m.doi ? `, doi:${m.doi}` : (m.url ? `, ${m.url}` : "");
  return `${auth} "${m.title}."${jrnl}${yr}${pg}${link}.`.replace(/\s+/g, " ").trim();
}

function citationVancouver(m: PaperMeta, idx: number): string {
  const a = authorList(m);
  const auth = a.length ? a.join(", ") + "." : "";
  const jrnl = m.journal ? ` ${m.journal}.` : "";
  const tail = `${m.year ? ` ${m.year}` : ""}${m.volume ? `;${m.volume}` : ""}${m.issue ? `(${m.issue})` : ""}${m.pages ? `:${m.pages}` : ""}`;
  return `${idx}. ${auth} ${m.title}.${jrnl}${tail ? `${tail}.` : ""}`.replace(/\s+/g, " ").trim();
}

function citationChicago(m: PaperMeta): string {
  const a = authorList(m);
  const auth = a.length ? a.join(", ") + "." : "";
  const yr = m.year ? ` ${m.year}.` : " n.d.";
  const jrnl = m.journal
    ? ` ${m.journal}${m.volume ? ` ${m.volume}` : ""}${m.issue ? `, no. ${m.issue}` : ""}${m.pages ? `: ${m.pages}` : ""}.`
    : "";
  const link = m.doi ? ` https://doi.org/${m.doi}.` : "";
  return `${auth}${yr} "${m.title}."${jrnl}${link}`.replace(/\s+/g, " ").trim();
}

type CiteFormat = "BibTeX" | "RIS" | "APA" | "MLA" | "Vancouver" | "Chicago";

const CITE_FORMATS: { value: CiteFormat; ext: string; mime: string; joiner: string }[] = [
  { value: "BibTeX",    ext: "bib", mime: "application/x-bibtex", joiner: "\n\n" },
  { value: "RIS",       ext: "ris", mime: "application/x-research-info-systems", joiner: "\n\n" },
  { value: "APA",       ext: "txt", mime: "text/plain", joiner: "\n\n" },
  { value: "MLA",       ext: "txt", mime: "text/plain", joiner: "\n\n" },
  { value: "Vancouver", ext: "txt", mime: "text/plain", joiner: "\n" },
  { value: "Chicago",   ext: "txt", mime: "text/plain", joiner: "\n\n" },
];

function formatCitation(m: PaperMeta, fmt: CiteFormat, idx: number): string {
  switch (fmt) {
    case "RIS":       return toRisEntry(m);
    case "APA":       return citationAPA(m);
    case "MLA":       return citationMLA(m);
    case "Vancouver": return citationVancouver(m, idx);
    case "Chicago":   return citationChicago(m);
    case "BibTeX":
    default:          return toBibTeXEntry(m, idx);
  }
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

  // Enrichment cache + generated summary live in the store so switching tabs
  // (which unmounts this page) doesn't re-fetch metadata every time.
  const enriched = s.writingEnriched;
  const setEnriched = s.setWritingEnriched;
  const summary = s.writingSummary;
  const setSummary = s.setWritingSummary;
  const [enriching, setEnriching] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [format, setFormat] = useState<CiteFormat>("BibTeX");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [searchDate, setSearchDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [editingSummary, setEditingSummary] = useState(false);

  const merged = useMemo<PaperMeta[]>(
    () => includedPapers.map(p => ({ ...p, ...(enriched[p.paper_id] ?? {}) })),
    [includedPapers, enriched],
  );

  // Number every paper by its stable position, then filter for display so a
  // search never renumbers the citations.
  const rows = useMemo(() => merged.map((p, i) => ({ p, n: i + 1 })), [merged]);
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter(({ p }) =>
      [p.title, p.authors, p.journal, p.source, p.doi, p.year != null ? String(p.year) : ""]
        .some(f => (f || "").toLowerCase().includes(q)),
    );
  }, [rows, query]);

  const selectedRow = rows.find(r => r.p.paper_id === selectedId) ?? rows[0] ?? null;

  // ── Enrichment ──────────────────────────────────────────────────────────────

  async function enrichAll(papers = includedPapers, force = false) {
    const todo = papers.filter(p => force || !enriched[p.paper_id]);
    if (todo.length === 0) return;   // already cached — no refetch
    setEnriching(true);
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

  const fmtCfg = CITE_FORMATS.find(f => f.value === format) ?? CITE_FORMATS[0];

  function allFormatted(): string {
    return merged.map((p, i) => formatCitation(p, format, i + 1)).join(fmtCfg.joiner);
  }

  function downloadAll() {
    downloadFile(allFormatted(), `references.${fmtCfg.ext}`, fmtCfg.mime);
    toast.success(
      format === "RIS"
        ? "Downloaded references.ris — import into Zotero via File → Import"
        : `Downloaded references.${fmtCfg.ext}`,
    );
  }

  function copyAll() {
    navigator.clipboard.writeText(allFormatted());
    toast.success(`All ${merged.length} citations copied as ${format}`);
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
        // RAISE: stages where AI made/suggested judgements (for the AI-use subsection)
        ai_model: modelName,
        ai_steps: aiSteps.map(st => ({ stage: st.stage, purpose: st.purpose, oversight: st.oversight })),
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

  // ── PRISMA-S search-strategy data (deterministic, from stored data) ──────────
  const apiDbs = s.sources.filter(x => x !== "Local PDFs");
  const baseQuery = (s.unifiedSearchQuery || s.query || "").trim();
  const countOf = (src: string): number | null => {
    if (s.simulation && s.simulation[src] != null) return s.simulation[src];
    if (s.dbTestResults?.[src]?.total_found != null) return s.dbTestResults[src].total_found;
    if (s.rawPapers) { const c = s.rawPapers.filter(p => p.source === src).length; return c || null; }
    return null;
  };
  const searchRows = apiDbs.map(src => ({
    db: src,
    query: (s.perDbQueries[src] || baseQuery || "").trim(),
    count: countOf(src),
  }));
  const totalRetrieved = searchRows.reduce((a, r) => a + (r.count ?? 0), 0);
  const identified = s.rawPapers?.length ?? (searchRows.some(r => r.count != null) ? totalRetrieved : null);
  const dupRemoved = s.duplicatesCount ?? 0;
  const afterDedup = s.uniquePapers?.length ?? (identified != null ? Math.max(0, identified - dupRemoved) : null);

  function buildSearchAppendix(): string {
    const L: string[] = [];
    L.push("SEARCH STRATEGY (PRISMA-S)");
    L.push("");
    L.push(`Databases searched (${apiDbs.length}): ${apiDbs.join(", ") || "none"}.`);
    if (s.sources.includes("Local PDFs")) L.push(`Additional records: uploaded local PDFs (${s.files.length}).`);
    L.push(`Search date: ${searchDate}.`);
    if (baseQuery) {
      L.push("");
      L.push("Base query (applied to each database unless a custom string is shown below):");
      L.push(baseQuery);
    }
    L.push("");
    L.push("Per-database searches");
    L.push("─────────────────────");
    for (const r of searchRows) {
      L.push("");
      L.push(r.db);
      L.push(`  Query run:         ${r.query || "(not specified)"}`);
      L.push(`  Date searched:     ${searchDate}`);
      L.push(`  Records retrieved: ${r.count != null ? r.count.toLocaleString() : "not run"}`);
    }
    L.push("");
    L.push("Identification");
    L.push("──────────────");
    if (identified != null) L.push(`  Records identified across databases: ${identified.toLocaleString()}`);
    L.push(`  Duplicate records removed:           ${dupRemoved.toLocaleString()}`);
    if (afterDedup != null) L.push(`  Records after de-duplication:         ${afterDedup.toLocaleString()}`);
    return L.join("\n");
  }

  function appendixCsv(): string {
    const esc = (v: string) => `"${String(v).replace(/"/g, '""')}"`;
    const rows = [["Database", "Query", "Date searched", "Records retrieved"]];
    for (const r of searchRows) rows.push([r.db, r.query, searchDate, r.count != null ? String(r.count) : "not run"]);
    return rows.map(row => row.map(esc).join(",")).join("\n");
  }

  // ── RAISE AI use disclosure (grounded in stages that actually ran) ───────────
  // RAISE requires transparent reporting of any AI use that makes or suggests
  // judgements (eligibility, appraisal, extraction, synthesis, drafting).
  const modelName = s.model || "the configured large language model";
  const aiSteps = useMemo(() => {
    const steps: { stage: string; purpose: string; judgement: boolean; oversight: string }[] = [];
    const has = (v: any) => Array.isArray(v) ? v.length > 0 : !!v;
    if (has(s.unifiedSearchQuery) || has(s.simulationRuns) || has(s.history))
      steps.push({ stage: "Search strategy development", purpose: "Suggested and expanded database search strings (synonyms, controlled vocabulary).", judgement: true, oversight: "Search strings were reviewed and editable by the reviewer before execution." });
    if (has(s.rerankResults))
      steps.push({ stage: "Relevance ranking", purpose: "Ranked retrieved records by relevance to the review question.", judgement: true, oversight: "Used to prioritise screening order; no records were excluded on this basis alone." });
    if (has(s.results)) {
      const ov = Object.keys(s.abstractOverrides || {}).length;
      steps.push({ stage: "Title and abstract screening", purpose: "Suggested eligibility decisions (include/exclude) with reasons from title/abstract.", judgement: true, oversight: `Reviewer-facing decisions with override capability${ov ? `; ${ov} AI decision(s) were overridden by a reviewer` : ""}.` });
    }
    if (has(s.snowballScreened))
      steps.push({ stage: "Citation snowballing screening", purpose: "Suggested eligibility of references identified via backward/forward citation searching.", judgement: true, oversight: "Reviewer selected which records to carry forward." });
    if (has(s.fullTextResults)) {
      const ov = Object.keys(s.fullTextOverrides || {}).length;
      steps.push({ stage: "Full-text eligibility assessment", purpose: "Suggested full-text eligibility against inclusion/exclusion criteria, with reasons.", judgement: true, oversight: `Reviewer-facing decisions with override capability${ov ? `; ${ov} AI decision(s) were overridden by a reviewer` : ""}.` });
    }
    if (has(s.qualityReports)) {
      const ov = (s.qualityOverrides || []).length;
      steps.push({ stage: "Risk-of-bias / quality assessment", purpose: "Suggested domain-level risk-of-bias / methodological appraisal judgements.", judgement: true, oversight: `Each AI judgement was reviewer-editable${ov ? `; ${ov} judgement(s) were overridden by a reviewer` : ""}.` });
    }
    if (has(s.extractedPapers) || has(s.textExtractions))
      steps.push({ stage: "Data extraction", purpose: "Extracted numerical and/or qualitative data items from study reports.", judgement: true, oversight: "Extracted values were displayed against the source for reviewer verification and correction." });
    if (has(s.writingSummary))
      steps.push({ stage: "Manuscript drafting", purpose: "Drafted the methods / search-strategy appendix text.", judgement: true, oversight: "Draft text was reviewer-editable before use." });
    return steps;
  }, [s.unifiedSearchQuery, s.simulationRuns, s.history, s.rerankResults, s.results, s.abstractOverrides, s.snowballScreened, s.fullTextResults, s.fullTextOverrides, s.qualityReports, s.qualityOverrides, s.extractedPapers, s.textExtractions, s.writingSummary]);

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

  const enrichedCount = Object.keys(enriched).length;

  return (
    <div className="space-y-3">
      {/* ── Compact header: stats + bulk export ─────────────────────────────── */}
      <Card className="p-3">
        <div className="flex items-end gap-3 flex-wrap">
          <div className="mr-auto min-w-0">
            <h2 className="font-medium leading-tight">Writing Assistant</h2>
            <div className="flex flex-wrap items-center gap-1.5 mt-1">
              <Pill icon={BookOpen}>{merged.length} papers</Pill>
              <Pill icon={Layers} title="Pipeline stage these papers came from">{stage}</Pill>
              {enrichedCount > 0 && <Pill icon={CheckCircle2} tone="green" title="Full metadata fetched (cached this session)">{enrichedCount} enriched</Pill>}
              {enriching && <Pill icon={Loader2} spin>fetching metadata…</Pill>}
            </div>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <Button variant="outline" size="sm" onClick={() => enrichAll(includedPapers, true)} disabled={enriching}>
              <RefreshCw className="size-3.5 mr-1.5" />Re-fetch
            </Button>
            <Select value={format} onValueChange={v => setFormat(v as CiteFormat)}>
              <SelectTrigger className="h-9 w-[7.5rem]"><SelectValue /></SelectTrigger>
              <SelectContent>
                {CITE_FORMATS.map(f => <SelectItem key={f.value} value={f.value}>{f.value}</SelectItem>)}
              </SelectContent>
            </Select>
            <Button variant="outline" size="sm" onClick={copyAll}>
              <Copy className="size-3.5 mr-1.5" />Copy all
            </Button>
            <Button size="sm" onClick={downloadAll}>
              <Download className="size-3.5 mr-1.5" />.{fmtCfg.ext}
            </Button>
          </div>
        </div>
      </Card>

      <Tabs defaultValue="citations">
        <TabsList>
          <TabsTrigger value="citations" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-sm">Citations</TabsTrigger>
          <TabsTrigger value="methods" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-sm">Search strategy</TabsTrigger>
        </TabsList>

        {/* ── Citations: searchable paper list (left) + citation detail (right) ── */}
        <TabsContent value="citations" className="mt-3">
          <div className="flex gap-4 h-[calc(100vh-15rem)] min-h-[28rem]">
            <Card className="w-80 shrink-0 p-0 overflow-hidden flex flex-col">
              <div className="p-2 border-b">
                <div className="relative">
                  <Search className="size-3.5 absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    placeholder={`Filter ${merged.length} papers…`}
                    className="pl-7 h-8 text-sm"
                  />
                </div>
              </div>
              <div className="overflow-auto flex-1">
                {filtered.map(({ p, n }) => {
                  const active = p.paper_id === selectedRow?.p.paper_id;
                  return (
                    <button
                      key={p.paper_id}
                      onClick={() => setSelectedId(p.paper_id)}
                      className={`w-full text-left px-3 py-2.5 border-b transition-colors ${active ? "bg-primary/10 border-l-2 border-l-primary" : "border-l-2 border-l-transparent hover:bg-muted/50"}`}
                    >
                      <div className="flex items-center gap-2 mb-1 text-[10px] text-muted-foreground">
                        <span className="tabular-nums">[{n}]</span>
                        <Badge variant="outline" className="text-[10px]">{p.source}</Badge>
                        {p.year && <span className="tabular-nums">{p.year}</span>}
                      </div>
                      <div className="text-sm leading-snug line-clamp-2 max-h-[2.75em] overflow-hidden">{p.title}</div>
                    </button>
                  );
                })}
                {filtered.length === 0 && (
                  <div className="p-4 text-sm text-muted-foreground">No papers match “{query}”.</div>
                )}
              </div>
            </Card>

            <Card className="flex-1 min-w-0 p-0 overflow-hidden flex flex-col">
              {!selectedRow ? (
                <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">Select a paper on the left.</div>
              ) : (
                <CitationDetail row={selectedRow} format={format} enriching={enriching} />
              )}
            </Card>
          </div>
        </TabsContent>

        {/* ── Search strategy (PRISMA-S) — per-database table ── */}
        <TabsContent value="methods" className="mt-3 space-y-3">
          <Card className="p-4 space-y-4">
            <div className="flex items-start justify-between gap-3 flex-wrap">
              <div>
                <h3 className="font-medium">Search strategy (PRISMA-S)</h3>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Per-database query, date run, and records retrieved — report this as a supplement.
                </p>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <label className="text-xs text-muted-foreground flex items-center gap-1.5">
                  Search date
                  <Input type="date" value={searchDate} onChange={e => setSearchDate(e.target.value)} className="h-8 w-[9.5rem] text-sm" />
                </label>
                <Button variant="outline" size="sm" onClick={() => { navigator.clipboard.writeText(buildSearchAppendix()); toast.success("Copied appendix"); }}>
                  <Copy className="size-3.5 mr-1.5" />Copy
                </Button>
                <Button variant="outline" size="sm" onClick={() => downloadFile(buildSearchAppendix(), "search_strategy.txt", "text/plain")}>
                  <Download className="size-3.5 mr-1.5" />.txt
                </Button>
                <Button variant="outline" size="sm" onClick={() => downloadFile(appendixCsv(), "search_strategy.csv", "text/csv")}>
                  <Download className="size-3.5 mr-1.5" />.csv
                </Button>
              </div>
            </div>

            {baseQuery && (
              <div>
                <div className="text-xs font-semibold text-muted-foreground mb-1.5">Base query (applied unless a database has a custom string)</div>
                <pre className="rounded-md border bg-muted/30 p-3 text-xs font-mono whitespace-pre-wrap break-words leading-relaxed">{baseQuery}</pre>
              </div>
            )}

            <div className="rounded-md border overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-40">Database</TableHead>
                    <TableHead>Query run</TableHead>
                    <TableHead className="w-28">Date</TableHead>
                    <TableHead className="w-24 text-right">Records</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {searchRows.map(r => (
                    <TableRow key={r.db} className="align-top">
                      <TableCell className="font-medium">{r.db}</TableCell>
                      <TableCell>
                        <pre className="font-mono text-xs whitespace-pre-wrap break-words leading-relaxed max-w-[44rem]">{r.query || "(not specified)"}</pre>
                      </TableCell>
                      <TableCell className="text-sm tabular-nums whitespace-nowrap">{searchDate}</TableCell>
                      <TableCell className="text-right text-sm tabular-nums">
                        {r.count != null ? r.count.toLocaleString() : <span className="text-muted-foreground">not run</span>}
                      </TableCell>
                    </TableRow>
                  ))}
                  {searchRows.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={4} className="text-sm text-muted-foreground text-center py-6">
                        No databases selected. Pick sources in the sidebar.
                      </TableCell>
                    </TableRow>
                  )}
                  {searchRows.length > 0 && (
                    <>
                      <TableRow className="border-t-2 bg-muted/30">
                        <TableCell colSpan={3} className="font-medium">Total records retrieved</TableCell>
                        <TableCell className="text-right text-sm font-semibold tabular-nums">
                          {identified != null ? identified.toLocaleString() : "—"}
                        </TableCell>
                      </TableRow>
                      <TableRow className="bg-muted/30">
                        <TableCell colSpan={3} className="font-medium">Duplicates removed</TableCell>
                        <TableCell className="text-right text-sm font-semibold tabular-nums">{dupRemoved.toLocaleString()}</TableCell>
                      </TableRow>
                      <TableRow className="bg-muted/30">
                        <TableCell colSpan={3} className="font-medium">Unique records screened</TableCell>
                        <TableCell className="text-right text-sm font-semibold tabular-nums">
                          {afterDedup != null ? afterDedup.toLocaleString() : "—"}
                        </TableCell>
                      </TableRow>
                    </>
                  )}
                </TableBody>
              </Table>
            </div>
          </Card>

          {/* Search-strategy methods appendix (Design / Sources / Eligibility / Selection / Synthesis) */}
          <Card className="p-4 space-y-3">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div>
                <h3 className="font-medium">Search strategy methods (appendix)</h3>
                <p className="text-xs text-muted-foreground mt-0.5">Prose methods appendix: design &amp; scope, data sources, eligibility, study selection, synthesis, and a RAISE-compliant AI &amp; automation use declaration.</p>
              </div>
              <Button size="sm" variant="outline" onClick={generateSummary} disabled={generating}>
                {generating
                  ? <><Loader2 className="size-4 mr-2 animate-spin" />Generating…</>
                  : <><BookOpen className="size-4 mr-2" />{summary ? "Regenerate" : "Generate with AI"}</>}
              </Button>
            </div>
            {summary && (
              <>
                <Separator />
                {editingSummary ? (
                  <Textarea value={summary} onChange={e => setSummary(e.target.value)} rows={16} className="font-mono text-xs leading-relaxed" />
                ) : (
                  <div className="space-y-3 font-serif">{renderMethodsAppendix(summary)}</div>
                )}
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={() => setEditingSummary(v => !v)}>
                    {editingSummary ? "Done" : "Edit"}
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => { navigator.clipboard.writeText(summary); toast.success("Copied"); }}>
                    <Copy className="size-3.5 mr-1.5" />Copy
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => downloadFile(summary, "methods_appendix.txt", "text/plain")}>
                    <Download className="size-3.5 mr-1.5" />Download
                  </Button>
                </div>
              </>
            )}
          </Card>

        </TabsContent>
      </Tabs>
    </div>
  );
}

// ── Selected-citation detail (right pane) ─────────────────────────────────────
function CitationDetail({ row, format, enriching }: { row: { p: PaperMeta; n: number }; format: CiteFormat; enriching: boolean }) {
  const { p, n } = row;
  const meta: { label: string; value?: string }[] = [
    { label: "Authors", value: p.authors },
    { label: "Journal", value: p.journal },
    { label: "Year", value: p.year != null ? String(p.year) : undefined },
    { label: "Volume", value: [p.volume, p.issue && `(${p.issue})`].filter(Boolean).join("") || undefined },
    { label: "Pages", value: p.pages },
  ].filter(m => m.value);

  return (
    <>
      <div className="border-b p-4 space-y-2">
        <div className="flex items-start gap-2">
          <span className="text-xs text-muted-foreground tabular-nums shrink-0 pt-0.5">[{n}]</span>
          <a href={p.url} target="_blank" rel="noreferrer" className="font-medium leading-snug hover:underline inline-flex items-start gap-1">
            <span>{p.title}</span>
            <ExternalLink className="size-3 mt-1 shrink-0 text-muted-foreground" />
          </a>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <Badge variant="secondary">{p.source}</Badge>
          {p.doi && (
            <a href={`https://doi.org/${p.doi}`} target="_blank" rel="noreferrer" className="text-primary hover:underline">doi:{p.doi}</a>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-4">
        <div>
          <div className="text-xs font-semibold text-muted-foreground mb-2">Metadata</div>
          {meta.length > 0 ? (
            <dl className="grid grid-cols-[5rem_1fr] gap-x-3 gap-y-1 text-sm">
              {meta.map(m => (
                <Fragment key={m.label}>
                  <dt className="text-muted-foreground">{m.label}</dt>
                  <dd className="min-w-0 break-words">{m.value}</dd>
                </Fragment>
              ))}
            </dl>
          ) : (
            <p className="text-sm text-muted-foreground italic">
              {enriching ? "Fetching metadata…" : "No author/journal metadata available for this record."}
            </p>
          )}
        </div>

        <div>
          <div className="flex items-center justify-between mb-1.5">
            <div className="text-xs font-semibold text-muted-foreground">{format} citation</div>
            <div className="flex gap-1">
              <Button size="sm" variant="ghost" className="h-7 text-xs"
                onClick={() => { navigator.clipboard.writeText(formatCitation(p, format, n)); toast.success(`Copied ${format}`); }}>
                <Copy className="size-3 mr-1" />Copy {format}
              </Button>
              {format !== "BibTeX" && (
                <Button size="sm" variant="ghost" className="h-7 text-xs"
                  onClick={() => { navigator.clipboard.writeText(toBibTeXEntry(p, n)); toast.success("Copied BibTeX"); }}>BibTeX</Button>
              )}
              {format !== "RIS" && (
                <Button size="sm" variant="ghost" className="h-7 text-xs"
                  onClick={() => { navigator.clipboard.writeText(toRisEntry(p)); toast.success("Copied RIS"); }}>RIS</Button>
              )}
            </div>
          </div>
          <pre className="rounded-md border bg-muted/30 p-3 text-xs font-mono whitespace-pre-wrap break-words leading-relaxed">
            {formatCitation(p, format, n)}
          </pre>
        </div>
      </div>
    </>
  );
}

// Compact count pill (matches the other pages).
function Pill({
  icon: Icon, children, tone = "default", title, spin = false,
}: {
  icon: React.ComponentType<{ className?: string }>;
  children: React.ReactNode;
  tone?: "default" | "green";
  title?: string;
  spin?: boolean;
}) {
  const cls = tone === "green"
    ? "bg-emerald-50 text-emerald-700 border-emerald-200"
    : "bg-muted text-muted-foreground border-transparent";
  return (
    <span title={title} className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${cls}`}>
      <Icon className={`size-3 ${spin ? "animate-spin" : ""}`} />{children}
    </span>
  );
}
