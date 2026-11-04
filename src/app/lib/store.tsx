import { createContext, useContext, useEffect, useRef, useState, ReactNode } from "react";
import { Pico, Analysis, ScreenResult, FullTextResult, Paper, QualityReport, QualityOverride, GradeOutcome } from "./mockServices";
import type { ExtractedTable } from "./apiClient";
import { apiConfig, RerankResult, StudyEffect, MetaRunResult, EffectMeasure, Tau2Method, SearchLogEntry, ReviewProtocol, ProtocolDeviation, SelectionStrategy, SearchFilters } from "./apiClient";
import { FRESH_LAUNCH } from "./launchFlags";
import { idbGet, idbSet, idbDel } from "./idb";
import { compactFullTexts, compactArchive, expandArchive, compactRawPapers, expandRawPapers } from "./snapshotCompact";
import { fetchFullTextBody, putFullTextBody } from "./fulltextCache";
import type { FrameworkId } from "./frameworks";

export type PageId = "home" | "simulation" | "quality" | "abstract" | "acquisition" | "fulltext" | "snowball" | "extraction" | "textextraction" | "prisma" | "meta" | "projects" | "writing";

export type SimulationRun = {
  id: string;
  timestamp: number;
  label: string;
  unifiedQuery: string;
  perDbQueries: Record<string, string>;
  counts: Record<string, number>;
  totalYield: number;
  source: "manual" | "ai-optimize";
};

/** The planning run screening should use.
 *
 *  One resolver, used by both the Planning page and Abstract Screening, so the
 *  run shown as "in use" is provably the run screening retrieves with. It used
 *  to be inferred by matching query text against the editor, which meant an
 *  edit made after a run silently detached the counts from the retrieval and
 *  the two pages could disagree about which plan was live.
 *
 *  Defaults to the most recent run; an explicit `inUseRunId` pins another. */
export function resolveInUseRun(
  runs: SimulationRun[] | null | undefined,
  inUseRunId: string | null | undefined,
): SimulationRun | null {
  const list = runs || [];
  if (!list.length) return null;
  return (inUseRunId && list.find(r => r.id === inUseRunId)) || list[list.length - 1];
}

/** How a snowball run selected what it kept, and what it left behind. */
export type SnowballPlan = {
  /** Which ordering decided the survivors. */
  sort: SnowballSort;
  /** The per-seed cap the reviewer set. */
  maxPerSeed: number;
  direction: string;
  sources: string[];
  /** Per seed: how many its sources hold, how many were pulled, how many kept. */
  seeds: { title: string; available: number; fetched: number; kept: number }[];
  /** The backend's per-source fetch ceiling, so a capped pull is attributable. */
  ceiling: number;
  ranAt: string;
};

export type SnowballSort = "relevance" | "cited" | "recent" | "source";

export type FullTextRecord = {
  paper_id: string;
  title: string;
  url: string;
  source: string;                       // paper's originating database
  status: "found" | "missing" | "pending";
  text?: string;
  /** Key of this body in the backend text cache. When set, the saved session omits
   *  `text` and the store pulls the body back from the cache after a load. */
  text_key?: string;
  reason?: string;
  /** Path to the acquired PDF cached by the backend (relative to the API base),
   *  when the full text came from a real PDF source. Absent for XML/HTML tiers. */
  pdf_url?: string;
  /** Tables recognised in this full text, when a caller has attached them. Nothing
   *  populates this yet: tables come from the separate /extract/tables endpoint and
   *  live on ExtractedPaper, so ExtractionWorkspace's table context is always empty.
   *  Declared so that call site type-checks; wiring it up is a behaviour change. */
  tables?: ExtractedTable[];
  /** Where the full text was retrieved from on this fetch: e.g.
   *  "Europe PMC (XML)", "PMC PDF (PMC1234567)", "Open access (nature.com)",
   *  "arXiv PDF (2304.12345)", "HTML scrape (publisher.com)". Different from
   *  `source` (which is the paper's database). */
  retrieved_via?: string;
  /** Why a fetch came back missing, as a machine-readable class so the UI can
   *  tell "paywalled, go to the library" apart from "we could not resolve it".
   *  One of: no_doi | paywalled | oa_blocked | unresolved. */
  reason_code?: string;
  /** Open-access state from the OA index: pdf | landing | closed | unknown. */
  oa_status?: string;
  doi?: string;
  pmid?: string;
  /** Ready-made places a human can go to get this paper by hand. */
  links?: Record<string, string>;
};
export type TextEvidenceItem = {
  quote: string;
  why?: string;
  section?: string;
  start: number;
  end: number;
};
export type TextExtractedValue = {
  field: string;
  value: string;
  quote?: string;
  section?: string;
  start?: number;
  end?: number;
};
export type TextExtractionResult = {
  paper_id: string;
  title: string;
  query: string;
  answer?: string;                                        // synthesised natural-language answer
  summary: string;                                        // back-compat alias for answer
  evidence?: TextEvidenceItem[];                          // new structured evidence list
  spans: { start: number; end: number; label?: string }[]; // legacy spans (for back-compat with renderHighlighted)
  values: TextExtractedValue[];
};

export type HistoryEntry = {
  goal: string;
  query: string;
  formal_question: string;
  summary: string;
  references?: { title: string; url: string; source: string; id: string }[];
  pico_dict: Analysis;
  suggestions: string[];
  inclusion: string[];
  exclusion: string[];
  adversarial_query: string;
  ts?: number;                   // creation time, so search and chat turns can be interleaved chronologically
  // Deep-scan adaptive round: the supplementary query the agent ran (or the user
  // edited), why, and what it added. Present only when deep scan was on.
  deep_scan?: { query: string; rationale: string; tactic: string; retrieved: number; added: number };
};

// A document Q&A exchange on Home: a question answered from the documents in
// play, with the sources cited. Persisted so it survives a refresh like history.
export type SummaryRow = { id: string; title: string; design: string; population: string; intervention: string; comparator: string; outcomes: string; key_finding: string };
export type DocQaTurn = { question: string; answer: string; sources: { n: number; id: string; title: string }[]; busy?: boolean; status?: string; note?: string; ts?: number; table?: SummaryRow[] };

// Re-exported rather than redeclared. A second, narrower definition here would
// typecheck fine and silently drop the page/tier provenance the backend now
// sends, which is the whole point of extracting tables deterministically.
export type { ExtractedTable } from "./apiClient";
export type ExtractedPaper = { Paper_Title: string; Paper_URL: string; Source: string; Extracted_Tables: ExtractedTable[] };

// One archived title/abstract screening pass, kept so prior runs can be
// compared or restored. `results` is the full result set after the pass, so a
// pass can be restored wholesale. Incremental passes only re-screen papers not
// already screened (e.g. records added later from citation snowballing).
export type ScreeningPass = {
  id: string;
  ranAt: string;                 // ISO timestamp
  mode: "full" | "incremental";
  model: string;
  screenedNow: number;           // papers screened in this pass
  totalAfter: number;            // total records after this pass
  included: number;              // AI includes among the total
  excluded: number;
  durationSec: number;
  results: ScreenResult[];       // snapshot of the full set (for restore/compare)
  /** Set only inside a saved session: this pass's results ARE the live results
   *  table, so the copy is left out and put back on load (see snapshotCompact). */
  resultsRef?: "current";
  label?: string;                // optional user nickname for the run
};

export type TaskStage = {
  id: string;
  label: string;
  status: "pending" | "running" | "done" | "error" | "canceled";
  detail?: string;
};

export type TaskKind =
  | "home-analysis"
  | "ai-optimize"
  | "quality-assess"
  | "abstract-screen"
  | "retrieval"
  | "fulltext-fetch"
  | "full-text-screen"
  | "snowball"
  | "snowball-screen"
  | "table-extract"
  | "text-extract";

export type TaskRecord = {
  kind: TaskKind;
  taskId: string;     // Server-side cancel handle. Sent to endpoints that support
                      // mid-flight cancellation (currently the agentic optimizer).
  status: "running" | "done" | "canceled" | "error";
  startedAt: number;
  stages: TaskStage[];
  log: string[];
  detail?: string;
  progress?: { done: number; total: number; label?: string };
  // Non-serialised: AbortController for the in-flight HTTP call(s). Kept off snapshots.
  abort?: AbortController;
};

type Ctx = {
  // Sidebar
  page: PageId; setPage: (p: PageId) => void;
  // Transient UI: whether the Home page's Strategy Review drawer is open (lives
  // in the store so the header bar can toggle it). Not persisted.
  reviewOpen: boolean; setReviewOpen: (v: boolean) => void;
  model: string; setModel: (v: string) => void;
  sources: string[]; setSources: (v: string[]) => void;
  numPerSource: number; setNumPerSource: (v: number) => void;
  // Per-database paper cap overrides (source name -> max). Sources without an
  // entry fall back to numPerSource. Edited in the Strategy Review modal.
  perSourceLimits: Record<string, number>; setPerSourceLimits: React.Dispatch<React.SetStateAction<Record<string, number>>>;
  // Which records to keep when a source matches more than its cap: best match
  // ("relevance", default) or newest ("recent"). Chosen in the Review modal.
  searchSelection: SelectionStrategy; setSearchSelection: (v: SelectionStrategy) => void;
  // Researcher-set search filters (e.g. publication-year window). Applied at
  // fetch time in the main search.
  searchFilters: SearchFilters; setSearchFilters: React.Dispatch<React.SetStateAction<SearchFilters>>;
  // Manual include/exclude overrides on the relevance rerank, keyed by paper id.
  // "keep"/"drop" override the auto threshold decision in the Relevance explorer.
  relevanceOverrides: Record<string, "keep" | "drop">; setRelevanceOverrides: React.Dispatch<React.SetStateAction<Record<string, "keep" | "drop">>>;
  files: File[]; setFiles: (v: File[]) => void;

  // Strategy
  history: HistoryEntry[]; setHistory: React.Dispatch<React.SetStateAction<HistoryEntry[]>>;
  docQa: DocQaTurn[]; setDocQa: React.Dispatch<React.SetStateAction<DocQaTurn[]>>;
  pico: Pico; setPico: React.Dispatch<React.SetStateAction<Pico>>;
  framework: FrameworkId; setFramework: (v: FrameworkId) => void;
  inclusion: string[]; setInclusion: (v: string[]) => void;
  exclusion: string[]; setExclusion: (v: string[]) => void;
  query: string; setQuery: (v: string) => void;

  // Simulation
  unifiedSearchQuery: string; setUnifiedSearchQuery: (v: string) => void;
  perDbQueries: Record<string, string>; setPerDbQueries: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  simulation: Record<string, number> | null; setSimulation: (v: Record<string, number> | null) => void;
  /** The query those yield counts were measured against.
   *
   *  Without it, a planned-vs-retrieved gap is unattributable: the counts are
   *  frozen at Planning time while the fetch uses whatever query is current, so
   *  editing the query afterwards makes every source look like it errored. */
  simulationQuery: string | null; setSimulationQuery: (v: string | null) => void;
  dbTestResults: Record<string, { query: string; total_found: number; papers: { title: string; url: string }[] }> | null;
  setDbTestResults: React.Dispatch<React.SetStateAction<Record<string, { query: string; total_found: number; papers: { title: string; url: string }[] }> | null>>;
  agenticTrace: any[] | null; setAgenticTrace: (v: any[] | null) => void;
  agenticSummary: { iterations_run: number; total_papers_found: number; best_relevance: number } | null;
  setAgenticSummary: (v: { iterations_run: number; total_papers_found: number; best_relevance: number } | null) => void;
  simulationRuns: SimulationRun[];
  /** Which planning run screening should use. `null` means "the most recent",
   *  which is the default; an explicit id pins a specific run. */
  inUseRunId: string | null; setInUseRunId: (id: string | null) => void;
  addSimulationRun: (run: Omit<SimulationRun, "id" | "timestamp" | "label">) => void;
  removeSimulationRun: (id: string) => void;
  renameSimulationRun: (id: string, label: string) => void;
  clearSimulationRuns: () => void;

  // Quality Assessment
  rawPapers: Paper[] | null; setRawPapers: (v: Paper[] | null) => void;
  // Records the reviewer marked as relevant "seed" studies (relevance feedback);
  // fed back into the grounded query builder to broaden concepts.
  seedIds: Set<string>; setSeedIds: React.Dispatch<React.SetStateAction<Set<string>>>;
  uniquePapers: Paper[] | null; setUniquePapers: (v: Paper[] | null) => void;
  duplicatesCount: number; setDuplicatesCount: (v: number) => void;
  qualityReports: QualityReport[] | null; setQualityReports: (v: QualityReport[] | null) => void;
  // Saved alternate appraisals: prior runs kept when a paper is re-appraised with a
  // different instrument. Invariant: at most one entry per (paper_id, instrument_id),
  // and none duplicating the active report in qualityReports.
  qualityArchive: QualityReport[]; setQualityArchive: React.Dispatch<React.SetStateAction<QualityReport[]>>;
  excludedByQuality: Set<string>; setExcludedByQuality: React.Dispatch<React.SetStateAction<Set<string>>>;

  // Reviewer overrides on screening decisions. Keyed by paper_id; value is
  // the reviewer's effective decision ("INCLUDE" / "EXCLUDE" for abstract,
  // "Include" / "Exclude" for full-text, matching the case the screener
  // already uses for each stage). The AI's original Decision stays on the
  // result row so the override is auditable.
  abstractOverrides: Record<string, "INCLUDE" | "EXCLUDE">;
  setAbstractOverride: (paperId: string, decision: "INCLUDE" | "EXCLUDE") => void;
  clearAbstractOverride: (paperId: string) => void;
  setAbstractOverrides: (v: Record<string, "INCLUDE" | "EXCLUDE">) => void;
  fullTextOverrides: Record<string, "Include" | "Exclude">;
  setFullTextOverride: (paperId: string, decision: "Include" | "Exclude") => void;
  clearFullTextOverride: (paperId: string) => void;
  setFullTextOverrides: (v: Record<string, "Include" | "Exclude">) => void;
  // Audit log of reviewer overrides on AI-generated RoB judgments. Stored as
  // an append-only list; the most recent entry per (paper, domain) is the
  // current effective judgment.
  qualityOverrides: QualityOverride[];
  addQualityOverride: (o: QualityOverride) => void;
  clearQualityOverrides: (paperId?: string) => void;
  setQualityOverrides: (v: QualityOverride[]) => void;
  gradeOutcomes: GradeOutcome[]; setGradeOutcomes: React.Dispatch<React.SetStateAction<GradeOutcome[]>>;
  searchLog: SearchLogEntry[]; setSearchLog: React.Dispatch<React.SetStateAction<SearchLogEntry[]>>;
  protocol: ReviewProtocol | null; setProtocol: (v: ReviewProtocol | null) => void;
  protocolDeviations: ProtocolDeviation[]; setProtocolDeviations: React.Dispatch<React.SetStateAction<ProtocolDeviation[]>>;
  prismaChecklist: Record<string, { status: string; note: string }>; setPrismaChecklist: React.Dispatch<React.SetStateAction<Record<string, { status: string; note: string }>>>;

  // Relevance reranking: LEADS-scored papers from the home-analysis pipeline.
  // Threshold is user-tunable in the sidebar; rerankResults holds the full
  // ranked list so the UI can re-filter without re-running LLM calls.
  rerankThreshold: number; setRerankThreshold: (v: number) => void;
  rerankResults: RerankResult | null; setRerankResults: (v: RerankResult | null) => void;

  // Meta-analysis agent: extracted effect sizes + full analysis bundle
  // (pool + subgroup + LOO + funnel + Egger + Begg + trim-and-fill +
  // meta-regression). All editable by the user; rows can be removed or
  // corrected without re-running LLM extraction, then re-run the analysis
  // against the edited list.
  metaOutcome: string; setMetaOutcome: (v: string) => void;
  metaMeasure: EffectMeasure | ""; setMetaMeasure: (v: EffectMeasure | "") => void;
  metaTau2Method: Tau2Method; setMetaTau2Method: (v: Tau2Method) => void;
  metaUseKnappHartung: boolean; setMetaUseKnappHartung: (v: boolean) => void;
  metaExtractions: StudyEffect[] | null; setMetaExtractions: React.Dispatch<React.SetStateAction<StudyEffect[] | null>>;
  metaRun: MetaRunResult | null; setMetaRun: (v: MetaRunResult | null) => void;

  // Results
  results: ScreenResult[] | null; setResults: (v: ScreenResult[] | null) => void;
  screeningDuration: number; setScreeningDuration: (v: number) => void;
  // Provenance for the corpus that abstract screening actually retrieved: the
  // per-database query and planned yield used for each source. PRISMA requires
  // the search to be reported per database, and without this the link between
  // the planning run and the screened corpus is not recorded anywhere.
  screeningPlan: ScreeningPlan | null; setScreeningPlan: (v: ScreeningPlan | null) => void;
  // Archive of every title/abstract screening pass (for compare / restore).
  screeningArchive: ScreeningPass[]; setScreeningArchive: React.Dispatch<React.SetStateAction<ScreeningPass[]>>;
  fullTextResults: FullTextResult[] | null; setFullTextResults: (v: FullTextResult[] | null) => void;
  ftDuration: number; setFtDuration: (v: number) => void;

  // Snowball
  snowballResults: any[] | null; setSnowballResults: (v: any[] | null) => void;
  snowballScreened: ScreenResult[] | null; setSnowballScreened: (v: ScreenResult[] | null) => void;
  // Seed titles the last citation-snowball ran over (session-only), so a later
  // change to the included set can flag stale results and prompt a re-run.
  snowballSeeds: string[] | null; setSnowballSeeds: (v: string[] | null) => void;
  /** What the snowball pull found against what it kept.
   *
   *  Snowballing truncates: a seed with 85 references, capped at 20, loses 65.
   *  That was invisible, which made the search irreproducible from what the
   *  app recorded and left PRISMA with nothing to state. */
  snowballPlan: SnowballPlan | null; setSnowballPlan: (v: SnowballPlan | null) => void;
  // Citation ids the reviewer has selected to carry forward. Persisted (and null
  // until first seeded) so the selection survives tab switches and isn't re-seeded
  // from the AI verdicts on every remount.
  snowballChosen: string[] | null; setSnowballChosen: React.Dispatch<React.SetStateAction<string[] | null>>;

  // Full-text acquisition
  fullTexts: Record<string, FullTextRecord>; setFullTexts: React.Dispatch<React.SetStateAction<Record<string, FullTextRecord>>>;

  // Extraction
  extractedPapers: ExtractedPaper[] | null; setExtractedPapers: (v: ExtractedPaper[] | null) => void;
  textExtractions: TextExtractionResult[]; setTextExtractions: React.Dispatch<React.SetStateAction<TextExtractionResult[]>>;
  // Writing Assistant: cached citation metadata enrichment + generated methods
  // summary, so switching tabs doesn't re-fetch everything.
  writingEnriched: Record<string, Record<string, any>>; setWritingEnriched: React.Dispatch<React.SetStateAction<Record<string, Record<string, any>>>>;
  writingSummary: string; setWritingSummary: (v: string) => void;
  writingMethodsMain: string; setWritingMethodsMain: (v: string) => void;

  // PRISMA
  prisma: PrismaCounts; setPrisma: React.Dispatch<React.SetStateAction<PrismaCounts>>;

  // Session persistence
  currentSessionId: string | null; setCurrentSessionId: (v: string | null) => void;
  currentSessionTitle: string; setCurrentSessionTitle: (v: string) => void;
  snapshot: () => any;
  hydrate: (data: any, authoritative?: boolean) => void;
  reset: () => void;

  // Multi-reviewer project context. When `currentProjectId` is set the
  // platform is in "project mode": screening writes flow through the project
  // API (multi-reviewer aware), the PRISMA flow honours adjudicated decisions,
  // and blinding is enforced server-side. When null we are in legacy single-
  // user mode.
  currentProjectId: string | null; setCurrentProjectId: (v: string | null) => void;
  currentProjectName: string; setCurrentProjectName: (v: string) => void;
  currentProjectRole: "lead" | "reviewer" | "adjudicator" | "viewer" | null;
  setCurrentProjectRole: (v: "lead" | "reviewer" | "adjudicator" | "viewer" | null) => void;
  currentProjectMode: "single" | "dual" | "dual_blinded" | null;
  setCurrentProjectMode: (v: "single" | "dual" | "dual_blinded" | null) => void;

  // Long-running tasks (survive page changes)
  tasks: Record<string, TaskRecord>;
  startTask: (kind: TaskRecord["kind"], stages: TaskStage[]) => { abort: AbortController; taskId: string };
  updateTask: (kind: TaskRecord["kind"], patch: Partial<TaskRecord>) => void;
  updateTaskStage: (kind: TaskRecord["kind"], stageId: string, patch: Partial<TaskStage>) => void;
  appendTaskLog: (kind: TaskRecord["kind"], line: string) => void;
  cancelTask: (kind: TaskRecord["kind"]) => void;
  clearTask: (kind: TaskRecord["kind"]) => void;
};

export type ScreeningPlan = {
  /** Sources actually searched, after filtering out non-API entries. */
  sources: string[];
  /** The exact query sent to each source. */
  perDbQueries: Record<string, string>;
  /** Planned yield per source at the time of the run (null if never simulated). */
  counts: Record<string, number> | null;
  /** How many records each source actually returned. */
  retrieved: Record<string, number>;
  /** Why each source returned what it did: ok / empty / error, the error text,
   *  whether the result was capped, and how long the call took. Recorded so a
   *  shortfall can be explained instead of guessed at. */
  sourceStatus?: Record<string, {
    status: "ok" | "empty" | "error"; count: number; requested: number;
    capped: boolean; error?: string; error_type?: string; ms: number;
  }>;
  /** The query `counts` were measured against. Lets a planned-vs-retrieved gap
   *  be attributed: a stale plan reads very differently from a source failing. */
  countedQuery?: string | null;
  /** The PER-SOURCE query each count in `counts` was measured against. Retrieval
   *  is per-source, so a unified-query comparison could not see a per-database
   *  edit and would blame the database for a gap the app itself created. */
  countedPerDb?: Record<string, string>;
  /** The planning run this plan was built from, so the provenance is nameable
   *  rather than inferred by matching query text. */
  runId?: string | null;
  runLabel?: string | null;
  ranAt: string;
};

export type PrismaCounts = {
  identified: number;
  source_counts: Record<string, number>;
  duplicates_removed: number;
  screened: number;
  excluded_total: number;
  exclusion_breakdown: Record<string, number>;
  ft_exclusion_breakdown?: Record<string, number>;
  included_final: number;
};

const StoreCtx = createContext<Ctx | null>(null);
// The "Saved" indicator has its own context: it updates after every autosave, and
// in the data context that update re-rendered every page and table for a timestamp.
const SaveStatusCtx = createContext<number | null>(null);

const PAGE_STORAGE_KEY = "ee:page";
export const SESSION_STORAGE_KEY = "ee:sessionId";
// Local autosave of the working snapshot, so a browser refresh restores the
// review even in demo / logged-out mode (where backend sessions don't save).
const LOCAL_SNAPSHOT_KEY = "ee:snapshot";
const VALID_PAGES: PageId[] = [
  "home", "simulation", "quality", "abstract", "acquisition", "fulltext",
  "snowball", "extraction", "textextraction", "prisma", "meta", "projects", "writing",
];
function loadPage(): PageId {
  if (FRESH_LAUNCH) return "home";   // a fresh app launch always starts on Home
  try {
    const v = localStorage.getItem(PAGE_STORAGE_KEY) as PageId | null;
    if (v && VALID_PAGES.includes(v)) return v;
  } catch { /* localStorage unavailable (private mode, etc.) */ }
  return "home";
}

export function StoreProvider({ children }: { children: ReactNode }) {
  // Restore the last-viewed tab so a browser refresh keeps you in place.
  const [page, setPage] = useState<PageId>(loadPage);
  const [reviewOpen, setReviewOpen] = useState(false);
  useEffect(() => {
    try { localStorage.setItem(PAGE_STORAGE_KEY, page); } catch { /* ignore */ }
  }, [page]);
  // Default to LEADS, the benchmark's highest-performing screening model
  // (LEADS-mistral-7b × LEADS-native @ score ≥ +0.20: recall=1.000,
  // specificity=0.676, MCC=+0.260 on van_Dis_2020). The Backend resolves
  // "leads" to the full GGUF tag and routes to the LEADS-native pipeline.
  const [model, setModel] = useState("leads");
  const [sources, setSources] = useState<string[]>(["PubMed", "Europe PMC", "Semantic Scholar"]);
  const [numPerSource, setNumPerSource] = useState(15);
  const [perSourceLimits, setPerSourceLimits] = useState<Record<string, number>>({});
  const [searchSelection, setSearchSelection] = useState<SelectionStrategy>("relevance");
  const [searchFilters, setSearchFilters] = useState<SearchFilters>({});
  const [relevanceOverrides, setRelevanceOverrides] = useState<Record<string, "keep" | "drop">>({});
  const [files, setFiles] = useState<File[]>([]);

  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [docQa, setDocQa] = useState<DocQaTurn[]>([]);
  const [pico, setPico] = useState<Pico>({ population: "", intervention: "", comparator: "", outcome: "" });
  const [framework, setFramework] = useState<FrameworkId>("pico");
  const [inclusion, setInclusion] = useState<string[]>([]);
  const [exclusion, setExclusion] = useState<string[]>([]);
  const [query, setQuery] = useState("");

  const [unifiedSearchQuery, setUnifiedSearchQuery] = useState("");
  const [perDbQueries, setPerDbQueries] = useState<Record<string, string>>({});
  const [simulation, setSimulation] = useState<Record<string, number> | null>(null);
  const [dbTestResults, setDbTestResults] = useState<Record<string, { query: string; total_found: number; papers: { title: string; url: string }[] }> | null>(null);
  const [agenticTrace, setAgenticTrace] = useState<any[] | null>(null);
  const [agenticSummary, setAgenticSummary] = useState<{ iterations_run: number; total_papers_found: number; best_relevance: number } | null>(null);
  const [simulationQuery, setSimulationQuery] = useState<string | null>(null);
  const [simulationRuns, setSimulationRuns] = useState<SimulationRun[]>([]);
  const [inUseRunId, setInUseRunId] = useState<string | null>(null);
  const addSimulationRun = (run: Omit<SimulationRun, "id" | "timestamp" | "label">) => {
    const id = crypto.randomUUID();
    setSimulationRuns(prev => [...prev, {
      ...run, id, timestamp: Date.now(), label: `Run ${prev.length + 1}`,
    }].slice(-20)); // keep last 20
    // A run the reviewer just executed becomes the one screening uses. Pinning
    // an older run stays in effect only until the next run is made.
    setInUseRunId(id);
  };
  // Labels are NOT renumbered after a removal. "Run 3" is how a reviewer refers
  // to a run in their notes and in a comparison they have open; resequencing
  // would silently rename every later run and make those references wrong.
  const removeSimulationRun = (id: string) => {
    setSimulationRuns(prev => prev.filter(r => r.id !== id));
    // Fall back to "most recent" rather than pointing at a run that is gone.
    setInUseRunId(prev => (prev === id ? null : prev));
  };
  // A run's label is how a reviewer identifies it later ("the one where I
  // dropped the population block"), so it is editable. An empty rename is
  // ignored rather than accepted: a blank row in the history is worse than the
  // generated "Run 3".
  const renameSimulationRun = (id: string, label: string) => {
    const next = (label || "").trim().slice(0, 80);
    if (!next) return;
    setSimulationRuns(prev => prev.map(r => (r.id === id ? { ...r, label: next } : r)));
  };
  const clearSimulationRuns = () => { setSimulationRuns([]); setInUseRunId(null); };

  const [rawPapers, setRawPapers] = useState<Paper[] | null>(null);
  const [seedIds, setSeedIds] = useState<Set<string>>(new Set());
  const [uniquePapers, setUniquePapers] = useState<Paper[] | null>(null);
  const [duplicatesCount, setDuplicatesCount] = useState(0);
  const [qualityReports, setQualityReports] = useState<QualityReport[] | null>(null);
  const [qualityArchive, setQualityArchive] = useState<QualityReport[]>([]);
  const [excludedByQuality, setExcludedByQuality] = useState<Set<string>>(new Set());
  const [qualityOverrides, setQualityOverrides] = useState<QualityOverride[]>([]);
  const [gradeOutcomes, setGradeOutcomes] = useState<GradeOutcome[]>([]);
  const [searchLog, setSearchLog] = useState<SearchLogEntry[]>([]);
  const [protocol, setProtocol] = useState<ReviewProtocol | null>(null);
  const [protocolDeviations, setProtocolDeviations] = useState<ProtocolDeviation[]>([]);
  const [prismaChecklist, setPrismaChecklist] = useState<Record<string, { status: string; note: string }>>({});
  const [lastSavedAt, setLastSavedAt] = useState<number | null>(null);

  const [abstractOverrides, setAbstractOverrides] = useState<Record<string, "INCLUDE" | "EXCLUDE">>({});
  const setAbstractOverride = (paperId: string, decision: "INCLUDE" | "EXCLUDE") => {
    setAbstractOverrides(prev => ({ ...prev, [paperId]: decision }));
  };
  const clearAbstractOverride = (paperId: string) => {
    setAbstractOverrides(prev => {
      const { [paperId]: _, ...rest } = prev;
      return rest;
    });
  };

  const [fullTextOverrides, setFullTextOverrides] = useState<Record<string, "Include" | "Exclude">>({});
  const setFullTextOverride = (paperId: string, decision: "Include" | "Exclude") => {
    setFullTextOverrides(prev => ({ ...prev, [paperId]: decision }));
  };
  const clearFullTextOverride = (paperId: string) => {
    setFullTextOverrides(prev => {
      const { [paperId]: _, ...rest } = prev;
      return rest;
    });
  };

  const addQualityOverride = (o: QualityOverride) => {
    setQualityOverrides(prev => [...prev, o]);
  };
  const clearQualityOverrides = (paperId?: string) => {
    if (paperId === undefined) {
      setQualityOverrides([]);
    } else {
      setQualityOverrides(prev => prev.filter(o => o.paper_id !== paperId));
    }
  };

  // Relevance rerank: -0.2 keeps "maybe relevant" and better. Tunable in sidebar.
  const [rerankThreshold, setRerankThreshold] = useState<number>(-0.2);
  const [rerankResults, setRerankResults] = useState<RerankResult | null>(null);

  // Meta-analysis agent state. The extractions list is mutable so the user
  // can correct individual rows before pooling. metaRun holds all derived
  // analyses (pool, subgroup, LOO, funnel, Egger, Begg, trim-and-fill,
  // meta-regression) so the tabs can render without re-hitting the backend.
  const [metaOutcome, setMetaOutcome] = useState<string>("");
  const [metaMeasure, setMetaMeasure] = useState<EffectMeasure | "">("");
  const [metaTau2Method, setMetaTau2Method] = useState<Tau2Method>("DL");
  const [metaUseKnappHartung, setMetaUseKnappHartung] = useState<boolean>(false);
  const [metaExtractions, setMetaExtractions] = useState<StudyEffect[] | null>(null);
  const [metaRun, setMetaRun] = useState<MetaRunResult | null>(null);

  const [results, setResults] = useState<ScreenResult[] | null>(null);
  const [screeningDuration, setScreeningDuration] = useState(0);
  const [screeningPlan, setScreeningPlan] = useState<ScreeningPlan | null>(null);
  const [screeningArchive, setScreeningArchive] = useState<ScreeningPass[]>([]);
  const [fullTextResults, setFullTextResults] = useState<FullTextResult[] | null>(null);
  const [ftDuration, setFtDuration] = useState(0);

  const [snowballResults, setSnowballResults] = useState<any[] | null>(null);
  const [snowballScreened, setSnowballScreened] = useState<ScreenResult[] | null>(null);
  const [snowballChosen, setSnowballChosen] = useState<string[] | null>(null);
  const [snowballSeeds, setSnowballSeeds] = useState<string[] | null>(null);
  const [snowballPlan, setSnowballPlan] = useState<SnowballPlan | null>(null);

  const [extractedPapers, setExtractedPapers] = useState<ExtractedPaper[] | null>(null);
  const [fullTexts, setFullTexts] = useState<Record<string, FullTextRecord>>({});
  const [textExtractions, setTextExtractions] = useState<TextExtractionResult[]>([]);
  const [writingEnriched, setWritingEnriched] = useState<Record<string, Record<string, any>>>({});
  const [writingSummary, setWritingSummary] = useState("");
  const [writingMethodsMain, setWritingMethodsMain] = useState("");

  const [prisma, setPrisma] = useState<PrismaCounts>({
    identified: 0, source_counts: {}, duplicates_removed: 0, screened: 0,
    excluded_total: 0, exclusion_breakdown: {}, included_final: 0,
  });

  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [currentSessionTitle, setCurrentSessionTitle] = useState<string>("Untitled session");
  // Remember the active session so a refresh can silently restore it (see
  // SessionsPanel's auto-restore effect).
  useEffect(() => {
    try {
      if (currentSessionId) localStorage.setItem(SESSION_STORAGE_KEY, currentSessionId);
      else localStorage.removeItem(SESSION_STORAGE_KEY);
    } catch { /* ignore */ }
  }, [currentSessionId]);


  const [currentProjectId, setCurrentProjectId] = useState<string | null>(null);
  const [currentProjectName, setCurrentProjectName] = useState<string>("");
  const [currentProjectRole, setCurrentProjectRole] = useState<"lead" | "reviewer" | "adjudicator" | "viewer" | null>(null);
  const [currentProjectMode, setCurrentProjectMode] = useState<"single" | "dual" | "dual_blinded" | null>(null);

  const [tasks, setTasks] = useState<Record<string, TaskRecord>>({});

  const startTask = (kind: TaskRecord["kind"], stages: TaskStage[]) => {
    const abort = new AbortController();
    const taskId =
      typeof crypto !== "undefined" && (crypto as any).randomUUID
        ? (crypto as any).randomUUID()
        : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    setTasks(t => ({
      ...t,
      [kind]: {
        kind,
        taskId,
        status: "running",
        startedAt: Date.now(),
        stages,
        log: [],
        abort,
      },
    }));
    return { abort, taskId };
  };
  const updateTask: Ctx["updateTask"] = (kind, patch) =>
    setTasks(t => (t[kind] ? { ...t, [kind]: { ...t[kind], ...patch } } : t));
  const updateTaskStage: Ctx["updateTaskStage"] = (kind, stageId, patch) =>
    setTasks(t => {
      const tk = t[kind];
      if (!tk) return t;
      return { ...t, [kind]: { ...tk, stages: tk.stages.map(st => (st.id === stageId ? { ...st, ...patch } : st)) } };
    });
  const appendTaskLog: Ctx["appendTaskLog"] = (kind, line) =>
    setTasks(t => (t[kind] ? { ...t, [kind]: { ...t[kind], log: [...t[kind].log, line] } } : t));
  const cancelTask: Ctx["cancelTask"] = kind => {
    setTasks(t => {
      const tk = t[kind];
      if (!tk) return t;
      // Signal server-side cancel (harmless if the endpoint doesn't register).
      if (tk.taskId) {
        fetch(`${apiConfig.baseUrl}/tasks/cancel`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task_id: tk.taskId }),
          keepalive: true,
        }).catch(() => { /* fire-and-forget */ });
      }
      try { tk.abort?.abort(); } catch { /* noop */ }
      return { ...t, [kind]: { ...tk, status: "canceled", abort: undefined } };
    });
  };
  const clearTask: Ctx["clearTask"] = kind =>
    setTasks(t => {
      const next = { ...t };
      delete next[kind];
      return next;
    });

  useEffect(() => { apiConfig.model = model; }, [model]);

  const snapshot = () => ({
    history, docQa, pico, framework, inclusion, exclusion, query, unifiedSearchQuery, perDbQueries,
    sources, numPerSource, perSourceLimits, searchSelection, searchFilters, relevanceOverrides, model,
    // rawPapers repeats uniquePapers almost record for record; shared abstracts go once.
    rawPapers: compactRawPapers(rawPapers, uniquePapers), uniquePapers, duplicatesCount, qualityReports, qualityArchive,
    excludedByQuality: Array.from(excludedByQuality),
    qualityOverrides,
    gradeOutcomes,
    searchLog, protocol, protocolDeviations, prismaChecklist,
    abstractOverrides,
    fullTextOverrides,
    rerankThreshold, rerankResults,
    // The newest pass's results copy is the live table itself; store a marker.
    results, screeningArchive: compactArchive(screeningArchive, results), screeningDuration, screeningPlan, fullTextResults, ftDuration, snowballResults, snowballScreened, snowballChosen, snowballPlan, extractedPapers, prisma,
    // Planning (search-design) outputs + per-tab run results so a session keeps
    // everything that's been run, including acquired full texts. Full-text BODIES
    // stay in the backend text cache and only their keys are saved; the store
    // pulls the bodies back after a load. Bodies without a key are kept as-is.
    simulation, simulationQuery, simulationRuns, inUseRunId, dbTestResults, agenticTrace, agenticSummary,
    textExtractions, fullTexts: compactFullTexts(fullTexts),
    writingEnriched, writingSummary, writingMethodsMain,
    // Meta-analysis. These were absent, so every extraction and pooled result
    // was discarded on reload or session switch: the reviewer paid for an LLM
    // pass over every included study, edited the rows by hand, and lost all of
    // it by opening another tab.
    metaOutcome, metaMeasure, metaTau2Method, metaUseKnappHartung,
    metaExtractions, metaRun,
  });

  // `authoritative` = true replaces every field outright (explicit session
  // load/switch: the incoming session IS the truth, even where it's empty).
  // `authoritative` = false is a RECONCILIATION (restore-on-open, where the
  // local snapshot and the backend session are two views of the same session):
  // an absent or empty incoming collection must NOT clobber one the store
  // already has populated. Without this, a leaner/staler source (or a session
  // whose late-stage tabs were dropped) wipes extractions/quality/snowball on
  // reopen and autosave then persists the empties: the tabs-go-blank bug.
  const hydrate = (d: any, authoritative = true) => {
    if (!d) return;
    // Prefer a non-empty incoming collection; otherwise keep what we have.
    const isEmpty = (v: any) =>
      v == null ||
      (Array.isArray(v) && v.length === 0) ||
      (typeof v === "object" && !Array.isArray(v) && Object.keys(v).length === 0);
    const pick = <T,>(incoming: T | null | undefined, cur: T | null, fallback: T | null = null): T | null => {
      if (authoritative) return (incoming ?? fallback) as T | null;
      // Reconciliation: keep whatever is already populated (the local snapshot
      // is written more often, so it's the fresher of the two views of this
      // session); only fill in fields the current store is still missing.
      if (!isEmpty(cur)) return cur;
      return (incoming ?? fallback) as T | null;
    };

    setHistory(d.history || []);
    setDocQa((d.docQa || []).filter((t: DocQaTurn) => !t.busy));   // drop any turn that was mid-flight when saved
    setPico(d.pico || { population: "", intervention: "", comparator: "", outcome: "" });
    setFramework(d.framework === "pcc" ? "pcc" : "pico");
    setInclusion(d.inclusion || []);
    setExclusion(d.exclusion || []);
    setQuery(d.query || "");
    setUnifiedSearchQuery(prev => pick(d.unifiedSearchQuery, prev, "") ?? "");
    setPerDbQueries(prev => pick(d.perDbQueries, prev, {}) ?? {});
    if (Array.isArray(d.sources)) setSources(d.sources);
    if (typeof d.numPerSource === "number") setNumPerSource(d.numPerSource);
    if (d.searchSelection === "relevance" || d.searchSelection === "recent") setSearchSelection(d.searchSelection);
    if (d.searchFilters && typeof d.searchFilters === "object") setSearchFilters(d.searchFilters);
    if (d.relevanceOverrides && typeof d.relevanceOverrides === "object") setRelevanceOverrides(d.relevanceOverrides);
    setPerSourceLimits(d.perSourceLimits && typeof d.perSourceLimits === "object" ? d.perSourceLimits : {});
    if (d.model) setModel(d.model);
    setRawPapers(prev => pick(expandRawPapers(d.rawPapers, d.uniquePapers) ?? null, prev));
    setUniquePapers(prev => pick(d.uniquePapers, prev));
    setDuplicatesCount(d.duplicatesCount ?? 0);
    setQualityReports(prev => pick(d.qualityReports, prev));
    setQualityArchive(prev => pick(d.qualityArchive, prev, []) ?? []);
    setExcludedByQuality(new Set(d.excludedByQuality || []));
    setQualityOverrides(Array.isArray(d.qualityOverrides) ? d.qualityOverrides : []);
    setGradeOutcomes(prev => pick(d.gradeOutcomes, prev, []) ?? []);
    setSearchLog(prev => pick(d.searchLog, prev, []) ?? []);
    if (authoritative) setProtocol(d.protocol ?? null); else if (d.protocol) setProtocol(d.protocol);
    setProtocolDeviations(prev => pick(d.protocolDeviations, prev, []) ?? []);
    setPrismaChecklist(prev => pick(d.prismaChecklist, prev, {}) ?? {});
    // Meta-analysis. metaExtractions carries hand-corrected numbers, so on a
    // reconciliation it follows the same keep-what-is-populated rule as the
    // other expensive collections.
    if (typeof d.metaOutcome === "string") setMetaOutcome(d.metaOutcome);
    if (typeof d.metaMeasure === "string") setMetaMeasure(d.metaMeasure as EffectMeasure | "");
    if (d.metaTau2Method) setMetaTau2Method(d.metaTau2Method as Tau2Method);
    if (typeof d.metaUseKnappHartung === "boolean") setMetaUseKnappHartung(d.metaUseKnappHartung);
    setMetaExtractions(prev => pick(d.metaExtractions, prev));
    if (authoritative) setMetaRun(d.metaRun ?? null); else if (d.metaRun) setMetaRun(d.metaRun);
    setAbstractOverrides(d.abstractOverrides && typeof d.abstractOverrides === "object" ? d.abstractOverrides : {});
    setFullTextOverrides(d.fullTextOverrides && typeof d.fullTextOverrides === "object" ? d.fullTextOverrides : {});
    if (typeof d.rerankThreshold === "number") setRerankThreshold(d.rerankThreshold);
    setRerankResults(prev => pick(d.rerankResults, prev));
    setResults(prev => pick(d.results, prev));
    setScreeningArchive(prev => pick(expandArchive(d.screeningArchive, d.results) ?? null, prev, []) ?? []);
    // Run timings. On an authoritative load the incoming session is the truth,
    // so take its value (0 if the session predates this field). On a
    // reconciliation, a positive incoming duration fills in a missing one but a
    // 0/absent value must not overwrite a real measurement the store already has.
    setScreeningPlan(pick(d.screeningPlan, screeningPlan, null) ?? null);
    if (authoritative) setScreeningDuration(typeof d.screeningDuration === "number" ? d.screeningDuration : 0);
    else if (typeof d.screeningDuration === "number" && d.screeningDuration > 0) setScreeningDuration(d.screeningDuration);
    setFullTextResults(prev => pick(d.fullTextResults, prev));
    if (authoritative) setFtDuration(typeof d.ftDuration === "number" ? d.ftDuration : 0);
    else if (typeof d.ftDuration === "number" && d.ftDuration > 0) setFtDuration(d.ftDuration);
    setSnowballResults(prev => pick(d.snowballResults, prev));
    setSnowballPlan(pick(d.snowballPlan, snowballPlan, null) ?? null);
    setSnowballScreened(prev => pick(d.snowballScreened, prev));
    setSnowballChosen(Array.isArray(d.snowballChosen) ? d.snowballChosen : null);
    setExtractedPapers(prev => pick(d.extractedPapers, prev));
    if (d.prisma) setPrisma(d.prisma);
    // Planning + per-tab run outputs.
    setSimulation(prev => pick(d.simulation, prev));
    setSimulationRuns(prev => pick(d.simulationRuns, prev, []) ?? []);
    if (typeof d.simulationQuery === "string") setSimulationQuery(d.simulationQuery);
    if (typeof d.inUseRunId === "string" || d.inUseRunId === null) setInUseRunId(d.inUseRunId ?? null);
    setDbTestResults(prev => pick(d.dbTestResults, prev));
    setAgenticTrace(prev => pick(d.agenticTrace, prev));
    setAgenticSummary(prev => pick(d.agenticSummary, prev));
    setTextExtractions(prev => pick(d.textExtractions, prev, []) ?? []);
    if (d.writingEnriched && typeof d.writingEnriched === "object") setWritingEnriched(d.writingEnriched);
    if (typeof d.writingSummary === "string") setWritingSummary(d.writingSummary);
    if (typeof d.writingMethodsMain === "string") setWritingMethodsMain(d.writingMethodsMain);
    setFullTexts(prev => pick(d.fullTexts, prev, {}) ?? {});
  };

  // ── Full-text bodies ──────────────────────────────────────────────────────
  // The saved session carries a cache key per acquired paper, not the text. After
  // a load, pull the bodies back in the background in small batches (one state
  // update per batch); and push any body that predates the cache so the next save
  // can drop it. Each id is attempted once per app run, so a miss never loops.
  const textSync = useRef<Set<string>>(new Set());
  useEffect(() => {
    const recs = Object.values(fullTexts);
    const pull = recs.filter(r => r.status === "found" && r.text_key && r.text === undefined && !textSync.current.has(`get:${r.paper_id}`));
    const push = recs.filter(r => r.status === "found" && !r.text_key && r.text && !textSync.current.has(`put:${r.paper_id}`));
    if (!pull.length && !push.length) return;
    (async () => {
      for (let i = 0; i < pull.length; i += 20) {
        const batch = pull.slice(i, i + 20);
        batch.forEach(r => textSync.current.add(`get:${r.paper_id}`));
        const got = await Promise.all(batch.map(async r => {
          try { return [r.paper_id, await fetchFullTextBody(r.text_key!)] as const; }
          catch { return [r.paper_id, null] as const; }
        }));
        const found = got.filter(([, t]) => typeof t === "string" && t);
        if (found.length) setFullTexts(prev => {
          const next = { ...prev };
          for (const [id, t] of found) if (next[id] && next[id].text === undefined) next[id] = { ...next[id], text: t! };
          return next;
        });
      }
      for (let i = 0; i < push.length; i += 10) {
        const batch = push.slice(i, i + 10);
        batch.forEach(r => textSync.current.add(`put:${r.paper_id}`));
        const keys = await Promise.all(batch.map(async r => {
          try { return [r.paper_id, await putFullTextBody(r.paper_id, r.text!)] as const; }
          catch { return [r.paper_id, null] as const; }
        }));
        const stored = keys.filter(([, k]) => k);
        if (stored.length) setFullTexts(prev => {
          const next = { ...prev };
          for (const [id, k] of stored) if (next[id] && !next[id].text_key) next[id] = { ...next[id], text_key: k! };
          return next;
        });
      }
    })();
  }, [fullTexts]);

  // ── Local persistence ─────────────────────────────────────────────────────
  // Restore the locally-saved snapshot once on mount, then keep it in sync
  // (debounced) so a refresh lands the user back in their review. Guarded so
  // the first empty render never wipes the saved snapshot before it's read.
  const localRestored = useRef(false);
  const lastSavedJson = useRef<string | null>(null);
  useEffect(() => {
    if (!localRestored.current) return;          // wait until restore has run
    // Debounce, then wait for an idle moment: serialising the session is main-
    // thread work of the size that drops frames if it lands mid-animation.
    const w = window as any;
    let idleId: number | null = null;
    const hasIdle = typeof w.requestIdleCallback === "function";
    const t = setTimeout(() => { if (hasIdle) idleId = w.requestIdleCallback(save, { timeout: 4000 }); else save(); }, 600);
    function save() {
      if (history.length === 0) {
        try { localStorage.removeItem(LOCAL_SNAPSHOT_KEY); } catch { /* ignore */ }
        idbDel(LOCAL_SNAPSHOT_KEY).catch(() => { /* ignore */ });
        return;
      }
      // Envelope keeps the session identity with the data, so a refresh keeps
      // editing the SAME session instead of spawning a duplicate.
      const env = { data: snapshot(), sessionId: currentSessionId, sessionTitle: currentSessionTitle };
      // Serialise once. A save whose bytes equal the last one is skipped, so a tab
      // that re-sets identical state on mount costs nothing; storing the string
      // rather than the object graph also spares IndexedDB a deep structured clone.
      let json: string;
      try { json = JSON.stringify(env); } catch { return; }
      if (json === lastSavedJson.current) return;
      // Primary store: IndexedDB, a much larger quota than localStorage (~5 MB),
      // so big full-text sets (many uploaded PDFs) persist intact across reloads.
      idbSet(LOCAL_SNAPSHOT_KEY, json)
        .then(() => { lastSavedJson.current = json; setLastSavedAt(Date.now()); try { localStorage.removeItem(LOCAL_SNAPSHOT_KEY); } catch { /* ignore */ } })
        .catch(() => {
          // IndexedDB unavailable (rare) → fall back to localStorage, shedding the
          // heaviest re-derivable fields one tier at a time so structured stage
          // outputs still persist rather than the whole write being dropped.
          const SHED_TIERS: string[][] = [
            [],
            ["fullTexts"],
            ["fullTexts", "agenticTrace", "simulationRuns", "dbTestResults"],
            ["fullTexts", "agenticTrace", "simulationRuns", "dbTestResults", "rawPapers"],
          ];
          for (const drop of SHED_TIERS) {
            const data: any = { ...env.data };
            for (const k of drop) delete data[k];
            try { localStorage.setItem(LOCAL_SNAPSHOT_KEY, JSON.stringify({ ...env, data })); break; }
            catch { /* over quota, try the next, leaner tier */ }
          }
        });
    }
    return () => { clearTimeout(t); if (idleId != null && hasIdle) w.cancelIdleCallback(idleId); };
  }, [history, docQa, pico, framework, inclusion, exclusion, query, unifiedSearchQuery, perDbQueries,
      sources, numPerSource, perSourceLimits, searchSelection, searchFilters, relevanceOverrides, model, rawPapers, uniquePapers, duplicatesCount,
      qualityReports, qualityArchive, excludedByQuality, qualityOverrides, gradeOutcomes,
      searchLog, protocol, protocolDeviations, prismaChecklist, abstractOverrides,
      fullTextOverrides, rerankThreshold, rerankResults, results, screeningArchive, fullTextResults,
      snowballResults, snowballScreened, snowballChosen, snowballPlan, extractedPapers, prisma,
      simulation, simulationRuns, inUseRunId, dbTestResults, agenticTrace, agenticSummary, textExtractions, fullTexts,
      writingEnriched, writingSummary, writingMethodsMain,
      currentSessionId, currentSessionTitle]);

  const restoreStarted = useRef(false);
  useEffect(() => {
    if (restoreStarted.current) return;
    restoreStarted.current = true;
    // Fresh app launch → start a brand-new session on Home; don't restore the
    // last one. Past sessions remain saved and loadable from the Sessions panel.
    if (FRESH_LAUNCH) { localRestored.current = true; return; }
    (async () => {
      try {
        // Prefer IndexedDB; fall back to (and migrate from) an old localStorage snapshot.
        let parsed: any = await idbGet(LOCAL_SNAPSHOT_KEY).catch(() => undefined);
        if (typeof parsed === "string") {            // newer snapshots are stored serialised
          try { parsed = JSON.parse(parsed); } catch { parsed = undefined; }
        }
        if (!parsed) {
          try {
            const raw = localStorage.getItem(LOCAL_SNAPSHOT_KEY);
            if (raw) parsed = JSON.parse(raw);
          } catch { /* ignore */ }
        }
        if (parsed) {
          // Back-compat: older snapshots stored the bare data object.
          const data = parsed && typeof parsed === "object" && "data" in parsed ? parsed.data : parsed;
          hydrate(data, false);   // reconcile with the backend copy, don't clobber
          if (parsed.sessionId) setCurrentSessionId(parsed.sessionId);
          if (parsed.sessionTitle) setCurrentSessionTitle(parsed.sessionTitle);
        }
      } catch { /* corrupt snapshot, ignore */ }
      finally { localRestored.current = true; }   // only now may auto-save run
    })();
  }, []);

  const reset = () => {
    setHistory([]); setDocQa([]); setPico({ population: "", intervention: "", comparator: "", outcome: "" }); setFramework("pico");
    setInclusion([]); setExclusion([]); setQuery(""); setUnifiedSearchQuery(""); setPerDbQueries({});
    setSimulation(null); setDbTestResults(null); setAgenticTrace(null); setAgenticSummary(null);
    setRawPapers(null); setUniquePapers(null); setDuplicatesCount(0);
    setQualityReports(null); setQualityArchive([]); setExcludedByQuality(new Set()); setQualityOverrides([]); setGradeOutcomes([]);
    setSearchLog([]); setProtocol(null); setProtocolDeviations([]); setPrismaChecklist({});
    setAbstractOverrides({}); setFullTextOverrides({});
    setRerankThreshold(-0.2); setRerankResults(null);
    setMetaOutcome(""); setMetaMeasure(""); setMetaTau2Method("DL");
    setMetaUseKnappHartung(false);
    setMetaExtractions(null); setMetaRun(null);
    setResults(null); setScreeningArchive([]); setScreeningDuration(0); setFullTextResults(null); setFtDuration(0);
    setSnowballResults(null); setSnowballScreened(null); setSnowballChosen(null); setSnowballSeeds(null); setSnowballPlan(null); setExtractedPapers(null);
    setFullTexts({}); setTextExtractions([]);
    setWritingEnriched({}); setWritingSummary(""); setWritingMethodsMain("");
    setPrisma({ identified: 0, source_counts: {}, duplicates_removed: 0, screened: 0, excluded_total: 0, exclusion_breakdown: {}, included_final: 0 });
    setCurrentSessionId(null); setCurrentSessionTitle("Untitled session");
    setCurrentProjectId(null); setCurrentProjectName(""); setCurrentProjectRole(null); setCurrentProjectMode(null);
    setPage("home");
  };

  const value: Ctx = {
    page, setPage, reviewOpen, setReviewOpen, model, setModel, sources, setSources, numPerSource, setNumPerSource, perSourceLimits, setPerSourceLimits, searchSelection, setSearchSelection, searchFilters, setSearchFilters, relevanceOverrides, setRelevanceOverrides, files, setFiles,
    history, setHistory, docQa, setDocQa, pico, setPico, framework, setFramework, inclusion, setInclusion, exclusion, setExclusion, query, setQuery,
    unifiedSearchQuery, setUnifiedSearchQuery, perDbQueries, setPerDbQueries, simulation, setSimulation,
    simulationQuery, setSimulationQuery,
    dbTestResults, setDbTestResults, agenticTrace, setAgenticTrace, agenticSummary, setAgenticSummary,
    simulationRuns, addSimulationRun, removeSimulationRun, renameSimulationRun, clearSimulationRuns,
    inUseRunId, setInUseRunId,
    rawPapers, setRawPapers, seedIds, setSeedIds, uniquePapers, setUniquePapers, duplicatesCount, setDuplicatesCount,
    qualityReports, setQualityReports, qualityArchive, setQualityArchive, excludedByQuality, setExcludedByQuality,
    qualityOverrides, setQualityOverrides, addQualityOverride, clearQualityOverrides,
    gradeOutcomes, setGradeOutcomes,
    searchLog, setSearchLog, protocol, setProtocol, protocolDeviations, setProtocolDeviations,
    prismaChecklist, setPrismaChecklist,
    abstractOverrides, setAbstractOverride, clearAbstractOverride, setAbstractOverrides,
    fullTextOverrides, setFullTextOverride, clearFullTextOverride, setFullTextOverrides,
    rerankThreshold, setRerankThreshold, rerankResults, setRerankResults,
    metaOutcome, setMetaOutcome, metaMeasure, setMetaMeasure,
    metaTau2Method, setMetaTau2Method, metaUseKnappHartung, setMetaUseKnappHartung,
    metaExtractions, setMetaExtractions, metaRun, setMetaRun,
    results, setResults, screeningArchive, setScreeningArchive, screeningDuration, setScreeningDuration, screeningPlan, setScreeningPlan, fullTextResults, setFullTextResults, ftDuration, setFtDuration,
    snowballResults, setSnowballResults, snowballScreened, setSnowballScreened, snowballChosen, setSnowballChosen, snowballSeeds, setSnowballSeeds,
    snowballPlan, setSnowballPlan,
    extractedPapers, setExtractedPapers, fullTexts, setFullTexts, textExtractions, setTextExtractions, prisma, setPrisma,
    writingEnriched, setWritingEnriched, writingSummary, setWritingSummary,
    writingMethodsMain, setWritingMethodsMain,
    currentSessionId, setCurrentSessionId, currentSessionTitle, setCurrentSessionTitle,
    currentProjectId, setCurrentProjectId,
    currentProjectName, setCurrentProjectName,
    currentProjectRole, setCurrentProjectRole,
    currentProjectMode, setCurrentProjectMode,
    snapshot, hydrate, reset,
    tasks, startTask, updateTask, updateTaskStage, appendTaskLog, cancelTask, clearTask,
  };
  return (
    <SaveStatusCtx.Provider value={lastSavedAt}>
      <StoreCtx.Provider value={value}>{children}</StoreCtx.Provider>
    </SaveStatusCtx.Provider>
  );
}

export function useStore() {
  const c = useContext(StoreCtx);
  if (!c) throw new Error("StoreProvider missing");
  return c;
}

/** When the working snapshot was last written locally (ms epoch), or null. */
export function useSaveStatus() {
  return useContext(SaveStatusCtx);
}
