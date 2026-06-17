import { createContext, useContext, useEffect, useRef, useState, ReactNode } from "react";
import { Pico, Analysis, ScreenResult, FullTextResult, Paper, QualityReport, QualityOverride } from "./mockServices";
import { apiConfig, RerankResult, StudyEffect, MetaRunResult, EffectMeasure, Tau2Method } from "./apiClient";

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

export type FullTextRecord = {
  paper_id: string;
  title: string;
  url: string;
  source: string;                       // paper's originating database
  status: "found" | "missing" | "pending";
  text?: string;
  reason?: string;
  /** Where the full text was retrieved from on this fetch: e.g.
   *  "Europe PMC (XML)", "PMC PDF (PMC1234567)", "Unpaywall PDF (nature.com)",
   *  "arXiv PDF (2304.12345)", "HTML scrape (publisher.com)". Different from
   *  `source` (which is the paper's database). */
  retrieved_via?: string;
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
};

export type ExtractedTable = { title: string; type: string; data: string[][]; caption?: string };
export type ExtractedPaper = { Paper_Title: string; Paper_URL: string; Source: string; Extracted_Tables: ExtractedTable[] };

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
  model: string; setModel: (v: string) => void;
  sources: string[]; setSources: (v: string[]) => void;
  numPerSource: number; setNumPerSource: (v: number) => void;
  files: File[]; setFiles: (v: File[]) => void;

  // Strategy
  history: HistoryEntry[]; setHistory: React.Dispatch<React.SetStateAction<HistoryEntry[]>>;
  pico: Pico; setPico: React.Dispatch<React.SetStateAction<Pico>>;
  inclusion: string[]; setInclusion: (v: string[]) => void;
  exclusion: string[]; setExclusion: (v: string[]) => void;
  query: string; setQuery: (v: string) => void;

  // Simulation
  unifiedSearchQuery: string; setUnifiedSearchQuery: (v: string) => void;
  perDbQueries: Record<string, string>; setPerDbQueries: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  simulation: Record<string, number> | null; setSimulation: (v: Record<string, number> | null) => void;
  dbTestResults: Record<string, { query: string; total_found: number; papers: { title: string; url: string }[] }> | null;
  setDbTestResults: React.Dispatch<React.SetStateAction<Record<string, { query: string; total_found: number; papers: { title: string; url: string }[] }> | null>>;
  agenticTrace: any[] | null; setAgenticTrace: (v: any[] | null) => void;
  agenticSummary: { iterations_run: number; total_papers_found: number; best_relevance: number } | null;
  setAgenticSummary: (v: { iterations_run: number; total_papers_found: number; best_relevance: number } | null) => void;
  simulationRuns: SimulationRun[];
  addSimulationRun: (run: Omit<SimulationRun, "id" | "timestamp" | "label">) => void;
  clearSimulationRuns: () => void;

  // Quality Assessment
  rawPapers: Paper[] | null; setRawPapers: (v: Paper[] | null) => void;
  uniquePapers: Paper[] | null; setUniquePapers: (v: Paper[] | null) => void;
  duplicatesCount: number; setDuplicatesCount: (v: number) => void;
  qualityReports: QualityReport[] | null; setQualityReports: (v: QualityReport[] | null) => void;
  excludedByQuality: Set<string>; setExcludedByQuality: React.Dispatch<React.SetStateAction<Set<string>>>;

  // Reviewer overrides on screening decisions. Keyed by paper_id; value is
  // the reviewer's effective decision ("INCLUDE" / "EXCLUDE" for abstract,
  // "Include" / "Exclude" for full-text — matching the case the screener
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

  // Relevance reranking — LEADS-scored papers from the home-analysis pipeline.
  // Threshold is user-tunable in the sidebar; rerankResults holds the full
  // ranked list so the UI can re-filter without re-running LLM calls.
  rerankThreshold: number; setRerankThreshold: (v: number) => void;
  rerankResults: RerankResult | null; setRerankResults: (v: RerankResult | null) => void;

  // Meta-analysis agent — extracted effect sizes + full analysis bundle
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
  fullTextResults: FullTextResult[] | null; setFullTextResults: (v: FullTextResult[] | null) => void;
  ftDuration: number; setFtDuration: (v: number) => void;

  // Snowball
  snowballResults: any[] | null; setSnowballResults: (v: any[] | null) => void;
  snowballScreened: ScreenResult[] | null; setSnowballScreened: (v: ScreenResult[] | null) => void;

  // Full-text acquisition
  fullTexts: Record<string, FullTextRecord>; setFullTexts: React.Dispatch<React.SetStateAction<Record<string, FullTextRecord>>>;

  // Extraction
  extractedPapers: ExtractedPaper[] | null; setExtractedPapers: (v: ExtractedPaper[] | null) => void;
  textExtractions: TextExtractionResult[]; setTextExtractions: React.Dispatch<React.SetStateAction<TextExtractionResult[]>>;

  // PRISMA
  prisma: PrismaCounts; setPrisma: React.Dispatch<React.SetStateAction<PrismaCounts>>;

  // Elsevier institutional access.
  // EZProxy mode (primary — no registration needed): after the user clicks
  // "Connect via UCSF" and completes MyAccess, the browser holds a live
  // proxy.library.ucsf.edu session cookie.  The frontend fetches Embase/
  // Scopus directly through the proxied URL and merges with backend results.
  ezproxyConnected: boolean; setEzproxyConnected: (v: boolean) => void;
  // OAuth mode (optional upgrade): set when ELSEVIER_OAUTH_CLIENT_ID is
  // configured and the user completes the OAuth popup flow.
  elsevierToken: string; setElsevierToken: (v: string) => void;

  // Session persistence
  currentSessionId: string | null; setCurrentSessionId: (v: string | null) => void;
  currentSessionTitle: string; setCurrentSessionTitle: (v: string) => void;
  snapshot: () => any;
  hydrate: (data: any) => void;
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
  try {
    const v = localStorage.getItem(PAGE_STORAGE_KEY) as PageId | null;
    if (v && VALID_PAGES.includes(v)) return v;
  } catch { /* localStorage unavailable (private mode, etc.) */ }
  return "home";
}

export function StoreProvider({ children }: { children: ReactNode }) {
  // Restore the last-viewed tab so a browser refresh keeps you in place.
  const [page, setPage] = useState<PageId>(loadPage);
  useEffect(() => {
    try { localStorage.setItem(PAGE_STORAGE_KEY, page); } catch { /* ignore */ }
  }, [page]);
  // Default to LEADS — the benchmark's highest-performing screening model
  // (LEADS-mistral-7b × LEADS-native @ score ≥ +0.20: recall=1.000,
  // specificity=0.676, MCC=+0.260 on van_Dis_2020). The Backend resolves
  // "leads" to the full GGUF tag and routes to the LEADS-native pipeline.
  const [model, setModel] = useState("leads");
  const [sources, setSources] = useState<string[]>(["PubMed", "Europe PMC", "Semantic Scholar"]);
  const [numPerSource, setNumPerSource] = useState(15);
  const [files, setFiles] = useState<File[]>([]);

  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [pico, setPico] = useState<Pico>({ population: "", intervention: "", comparator: "", outcome: "" });
  const [inclusion, setInclusion] = useState<string[]>([]);
  const [exclusion, setExclusion] = useState<string[]>([]);
  const [query, setQuery] = useState("");

  const [unifiedSearchQuery, setUnifiedSearchQuery] = useState("");
  const [perDbQueries, setPerDbQueries] = useState<Record<string, string>>({});
  const [simulation, setSimulation] = useState<Record<string, number> | null>(null);
  const [dbTestResults, setDbTestResults] = useState<Record<string, { query: string; total_found: number; papers: { title: string; url: string }[] }> | null>(null);
  const [agenticTrace, setAgenticTrace] = useState<any[] | null>(null);
  const [agenticSummary, setAgenticSummary] = useState<{ iterations_run: number; total_papers_found: number; best_relevance: number } | null>(null);
  const [simulationRuns, setSimulationRuns] = useState<SimulationRun[]>([]);
  const addSimulationRun = (run: Omit<SimulationRun, "id" | "timestamp" | "label">) => {
    setSimulationRuns(prev => {
      const next = [...prev, {
        ...run,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        label: `Run ${prev.length + 1}`,
      }].slice(-20); // keep last 20
      return next;
    });
  };
  const clearSimulationRuns = () => setSimulationRuns([]);

  const [rawPapers, setRawPapers] = useState<Paper[] | null>(null);
  const [uniquePapers, setUniquePapers] = useState<Paper[] | null>(null);
  const [duplicatesCount, setDuplicatesCount] = useState(0);
  const [qualityReports, setQualityReports] = useState<QualityReport[] | null>(null);
  const [excludedByQuality, setExcludedByQuality] = useState<Set<string>>(new Set());
  const [qualityOverrides, setQualityOverrides] = useState<QualityOverride[]>([]);

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
  const [fullTextResults, setFullTextResults] = useState<FullTextResult[] | null>(null);
  const [ftDuration, setFtDuration] = useState(0);

  const [snowballResults, setSnowballResults] = useState<any[] | null>(null);
  const [snowballScreened, setSnowballScreened] = useState<ScreenResult[] | null>(null);

  const [extractedPapers, setExtractedPapers] = useState<ExtractedPaper[] | null>(null);
  const [fullTexts, setFullTexts] = useState<Record<string, FullTextRecord>>({});
  const [textExtractions, setTextExtractions] = useState<TextExtractionResult[]>([]);

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

  const [ezproxyConnected, setEzproxyConnected] = useState(false);
  const [elsevierToken, setElsevierToken] = useState("");

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
    history, pico, inclusion, exclusion, query, unifiedSearchQuery, perDbQueries,
    sources, numPerSource, model,
    rawPapers, uniquePapers, duplicatesCount, qualityReports,
    excludedByQuality: Array.from(excludedByQuality),
    qualityOverrides,
    abstractOverrides,
    fullTextOverrides,
    rerankThreshold, rerankResults,
    results, fullTextResults, snowballResults, snowballScreened, extractedPapers, prisma,
  });

  const hydrate = (d: any) => {
    if (!d) return;
    setHistory(d.history || []);
    setPico(d.pico || { population: "", intervention: "", comparator: "", outcome: "" });
    setInclusion(d.inclusion || []);
    setExclusion(d.exclusion || []);
    setQuery(d.query || "");
    setUnifiedSearchQuery(d.unifiedSearchQuery || "");
    setPerDbQueries(d.perDbQueries || {});
    if (Array.isArray(d.sources)) setSources(d.sources);
    if (typeof d.numPerSource === "number") setNumPerSource(d.numPerSource);
    if (d.model) setModel(d.model);
    setRawPapers(d.rawPapers ?? null);
    setUniquePapers(d.uniquePapers ?? null);
    setDuplicatesCount(d.duplicatesCount ?? 0);
    setQualityReports(d.qualityReports ?? null);
    setExcludedByQuality(new Set(d.excludedByQuality || []));
    setQualityOverrides(Array.isArray(d.qualityOverrides) ? d.qualityOverrides : []);
    setAbstractOverrides(d.abstractOverrides && typeof d.abstractOverrides === "object" ? d.abstractOverrides : {});
    setFullTextOverrides(d.fullTextOverrides && typeof d.fullTextOverrides === "object" ? d.fullTextOverrides : {});
    if (typeof d.rerankThreshold === "number") setRerankThreshold(d.rerankThreshold);
    setRerankResults(d.rerankResults ?? null);
    setResults(d.results ?? null);
    setFullTextResults(d.fullTextResults ?? null);
    setSnowballResults(d.snowballResults ?? null);
    setSnowballScreened(d.snowballScreened ?? null);
    setExtractedPapers(d.extractedPapers ?? null);
    if (d.prisma) setPrisma(d.prisma);
  };

  // ── Local persistence ─────────────────────────────────────────────────────
  // Restore the locally-saved snapshot once on mount, then keep it in sync
  // (debounced) so a refresh lands the user back in their review. Guarded so
  // the first empty render never wipes the saved snapshot before it's read.
  const localRestored = useRef(false);
  useEffect(() => {
    if (!localRestored.current) return;          // wait until restore has run
    const t = setTimeout(() => {
      try {
        if (history.length === 0) { localStorage.removeItem(LOCAL_SNAPSHOT_KEY); return; }
        localStorage.setItem(LOCAL_SNAPSHOT_KEY, JSON.stringify(snapshot()));
      } catch { /* quota exceeded or unserialisable — skip this cycle */ }
    }, 600);
    return () => clearTimeout(t);
  }, [history, pico, inclusion, exclusion, query, unifiedSearchQuery, perDbQueries,
      sources, numPerSource, model, rawPapers, uniquePapers, duplicatesCount,
      qualityReports, excludedByQuality, qualityOverrides, abstractOverrides,
      fullTextOverrides, rerankThreshold, rerankResults, results, fullTextResults,
      snowballResults, snowballScreened, extractedPapers, prisma]);

  useEffect(() => {
    if (localRestored.current) return;
    localRestored.current = true;
    try {
      const raw = localStorage.getItem(LOCAL_SNAPSHOT_KEY);
      if (raw) hydrate(JSON.parse(raw));
    } catch { /* corrupt snapshot — ignore */ }
  }, []);

  const reset = () => {
    setHistory([]); setPico({ population: "", intervention: "", comparator: "", outcome: "" });
    setInclusion([]); setExclusion([]); setQuery(""); setUnifiedSearchQuery(""); setPerDbQueries({});
    setSimulation(null); setDbTestResults(null); setAgenticTrace(null); setAgenticSummary(null);
    setRawPapers(null); setUniquePapers(null); setDuplicatesCount(0);
    setQualityReports(null); setExcludedByQuality(new Set()); setQualityOverrides([]);
    setAbstractOverrides({}); setFullTextOverrides({});
    setRerankThreshold(-0.2); setRerankResults(null);
    setMetaOutcome(""); setMetaMeasure(""); setMetaTau2Method("DL");
    setMetaUseKnappHartung(false);
    setMetaExtractions(null); setMetaRun(null);
    setResults(null); setScreeningDuration(0); setFullTextResults(null); setFtDuration(0);
    setSnowballResults(null); setSnowballScreened(null); setExtractedPapers(null);
    setFullTexts({}); setTextExtractions([]);
    setPrisma({ identified: 0, source_counts: {}, duplicates_removed: 0, screened: 0, excluded_total: 0, exclusion_breakdown: {}, included_final: 0 });
    setCurrentSessionId(null); setCurrentSessionTitle("Untitled session");
    setCurrentProjectId(null); setCurrentProjectName(""); setCurrentProjectRole(null); setCurrentProjectMode(null);
    setPage("home");
  };

  const value: Ctx = {
    page, setPage, model, setModel, sources, setSources, numPerSource, setNumPerSource, files, setFiles,
    history, setHistory, pico, setPico, inclusion, setInclusion, exclusion, setExclusion, query, setQuery,
    unifiedSearchQuery, setUnifiedSearchQuery, perDbQueries, setPerDbQueries, simulation, setSimulation,
    dbTestResults, setDbTestResults, agenticTrace, setAgenticTrace, agenticSummary, setAgenticSummary,
    simulationRuns, addSimulationRun, clearSimulationRuns,
    rawPapers, setRawPapers, uniquePapers, setUniquePapers, duplicatesCount, setDuplicatesCount,
    qualityReports, setQualityReports, excludedByQuality, setExcludedByQuality,
    qualityOverrides, setQualityOverrides, addQualityOverride, clearQualityOverrides,
    abstractOverrides, setAbstractOverride, clearAbstractOverride, setAbstractOverrides,
    fullTextOverrides, setFullTextOverride, clearFullTextOverride, setFullTextOverrides,
    rerankThreshold, setRerankThreshold, rerankResults, setRerankResults,
    metaOutcome, setMetaOutcome, metaMeasure, setMetaMeasure,
    metaTau2Method, setMetaTau2Method, metaUseKnappHartung, setMetaUseKnappHartung,
    metaExtractions, setMetaExtractions, metaRun, setMetaRun,
    results, setResults, screeningDuration, setScreeningDuration, fullTextResults, setFullTextResults, ftDuration, setFtDuration,
    snowballResults, setSnowballResults, snowballScreened, setSnowballScreened,
    extractedPapers, setExtractedPapers, fullTexts, setFullTexts, textExtractions, setTextExtractions, prisma, setPrisma,
    currentSessionId, setCurrentSessionId, currentSessionTitle, setCurrentSessionTitle,
    ezproxyConnected, setEzproxyConnected,
    elsevierToken, setElsevierToken,
    currentProjectId, setCurrentProjectId,
    currentProjectName, setCurrentProjectName,
    currentProjectRole, setCurrentProjectRole,
    currentProjectMode, setCurrentProjectMode,
    snapshot, hydrate, reset,
    tasks, startTask, updateTask, updateTaskStage, appendTaskLog, cancelTask, clearTask,
  };
  return <StoreCtx.Provider value={value}>{children}</StoreCtx.Provider>;
}

export function useStore() {
  const c = useContext(StoreCtx);
  if (!c) throw new Error("StoreProvider missing");
  return c;
}
