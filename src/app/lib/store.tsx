import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { Pico, Analysis, ScreenResult, FullTextResult, Paper, QualityReport } from "./mockServices";
import { apiConfig } from "./apiClient";

export type PageId = "home" | "simulation" | "quality" | "abstract" | "acquisition" | "fulltext" | "snowball" | "extraction" | "textextraction" | "prisma" | "benchmark";

export type FullTextRecord = { paper_id: string; title: string; url: string; source: string; status: "found" | "missing" | "pending"; text?: string; reason?: string };
export type TextExtractionResult = { paper_id: string; title: string; query: string; summary: string; spans: { start: number; end: number; label?: string }[]; values: { field: string; value: string; quote?: string }[] };

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

  // Quality Assessment
  rawPapers: Paper[] | null; setRawPapers: (v: Paper[] | null) => void;
  uniquePapers: Paper[] | null; setUniquePapers: (v: Paper[] | null) => void;
  duplicatesCount: number; setDuplicatesCount: (v: number) => void;
  qualityReports: QualityReport[] | null; setQualityReports: (v: QualityReport[] | null) => void;
  excludedByQuality: Set<string>; setExcludedByQuality: React.Dispatch<React.SetStateAction<Set<string>>>;

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

  // Session persistence
  currentSessionId: string | null; setCurrentSessionId: (v: string | null) => void;
  currentSessionTitle: string; setCurrentSessionTitle: (v: string) => void;
  snapshot: () => any;
  hydrate: (data: any) => void;
  reset: () => void;

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

export function StoreProvider({ children }: { children: ReactNode }) {
  const [page, setPage] = useState<PageId>("home");
  const [model, setModel] = useState("llama3.1");
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

  const [rawPapers, setRawPapers] = useState<Paper[] | null>(null);
  const [uniquePapers, setUniquePapers] = useState<Paper[] | null>(null);
  const [duplicatesCount, setDuplicatesCount] = useState(0);
  const [qualityReports, setQualityReports] = useState<QualityReport[] | null>(null);
  const [excludedByQuality, setExcludedByQuality] = useState<Set<string>>(new Set());

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
    setResults(d.results ?? null);
    setFullTextResults(d.fullTextResults ?? null);
    setSnowballResults(d.snowballResults ?? null);
    setSnowballScreened(d.snowballScreened ?? null);
    setExtractedPapers(d.extractedPapers ?? null);
    if (d.prisma) setPrisma(d.prisma);
  };

  const reset = () => {
    setHistory([]); setPico({ population: "", intervention: "", comparator: "", outcome: "" });
    setInclusion([]); setExclusion([]); setQuery(""); setUnifiedSearchQuery(""); setPerDbQueries({});
    setSimulation(null); setDbTestResults(null); setAgenticTrace(null); setAgenticSummary(null);
    setRawPapers(null); setUniquePapers(null); setDuplicatesCount(0);
    setQualityReports(null); setExcludedByQuality(new Set());
    setResults(null); setScreeningDuration(0); setFullTextResults(null); setFtDuration(0);
    setSnowballResults(null); setSnowballScreened(null); setExtractedPapers(null);
    setFullTexts({}); setTextExtractions([]);
    setPrisma({ identified: 0, source_counts: {}, duplicates_removed: 0, screened: 0, excluded_total: 0, exclusion_breakdown: {}, included_final: 0 });
    setCurrentSessionId(null); setCurrentSessionTitle("Untitled session"); setPage("home");
  };

  const value: Ctx = {
    page, setPage, model, setModel, sources, setSources, numPerSource, setNumPerSource, files, setFiles,
    history, setHistory, pico, setPico, inclusion, setInclusion, exclusion, setExclusion, query, setQuery,
    unifiedSearchQuery, setUnifiedSearchQuery, perDbQueries, setPerDbQueries, simulation, setSimulation,
    dbTestResults, setDbTestResults, agenticTrace, setAgenticTrace, agenticSummary, setAgenticSummary,
    rawPapers, setRawPapers, uniquePapers, setUniquePapers, duplicatesCount, setDuplicatesCount,
    qualityReports, setQualityReports, excludedByQuality, setExcludedByQuality,
    results, setResults, screeningDuration, setScreeningDuration, fullTextResults, setFullTextResults, ftDuration, setFtDuration,
    snowballResults, setSnowballResults, snowballScreened, setSnowballScreened,
    extractedPapers, setExtractedPapers, fullTexts, setFullTexts, textExtractions, setTextExtractions, prisma, setPrisma,
    currentSessionId, setCurrentSessionId, currentSessionTitle, setCurrentSessionTitle,
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
