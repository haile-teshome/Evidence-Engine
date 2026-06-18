// Real HTTP client for the Evidence Engine backend.
// Mirrors the contract of `mockServices.ts` so pages stay unchanged.

export type Pico = { population: string; intervention: string; comparator: string; outcome: string };
export type Paper = { id: string; source: string; title: string; abstract: string; url: string; year?: number; authors?: string };

export type Analysis = {
  p: string; i: string; c: string; o: string;
  inclusion: string[]; exclusion: string[]; query: string;
};

export type ClarifyingQuestion = {
  id: string;             // e.g. "population", "outcome", or a free-form key
  title: string;          // question text shown at the top of the modal
  options: { id: string; label: string }[];
};

export type AgentVote = { vote: "PASS" | "FAIL" | "N/A"; reasoning: string; evidence?: string };
export type AgentTrace = Record<string, AgentVote>;

// Per-PICO structured appraisal returned by the screen-abstract endpoint.
// vote semantics:
//   PASS     — abstract clearly satisfies this PICO element
//   PARTIAL  — partial / related match
//   FAIL     — clearly does not match
//   NA       — abstract lacks enough information to judge
export type PicoVote = "PASS" | "PARTIAL" | "FAIL" | "NA";

export type PicoFieldAssessment = {
  vote: PicoVote;
  evidence: string;     // short verbatim quote from the abstract, may be ""
  reasoning: string;    // one-sentence explanation
};

export type PicoAssessment = {
  population: PicoFieldAssessment;
  intervention: PicoFieldAssessment;
  comparator: PicoFieldAssessment;
  outcome: PicoFieldAssessment;
  overall_reasoning: string;  // 2-3 sentence synthesis across PICO
};

export type ScreenResult = {
  paper_id: string;
  Source: string; Title: string; URL: string; Abstract: string;
  Decision: "INCLUDE" | "EXCLUDE";
  Reason: string;
  Agent_Trace: AgentTrace;
  Pico_Assessment?: PicoAssessment;
};

export type CriterionEvidence = { decision: "INCLUDE" | "EXCLUDE"; evidence: string; reasoning: string };
export type PicoEvidence = {
  population: { evidence: string; match: "yes" | "partial" | "no"; value: string };
  intervention: { evidence: string; match: "yes" | "partial" | "no"; value: string };
  comparator: { evidence: string; match: "yes" | "partial" | "no"; value: string };
  outcome: { evidence: string; match: "yes" | "partial" | "no"; value: string };
};
export type FullTextResult = {
  paper_id: string;
  Title: string; URL: string; Source: string; Abstract: string;
  Decision: "Include" | "Exclude";
  Reason: string;
  criteriaEval: Record<string, "INCLUDE" | "EXCLUDE">;
  criteriaEvidence: Record<string, CriterionEvidence>;
  picoEvidence?: PicoEvidence;
  inclusion_score: number;
  exclusion_violations: number;
};

// Legacy type kept exported only because mockServices re-exports it.
// The new QualityReport schema does not use it.
export type QualityIssue = {
  severity: "high" | "medium" | "low";
  category: string;
  message: string;
  evidence?: string;
};

export type RoBJudgment =
  | "Low"
  | "Some Concerns"
  | "High"
  | "No information"
  | "Not applicable";

export type RoBDomain = {
  id: string;
  name: string;
  judgment: RoBJudgment;
  rationale: string;
  supporting_quote: string;
  section: string;        // "Methods" | "Results" | "Discussion" | "Abstract" | "Other" | ""
};

export type QualityReport = {
  paper_id: string;
  title: string;
  source: string;
  url: string;
  abstract: string;
  study_design: string;        // detected design label
  rubric: string;              // "RoB 2" | "ROBINS-I" | "JBI cross-sectional" | "JBI qualitative" | "AMSTAR 2"
  domains: RoBDomain[];
  overall_judgment: RoBJudgment;
  overall_rationale: string;
  used_full_text: boolean;
};

// Reviewer override on an AI-generated domain judgment. Captured for the
// audit log so every change has a timestamp, original value, and rationale.
export type QualityOverride = {
  paper_id: string;
  domain_id: string;
  original_judgment: RoBJudgment;
  new_judgment: RoBJudgment;
  reason: string;
  reviewer: string;       // optional; populated when auth is configured
  timestamp: string;      // ISO-8601
};

// ---------------------------------------------------------------------------
// Shared config
// ---------------------------------------------------------------------------

// Mutable global so the React store can update `model` without prop-drilling.
export const apiConfig: { model: string; baseUrl: string } = {
  model: "llama3.1",
  baseUrl: (import.meta as any)?.env?.VITE_API_BASE_URL || "/api",
};

const AGENTS = ["Population Agent", "Intervention Agent", "Outcome Agent", "Study Design Agent"];
const SOURCES_POOL = [
  "PubMed",
  "Europe PMC",
  "Semantic Scholar",
  "OpenAlex",
  "CrossRef",
  "arXiv",
  "bioRxiv",
  "medRxiv",
  "DOAJ",
  "CORE",
  "Scopus",
  "Embase",
  "Local PDFs",
];

async function postJSON<T = any>(path: string, body: any, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${apiConfig.baseUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
    signal,
  });
  const text = await res.text();
  let json: any = null;
  try { json = text ? JSON.parse(text) : null; } catch { /* non-json */ }
  if (!res.ok) {
    const msg = json?.detail || json?.error || text || `Request failed (${res.status})`;
    console.error(`API ${path} failed:`, msg);
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return json as T;
}

// ---------------------------------------------------------------------------
// AIService
// ---------------------------------------------------------------------------

export const AIService = {
  async inferPicoAndQuery(input: string, prior?: Partial<Analysis> | null, signal?: AbortSignal): Promise<Analysis> {
    return postJSON<Analysis>("/pico/infer", { input, prior: prior || undefined, model: apiConfig.model }, signal);
  },

  // Clarifying questions — called BEFORE the search runs. Returns 1-3 multiple-
  // choice questions that disambiguate underspecified PICO elements. The Home
  // page shows them in a Claude-style modal so the user owns the answer.
  async getClarifyingQuestions(input: string, signal?: AbortSignal): Promise<ClarifyingQuestion[]> {
    const r = await postJSON<{ questions: ClarifyingQuestion[] }>(
      "/pico/clarify-questions",
      { input, model: apiConfig.model },
      signal,
    );
    return r.questions || [];
  },

  // Conversational PICO clarifier. Returns one question at a time (exactly 3
  // specific options) or { done: true } once all PICO elements are SR-ready.
  async getClarifyNext(
    goal: string,
    picoSoFar: Record<string, string>,
    round: number,
    asked: string[] = [],
    signal?: AbortSignal,
  ): Promise<{ done: boolean; question?: ClarifyingQuestion }> {
    return postJSON<{ done: boolean; question?: ClarifyingQuestion }>(
      "/pico/clarify-next",
      { goal, pico_so_far: picoSoFar, round, asked, model: apiConfig.model },
      signal,
    );
  },

  async generateFormalQuestion(pico: Pico, signal?: AbortSignal): Promise<string> {
    const r = await postJSON<{ question: string }>("/pico/formal-question", {
      pico, model: apiConfig.model, history: [],
    }, signal);
    return r.question || "";
  },

  async generateComprehensiveSummary(goal = "", papers: Paper[] = [], signal?: AbortSignal): Promise<string> {
    const r = await postJSON<{ summary: string }>("/pico/summary", {
      goal, papers, model: apiConfig.model,
    }, signal);
    return r.summary || "";
  },

  async generateComprehensiveSummaryWithRefs(goal = "", papers: Paper[] = [], signal?: AbortSignal): Promise<{ summary: string; references: { title: string; url: string; source: string; id: string }[] }> {
    const r = await postJSON<{ summary: string; references: { title: string; url: string; source: string; id: string }[] }>("/pico/summary", {
      goal, papers, model: apiConfig.model,
    }, signal);
    return { summary: r.summary || "", references: r.references || [] };
  },

  async getRefinementSuggestions(goal = "", papers: Paper[] = [], signal?: AbortSignal): Promise<string[]> {
    const r = await postJSON<{ suggestions: string[] }>("/pico/suggestions", {
      goal, papers, model: apiConfig.model,
    }, signal);
    return r.suggestions || [];
  },

  async generateAdversarialQuery(pico: Pico, signal?: AbortSignal): Promise<string> {
    const r = await postJSON<{ query: string }>("/pico/adversarial", {
      pico, model: apiConfig.model,
    }, signal);
    return r.query || "";
  },

  async getPicoSuggestion(goal: string, category: string): Promise<string[]> {
    const r = await postJSON<{ suggestions: string[] }>("/pico/brainstorm", {
      goal: goal || "", element: category,
    });
    return r.suggestions || [];
  },

  async refinePico(pico: Pico, goal = ""): Promise<{ field: "population" | "intervention" | "comparator" | "outcome" | null; current: string; suggested: string; reason: string; is_clarification?: boolean }> {
    return postJSON("/pico/refine", { pico, goal, model: apiConfig.model });
  },

  async generateSessionTitle(goal: string, signal?: AbortSignal): Promise<string> {
    const r = await postJSON<{ title: string }>("/sessions/title", { goal, model: apiConfig.model }, signal);
    return r.title || "Untitled session";
  },

  async screenPaperMultiAgent(paper: Paper, pico: Pico, inclusion: string[] = [], exclusion: string[] = [], signal?: AbortSignal): Promise<ScreenResult> {
    return postJSON<ScreenResult>("/screen/abstract", {
      paper, pico, inclusion, exclusion, model: apiConfig.model,
    }, signal);
  },

  async screenPaperMultiAgentBatch(papers: Paper[], pico: Pico, inclusion: string[] = [], exclusion: string[] = [], signal?: AbortSignal): Promise<ScreenResult[]> {
    const r = await postJSON<{ results: ScreenResult[] }>("/screen/abstract-batch", {
      papers, pico, inclusion, exclusion, model: apiConfig.model,
    }, signal);
    return r.results || [];
  },

  async screenFullTextMultiAgent(
    row: { Title: string; URL: string; Source: string; Abstract: string; paper_id: string },
    inclusion: string[], exclusion: string[], fullText?: string, signal?: AbortSignal,
    pico?: Pico,
  ): Promise<FullTextResult> {
    return postJSON<FullTextResult>("/screen/fulltext", {
      paper: {
        id: row.paper_id,
        source: row.Source,
        title: row.Title,
        abstract: row.Abstract,
        url: row.URL,
      },
      pico: pico || { population: "", intervention: "", comparator: "", outcome: "" },
      inclusion, exclusion,
      fullText,
      model: apiConfig.model,
    }, signal);
  },

  async agenticOptimizePerSource(
    baseQuery: string, pico: Pico, sources: string[],
    onProgress?: (iter: number, total: number, source: string, count: number, relevance: number, reasoning: string) => void,
    signal?: AbortSignal,
    taskId?: string,
  ): Promise<{
    iterations_run: number;
    total_papers_found: number;
    best_relevance: number;
    per_source_queries: Record<string, string>;
    trace: { iteration: number; sources: Record<string, {
      count: number;
      relevance_score: number;
      quality_rating: string;
      query: string;
      titles: string[];
      iteration_reasoning: string;
      tactic?: string;
      query_diff?: { added: string[]; removed: string[] };
      stopped?: boolean;
      // "new_best" → this iteration improved on the running best and was adopted.
      // "tied_better_yield" → relevance matched best but more papers; kept query but still counts as non-improvement.
      // "backtrack" → this iteration scored below the running best; the previous best is preserved.
      // "stopped" → source has been removed from the loop after consecutive non-improvements.
      action?: "new_best" | "tied_better_yield" | "backtrack" | "stopped" | "tested";
      best_so_far?: {
        iteration: number;
        tactic: string;
        query: string;
        relevance_score: number;
        count: number;
      };
    }> }[];
  }> {
    // Stream via SSE so iterations show up live.
    const res = await fetch(`${apiConfig.baseUrl}/simulation/agentic/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_query: baseQuery, pico, sources, model: apiConfig.model, task_id: taskId }),
      signal,
    });
    if (!res.ok || !res.body) throw new Error(`agentic stream failed (${res.status})`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let result: any = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      // SSE events are separated by blank lines
      let idx: number;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        let event = "message";
        let data = "";
        for (const line of raw.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;
        let parsed: any;
        try { parsed = JSON.parse(data); } catch { continue; }
        if (event === "progress" && onProgress) {
          onProgress(
            parsed.iteration, parsed.total, parsed.source,
            parsed.count || 0, parsed.relevance || 0, parsed.reasoning || "",
          );
        } else if (event === "done") {
          result = parsed;
        } else if (event === "canceled") {
          const err: any = new Error("Canceled");
          err.name = "AbortError";
          throw err;
        } else if (event === "error") {
          throw new Error(parsed?.message || "agentic stream error");
        }
      }
    }
    if (!result) throw new Error("agentic stream ended without result");
    return result;
  },

  async fetchCitations(seedTitle: string, type: "Both" | "Backward (References)" | "Forward (Cited by)", maxPer: number, sources: string[], signal?: AbortSignal): Promise<any[]> {
    const r = await postJSON<{ citations: any[] }>("/citations", {
      paper_id: "", source: "", title: seedTitle, snowball_type: type, max_per: maxPer, sources,
    }, signal);
    return (r.citations || []).map((c: any) => ({
      id: c.id || c.paper_id || `${(c.title || "").slice(0, 12)}-${Math.random().toString(36).slice(2, 6)}`,
      title: c.title,
      source: c.source,
      abstract: c.abstract,
      url: c.url,
      citation_type: c.citation_type,
    }));
  },

  async fetchFullText(paper: { Title: string; URL: string; Source: string; paper_id?: string }, signal?: AbortSignal): Promise<{ status: "found" | "missing"; text?: string; reason?: string; source?: string }> {
    return postJSON("/fulltext/fetch", {
      Title: paper.Title, URL: paper.URL, Source: paper.Source, paper_id: paper.paper_id || null,
    }, signal);
  },

  async extractFromText(
    text: string,
    query: string,
    signal?: AbortSignal,
  ): Promise<{
    answer?: string;
    summary: string;
    evidence?: { quote: string; why?: string; section?: string; start: number; end: number }[];
    spans: { start: number; end: number; label?: string }[];
    values: { field: string; value: string; quote?: string; section?: string; start?: number; end?: number }[];
  }> {
    return postJSON("/extract/text", { text, query, model: apiConfig.model }, signal);
  },

  async extractTables(
    paper: {
      Title: string; URL: string; Source: string;
      paper_id?: string; Abstract?: string; full_text?: string;
    },
    signal?: AbortSignal,
  ): Promise<{ title: string; type: string; data: string[][]; caption?: string }[]> {
    const r = await postJSON<{ tables: { title: string; type: string; data: string[][]; caption?: string }[] }>("/extract/tables", {
      Title: paper.Title,
      URL: paper.URL,
      Source: paper.Source,
      paper_id: paper.paper_id || null,
      // Backend uses Abstract + full_text to seed the LLM fallback when the
      // open-access XML / HTML scrape produces nothing. Without these the
      // fallback only sees the title and never extracts anything.
      abstract: paper.Abstract || "",
      full_text: paper.full_text || "",
      model: apiConfig.model,
    }, signal);
    return r.tables || [];
  },
};

// ---------------------------------------------------------------------------
// QualityService
// ---------------------------------------------------------------------------

export const QualityService = {
  async assessPaper(
    paper: Paper,
    signal?: AbortSignal,
    opts: { fullText?: string; rubricOverride?: string } = {},
  ): Promise<QualityReport> {
    return postJSON<QualityReport>(
      "/quality/assess",
      {
        paper,
        full_text: opts.fullText,
        rubric_override: opts.rubricOverride,
        model: apiConfig.model,
      },
      signal,
    );
  },
};

// ---------------------------------------------------------------------------
// DataAggregator
// ---------------------------------------------------------------------------

export type RerankItem = {
  paper: Paper;
  leads_score: number;
  decision: string;
  reason: string;
};

export type RerankResult = {
  ranked: RerankItem[];
  kept: RerankItem[];
  threshold: number;
  quantile_keep?: number | null;
  quantile_cutoff?: number | null;
  effective_floor?: number;
  total_scored: number;
  total_kept: number;
  model_used: string;
};

// ---------------------------------------------------------------------------
// EZProxy browser-side Elsevier fetch
// ---------------------------------------------------------------------------
// After the user authenticates through UCSF's EZProxy, their browser holds a
// live session cookie for proxy.library.ucsf.edu.  These fetches route through
// that proxy so the browser sends the cookie automatically — no API key or
// institutional token required.  Results are merged with backend-fetched papers
// inside fetchAll/simulateYield below.

const EZPROXY_ELSEVIER = "https://api.elsevier.com.proxy.library.ucsf.edu";

async function _ezproxyElsevierFetch(
  query: string,
  sources: string[],
  maxPerSource: number,
): Promise<{ papers: Paper[]; sourceCounts: Record<string, number> }> {
  const papers: Paper[] = [];
  const sourceCounts: Record<string, number> = {};
  const clean = query.replace(/\[[^\]]+\]/g, "").trim() || query;

  const tasks: Promise<void>[] = [];

  if (sources.includes("Scopus")) {
    tasks.push((async () => {
      try {
        const url = `${EZPROXY_ELSEVIER}/content/search/scopus?` +
          `query=${encodeURIComponent(`TITLE-ABS-KEY(${clean})`)}&count=${maxPerSource}` +
          `&field=dc:title,dc:description,prism:doi,dc:identifier,prism:url`;
        const resp = await fetch(url, { credentials: "include" });
        if (!resp.ok) { sourceCounts["Scopus"] = 0; return; }
        const data = await resp.json();
        const entries: any[] = (data["search-results"]?.entry) || [];
        for (const e of entries) {
          const doi = (e["prism:doi"] || "").trim();
          papers.push({
            id: e["dc:identifier"] || doi || "",
            source: "Scopus",
            title: (e["dc:title"] || "").trim(),
            abstract: (e["dc:description"] || "").trim(),
            url: e["prism:url"] || (doi ? `https://doi.org/${doi}` : ""),
          });
        }
        sourceCounts["Scopus"] = entries.length;
      } catch { sourceCounts["Scopus"] = 0; }
    })());
  }

  if (sources.includes("Embase")) {
    tasks.push((async () => {
      try {
        const url = `${EZPROXY_ELSEVIER}/content/search/embase?` +
          `query=${encodeURIComponent(clean)}&count=${maxPerSource}` +
          `&field=dc:title,dc:description,prism:doi,dc:identifier,prism:url`;
        const resp = await fetch(url, { credentials: "include" });
        if (!resp.ok) { sourceCounts["Embase"] = 0; return; }
        const data = await resp.json();
        const entries: any[] = (data["search-results"]?.entry) || [];
        for (const e of entries) {
          const doi = (e["prism:doi"] || "").trim();
          papers.push({
            id: e["dc:identifier"] || doi || "",
            source: "Embase",
            title: (e["dc:title"] || "").trim(),
            abstract: (e["dc:description"] || "").trim(),
            url: e["prism:url"] || (doi ? `https://doi.org/${doi}` : ""),
          });
        }
        sourceCounts["Embase"] = entries.length;
      } catch { sourceCounts["Embase"] = 0; }
    })());
  }

  await Promise.all(tasks);
  return { papers, sourceCounts };
}

async function _ezproxyElsevierCount(
  query: string,
  sources: string[],
): Promise<Record<string, number>> {
  const counts: Record<string, number> = {};
  const clean = query.replace(/\[[^\]]+\]/g, "").trim() || query;

  const tasks: Promise<void>[] = [];

  if (sources.includes("Scopus")) {
    tasks.push((async () => {
      try {
        const url = `${EZPROXY_ELSEVIER}/content/search/scopus?` +
          `query=${encodeURIComponent(`TITLE-ABS-KEY(${clean})`)}&count=1&field=dc:identifier`;
        const resp = await fetch(url, { credentials: "include" });
        if (!resp.ok) { counts["Scopus"] = 0; return; }
        const data = await resp.json();
        counts["Scopus"] = parseInt(data["search-results"]?.["opensearch:totalResults"] || "0", 10) || 0;
      } catch { counts["Scopus"] = 0; }
    })());
  }

  if (sources.includes("Embase")) {
    tasks.push((async () => {
      try {
        const url = `${EZPROXY_ELSEVIER}/content/search/embase?` +
          `query=${encodeURIComponent(clean)}&count=1&field=dc:identifier`;
        const resp = await fetch(url, { credentials: "include" });
        if (!resp.ok) { counts["Embase"] = 0; return; }
        const data = await resp.json();
        counts["Embase"] = parseInt(data["search-results"]?.["opensearch:totalResults"] || "0", 10) || 0;
      } catch { counts["Embase"] = 0; }
    })());
  }

  await Promise.all(tasks);
  return counts;
}

export const DataAggregator = {
  // `maxPerSource` is a download budget, NOT a relevance cap. The downstream
  // rerank stage auto-detects the relevance break — anything that fetched
  // matters only insofar as LEADS can score it. Default raised to 50 so the
  // candidate pool is broad enough to find the natural break without missing
  // relevant papers from a single source.
  async fetchAll(query: string, sources: string[], _pico: Pico, maxPerSource = 50, signal?: AbortSignal, elsevierToken = "", ezproxyConnected = false): Promise<{ papers: Paper[]; sourceCounts: Record<string, number> }> {
    // When EZProxy is active the browser-side fetch handles Scopus/Embase;
    // remove them from the backend request to avoid double-fetching.
    const elsevierSources = ["Scopus", "Embase"];
    const backendSources = ezproxyConnected ? sources.filter(s => !elsevierSources.includes(s)) : sources;

    const result = await postJSON<{ papers: Paper[]; sourceCounts: Record<string, number> }>(
      "/papers/fetch",
      { query, sources: backendSources, max_per_source: maxPerSource, elsevier_token: elsevierToken },
      signal,
    );

    if (ezproxyConnected) {
      const els = await _ezproxyElsevierFetch(query, sources, maxPerSource);
      result.papers.push(...els.papers);
      Object.assign(result.sourceCounts, els.sourceCounts);
    }

    return result;
  },

  async simulateYield(query: string, sources: string[], signal?: AbortSignal, elsevierToken = "", ezproxyConnected = false): Promise<Record<string, number>> {
    const elsevierSources = ["Scopus", "Embase"];
    const backendSources = ezproxyConnected ? sources.filter(s => !elsevierSources.includes(s)) : sources;

    const r = await postJSON<{ counts: Record<string, number> }>("/simulation/yield", { query, sources: backendSources, elsevier_token: elsevierToken }, signal);
    const counts = r.counts || {};

    if (ezproxyConnected) {
      const els = await _ezproxyElsevierCount(query, sources);
      Object.assign(counts, els);
    }

    return counts;
  },

  // Score fetched papers for relevance against PICO using LEADS-native.
  // Returns both the full ranked list (with scores) and the subset that passed
  // the relevance threshold. Feed `kept` to the summariser.
  //
  // `threshold` is the absolute floor in [-1, +1]. `quantileKeep`, if set,
  // additionally requires the paper to be in the top `quantileKeep` fraction of
  // the scored corpus (e.g. 0.30 = top 30 %). The effective acceptance bar is
  // max(absolute floor, quantile cutoff) — both gates must be cleared.
  async rerankByRelevance(
    papers: Paper[],
    pico: Pico,
    inclusion: string[] = [],
    exclusion: string[] = [],
    threshold = -0.2,
    topK?: number,
    signal?: AbortSignal,
    quantileKeep?: number,
  ): Promise<RerankResult> {
    return postJSON<RerankResult>(
      "/papers/rerank",
      {
        papers, pico, inclusion, exclusion,
        threshold,
        quantile_keep: quantileKeep,
        top_k: topK,
        model: apiConfig.model,
      },
      signal,
    );
  },
};

// ---------------------------------------------------------------------------
// Deduplicator — kept client-side (pure logic, no backend round-trip)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Meta-analysis agent
// ---------------------------------------------------------------------------

export type EffectMeasure =
  | "OR" | "RR" | "RD" | "PETO_OR" | "HR"
  | "MD" | "SMD"
  | "PROP" | "IR" | "ZCOR"
  | "GENERIC" | "UNKNOWN";

export type Tau2Method = "DL" | "PM" | "REML";

export type StudyEffect = {
  paper_id: string;
  title: string;
  url?: string | null;
  outcome: string;
  effect_measure: EffectMeasure;

  // Binary 2x2
  events_t?: number | null; n_t?: number | null;
  events_c?: number | null; n_c?: number | null;

  // Continuous two-arm
  mean_t?: number | null; sd_t?: number | null;
  mean_c?: number | null; sd_c?: number | null;

  // Time-to-event reported as HR + CI
  hr?: number | null;
  hr_ci_low?: number | null;
  hr_ci_high?: number | null;
  log_hr?: number | null;
  log_hr_se?: number | null;

  // Single-arm prevalence / proportion
  events_total?: number | null;
  n_total?: number | null;

  // Single-arm incidence rate
  person_time?: number | null;

  // Correlation
  correlation?: number | null;

  // Generic IV / computed-effect storage
  yi?: number | null; vi?: number | null; se?: number | null;
  ci_low?: number | null; ci_high?: number | null;

  // Moderators
  subgroup?: string | null;
  moderator?: number | null;

  // LLM transparency
  extraction_quote?: string | null;
  extraction_confidence?: number | null;
  extraction_notes?: string | null;
  error?: string | null;
  weight_fe?: number;
  weight_re?: number;
};

export type PooledEstimate = {
  estimate: number; se: number; ci_low: number; ci_high: number;
  z?: number; p_value?: number;
};

export type MetaPoolResult = {
  k: number;
  effect_measures: EffectMeasure[];
  tau2_method: Tau2Method;
  use_knapp_hartung: boolean;
  valid_studies: StudyEffect[];
  invalid_studies: StudyEffect[];
  fixed: PooledEstimate | null;
  random: (PooledEstimate & { tau2: number; tau: number }) | null;
  heterogeneity: { Q: number; df: number; Q_p_value: number; I2_pct: number; H2: number; tau2: number } | null;
  prediction_interval: { low: number; high: number } | null;
};

export type SubgroupResult = {
  groups: Array<{
    name: string;
    k: number;
    estimate: number | null;
    ci_low: number | null;
    ci_high: number | null;
    se: number | null;
    tau2: number | null;
    I2_pct: number | null;
  }>;
  Q_between: number;
  df_between: number;
  Q_between_p: number;
};

export type LOORow = {
  paper_id: string;
  title: string;
  estimate_without: number | null;
  ci_low_without: number | null;
  ci_high_without: number | null;
  delta: number | null;
};

export type CumulativeRow = {
  k: number;
  last_added: string;
  estimate: number | null;
  ci_low: number | null;
  ci_high: number | null;
  I2_pct: number | null;
};

export type FunnelData = {
  center: number | null;
  studies: Array<{ paper_id: string; title: string; yi: number; se: number }>;
};

export type EggerResult = {
  intercept: number | null; se: number | null;
  t: number | null; p_value: number | null;
  k: number; note?: string;
};

export type BeggResult = {
  tau: number | null; z?: number; p_value: number | null;
  k: number; note?: string;
};

export type TrimFillResult = {
  k0: number; side?: "left" | "right";
  filled_estimate: number | null;
  filled_ci_low: number | null;
  filled_ci_high: number | null;
  filled_k?: number;
  note?: string;
};

export type MetaRegressionResult = {
  slope: number | null; se: number | null;
  intercept: number | null; z: number | null;
  p_value: number | null; R2: number | null;
  k: number; tau2?: number; note?: string;
};

export type MetaRunResult = {
  pool: MetaPoolResult;
  subgroup: SubgroupResult;
  leave_one_out: LOORow[];
  cumulative: CumulativeRow[];
  funnel: FunnelData;
  egger: EggerResult;
  begg: BeggResult;
  trim_fill: TrimFillResult;
  meta_regression: MetaRegressionResult;
};

export const MetaAnalysisService = {
  async extract(
    papers: Paper[],
    outcome: string,
    measure: string = "",
    fullTexts: Record<string, string> = {},
    signal?: AbortSignal,
  ): Promise<{ extractions: StudyEffect[]; model_used: string; outcome: string }> {
    return postJSON<{ extractions: StudyEffect[]; model_used: string; outcome: string }>(
      "/meta/extract",
      { papers, outcome, measure, full_texts: fullTexts, model: apiConfig.model },
      signal,
    );
  },

  async pool(
    extractions: StudyEffect[],
    tau2Method: Tau2Method = "DL",
    useKnappHartung: boolean = false,
    signal?: AbortSignal,
  ): Promise<MetaPoolResult> {
    return postJSON<MetaPoolResult>(
      "/meta/pool",
      { extractions, tau2_method: tau2Method, use_knapp_hartung: useKnappHartung },
      signal,
    );
  },

  async run(
    extractions: StudyEffect[],
    tau2Method: Tau2Method = "DL",
    useKnappHartung: boolean = false,
    signal?: AbortSignal,
  ): Promise<MetaRunResult> {
    return postJSON<MetaRunResult>(
      "/meta/run",
      { extractions, tau2_method: tau2Method, use_knapp_hartung: useKnappHartung },
      signal,
    );
  },
};

export const Deduplicator = {
  run(papers: Paper[]): { unique: Paper[]; duplicates: Paper[] } {
    const seen = new Set<string>();
    const unique: Paper[] = [];
    const duplicates: Paper[] = [];
    for (const p of papers) {
      const key = (p.title || "").toLowerCase().trim();
      if (seen.has(key)) duplicates.push(p);
      else { seen.add(key); unique.push(p); }
    }
    return { unique, duplicates };
  },
};

// ---------------------------------------------------------------------------
// Constants + helpers (unchanged from mockServices)
// ---------------------------------------------------------------------------

export const ALL_SOURCES = SOURCES_POOL;
export const AGENT_NAMES = AGENTS;

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}
