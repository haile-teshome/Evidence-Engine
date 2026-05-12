// Real HTTP client for the Evidence Engine backend.
// Mirrors the contract of `mockServices.ts` so pages stay unchanged.

export type Pico = { population: string; intervention: string; comparator: string; outcome: string };
export type Paper = { id: string; source: string; title: string; abstract: string; url: string; year?: number; authors?: string };

export type Analysis = {
  p: string; i: string; c: string; o: string;
  inclusion: string[]; exclusion: string[]; query: string;
};

export type AgentVote = { vote: "PASS" | "FAIL" | "N/A"; reasoning: string; evidence?: string };
export type AgentTrace = Record<string, AgentVote>;

export type ScreenResult = {
  paper_id: string;
  Source: string; Title: string; URL: string; Abstract: string;
  Decision: "INCLUDE" | "EXCLUDE";
  Reason: string;
  Agent_Trace: AgentTrace;
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

export type QualityIssue = {
  severity: "high" | "medium" | "low";
  category: string;
  message: string;
  evidence?: string;
};

export type QualityReport = {
  paper_id: string;
  title: string;
  source: string;
  url: string;
  abstract: string;
  score: number;
  rating: "Excellent" | "Good" | "Fair" | "Poor";
  issues: QualityIssue[];
  highlightedAbstract: { text: string; flagged: boolean; reason?: string }[];
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
  async inferPicoAndQuery(input: string, signal?: AbortSignal): Promise<Analysis> {
    return postJSON<Analysis>("/pico/infer", { input, model: apiConfig.model }, signal);
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

  async refinePico(pico: Pico, goal = ""): Promise<{ field: "population" | "intervention" | "comparator" | "outcome" | null; current: string; suggested: string; reason: string }> {
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
    trace: { iteration: number; sources: Record<string, { count: number; relevance_score: number; quality_rating: string; query: string; titles: string[]; iteration_reasoning: string }> }[];
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
      let idx;
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

  async extractFromText(text: string, query: string, signal?: AbortSignal): Promise<{ summary: string; spans: { start: number; end: number; label?: string }[]; values: { field: string; value: string; quote?: string }[] }> {
    return postJSON("/extract/text", { text, query }, signal);
  },

  async extractTables(paper: { Title: string; URL: string; Source: string; paper_id?: string }, signal?: AbortSignal): Promise<{ title: string; type: string; data: string[][]; caption?: string }[]> {
    const r = await postJSON<{ tables: { title: string; type: string; data: string[][]; caption?: string }[] }>("/extract/tables", {
      Title: paper.Title, URL: paper.URL, Source: paper.Source, paper_id: paper.paper_id || null, model: apiConfig.model,
    }, signal);
    return r.tables || [];
  },
};

// ---------------------------------------------------------------------------
// QualityService
// ---------------------------------------------------------------------------

export const QualityService = {
  async assessPaper(paper: Paper, signal?: AbortSignal): Promise<QualityReport> {
    return postJSON<QualityReport>("/quality/assess", { paper }, signal);
  },
};

// ---------------------------------------------------------------------------
// DataAggregator
// ---------------------------------------------------------------------------

export const DataAggregator = {
  async fetchAll(query: string, sources: string[], _pico: Pico, maxPerSource = 10, signal?: AbortSignal): Promise<{ papers: Paper[]; sourceCounts: Record<string, number> }> {
    return postJSON("/papers/fetch", { query, sources, max_per_source: maxPerSource }, signal);
  },

  async simulateYield(query: string, sources: string[], signal?: AbortSignal): Promise<Record<string, number>> {
    const r = await postJSON<{ counts: Record<string, number> }>("/simulation/yield", { query, sources }, signal);
    return r.counts || {};
  },
};

// ---------------------------------------------------------------------------
// Deduplicator — kept client-side (pure logic, no backend round-trip)
// ---------------------------------------------------------------------------

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
