// ---------------------------------------------------------------------------
// Auditable AI decision log.
//
// Compiles every AI-made or AI-suggested decision the app has stored (relevance
// screening and risk-of-bias appraisal, each already criterion-level) and stamps
// them with the run manifest (model, seed, prompt version) so the whole review is
// transparent and reproducible. Exportable as JSON or CSV for a PRISMA-AI / RAISE
// AI-use declaration. This reads what is already in the store; nothing new is sent
// to any model.
// ---------------------------------------------------------------------------

export type AuditManifest = {
  model?: string;
  seed?: number;
  prompt_version?: string;
  temperature?: number;
  local_models?: { name: string; digest: string }[];
};

export type AuditCriterion = { name: string; verdict: string; reasoning: string };

export type AuditDecision = {
  stage: string;
  id: string;
  item: string;
  verdict: string;
  score?: number | null;
  reasoning: string;
  criteria?: AuditCriterion[];
  assessed_at?: string;
};

export type AuditLog = {
  generated_at: string;
  manifest: AuditManifest;
  decisions: AuditDecision[];
  note: string;
};

/** Build the decision log from stored results + the reproducibility manifest. */
export function compileAuditLog(opts: {
  qualityReports?: any[] | null;
  rerankResults?: any | null;
  model?: string;
  manifest?: AuditManifest | null;
  generatedAt?: string;
}): AuditLog {
  const decisions: AuditDecision[] = [];

  // Relevance / abstract screening (LEADS rerank): decision + reason per paper.
  const ranked = opts.rerankResults?.ranked || [];
  for (const r of ranked) {
    decisions.push({
      stage: "Relevance screening",
      id: r?.paper?.id || "",
      item: r?.paper?.title || "Untitled",
      verdict: r?.decision || "",
      score: typeof r?.leads_score === "number" ? r.leads_score : null,
      reasoning: r?.reason || "",
    });
  }

  // Risk of bias / quality appraisal: overall + per-domain (criterion-level).
  for (const q of opts.qualityReports || []) {
    decisions.push({
      stage: `Risk of bias (${q?.instrument || q?.rubric || "RoB"})`,
      id: q?.paper_id || "",
      item: q?.title || "Untitled",
      verdict: q?.overall_judgment || "",
      reasoning: q?.overall_rationale || "",
      assessed_at: q?.assessed_at,
      criteria: (q?.domains || []).map((d: any) => ({
        name: d?.name || d?.domain || d?.id || "domain",
        verdict: d?.judgment || d?.judgement || "",
        reasoning: d?.rationale || d?.reasoning || "",
      })),
    });
  }

  return {
    // Default to now. This is a compliance artifact: a RAISE / PRISMA-AI
    // disclosure has to state when the AI-assisted decisions were made, and
    // the only caller does not pass a timestamp, so every export carried "".
    generated_at: opts.generatedAt || new Date().toISOString(),
    manifest: { ...(opts.manifest || {}), model: opts.model },
    decisions,
    note: "Each decision was produced by the model and run settings in the manifest. "
      + "Local (Ollama) and OpenAI runs at temperature 0 with the listed seed reproduce "
      + "to identical output; the reviewer can override any AI judgment.",
  };
}

export function auditToJson(log: AuditLog): string {
  return JSON.stringify(log, null, 2);
}

export function auditToCsv(log: AuditLog): string {
  const m = log.manifest;
  const head = [
    "Stage", "Item", "ID", "Verdict", "Score", "Reasoning",
    "Criterion", "Criterion verdict", "Criterion reasoning",
    "Model", "Seed", "Prompt version",
  ];
  const rows: string[][] = [head];
  const base = (d: AuditDecision) => [
    d.stage, d.item, d.id, d.verdict, d.score == null ? "" : String(d.score), d.reasoning,
  ];
  const tail = [m.model || "", m.seed == null ? "" : String(m.seed), m.prompt_version || ""];
  for (const d of log.decisions) {
    if (d.criteria && d.criteria.length) {
      for (const c of d.criteria) rows.push([...base(d), c.name, c.verdict, c.reasoning, ...tail]);
    } else {
      rows.push([...base(d), "", "", "", ...tail]);
    }
  }
  const esc = (v: string) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  return rows.map(r => r.map(esc).join(",")).join("\n");
}
