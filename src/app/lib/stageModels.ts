// Which model runs at which stage.
//
// One model for the whole review was the wrong unit of choice, because the
// stages do not ask for the same thing. Screening wants the model fine-tuned on
// screening verdicts, and on the benchmark that is LEADS-Mistral; asked to fill
// an extraction form the same model returns no usable JSON at all. Extraction
// wants a general model that will follow a schema, and measurably that is
// qwen2.5:7b. A reviewer who picks one model is therefore accepting a
// compromise at every stage but one.
//
// So the model is chosen per stage, over a default. `StageModels` holds only
// the stages the reviewer has decided about; every other stage reads the
// default, which keeps a fresh session to a single decision and keeps the
// existing behaviour for anyone who never opens the per-stage list.

/**
 * A stage that runs a model.
 *
 * Full-Text Acquisition, Diagramming and Projects are missing because they run
 * no model: acquisition fetches files, the diagram is drawn from counts. A
 * stage with no model call would be a picker that changes nothing.
 */
export type ModelStage =
  | "home"
  | "planning"
  | "abstract-screening"
  | "full-text-screening"
  | "citation-snowball"
  | "table-extraction"
  | "text-extraction"
  | "quality-assessment"
  | "meta-analysis"
  | "writing";

/** The reviewer's per-stage choices. Absent key means "use the default". */
export type StageModels = Partial<Record<ModelStage, string>>;

/**
 * The stages, in the order the sidebar lists them, with what each one asks of a
 * model. `what` is shown under the picker: "Text Extraction" does not tell a
 * reviewer why they might want a different model there, and "fills your
 * extraction form field by field" does.
 */
export const MODEL_STAGES: { id: ModelStage; label: string; what: string }[] = [
  { id: "home",               label: "Home",                 what: "chat, PICO, search strings, summaries" },
  { id: "planning",           label: "Planning",             what: "per-database query adaptation" },
  { id: "abstract-screening", label: "Abstract Screening",   what: "include/exclude verdicts on abstracts" },
  { id: "full-text-screening", label: "Full-Text Screening", what: "eligibility against the full text" },
  { id: "citation-snowball",  label: "Citation Snowball",    what: "ranking and screening cited work" },
  { id: "table-extraction",   label: "Table Extraction",     what: "finding and naming results tables" },
  { id: "text-extraction",    label: "Text Extraction",      what: "filling your extraction form" },
  { id: "quality-assessment", label: "Quality Assessment",   what: "risk-of-bias judgements" },
  { id: "meta-analysis",      label: "Meta-analysis",        what: "pulling effect sizes to pool" },
  { id: "writing",            label: "Writing Assistant",    what: "protocol and manuscript drafting" },
];

const STAGE_IDS = new Set<string>(MODEL_STAGES.map(s => s.id));

/** Is this a stage whose model is asked for a screening verdict? */
export function isScreeningStage(stage: ModelStage): boolean {
  return stage === "abstract-screening"
      || stage === "full-text-screening"
      || stage === "citation-snowball";
}

/**
 * The model that should run `stage`.
 *
 * An override wins; otherwise the default. An empty or whitespace override is
 * treated as absent, because that is what a half-finished edit leaves behind
 * and sending "" as a model name fails the request rather than falling back.
 */
export function modelForStage(
  stage: ModelStage,
  overrides: StageModels | null | undefined,
  fallback: string,
): string {
  const pick = overrides?.[stage];
  return typeof pick === "string" && pick.trim() ? pick : fallback;
}

/**
 * Drop anything that is not a known stage or not a non-empty string.
 *
 * Stage choices are persisted and restored from a saved session, which may have
 * been written by an older build that named a stage differently. A stale key
 * would otherwise sit in the map for ever, invisible in the UI and impossible
 * to clear.
 */
export function cleanStageModels(raw: unknown): StageModels {
  if (!raw || typeof raw !== "object") return {};
  const out: StageModels = {};
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    if (STAGE_IDS.has(key) && typeof value === "string" && value.trim()) {
      out[key as ModelStage] = value;
    }
  }
  return out;
}
