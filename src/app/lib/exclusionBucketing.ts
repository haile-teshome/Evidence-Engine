// Shared exclusion-reason bucketing. Used by both the screening pages (to
// store a fresh count when the screen runs) and the PRISMA page (to re-derive
// the breakdown live, so stale long-sentence labels from older runs don't
// linger in the diagram).
//
// Label style: short reviewer phrases that match how a methods section would
// describe the exclusion ("Wrong study population", "Wrong intervention", …).

import type { ScreenResult, FullTextResult } from "./mockServices";

// ---- effective-decision helpers ------------------------------------------
//
// "Effective" decision = reviewer override if present, else the AI's call.
// All UI and aggregation code should go through these so a manual "Keep this
// paper" toggle ripples consistently through stats, PRISMA, exports, and
// downstream stages.

export type AbstractDecision = "INCLUDE" | "EXCLUDE";
export type FullTextDecision = "Include" | "Exclude";

export function effectiveAbstractDecision(
  r: ScreenResult,
  overrides: Record<string, AbstractDecision>,
): AbstractDecision {
  return overrides[r.paper_id] ?? r.Decision;
}

export function effectiveFullTextDecision(
  r: FullTextResult,
  overrides: Record<string, FullTextDecision>,
): FullTextDecision {
  return overrides[r.paper_id] ?? r.Decision;
}

export function categoriseAbstractExclusion(r: ScreenResult): string {
  const pa = r.Pico_Assessment;
  if (!pa) return "Other reason";

  const votes = {
    Population:   pa.population.vote,
    Intervention: pa.intervention.vote,
    Comparator:   pa.comparator.vote,
    Outcome:      pa.outcome.vote,
  };

  const FAIL_LABEL: Record<string, string> = {
    Population:   "Wrong study population",
    Intervention: "Wrong intervention",
    Comparator:   "Wrong comparator",
    Outcome:      "Wrong outcome",
  };

  const failed = Object.entries(votes).filter(([, v]) => v === "FAIL").map(([k]) => k);
  if (failed.length === 1) return FAIL_LABEL[failed[0]] || "Other reason";
  if (failed.length >= 2)  return "Multiple PICO mismatches";

  const values = Object.values(votes);
  const naCount      = values.filter(v => v === "NA").length;
  const partialCount = values.filter(v => v === "PARTIAL").length;
  const passCount    = values.filter(v => v === "PASS").length;

  if (naCount >= 3)      return "Insufficient abstract detail";
  if (passCount === 0)   return "No PICO match";
  if (partialCount >= 3) return "Partial match only";
  return "Other reason";
}

export function categoriseFullTextExclusion(r: FullTextResult): string {
  const pe = r.picoEvidence;
  const FAIL_LABEL: Record<string, string> = {
    population:   "Wrong study population",
    intervention: "Wrong intervention",
    comparator:   "Wrong comparator",
    outcome:      "Wrong outcome",
  };

  if (pe) {
    const noMatches: string[] = [];
    if (pe.population?.match === "no")   noMatches.push("population");
    if (pe.intervention?.match === "no") noMatches.push("intervention");
    if (pe.comparator?.match === "no")   noMatches.push("comparator");
    if (pe.outcome?.match === "no")      noMatches.push("outcome");
    if (noMatches.length === 1) return FAIL_LABEL[noMatches[0]] || "Other reason";
    if (noMatches.length >= 2)  return "Multiple PICO mismatches";
  }

  if ((r.exclusion_violations ?? 0) > 0) return "Exclusion criterion met";
  if ((r.inclusion_score ?? 0) === 0)    return "Inclusion criteria not met";
  return "Other reason";
}

/** Aggregate a list of screening results into a category → count map.
 *
 * Honours reviewer overrides: a paper with `Decision === "EXCLUDE"` whose
 * reviewer override is `"INCLUDE"` is treated as included and NOT counted in
 * any exclusion bucket. A bucket labelled `"Reviewer kept"` summarises how
 * many AI-excluded papers were rescued by a reviewer.
 */
export function bucketAbstractExclusions(
  results: ScreenResult[] | null | undefined,
  overrides: Record<string, AbstractDecision> = {},
): Record<string, number> {
  const out: Record<string, number> = {};
  if (!results) return out;
  for (const r of results) {
    const eff = effectiveAbstractDecision(r, overrides);
    if (eff !== "EXCLUDE") continue;
    const k = categoriseAbstractExclusion(r);
    out[k] = (out[k] || 0) + 1;
  }
  return out;
}

export function bucketFullTextExclusions(
  results: FullTextResult[] | null | undefined,
  overrides: Record<string, FullTextDecision> = {},
): Record<string, number> {
  const out: Record<string, number> = {};
  if (!results) return out;
  for (const r of results) {
    const eff = effectiveFullTextDecision(r, overrides);
    if (eff !== "Exclude") continue;
    const k = categoriseFullTextExclusion(r);
    out[k] = (out[k] || 0) + 1;
  }
  return out;
}
