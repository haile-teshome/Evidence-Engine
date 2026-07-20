// Shared exclusion-reason bucketing. Used by both the screening pages (to
// store a fresh count when the screen runs) and the PRISMA page (to re-derive
// the breakdown live, so stale long-sentence labels from older runs don't
// linger in the diagram).
//
// Bucketing priority:
//   1. If the screening's per-criterion trace flagged at least one of the
//      user's *named* inclusion/exclusion criteria, that criterion's text
//      becomes the bucket label. This is what the user actually wants to see:
//      which of their specific criteria are doing the work.
//   2. If multiple criteria flagged, the paper is bucketed as "Multiple
//      criteria failed" so the sum-of-buckets still equals the excluded count.
//   3. Otherwise we fall back to the PICO-based generic buckets.

import type { ScreenResult, FullTextResult } from "./mockServices";

// ---- effective-decision helpers ------------------------------------------

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

// ---- helpers --------------------------------------------------------------

/** Trim long criterion text for an aside label so it stays readable. */
function shorten(s: string, max = 70): string {
  const t = s.trim();
  return t.length <= max ? t : t.slice(0, max - 1) + "…";
}

/** Walk a screening result's Agent_Trace and return the user-defined criteria
 *  that voted FAIL, separated by list. Criteria not found in either list are
 *  ignored (they're typically PICO element names left over from the legacy
 *  screener). */
function failedNamedCriteria(
  agentTrace: Record<string, { vote: "PASS" | "FAIL" | "N/A"; reasoning?: string; evidence?: string }> | undefined,
  inclusion: string[],
  exclusion: string[],
): { inc: string[]; exc: string[] } {
  const inc: string[] = [];
  const exc: string[] = [];
  if (!agentTrace) return { inc, exc };
  const incSet = new Set(inclusion.map(c => c.trim()).filter(Boolean));
  const excSet = new Set(exclusion.map(c => c.trim()).filter(Boolean));
  for (const [crit, info] of Object.entries(agentTrace)) {
    if (!info || info.vote !== "FAIL") continue;
    const key = crit.trim();
    if (incSet.has(key)) inc.push(key);
    else if (excSet.has(key)) exc.push(key);
  }
  return { inc, exc };
}

// ---- abstract -------------------------------------------------------------

export function categoriseAbstractExclusion(
  r: ScreenResult,
  inclusion: string[] = [],
  exclusion: string[] = [],
): string {
  // 1. User-defined criteria first.
  const { inc, exc } = failedNamedCriteria(r.Agent_Trace, inclusion, exclusion);
  const total = inc.length + exc.length;
  if (total >= 1) {
    // One brief primary reason per study (matched exclusion is decisive, else
    // the first unmet inclusion), kept short so reasons aggregate.
    const primary = exc[0] ?? inc[0];
    return shorten(primary, 80);
  }

  // 2. Fall back to PICO bucket.
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

  // One reason per study: the highest-priority failed element in PICO order
  // (population > intervention > comparator > outcome), since `votes` is built
  // in that order. Keeps the PRISMA box to a single clear reason per line.
  const failed = Object.entries(votes).filter(([, v]) => v === "FAIL").map(([k]) => k);
  if (failed.length >= 1) return FAIL_LABEL[failed[0]] || "Other reason";

  const values = Object.values(votes);
  const naCount      = values.filter(v => v === "NA").length;
  const partialCount = values.filter(v => v === "PARTIAL").length;
  const passCount    = values.filter(v => v === "PASS").length;

  if (naCount >= 3)      return "Insufficient abstract detail";
  if (passCount === 0)   return "No PICO match";
  if (partialCount >= 3) return "Partial match only";
  return "Other reason";
}

// ---- full text ------------------------------------------------------------

/** Full-text screening doesn't write Agent_Trace; it writes criteriaEval with
 *  INCLUDE/EXCLUDE strings per criterion. Mirror the same prioritised logic:
 *  user-named criteria first, then PICO match status. */
export function categoriseFullTextExclusion(
  r: FullTextResult,
  inclusion: string[] = [],
  exclusion: string[] = [],
): string {
  // 1. User-defined criteria first. For inclusion criteria, EXCLUDE in
  //    criteriaEval means the paper didn't meet the criterion. For exclusion
  //    criteria, EXCLUDE means the paper violated the criterion (per the
  //    backend's screen_full_text counting in `exclusion_violations`).
  const incFailed: string[] = [];
  const excFailed: string[] = [];
  for (const c of inclusion) {
    const v = r.criteriaEval?.[c];
    if (v === "EXCLUDE") incFailed.push(c);
  }
  for (const c of exclusion) {
    const v = r.criteriaEval?.[c];
    if (v === "EXCLUDE") excFailed.push(c);
  }
  const total = incFailed.length + excFailed.length;
  if (total >= 1) {
    // One brief primary reason per study, so the PRISMA box stays short and
    // studies failing the same criterion aggregate. A matched exclusion
    // criterion is decisive; otherwise the first unmet inclusion criterion.
    // (Every label is inline-editable in the diagram.)
    const primary = excFailed[0] ?? incFailed[0];
    return shorten(primary, 80);
  }

  // 2. Fall back to PICO mismatch.
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
    // One reason per study: the highest-priority mismatch in PICO order
    // (population > intervention > comparator > outcome), since `noMatches` is
    // built in that order.
    if (noMatches.length >= 1) return FAIL_LABEL[noMatches[0]] || "Other reason";
  }

  if ((r.exclusion_violations ?? 0) > 0) return "Exclusion criterion met";
  if ((r.inclusion_score ?? 0) === 0)    return "Inclusion criteria not met";
  return "Other reason";
}

// ---- aggregation ----------------------------------------------------------

/** Same as bucketAbstractExclusions but groups the ScreenResult objects instead
 *  of counting them — so callers can show per-paper breakdowns in the PRISMA UI. */
export function bucketAbstractExclusionsByPaper(
  results: ScreenResult[] | null | undefined,
  overrides: Record<string, AbstractDecision> = {},
  inclusion: string[] = [],
  exclusion: string[] = [],
): Record<string, ScreenResult[]> {
  const out: Record<string, ScreenResult[]> = {};
  if (!results) return out;
  for (const r of results) {
    const eff = effectiveAbstractDecision(r, overrides);
    if (eff !== "EXCLUDE") continue;
    const k = categoriseAbstractExclusion(r, inclusion, exclusion);
    if (!out[k]) out[k] = [];
    out[k].push(r);
  }
  return out;
}

/** Same as bucketFullTextExclusions but groups FullTextResult objects. */
export function bucketFullTextExclusionsByPaper(
  results: FullTextResult[] | null | undefined,
  overrides: Record<string, FullTextDecision> = {},
  inclusion: string[] = [],
  exclusion: string[] = [],
): Record<string, FullTextResult[]> {
  const out: Record<string, FullTextResult[]> = {};
  if (!results) return out;
  for (const r of results) {
    const eff = effectiveFullTextDecision(r, overrides);
    if (eff !== "Exclude") continue;
    const k = categoriseFullTextExclusion(r, inclusion, exclusion);
    if (!out[k]) out[k] = [];
    out[k].push(r);
  }
  return out;
}

export function bucketAbstractExclusions(
  results: ScreenResult[] | null | undefined,
  overrides: Record<string, AbstractDecision> = {},
  inclusion: string[] = [],
  exclusion: string[] = [],
): Record<string, number> {
  const out: Record<string, number> = {};
  if (!results) return out;
  for (const r of results) {
    const eff = effectiveAbstractDecision(r, overrides);
    if (eff !== "EXCLUDE") continue;
    const k = categoriseAbstractExclusion(r, inclusion, exclusion);
    out[k] = (out[k] || 0) + 1;
  }
  return out;
}

export function bucketFullTextExclusions(
  results: FullTextResult[] | null | undefined,
  overrides: Record<string, FullTextDecision> = {},
  inclusion: string[] = [],
  exclusion: string[] = [],
): Record<string, number> {
  const out: Record<string, number> = {};
  if (!results) return out;
  for (const r of results) {
    const eff = effectiveFullTextDecision(r, overrides);
    if (eff !== "Exclude") continue;
    const k = categoriseFullTextExclusion(r, inclusion, exclusion);
    out[k] = (out[k] || 0) + 1;
  }
  return out;
}
