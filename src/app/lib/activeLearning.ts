// Active-learning ranker for prioritized screening. As the reviewer labels
// records (include/exclude), a model learns from those labels and re-ranks the
// remaining unscreened records by predicted relevance. Before there are enough
// labels it cold-starts from the AI decision. The backend also returns two stop
// indicators computed from the decision history (see Backend/stopping.py): a
// heuristic "likely complete" and a hypergeometric "safe to stop".
//
// Two rankers, and which one ran is always reported:
//
//   bge+tfidf  the benchmarked configuration (WSS@95 0.679 over the 26 SYNERGY
//              reviews). It runs in the backend because a BGE-large embedding
//              model cannot run in a browser tab. See Backend/ranking.py.
//   nb         the Multinomial Naive Bayes below. Pure, dependency-free, and
//              the only option when the backend is unreachable. It has NOT been
//              benchmarked, so callers must not present it as though it had.
//
// The distinction is load-bearing rather than pedantic: the published figure
// describes bge+tfidf, and for a while the app shipped only the Naive Bayes
// while the manuscript described the ensemble.

const STOP = new Set(
  ("a an and are as at be by for from has have in into is it its of on or that the to was were with this these those we our study studies " +
   "results result method methods using used based which be been also can may our their they than then").split(" "),
);

export function tokenize(text: string): string[] {
  return (String(text || "").toLowerCase().match(/[a-z0-9]+/g) || [])
    .filter(t => t.length > 2 && !STOP.has(t));
}

type NBModel = { logPrior: [number, number]; logLik: Map<string, [number, number]>; classTok: [number, number]; V: number };

function trainNB(labeled: { tokens: string[]; label: 0 | 1 }[]): NBModel {
  const counts = new Map<string, [number, number]>();
  const classTok: [number, number] = [0, 0];
  const classDoc: [number, number] = [0, 0];
  for (const d of labeled) {
    classDoc[d.label]++;
    for (const t of d.tokens) {
      const c = counts.get(t) || [0, 0];
      c[d.label]++; counts.set(t, c);
      classTok[d.label]++;
    }
  }
  const n = labeled.length;
  const V = counts.size || 1;
  const logLik = new Map<string, [number, number]>();
  for (const [t, c] of counts) {
    logLik.set(t, [
      Math.log((c[0] + 1) / (classTok[0] + V)),
      Math.log((c[1] + 1) / (classTok[1] + V)),
    ]);
  }
  return { logPrior: [Math.log((classDoc[0] + 1) / (n + 2)), Math.log((classDoc[1] + 1) / (n + 2))], logLik, classTok, V };
}

function scoreNB(m: NBModel, tokens: string[]): number {
  let s0 = m.logPrior[0], s1 = m.logPrior[1];
  const miss0 = Math.log(1 / (m.classTok[0] + m.V)), miss1 = Math.log(1 / (m.classTok[1] + m.V));
  for (const t of tokens) {
    const ll = m.logLik.get(t);
    if (ll) { s0 += ll[0]; s1 += ll[1]; } else { s0 += miss0; s1 += miss1; }
  }
  const mx = Math.max(s0, s1);
  const e0 = Math.exp(s0 - mx), e1 = Math.exp(s1 - mx);
  return e1 / (e0 + e1);
}

export type ALItem = { id: string; title?: string; text?: string; aiInclude: boolean; override?: "include" | "exclude" };

export type ALResult = {
  order: string[];                      // unlabeled ids, most-relevant first
  scores: Record<string, number>;       // predicted P(relevant) for unlabeled
  includesFound: number;                // labeled includes so far
  reviewed: number;                     // total labeled
  predictedRemaining: number;           // unlabeled predicted relevant (P >= 0.5); a count, not a recall estimate
  sinceLastInclude: number;             // decisions since the last include (backend only)
  likelyComplete: boolean;              // heuristic stop prompt (backend only)
  safeToStop: boolean;                  // hypergeometric stopping test at 95% recall (backend only)
  stopPValue: number | null;
  stopDetail: string;
  trained: boolean;                     // model trained vs cold-start
};

const MIN_TO_TRAIN = 4;

export function activeRank(items: ALItem[]): ALResult {
  const docs = items.map(it => ({
    id: it.id,
    tokens: tokenize(`${it.title || ""} ${it.text || ""}`),
    aiInclude: it.aiInclude,
    label: it.override ? (it.override === "include" ? 1 : 0) as 0 | 1 : undefined,
  }));
  const labeled = docs.filter(d => d.label !== undefined) as { tokens: string[]; label: 0 | 1 }[];
  const unlabeled = docs.filter(d => d.label === undefined);
  const hasBoth = labeled.some(d => d.label === 1) && labeled.some(d => d.label === 0);
  const trained = hasBoth && labeled.length >= MIN_TO_TRAIN;

  const scores: Record<string, number> = {};
  if (trained) {
    const m = trainNB(labeled);
    for (const d of unlabeled) scores[d.id] = scoreNB(m, d.tokens);
  } else {
    // Cold start: trust the AI decision as the prior ordering.
    for (const d of unlabeled) scores[d.id] = d.aiInclude ? 0.85 : 0.15;
  }

  const order = unlabeled.map(d => d.id).sort((a, b) => scores[b] - scores[a]);
  const predictedRemaining = unlabeled.filter(d => scores[d.id] >= 0.5).length;
  const includesFound = docs.filter(d => d.label === 1).length;
  // The in-browser ranker never claims a stopping point: the indicators need the
  // decision history and are computed by the backend (Backend/stopping.py).
  return {
    order, scores, includesFound, reviewed: labeled.length, predictedRemaining,
    sinceLastInclude: 0, likelyComplete: false, safeToStop: false, stopPValue: null,
    stopDetail: "stop indicators need the backend ranker", trained,
  };
}

// ---------------------------------------------------------------------------
// The benchmarked path.
// ---------------------------------------------------------------------------

import { rankRemote, type RankTier } from "./apiClient";

export type RankedState = ALResult & {
  tier: RankTier;   // which ranker produced `order`
  detail: string;   // human-readable qualifier for the UI
};

/**
 * Rank with the benchmarked backend ranker, falling back to the in-browser
 * Naive Bayes only when the backend cannot answer.
 *
 * The fallback is deliberately NOT silent: `tier` comes back as "nb" and the
 * caller is expected to say so, because the ordering's validation depends
 * entirely on which model produced it.
 *
 * `history` is the order in which the reviewer decided records (latest last);
 * the backend's stop indicators are functions of that sequence.
 */
export async function rankRecords(items: ALItem[], signal?: AbortSignal, history: string[] = []): Promise<RankedState> {
  const labels: Record<string, 0 | 1> = {};
  for (const it of items) {
    if (it.override) labels[it.id] = it.override === "include" ? 1 : 0;
  }

  try {
    const r = await rankRemote(
      items.map(it => ({ id: it.id, title: it.title, text: it.text })),
      labels,
      "auto",
      signal,
      history,
    );
    // Too few labels to fit anything. The local cold start orders by the AI
    // decision, which beats the arbitrary order the backend returns here.
    if (r.tier === "cold" || !r.trained) {
      return { ...activeRank(items), tier: "cold", detail: r.detail };
    }
    return {
      order: r.order,
      scores: r.scores,
      includesFound: r.includes_found,
      reviewed: r.reviewed,
      predictedRemaining: r.predicted_remaining,
      sinceLastInclude: r.since_last_include,
      likelyComplete: r.likely_complete,
      safeToStop: r.safe_to_stop,
      stopPValue: r.stop_p_value,
      stopDetail: r.stop_detail,
      trained: r.trained,
      tier: r.tier,
      detail: r.detail,
    };
  } catch (err) {
    if ((err as any)?.name === "AbortError") throw err;
    return {
      ...activeRank(items),
      tier: "nb",
      detail: "backend unavailable; ranked in-browser with the unbenchmarked model",
    };
  }
}
