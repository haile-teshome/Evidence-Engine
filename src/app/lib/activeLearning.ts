// Active-learning ranker for prioritized screening. As the reviewer labels
// records (include/exclude), a lightweight Multinomial Naive Bayes model over
// the title + abstract text learns from those labels and re-ranks the remaining
// unscreened records by predicted relevance. Before there are enough labels it
// cold-starts from the AI decision. It also estimates recall so the reviewer
// knows when it is safe to stop. Pure and dependency-free.

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
  predictedRemaining: number;           // unlabeled predicted relevant (P >= 0.5)
  estRecall: number | null;             // found / (found + predicted remaining)
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
  const denom = includesFound + predictedRemaining;
  const estRecall = denom > 0 ? includesFound / denom : null;
  return { order, scores, includesFound, reviewed: labeled.length, predictedRemaining, estRecall, trained };
}
