"""Comprehensive metrics for screening benchmark.

Tier 1 (primary): recall, WSS@95, MCC, F1
Tier 2 (secondary): accuracy, precision, specificity, Cohen's kappa
Tier 3 (interrater): pairwise kappa, Fleiss' kappa, Krippendorff's alpha
Tier 4 (stability): cross-repeat consistency
Tier 5 (statistical): McNemar's test, bootstrap CIs, Holm-Bonferroni correction
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from typing import Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Per-cell binary classification metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float          # = sensitivity
    specificity: float
    f1: float
    mcc: float
    kappa: float           # Cohen's kappa vs gold standard
    wss_at_95: float       # 0..1; higher = more work saved at 95% recall

    def as_dict(self) -> dict:
        return self.__dict__


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1: tp += 1
        elif yt == 0 and yp == 1: fp += 1
        elif yt == 0 and yp == 0: tn += 1
        elif yt == 1 and yp == 0: fn += 1
    return tp, fp, tn, fn


def _safe(num: float, den: float) -> float:
    return num / den if den else 0.0


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa for two binary raters / one rater vs gold."""
    n = len(a)
    if not n or len(b) != n:
        return 0.0
    agree = sum(1 for x, y in zip(a, b) if x == y)
    po = agree / n
    p_a1 = sum(a) / n
    p_b1 = sum(b) / n
    pe = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    return (po - pe) / (1 - pe) if (1 - pe) else 0.0


def compute(y_true: List[int], y_pred: List[int], confidence: List[float] | None = None) -> Metrics:
    n = len(y_true)
    tp, fp, tn, fn = confusion(y_true, y_pred)
    precision = _safe(tp, tp + fp)
    recall = _safe(tp, tp + fn)
    specificity = _safe(tn, tn + fp)
    accuracy = _safe(tp + tn, n)
    f1 = _safe(2 * precision * recall, precision + recall)
    den = sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / den if den else 0.0
    kappa = cohens_kappa(y_true, y_pred)

    # WSS@95
    if confidence and len(confidence) == n:
        ranked = sorted(zip(confidence, y_true), key=lambda x: -x[0])
    else:
        ranked = sorted(zip(y_pred, y_true), key=lambda x: -x[0])
    total_pos = sum(y_true)
    if total_pos == 0:
        wss = 0.0
    else:
        target = int(round(0.95 * total_pos))
        found = 0
        reviewed = 0
        for _, yt in ranked:
            reviewed += 1
            if yt == 1: found += 1
            if found >= target: break
        baseline = 0.95 * n
        wss = max(0.0, (baseline - reviewed) / n)

    return Metrics(
        n=n, tp=tp, fp=fp, tn=tn, fn=fn,
        accuracy=accuracy, precision=precision, recall=recall,
        specificity=specificity, f1=f1, mcc=mcc, kappa=kappa, wss_at_95=wss,
    )


# ---------------------------------------------------------------------------
# Interrater reliability (architectures as joint panel)
# ---------------------------------------------------------------------------

def fleiss_kappa(ratings: List[List[int]]) -> float:
    """Fleiss' kappa for N items × R raters (binary 0/1).

    ratings[i] = list of R binary ratings for item i.
    """
    if not ratings:
        return 0.0
    n = len(ratings)
    R = len(ratings[0])
    if R < 2:
        return 0.0
    cat_counts = [[r.count(0), r.count(1)] for r in ratings]
    # Per-item agreement
    p_i = [(c[0] * (c[0] - 1) + c[1] * (c[1] - 1)) / (R * (R - 1)) for c in cat_counts]
    p_bar = sum(p_i) / n
    # Category marginals
    p_0 = sum(c[0] for c in cat_counts) / (n * R)
    p_1 = sum(c[1] for c in cat_counts) / (n * R)
    p_e = p_0 ** 2 + p_1 ** 2
    return (p_bar - p_e) / (1 - p_e) if (1 - p_e) else 0.0


def krippendorff_alpha_binary(ratings: List[List[int | None]]) -> float:
    """Krippendorff's alpha for binary nominal data. Tolerates None for missing."""
    if not ratings:
        return 0.0
    # Coincidence matrix
    coincidences = {(0, 0): 0.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.0}
    n_eff = 0
    for row in ratings:
        present = [r for r in row if r is not None]
        m = len(present)
        if m < 2:
            continue
        n_eff += 1
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                coincidences[(present[i], present[j])] += 1 / (m - 1)
    n_dot = {0: coincidences[(0, 0)] + coincidences[(0, 1)],
             1: coincidences[(1, 0)] + coincidences[(1, 1)]}
    total = sum(n_dot.values())
    if total == 0:
        return 0.0
    do = coincidences[(0, 1)] + coincidences[(1, 0)]
    de = 2 * n_dot[0] * n_dot[1] / (total - 1) if (total - 1) else 0.0
    return 1 - (do / de) if de else 0.0


def pairwise_kappa(rater_labels: Dict[str, List[int]]) -> Dict[Tuple[str, str], float]:
    """Pairwise Cohen's kappa between every pair of raters."""
    out: Dict[Tuple[str, str], float] = {}
    names = sorted(rater_labels.keys())
    for a, b in combinations(names, 2):
        out[(a, b)] = cohens_kappa(rater_labels[a], rater_labels[b])
    return out


# ---------------------------------------------------------------------------
# Stability across repeats
# ---------------------------------------------------------------------------

def stability_score(repeats: List[List[int]]) -> Dict[str, float]:
    """Given K runs of the same arch (each list of n predictions), return:
       - consistency: fraction of papers where ALL repeats agree
       - mean_pairwise_kappa: average Cohen's kappa across all pairs of repeats
    """
    if len(repeats) < 2:
        return {"consistency": 1.0, "mean_pairwise_kappa": 1.0}
    n = len(repeats[0])
    if any(len(r) != n for r in repeats):
        return {"consistency": 0.0, "mean_pairwise_kappa": 0.0}
    consistent = sum(1 for i in range(n) if len(set(r[i] for r in repeats)) == 1)
    kappas = [cohens_kappa(a, b) for a, b in combinations(repeats, 2)]
    return {
        "consistency": consistent / n,
        "mean_pairwise_kappa": sum(kappas) / len(kappas) if kappas else 1.0,
    }


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def mcnemar_test(a: Sequence[int], b: Sequence[int], y_true: Sequence[int]) -> Tuple[float, float]:
    """McNemar's paired test on agreement-with-gold. Returns (chi2, p_value).

    b00: both wrong
    b01: a wrong, b correct
    b10: a correct, b wrong
    b11: both correct
    Tests H0: b01 == b10 (i.e., no difference in error rate).
    """
    b01 = b10 = 0
    for yt, ya, yb in zip(y_true, a, b):
        a_correct = (ya == yt)
        b_correct = (yb == yt)
        if not a_correct and b_correct: b01 += 1
        elif a_correct and not b_correct: b10 += 1
    if b01 + b10 == 0:
        return 0.0, 1.0
    # Continuity-corrected McNemar chi-squared, 1 df
    chi2 = ((abs(b01 - b10) - 1) ** 2) / (b01 + b10)
    # Approximate p-value via chi-squared 1df survival
    p = math.exp(-chi2 / 2) / math.sqrt(2 * math.pi) if chi2 < 0 else _chi2_sf_1df(chi2)
    return chi2, p


def _chi2_sf_1df(x: float) -> float:
    """Survival function (1 - CDF) of chi-squared with 1 d.f.
    For chi2(1), 1 - CDF(x) = erfc(sqrt(x/2)).
    """
    return math.erfc(math.sqrt(max(0.0, x) / 2))


def holm_bonferroni(pvals: Dict[Tuple[str, str], float], alpha: float = 0.05) -> Dict[Tuple[str, str], bool]:
    """Holm-Bonferroni step-down. Returns dict mapping pair -> reject_H0."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[Tuple[str, str], bool] = {}
    rejected_all_below = True
    for i, (pair, p) in enumerate(items):
        thresh = alpha / (m - i)
        if rejected_all_below and p < thresh:
            out[pair] = True
        else:
            out[pair] = False
            rejected_all_below = False
    return out


def bootstrap_ci(
    y_true: Sequence[int], y_pred: Sequence[int],
    metric_fn=None, n_boot: int = 1000, seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap 95% CI for a metric. Returns (point, ci_low, ci_high).
    metric_fn signature: (y_true, y_pred) -> float. Defaults to F1.
    """
    if metric_fn is None:
        def metric_fn(yt, yp):
            return compute(list(yt), list(yp)).f1
    rng = random.Random(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        stats.append(metric_fn(yt, yp))
    stats.sort()
    low = stats[int(0.025 * n_boot)]
    high = stats[int(0.975 * n_boot)]
    return point, low, high


# ---------------------------------------------------------------------------
# Stratified analysis helpers
# ---------------------------------------------------------------------------

def metrics_by_stratum(
    y_true: Sequence[int], y_pred: Sequence[int], strata: Sequence[str],
) -> Dict[str, Metrics]:
    """Compute metrics separately for each value of `strata` (parallel list)."""
    buckets: Dict[str, Tuple[List[int], List[int]]] = {}
    for yt, yp, s in zip(y_true, y_pred, strata):
        if s not in buckets:
            buckets[s] = ([], [])
        buckets[s][0].append(yt)
        buckets[s][1].append(yp)
    return {s: compute(yt, yp) for s, (yt, yp) in buckets.items() if len(yt) >= 5}
