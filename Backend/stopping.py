"""When is it safe to stop screening? Two answers, computed from the labelled history alone.

Neither rule looks at model scores. Both need only the sequence of reviewer decisions in the order they
were made and the size of the corpus, so they are exact functions of what the reviewer has done, not
estimates from a classifier that has seen a handful of labels.

The rule this module replaces counted unlabelled records whose summed score reached 1.0 and called the
reviewer done when the count hit zero. Simulated on the 26 SYNERGY reviews with the shipped ranker, it
fired at a median of 2% of records screened with median true recall 0.30 (only 2 of 26 reviews were at
95% recall when it fired). The two rules below were evaluated on the same simulation
(screening-benchmark/cluster/stop_rule_eval.py, log backend_ranker_rounds_30241.csv):

    likely_complete   no new include in the last max(10% of records, 200 records), after 20% screened:
                      fired on 24 of 26 reviews with recall >= 0.95 on 24, mean work saved 0.51
                      (an oracle stopping exactly at 95% recall saves 0.73). A heuristic, so it is a
                      prompt to review, not a verdict.
    safe_to_stop      the hypergeometric stopping test of Callaghan and Mueller-Hansen (Systematic Reviews
                      2020, 9:273) at alpha 0.05 over every window ending at the current record: recall
                      >= 0.95 on 26 of 26 reviews (minimum 0.99), mean work saved 0.22. Conservative,
                      and the only one of the two that can carry the word "safe".
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

WINDOW_FRAC = 0.10       # likely_complete: the tail that must contain no include
MIN_WINDOW = 200         # ... and never shorter than this many records
MIN_SCREENED_FRAC = 0.20 # ... and never before this fraction of the corpus is screened
TARGET_RECALL = 0.95
ALPHA = 0.05


def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_cdf(k: int, M: int, K: int, n: int) -> float:
    """P[X <= k] for X ~ Hypergeometric(population M, successes K, draws n), without scipy."""
    if n <= 0 or K <= 0:
        return 1.0
    lo, hi = max(0, n - (M - K)), min(K, n)
    if k >= hi:
        return 1.0
    if k < lo:
        return 0.0
    denom = _log_choose(M, n)
    total = 0.0
    for x in range(lo, k + 1):
        total += math.exp(_log_choose(K, x) + _log_choose(M - K, n - x) - denom)
    return min(1.0, total)


def order_history(labels: Dict[str, int], history: Optional[Sequence[str]]) -> List[int]:
    """Decisions in screening order (1 include, 0 exclude). `history` lists record ids in the order they
    were decided; labelled ids missing from it are placed first, since their order is unknown and the
    conservative reading is that they were screened earliest."""
    seen, seq = set(), []
    for rid in history or []:
        if rid in labels and rid not in seen:
            seen.add(rid); seq.append(rid)
    early = [rid for rid in labels if rid not in seen]
    return [1 if int(labels[r]) == 1 else 0 for r in early + seq]


def since_last_include(decisions: Sequence[int]) -> int:
    n = 0
    for d in reversed(decisions):
        if d == 1:
            break
        n += 1
    return n


def likely_complete(decisions: Sequence[int], n_total: int,
                    window_frac: float = WINDOW_FRAC, min_window: int = MIN_WINDOW,
                    min_screened_frac: float = MIN_SCREENED_FRAC) -> Dict[str, object]:
    """Heuristic: enough screened, at least one include found, and the tail of the history holds none."""
    screened = len(decisions)
    window = max(min_window, int(math.ceil(window_frac * n_total)))
    window = min(window, max(1, n_total))
    tail = since_last_include(decisions)
    found = sum(1 for d in decisions if d == 1)
    ok = (n_total > 0 and screened >= min_screened_frac * n_total and found > 0 and tail >= window)
    return {"likely_complete": bool(ok), "since_last_include": tail, "window": window,
            "min_screened": int(math.ceil(min_screened_frac * n_total))}


def safe_to_stop(decisions: Sequence[int], n_total: int,
                 target: float = TARGET_RECALL, alpha: float = ALPHA) -> Dict[str, object]:
    """Hypergeometric stopping test (Callaghan and Mueller-Hansen 2020).

    Null hypothesis: recall is still below `target`, i.e. the corpus holds at least R0 = floor(r/target)+1
    includes, r being the number found. For a window covering the last k screened records, K0 = R0 minus
    the includes found before the window were still unscreened when the window began, in a pool of M
    records; the p-value is the probability of finding at most the observed number of includes in k
    draws from that pool. Screening in ranked order makes the test conservative. Windows that start right
    after an include (or at the very beginning) dominate all others, so only those are evaluated.
    """
    screened = len(decisions)
    found = sum(1 for d in decisions if d == 1)
    if n_total <= 0 or screened == 0 or found == 0:
        return {"safe_to_stop": False, "p_value": None, "found": found}
    if screened >= n_total:
        return {"safe_to_stop": True, "p_value": 0.0, "found": found}
    R0 = int(math.floor(found / target)) + 1
    starts = [0] + [i + 1 for i, d in enumerate(decisions) if d == 1 and i + 1 < screened]
    best = 1.0
    cum = 0
    found_before = {}
    for i, d in enumerate(decisions):
        found_before[i] = cum
        cum += 1 if d == 1 else 0
    for s in starts:
        k = screened - s
        r_before = found_before[s]
        K0 = R0 - r_before
        M = n_total - s
        if K0 <= 0 or K0 > M or k <= 0:
            continue
        p = hypergeom_cdf(found - r_before, M, K0, k)
        best = min(best, p)
    return {"safe_to_stop": bool(best < alpha), "p_value": best, "found": found}


def stop_indicators(labels: Dict[str, int], history: Optional[Sequence[str]], n_total: int) -> Dict[str, object]:
    decisions = order_history(labels, history)
    lc = likely_complete(decisions, n_total)
    st = safe_to_stop(decisions, n_total)
    if st["safe_to_stop"]:
        detail = (f"no evidence of missed includes: at most {math.floor(st['found'] / TARGET_RECALL) + 1 - st['found']} "
                  f"could remain for recall to be below {int(TARGET_RECALL * 100)}% (p = {st['p_value']:.3f})")
    elif lc["likely_complete"]:
        detail = (f"no include in the last {lc['since_last_include']} records; most reviews are complete here, "
                  f"but small reviews can still hold late includes")
    elif len(decisions) < lc["min_screened"]:
        detail = f"stop indicators start after {lc['min_screened']} records"
    else:
        detail = f"{lc['since_last_include']} records since the last include (window {lc['window']})"
    return {"since_last_include": lc["since_last_include"], "likely_complete": lc["likely_complete"],
            "safe_to_stop": st["safe_to_stop"], "stop_p_value": st["p_value"], "stop_detail": detail}
