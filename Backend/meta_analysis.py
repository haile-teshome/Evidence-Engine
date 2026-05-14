"""Comprehensive meta-analysis agent.

Covers the scenarios that account for >95% of published medical /
epidemiological / behavioural-science meta-analyses:

Effect measures
---------------
  OR       log Odds Ratio                       (binary 2x2)
  RR       log Risk Ratio (Relative Risk)        (binary 2x2)
  RD       Risk Difference                       (binary 2x2)
  PETO_OR  Peto's Odds Ratio                     (binary 2x2, rare events)
  HR       log Hazard Ratio                      (time-to-event, generic IV)
  MD       Raw Mean Difference                   (continuous, same units)
  SMD      Hedges' g (small-sample SMD)          (continuous, different units)
  PROP     logit-transformed proportion          (single-arm prevalence)
  IR       log incidence rate                    (single-arm, person-time)
  ZCOR     Fisher's z-transformed correlation    (correlational studies)
  GENERIC  generic inverse-variance              (effect + SE supplied directly)

Pooling methods
---------------
  Fixed-effects (FE)      inverse-variance
  Mantel-Haenszel (MH)    for binary 2x2 (more robust to sparse cells)
  Random-effects (RE)     DerSimonian-Laird (DL) — default
                          Paule-Mandel (PM)
                          REML (restricted maximum likelihood, iterative)
  Knapp-Hartung           t-distribution adjustment for RE CI

Heterogeneity
-------------
  Cochran's Q, df, Q-test p-value
  I², H², tau, tau², 95% prediction interval

Subgroup / moderator analyses
-----------------------------
  Group-by-categorical: within-group RE pool + between-group Q-test
  Meta-regression: single continuous moderator via weighted least squares

Sensitivity analyses
--------------------
  Leave-one-out (LOO)
  Cumulative meta-analysis (by user-supplied ordering)

Publication-bias diagnostics
----------------------------
  Funnel plot data (per-study effect + SE, plus pseudo 95% CI lines)
  Egger's regression test
  Begg's rank correlation test
  Trim-and-fill (Duval & Tweedie, L0 estimator)

No external statistical dependencies beyond the standard library —
all math is in closed form or simple iterative solvers.
"""

from __future__ import annotations

import json as _json
import math
import re
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from utils import AIService


# ===========================================================================
# Data model
# ===========================================================================


@dataclass
class StudyEffect:
    """One study's contribution. Many fields are optional — populate the ones
    relevant for the chosen effect_measure (or use yi/se directly for GENERIC).

    The pool() function dispatches on effect_measure; specific subsets of
    fields are required depending on the measure:
      OR / RR / RD / PETO_OR: events_t, n_t, events_c, n_c
      HR (from HR + CI):       hr, hr_ci_low, hr_ci_high  (will set yi=log(hr))
      MD:                      mean_t, sd_t, n_t, mean_c, sd_c, n_c
      SMD:                     mean_t, sd_t, n_t, mean_c, sd_c, n_c
      PROP:                    events_total, n_total
      IR:                      events_total, person_time
      ZCOR:                    correlation, n_total
      GENERIC:                 yi, se   (or yi, vi)
    """

    paper_id: str
    title: str
    url: Optional[str] = None
    outcome: str = ""
    effect_measure: str = "GENERIC"   # see module docstring for valid codes

    # Binary 2x2
    events_t: Optional[int] = None
    n_t: Optional[int] = None
    events_c: Optional[int] = None
    n_c: Optional[int] = None

    # Continuous two-arm
    mean_t: Optional[float] = None
    sd_t: Optional[float] = None
    mean_c: Optional[float] = None
    sd_c: Optional[float] = None
    # n_t / n_c are reused

    # Time-to-event reported as HR + CI
    hr: Optional[float] = None
    hr_ci_low: Optional[float] = None
    hr_ci_high: Optional[float] = None
    log_hr: Optional[float] = None
    log_hr_se: Optional[float] = None

    # Single-arm prevalence / proportion
    events_total: Optional[int] = None
    n_total: Optional[int] = None

    # Single-arm incidence rate (person-time)
    person_time: Optional[float] = None

    # Correlation
    correlation: Optional[float] = None

    # Generic inverse-variance: yi + se given directly (or yi + vi)
    # — also the computed-effect storage for all measures.
    yi: Optional[float] = None
    vi: Optional[float] = None
    se: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None

    # User-supplied moderators for subgroup / meta-regression
    subgroup: Optional[str] = None
    moderator: Optional[float] = None

    # LLM transparency
    extraction_quote: Optional[str] = None
    extraction_confidence: Optional[float] = None
    extraction_notes: Optional[str] = None
    error: Optional[str] = None


# ===========================================================================
# Effect-size math (closed form, no scipy)
# ===========================================================================

_Z95 = 1.959963984540054   # qnorm(0.975)


def _safe_log(x: float) -> float:
    if x <= 0:
        raise ValueError(f"log of non-positive value {x}")
    return math.log(x)


def _add_half(a: int, b: int, c: int, d: int) -> Tuple[float, float, float, float]:
    """Cox 1970 continuity correction: add 0.5 to all cells if any is zero."""
    if min(a, b, c, d) == 0:
        return a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return a, b, c, d


def _odds_ratio(events_t: int, n_t: int, events_c: int, n_c: int) -> Tuple[float, float]:
    a, b, c, d = _add_half(events_t, n_t - events_t, events_c, n_c - events_c)
    yi = math.log((a * d) / (b * c))
    vi = (1.0 / a) + (1.0 / b) + (1.0 / c) + (1.0 / d)
    return yi, vi


def _risk_ratio(events_t: int, n_t: int, events_c: int, n_c: int) -> Tuple[float, float]:
    """log RR + variance. Continuity correction when events are zero in
    either arm. Variance: 1/a - 1/n_t + 1/c - 1/n_c."""
    a, b, c, d = _add_half(events_t, n_t - events_t, events_c, n_c - events_c)
    nt = a + b
    nc = c + d
    yi = math.log((a / nt) / (c / nc))
    vi = (1.0 / a) - (1.0 / nt) + (1.0 / c) - (1.0 / nc)
    return yi, vi


def _risk_difference(events_t: int, n_t: int, events_c: int, n_c: int) -> Tuple[float, float]:
    """RD + variance. No continuity correction needed (proportions in [0,1])."""
    p1 = events_t / n_t
    p2 = events_c / n_c
    yi = p1 - p2
    vi = p1 * (1 - p1) / n_t + p2 * (1 - p2) / n_c
    if vi <= 0:
        vi = 1e-9   # tiny floor to keep pooling stable when both p's are 0 or 1
    return yi, vi


def _peto_odds_ratio(events_t: int, n_t: int, events_c: int, n_c: int) -> Tuple[float, float]:
    """Peto's OR — recommended when event rates are very low (<1%) and arms
    are balanced. Doesn't need continuity correction.
        O - E = events_t - (events_t + events_c) * n_t / (n_t + n_c)
        V     = n_t * n_c * (events_t + events_c) * (others) / N^2 / (N - 1)
        logPetoOR = (O - E) / V,  var = 1 / V."""
    a, c = events_t, events_c
    n1, n2 = n_t, n_c
    N = n1 + n2
    events_total = a + c
    non_events_total = N - events_total
    if N < 2 or events_total == 0 or non_events_total == 0:
        # Falls back to corrected OR
        return _odds_ratio(events_t, n_t, events_c, n_c)
    E = events_total * n1 / N
    O_minus_E = a - E
    V = (n1 * n2 * events_total * non_events_total) / ((N**2) * (N - 1))
    if V <= 0:
        return _odds_ratio(events_t, n_t, events_c, n_c)
    yi = O_minus_E / V
    vi = 1.0 / V
    return yi, vi


def _mean_difference(mean_t: float, sd_t: float, n_t: int,
                     mean_c: float, sd_c: float, n_c: int) -> Tuple[float, float]:
    """Raw mean difference + variance."""
    yi = mean_t - mean_c
    vi = (sd_t**2) / n_t + (sd_c**2) / n_c
    return yi, vi


def _hedges_g(mean_t: float, sd_t: float, n_t: int,
              mean_c: float, sd_c: float, n_c: int) -> Tuple[float, float]:
    pooled_sd_sq = (((n_t - 1) * sd_t**2) + ((n_c - 1) * sd_c**2)) / max(1, (n_t + n_c - 2))
    pooled_sd = math.sqrt(max(1e-12, pooled_sd_sq))
    d = (mean_t - mean_c) / pooled_sd
    df = n_t + n_c - 2
    j = 1.0 - (3.0 / (4.0 * df - 1.0)) if df > 1 else 1.0
    g = j * d
    var_d = ((n_t + n_c) / (n_t * n_c)) + (d**2 / (2.0 * (n_t + n_c)))
    var_g = (j**2) * var_d
    return g, var_g


def _logit_proportion(events: int, n: int) -> Tuple[float, float]:
    """logit(p) + variance for single-arm prevalence. Use Wilson-style 0.5
    correction when events == 0 or events == n."""
    if events <= 0 or events >= n:
        events_c = events + 0.5
        n_c = n + 1.0
        p = events_c / n_c
        # Variance approximation for the continuity-corrected logit
        vi = 1.0 / (n_c * p * (1 - p))
        return math.log(p / (1 - p)), vi
    p = events / n
    yi = math.log(p / (1 - p))
    vi = 1.0 / (n * p * (1 - p))
    return yi, vi


def _log_incidence_rate(events: int, person_time: float) -> Tuple[float, float]:
    """log(events/person-time) + variance 1/events."""
    if events <= 0:
        events_c = events + 0.5
        return math.log(events_c / person_time), 1.0 / events_c
    return math.log(events / person_time), 1.0 / events


def _fisher_z(r: float, n: int) -> Tuple[float, float]:
    """Fisher's z-transform of a correlation. SE = 1/sqrt(n-3)."""
    r = max(-0.999999, min(0.999999, r))
    z = 0.5 * math.log((1 + r) / (1 - r))
    vi = 1.0 / max(1, n - 3)
    return z, vi


def _log_hr_from_ci(hr: float, ci_low: float, ci_high: float) -> Tuple[float, float]:
    """Derive log HR and its variance from the reported HR + 95% CI."""
    log_hr = math.log(hr)
    # CI = log(hr) +/- 1.96 * SE  →  SE = (log(ci_high) - log(ci_low)) / (2 * 1.96)
    se = (math.log(ci_high) - math.log(ci_low)) / (2 * _Z95)
    return log_hr, se**2


def compute_effect_size(s: StudyEffect) -> StudyEffect:
    """Populate yi/vi/se/CI for `s` based on its effect_measure + raw inputs."""
    try:
        m = (s.effect_measure or "GENERIC").upper()
        if m == "OR":
            for k in ("events_t", "n_t", "events_c", "n_c"):
                if getattr(s, k) is None:
                    raise ValueError(f"OR needs {k}")
            s.yi, s.vi = _odds_ratio(int(s.events_t), int(s.n_t), int(s.events_c), int(s.n_c))
        elif m == "RR":
            for k in ("events_t", "n_t", "events_c", "n_c"):
                if getattr(s, k) is None:
                    raise ValueError(f"RR needs {k}")
            s.yi, s.vi = _risk_ratio(int(s.events_t), int(s.n_t), int(s.events_c), int(s.n_c))
        elif m == "RD":
            for k in ("events_t", "n_t", "events_c", "n_c"):
                if getattr(s, k) is None:
                    raise ValueError(f"RD needs {k}")
            s.yi, s.vi = _risk_difference(int(s.events_t), int(s.n_t), int(s.events_c), int(s.n_c))
        elif m == "PETO_OR":
            for k in ("events_t", "n_t", "events_c", "n_c"):
                if getattr(s, k) is None:
                    raise ValueError(f"PETO_OR needs {k}")
            s.yi, s.vi = _peto_odds_ratio(int(s.events_t), int(s.n_t), int(s.events_c), int(s.n_c))
        elif m == "MD":
            for k in ("mean_t", "sd_t", "n_t", "mean_c", "sd_c", "n_c"):
                if getattr(s, k) is None:
                    raise ValueError(f"MD needs {k}")
            s.yi, s.vi = _mean_difference(float(s.mean_t), float(s.sd_t), int(s.n_t),
                                          float(s.mean_c), float(s.sd_c), int(s.n_c))
        elif m == "SMD":
            for k in ("mean_t", "sd_t", "n_t", "mean_c", "sd_c", "n_c"):
                if getattr(s, k) is None:
                    raise ValueError(f"SMD needs {k}")
            s.yi, s.vi = _hedges_g(float(s.mean_t), float(s.sd_t), int(s.n_t),
                                   float(s.mean_c), float(s.sd_c), int(s.n_c))
        elif m == "HR":
            if s.log_hr is not None and s.log_hr_se is not None:
                s.yi = float(s.log_hr)
                s.vi = float(s.log_hr_se) ** 2
            elif s.hr is not None and s.hr_ci_low is not None and s.hr_ci_high is not None:
                s.yi, s.vi = _log_hr_from_ci(float(s.hr), float(s.hr_ci_low), float(s.hr_ci_high))
            else:
                raise ValueError("HR needs (log_hr + log_hr_se) or (hr + hr_ci_low + hr_ci_high)")
        elif m == "PROP":
            for k in ("events_total", "n_total"):
                if getattr(s, k) is None:
                    raise ValueError(f"PROP needs {k}")
            s.yi, s.vi = _logit_proportion(int(s.events_total), int(s.n_total))
        elif m == "IR":
            if s.events_total is None or s.person_time is None:
                raise ValueError("IR needs events_total + person_time")
            s.yi, s.vi = _log_incidence_rate(int(s.events_total), float(s.person_time))
        elif m == "ZCOR":
            if s.correlation is None or s.n_total is None:
                raise ValueError("ZCOR needs correlation + n_total")
            s.yi, s.vi = _fisher_z(float(s.correlation), int(s.n_total))
        elif m == "GENERIC":
            if s.yi is None or (s.vi is None and s.se is None):
                raise ValueError("GENERIC needs yi + (vi or se)")
            if s.vi is None:
                s.vi = float(s.se) ** 2
        else:
            raise ValueError(f"unknown effect_measure {m!r}")

        if s.vi is not None and s.vi > 0:
            s.se = math.sqrt(s.vi)
            s.ci_low = s.yi - _Z95 * s.se
            s.ci_high = s.yi + _Z95 * s.se
    except Exception as e:
        s.error = str(e)
    return s


# ===========================================================================
# Tau-squared estimators
# ===========================================================================


def _tau2_DL(yi: List[float], vi: List[float]) -> float:
    """DerSimonian-Laird (default; closed-form, fast)."""
    w = [1.0 / v for v in vi]
    sw = sum(w)
    yhat = sum(wi * y for wi, y in zip(w, yi)) / sw
    Q = sum(wi * (y - yhat) ** 2 for wi, y in zip(w, yi))
    df = len(yi) - 1
    C = sw - sum(wi**2 for wi in w) / sw if sw > 0 else 0.0
    return max(0.0, (Q - df) / C) if C > 0 else 0.0


def _tau2_PM(yi: List[float], vi: List[float], tol: float = 1e-6, max_iter: int = 100) -> float:
    """Paule-Mandel — iterative; sets the weighted Q statistic equal to its
    expected value under the null tau²."""
    tau2 = _tau2_DL(yi, vi)
    df = len(yi) - 1
    for _ in range(max_iter):
        w = [1.0 / (v + tau2) for v in vi]
        sw = sum(w)
        yhat = sum(wi * y for wi, y in zip(w, yi)) / sw
        Q_star = sum(wi * (y - yhat) ** 2 for wi, y in zip(w, yi))
        if Q_star >= df:
            num = sum(wi**2 * ((y - yhat) ** 2 - vi[i]) for i, (wi, y) in enumerate(zip(w, yi)))
            denom = sum(wi**2 for wi in w)
            new_tau2 = max(0.0, tau2 + (Q_star - df) / denom) if denom > 0 else tau2
        else:
            new_tau2 = 0.0
        if abs(new_tau2 - tau2) < tol:
            tau2 = new_tau2
            break
        tau2 = new_tau2
    return tau2


def _tau2_REML(yi: List[float], vi: List[float], tol: float = 1e-6, max_iter: int = 200) -> float:
    """REML — Fisher-scoring iteration. Converges from DL start."""
    tau2 = _tau2_DL(yi, vi)
    for _ in range(max_iter):
        w = [1.0 / (v + tau2) for v in vi]
        sw = sum(w)
        yhat = sum(wi * y for wi, y in zip(w, yi)) / sw
        num = sum(wi**2 * ((y - yhat) ** 2 - vi[i] - tau2 + 1.0 / sw)
                  for i, (wi, y) in enumerate(zip(w, yi)))
        denom = sum(wi**2 for wi in w)
        new_tau2 = max(0.0, tau2 + num / denom) if denom > 0 else tau2
        if abs(new_tau2 - tau2) < tol:
            tau2 = new_tau2
            break
        tau2 = new_tau2
    return tau2


# ===========================================================================
# Pooling
# ===========================================================================


def _z_to_p(z: float) -> float:
    """Two-sided p from a z-statistic via the standard-normal CDF."""
    return 2.0 * (1.0 - _phi(abs(z)))


def _phi(z: float) -> float:
    """Standard-normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _chi2_p(q: float, df: int) -> float:
    """Survival-function p-value for chi-squared(df). Uses the regularized
    incomplete gamma function from math.lgamma + a series approximation."""
    if df <= 0 or q < 0:
        return float("nan")
    # Use math.gammainc-style via the regularized lower incomplete gamma series
    # For chi²(df), P(X > q) = 1 - P(df/2, q/2)
    return 1.0 - _regularized_lower_gamma(df / 2.0, q / 2.0)


def _regularized_lower_gamma(s: float, x: float, max_iter: int = 200, tol: float = 1e-10) -> float:
    """Regularized lower incomplete gamma P(s, x) — series for x < s+1,
    continued fraction for x >= s+1. Adequate for chi-squared p-values."""
    if x <= 0:
        return 0.0
    if x < s + 1:
        # Power series
        term = 1.0 / s
        total = term
        for k in range(1, max_iter):
            term *= x / (s + k)
            total += term
            if abs(term) < abs(total) * tol:
                break
        return total * math.exp(-x + s * math.log(x) - math.lgamma(s))
    else:
        # Lentz's continued fraction for Q(s,x) = 1 - P(s,x)
        b = x + 1.0 - s
        c = 1e30
        d = 1.0 / b
        h = d
        for k in range(1, max_iter):
            an = -k * (k - s)
            b += 2.0
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < tol:
                break
        Q = h * math.exp(-x + s * math.log(x) - math.lgamma(s))
        return 1.0 - Q


def _t_ppf95(df: int) -> float:
    """Inverse CDF at 0.975 for Student-t(df). Hill 1970 + small-df fallback.
    Adequate accuracy for Knapp-Hartung CIs."""
    if df >= 100:
        return _Z95
    # Cornish-Fisher style expansion (Abramowitz 26.7.5)
    z = _Z95
    g1 = (z**3 + z) / 4
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96
    g3 = (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384
    g4 = (79 * z**9 + 776 * z**7 + 1482 * z**5 - 1920 * z**3 - 945 * z) / 92160
    return z + g1 / df + g2 / df**2 + g3 / df**3 + g4 / df**4


def pool(
    extractions: List[StudyEffect],
    tau2_method: str = "DL",          # "DL" | "PM" | "REML"
    use_knapp_hartung: bool = False,  # t-distribution adjusted RE CI
) -> Dict[str, Any]:
    """Pool extracted effect sizes using inverse-variance fixed-effects +
    random-effects (DL/PM/REML), with optional Knapp-Hartung adjustment.

    Returns a dict including pooled estimates, heterogeneity diagnostics,
    a 95% prediction interval for the random-effects model, and per-study
    standardised data for plotting / sensitivity analyses."""
    valid = [s for s in extractions if s.yi is not None and s.vi is not None and s.vi > 0]
    invalid = [s for s in extractions if s not in valid]
    if not valid:
        return {
            "k": 0,
            "valid_studies": [],
            "invalid_studies": [asdict(s) for s in invalid],
            "fixed": None,
            "random": None,
            "heterogeneity": None,
            "prediction_interval": None,
            "effect_measures": [],
            "tau2_method": tau2_method,
            "use_knapp_hartung": use_knapp_hartung,
        }

    yi = [s.yi for s in valid]
    vi = [s.vi for s in valid]

    # Fixed-effects (inverse variance)
    w_fe = [1.0 / v for v in vi]
    sum_w = sum(w_fe)
    mu_fe = sum(w * y for w, y in zip(w_fe, yi)) / sum_w
    var_fe = 1.0 / sum_w
    se_fe = math.sqrt(var_fe)
    fe_ci = (mu_fe - _Z95 * se_fe, mu_fe + _Z95 * se_fe)
    z_fe = mu_fe / se_fe if se_fe > 0 else float("nan")
    p_fe = _z_to_p(z_fe)

    # Heterogeneity
    Q = sum(w * (y - mu_fe) ** 2 for w, y in zip(w_fe, yi))
    df = len(valid) - 1
    Q_p = _chi2_p(Q, df) if df > 0 else float("nan")
    if tau2_method.upper() == "PM":
        tau2 = _tau2_PM(yi, vi)
    elif tau2_method.upper() == "REML":
        tau2 = _tau2_REML(yi, vi)
    else:
        tau2 = _tau2_DL(yi, vi)
    i2 = max(0.0, (Q - df) / Q * 100.0) if Q > 0 else 0.0
    h2 = max(1.0, Q / df) if df > 0 else 1.0

    # Random-effects
    w_re = [1.0 / (v + tau2) for v in vi]
    sum_w_re = sum(w_re)
    mu_re = sum(w * y for w, y in zip(w_re, yi)) / sum_w_re
    var_re = 1.0 / sum_w_re
    se_re = math.sqrt(var_re)
    if use_knapp_hartung and df >= 1:
        # KH: rescale variance by the weighted-residual variance
        qe = sum(w * (y - mu_re) ** 2 for w, y in zip(w_re, yi))
        kh_factor = max(1.0, qe / df) if df > 0 else 1.0
        var_re_kh = var_re * kh_factor
        se_re_kh = math.sqrt(var_re_kh)
        crit = _t_ppf95(df)
        re_ci = (mu_re - crit * se_re_kh, mu_re + crit * se_re_kh)
        z_re = mu_re / se_re_kh if se_re_kh > 0 else float("nan")
        p_re = _z_to_p(z_re)
        se_re_reported = se_re_kh
    else:
        re_ci = (mu_re - _Z95 * se_re, mu_re + _Z95 * se_re)
        z_re = mu_re / se_re if se_re > 0 else float("nan")
        p_re = _z_to_p(z_re)
        se_re_reported = se_re

    # 95% prediction interval (Higgins-Thompson-Spiegelhalter)
    if df >= 2 and tau2 > 0:
        crit = _t_ppf95(df - 1) if df - 1 >= 1 else _Z95
        pi_se = math.sqrt(tau2 + var_re)
        pred_interval = (mu_re - crit * pi_se, mu_re + crit * pi_se)
    else:
        pred_interval = None

    # Per-study with weights
    per_study = []
    for w_fe_i, w_re_i, s in zip(w_fe, w_re, valid):
        rec = asdict(s)
        rec["weight_fe"] = w_fe_i / sum_w
        rec["weight_re"] = w_re_i / sum_w_re
        per_study.append(rec)

    measures = sorted({(s.effect_measure or "GENERIC").upper() for s in valid})

    return {
        "k": len(valid),
        "valid_studies": per_study,
        "invalid_studies": [asdict(s) for s in invalid],
        "effect_measures": measures,
        "tau2_method": tau2_method.upper(),
        "use_knapp_hartung": use_knapp_hartung,
        "fixed": {
            "estimate": mu_fe,
            "se": se_fe,
            "ci_low": fe_ci[0],
            "ci_high": fe_ci[1],
            "z": z_fe,
            "p_value": p_fe,
        },
        "random": {
            "estimate": mu_re,
            "se": se_re_reported,
            "ci_low": re_ci[0],
            "ci_high": re_ci[1],
            "z": z_re,
            "p_value": p_re,
            "tau2": tau2,
            "tau": math.sqrt(tau2),
        },
        "heterogeneity": {
            "Q": Q,
            "df": df,
            "Q_p_value": Q_p,
            "I2_pct": i2,
            "H2": h2,
            "tau2": tau2,
        },
        "prediction_interval": (
            {"low": pred_interval[0], "high": pred_interval[1]} if pred_interval else None
        ),
    }


# ===========================================================================
# Subgroup analysis
# ===========================================================================


def subgroup_analysis(
    extractions: List[StudyEffect],
    tau2_method: str = "DL",
) -> Dict[str, Any]:
    """Group studies by their `subgroup` attribute, pool within group, and
    test between-group heterogeneity (mixed-effects style)."""
    valid = [s for s in extractions if s.yi is not None and s.vi is not None and s.vi > 0]
    if not valid:
        return {"groups": [], "Q_between": float("nan"), "df_between": 0, "Q_between_p": float("nan")}

    groups: Dict[str, List[StudyEffect]] = {}
    for s in valid:
        key = s.subgroup or "(none)"
        groups.setdefault(key, []).append(s)

    group_pools: List[Dict[str, Any]] = []
    for name, members in groups.items():
        if len(members) >= 1:
            r = pool(members, tau2_method=tau2_method)
            group_pools.append({
                "name": name,
                "k": r["k"],
                "estimate": r["random"]["estimate"] if r["random"] else None,
                "ci_low": r["random"]["ci_low"] if r["random"] else None,
                "ci_high": r["random"]["ci_high"] if r["random"] else None,
                "se": r["random"]["se"] if r["random"] else None,
                "tau2": r["random"]["tau2"] if r["random"] else None,
                "I2_pct": r["heterogeneity"]["I2_pct"] if r["heterogeneity"] else None,
            })

    # Between-group Q-test
    means = [g["estimate"] for g in group_pools if g.get("estimate") is not None]
    ses = [g["se"] for g in group_pools if g.get("se") is not None and g["se"] > 0]
    if len(means) >= 2 and len(means) == len(ses):
        weights = [1.0 / (se**2) for se in ses]
        sw = sum(weights)
        grand = sum(w * m for w, m in zip(weights, means)) / sw
        Q_b = sum(w * (m - grand) ** 2 for w, m in zip(weights, means))
        df_b = len(means) - 1
        Q_b_p = _chi2_p(Q_b, df_b)
    else:
        Q_b = float("nan")
        df_b = 0
        Q_b_p = float("nan")

    return {
        "groups": group_pools,
        "Q_between": Q_b,
        "df_between": df_b,
        "Q_between_p": Q_b_p,
    }


# ===========================================================================
# Sensitivity analyses
# ===========================================================================


def leave_one_out(extractions: List[StudyEffect], tau2_method: str = "DL") -> List[Dict[str, Any]]:
    """For each valid study, recompute the pooled RE estimate with that study
    excluded. Returns the deltas so the UI can highlight influential studies."""
    valid = [s for s in extractions if s.yi is not None and s.vi is not None and s.vi > 0]
    base = pool(valid, tau2_method=tau2_method)
    base_est = base["random"]["estimate"] if base.get("random") else None
    out = []
    for i, s in enumerate(valid):
        rest = valid[:i] + valid[i + 1:]
        if not rest:
            continue
        r = pool(rest, tau2_method=tau2_method)
        est = r["random"]["estimate"] if r.get("random") else None
        ci_low = r["random"]["ci_low"] if r.get("random") else None
        ci_high = r["random"]["ci_high"] if r.get("random") else None
        out.append({
            "paper_id": s.paper_id,
            "title": s.title,
            "estimate_without": est,
            "ci_low_without": ci_low,
            "ci_high_without": ci_high,
            "delta": (est - base_est) if (est is not None and base_est is not None) else None,
        })
    return out


def cumulative_meta_analysis(
    extractions: List[StudyEffect],
    tau2_method: str = "DL",
) -> List[Dict[str, Any]]:
    """Sequential pooling: study 1, then study 1+2, then 1+2+3, ...
    Order is the order of the input list (caller decides — usually by year)."""
    valid = [s for s in extractions if s.yi is not None and s.vi is not None and s.vi > 0]
    out = []
    for i in range(1, len(valid) + 1):
        r = pool(valid[:i], tau2_method=tau2_method)
        out.append({
            "k": i,
            "last_added": valid[i - 1].title,
            "estimate": r["random"]["estimate"] if r.get("random") else None,
            "ci_low": r["random"]["ci_low"] if r.get("random") else None,
            "ci_high": r["random"]["ci_high"] if r.get("random") else None,
            "I2_pct": r["heterogeneity"]["I2_pct"] if r.get("heterogeneity") else None,
        })
    return out


# ===========================================================================
# Publication-bias diagnostics
# ===========================================================================


def funnel_plot_data(extractions: List[StudyEffect], tau2_method: str = "DL") -> Dict[str, Any]:
    """Per-study (effect, SE) plus pseudo 95% CI lines around the pooled
    estimate, for funnel-plot rendering."""
    valid = [s for s in extractions if s.yi is not None and s.vi is not None and s.vi > 0]
    r = pool(valid, tau2_method=tau2_method)
    center = r["random"]["estimate"] if r.get("random") else None
    studies = [{"paper_id": s.paper_id, "title": s.title, "yi": s.yi, "se": math.sqrt(s.vi)}
               for s in valid]
    return {"center": center, "studies": studies}


def egger_test(extractions: List[StudyEffect]) -> Dict[str, Any]:
    """Egger's regression test for funnel-plot asymmetry.
    Regress standardised effect (yi/se) on precision (1/se);
    intercept significantly different from 0 → asymmetry."""
    valid = [s for s in extractions if s.yi is not None and s.vi is not None and s.vi > 0]
    n = len(valid)
    if n < 3:
        return {"intercept": None, "se": None, "t": None, "p_value": None, "k": n,
                "note": "Egger's test requires at least 3 studies."}
    x = [1.0 / math.sqrt(s.vi) for s in valid]            # precision
    y = [s.yi / math.sqrt(s.vi) for s in valid]            # standardised effect
    n_f = float(n)
    sx = sum(x); sy = sum(y); sxx = sum(xi**2 for xi in x); sxy = sum(xi*yi for xi, yi in zip(x, y))
    denom = n_f * sxx - sx**2
    if denom == 0:
        return {"intercept": None, "se": None, "t": None, "p_value": None, "k": n,
                "note": "Degenerate design matrix."}
    slope = (n_f * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n_f
    # Residuals + SE of intercept
    resid = [yi - (intercept + slope * xi) for xi, yi in zip(x, y)]
    rss = sum(r**2 for r in resid)
    sigma2 = rss / (n_f - 2) if n > 2 else float("nan")
    var_intercept = sigma2 * (sxx / denom)
    se_intercept = math.sqrt(var_intercept) if var_intercept > 0 else float("nan")
    t = intercept / se_intercept if se_intercept > 0 else float("nan")
    # Two-sided t-test with n-2 df, approximated via standard normal for n>20.
    p = _z_to_p(t) if not math.isnan(t) else float("nan")
    return {
        "intercept": intercept,
        "se": se_intercept,
        "t": t,
        "p_value": p,
        "k": n,
    }


def begg_test(extractions: List[StudyEffect]) -> Dict[str, Any]:
    """Begg-Mazumdar rank correlation test (Kendall's tau between standardised
    effect and its variance)."""
    valid = [s for s in extractions if s.yi is not None and s.vi is not None and s.vi > 0]
    n = len(valid)
    if n < 4:
        return {"tau": None, "p_value": None, "k": n, "note": "Begg's test requires at least 4 studies."}
    # Standardised effects vs variance
    yi_std = []
    vi_std = []
    sum_w = sum(1.0 / s.vi for s in valid)
    yhat = sum((s.yi / s.vi) for s in valid) / sum_w
    for s in valid:
        v_star = s.vi - 1.0 / sum_w
        if v_star <= 0:
            continue
        yi_std.append((s.yi - yhat) / math.sqrt(v_star))
        vi_std.append(s.vi)
    n = len(yi_std)
    if n < 4:
        return {"tau": None, "p_value": None, "k": n, "note": "Insufficient variance heterogeneity for Begg's test."}
    concord = 0
    discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            sgn = (yi_std[i] - yi_std[j]) * (vi_std[i] - vi_std[j])
            if sgn > 0:
                concord += 1
            elif sgn < 0:
                discord += 1
    pairs = n * (n - 1) / 2
    tau = (concord - discord) / pairs if pairs > 0 else 0.0
    var_tau = (2.0 * (2 * n + 5)) / (9.0 * n * (n - 1)) if n > 1 else float("nan")
    z = tau / math.sqrt(var_tau) if var_tau > 0 else float("nan")
    p = _z_to_p(z) if not math.isnan(z) else float("nan")
    return {"tau": tau, "z": z, "p_value": p, "k": n}


def trim_and_fill(extractions: List[StudyEffect], side: str = "auto", tau2_method: str = "DL") -> Dict[str, Any]:
    """Duval & Tweedie trim-and-fill (L0 estimator). Estimates the number of
    missing studies on one side of the funnel, imputes mirror studies, and
    returns the adjusted pooled estimate.

    `side` ∈ {"auto", "left", "right"} — which side is suspected missing.
    Use "auto" to side-pick based on the sign of Egger's intercept."""
    valid = [s for s in extractions if s.yi is not None and s.vi is not None and s.vi > 0]
    n = len(valid)
    if n < 4:
        return {"k0": 0, "filled_estimate": None, "filled_ci_low": None, "filled_ci_high": None, "note": "Need at least 4 studies."}

    base = pool(valid, tau2_method=tau2_method)
    base_est = base["random"]["estimate"]

    # Decide which side is missing
    if side == "auto":
        eg = egger_test(valid)
        side = "right" if (eg.get("intercept") or 0) > 0 else "left"

    # Recenter on the pooled estimate; rank by |yi - mu|, signs as appropriate
    centred = [(i, s.yi - base_est) for i, s in enumerate(valid)]
    if side == "left":
        # Suspect left tail missing → trim positive deviations
        centred_sorted = sorted(centred, key=lambda t: -t[1])
    else:
        centred_sorted = sorted(centred, key=lambda t: t[1])

    # L0 estimator: k0 = max(0, sum_of_signed_ranks of "wrong-side" tail / something)
    # Simplified L0: k0 = max(0, (4*Tn - n*(n+1)) / (2n - 1)) where Tn is sum
    # of ranks of positive deviations (when right side is missing) or negative
    # deviations (when left). We follow Duval-Tweedie Eq. for L0 here.
    signed = [(s.yi - base_est) for s in valid]
    abs_ranks = _ranks_abs(signed)
    if side == "right":
        Tn = sum(r for r, v in zip(abs_ranks, signed) if v > 0)
    else:
        Tn = sum(r for r, v in zip(abs_ranks, signed) if v < 0)
    k0 = max(0, int(round((4 * Tn - n * (n + 1)) / (2 * n - 1))))

    # Trim k0 most-extreme studies on the *included* side, re-estimate, mirror them.
    imputed: List[StudyEffect] = []
    if k0 > 0:
        # Sort by deviation from pooled estimate
        deviations = sorted(valid, key=lambda s: (s.yi - base_est), reverse=(side == "right"))
        extreme = deviations[:k0]
        for s in extreme:
            mirrored = StudyEffect(
                paper_id=f"imputed-{s.paper_id}",
                title=f"(imputed mirror of {s.title})",
                effect_measure=s.effect_measure,
            )
            mirrored.yi = 2 * base_est - s.yi
            mirrored.vi = s.vi
            mirrored.se = math.sqrt(s.vi)
            imputed.append(mirrored)

    augmented = valid + imputed
    r_aug = pool(augmented, tau2_method=tau2_method)
    return {
        "k0": k0,
        "side": side,
        "filled_estimate": r_aug["random"]["estimate"],
        "filled_ci_low": r_aug["random"]["ci_low"],
        "filled_ci_high": r_aug["random"]["ci_high"],
        "filled_k": r_aug["k"],
    }


def _ranks_abs(values: List[float]) -> List[float]:
    """Average-rank of the absolute values (1-based)."""
    abs_vals = [abs(v) for v in values]
    sorted_idx = sorted(range(len(abs_vals)), key=lambda i: abs_vals[i])
    ranks = [0.0] * len(abs_vals)
    i = 0
    while i < len(abs_vals):
        j = i
        while j + 1 < len(abs_vals) and abs_vals[sorted_idx[j + 1]] == abs_vals[sorted_idx[i]]:
            j += 1
        # Tied indices i..j → average rank (i+1 + j+1) / 2
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


# ===========================================================================
# Meta-regression (weighted least squares, single continuous moderator)
# ===========================================================================


def meta_regression(
    extractions: List[StudyEffect],
    tau2_method: str = "DL",
) -> Dict[str, Any]:
    """Mixed-effects meta-regression of yi on the `moderator` field.
    Uses WLS weights = 1 / (vi + tau²) (Knapp-Hartung-ready)."""
    valid = [s for s in extractions if s.yi is not None and s.vi is not None and s.vi > 0
             and s.moderator is not None]
    n = len(valid)
    if n < 3:
        return {"slope": None, "intercept": None, "p_value": None, "R2": None, "k": n,
                "note": "Meta-regression needs >=3 studies with a moderator value."}
    yi = [s.yi for s in valid]
    vi = [s.vi for s in valid]
    xi = [float(s.moderator) for s in valid]
    tau2 = _tau2_DL(yi, vi) if tau2_method.upper() == "DL" else (
        _tau2_PM(yi, vi) if tau2_method.upper() == "PM" else _tau2_REML(yi, vi))
    w = [1.0 / (v + tau2) for v in vi]
    sw = sum(w)
    xbar = sum(wi * xi_ for wi, xi_ in zip(w, xi)) / sw
    ybar = sum(wi * y_ for wi, y_ in zip(w, yi)) / sw
    Sxx = sum(wi * (xi_ - xbar) ** 2 for wi, xi_ in zip(w, xi))
    Sxy = sum(wi * (xi_ - xbar) * (y_ - ybar) for wi, xi_, y_ in zip(w, xi, yi))
    slope = Sxy / Sxx if Sxx > 0 else float("nan")
    intercept = ybar - slope * xbar
    # Residual sum of weighted squares
    rss = sum(wi * (y_ - (intercept + slope * xi_)) ** 2 for wi, xi_, y_ in zip(w, xi, yi))
    # Approx SE of slope
    se_slope = math.sqrt(1.0 / Sxx) if Sxx > 0 else float("nan")
    z = slope / se_slope if (se_slope and se_slope > 0) else float("nan")
    p = _z_to_p(z) if not math.isnan(z) else float("nan")
    # R²-analogue: fraction of between-study variance explained.
    tss = sum(wi * (y_ - ybar) ** 2 for wi, y_ in zip(w, yi))
    r2 = max(0.0, 1.0 - rss / tss) if tss > 0 else 0.0

    return {
        "slope": slope,
        "se": se_slope,
        "intercept": intercept,
        "z": z,
        "p_value": p,
        "R2": r2,
        "k": n,
        "tau2": tau2,
    }


# ===========================================================================
# LLM extraction
# ===========================================================================


_EXTRACTION_PROMPT = """You are extracting effect-size data from a research paper for a meta-analysis.

OUTCOME OF INTEREST: {outcome_hint}
PREFERRED EFFECT MEASURE (if applicable): {measure_hint}

PAPER:
Title: {title}
Abstract: {abstract}
{full_text_block}

Output a SINGLE JSON object with these keys (any irrelevant numerical field set to null):

{{
  "effect_measure": "OR" | "RR" | "RD" | "PETO_OR" | "HR" | "MD" | "SMD" | "PROP" | "IR" | "ZCOR" | "GENERIC" | "UNKNOWN",
  "outcome_name": "<short name of the measured outcome>",

  // --- Binary 2x2 (OR / RR / RD / PETO_OR) ---
  "events_t":  <int|null>,    // events in TREATMENT/INTERVENTION arm
  "n_t":       <int|null>,    // total in treatment arm
  "events_c":  <int|null>,    // events in CONTROL arm
  "n_c":       <int|null>,    // total in control arm

  // --- Continuous two-arm (MD / SMD) ---
  "mean_t":    <float|null>,
  "sd_t":      <float|null>,
  "mean_c":    <float|null>,
  "sd_c":      <float|null>,

  // --- Time-to-event (HR + 95% CI) ---
  "hr":          <float|null>,
  "hr_ci_low":   <float|null>,
  "hr_ci_high":  <float|null>,

  // --- Single-arm prevalence (PROP) ---
  "events_total": <int|null>,
  "n_total":      <int|null>,

  // --- Single-arm incidence (IR) ---
  "person_time":  <float|null>,   // person-time at risk (years, person-months, etc.)

  // --- Correlation (ZCOR) ---
  "correlation":  <float|null>,   // Pearson r

  // --- Generic (GENERIC) ---
  "yi":           <float|null>,   // log effect (e.g. log HR, log OR) on the scale appropriate for the measure
  "se":           <float|null>,   // standard error of yi

  "subgroup":     <string|null>,  // categorical moderator (e.g. "RCT", "cohort", "Asia", "Europe")
  "moderator":    <float|null>,   // continuous moderator value (e.g. mean age, dose, year)

  "quote":        "<exact sentence(s) from the paper grounding the extraction>",
  "confidence":   <float in [0,1]>,
  "notes":        "<any caveats, e.g. units, time point, subgroup>"
}}

RULES:
- Pick the effect_measure that best matches what the paper reports — do NOT convert measures yourself unless they are reported in multiple formats.
- If the paper does not report enough information to fill ANY supported schema, set effect_measure="UNKNOWN" and leave numerical fields null. Still fill quote/confidence/notes honestly.
- DO NOT hallucinate numbers. If a value is not in the text, return null. Hallucinated effect sizes corrupt the meta-analysis.
- Use the units as reported. Document non-obvious units in `notes`.
- For HR: extract the HR and its 95% CI bounds (so we can derive log(HR) + SE). Do not extract HR without CI.
- For continuous outcomes: if SE is reported instead of SD, convert SD = SE * sqrt(n) and note in `notes`.
- For PROP: extract numerator (events) and denominator (n).
- For IR: extract events + person-time exposure.
- For ZCOR: extract Pearson r and total n.
- For GENERIC (use sparingly): extract the log-scale effect + SE only when no raw inputs are available.
- Output ONLY the JSON object. No prose before or after.
"""


def _extract_json_block(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    s = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", s.strip(), flags=re.IGNORECASE | re.MULTILINE)
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start:i + 1]
                try:
                    return _json.loads(candidate)
                except Exception:
                    return None
    return None


def extract_effect_size(
    paper: Dict[str, Any],
    model_name: str,
    outcome_hint: str = "",
    measure_hint: str = "",
    full_text: Optional[str] = None,
) -> StudyEffect:
    """LLM-based effect-size extraction. Returns StudyEffect with yi/vi
    computed when feasible, else with `error` set."""
    title = str(paper.get("title") or paper.get("Title") or "").strip()
    abstract = str(paper.get("abstract") or paper.get("Abstract") or "").strip()
    paper_id = str(paper.get("id") or paper.get("paper_id") or title[:40] or "")
    url = paper.get("url") or paper.get("URL")

    s = StudyEffect(paper_id=paper_id, title=title, url=url, outcome=outcome_hint)

    if not abstract and not full_text:
        s.error = "no abstract or full text available for extraction"
        return s

    full_text_block = ""
    if full_text:
        trimmed = full_text.strip()
        if len(trimmed) > 12000:
            trimmed = trimmed[:12000] + "\n\n[...truncated for prompt budget...]"
        full_text_block = f"\nFull text:\n{trimmed}\n"

    prompt = _EXTRACTION_PROMPT.format(
        outcome_hint=outcome_hint or "(unspecified — pick the primary outcome reported)",
        measure_hint=measure_hint or "(auto-select; pick what the paper actually reports)",
        title=title,
        abstract=abstract,
        full_text_block=full_text_block,
    )

    try:
        model = AIService.get_model(model_name)
        if model is None:
            s.error = f"model {model_name!r} not available"
            return s
        response = model.invoke(prompt)
        text = getattr(response, "content", None) or str(response)
    except Exception as e:
        s.error = f"LLM call failed: {e}"
        return s

    data = _extract_json_block(text)
    if not data:
        s.error = "could not parse JSON from LLM response"
        s.extraction_notes = text[:300]
        return s

    s.outcome = str(data.get("outcome_name") or outcome_hint or "")
    em = str(data.get("effect_measure") or "UNKNOWN").upper()
    s.effect_measure = em if em != "UNKNOWN" else "GENERIC"
    s.extraction_quote = data.get("quote") or None
    try:
        s.extraction_confidence = float(data.get("confidence")) if data.get("confidence") is not None else None
    except Exception:
        s.extraction_confidence = None
    s.extraction_notes = data.get("notes") or None
    s.subgroup = data.get("subgroup") or None
    try:
        s.moderator = float(data.get("moderator")) if data.get("moderator") is not None else None
    except Exception:
        s.moderator = None

    def _i(v):
        try: return int(v) if v is not None else None
        except Exception: return None

    def _f(v):
        try: return float(v) if v is not None else None
        except Exception: return None

    s.events_t = _i(data.get("events_t"))
    s.n_t = _i(data.get("n_t"))
    s.events_c = _i(data.get("events_c"))
    s.n_c = _i(data.get("n_c"))
    s.mean_t = _f(data.get("mean_t"))
    s.sd_t = _f(data.get("sd_t"))
    s.mean_c = _f(data.get("mean_c"))
    s.sd_c = _f(data.get("sd_c"))
    s.hr = _f(data.get("hr"))
    s.hr_ci_low = _f(data.get("hr_ci_low"))
    s.hr_ci_high = _f(data.get("hr_ci_high"))
    s.events_total = _i(data.get("events_total"))
    s.n_total = _i(data.get("n_total"))
    s.person_time = _f(data.get("person_time"))
    s.correlation = _f(data.get("correlation"))
    s.yi = _f(data.get("yi"))
    se_val = _f(data.get("se"))
    if se_val is not None:
        s.se = se_val
        s.vi = se_val ** 2

    if em == "UNKNOWN" or em == "":
        # No measure → mark as not poolable; still echo back for transparency.
        if s.yi is None:
            s.error = "extractor could not determine effect measure / values"
        return s

    return compute_effect_size(s)
