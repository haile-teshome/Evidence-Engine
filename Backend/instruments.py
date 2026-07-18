"""Quality-appraisal instruments as data + deterministic roll-up.

Three separate axes are modelled here and MUST NOT be conflated:

  * internal_validity  — risk-of-bias tools (RoB 2, ROBINS-I, QUADAS-2, PROBAST, …)
  * reporting          — reporting-completeness checklists (TRIPOD+AI, CONSORT-AI, …)
  * certainty          — GRADE / GRADE-CERQual (OUTCOME level, not study level)

Design (see the engine in api.py):
  * Every instrument is DATA: domains → signalling questions → ordinal response
    options. One generic renderer/engine drives them all; adding a tool is a data
    change, never new UI.
  * The LLM only ANSWERS the signalling questions (with an evidence quote). The
    signalling-answer → domain-judgment → overall roll-up is done HERE in
    deterministic code, following each tool's published algorithm. There is never
    a single holistic "quality score".

Study design routes the instrument automatically (`instrument_for_design`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Cochrane-style signalling response options, most→least reassuring for a
# question phrased so that "Yes" is reassuring. Individual signals flag their own
# polarity via `concern_if` so the roll-up knows which answers indicate a problem.
YN = ["Y", "PY", "PN", "N", "NI"]        # Yes / Probably yes / Probably no / No / No information
YN_LABEL = {"Y": "Yes", "PY": "Probably yes", "PN": "Probably no", "N": "No", "NI": "No information"}

# Judgment scales per instrument family.
SCALE_ROB2 = ["Low", "Some concerns", "High"]
SCALE_ROBINS = ["Low", "Moderate", "Serious", "Critical", "No information"]
SCALE_LHU = ["Low", "High", "Unclear"]            # QUADAS-2 / PROBAST risk of bias
SCALE_GRADE = ["High", "Moderate", "Low", "Very low"]


def _ans_in(a: Optional[str], opts) -> bool:
    return (a or "NI") in opts


# ---------------------------------------------------------------------------
# Per-instrument domain + overall roll-up algorithms
# ---------------------------------------------------------------------------
# Each `domain_judgment(domain_id, answers)` returns an ordinal judgment string;
# `answers` maps signal_id → one of YN. Each `overall(domain_judgments)` maps the
# list of domain judgments to the instrument's overall judgment.

# ---- RoB 2 (Sterne 2019) ----
# Faithful to the RoB 2 algorithm's dominant paths. "reassuring" = Y/PY; "concern"
# = N/PN. NI is treated as inability to reassure (→ some concerns at most).
def rob2_domain(domain_id: str, a: Dict[str, str]) -> str:
    # Signal polarity differs per question: for some, "Yes" is reassuring; for
    # others (e.g. baseline imbalance, selective reporting) "Yes" IS the problem.
    good = lambda k: _ans_in(a.get(k, "NI"), ("Y", "PY"))   # answered yes-ish
    no = lambda k: _ans_in(a.get(k, "NI"), ("N", "PN"))     # answered no-ish
    if domain_id == "d1":  # randomization
        if good("1.3"):                          # baseline imbalance suggests a problem
            return "High"
        if good("1.1") and good("1.2"):          # random sequence + concealed
            return "Low"
        return "Some concerns"
    if domain_id == "d2":  # deviations from intended interventions
        if no("2.6") or good("2.7"):             # no appropriate (ITT) analysis, or impactful deviations
            return "High"
        if good("2.6") and no("2.7"):
            return "Low"
        return "Some concerns"
    if domain_id == "d3":  # missing outcome data
        if good("3.1") or good("3.3"):           # ~complete data, or evidence result is unbiased
            return "Low"
        if good("3.4"):                          # missingness could depend on the true value
            return "High"
        return "Some concerns"
    if domain_id == "d4":  # measurement of the outcome
        if good("4.1") or good("4.2") or good("4.5"):   # inappropriate / differential / influenced
            return "High"
        if no("4.1") and no("4.2") and (no("4.3") or no("4.5")):
            return "Low"
        return "Some concerns"
    if domain_id == "d5":  # selection of the reported result
        if good("5.2") or good("5.3") or no("5.1"):     # result cherry-picked / not pre-specified
            return "High"
        if good("5.1") and no("5.2") and no("5.3"):
            return "Low"
        return "Some concerns"
    return "Some concerns"


def rob2_overall(dj: List[str]) -> str:
    if not dj or all(x == "No information" for x in dj):
        return "Some concerns"
    if any(x == "High" for x in dj):
        return "High"
    if sum(1 for x in dj if x == "Some concerns") >= 3:
        return "High"                           # multiple concerns → overall high
    if any(x == "Some concerns" for x in dj):
        return "Some concerns"
    return "Low"


# ---- ROBINS-I V2 (Sterne 2016 + 2023 update) ----
# Worst-domain roll-up. Each domain maps concern signals to Low/Moderate/Serious/
# Critical. NI → "No information" for the domain.
def robins_domain(domain_id: str, a: Dict[str, str]) -> str:
    vals = [a.get(k, "NI") for k in a]
    if a and all(v == "NI" for v in vals):
        return "No information"
    bad = lambda k: _ans_in(a.get(k, "NI"), ("N", "PN"))
    good = lambda k: _ans_in(a.get(k, "NI"), ("Y", "PY"))
    # Critical: any signal explicitly flagged critical (suffix "!crit") answered No.
    crit = [k for k in a if k.endswith("c")]
    if any(bad(k) for k in crit):
        return "Critical"
    concerns = sum(1 for k in a if not k.endswith("c") and bad(k))
    if concerns == 0 and all(good(k) or a.get(k) == "NI" for k in a):
        return "Low" if any(good(k) for k in a) else "Moderate"
    if concerns >= 2:
        return "Serious"
    if concerns == 1:
        return "Moderate"
    return "Moderate"


def robins_overall(dj: List[str]) -> str:
    order = ["No information", "Critical", "Serious", "Moderate", "Low"]
    if not dj:
        return "Moderate"
    if "Critical" in dj:
        return "Critical"
    if "Serious" in dj:
        return "Serious"
    if "Moderate" in dj:
        return "Moderate"
    if all(x == "No information" for x in dj):
        return "No information"
    return "Low"


# ---- QUADAS-2 / PROBAST (Low / High / Unclear) ----
def lhu_domain(domain_id: str, a: Dict[str, str]) -> str:
    vals = list(a.values())
    if not vals or all(v == "NI" for v in vals):
        return "Unclear"
    if any(_ans_in(v, ("N", "PN")) for v in vals):
        return "High"
    if all(_ans_in(v, ("Y", "PY")) for v in vals):
        return "Low"
    return "Unclear"


def lhu_overall(dj: List[str]) -> str:
    if any(x == "High" for x in dj):
        return "High"
    if any(x == "Unclear" for x in dj):
        return "Unclear"
    return "Low" if dj else "Unclear"


# ---------------------------------------------------------------------------
# Instrument definitions (data)
# ---------------------------------------------------------------------------

def _sig(sid: str, text: str) -> Dict[str, Any]:
    return {"id": sid, "text": text, "options": YN, "labels": YN_LABEL}


ROB2 = {
    "id": "rob2",
    "name": "RoB 2 — risk of bias in randomized trials",
    "short_name": "RoB 2",
    "axis": "internal_validity",
    "scale": SCALE_ROB2,
    "reference": "Sterne JAC et al. BMJ 2019;366:l4898",
    "applies_to": ["rct", "randomized controlled trial", "randomised controlled trial", "randomized trial"],
    "domains": [
        {"id": "d1", "name": "Bias arising from the randomization process", "signals": [
            _sig("1.1", "Was the allocation sequence random?"),
            _sig("1.2", "Was the allocation sequence concealed until participants were enrolled and assigned?"),
            _sig("1.3", "Did baseline differences between groups suggest a problem with the randomization process?"),
        ]},
        {"id": "d2", "name": "Bias due to deviations from intended interventions", "signals": [
            _sig("2.1", "Were participants aware of their assigned intervention?"),
            _sig("2.2", "Were carers/people delivering the interventions aware of assignments?"),
            _sig("2.6", "Was an appropriate analysis used to estimate the effect of assignment to intervention (e.g. ITT)?"),
            _sig("2.7", "Were there deviations from intended intervention that arose because of the trial context, likely affecting the outcome?"),
        ]},
        {"id": "d3", "name": "Bias due to missing outcome data", "signals": [
            _sig("3.1", "Were data for this outcome available for all, or nearly all, participants?"),
            _sig("3.3", "Is there evidence that the result was not biased by missing outcome data?"),
            _sig("3.4", "Could missingness in the outcome depend on its true value?"),
        ]},
        {"id": "d4", "name": "Bias in measurement of the outcome", "signals": [
            _sig("4.1", "Was the method of measuring the outcome inappropriate?"),
            _sig("4.2", "Could measurement/ascertainment of the outcome have differed between groups?"),
            _sig("4.3", "Were outcome assessors aware of the intervention received?"),
            _sig("4.4", "Could assessment of the outcome have been influenced by knowledge of the intervention?"),
            _sig("4.5", "Is it likely that assessment was influenced by knowledge of the intervention?"),
        ]},
        {"id": "d5", "name": "Bias in selection of the reported result", "signals": [
            _sig("5.1", "Were the data analysed in accordance with a pre-specified analysis plan finalised before unblinded outcome data were available?"),
            _sig("5.2", "Was the numerical result being assessed likely selected from multiple eligible outcome measurements?"),
            _sig("5.3", "Was the result likely selected from multiple eligible analyses of the data?"),
        ]},
    ],
    "_domain_fn": "rob2_domain",
    "_overall_fn": "rob2_overall",
}

ROBINS_I = {
    "id": "robins_i",
    "name": "ROBINS-I V2 — non-randomized studies of interventions",
    "short_name": "ROBINS-I",
    "axis": "internal_validity",
    "scale": SCALE_ROBINS,
    "reference": "Sterne JAC et al. BMJ 2016;355:i4919 (V2 update 2023)",
    "applies_to": ["cohort", "case-control", "case control", "non-randomised", "non-randomized",
                   "quasi-experimental", "controlled before-after", "interrupted time series"],
    "domains": [
        {"id": "d1", "name": "Bias due to confounding", "signals": [
            _sig("1.1c", "Did the analysis appropriately control for all important confounding domains?"),
            _sig("1.2", "Were confounders measured validly and reliably?"),
        ]},
        {"id": "d2", "name": "Bias arising from measurement of the exposure/intervention", "signals": [
            _sig("2.1", "Were intervention groups clearly defined and classified using data recorded at the start of the intervention?"),
        ]},
        {"id": "d3", "name": "Bias in selection of participants into the study", "signals": [
            _sig("3.1", "Was selection into the study unrelated to intervention or outcome?"),
        ]},
        {"id": "d4", "name": "Bias due to post-intervention deviations from intended interventions", "signals": [
            _sig("4.1", "Was the intervention implemented as intended, without important co-interventions differing between groups?"),
        ]},
        {"id": "d5", "name": "Bias due to missing data", "signals": [
            _sig("5.1", "Were outcome and intervention/confounder data reasonably complete?"),
        ]},
        {"id": "d6", "name": "Bias arising from measurement of the outcome", "signals": [
            _sig("6.1", "Was the outcome measure objective, or were assessors blinded to intervention status?"),
        ]},
        {"id": "d7", "name": "Bias in selection of the reported result", "signals": [
            _sig("7.1", "Was the reported result unlikely to be selected from multiple analyses/subgroups/outcomes?"),
        ]},
    ],
    "_domain_fn": "robins_domain",
    "_overall_fn": "robins_overall",
}

QUADAS_2 = {
    "id": "quadas_2",
    "name": "QUADAS-2 — diagnostic test accuracy",
    "short_name": "QUADAS-2",
    "axis": "internal_validity",
    "scale": SCALE_LHU,
    "reference": "Whiting PF et al. Ann Intern Med 2011;155:529",
    "applies_to": ["diagnostic", "diagnostic test accuracy", "dta", "test accuracy"],
    "domains": [
        {"id": "d1", "name": "Patient selection", "signals": [
            _sig("1.1", "Was a consecutive or random sample of patients enrolled?"),
            _sig("1.2", "Was a case-control design avoided?"),
            _sig("1.3", "Did the study avoid inappropriate exclusions?"),
        ]},
        {"id": "d2", "name": "Index test", "signals": [
            _sig("2.1", "Were the index test results interpreted without knowledge of the reference standard?"),
            _sig("2.2", "If a threshold was used, was it pre-specified?"),
        ]},
        {"id": "d3", "name": "Reference standard", "signals": [
            _sig("3.1", "Is the reference standard likely to correctly classify the target condition?"),
            _sig("3.2", "Were reference-standard results interpreted without knowledge of the index test?"),
        ]},
        {"id": "d4", "name": "Flow and timing", "signals": [
            _sig("4.1", "Was there an appropriate interval between index test and reference standard?"),
            _sig("4.2", "Did all patients receive the same reference standard, and were all included in the analysis?"),
        ]},
    ],
    "_domain_fn": "lhu_domain",
    "_overall_fn": "lhu_overall",
}

PROBAST_AI = {
    "id": "probast_ai",
    "name": "PROBAST+AI — prediction / AI model risk of bias",
    "short_name": "PROBAST+AI",
    "axis": "internal_validity",
    "scale": SCALE_LHU,
    "reference": "Wolff RF et al. Ann Intern Med 2019;170:51 (+AI extension)",
    "applies_to": ["prediction model", "prognostic model", "diagnostic model", "machine learning",
                   "ai model", "clinical prediction", "risk model"],
    "domains": [
        {"id": "d1", "name": "Participants", "signals": [
            _sig("1.1", "Were appropriate data sources and inclusion/exclusion criteria used?"),
        ]},
        {"id": "d2", "name": "Predictors", "signals": [
            _sig("2.1", "Were predictors defined and assessed similarly for all participants, blinded to outcome?"),
            _sig("2.2", "(AI) Were data leakage and inappropriate feature selection avoided?"),
        ]},
        {"id": "d3", "name": "Outcome", "signals": [
            _sig("3.1", "Was the outcome defined and determined appropriately, without knowledge of predictor information?"),
        ]},
        {"id": "d4", "name": "Analysis", "signals": [
            _sig("4.1", "Were there a reasonable number of events per variable and appropriate handling of missing data?"),
            _sig("4.2", "Were model performance measures (calibration, discrimination) evaluated appropriately?"),
            _sig("4.3", "(AI) Was the model validated on a separate/external dataset with no train-test contamination?"),
        ]},
    ],
    "_domain_fn": "lhu_domain",
    "_overall_fn": "lhu_overall",
}

# --- Scaffolded internal-validity / reporting tools (definitions wired into the
#     same engine; signalling sets to be fully populated in a later pass). ---
def _scaffold(iid, name, short, axis, scale, applies=None, ref=""):
    return {"id": iid, "name": name, "short_name": short, "axis": axis, "scale": scale,
            "reference": ref, "applies_to": applies or [], "domains": [], "scaffold": True,
            "_domain_fn": "lhu_domain", "_overall_fn": "lhu_overall"}


SCAFFOLDS = [
    _scaffold("robins_e", "ROBINS-E — non-randomized studies of exposures", "ROBINS-E", "internal_validity", SCALE_ROBINS, ["exposure", "environmental"]),
    _scaffold("quadas_c", "QUADAS-C — comparative diagnostic accuracy", "QUADAS-C", "internal_validity", SCALE_LHU, ["comparative diagnostic"]),
    _scaffold("quips", "QUIPS — prognostic factor studies", "QUIPS", "internal_validity", SCALE_LHU, ["prognostic factor"]),
    _scaffold("jbi_xsectional", "JBI critical appraisal — analytical cross-sectional", "JBI cross-sectional", "internal_validity", SCALE_LHU, ["cross-sectional", "cross sectional"]),
    _scaffold("jbi_prevalence", "JBI critical appraisal — prevalence studies", "JBI prevalence", "internal_validity", SCALE_LHU, ["prevalence"]),
    _scaffold("jbi_caseseries", "JBI critical appraisal — case series", "JBI case series", "internal_validity", SCALE_LHU, ["case series"]),
    _scaffold("jbi_casereport", "JBI critical appraisal — case report", "JBI case report", "internal_validity", SCALE_LHU, ["case report"]),
    _scaffold("jbi_qualitative", "JBI critical appraisal — qualitative research", "JBI qualitative", "internal_validity", SCALE_LHU, ["qualitative"]),
    _scaffold("jbi_economic", "JBI critical appraisal — economic evaluations", "JBI economic", "internal_validity", SCALE_LHU, ["economic"]),
    _scaffold("amstar_2", "AMSTAR 2 — systematic reviews", "AMSTAR 2", "internal_validity", SCALE_LHU, ["systematic review", "meta-analysis"]),
    _scaffold("robis", "ROBIS — risk of bias in systematic reviews", "ROBIS", "internal_validity", SCALE_LHU, ["systematic review"]),
    _scaffold("nos", "Newcastle-Ottawa Scale (legacy — methodologically weaker; prefer ROBINS-I)", "NOS", "internal_validity", SCALE_LHU, [], "legacy"),
    # Reporting axis (NOT risk of bias)
    _scaffold("tripod_ai", "TRIPOD+AI — reporting of prediction models", "TRIPOD+AI", "reporting", SCALE_LHU, ["prediction model", "ai model"]),
    _scaffold("consort_ai", "CONSORT-AI — reporting of AI trials", "CONSORT-AI", "reporting", SCALE_LHU, ["rct", "ai model"]),
    _scaffold("spirit_ai", "SPIRIT-AI — reporting of AI trial protocols", "SPIRIT-AI", "reporting", SCALE_LHU, ["protocol", "ai model"]),
    _scaffold("quadas_ai", "QUADAS-AI — reporting of AI diagnostic accuracy", "QUADAS-AI", "reporting", SCALE_LHU, ["diagnostic", "ai model"]),
    _scaffold("claim", "CLAIM — checklist for AI in medical imaging", "CLAIM", "reporting", SCALE_LHU, ["imaging", "ai model"]),
]

INSTRUMENTS: Dict[str, Dict[str, Any]] = {
    i["id"]: i for i in [ROB2, ROBINS_I, QUADAS_2, PROBAST_AI, *SCAFFOLDS]
}

_ROLLUP = {
    "rob2_domain": rob2_domain, "rob2_overall": rob2_overall,
    "robins_domain": robins_domain, "robins_overall": robins_overall,
    "lhu_domain": lhu_domain, "lhu_overall": lhu_overall,
}


# ---------------------------------------------------------------------------
# Design → instrument routing (req A: chosen by design, not free pick)
# ---------------------------------------------------------------------------
DESIGN_ROUTES = [
    (("randomized controlled trial", "randomised controlled trial", "randomized trial", "rct"), "rob2"),
    (("diagnostic test accuracy", "diagnostic", "dta", "test accuracy"), "quadas_2"),
    (("prediction model", "prognostic model", "clinical prediction", "risk model", "ai model", "machine learning"), "probast_ai"),
    (("prognostic factor",), "quips"),
    (("exposure",), "robins_e"),
    (("cohort", "case-control", "case control", "non-randomised", "non-randomized", "quasi-experimental",
      "controlled before-after", "interrupted time series"), "robins_i"),
    (("cross-sectional", "cross sectional"), "jbi_xsectional"),
    (("prevalence",), "jbi_prevalence"),
    (("case series",), "jbi_caseseries"),
    (("case report",), "jbi_casereport"),
    (("qualitative",), "jbi_qualitative"),
    (("economic",), "jbi_economic"),
    (("systematic review", "meta-analysis", "meta analysis", "scoping review", "umbrella review"), "amstar_2"),
]


def instrument_for_design(design: Optional[str]) -> str:
    key = (design or "").strip().lower()
    for needles, iid in DESIGN_ROUTES:
        if any(n in key for n in needles):
            return iid
    return "jbi_xsectional"   # broadest fallback


def roll_up(instrument_id: str, domain_answers: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Deterministically compute per-domain judgments + overall from signalling
    answers. `domain_answers` maps domain_id → {signal_id: answer}. Returns
    {"domains": [{id, judgment}], "overall": <judgment>}.
    """
    inst = INSTRUMENTS[instrument_id]
    dfn = _ROLLUP[inst["_domain_fn"]]
    ofn = _ROLLUP[inst["_overall_fn"]]
    domains = []
    for d in inst["domains"]:
        j = dfn(d["id"], domain_answers.get(d["id"], {}))
        domains.append({"id": d["id"], "judgment": j})
    overall = ofn([d["judgment"] for d in domains])
    return {"domains": domains, "overall": overall}


# ---------------------------------------------------------------------------
# GRADE — certainty of the body of evidence (OUTCOME level, req B/certainty)
# ---------------------------------------------------------------------------
# Start High for randomized evidence, Low for observational. Five downgrade
# domains (0 / -1 serious / -2 very serious), three upgrade factors for
# observational evidence. Final certainty clamped to High..Very low.
GRADE_DOWNGRADE = ["risk_of_bias", "inconsistency", "indirectness", "imprecision", "publication_bias"]
GRADE_UPGRADE = ["large_effect", "dose_response", "plausible_confounding"]
_GRADE_ORDER = ["Very low", "Low", "Moderate", "High"]


def grade_certainty(starting: str, downgrades: Dict[str, int], upgrades: Dict[str, int]) -> Dict[str, Any]:
    """starting: 'randomized' | 'observational'. downgrades: domain → 0/-1/-2.
    upgrades: factor → 0/+1/+2 (only applied to observational evidence)."""
    start_idx = 3 if str(starting).lower().startswith("random") else 1   # High vs Low
    down = sum(max(-2, min(0, int(downgrades.get(k, 0)))) for k in GRADE_DOWNGRADE)
    up = 0
    if not str(starting).lower().startswith("random"):
        up = sum(max(0, min(2, int(upgrades.get(k, 0)))) for k in GRADE_UPGRADE)
    idx = max(0, min(3, start_idx + down + up))
    return {"certainty": _GRADE_ORDER[idx], "start": _GRADE_ORDER[start_idx], "net": down + up}


def public_instrument(inst: Dict[str, Any]) -> Dict[str, Any]:
    """A JSON-serialisable instrument definition for the frontend (no functions)."""
    return {
        "id": inst["id"], "name": inst["name"], "short_name": inst["short_name"],
        "axis": inst["axis"], "scale": inst["scale"], "reference": inst.get("reference", ""),
        "applies_to": inst.get("applies_to", []), "scaffold": inst.get("scaffold", False),
        "domains": [{"id": d["id"], "name": d["name"], "signals": d["signals"]} for d in inst["domains"]],
    }
