"""Quality-appraisal instruments as data + deterministic roll-up.

Three separate axes are modelled here and MUST NOT be conflated:

  * internal_validity  — risk-of-bias tools (RoB 2, ROBINS-I, QUADAS-2, PROBAST, …)
  * reporting          — reporting-completeness checklists (TRIPOD+AI, CONSORT-AI, …)
  * certainty          — GRADE / GRADE-CERQual (OUTCOME level, not study level)

Every instrument is DATA: domains → signalling questions (each with a `concern`
polarity saying which answer flags a problem) → ordinal response options. One
generic engine drives them all. The LLM only ANSWERS the signalling questions
(with an evidence quote); the signalling-answer → domain-judgment → overall
roll-up is deterministic code here, following each tool's published algorithm.
There is never a single holistic "quality score".

Study design routes the instrument automatically (`instrument_for_design`), but a
user may override the choice (the API accepts `instrument_id`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

YN = ["Y", "PY", "PN", "N", "NI"]        # Yes / Probably yes / Probably no / No / No information
YN_LABEL = {"Y": "Yes", "PY": "Probably yes", "PN": "Probably no", "N": "No", "NI": "No information"}

SCALE_ROB2 = ["Low", "Some concerns", "High"]
SCALE_ROBINS = ["Low", "Moderate", "Serious", "Critical", "No information"]
SCALE_LHU = ["Low", "High", "Unclear"]            # QUADAS-2 / PROBAST / JBI / NOS risk of bias
SCALE_GRADE = ["High", "Moderate", "Low", "Very low"]


def _yes(a: Optional[str]) -> bool:
    return (a or "NI") in ("Y", "PY")


def _no(a: Optional[str]) -> bool:
    return (a or "NI") in ("N", "PN")


def _sig(sid: str, text: str, concern: str = "no", crit: bool = False) -> Dict[str, Any]:
    """A signalling question. `concern` = which answer flags a problem:
    "no"  → answering No / Probably no indicates concern (question phrased "was it done well?")
    "yes" → answering Yes / Probably yes indicates concern (question phrased "could it bias?")
    `crit` marks a question that, when concerning, can drive a Critical judgment (ROBINS)."""
    return {"id": sid, "text": text, "options": YN, "labels": YN_LABEL, "concern": concern, "crit": crit}


def _is_concern(sig: Dict[str, Any], a: Optional[str]) -> bool:
    return _yes(a) if sig.get("concern") == "yes" else _no(a)


def _is_reassuring(sig: Dict[str, Any], a: Optional[str]) -> bool:
    return _no(a) if sig.get("concern") == "yes" else _yes(a)


def _domain_stats(domain: Dict[str, Any], answers: Dict[str, str]):
    sigs = domain["signals"]
    ans = [answers.get(s["id"], "NI") for s in sigs]
    concerns = sum(1 for s in sigs if _is_concern(s, answers.get(s["id"])))
    crit = any(s.get("crit") and _is_concern(s, answers.get(s["id"])) for s in sigs)
    all_ni = bool(sigs) and all(a == "NI" for a in ans)
    all_reassuring = bool(sigs) and all(_is_reassuring(s, answers.get(s["id"])) for s in sigs)
    return concerns, crit, all_ni, all_reassuring


# ---------------------------------------------------------------------------
# Domain + overall roll-up algorithms (polarity-aware, generic where possible)
# ---------------------------------------------------------------------------

def rob2_domain(domain: Dict[str, Any], a: Dict[str, str]) -> str:
    """RoB 2 per-domain algorithm (dominant published paths)."""
    did = domain["id"]
    yes = lambda k: _yes(a.get(k)); no = lambda k: _no(a.get(k))
    if did == "d1":
        if yes("1.3"): return "High"                       # baseline imbalance = problem
        if yes("1.1") and yes("1.2"): return "Low"
        return "Some concerns"
    if did == "d2":
        if no("2.6") or yes("2.7"): return "High"          # no ITT analysis, or impactful deviations
        if yes("2.6") and no("2.7"): return "Low"
        return "Some concerns"
    if did == "d3":
        if yes("3.1") or yes("3.3"): return "Low"          # ~complete data / evidence unbiased
        if yes("3.4"): return "High"                       # missingness could depend on true value
        return "Some concerns"
    if did == "d4":
        if yes("4.1") or yes("4.2") or yes("4.5"): return "High"
        if no("4.1") and no("4.2") and (no("4.3") or no("4.5")): return "Low"
        return "Some concerns"
    if did == "d5":
        if yes("5.2") or yes("5.3") or no("5.1"): return "High"
        if yes("5.1") and no("5.2") and no("5.3"): return "Low"
        return "Some concerns"
    return "Some concerns"


def rob2_overall(dj: List[str]) -> str:
    if not dj or all(x == "No information" for x in dj):
        return "Some concerns"
    if any(x == "High" for x in dj):
        return "High"
    if sum(1 for x in dj if x == "Some concerns") >= 3:
        return "High"
    if any(x == "Some concerns" for x in dj):
        return "Some concerns"
    return "Low"


def robins_domain(domain: Dict[str, Any], a: Dict[str, str]) -> str:
    """ROBINS-I / ROBINS-E per-domain: Low / Moderate / Serious / Critical / No information."""
    concerns, crit, all_ni, all_reassuring = _domain_stats(domain, a)
    if all_ni:
        return "No information"
    if crit:
        return "Critical"
    if concerns == 0 and all_reassuring:
        return "Low"
    if concerns >= 2:
        return "Serious"
    if concerns == 1:
        return "Moderate"
    return "Moderate"


def robins_overall(dj: List[str]) -> str:
    if not dj:
        return "Moderate"
    if "Critical" in dj: return "Critical"
    if "Serious" in dj: return "Serious"
    if "Moderate" in dj: return "Moderate"
    if all(x == "No information" for x in dj): return "No information"
    return "Low"


def lhu_domain(domain: Dict[str, Any], a: Dict[str, str]) -> str:
    """QUADAS-2 / PROBAST / JBI / NOS per-domain: Low / High / Unclear."""
    concerns, _crit, all_ni, all_reassuring = _domain_stats(domain, a)
    if all_ni:
        return "Unclear"
    if concerns > 0:
        return "High"
    if all_reassuring:
        return "Low"
    return "Unclear"


def lhu_overall(dj: List[str]) -> str:
    if any(x == "High" for x in dj): return "High"
    if any(x == "Unclear" for x in dj): return "Unclear"
    return "Low" if dj else "Unclear"


# ---------------------------------------------------------------------------
# Instrument definitions (data) — full signalling sets
# ---------------------------------------------------------------------------

ROB2 = {
    "id": "rob2", "name": "RoB 2 — risk of bias in randomized trials", "short_name": "RoB 2",
    "axis": "internal_validity", "scale": SCALE_ROB2,
    "reference": "Sterne JAC et al. BMJ 2019;366:l4898",
    "applies_to": ["rct", "randomized controlled trial", "randomised controlled trial", "randomized trial", "cluster-randomized", "crossover trial"],
    "domains": [
        {"id": "d1", "name": "Bias arising from the randomization process", "signals": [
            _sig("1.1", "Was the allocation sequence random?", "no"),
            _sig("1.2", "Was the allocation sequence concealed until participants were enrolled and assigned?", "no"),
            _sig("1.3", "Did baseline differences between groups suggest a problem with the randomization process?", "yes"),
        ]},
        {"id": "d2", "name": "Bias due to deviations from intended interventions", "signals": [
            _sig("2.1", "Were participants aware of their assigned intervention during the trial?", "yes"),
            _sig("2.2", "Were carers and people delivering the interventions aware of assignments?", "yes"),
            _sig("2.3", "If yes, were there deviations from the intended intervention beyond what would occur in usual care?", "yes"),
            _sig("2.4", "If yes, were these deviations likely to have affected the outcome?", "yes"),
            _sig("2.5", "If yes, were these deviations balanced between groups?", "no"),
            _sig("2.6", "Was an appropriate analysis used to estimate the effect of assignment to intervention (e.g. intention-to-treat)?", "no"),
            _sig("2.7", "If not, was there potential for a substantial impact of the failure to analyse participants as randomized?", "yes"),
        ]},
        {"id": "d3", "name": "Bias due to missing outcome data", "signals": [
            _sig("3.1", "Were data for this outcome available for all, or nearly all, randomized participants?", "no"),
            _sig("3.2", "If not, is there evidence that the result was not biased by missing outcome data?", "no"),
            _sig("3.3", "If not, could missingness in the outcome depend on its true value?", "yes"),
            _sig("3.4", "Is it likely that missingness in the outcome depended on its true value?", "yes"),
        ]},
        {"id": "d4", "name": "Bias in measurement of the outcome", "signals": [
            _sig("4.1", "Was the method of measuring the outcome inappropriate?", "yes"),
            _sig("4.2", "Could measurement or ascertainment of the outcome have differed between intervention groups?", "yes"),
            _sig("4.3", "Were outcome assessors aware of the intervention received by study participants?", "yes"),
            _sig("4.4", "Could assessment of the outcome have been influenced by knowledge of the intervention received?", "yes"),
            _sig("4.5", "Is it likely that assessment of the outcome was influenced by knowledge of the intervention received?", "yes"),
        ]},
        {"id": "d5", "name": "Bias in selection of the reported result", "signals": [
            _sig("5.1", "Were the data analysed in accordance with a pre-specified analysis plan finalised before unblinded outcome data were available?", "no"),
            _sig("5.2", "Was the numerical result being assessed likely selected from multiple eligible outcome measurements?", "yes"),
            _sig("5.3", "Was the result likely selected from multiple eligible analyses of the data?", "yes"),
        ]},
    ],
    "_domain_fn": "rob2_domain", "_overall_fn": "rob2_overall",
}

ROBINS_I = {
    "id": "robins_i", "name": "ROBINS-I V2 — non-randomized studies of interventions", "short_name": "ROBINS-I",
    "axis": "internal_validity", "scale": SCALE_ROBINS,
    "reference": "Sterne JAC et al. BMJ 2016;355:i4919 (V2, 2023)",
    "applies_to": ["cohort", "case-control", "case control", "non-randomised", "non-randomized",
                   "quasi-experimental", "controlled before-after", "interrupted time series", "nested case-control"],
    "domains": [
        {"id": "d1", "name": "Bias due to confounding", "signals": [
            _sig("1.1", "Is there potential for confounding of the effect of intervention in this study?", "yes"),
            _sig("1.2", "Was the analysis based on splitting participants' follow-up time by intervention received?", "yes"),
            _sig("1.3", "Were intervention discontinuations or switches likely to be related to factors that are prognostic for the outcome?", "yes"),
            _sig("1.4", "Did the authors use an appropriate analysis method that controlled for all the important confounding domains?", "no", crit=True),
            _sig("1.5", "Were confounding domains that were controlled for measured validly and reliably?", "no"),
            _sig("1.6", "Did the authors control for any post-intervention variables that could have been affected by the intervention?", "yes"),
        ]},
        {"id": "d2", "name": "Bias in classification of interventions", "signals": [
            _sig("2.1", "Were intervention groups clearly defined?", "no"),
            _sig("2.2", "Was information used to define intervention groups recorded at the start of the intervention?", "no"),
            _sig("2.3", "Could classification of intervention status have been affected by knowledge of the outcome or risk of the outcome?", "yes"),
        ]},
        {"id": "d3", "name": "Bias in selection of participants into the study", "signals": [
            _sig("3.1", "Was selection into the study (or into the analysis) based on participant characteristics observed after the start of intervention?", "yes"),
            _sig("3.2", "Were the post-intervention variables that influenced selection likely to be associated with intervention and outcome?", "yes"),
            _sig("3.3", "Did start of follow-up and start of intervention coincide for most participants?", "no"),
            _sig("3.4", "Were adjustment techniques used that are likely to correct for the presence of selection biases?", "no"),
        ]},
        {"id": "d4", "name": "Bias due to deviations from intended interventions", "signals": [
            _sig("4.1", "Were there deviations from the intended intervention beyond what would be expected in usual practice?", "yes"),
            _sig("4.2", "Were these deviations from intended intervention unbalanced between groups and likely to have affected the outcome?", "yes"),
            _sig("4.3", "Were important co-interventions balanced across intervention groups?", "no"),
            _sig("4.4", "Was the intervention implemented successfully for most participants?", "no"),
        ]},
        {"id": "d5", "name": "Bias due to missing data", "signals": [
            _sig("5.1", "Were outcome data available for all, or nearly all, participants?", "no"),
            _sig("5.2", "Were participants excluded due to missing data on intervention status or other variables?", "yes"),
            _sig("5.3", "Are the proportion and reasons for missing participants similar across interventions?", "no"),
            _sig("5.4", "Is there evidence that the result was not biased by missing data?", "no"),
        ]},
        {"id": "d6", "name": "Bias in measurement of the outcome", "signals": [
            _sig("6.1", "Could the outcome measure have been influenced by knowledge of the intervention received?", "yes"),
            _sig("6.2", "Were outcome assessors aware of the intervention received by study participants?", "yes"),
            _sig("6.3", "Were the methods of outcome assessment comparable across intervention groups?", "no"),
            _sig("6.4", "Were any systematic errors in measurement of the outcome unrelated to intervention received?", "no"),
        ]},
        {"id": "d7", "name": "Bias in selection of the reported result", "signals": [
            _sig("7.1", "Is the reported effect estimate likely to be selected, on the basis of the results, from multiple outcome measurements?", "yes"),
            _sig("7.2", "Is the reported effect estimate likely selected from multiple analyses of the intervention-outcome relationship?", "yes"),
            _sig("7.3", "Is the reported estimate likely selected from different subgroups?", "yes"),
        ]},
    ],
    "_domain_fn": "robins_domain", "_overall_fn": "robins_overall",
}

QUADAS_2 = {
    "id": "quadas_2", "name": "QUADAS-2 — diagnostic test accuracy", "short_name": "QUADAS-2",
    "axis": "internal_validity", "scale": SCALE_LHU,
    "reference": "Whiting PF et al. Ann Intern Med 2011;155:529",
    "applies_to": ["diagnostic", "diagnostic test accuracy", "dta", "test accuracy", "sensitivity specificity"],
    "domains": [
        {"id": "d1", "name": "Patient selection", "signals": [
            _sig("1.1", "Was a consecutive or random sample of patients enrolled?", "no"),
            _sig("1.2", "Was a case-control design avoided?", "no"),
            _sig("1.3", "Did the study avoid inappropriate exclusions?", "no"),
            _sig("1.4", "(Applicability) Do the included patients match the review question?", "no"),
        ]},
        {"id": "d2", "name": "Index test", "signals": [
            _sig("2.1", "Were the index test results interpreted without knowledge of the results of the reference standard?", "no"),
            _sig("2.2", "If a threshold was used, was it pre-specified?", "no"),
            _sig("2.3", "(Applicability) Do the test, its conduct, and interpretation match the review question?", "no"),
        ]},
        {"id": "d3", "name": "Reference standard", "signals": [
            _sig("3.1", "Is the reference standard likely to correctly classify the target condition?", "no"),
            _sig("3.2", "Were the reference-standard results interpreted without knowledge of the index test?", "no"),
            _sig("3.3", "(Applicability) Does the target condition as defined by the reference standard match the question?", "no"),
        ]},
        {"id": "d4", "name": "Flow and timing", "signals": [
            _sig("4.1", "Was there an appropriate interval between index test and reference standard?", "no"),
            _sig("4.2", "Did all patients receive a reference standard?", "no"),
            _sig("4.3", "Did all patients receive the same reference standard?", "no"),
            _sig("4.4", "Were all patients included in the analysis?", "no"),
        ]},
    ],
    "_domain_fn": "lhu_domain", "_overall_fn": "lhu_overall",
}

PROBAST_AI = {
    "id": "probast_ai", "name": "PROBAST+AI — prediction / AI model risk of bias", "short_name": "PROBAST+AI",
    "axis": "internal_validity", "scale": SCALE_LHU,
    "reference": "Wolff RF et al. Ann Intern Med 2019;170:51 (+AI extension)",
    "applies_to": ["prediction model", "prognostic model", "diagnostic model", "machine learning",
                   "ai model", "clinical prediction", "risk model", "deep learning", "radiomics"],
    "domains": [
        {"id": "d1", "name": "Participants", "signals": [
            _sig("1.1", "Were appropriate data sources used (e.g. cohort, RCT, registry rather than case-control)?", "no"),
            _sig("1.2", "Were all inclusions and exclusions of participants appropriate?", "no"),
        ]},
        {"id": "d2", "name": "Predictors", "signals": [
            _sig("2.1", "Were predictors defined and assessed in a similar way for all participants?", "no"),
            _sig("2.2", "Were predictor assessments made without knowledge of outcome data?", "no"),
            _sig("2.3", "Are all predictors available at the time the model is intended to be used?", "no"),
            _sig("2.4", "(AI) Were data leakage between predictors and outcome, and inappropriate feature selection, avoided?", "no"),
        ]},
        {"id": "d3", "name": "Outcome", "signals": [
            _sig("3.1", "Was the outcome determined appropriately, using a standard/adequate definition?", "no"),
            _sig("3.2", "Was the outcome defined and determined in a similar way for all participants?", "no"),
            _sig("3.3", "Was the outcome determined without knowledge of predictor information?", "no"),
            _sig("3.4", "Was the time interval between predictor assessment and outcome determination appropriate?", "no"),
        ]},
        {"id": "d4", "name": "Analysis", "signals": [
            _sig("4.1", "Were there a reasonable number of participants with the outcome (events per variable)?", "no"),
            _sig("4.2", "Were continuous and categorical predictors handled appropriately?", "no"),
            _sig("4.3", "Were all enrolled participants included in the analysis?", "no"),
            _sig("4.4", "Were participants with missing data handled appropriately (not simply excluded)?", "no"),
            _sig("4.5", "Was selection of predictors based on univariable analysis avoided?", "no"),
            _sig("4.6", "Were relevant model performance measures evaluated (calibration and discrimination)?", "no"),
            _sig("4.7", "Were model overfitting, underfitting, and optimism accounted for?", "no"),
            _sig("4.8", "(AI) Was the model validated on a separate/external dataset with no train-test contamination?", "no"),
        ]},
    ],
    "_domain_fn": "lhu_domain", "_overall_fn": "lhu_overall",
}

ROBINS_E = {
    "id": "robins_e", "name": "ROBINS-E — non-randomized studies of exposures", "short_name": "ROBINS-E",
    "axis": "internal_validity", "scale": SCALE_ROBINS,
    "reference": "ROBINS-E Development Group, 2023",
    "applies_to": ["exposure", "environmental", "occupational", "observational exposure"],
    "domains": [
        {"id": "d1", "name": "Bias due to confounding", "signals": [
            _sig("1.1", "Did the authors control for all important confounding factors?", "no", crit=True),
            _sig("1.2", "Were the confounding factors measured validly and reliably?", "no"),
        ]},
        {"id": "d2", "name": "Bias arising from measurement of the exposure", "signals": [
            _sig("2.1", "Was exposure characterised using valid and reliable measurements?", "no"),
            _sig("2.2", "Could exposure measurement have been influenced by the outcome or risk of outcome?", "yes"),
        ]},
        {"id": "d3", "name": "Bias in selection of participants into the study", "signals": [
            _sig("3.1", "Was selection into the study unrelated to the exposure or the outcome?", "no"),
        ]},
        {"id": "d4", "name": "Bias due to post-exposure interventions", "signals": [
            _sig("4.1", "Were post-exposure interventions that could affect the outcome balanced across exposure groups?", "no"),
        ]},
        {"id": "d5", "name": "Bias due to missing data", "signals": [
            _sig("5.1", "Were outcome and exposure data reasonably complete?", "no"),
        ]},
        {"id": "d6", "name": "Bias arising from measurement of the outcome", "signals": [
            _sig("6.1", "Could outcome measurement have differed by exposure level?", "yes"),
        ]},
        {"id": "d7", "name": "Bias in selection of the reported result", "signals": [
            _sig("7.1", "Was the reported result selected from multiple exposures, outcomes, or analyses?", "yes"),
        ]},
    ],
    "_domain_fn": "robins_domain", "_overall_fn": "robins_overall",
}

AMSTAR_2 = {
    "id": "amstar_2", "name": "AMSTAR 2 — appraisal of systematic reviews", "short_name": "AMSTAR 2",
    "axis": "internal_validity", "scale": SCALE_LHU,
    "reference": "Shea BJ et al. BMJ 2017;358:j4008",
    "applies_to": ["systematic review", "meta-analysis", "meta analysis"],
    "domains": [
        {"id": "d1", "name": "PICO and protocol (critical: registered protocol)", "signals": [
            _sig("1.1", "Did the research questions and inclusion criteria include the components of PICO?", "no"),
            _sig("1.2", "(Critical) Was there a registered protocol established before the review?", "no", crit=True),
        ]},
        {"id": "d2", "name": "Study selection and search (critical: comprehensive search)", "signals": [
            _sig("2.1", "Did the review authors explain their selection of study designs for inclusion?", "no"),
            _sig("2.2", "(Critical) Was a comprehensive literature search strategy used (≥2 databases + supplementary sources)?", "no", crit=True),
            _sig("2.3", "Was study selection performed in duplicate?", "no"),
            _sig("2.4", "Was data extraction performed in duplicate?", "no"),
        ]},
        {"id": "d3", "name": "Excluded studies and study description (critical: list of exclusions)", "signals": [
            _sig("3.1", "(Critical) Did the authors provide a list of excluded full-text studies with justification?", "no", crit=True),
            _sig("3.2", "Did the authors describe the included studies in adequate detail?", "no"),
        ]},
        {"id": "d4", "name": "Risk of bias and its use (critical)", "signals": [
            _sig("4.1", "(Critical) Did the authors use a satisfactory technique for assessing risk of bias in included studies?", "no", crit=True),
            _sig("4.2", "Did the authors account for risk of bias when interpreting/discussing the results?", "no"),
        ]},
        {"id": "d5", "name": "Meta-analysis methods (critical: appropriate methods)", "signals": [
            _sig("5.1", "(Critical) If meta-analysis was performed, were appropriate methods used for statistical combination?", "no", crit=True),
            _sig("5.2", "Did the authors assess the potential impact of risk of bias on the meta-analysis results?", "no"),
            _sig("5.3", "(Critical) Did the authors investigate and account for publication bias?", "no", crit=True),
        ]},
        {"id": "d6", "name": "Funding and conflicts", "signals": [
            _sig("6.1", "Did the authors report sources of funding for the review and included studies, and any conflicts of interest?", "no"),
        ]},
    ],
    "_domain_fn": "lhu_domain", "_overall_fn": "lhu_overall",
}

JBI_XSECTIONAL = {
    "id": "jbi_xsectional", "name": "JBI — analytical cross-sectional studies", "short_name": "JBI cross-sectional",
    "axis": "internal_validity", "scale": SCALE_LHU,
    "reference": "JBI Critical Appraisal Checklist (Moola et al. 2020)",
    "applies_to": ["cross-sectional", "cross sectional", "prevalence"],
    "domains": [
        {"id": "d1", "name": "Sample and setting", "signals": [
            _sig("1.1", "Were the criteria for inclusion in the sample clearly defined?", "no"),
            _sig("1.2", "Were the study subjects and setting described in detail?", "no"),
        ]},
        {"id": "d2", "name": "Exposure and condition measurement", "signals": [
            _sig("2.1", "Was the exposure measured in a valid and reliable way?", "no"),
            _sig("2.2", "Were objective, standard criteria used for measurement of the condition?", "no"),
            _sig("2.3", "Were the outcomes measured in a valid and reliable way?", "no"),
        ]},
        {"id": "d3", "name": "Confounding", "signals": [
            _sig("3.1", "Were confounding factors identified?", "no"),
            _sig("3.2", "Were strategies to deal with confounding factors stated?", "no"),
        ]},
        {"id": "d4", "name": "Analysis", "signals": [
            _sig("4.1", "Was appropriate statistical analysis used?", "no"),
        ]},
    ],
    "_domain_fn": "lhu_domain", "_overall_fn": "lhu_overall",
}

NOS = {
    "id": "nos", "name": "Newcastle-Ottawa Scale (legacy — methodologically weaker; prefer ROBINS-I)", "short_name": "NOS",
    "axis": "internal_validity", "scale": SCALE_LHU, "legacy": True,
    "reference": "Wells GA et al. (Ottawa Hospital Research Institute)",
    "applies_to": [],
    "domains": [
        {"id": "d1", "name": "Selection", "signals": [
            _sig("1.1", "Was the exposed/case cohort truly or somewhat representative?", "no"),
            _sig("1.2", "Was the non-exposed/control group drawn from the same community?", "no"),
            _sig("1.3", "Was exposure/case status ascertained securely (records or structured interview)?", "no"),
            _sig("1.4", "Was it demonstrated that the outcome of interest was not present at start?", "no"),
        ]},
        {"id": "d2", "name": "Comparability", "signals": [
            _sig("2.1", "Were groups comparable on the most important factor (design or analysis)?", "no"),
            _sig("2.2", "Were groups comparable on any additional factor?", "no"),
        ]},
        {"id": "d3", "name": "Outcome / Exposure", "signals": [
            _sig("3.1", "Was the outcome assessed independently/blindly, or by record linkage?", "no"),
            _sig("3.2", "Was follow-up long enough for outcomes to occur?", "no"),
            _sig("3.3", "Was follow-up adequate/complete?", "no"),
        ]},
    ],
    "_domain_fn": "lhu_domain", "_overall_fn": "lhu_overall",
}


def _scaffold(iid, name, short, axis, scale, applies=None, ref=""):
    return {"id": iid, "name": name, "short_name": short, "axis": axis, "scale": scale,
            "reference": ref, "applies_to": applies or [], "domains": [], "scaffold": True,
            "_domain_fn": "lhu_domain", "_overall_fn": "lhu_overall"}


SCAFFOLDS = [
    _scaffold("quadas_c", "QUADAS-C — comparative diagnostic accuracy", "QUADAS-C", "internal_validity", SCALE_LHU, ["comparative diagnostic"]),
    _scaffold("quips", "QUIPS — prognostic factor studies", "QUIPS", "internal_validity", SCALE_LHU, ["prognostic factor"]),
    _scaffold("jbi_caseseries", "JBI — case series", "JBI case series", "internal_validity", SCALE_LHU, ["case series"]),
    _scaffold("jbi_casereport", "JBI — case report", "JBI case report", "internal_validity", SCALE_LHU, ["case report"]),
    _scaffold("jbi_qualitative", "JBI — qualitative research", "JBI qualitative", "internal_validity", SCALE_LHU, ["qualitative"]),
    _scaffold("jbi_economic", "JBI — economic evaluations", "JBI economic", "internal_validity", SCALE_LHU, ["economic"]),
    _scaffold("robis", "ROBIS — risk of bias in systematic reviews", "ROBIS", "internal_validity", SCALE_LHU, ["systematic review"]),
    _scaffold("tripod_ai", "TRIPOD+AI — reporting of prediction models", "TRIPOD+AI", "reporting", SCALE_LHU, ["prediction model", "ai model"]),
    _scaffold("consort_ai", "CONSORT-AI — reporting of AI trials", "CONSORT-AI", "reporting", SCALE_LHU, ["rct", "ai model"]),
    _scaffold("spirit_ai", "SPIRIT-AI — reporting of AI trial protocols", "SPIRIT-AI", "reporting", SCALE_LHU, ["protocol", "ai model"]),
    _scaffold("quadas_ai", "QUADAS-AI — reporting of AI diagnostic accuracy", "QUADAS-AI", "reporting", SCALE_LHU, ["diagnostic", "ai model"]),
    _scaffold("claim", "CLAIM — checklist for AI in medical imaging", "CLAIM", "reporting", SCALE_LHU, ["imaging", "ai model"]),
]

INSTRUMENTS: Dict[str, Dict[str, Any]] = {
    i["id"]: i for i in [ROB2, ROBINS_I, ROBINS_E, QUADAS_2, PROBAST_AI, AMSTAR_2, JBI_XSECTIONAL, NOS, *SCAFFOLDS]
}

_ROLLUP = {
    "rob2_domain": rob2_domain, "rob2_overall": rob2_overall,
    "robins_domain": robins_domain, "robins_overall": robins_overall,
    "lhu_domain": lhu_domain, "lhu_overall": lhu_overall,
}


# ---------------------------------------------------------------------------
# Design → instrument routing (default; a user may override via instrument_id)
# ---------------------------------------------------------------------------
DESIGN_ROUTES = [
    (("randomized controlled trial", "randomised controlled trial", "randomized trial", "rct", "cluster-randomized", "crossover"), "rob2"),
    (("diagnostic test accuracy", "diagnostic", "dta", "test accuracy"), "quadas_2"),
    (("prediction model", "prognostic model", "clinical prediction", "risk model", "ai model", "machine learning", "deep learning", "radiomics"), "probast_ai"),
    (("prognostic factor",), "quips"),
    (("exposure", "environmental", "occupational"), "robins_e"),
    (("cohort", "case-control", "case control", "non-randomised", "non-randomized", "quasi-experimental",
      "controlled before-after", "interrupted time series"), "robins_i"),
    (("cross-sectional", "cross sectional"), "jbi_xsectional"),
    (("prevalence",), "jbi_xsectional"),
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
    return "jbi_xsectional"


def roll_up(instrument_id: str, domain_answers: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Deterministically compute per-domain judgments + overall from signalling
    answers. `domain_answers` maps domain_id → {signal_id: answer}."""
    inst = INSTRUMENTS[instrument_id]
    dfn = _ROLLUP[inst["_domain_fn"]]
    ofn = _ROLLUP[inst["_overall_fn"]]
    domains = []
    for d in inst["domains"]:
        domains.append({"id": d["id"], "judgment": dfn(d, domain_answers.get(d["id"], {}))})
    return {"domains": domains, "overall": ofn([d["judgment"] for d in domains])}


# ---------------------------------------------------------------------------
# GRADE — certainty of the body of evidence (OUTCOME level)
# ---------------------------------------------------------------------------
GRADE_DOWNGRADE = ["risk_of_bias", "inconsistency", "indirectness", "imprecision", "publication_bias"]
GRADE_UPGRADE = ["large_effect", "dose_response", "plausible_confounding"]
_GRADE_ORDER = ["Very low", "Low", "Moderate", "High"]


def grade_certainty(starting: str, downgrades: Dict[str, int], upgrades: Dict[str, int]) -> Dict[str, Any]:
    start_idx = 3 if str(starting).lower().startswith("random") else 1
    down = sum(max(-2, min(0, int(downgrades.get(k, 0)))) for k in GRADE_DOWNGRADE)
    up = 0
    if not str(starting).lower().startswith("random"):
        up = sum(max(0, min(2, int(upgrades.get(k, 0)))) for k in GRADE_UPGRADE)
    idx = max(0, min(3, start_idx + down + up))
    return {"certainty": _GRADE_ORDER[idx], "start": _GRADE_ORDER[start_idx], "net": down + up}


def public_instrument(inst: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": inst["id"], "name": inst["name"], "short_name": inst["short_name"],
        "axis": inst["axis"], "scale": inst["scale"], "reference": inst.get("reference", ""),
        "applies_to": inst.get("applies_to", []), "scaffold": inst.get("scaffold", False),
        "legacy": inst.get("legacy", False),
        "domains": [{"id": d["id"], "name": d["name"], "signals": [
            {"id": s["id"], "text": s["text"], "options": s["options"], "labels": s["labels"], "concern": s.get("concern", "no")}
            for s in d["signals"]]} for d in inst["domains"]],
    }
