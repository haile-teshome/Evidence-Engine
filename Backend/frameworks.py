# ============================================================================
# FILE: frameworks.py
# Question-frame registry: PICO (intervention reviews) and PCC (JBI scoping
# reviews) as data, so the inference, clarifier, query-building, screening and
# writing code read the element list from here instead of hardcoding P/I/C/O.
# Adding another frame (SPIDER, PEO, …) is a new entry, not a code change.
# ============================================================================
from __future__ import annotations
import re
from typing import Dict, List

# Each element: id (canonical key), label (display noun), letter (badge),
# desc (one-line guidance used in prompts and tooltips).
FRAMEWORKS: Dict[str, dict] = {
    "pico": {
        "id": "pico",
        "label": "PICO",
        "review_type": "Intervention / effectiveness review",
        "elements": [
            {"id": "population", "label": "Population", "letter": "P",
             "desc": "the patients / participants — condition, age/sex, severity, setting"},
            {"id": "intervention", "label": "Intervention", "letter": "I",
             "desc": "the treatment, exposure, drug, procedure or device under study"},
            {"id": "comparator", "label": "Comparator", "letter": "C",
             "desc": "what the intervention is compared against (placebo, usual care, another arm)"},
            {"id": "outcome", "label": "Outcome", "letter": "O",
             "desc": "the measured endpoints and effects"},
        ],
        # search-query roles: anchors are the must-include concept blocks;
        # context elements become optional filter blocks.
        "anchor_ids": ["intervention", "outcome"],
        "context_ids": ["population"],
        # Screening roles: a FAIL on a DISCRIMINATING element excludes the paper.
        # Everything else is advisory — abstracts routinely omit the comparator and
        # report surrogate outcomes, so failing on those would exclude eligible work.
        "discriminating_ids": ["population", "intervention"],
        "formal_template": "In [population], does [intervention] compared to [comparator] affect [outcome]?",
    },
    "pcc": {
        "id": "pcc",
        "label": "PCC",
        "review_type": "Scoping review (JBI)",
        "elements": [
            {"id": "population", "label": "Population", "letter": "P",
             "desc": "the participants / groups of interest — may be broad in a scoping review"},
            {"id": "concept", "label": "Concept", "letter": "C",
             "desc": "the core concept examined — an intervention, phenomenon, or idea being mapped"},
            {"id": "context", "label": "Context", "letter": "C",
             "desc": "the setting / context — geography, setting, timeframe, or cultural factors"},
        ],
        "anchor_ids": ["concept"],
        "context_ids": ["population", "context"],
        # JBI scoping reviews are deliberately BROAD on population and context —
        # "adults 18-65" is a description of scope, not a gate, and excluding a
        # paper for studying "CKD patients" instead would be wrong. Only the
        # Concept (the thing being mapped) discriminates.
        "discriminating_ids": ["concept"],
        "formal_template": "This scoping review maps [concept] among [population] in [context].",
    },
}

DEFAULT_FRAMEWORK = "pico"


def normalize(name: str | None) -> str:
    n = (name or "").strip().lower()
    return n if n in FRAMEWORKS else DEFAULT_FRAMEWORK


def framework_of(name: str | None) -> dict:
    return FRAMEWORKS[normalize(name)]


def element_ids(name: str | None) -> List[str]:
    return [e["id"] for e in framework_of(name)["elements"]]


def element_defs(name: str | None) -> List[dict]:
    return framework_of(name)["elements"]


def discriminating_ids(name: str | None) -> List[str]:
    """Element ids whose FAIL vote is sufficient to exclude a paper at screening.

    Non-discriminating elements still get a vote (the reviewer sees it), but a
    FAIL there never drives the decision on its own — it flags, it does not cut.
    Falls back to the search anchors when a frame omits the key.
    """
    fw = framework_of(name)
    return list(fw.get("discriminating_ids") or fw.get("anchor_ids") or [])


def label_for(name: str | None, element_id: str) -> str:
    for e in framework_of(name)["elements"]:
        if e["id"] == element_id:
            return e["label"]
    return element_id.capitalize()


# ---- lightweight auto-detect (heuristic; the caller may override) -----------
_PCC_HINTS = re.compile(
    r"\b(scoping review|scope the literature|map (the |out )?(the )?(literature|evidence|research)|"
    r"extent (and|&) (nature|range)|what is known|breadth of|knowledge gaps?|"
    r"types of evidence|concept(s)? of|characteris(e|ing|ation)|landscape of)\b",
    re.IGNORECASE,
)
_PICO_HINTS = re.compile(
    r"\b(compared? (to|with)|versus|vs\.?|effect(iveness)? of|efficacy|randomi[sz]ed|"
    r"reduce|improve|increase|risk of|outcome|placebo|treatment of|does .* affect)\b",
    re.IGNORECASE,
)


def detect_framework(text: str) -> str:
    """Heuristic pre-selection of the likely frame from the goal text. The user
    can always override in the UI, so this only needs to be right-ish."""
    t = text or ""
    pcc = len(_PCC_HINTS.findall(t))
    pico = len(_PICO_HINTS.findall(t))
    if pcc > pico:
        return "pcc"
    return "pico"
