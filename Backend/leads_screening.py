"""LEADS-native screening for the production API.

Ports the highest-performing benchmark cell (leads_native × leads-mistral-7b at
threshold +0.20) into the Backend so screen-abstract requests get the same
PICO-element prompt + 4-way verdict + averaged-score decision that achieved
recall=1.000, specificity=0.676, MCC=+0.260 on van_Dis_2020.

Public surface:
  - LEADS_MODEL_NAME            : full Ollama GGUF tag
  - LEADS_SCORE_THRESHOLD       : +0.20 (operationalised sweet spot)
  - is_leads_model(name)        : alias / full-name detector
  - resolve_model_name(name)    : "leads" → full GGUF tag, else passthrough
  - screen_paper_leads(...)     : returns the same dict shape as AIService.screen_paper
"""

from __future__ import annotations

import json as _json
import os
import re
from typing import Any

# ---------------------------------------------------------------------------
# Model + threshold constants
# ---------------------------------------------------------------------------

LEADS_MODEL_NAME = os.getenv(
    "LEADS_MODEL_NAME",
    "hf.co/mradermacher/leads-mistral-7b-v1-GGUF:latest",
)

# Sweet-spot threshold from the benchmark (LEADS aggregate score >= +0.20 →
# INCLUDE). recall=1.000, specificity=0.676, MCC=+0.260, WSS@95=0.61 on
# van_Dis_2020 (288 stratified papers).
LEADS_SCORE_THRESHOLD: float = float(os.getenv("LEADS_SCORE_THRESHOLD", "0.20"))


def is_leads_model(name: str | None) -> bool:
    if not name:
        return False
    n = name.lower()
    return n == "leads" or "leads-mistral" in n or "leads_mistral" in n


def resolve_model_name(name: str | None) -> str | None:
    """Map the short 'leads' alias to the full GGUF tag for Ollama."""
    if name and name.lower() == "leads":
        return LEADS_MODEL_NAME
    return name


# Model used for tasks LEADS-Mistral wasn't fine-tuned on (PICO formulation,
# query generation, summarisation, refinement, etc.). LEADS shines at screening
# verdicts; for everything else we route to a general-purpose model.
NON_SCREENING_FALLBACK_MODEL = os.getenv(
    "NON_SCREENING_FALLBACK_MODEL",
    "qwen2.5:7b",
)


def resolve_for_thinking(name: str | None) -> str:
    """Return a model name appropriate for non-screening tasks.

    LEADS-Mistral-7b is fine-tuned exclusively for per-PICO screening verdicts.
    Calling it for PICO inference, MeSH generation, summaries, or refinements
    produces template-leaking output (e.g. the model echoes "# RESPONSE # You
    are required to output a JSON object…" instead of producing the JSON).

    If the caller selected a LEADS model, redirect to the configured
    general-purpose fallback. Otherwise pass through.
    """
    if is_leads_model(name):
        return NON_SCREENING_FALLBACK_MODEL
    return name or NON_SCREENING_FALLBACK_MODEL


# ---------------------------------------------------------------------------
# LEADS-native prompt (verbatim from the LEADS repo)
# ---------------------------------------------------------------------------

LEADS_SCREENING_PROMPT = """
# CONTEXT #
You are a clinical specialist tasked with assessing research papers for inclusion in a systematic literature review based on specific eligibility criteria.

# OBJECTIVE #
Evaluate each criterion of a given paper to determine its eligibility for inclusion in the review. Provide a list of decisions ("YES", "PARTIAL", "NO", or "UNCERTAIN") for each eligibility criterion. You must deliver exactly {num_criteria} responses.
1. YES: Meets the criteria.
2. PARTIAL: Partially meets the criteria but not completely.
3. NO: Does not meet the criteria.
4. UNCERTAIN: Uncertain if it meets the criteria.

# IMPORTANT NOTE #
If the information within the provided paper content is insufficient to conclusively evaluate a criterion, you must opt for "UNCERTAIN" as your response. Avoid making assumptions or extrapolating beyond the provided data, as accurate and reliable responses are crucial, and fabricating information (hallucinations) could lead to serious errors in the systematic review.
If the information is not applicable N/A, you also must opt for "UNCERTAIN".
Use "PARTIAL" when the paper meets some aspects of the criterion but not all; ensure that the partial fulfillment is based on the provided data and not on assumptions or incomplete information.

# PAPER DETAILS #
- Provided Paper: {paper_content}

# EVALUATION CRITERIA #
- Number of Criteria: {num_criteria}
- Criteria for Inclusion: {criteria_text}

# RESPONSE #
You are required to output a JSON object containing a list of decisions for each of the {num_criteria} eligibility criteria. Each decision should directly correspond to one of the criteria and be listed in the order they are presented. Ensure to use "UNCERTAIN" wherever the paper does not explicitly support a "YES", "PARTIAL", or "NO" decision.
The length of "evaluation" should be exactly {num_criteria}.
For example:
```json
{{
    "evaluations": [
        {{"eligibility": "YES", "rationale": "..."}},
        {{"eligibility": "PARTIAL", "rationale": "..."}},
        {{"eligibility": "NO", "rationale": "..."}},
        {{"eligibility": "UNCERTAIN", "rationale": "..."}}
    ]
}}
```
"""


PICO_ELEMENTS = [
    ("Population", "population"),
    ("Intervention", "intervention"),
    ("Comparison", "comparator"),
    ("Outcome", "outcome"),
]


# ---------------------------------------------------------------------------
# Parsing helpers (tolerant of LLM JSON drift)
# ---------------------------------------------------------------------------

def _parse_evaluations(text: str) -> list[dict]:
    if not text:
        return []
    # 1) Direct JSON
    try:
        data = _json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("evaluations"), list):
            return data["evaluations"]
    except Exception:
        pass
    # 2) ```json ... ``` fence
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            data = _json.loads(m.group(1))
            if isinstance(data.get("evaluations"), list):
                return data["evaluations"]
        except Exception:
            pass
    # 3) Outermost balanced { ... }
    start, depth = text.find("{"), 0
    if start != -1:
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = _json.loads(text[start:i + 1])
                        if isinstance(data.get("evaluations"), list):
                            return data["evaluations"]
                    except Exception:
                        break
                    break
    # 4) Regex-scrape individual eligibility/rationale pairs
    out: list[dict] = []
    for m in re.finditer(
        r'"eligibility"\s*:\s*"([^"]+)"[\s,]*"rationale"\s*:\s*"([^"]*)"', text,
    ):
        out.append({"eligibility": m.group(1).upper(), "rationale": m.group(2)})
    return out


def _score(evaluations: list[dict]) -> float:
    if not evaluations:
        return 0.0
    s = 0.0
    for ev in evaluations:
        e = str(ev.get("eligibility") or "").upper()
        if e == "YES":
            s += 1
        elif e == "PARTIAL":
            s += 0.5
        elif e == "UNCERTAIN":
            s += 0
        elif e == "NO":
            s -= 1
    return s / len(evaluations)


# ---------------------------------------------------------------------------
# Public entry point — mirrors AIService.screen_paper's return shape
# ---------------------------------------------------------------------------

def screen_paper_leads(paper: Any, pico: Any) -> dict[str, Any]:
    """LEADS-native screening on a single paper.

    Args:
        paper: object with `.title` and `.abstract`
        pico:  object with `.population`, `.intervention`, `.comparator`, `.outcome`

    Returns:
        A dict shaped like AIService.screen_paper's output:
          {
            "decision":   "Include" | "Exclude",
            "bucket":     str,
            "reason":     str,
            "<criterion>": "INCLUDE" | "EXCLUDE" | "UNCERTAIN" | "PARTIAL",
            ...
            "_leads_score": float,
            "_leads_threshold": float,
          }
    """
    # Local imports keep this file free of langchain at import-time for
    # callers that just need the model-detection helpers.
    from langchain_core.messages import HumanMessage
    from langchain_ollama import ChatOllama

    criteria_labels = [
        f"{display}: {getattr(pico, attr, '') or 'any'}"
        for display, attr in PICO_ELEMENTS
    ]
    num_criteria = len(criteria_labels)
    criteria_text = ". ".join(f"{i + 1}: {c}" for i, c in enumerate(criteria_labels))

    prompt = LEADS_SCREENING_PROMPT.format(
        paper_content=f"Title: {paper.title}\n\nAbstract: {paper.abstract}",
        num_criteria=num_criteria,
        criteria_text=criteria_text,
    )

    model = ChatOllama(
        model=LEADS_MODEL_NAME,
        temperature=0,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    raw_text = ""
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        raw_text = getattr(response, "content", "") or ""
    except Exception as e:
        return {
            "decision": "Exclude",
            "bucket": "LEADS call failed",
            "reason": f"LEADS-mistral via Ollama failed: {e}. "
                      "Make sure Ollama is running and the model is pulled: "
                      f"`ollama pull {LEADS_MODEL_NAME}`",
            "_leads_score": 0.0,
            "_leads_threshold": LEADS_SCORE_THRESHOLD,
        }

    evaluations = _parse_evaluations(raw_text)
    if not evaluations:
        # Match LEADS's own fallback: all UNCERTAIN, score = 0, defaults to exclude.
        evaluations = [
            {"eligibility": "UNCERTAIN", "rationale": "Parse failure."}
            for _ in range(num_criteria)
        ]

    score = _score(evaluations)
    include = score >= LEADS_SCORE_THRESHOLD

    # Build the per-criterion vote dict using PICO element names. The downstream
    # _normalize_abstract_decision iterates over the user's inclusion/exclusion
    # criteria — those won't appear here (LEADS evaluates PICO, not free-text
    # criteria), so the front-end "criteria met" column will show 0/0. We add
    # PICO entries under their own names so they surface in raw responses /
    # debugging tools.
    per_pico: dict[str, str] = {}
    pico_summary: list[str] = []
    for label, ev in zip(criteria_labels, evaluations):
        verdict = str(ev.get("eligibility", "UNCERTAIN")).upper()
        # Map YES/PARTIAL → INCLUDE, NO → EXCLUDE, UNCERTAIN → UNCERTAIN (for the
        # normalizer to render correctly).
        if verdict in {"YES", "PARTIAL"}:
            vote = "INCLUDE"
        elif verdict == "NO":
            vote = "EXCLUDE"
        else:
            vote = "UNCERTAIN"
        per_pico[label] = vote
        pico_summary.append(f"{label.split(':')[0]}={verdict}")

    reason = (
        f"LEADS score = {score:+.2f} (threshold {LEADS_SCORE_THRESHOLD:+.2f}). "
        + "; ".join(pico_summary)
    )

    out: dict[str, Any] = {
        "decision": "Include" if include else "Exclude",
        "bucket": "LEADS-native (mistral-7b) @ score "
                  f"≥ {LEADS_SCORE_THRESHOLD:+.2f}",
        "reason": reason,
        "_leads_score": score,
        "_leads_threshold": LEADS_SCORE_THRESHOLD,
    }
    out.update(per_pico)
    return out
