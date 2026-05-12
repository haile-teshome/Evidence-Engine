"""LEADS-native architecture.

Implements the exact screening prompt + 4-way eligibility scoring from the
LEADS paper (Wang Zifeng et al., 2025; github.com/Keiji-AI/LEADS).

Key differences from our other architectures:
  - Evaluates per-PICO-element (Population, Intervention, Comparison, Outcome)
    rather than the user-supplied free-text inclusion/exclusion criteria.
  - 4-way verdict: YES / PARTIAL / NO / UNCERTAIN per criterion.
  - Explicit instruction to choose UNCERTAIN when info is missing
    (rather than defaulting to NO).
  - Aggregate score: YES=+1, PARTIAL=+0.5, UNCERTAIN=0, NO=-1, averaged.
    Include if score >= threshold (default 0.0).
"""

from __future__ import annotations

import json as _json
import re
import time
from typing import Any, List

from .base import Paper, ScreeningArchitecture, ScreeningContext, ScreeningResult, invoke


# Verbatim from the LEADS repo (leads/modules/screening.py).
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


PICO_KEY_MAP = {
    "P": ("Population", "population"),
    "I": ("Intervention", "intervention"),
    "C": ("Comparison", "comparator"),
    "O": ("Outcome", "outcome"),
}


def _build_pico_criteria(pico: dict) -> tuple[list[str], int]:
    criteria = [f"{display}: {pico.get(ours, '')}" for display, ours in PICO_KEY_MAP.values()]
    return criteria, len(criteria)


def _stringify(criteria: list[str]) -> str:
    return ". ".join(f"{i+1}: {c}" for i, c in enumerate(criteria))


def _score(evaluations: List[dict]) -> float:
    if not evaluations:
        return 0.0
    s = 0.0
    for ev in evaluations:
        e = str(ev.get("eligibility") or "").upper()
        if e == "YES": s += 1
        elif e == "PARTIAL": s += 0.5
        elif e == "UNCERTAIN": s += 0
        elif e == "NO": s -= 1
    return s / len(evaluations)


def _parse_evaluations(text: str) -> list[dict]:
    """Pull the LEADS evaluations array out of an LLM response, tolerantly."""
    if not text:
        return []
    # 1) Direct JSON
    try:
        data = _json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("evaluations"), list):
            return data["evaluations"]
    except Exception:
        pass
    # 2) ```json ... ``` block
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            data = _json.loads(m.group(1))
            if isinstance(data.get("evaluations"), list):
                return data["evaluations"]
        except Exception:
            pass
    # 3) Outermost { ... } block
    start, depth = text.find("{"), 0
    if start != -1:
        for i in range(start, len(text)):
            if text[i] == "{": depth += 1
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
    # 4) Regex-extract individual eligibility / rationale pairs
    out: list[dict] = []
    for m in re.finditer(r'"eligibility"\s*:\s*"([^"]+)"[\s,]*"rationale"\s*:\s*"([^"]*)"', text):
        out.append({"eligibility": m.group(1).upper(), "rationale": m.group(2)})
    return out


class LeadsNative(ScreeningArchitecture):
    """LEADS as designed: PICO-element eligibility, 4-way verdict, threshold-based decision."""

    name = "leads_native"
    threshold: float = 0.0  # score >= threshold => INCLUDE

    def screen(self, paper: Paper, ctx: ScreeningContext, model: Any) -> ScreeningResult:
        t0 = time.time()
        criteria, num_criteria = _build_pico_criteria(ctx.pico)
        prompt = LEADS_SCREENING_PROMPT.format(
            paper_content=f"Title: {paper.title}\n\nAbstract: {paper.abstract}",
            num_criteria=num_criteria,
            criteria_text=_stringify(criteria),
        )
        raw = invoke(model, prompt)
        evaluations = _parse_evaluations(raw)
        if not evaluations:
            # Match LEADS's fallback: all UNCERTAIN
            evaluations = [{"eligibility": "UNCERTAIN", "rationale": "Parse failure."} for _ in range(num_criteria)]

        score = _score(evaluations)
        prediction = 1 if score >= self.threshold else 0

        per_crit: dict[str, str] = {}
        for crit, ev in zip(criteria, evaluations[:num_criteria]):
            per_crit[crit] = str(ev.get("eligibility", "UNCERTAIN")).upper()

        reasoning = f"LEADS score={score:+.2f}. " + "; ".join(
            f"{crit.split(':')[0]}={ev.get('eligibility','?')}"
            for crit, ev in zip(criteria, evaluations[:num_criteria])
        )

        return ScreeningResult(
            paper_id=paper.paper_id,
            prediction=prediction,
            confidence=(score + 1) / 2,  # map [-1, 1] → [0, 1]
            reasoning=reasoning,
            per_criterion=per_crit,
            llm_calls=1,
            wall_time_s=time.time() - t0,
            raw_outputs=[raw],
        )
