"""Multi-persona LEADS: three expert personas each apply the LEADS prompt,
then we aggregate to a final decision.

Mimics the dual-screener convention in human SR practice — but at LLM scale
and with three distinct expert lenses on every paper.

Aggregation options (default: averaged score across personas, threshold ≥ 0):
  - "average":  mean LEADS score across personas; threshold ≥ 0 → INCLUDE
  - "majority": each persona makes a binary call; INCLUDE if ≥2/3 say so
  - "any":      INCLUDE if ANY persona says INCLUDE (highest recall)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

from .base import Paper, ScreeningArchitecture, ScreeningContext, ScreeningResult, invoke
from .leads_native import (
    LEADS_SCREENING_PROMPT, _build_pico_criteria, _parse_evaluations, _score, _stringify,
)


PERSONAS = [
    (
        "Methodologist",
        (
            "You are a senior clinical research methodologist with 20+ years of experience "
            "conducting systematic reviews and meta-analyses. You evaluate papers with "
            "particular attention to study design quality, randomization, allocation "
            "concealment, and risk of bias. You are conservative about marking criteria "
            "as YES unless the methodology is explicitly described as rigorous."
        ),
    ),
    (
        "Domain Specialist",
        (
            "You are a clinical specialist with deep expertise in anxiety-related disorders "
            "and cognitive-behavioral therapy. You evaluate papers by their clinical relevance: "
            "is the population truly representative of the disorder of interest, is the "
            "intervention faithful to evidence-based CBT, and are the outcomes clinically "
            "meaningful? You can recognize plausible inclusion even when the abstract is terse."
        ),
    ),
    (
        "Information Specialist",
        (
            "You are an information specialist trained in evidence synthesis methodology. "
            "You apply the screening criteria literally and consistently, calling UNCERTAIN "
            "whenever the abstract is silent on a criterion rather than guessing. Your role "
            "is to avoid premature exclusion based on missing information — when in doubt, "
            "preserve the paper for full-text review."
        ),
    ),
]


class LeadsMultiPersona(ScreeningArchitecture):
    """Three personas each apply the LEADS prompt; aggregate via averaged score."""

    name = "leads_multi_persona"
    threshold: float = 0.0       # avg score across personas; INCLUDE if >= threshold
    aggregation: str = "average"  # "average" | "majority" | "any"

    def screen(self, paper: Paper, ctx: ScreeningContext, model: Any) -> ScreeningResult:
        t0 = time.time()
        criteria, num_criteria = _build_pico_criteria(ctx.pico)
        prompt = LEADS_SCREENING_PROMPT.format(
            paper_content=f"Title: {paper.title}\n\nAbstract: {paper.abstract}",
            num_criteria=num_criteria,
            criteria_text=_stringify(criteria),
        )

        # Fire all 3 persona calls concurrently
        def _call(persona):
            name, system_msg = persona
            return name, invoke(model, prompt, system=system_msg)

        with ThreadPoolExecutor(max_workers=3) as ex:
            raw_by_persona = list(ex.map(_call, PERSONAS))

        # Parse each persona's PICO evaluations + score
        per_persona = []
        raws = []
        for name, raw in raw_by_persona:
            raws.append(raw)
            evals = _parse_evaluations(raw)
            if not evals:
                evals = [{"eligibility": "UNCERTAIN", "rationale": "Parse failure."} for _ in range(num_criteria)]
            score = _score(evals)
            per_persona.append({"name": name, "evals": evals, "score": score, "decision": 1 if score >= self.threshold else 0})

        # Aggregate
        if self.aggregation == "average":
            avg = sum(p["score"] for p in per_persona) / len(per_persona)
            prediction = 1 if avg >= self.threshold else 0
            confidence = (avg + 1) / 2
            agg_summary = f"avg={avg:+.2f}"
        elif self.aggregation == "majority":
            yes_votes = sum(p["decision"] for p in per_persona)
            prediction = 1 if yes_votes >= 2 else 0
            confidence = yes_votes / len(per_persona)
            agg_summary = f"votes={yes_votes}/{len(per_persona)}"
        elif self.aggregation == "any":
            prediction = 1 if any(p["decision"] == 1 for p in per_persona) else 0
            confidence = max(p["score"] for p in per_persona)
            confidence = (confidence + 1) / 2
            agg_summary = "any-include"
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Per-criterion majority vote across personas
        per_crit: dict[str, str] = {}
        for i, crit in enumerate(criteria):
            votes = []
            for p in per_persona:
                if i < len(p["evals"]):
                    v = str(p["evals"][i].get("eligibility", "UNCERTAIN")).upper()
                    votes.append(v)
            if votes:
                per_crit[crit] = max(set(votes), key=votes.count)

        reasoning = (
            f"LEADS multi-persona ({self.aggregation}, {agg_summary}). "
            + "; ".join(f"{p['name']}={p['score']:+.2f}" for p in per_persona)
        )

        return ScreeningResult(
            paper_id=paper.paper_id,
            prediction=prediction,
            confidence=confidence,
            reasoning=reasoning,
            per_criterion=per_crit,
            llm_calls=len(PERSONAS),
            wall_time_s=time.time() - t0,
            raw_outputs=raws,
        )
