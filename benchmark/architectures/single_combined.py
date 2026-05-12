"""Baseline: one LLM call evaluates the decision and every criterion at once.

Mirrors the production app's current behavior so the benchmark has a fair
reference point.
"""

from __future__ import annotations

import time
from .base import Paper, ScreeningArchitecture, ScreeningContext, ScreeningResult, extract_json, normalize_decision, invoke


class SingleCombined(ScreeningArchitecture):
    name = "single_combined"

    def screen(self, paper: Paper, ctx: ScreeningContext, model) -> ScreeningResult:
        t0 = time.time()
        pico = ctx.pico
        prompt = f"""You are screening a paper for a systematic review.

PICO:
- Population: {pico.get('population','')}
- Intervention: {pico.get('intervention','')}
- Comparator: {pico.get('comparator','')}
- Outcome: {pico.get('outcome','')}

Inclusion criteria:
{chr(10).join('- ' + c for c in ctx.inclusion)}

Exclusion criteria:
{chr(10).join('- ' + c for c in ctx.exclusion)}

PAPER
Title: {paper.title}
Abstract: {paper.abstract}

Return ONLY a JSON object with exactly these keys:
{{
  "decision": "INCLUDE" or "EXCLUDE",
  "reason": "<one-sentence rationale>",
  "criteria": {{
    "<criterion text>": "INCLUDE" | "EXCLUDE" | "UNCERTAIN",
    ...
  }}
}}
"""
        raw = invoke(model, prompt)
        data = extract_json(raw) or {}
        decision = normalize_decision(str(data.get("decision", "")))
        per_crit = {}
        for k, v in (data.get("criteria") or {}).items():
            per_crit[str(k)] = str(v).upper()
        return ScreeningResult(
            paper_id=paper.paper_id,
            prediction=decision,
            reasoning=str(data.get("reason", "")),
            per_criterion=per_crit,
            llm_calls=1,
            wall_time_s=time.time() - t0,
            raw_outputs=[raw],
        )
