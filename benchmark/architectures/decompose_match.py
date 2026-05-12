"""Two calls: (A) extract the paper's PICO + design from the abstract;
(B) match that structured extraction against the user's criteria."""

from __future__ import annotations

import time
from .base import Paper, ScreeningArchitecture, ScreeningContext, ScreeningResult, extract_json, normalize_decision, invoke


class DecomposeMatch(ScreeningArchitecture):
    name = "decompose_match"

    def screen(self, paper: Paper, ctx: ScreeningContext, model) -> ScreeningResult:
        t0 = time.time()
        raws: list[str] = []

        # Call A: extract the paper's PICO + study design
        extract_prompt = f"""Extract the study's PICO and design from this abstract.

PAPER
Title: {paper.title}
Abstract: {paper.abstract}

Return JSON with these keys (use null if not stated):
{{
  "population": "<who the study enrolled>",
  "intervention": "<what was given>",
  "comparator": "<what was compared>",
  "outcome": "<what was measured>",
  "design": "<RCT, cohort, case-control, review, etc>",
  "sample_size": "<n=... or null>"
}}
"""
        raw_a = invoke(model, extract_prompt)
        raws.append(raw_a)
        extracted = extract_json(raw_a) or {}

        # Call B: match extracted PICO against user's criteria
        match_prompt = f"""You are matching an extracted paper PICO against a systematic review's criteria.

USER'S CRITERIA
Population: {ctx.pico.get('population','')}
Intervention: {ctx.pico.get('intervention','')}
Comparator: {ctx.pico.get('comparator','')}
Outcome: {ctx.pico.get('outcome','')}

Inclusion:
{chr(10).join('- ' + c for c in ctx.inclusion)}

Exclusion:
{chr(10).join('- ' + c for c in ctx.exclusion)}

PAPER'S EXTRACTED PICO
{extracted}

Match each criterion. Return ONLY JSON:
{{
  "decision": "INCLUDE" or "EXCLUDE",
  "reason": "<one sentence>",
  "criteria": {{ "<criterion text>": "INCLUDE" | "EXCLUDE" | "UNCERTAIN" }}
}}
"""
        raw_b = invoke(model, match_prompt)
        raws.append(raw_b)
        data = extract_json(raw_b) or {}
        decision = normalize_decision(str(data.get("decision", "")))

        per_crit = {str(k): str(v).upper() for k, v in (data.get("criteria") or {}).items()}
        return ScreeningResult(
            paper_id=paper.paper_id,
            prediction=decision,
            reasoning=f"Extracted: {extracted}. Match: {data.get('reason','')}",
            per_criterion=per_crit,
            llm_calls=2,
            wall_time_s=time.time() - t0,
            raw_outputs=raws,
        )
