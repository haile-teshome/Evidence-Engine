"""Reviewer → Critic → Adjudicator. Three roles, three calls, one decision."""

from __future__ import annotations

import time
from .base import Paper, ScreeningArchitecture, ScreeningContext, ScreeningResult, extract_json, normalize_decision, invoke


class MultiAgent(ScreeningArchitecture):
    name = "multi_agent"

    def screen(self, paper: Paper, ctx: ScreeningContext, model) -> ScreeningResult:
        t0 = time.time()
        raws: list[str] = []
        pico = ctx.pico

        # 1. Reviewer: makes an initial INCLUDE/EXCLUDE call with rationale
        reviewer_prompt = f"""You are the primary screener. Decide INCLUDE or EXCLUDE for this paper.

PICO: {pico}
Inclusion: {ctx.inclusion}
Exclusion: {ctx.exclusion}

PAPER
Title: {paper.title}
Abstract: {paper.abstract}

Return JSON: {{ "decision": "INCLUDE"|"EXCLUDE", "rationale": "<2-3 sentences>" }}
"""
        raw_r = invoke(model, reviewer_prompt, system="You are a thorough but pragmatic SR screener.")
        raws.append(raw_r)
        rev = extract_json(raw_r) or {}
        rev_decision = str(rev.get("decision", "EXCLUDE")).upper()
        rev_rationale = str(rev.get("rationale", ""))

        # 2. Critic: argues AGAINST the reviewer
        critic_prompt = f"""You are the critic. The reviewer said: {rev_decision} — "{rev_rationale}".
Steelman the OPPOSITE position based on the same paper. Point out anything the reviewer missed
or overweighted. If you cannot find good counter-evidence, say so plainly.

PAPER
Title: {paper.title}
Abstract: {paper.abstract}

Return JSON: {{ "counter_argument": "<your rebuttal>", "strength": "weak"|"moderate"|"strong" }}
"""
        raw_c = invoke(model, critic_prompt, system="You are an adversarial reviewer who challenges accepted conclusions.")
        raws.append(raw_c)
        crit = extract_json(raw_c) or {}
        counter = str(crit.get("counter_argument", ""))
        strength = str(crit.get("strength", "weak")).lower()

        # 3. Adjudicator: makes the final call given both views
        adj_prompt = f"""You are the adjudicator. Make the final screening decision given the reviewer's view and the critic's counter-argument.

REVIEWER ({rev_decision}): {rev_rationale}
CRITIC ({strength}): {counter}

PAPER
Title: {paper.title}
Abstract: {paper.abstract}

Return JSON: {{ "decision": "INCLUDE"|"EXCLUDE", "reason": "<one sentence>" }}
"""
        raw_a = invoke(model, adj_prompt, system="You weigh both sides and make a calibrated final call.")
        raws.append(raw_a)
        adj = extract_json(raw_a) or {}
        decision = normalize_decision(str(adj.get("decision", "")))

        return ScreeningResult(
            paper_id=paper.paper_id,
            prediction=decision,
            reasoning=f"Reviewer:{rev_decision} | Critic:{strength} | Adjudicator: {adj.get('reason','')}",
            llm_calls=3,
            wall_time_s=time.time() - t0,
            raw_outputs=raws,
        )
