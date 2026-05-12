"""Cascade: cheap title-only triage first; thorough call only on YES/MAYBE."""

from __future__ import annotations

import time
from .base import Paper, ScreeningArchitecture, ScreeningContext, ScreeningResult, extract_json, normalize_decision, invoke
from .single_combined import SingleCombined


class CascadeTriage(ScreeningArchitecture):
    name = "cascade_triage"

    def __init__(self) -> None:
        self._deep = SingleCombined()

    def screen(self, paper: Paper, ctx: ScreeningContext, model) -> ScreeningResult:
        t0 = time.time()
        pico = ctx.pico
        first_sentence = (paper.abstract or "").split(". ", 1)[0][:200]
        triage_prompt = f"""You are doing a rapid first-pass triage for a systematic review.

Question topic — Population: {pico.get('population','')}; Intervention: {pico.get('intervention','')}; Outcome: {pico.get('outcome','')}

PAPER (title + 1 sentence only):
Title: {paper.title}
First sentence: {first_sentence}

Is this paper plausibly relevant to the question? Reply with ONE word: YES, MAYBE, or NO.
- NO only if the paper is clearly about a different topic.
- MAYBE if uncertain.
- YES if it appears topically relevant.
"""
        raw = invoke(model, triage_prompt)
        verdict = (raw or "").strip().upper().split()[0] if raw else "MAYBE"
        verdict = verdict if verdict in {"YES", "NO", "MAYBE"} else "MAYBE"

        if verdict == "NO":
            return ScreeningResult(
                paper_id=paper.paper_id,
                prediction=0,
                confidence=0.85,
                reasoning="Triage: not topically relevant.",
                llm_calls=1,
                wall_time_s=time.time() - t0,
                raw_outputs=[raw],
            )

        # YES / MAYBE: fall through to deep screening
        deep = self._deep.screen(paper, ctx, model)
        return ScreeningResult(
            paper_id=paper.paper_id,
            prediction=deep.prediction,
            confidence=deep.confidence,
            reasoning=f"Triage:{verdict}. {deep.reasoning}",
            per_criterion=deep.per_criterion,
            llm_calls=1 + deep.llm_calls,
            wall_time_s=time.time() - t0,
            raw_outputs=[raw] + deep.raw_outputs,
        )
