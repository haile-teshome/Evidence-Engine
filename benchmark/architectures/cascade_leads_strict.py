"""Two-stage cascade: LEADS-native first pass (high recall) → strict second
pass on the candidates that survived.

Designed to hit the SR sweet spot: keep nearly all true positives while
dramatically reducing false positives. Stage 1's role is "don't miss anything";
stage 2's role is "re-check borderline includes with strict criteria."

Final decision = INCLUDE iff stage 1 says INCLUDE AND stage 2 says INCLUDE.
Papers stage 1 already excludes get short-circuited (1 call instead of 2).
"""

from __future__ import annotations

import time
from typing import Any

from .base import Paper, ScreeningArchitecture, ScreeningContext, ScreeningResult
from .leads_native import LeadsNative
from .single_combined import SingleCombined


class CascadeLeadsStrict(ScreeningArchitecture):
    name = "cascade_leads_strict"

    def __init__(self) -> None:
        self.stage1 = LeadsNative()
        self.stage2 = SingleCombined()

    def screen(self, paper: Paper, ctx: ScreeningContext, model: Any) -> ScreeningResult:
        t0 = time.time()
        r1 = self.stage1.screen(paper, ctx, model)
        if r1.prediction == 0:
            # Stage 1 already excludes — accept and skip stage 2
            return ScreeningResult(
                paper_id=paper.paper_id,
                prediction=0,
                confidence=1 - r1.confidence,
                reasoning=f"Stage 1 (LEADS): EXCLUDE. {r1.reasoning}",
                per_criterion=r1.per_criterion,
                llm_calls=r1.llm_calls,
                wall_time_s=time.time() - t0,
                raw_outputs=r1.raw_outputs,
            )

        # Stage 1 said INCLUDE — re-screen with strict generic criteria
        r2 = self.stage2.screen(paper, ctx, model)
        prediction = 1 if (r1.prediction == 1 and r2.prediction == 1) else 0
        # Combined confidence: only confident-include if both agree
        confidence = r1.confidence * (r2.confidence if r2.prediction == 1 else 1 - r2.confidence)

        # Merge per-criterion votes (stage 2's named keys overwrite stage 1's)
        per_crit = dict(r1.per_criterion)
        per_crit.update(r2.per_criterion)

        return ScreeningResult(
            paper_id=paper.paper_id,
            prediction=prediction,
            confidence=confidence,
            reasoning=f"Stage 1 (LEADS): INCLUDE ({r1.reasoning[:100]}). Stage 2 (strict): "
                      f"{'INCLUDE' if r2.prediction == 1 else 'EXCLUDE'} — {r2.reasoning[:140]}",
            per_criterion=per_crit,
            llm_calls=r1.llm_calls + r2.llm_calls,
            wall_time_s=time.time() - t0,
            raw_outputs=r1.raw_outputs + r2.raw_outputs,
        )
