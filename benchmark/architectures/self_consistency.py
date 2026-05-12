"""Default one-shot screening; re-sample 3× and majority-vote when hedging detected."""

from __future__ import annotations

import re
import time
from collections import Counter
from .base import Paper, ScreeningArchitecture, ScreeningContext, ScreeningResult
from .single_combined import SingleCombined


_HEDGE_RE = re.compile(
    r"\b(unclear|appears|possibly|might|may\s|likely|seems|suggests|insufficient|"
    r"lacks?\s+detail|not\s+specified|ambig|uncertain|hard\s+to\s+tell)\b",
    re.I,
)


def _is_uncertain(result: ScreeningResult) -> bool:
    if _HEDGE_RE.search(result.reasoning or ""):
        return True
    # If per_criterion has mixed INCLUDE/EXCLUDE/UNCERTAIN, that's uncertainty.
    vals = list(result.per_criterion.values())
    if vals and "UNCERTAIN" in vals:
        return True
    inc = sum(1 for v in vals if v == "INCLUDE")
    exc = sum(1 for v in vals if v in ("EXCLUDE", "EXCLUDE_VIOLATION"))
    if vals and abs(inc - exc) <= 1 and (inc + exc) > 0:
        return True
    return False


class SelfConsistency(ScreeningArchitecture):
    name = "self_consistency"

    def __init__(self, resample: int = 3) -> None:
        self._base = SingleCombined()
        self._resample = resample

    def screen(self, paper: Paper, ctx: ScreeningContext, model) -> ScreeningResult:
        t0 = time.time()
        first = self._base.screen(paper, ctx, model)
        if not _is_uncertain(first):
            first.wall_time_s = time.time() - t0
            return first

        # Re-sample and majority-vote
        votes: Counter[int] = Counter([first.prediction])
        raws = list(first.raw_outputs)
        calls = first.llm_calls
        for _ in range(self._resample):
            extra = self._base.screen(paper, ctx, model)
            votes[extra.prediction] += 1
            raws.extend(extra.raw_outputs)
            calls += extra.llm_calls
        prediction, _ = votes.most_common(1)[0]
        return ScreeningResult(
            paper_id=paper.paper_id,
            prediction=prediction,
            reasoning=f"Self-consistency vote {dict(votes)}: {first.reasoning}",
            per_criterion=first.per_criterion,
            llm_calls=calls,
            wall_time_s=time.time() - t0,
            raw_outputs=raws,
        )
