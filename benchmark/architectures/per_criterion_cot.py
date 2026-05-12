"""One focused LLM call per inclusion/exclusion criterion, with chain-of-thought.

Per-criterion calls run in parallel via ThreadPoolExecutor — the underlying
LLM client must be thread-safe (langchain BaseChatModel implementations are).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

from .base import Paper, ScreeningArchitecture, ScreeningContext, ScreeningResult, extract_json, invoke


class PerCriterionCoT(ScreeningArchitecture):
    name = "per_criterion_cot"

    def __init__(self) -> None:
        # Fan-out for criterion-level calls. Stays bounded so we don't overload
        # the local Ollama context budget when many criteria run together.
        self.fanout = int(os.getenv("BENCH_CRITERIA_FANOUT", "4"))

    def _eval_one(self, paper: Paper, crit: str, is_inclusion: bool, model) -> tuple[str, str, str, str]:
        label = "INCLUSION" if is_inclusion else "EXCLUSION"
        ok_label = "INCLUDE" if is_inclusion else "NO_VIOLATION"
        fail_label = "EXCLUDE" if is_inclusion else "EXCLUDE_VIOLATION"
        prompt = f"""Evaluate ONE criterion against a paper. Think step-by-step before deciding.

CRITERION ({label}): {crit}

PAPER
Title: {paper.title}
Abstract: {paper.abstract}

Steps:
1. Identify the specific element this criterion is testing.
2. Quote the most relevant sentence (or note its absence) from the abstract.
3. Decide.

Return JSON exactly:
{{
  "thinking": "<step-by-step reasoning>",
  "evidence": "<quoted sentence or 'not stated'>",
  "decision": "{ok_label}" or "{fail_label}" or "UNCERTAIN"
}}
"""
        raw = invoke(model, prompt)
        data = extract_json(raw) or {}
        dec = str(data.get("decision", "")).upper()
        ev = str(data.get("evidence", ""))
        return crit, dec, ev, raw

    def screen(self, paper: Paper, ctx: ScreeningContext, model) -> ScreeningResult:
        t0 = time.time()
        per_crit: dict[str, str] = {}
        raw_outs: list[str] = []

        # Run all criteria (inclusion + exclusion) concurrently
        tasks = [(c, True) for c in ctx.inclusion] + [(c, False) for c in ctx.exclusion]
        with ThreadPoolExecutor(max_workers=max(1, self.fanout)) as ex:
            results = list(ex.map(lambda t: self._eval_one(paper, t[0], t[1], model), tasks))

        for crit, dec, _ev, raw in results:
            per_crit[crit] = dec
            raw_outs.append(raw)

        # Decision: every inclusion must be INCLUDE, no exclusion may be a violation
        passes_inclusion = all(per_crit.get(c) == "INCLUDE" for c in ctx.inclusion)
        exclusion_violated = any(
            per_crit.get(c) in ("EXCLUDE_VIOLATION", "EXCLUDE") for c in ctx.exclusion
        )
        prediction = 1 if (passes_inclusion and not exclusion_violated) else 0

        n_inc_met = sum(1 for c in ctx.inclusion if per_crit.get(c) == "INCLUDE")
        n_exc_viol = sum(1 for c in ctx.exclusion if per_crit.get(c) in ("EXCLUDE_VIOLATION", "EXCLUDE"))
        return ScreeningResult(
            paper_id=paper.paper_id,
            prediction=prediction,
            reasoning=f"{n_inc_met}/{len(ctx.inclusion)} inclusion met; {n_exc_viol} exclusion violations.",
            per_criterion=per_crit,
            llm_calls=len(tasks),
            wall_time_s=time.time() - t0,
            raw_outputs=raw_outs,
        )
