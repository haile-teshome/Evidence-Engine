"""Single LLM agent that can call deterministic tools to gather evidence."""

from __future__ import annotations

import re
import time
from .base import Paper, ScreeningArchitecture, ScreeningContext, ScreeningResult, extract_json, normalize_decision, invoke


_TOOLBOX_DESCRIPTION = """You have access to these tools. To call one, reply with a JSON object on a single line:
  {"tool": "<name>", "args": {...}}
Available tools:
- search_abstract(query: string) -> returns sentences mentioning the query (case-insensitive substring).
- pico_match(element: "population"|"intervention"|"comparator"|"outcome") -> returns the best-matching sentence for that PICO element.
- criterion_evidence(criterion: string) -> returns the best-matching sentence for a criterion.
When you have enough evidence, return a final JSON:
  {"final": true, "decision": "INCLUDE"|"EXCLUDE", "reason": "<one sentence>"}
"""


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def _best_match(query: str, text: str) -> str:
    toks = [t for t in re.split(r"\W+", (query or "").lower()) if len(t) > 3]
    if not toks:
        return ""
    best, best_score = "", 0
    for s in _sentences(text):
        lo = s.lower()
        score = sum(1 for t in toks if t in lo)
        if score > best_score:
            best, best_score = s, score
    return best


def _run_tool(call: dict, paper: Paper, ctx: ScreeningContext) -> str:
    tool = call.get("tool", "")
    args = call.get("args", {}) or {}
    text = paper.abstract or ""
    if tool == "search_abstract":
        q = args.get("query", "")
        hits = [s for s in _sentences(text) if q.lower() in s.lower()]
        return "\n".join(hits[:5]) or "(no matches)"
    if tool == "pico_match":
        el = args.get("element", "")
        v = ctx.pico.get(el, "")
        if not v:
            return f"(no PICO value for {el})"
        return _best_match(v, text) or "(no matching sentence)"
    if tool == "criterion_evidence":
        return _best_match(args.get("criterion", ""), text) or "(no matching sentence)"
    return f"(unknown tool: {tool})"


class SingleAgentTools(ScreeningArchitecture):
    name = "single_agent_tools"
    max_steps: int = 6

    def screen(self, paper: Paper, ctx: ScreeningContext, model) -> ScreeningResult:
        t0 = time.time()
        system = "You are a careful systematic-review screener that uses tools to verify claims before deciding."
        conversation = f"""{_TOOLBOX_DESCRIPTION}

PICO: {ctx.pico}
Inclusion criteria: {ctx.inclusion}
Exclusion criteria: {ctx.exclusion}

PAPER
Title: {paper.title}
Abstract: {paper.abstract}

Use the tools as needed, then return your final decision."""
        raws: list[str] = []
        calls = 0
        final_decision: int | None = None
        final_reason = ""
        transcript = conversation

        for step in range(self.max_steps):
            raw = invoke(model, transcript, system=system)
            raws.append(raw)
            calls += 1
            data = extract_json(raw)
            if isinstance(data, dict) and data.get("final"):
                final_decision = normalize_decision(str(data.get("decision", "")))
                final_reason = str(data.get("reason", ""))
                break
            if isinstance(data, dict) and "tool" in data:
                obs = _run_tool(data, paper, ctx)
                transcript += f"\n\nASSISTANT: {raw}\nTOOL_RESULT: {obs}\nContinue."
                continue
            # If model didn't emit a parseable tool/final, treat raw as the final.
            final_decision = normalize_decision(raw)
            final_reason = raw.strip()[:240]
            break

        if final_decision is None:
            final_decision = 0
            final_reason = "Did not converge within step budget; defaulting to exclude."

        return ScreeningResult(
            paper_id=paper.paper_id,
            prediction=final_decision,
            reasoning=final_reason,
            llm_calls=calls,
            wall_time_s=time.time() - t0,
            raw_outputs=raws,
        )
