"""Shared shapes + base class every architecture implements."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScreeningContext:
    """Per-dataset context shared by all papers in a run."""
    pico: Dict[str, str]
    inclusion: List[str]
    exclusion: List[str]


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    label: Optional[int] = None  # 1 = include, 0 = exclude, None = unknown


@dataclass
class ScreeningResult:
    paper_id: str
    prediction: int                  # 1 = include, 0 = exclude
    confidence: float = 0.5          # 0..1, optional self-reported
    reasoning: str = ""
    per_criterion: Dict[str, str] = field(default_factory=dict)  # criterion -> "INCLUDE"/"EXCLUDE"/"UNCERTAIN"
    llm_calls: int = 0
    wall_time_s: float = 0.0
    raw_outputs: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers used by multiple architectures
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Optional[Any]:
    """Pull the first JSON object/array out of an LLM response, tolerantly."""
    if not text:
        return None
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")
    # Try direct
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # Find first {...} or [...] block
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break
    return None


def normalize_decision(s: str) -> int:
    if not s:
        return 0
    s = s.strip().lower()
    if s.startswith("inc") or s in {"yes", "true", "pass", "1"}:
        return 1
    return 0


def invoke(model, prompt: str, *, system: Optional[str] = None) -> str:
    """Wrap a langchain BaseChatModel.invoke and return text content."""
    from langchain_core.messages import HumanMessage, SystemMessage
    msgs = []
    if system:
        msgs.append(SystemMessage(content=system))
    msgs.append(HumanMessage(content=prompt))
    r = model.invoke(msgs)
    return (r.content or "") if hasattr(r, "content") else str(r)


class ScreeningArchitecture(ABC):
    """One architecture, called once per paper."""

    name: str = "base"

    @abstractmethod
    def screen(self, paper: Paper, ctx: ScreeningContext, model: Any) -> ScreeningResult:
        ...

    def _timed(self, fn):
        t0 = time.time()
        out = fn()
        return out, time.time() - t0
