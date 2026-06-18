"""FastAPI HTTP layer for Evidence Engine.

Run:
    cd Backend
    cp .env.example .env   # fill in API keys
    pip install -r requirements.txt
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import os
import re
import math
import json as _json
import queue
import threading
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()

# IMPORTANT: install the headless shim BEFORE importing utils / data_services
from streamlit_shim import install as _install_shim, session_state as _ss

_install_shim()

import hashlib
import base64
import secrets
import urllib.parse as _urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse, HTMLResponse
from pydantic import BaseModel, Field

from config import Config
from models import Paper as BackendPaper, PICOCriteria, clean_markup
from utils import AIService, Deduplicator, AITableExtractor
from data_services import DataAggregator
from leads_screening import (
    LEADS_MODEL_NAME,
    LEADS_SCORE_THRESHOLD,
    is_leads_model,
    resolve_for_thinking,
    resolve_model_name,
    screen_paper_leads,
)

import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="Evidence Engine API", version="0.1.0")


# ---------------------------------------------------------------------------
# Server-side cancellation registry
# ---------------------------------------------------------------------------
# Long-running endpoints register a threading.Event under a task_id. When the
# client calls /api/tasks/cancel with that id, the event is set and the
# background worker checks it between iterations to bail out cleanly.

_cancel_events: Dict[str, threading.Event] = {}
_cancel_lock = threading.Lock()


class TaskCanceled(BaseException):
    """Raised inside a progress callback to abort an iterative LLM loop.

    Inherits from BaseException (not Exception) so legacy try/except
    Exception blocks inside the iterative functions don't swallow it.
    """


def _register_cancel(task_id: Optional[str]) -> Optional[threading.Event]:
    if not task_id:
        return None
    ev = threading.Event()
    with _cancel_lock:
        _cancel_events[task_id] = ev
    return ev


def _unregister_cancel(task_id: Optional[str]) -> None:
    if not task_id:
        return
    with _cancel_lock:
        _cancel_events.pop(task_id, None)


class CancelRequest(BaseModel):
    task_id: str


@app.post("/api/tasks/cancel")
def cancel_task(req: CancelRequest):
    """Signal a registered task to stop. Returns whether anything matched."""
    with _cancel_lock:
        ev = _cancel_events.get(req.task_id)
    if ev:
        ev.set()
        return {"canceled": True, "task_id": req.task_id}
    return {"canceled": False, "task_id": req.task_id, "reason": "not_found"}

_default_origins = [
    "http://localhost:5173",
    "http://localhost:4173",
    "http://127.0.0.1:5173",
]
_extra = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _default_model() -> str:
    """Default screening model.

    Order of precedence:
      1. `DEFAULT_MODEL` env var (explicit override)
      2. `Config.DEFAULT_MODEL` (legacy config.py value)
      3. `"leads"` — the highest-performing cell measured in the benchmark
         (LEADS-mistral-7b × LEADS-native @ threshold +0.20: recall=1.000,
         specificity=0.676, MCC=+0.260, WSS@95=0.61 on van_Dis_2020).

    Cloud-LLM keys (Claude/GPT) are not required for the default — LEADS runs
    locally on Ollama with no API key. Set `DEFAULT_MODEL=claude-sonnet-4-6`
    (or similar) in `.env` if you'd rather use a cloud model.
    """
    return os.getenv("DEFAULT_MODEL") or Config.DEFAULT_MODEL or "leads"


# ---------------------------------------------------------------------------
# Pydantic request/response models (mirror TypeScript shapes)
# ---------------------------------------------------------------------------


class PicoIn(BaseModel):
    population: str = ""
    intervention: str = ""
    comparator: str = ""
    outcome: str = ""


class PaperIn(BaseModel):
    id: str
    source: str = ""
    title: str = ""
    abstract: str = ""
    url: str = ""
    year: Optional[int] = None
    authors: Optional[str] = None


def _to_backend_paper(p: PaperIn | Dict[str, Any]) -> BackendPaper:
    if isinstance(p, dict):
        p = PaperIn(**p)
    return BackendPaper(
        source=p.source or "",
        id=p.id,
        title=p.title or "",
        abstract=p.abstract or "",
        url=p.url or "",
    )


def _to_pico(p: PicoIn) -> PICOCriteria:
    return PICOCriteria(
        population=p.population,
        intervention=p.intervention,
        comparator=p.comparator,
        outcome=p.outcome,
    )


def _paper_to_dict(p: BackendPaper) -> Dict[str, Any]:
    return {
        "id": str(p.id),
        "source": p.source,
        "title": p.title,
        "abstract": p.abstract,
        "url": p.url,
    }


# ---------------------------------------------------------------------------
# PICO / Strategy endpoints
# ---------------------------------------------------------------------------


class InferRequest(BaseModel):
    input: str
    model: Optional[str] = None
    previous_goal: Optional[str] = ""
    # Previous strategy to refine from (PICO + criteria), when this is a
    # follow-up message rather than a brand-new research goal.
    prior: Optional[Dict[str, Any]] = None


class Analysis(BaseModel):
    p: str
    i: str
    c: str
    o: str
    inclusion: List[str]
    exclusion: List[str]
    query: str


def _pico_value(v: Any) -> str:
    """Flatten whatever an LLM returned for a PICO field into a plain string.

    Smaller / instruction-following models sometimes return nested objects like
      {"specific_target_population": "adults with T2DM", "description": "..."}
    instead of a bare string. Join the values so downstream pydantic stays happy.
    """
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, dict):
        parts = [str(x).strip() for x in v.values() if x and isinstance(x, (str, int, float))]
        return "; ".join(parts) if parts else ""
    if isinstance(v, list):
        return "; ".join(p for p in (_pico_value(x) for x in v) if p)
    return str(v).strip()


def _coerce_str_list(v: Any) -> List[str]:
    """Inclusion / exclusion criteria may come back as a list of strings, a list
    of dicts, or even a single string. Normalise to List[str]."""
    if not v:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        out: List[str] = []
        for item in v:
            if isinstance(item, str):
                if item.strip():
                    out.append(item.strip())
            elif isinstance(item, dict):
                s = _pico_value(item)
                if s:
                    out.append(s)
            else:
                s = str(item).strip()
                if s:
                    out.append(s)
        return out
    return [_pico_value(v)] if _pico_value(v) else []


@app.post("/api/pico/infer", response_model=Analysis)
def pico_infer(req: InferRequest):
    model_name = resolve_for_thinking(req.model)
    data = AIService.infer_pico_and_query(req.input, model_name, req.previous_goal or "", prior=req.prior)
    p_str = _pico_value(data.get("p", ""))
    i_str = _pico_value(data.get("i", ""))
    c_str = _pico_value(data.get("c", ""))
    o_str = _pico_value(data.get("o", ""))
    pico = PICOCriteria(
        population=p_str,
        intervention=i_str,
        comparator=c_str,
        outcome=o_str,
    )
    try:
        query = AIService.generate_mesh_query(pico, model_name, goal=req.input or "")
    except Exception as e:
        print(f"[pico_infer] mesh query failed: {e}")
        query = ""
    return Analysis(
        p=p_str,
        i=i_str,
        c=c_str,
        o=o_str,
        inclusion=_coerce_str_list(data.get("inclusion")),
        exclusion=_coerce_str_list(data.get("exclusion")),
        query=query or "",
    )


class ClarifyQuestionsRequest(BaseModel):
    """Input to /api/pico/clarify-questions: the user's natural-language research
    goal, used to generate 1-3 multiple-choice questions that surface what the
    user did not specify (population focus, outcome scope, comparator, etc.)."""
    input: str
    model: Optional[str] = None


@app.post("/api/pico/clarify-questions")
def pico_clarify_questions(req: ClarifyQuestionsRequest):
    """Generate clarifying multiple-choice questions for an under-specified goal.

    Returns at most 3 questions. Each question has 4-5 option chips plus an
    implicit 'something else' free-form input on the frontend. The frontend
    shows a modal with one question at a time. The answers are then folded
    into the PICO before search runs, so the system stops silently inferring
    elements the user did not state.
    """
    from langchain_core.messages import HumanMessage

    goal = (req.input or "").strip()
    if not goal:
        return {"questions": []}

    model = AIService.get_model(resolve_for_thinking(req.model))
    if not model:
        return {"questions": []}

    prompt = f"""You are a clinical research methodologist helping a researcher refine a
systematic-review question BEFORE a literature search runs. The researcher typed the
goal below. Your job is to produce 1-3 short multiple-choice questions that will
let them disambiguate WITHOUT inventing details the system would otherwise have to
guess.

RESEARCH GOAL: "{goal}"

Generate clarifying questions ONLY for elements the researcher genuinely left ambiguous.
Skip questions that are already answered by what they wrote. The most common useful
questions are:
  • Population focus (general population vs. older adults vs. specific risk group)
  • Outcome scope (e.g. all-cause mortality vs. healthspan vs. specific biomarker)
  • Comparator (active comparator vs. placebo vs. usual care)
  • Time horizon / study-design preference

Each question must have 3-5 distinct options. Options should be short noun phrases
(3-12 words). The researcher will also see a free-text "something else" input on each
question, so make the options cover the COMMON cases — do not try to enumerate every
possibility.

Output ONLY a JSON object:
{{
  "questions": [
    {{
      "id": "population" | "intervention" | "comparator" | "outcome" | "design",
      "title": "Short question text ending in '?'",
      "options": [
        {{"id": "adults", "label": "Adults in the general population"}},
        {{"id": "elderly", "label": "Older adults (65+ years)"}},
        ...
      ]
    }},
    ...
  ]
}}

If the goal is already fully specified across population, intervention, comparator,
and outcome, return {{"questions": []}}.
"""
    try:
        r = model.invoke([HumanMessage(content=prompt)])
        data = AIService._extract_json(r.content) or {}
        raw_qs = data.get("questions") or []
        cleaned: List[Dict[str, Any]] = []
        for q in raw_qs[:3]:
            if not isinstance(q, dict):
                continue
            qid = str(q.get("id") or "").strip().lower() or f"q{len(cleaned)+1}"
            title = str(q.get("title") or "").strip()
            if not title:
                continue
            opts: List[Dict[str, str]] = []
            for o in (q.get("options") or [])[:5]:
                if isinstance(o, dict):
                    oid = str(o.get("id") or "").strip()
                    olabel = str(o.get("label") or "").strip()
                    if oid and olabel:
                        opts.append({"id": oid, "label": olabel})
                elif isinstance(o, str):
                    s = o.strip()
                    if s:
                        opts.append({"id": s.lower().replace(" ", "_")[:30], "label": s})
            if len(opts) >= 2:
                cleaned.append({"id": qid, "title": title, "options": opts})
        return {"questions": cleaned}
    except Exception as e:
        print(f"[clarify_questions] {e}")
        return {"questions": []}


class ClarifyNextRequest(BaseModel):
    goal: str
    pico_so_far: Dict[str, str] = Field(default_factory=dict)
    round: int = 0   # total questions answered so far — used for the safety cap
    asked: List[str] = Field(default_factory=list)  # PICO element ids already asked
    model: Optional[str] = None


@app.post("/api/pico/clarify-next")
def pico_clarify_next(req: ClarifyNextRequest):
    """Conversational PICO clarifier. Returns one question at a time with
    exactly 3 highly specific options, or { done: true } once all PICO
    elements are specific enough for a systematic review.

    Called repeatedly by the frontend after each answer until done."""
    from langchain_core.messages import HumanMessage

    goal = (req.goal or "").strip()
    if not goal:
        return {"done": True}

    model = AIService.get_model(resolve_for_thinking(req.model))
    if not model:
        return {"done": True}

    answered = {k: v for k, v in (req.pico_so_far or {}).items() if v and str(v).strip()}

    # At most ONE clarifying question per PICO element. Once all four have been
    # asked (or the safety cap is hit) we're done — no re-asking.
    PICO_IDS = ["population", "intervention", "comparator", "outcome"]
    asked = {str(a).strip().lower() for a in (req.asked or [])}
    remaining = [p for p in PICO_IDS if p not in asked]
    if req.round >= 4 or not remaining:
        return {"done": True}

    pico_lines = (
        "\n".join(f"  {k.upper()}: {v}" for k, v in answered.items())
        or "  (nothing yet)"
    )
    remaining_label = ", ".join(p.capitalize() for p in remaining)

    prompt = f"""You are a systematic-review librarian helping a researcher pin down their PICO
before a database search.

RESEARCHER'S GOAL: "{goal}"

PICO elements clarified so far:
{pico_lines}

TASK
────
Decide which PICO elements (Population, Intervention, Comparator, Outcome) still
NEED CLARIFICATION before a search. Only ask about elements that are NOT already
well-defined.

An element is ALREADY WELL-DEFINED (do NOT ask about it) when the goal names a
concrete, searchable concept for it — even if it is not maximally precise.
Reasonable specifics are fine and should be left alone, e.g. "older adults",
"type 2 diabetes", "mindfulness-based therapy", "pet ownership",
"depression symptoms". Do not push for extra precision on these.

An element NEEDS CLARIFICATION only when it is:
  • ABSENT — not stated in the goal and not reasonably implied; OR
  • UNSEARCHABLY GENERIC — a bare word with no domain, e.g. "treatment",
    "outcomes", "patients", "intervention" used on their own.

You may ONLY ask about these elements (the others were already asked — never
re-ask them): {remaining_label}.

Rules:
• Ask a question ONLY for an element that NEEDS CLARIFICATION by the test above.
  If the goal already specifies an element reasonably, treat it as defined and
  DO NOT ask about it.
• Ask AT MOST ONE question, about a single element from the allowed list.
• If every allowed element is already well-defined → return {{"done": true}}.
  Prefer {{"done": true}} whenever you are unsure — do not ask filler questions.
• Comparator is frequently left unspecified on purpose; only ask about it when
  the question clearly hinges on a specific comparison.
• When you do ask, pick the most important element that needs clarification
  (priority: Population > Intervention > Outcome > Comparator) and give EXACTLY
  3 concrete, measurable options relevant to "{goal}".
  GOOD options: "adults 18–65 with major depressive disorder (DSM-5)",
  "CBT ≥12 sessions", "remission at 8 weeks (PHQ-9 < 5)".

Return ONLY one of:

{{"done": true}}

{{
  "done": false,
  "question": {{
    "id": "population" | "intervention" | "comparator" | "outcome",
    "title": "<focused question ≤12 words ending in '?'>",
    "options": [
      {{"id": "a", "label": "<specific option 1>"}},
      {{"id": "b", "label": "<specific option 2>"}},
      {{"id": "c", "label": "<specific option 3>"}}
    ]
  }}
}}"""

    try:
        r = model.invoke([HumanMessage(content=prompt)])
        data = AIService._extract_json(r.content) or {}
        if data.get("done"):
            return {"done": True}
        q = data.get("question")
        if not isinstance(q, dict) or not q.get("title") or not q.get("options"):
            return {"done": True}
        # Enforce one-per-element: if the model picked an already-asked element
        # (or an unknown id), stop rather than loop.
        qid = str(q.get("id", "")).strip().lower()
        if qid in asked or qid not in PICO_IDS:
            return {"done": True}
        opts: List[Dict[str, str]] = []
        for o in (q.get("options") or [])[:3]:
            if isinstance(o, dict):
                oid = str(o.get("id") or "").strip() or f"opt{len(opts)+1}"
                olabel = str(o.get("label") or "").strip()
                if olabel:
                    opts.append({"id": oid, "label": olabel})
            elif isinstance(o, str) and o.strip():
                s = o.strip()
                opts.append({"id": s.lower().replace(" ", "_")[:30], "label": s})
        if len(opts) < 2:
            return {"done": True}
        return {
            "done": False,
            "question": {
                "id": str(q.get("id", "pico")).strip(),
                "title": str(q.get("title", "")).strip(),
                "options": opts[:3],
            },
        }
    except Exception as e:
        print(f"[clarify_next] {e}")
        return {"done": True}


class FormalQuestionRequest(BaseModel):
    pico: PicoIn
    model: Optional[str] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)


@app.post("/api/pico/formal-question")
def pico_formal_question(req: FormalQuestionRequest):
    q = AIService.generate_formal_question(_to_pico(req.pico), resolve_for_thinking(req.model), req.history)
    return {"question": q}


class SummaryRequest(BaseModel):
    goal: str
    papers: List[PaperIn] = Field(default_factory=list)
    model: Optional[str] = None


def _plain_summary(goal: str, papers: List[BackendPaper], model_name: str) -> str:
    """Plain-prose comprehensive evidence synthesis (no HTML, no markdown fences).

    The model is asked to produce something a researcher could read once and
    understand the topic well enough to ask follow-up questions, not a thin
    bullet list. It cites only the relevant subset of the provided literature.
    """
    if not papers:
        return ""
    from langchain_core.messages import HumanMessage

    model = AIService.get_model(model_name)
    if not model:
        return ""

    # All papers have already been LEADS-reranked, so every entry is relevant.
    # Cap at 50 to stay within context limits.
    subset = papers[:50]
    ctx = ""
    for idx, p in enumerate(subset):
        ctx += (
            f"[{idx + 1}] {p.title}\n"
            f"    Source: {p.source}\n"
            f"    Abstract: {(p.abstract or '')[:800]}\n\n"
        )

    prompt = f"""You are an expert evidence synthesist. Produce a COMPREHENSIVE plain-text briefing
on the research question — the kind of document a researcher could read once and walk away with a
working understanding of the topic, including what is known, what is contested, what is missing,
and what to ask next.

RESEARCH GOAL: {goal}

LITERATURE ({len(subset)} papers, numbered [1]-[{len(subset)}]):
{ctx}

Structure the response with exactly these section headers, each followed by a blank line, in this
order:

Research landscape overview
Arguments supporting the research question
Arguments against or challenging the research question
Mechanisms, effect sizes, and study characteristics
Open questions and follow-up considerations

REQUIREMENTS:
1. "Research landscape overview" — 1–2 paragraphs (≈ 4–8 sentences). Describe what the literature
   covers, what populations and settings have been studied, what study designs dominate, and where
   the evidence base is thin or fragmented. Cite the most representative papers inline.

2. "Arguments supporting the research question" — 4–7 substantive bullet points. Each bullet should
   make a SPECIFIC claim backed by at least one citation: name the mechanism, the effect size or
   direction, the population, and the study design where possible. Avoid generic statements.

3. "Arguments against or challenging the research question" — 3–6 substantive bullet points. Cover
   contradictory findings, null results, methodological limitations of supporting studies,
   confounders, or settings where the relationship breaks down. Cite specific evidence.

4. "Mechanisms, effect sizes, and study characteristics" — 1 paragraph (≈ 5–8 sentences) or 4–6
   bullets. Pull out concrete numbers where the abstracts supply them: sample sizes, follow-up
   durations, hazard ratios, percentages, p-values. Name the proposed biological / behavioural /
   methodological mechanisms when discussed.

5. "Open questions and follow-up considerations" — 3–5 specific questions a researcher might ask
   next based on gaps in the current literature. Phrase them as concrete refinements (e.g. "How
   does the effect change between Mediterranean diet adherence indices, and which index best
   predicts mortality?") rather than generic ones.

CITATION RULES:
  • Cite only papers that are actually relevant to the goal. If a paper is off-topic, ignore it
    completely — do not mention it, do not cite it.
  • Use inline citations like [3] or [5, 7]. Never invent a citation number not present in the
    provided literature.
  • Cite specific evidence — never write "[3] is relevant" without saying WHAT in [3] is relevant.

FAILURE MODE:
If FEWER THAN 3 of the provided papers are directly relevant to the goal, do not pad the response.
Write only the "Research landscape overview" section (1 paragraph) stating that the directly
relevant evidence base is thin, naming the closest-adjacent findings from the papers you do have,
and listing 3 ways to broaden or refocus the search. Leave the other sections out entirely.

FORMAT:
Plain text only. No HTML. No markdown bold/italics. No code fences. Dashes for bullets are fine.
Do NOT include a final reference list — the UI renders one separately.
"""
    try:
        r = model.invoke([HumanMessage(content=prompt)])
        text = (r.content or "").strip()
        # Even though the prompt forbids markdown, smaller models still emit
        # **bold** and *italic*. Strip the asterisks so the UI does not show
        # literal markup. (Underscore-italics are left alone to avoid breaking
        # legitimate text like "all-cause_mortality" tokens.)
        text = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", text)   # **bold** → bold
        text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)  # *italic* → italic
        # Collapse any remaining stray double-asterisks that didn't match.
        text = text.replace("**", "")
        return text
    except Exception as e:
        print(f"[plain_summary] {e}")
        return ""


_CITE_RE = re.compile(r"\[(\d+)\]")


def _strip_invalid_citations(summary: str, n_refs: int) -> str:
    """Remove citation markers that point outside the references range."""
    def _repl(m: "re.Match[str]") -> str:
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= n_refs else ""
    return _CITE_RE.sub(_repl, summary)


def _filter_to_cited(summary: str, papers: list) -> Tuple[str, list]:
    """Keep only papers that are actually cited in the summary text.

    Finds every [N] in the summary, discards papers that are never cited,
    and renumbers the remaining citations consecutively so the text stays
    consistent with the returned reference list.
    """
    # Collect 1-based indices that appear in the text (clamped to valid range).
    cited_indices: set = set()
    for m in _CITE_RE.finditer(summary):
        n = int(m.group(1))
        if 1 <= n <= len(papers):
            cited_indices.add(n)

    if not cited_indices:
        # LLM produced no valid citations — fall back to returning all papers.
        return summary, papers

    # Build old-index → new-index mapping (1-based, keeping cited order).
    old_to_new: dict = {}
    new_papers = []
    for old_idx in sorted(cited_indices):
        new_idx = len(new_papers) + 1
        old_to_new[old_idx] = new_idx
        new_papers.append(papers[old_idx - 1])

    # Rewrite citation numbers in the text.
    def _repl(m: "re.Match[str]") -> str:
        n = int(m.group(1))
        return f"[{old_to_new[n]}]" if n in old_to_new else ""

    new_summary = _CITE_RE.sub(_repl, summary)
    return new_summary, new_papers


@app.post("/api/pico/summary")
def pico_summary(req: SummaryRequest):
    bps = [_to_backend_paper(p) for p in req.papers]
    if not bps:
        return {"summary": "", "references": []}

    # Papers arriving here have already been LEADS-reranked and auto-cut by the
    # Home page. We do NOT additionally TF-IDF filter — everything that was
    # kept goes into the references list, even if the summary ultimately does
    # not cite every one of them.
    #
    # We DO reorder them so that papers from the same source are contiguous,
    # preserving rerank order (highest LEADS score first) within each source
    # group. The summary is generated against this grouped order, so the [N]
    # citation markers it emits line up with the source-grouped references the
    # UI displays. Source groups are ordered by first appearance in the
    # rerank-sorted list (so the source with the most relevant paper leads).
    grouped_papers: List[BackendPaper] = []
    by_source: Dict[str, List[BackendPaper]] = {}
    source_order: List[str] = []
    for p in bps:
        key = (p.source or "Other").strip() or "Other"
        if key not in by_source:
            by_source[key] = []
            source_order.append(key)
        by_source[key].append(p)
    for key in source_order:
        grouped_papers.extend(by_source[key])

    summary = _plain_summary(req.goal, grouped_papers, resolve_for_thinking(req.model))
    # Strip any hallucinated out-of-range citation numbers.
    summary = _strip_invalid_citations(summary, len(grouped_papers))
    # All grouped_papers passed the LEADS rerank — show them all as references.
    references = [
        {"title": (p.title or "").strip(), "url": p.url, "source": p.source, "id": str(p.id)}
        for p in grouped_papers
    ]
    return {"summary": summary, "references": references}


class RefinementRequest(BaseModel):
    goal: str
    papers: List[PaperIn] = Field(default_factory=list)
    model: Optional[str] = None


@app.post("/api/pico/suggestions")
def pico_suggestions(req: RefinementRequest):
    bps = [_to_backend_paper(p) for p in req.papers]
    suggs = AIService.get_refinement_suggestions(req.goal, bps, resolve_for_thinking(req.model))
    return {"suggestions": list(suggs or [])}


class AdversarialRequest(BaseModel):
    pico: PicoIn
    model: Optional[str] = None


@app.post("/api/pico/adversarial")
def pico_adversarial(req: AdversarialRequest):
    q = AIService.generate_adversarial_query(_to_pico(req.pico), resolve_for_thinking(req.model))
    return {"query": q}


class BrainstormRequest(BaseModel):
    goal: str = ""
    element: str  # "population" | "intervention" | "comparator" | "outcome"


@app.post("/api/pico/brainstorm")
def pico_brainstorm(req: BrainstormRequest):
    opts = AIService.get_pico_suggestion(req.goal, req.element)
    return {"suggestions": list(opts or [])}


class RefineRequest(BaseModel):
    pico: PicoIn
    goal: str = ""
    model: Optional[str] = None


class TitleRequest(BaseModel):
    goal: str
    model: Optional[str] = None


@app.post("/api/sessions/title")
def session_title(req: TitleRequest):
    """Generate a short 3-6 word title from a research goal (LLM-driven, with a
    string-slice fallback so the frontend always gets something usable)."""
    goal = (req.goal or "").strip()
    fallback = (goal[:50] + ("…" if len(goal) > 50 else "")) or "Untitled session"
    if not goal:
        return {"title": "Untitled session"}
    try:
        from langchain_core.messages import HumanMessage
        model = AIService.get_model(resolve_for_thinking(req.model))
        if not model:
            return {"title": fallback}
        prompt = (
            "Summarize this research goal in 3-6 words as a concise title. "
            "No quotes, no surrounding punctuation, no trailing period, no preamble. "
            "Return only the title text.\n\n"
            f"GOAL: {goal}\n\nTITLE:"
        )
        r = model.invoke([HumanMessage(content=prompt)])
        title = (r.content or "").strip()
        # Take only the first line and strip wrapping punctuation.
        title = title.split("\n")[0].strip().strip('"').strip("'").rstrip(".").strip()
        if not title or len(title) > 80:
            return {"title": fallback}
        return {"title": title}
    except Exception as e:
        print(f"[session_title] {e}")
        return {"title": fallback}


@app.post("/api/pico/refine")
def pico_refine(req: RefineRequest):
    """Surface ONE PICO field that the user should clarify or sharpen.

    Behaviour:
      • If any PICO field is blank, return a CLARIFYING QUESTION for the most
        important blank field. The response carries `is_clarification = True` and
        `suggested` holds a tentative starting value the user can accept / edit /
        replace via the Home-page popup.
      • If all PICO fields are filled but one is methodologically weak, fall back
        to the previous behaviour: propose a sharper replacement with
        `is_clarification = False`.
    """
    from langchain_core.messages import HumanMessage

    empty = {"field": None, "current": "", "suggested": "", "reason": "", "is_clarification": False}
    model = AIService.get_model(resolve_for_thinking(req.model))
    if not model:
        return {**empty, "reason": "Model unavailable."}

    # Prioritise blanks. Order matters: Population is the most load-bearing for
    # retrieval relevance, followed by Intervention, Outcome, Comparator.
    PRIORITY = ["population", "intervention", "outcome", "comparator"]
    values = {
        "population": (req.pico.population or "").strip(),
        "intervention": (req.pico.intervention or "").strip(),
        "comparator": (req.pico.comparator or "").strip(),
        "outcome": (req.pico.outcome or "").strip(),
    }
    blanks = [f for f in PRIORITY if not values[f]]

    if blanks:
        target = blanks[0]
        prompt = f"""You are a clinical research methodologist helping a researcher specify a
systematic-review PICO. The researcher's stated goal is below. They did NOT specify the
{target.upper()} element. Your job is to ask ONE concise clarifying question and offer ONE
plausible starting value the researcher can accept, edit, or reject.

RESEARCH GOAL: {req.goal or "(not provided)"}

CURRENT PICO (the blank field is the one we are asking about):
  Population: {values['population'] or '(blank)'}
  Intervention: {values['intervention'] or '(blank)'}
  Comparator: {values['comparator'] or '(blank)'}
  Outcome: {values['outcome'] or '(blank)'}

Rules for the clarifying question:
  • Phrase it as a question to the researcher, ≤ 18 words.
  • Reference only what the researcher actually wrote. Do NOT invent a different
    research topic.
  • The "suggested" starting value must be a reasonable default GIVEN the
    researcher's stated goal — but make clear it is one option among many.
  • The "suggested" must be 5–20 words.

Return ONLY a JSON object with these exact keys:
{{
  "field": "{target}",
  "current": "",
  "suggested": "<tentative starting value the researcher can accept or edit>",
  "reason": "<the clarifying question, ≤ 18 words, ending with '?'>"
}}
"""
        is_clarification = True
    else:
        prompt = f"""You are a clinical research methodologist reviewing a PICO breakdown for a
systematic review. Identify the ONE element that is most under-specified, ambiguous, or
methodologically weak, and propose a sharper replacement for that element only.

RESEARCH GOAL: {req.goal}

CURRENT PICO:
  Population: {values['population']}
  Intervention: {values['intervention']}
  Comparator: {values['comparator']}
  Outcome: {values['outcome']}

Pick the single weakest element and propose a concrete improvement. Be specific — name a
population subgroup, dose/duration, comparator type, or validated outcome measure. Do not
suggest changes to multiple elements; pick the most impactful one.

Return ONLY a JSON object with these exact keys:
{{
  "field": "population" | "intervention" | "comparator" | "outcome",
  "current": "<current value verbatim>",
  "suggested": "<sharper replacement, 5-20 words>",
  "reason": "<one-sentence rationale for why this change improves clarity or rigor>"
}}
"""
        is_clarification = False

    try:
        r = model.invoke([HumanMessage(content=prompt)])
        raw = (r.content or "").strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {**empty, "reason": "Could not parse model response."}
        data = _json.loads(m.group(0))
        field = str(data.get("field", "")).strip().lower()
        if field not in {"population", "intervention", "comparator", "outcome"}:
            return {**empty, "reason": "Model returned an invalid field."}
        return {
            "field": field,
            "current": str(data.get("current", "")).strip(),
            "suggested": str(data.get("suggested", "")).strip(),
            "reason": str(data.get("reason", "")).strip(),
            "is_clarification": is_clarification,
        }
    except Exception as e:
        print(f"[pico_refine] {e}")
        return {**empty, "reason": f"Refine error: {e}"}


# ---------------------------------------------------------------------------
# Elsevier institutional access — EZProxy + optional OAuth upgrade
# ---------------------------------------------------------------------------
# PRIMARY (no registration needed): EZProxy SSO
#   The user clicks "Connect via UCSF", a popup opens the library proxy login
#   page, they authenticate via MyAccess, and EZProxy redirects back here.
#   After that the browser holds a live EZProxy session cookie for
#   proxy.library.ucsf.edu, so all subsequent browser-side fetches to
#   *.proxy.library.ucsf.edu carry institutional access automatically.
#   The frontend makes Elsevier API calls directly from the browser through
#   the proxied URL and merges them with the backend results.
#
# OPTIONAL UPGRADE: Elsevier OAuth 2.0
#   If ELSEVIER_OAUTH_CLIENT_ID is set, the authorize/callback endpoints
#   below also handle PKCE-based OAuth so a Bearer token can be used on
#   backend-side requests instead.  Leave blank to use EZProxy only.

_UCSF_EZPROXY_LOGIN = "https://proxy.library.ucsf.edu/login"

_oauth_states: Dict[str, str] = {}  # state → code_verifier (for OAuth upgrade path)


@app.get("/api/auth/ezproxy/start")
def ezproxy_start():
    """Redirect the browser to the UCSF EZProxy login, targeting our callback."""
    callback = _urlparse.quote(f"{Config.APP_BASE_URL}/api/auth/ezproxy/callback", safe="")
    return RedirectResponse(url=f"{_UCSF_EZPROXY_LOGIN}?url={callback}")


@app.get("/api/auth/ezproxy/callback")
def ezproxy_callback():
    """EZProxy redirects here after MyAccess SSO.
    The browser now holds a live proxy.library.ucsf.edu session cookie.
    We just close the popup and tell the parent window we're connected.
    """
    return HTMLResponse("""<!DOCTYPE html>
<html><head><title>UCSF Library — Connected</title></head>
<body style="font-family:sans-serif;padding:2rem;background:#f8fafc">
  <h3 style="color:#16a34a">&#10003; Connected to UCSF Library</h3>
  <p style="color:#475569">You can close this window. Embase and Scopus are now accessible.</p>
  <script>
    window.opener?.postMessage({type: 'ezproxy_connected'}, '*');
    setTimeout(() => window.close(), 800);
  </script>
</body></html>""")


# Optional OAuth upgrade — only active when ELSEVIER_OAUTH_CLIENT_ID is set.

_ELSEVIER_AUTH_URL = "https://id.elsevier.com/as/authorization.oauth2"
_ELSEVIER_TOKEN_URL = "https://id.elsevier.com/as/token.oauth2"


@app.get("/api/auth/elsevier/status")
def elsevier_status():
    return {"oauth_configured": bool(Config.ELSEVIER_OAUTH_CLIENT_ID)}


@app.get("/api/auth/elsevier/authorize")
def elsevier_authorize():
    if not Config.ELSEVIER_OAUTH_CLIENT_ID:
        return HTMLResponse(
            "<p style='font-family:sans-serif;padding:2rem'>"
            "ELSEVIER_OAUTH_CLIENT_ID is not set — using EZProxy mode instead. "
            "This endpoint is only needed if you want backend-side Bearer-token access.</p>",
            status_code=503,
        )
    state = secrets.token_urlsafe(16)
    code_verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    _oauth_states[state] = code_verifier
    redirect_uri = f"{Config.APP_BASE_URL}/api/auth/elsevier/callback"
    params = {
        "client_id": Config.ELSEVIER_OAUTH_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return RedirectResponse(url=_ELSEVIER_AUTH_URL + "?" + _urlparse.urlencode(params))


@app.get("/api/auth/elsevier/callback")
def elsevier_callback(code: str = "", state: str = "", error: str = ""):
    if error:
        return HTMLResponse(
            f"<script>window.opener?.postMessage({{type:'elsevier_oauth',error:{_json.dumps(error)}}}, '*');window.close();</script>"
        )
    code_verifier = _oauth_states.pop(state, None)
    if not code_verifier:
        return HTMLResponse(
            "<script>window.opener?.postMessage({type:'elsevier_oauth',error:'invalid_state'}, '*');window.close();</script>",
            status_code=400,
        )
    redirect_uri = f"{Config.APP_BASE_URL}/api/auth/elsevier/callback"
    try:
        resp = requests.post(_ELSEVIER_TOKEN_URL, data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": Config.ELSEVIER_OAUTH_CLIENT_ID,
            "client_secret": Config.ELSEVIER_OAUTH_CLIENT_SECRET,
            "code_verifier": code_verifier,
        }, timeout=15)
        access_token = resp.json().get("access_token", "")
    except Exception as e:
        return HTMLResponse(
            f"<script>window.opener?.postMessage({{type:'elsevier_oauth',error:'token_exchange_failed'}}, '*');window.close();</script>",
            status_code=500,
        )
    if not access_token:
        return HTMLResponse(
            "<script>window.opener?.postMessage({type:'elsevier_oauth',error:'no_token'}, '*');window.close();</script>",
            status_code=400,
        )
    return HTMLResponse(f"""<!DOCTYPE html>
<html><head><title>Connected</title></head><body>
<p style="font-family:sans-serif;padding:2rem;color:#16a34a">&#10003; Connected to Elsevier</p>
<script>
  window.opener?.postMessage({{type: 'elsevier_oauth', token: {_json.dumps(access_token)}}}, '*');
  setTimeout(() => window.close(), 800);
</script></body></html>""")


# ---------------------------------------------------------------------------
# Search / data aggregation
# ---------------------------------------------------------------------------


class FetchAllRequest(BaseModel):
    query: str
    sources: List[str]
    max_per_source: int = 10
    limit: Optional[int] = None
    elsevier_token: str = ""


@app.post("/api/papers/fetch")
def papers_fetch(req: FetchAllRequest):
    papers, counts = DataAggregator.fetch_all(
        req.query, req.sources, max_per_source=req.max_per_source,
        uploaded_files=None, limit=req.limit, elsevier_token=req.elsevier_token,
    )
    return {
        "papers": [_paper_to_dict(p) for p in papers],
        "sourceCounts": counts,
    }


class SimulateYieldRequest(BaseModel):
    query: str
    sources: List[str]
    elsevier_token: str = ""


@app.post("/api/simulation/yield")
def simulation_yield(req: SimulateYieldRequest):
    counts = DataAggregator.simulate_yield(req.query, req.sources, elsevier_token=req.elsevier_token)
    return {"counts": counts}


class DedupeRequest(BaseModel):
    papers: List[PaperIn]


@app.post("/api/papers/dedupe")
def papers_dedupe(req: DedupeRequest):
    bps = [_to_backend_paper(p) for p in req.papers]
    unique, dups = Deduplicator.run(bps)
    return {
        "unique": [_paper_to_dict(p) for p in unique],
        "duplicates": [_paper_to_dict(p) for p in dups],
    }


# ---------------------------------------------------------------------------
# Screening (abstract + full-text)
# ---------------------------------------------------------------------------


class ScreenAbstractRequest(BaseModel):
    paper: PaperIn
    pico: PicoIn
    inclusion: List[str] = Field(default_factory=list)
    exclusion: List[str] = Field(default_factory=list)
    model: Optional[str] = None


_PICO_ASSESS_VOTES = {"PASS", "PARTIAL", "FAIL", "NA"}


def _pico_assess(paper: PaperIn, pico: PicoIn, model_name: str) -> Dict[str, Any]:
    """Run a single LLM call that returns per-PICO appraisal with evidence quotes.

    Output shape:
      {
        "population":    { "vote": "PASS"|"PARTIAL"|"FAIL"|"NA", "evidence": "<short quote>", "reasoning": "<one sentence>" },
        "intervention":  { ... },
        "comparator":    { ... },
        "outcome":       { ... },
        "overall_reasoning": "<2-3 sentence synthesis across PICO>",
      }

    The "evidence" string must be a short verbatim snippet from the abstract
    (best-effort; we re-anchor it against the abstract afterwards). A vote of
    "NA" means the abstract does not give enough information to judge that
    element.
    """
    from langchain_core.messages import HumanMessage

    abstract = (paper.abstract or "").strip()
    title = paper.title or "(untitled)"

    model = AIService.get_model(model_name)
    empty: Dict[str, Any] = {
        "population":   {"vote": "NA", "evidence": "", "reasoning": ""},
        "intervention": {"vote": "NA", "evidence": "", "reasoning": ""},
        "comparator":   {"vote": "NA", "evidence": "", "reasoning": ""},
        "outcome":      {"vote": "NA", "evidence": "", "reasoning": ""},
        "overall_reasoning": "",
    }
    if not model:
        return empty

    has_abstract = bool(abstract)
    abstract_block = abstract[:6000] if has_abstract else (
        "(NO ABSTRACT AVAILABLE — judge ONLY from the paper title above. "
        "Infer the likely population/intervention/outcome from the title, do NOT "
        "fabricate quotes, and explain your best-effort judgement.)"
    )

    prompt = f"""You are screening a paper against a PICO frame for a systematic review.
For EACH of the four PICO elements below, decide a vote of PASS, PARTIAL, or FAIL.
Never use "NA" or "UNCERTAIN" — pick the closest of the three labels:

  • PASS    — the title/abstract clearly satisfies this element (explicit match).
  • PARTIAL — it relates to this element but the match is implicit, broader,
              narrower, or otherwise "on par but not explicit" (e.g. broader
              population, surrogate outcome, related setting). USE THIS
              GENEROUSLY when the text touches the concept at all, and when you
              are inferring from a title alone.
  • FAIL    — the text addresses this element AND the match is clearly wrong, OR
              makes no mention whatsoever of anything relevant to this element.

For every vote also return:
  • evidence: a SHORT verbatim phrase or sentence copied directly from the
    abstract (≤ 200 characters) that best supports your vote. If there is no
    abstract, return an empty string for evidence — never invent a quote.
  • reasoning: one sentence, SPECIFIC to THIS paper — name what the paper
    actually studied (its real population/intervention/outcome as stated in the
    title or abstract) and why that earns the vote. Never leave it blank and
    never write a generic template sentence.

ALWAYS write a 2-3 sentence "overall_reasoning" that (a) names in one clause what
THIS paper is actually about (use the title when there's no abstract), then
(b) states the specific PICO element(s) it matches or fails and why, referencing
concrete terms from the title/abstract. It must NEVER be blank. Do NOT write a
generic sentence such as "the paper does not address any of the PICO elements"
or "it lacks specificity regarding validated outcomes" — always ground it in
this study's actual topic. When only a title is available, say what the title
implies and flag what remains uncertain pending full text.

PICO:
  Population:   {pico.population or '(unspecified)'}
  Intervention: {pico.intervention or '(unspecified)'}
  Comparator:   {pico.comparator or '(unspecified)'}
  Outcome:      {pico.outcome or '(unspecified)'}

PAPER TITLE: {title}

PAPER ABSTRACT:
{abstract_block}

Return ONLY a JSON object with EXACTLY this shape:
{{
  "population":    {{ "vote": "...", "evidence": "...", "reasoning": "..." }},
  "intervention":  {{ "vote": "...", "evidence": "...", "reasoning": "..." }},
  "comparator":    {{ "vote": "...", "evidence": "...", "reasoning": "..." }},
  "outcome":       {{ "vote": "...", "evidence": "...", "reasoning": "..." }},
  "overall_reasoning": "..."
}}

NEVER fabricate a quote that does not appear in the abstract.
"""

    try:
        r = model.invoke([HumanMessage(content=prompt)])
        data = AIService._extract_json(r.content) or {}
    except Exception as e:
        print(f"[pico_assess] LLM error: {e}")
        return empty

    # Pre-compute a normalised abstract + tokenised sentences for fast anchoring.
    _norm_abs = re.sub(r"\s+", " ", abstract).strip().lower()
    _abs_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if s.strip()]

    def _best_sentence(seed_text: str, min_overlap: int = 1) -> str:
        """Return the abstract sentence that shares the most ≥3-char tokens
        with `seed_text`. Returns "" if no sentence has at least `min_overlap`
        shared tokens."""
        toks = {t for t in re.findall(r"\w{3,}", seed_text.lower())}
        if not toks or not _abs_sentences:
            return ""
        best, best_score = "", 0
        for sentence in _abs_sentences:
            slo = sentence.lower()
            score = sum(1 for t in toks if t in slo)
            if score > best_score:
                best, best_score = sentence, score
        return best if best_score >= min_overlap else ""

    def _clean_field(raw: Any, pico_seed: str) -> Dict[str, str]:
        if not isinstance(raw, dict):
            # Abstract present but model returned nothing — try a PICO-keyword
            # rescue. If even that finds something, mark PARTIAL; otherwise NA.
            rescued = _best_sentence(pico_seed, min_overlap=1)[:240]
            if rescued:
                return {
                    "vote": "PARTIAL",
                    "evidence": rescued,
                    "reasoning": "Closest abstract sentence by keyword match — model did not return a structured response.",
                }
            return {"vote": "NA", "evidence": "", "reasoning": ""}

        vote = str(raw.get("vote") or "").strip().upper()
        # Normalise loose / legacy variants.
        if vote in {"YES", "INCLUDE", "TRUE"}: vote = "PASS"
        elif vote in {"NO", "EXCLUDE", "FALSE"}: vote = "FAIL"
        elif vote in {"PARTIAL"}: pass
        elif vote in {"UNCERTAIN", "UNKNOWN", "NA", ""}:
            # The new prompt forbids NA — but the model still emits it
            # sometimes. Treat NA / UNCERTAIN as "could be partial" and rely
            # on the quote-anchoring below to decide PARTIAL vs NA.
            vote = "NA"
        elif vote not in {"PASS", "FAIL"}:
            vote = "NA"

        raw_evidence = str(raw.get("evidence") or "").strip().strip('"').strip("'")
        evidence = ""

        # Tier 1: direct (whitespace-normalised) substring match.
        if raw_evidence:
            _norm_q = re.sub(r"\s+", " ", raw_evidence).strip().lower()
            if _norm_q and _norm_q in _norm_abs:
                evidence = raw_evidence[:240]
            else:
                # Tier 2: best sentence by token overlap with the model's quote.
                evidence = _best_sentence(raw_evidence, min_overlap=1)[:240]
        # Tier 3: PICO-keyword rescue — find the best sentence using the PICO
        # element text itself. Catches the case where the model emitted no
        # quote OR an unusable quote, but the abstract clearly addresses the
        # element through related vocabulary.
        if not evidence:
            evidence = _best_sentence(pico_seed, min_overlap=1)[:240]

        reasoning = str(raw.get("reasoning") or "").strip()[:300]

        # Reclassification rules (no more silent NA when the abstract exists):
        #   • If the model said NA but we DID anchor a quote → PARTIAL.
        #     This is the "on par but not explicit" case.
        #   • If the model said NA and we anchored nothing → keep NA.
        #     (Abstract genuinely doesn't address this element.)
        #   • If the model said PASS/PARTIAL/FAIL but we couldn't anchor a
        #     quote at all → downgrade to NA (we promised the user that every
        #     non-NA chip has a quote).
        if vote == "NA" and evidence:
            vote = "PARTIAL"
            if not reasoning:
                reasoning = "Abstract content relates to this PICO element but does not match explicitly."
        elif vote == "NA":
            reasoning = ""
        if vote != "NA" and not evidence:
            vote = "NA"
            reasoning = ""

        return {"vote": vote, "evidence": evidence, "reasoning": reasoning}

    population   = _clean_field(data.get("population"),   pico.population   or "")
    intervention = _clean_field(data.get("intervention"), pico.intervention or "")
    comparator   = _clean_field(data.get("comparator"),   pico.comparator   or "")
    outcome      = _clean_field(data.get("outcome"),      pico.outcome      or "")

    # Never leave the reason blank — fall back to a record-specific, best-effort
    # sentence grounded in the title (the PICO chips may still be NA when no
    # quote could be anchored, but the reviewer always gets a "why").
    overall_reasoning = str(data.get("overall_reasoning") or "").strip()[:800]
    if not overall_reasoning:
        t = title if title and title != "(untitled)" else "this record"
        overall_reasoning = (
            f'Based only on the title ("{title[:140]}"), a confident PICO match could not '
            f"be confirmed without an abstract — assess {t} at full text."
            if not has_abstract else
            f'The abstract for "{title[:140]}" could not be mapped to the PICO frame with '
            f"confidence — assess at full text."
        )

    return {
        "population":   population,
        "intervention": intervention,
        "comparator":   comparator,
        "outcome":      outcome,
        "overall_reasoning": overall_reasoning,
    }


def _normalize_abstract_decision(
    raw: Dict[str, Any],
    inclusion: List[str],
    exclusion: List[str],
    paper: PaperIn,
    pico: Optional[PicoIn] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    decision = str(raw.get("decision", "Exclude")).strip().lower()
    decision_upper = "INCLUDE" if decision.startswith("inc") else "EXCLUDE"
    reason = str(raw.get("reason") or raw.get("bucket") or "")
    agent_trace: Dict[str, Dict[str, str]] = {}
    abstract = paper.abstract or ""

    def _evidence_for(criterion: str) -> str:
        toks = [t for t in re.split(r"\W+", criterion.lower()) if len(t) > 3]
        if not toks or not abstract:
            return abstract[:200]
        sentences = re.split(r"(?<=[.!?])\s+", abstract)
        best, best_score = "", 0
        for s in sentences:
            lo = s.lower()
            score = sum(1 for t in toks if t in lo)
            if score > best_score:
                best, best_score = s, score
        return best or abstract[:200]

    all_criteria = list(inclusion) + list(exclusion)
    for crit in all_criteria:
        v = raw.get(crit)
        if isinstance(v, str):
            vu = v.strip().upper()
            vote = "PASS" if vu in {"INCLUDE", "PASS", "YES", "TRUE"} else (
                "FAIL" if vu in {"EXCLUDE", "FAIL", "NO", "FALSE"} else "N/A"
            )
        else:
            vote = "N/A"
        agent_trace[crit] = {
            "vote": vote,
            "reasoning": f"Criterion evaluation: {vote}",
            "evidence": _evidence_for(crit),
        }

    # Per-PICO structured assessment with evidence quotes. The "overall_reasoning"
    # synthesises across population/intervention/comparator/outcome and becomes
    # the new Reason string shown in the screening table.
    pico_assessment: Dict[str, Any] = {
        "population":   {"vote": "NA", "evidence": "", "reasoning": ""},
        "intervention": {"vote": "NA", "evidence": "", "reasoning": ""},
        "comparator":   {"vote": "NA", "evidence": "", "reasoning": ""},
        "outcome":      {"vote": "NA", "evidence": "", "reasoning": ""},
        "overall_reasoning": "",
    }
    if pico is not None and model_name:
        try:
            pico_assessment = _pico_assess(paper, pico, model_name)
        except Exception as e:
            print(f"[normalize_abstract] pico_assess failed: {e}")

    # _pico_assess always returns a record-specific, non-blank overall_reasoning
    # (it judges from the title when no abstract is present), so use it directly.
    # Only fall back to the screener's own reason, then a minimal default.
    overall_reasoning = (
        str(pico_assessment.get("overall_reasoning") or "").strip()
        or reason.strip()
        or ("Meets inclusion criteria" if decision_upper == "INCLUDE" else "Excluded")
    )

    return {
        "paper_id": paper.id,
        "Source": paper.source,
        "Title": paper.title,
        "URL": paper.url,
        "Abstract": abstract,
        "Decision": decision_upper,
        "Reason": overall_reasoning,
        "Agent_Trace": agent_trace,
        "Pico_Assessment": pico_assessment,
    }


def _screen_one(paper: BackendPaper, pico: PICOCriteria, model_name: str,
                inclusion: List[str], exclusion: List[str]) -> Dict[str, Any]:
    """Route a single paper to LEADS or to the generic screener depending on model."""
    if is_leads_model(model_name):
        return screen_paper_leads(paper, pico)
    return AIService.screen_paper(paper, pico, model_name, inclusion, exclusion)


@app.post("/api/screen/abstract")
def screen_abstract(req: ScreenAbstractRequest):
    paper = _to_backend_paper(req.paper)
    pico = _to_pico(req.pico)
    # Make criteria available to legacy functions that read session_state
    _ss["inclusion_list"] = list(req.inclusion or [])
    _ss["exclusion_list"] = list(req.exclusion or [])
    model_name = resolve_model_name(req.model) or resolve_model_name(_default_model())
    raw = _screen_one(paper, pico, model_name, req.inclusion, req.exclusion)
    # Per-PICO assessment uses the reasoning-tier model regardless of screening
    # model, because structured JSON output is its strength.
    pico_model = resolve_for_thinking(req.model)
    return _normalize_abstract_decision(
        raw, req.inclusion, req.exclusion, req.paper, pico=req.pico, model_name=pico_model,
    )


class ScreenAbstractBatchRequest(BaseModel):
    papers: List[PaperIn]
    pico: PicoIn
    inclusion: List[str] = Field(default_factory=list)
    exclusion: List[str] = Field(default_factory=list)
    model: Optional[str] = None


@app.post("/api/screen/abstract-batch")
def screen_abstract_batch(req: ScreenAbstractBatchRequest):
    _ss["inclusion_list"] = list(req.inclusion or [])
    _ss["exclusion_list"] = list(req.exclusion or [])
    pico = _to_pico(req.pico)
    model_name = resolve_model_name(req.model) or resolve_model_name(_default_model())

    pico_model = resolve_for_thinking(req.model)

    def _one(p_in: PaperIn) -> Dict[str, Any]:
        bp = _to_backend_paper(p_in)
        raw = _screen_one(bp, pico, model_name, req.inclusion, req.exclusion)
        return _normalize_abstract_decision(
            raw, req.inclusion, req.exclusion, p_in, pico=req.pico, model_name=pico_model,
        )

    results: List[Dict[str, Any]] = []
    workers = max(1, min(Config.PARALLEL_SCREENING_WORKERS, len(req.papers) or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_one, p) for p in req.papers]):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[screen_abstract_batch] worker error: {e}")
    return {"results": results}


def _auto_relevance_cutoff(scores: List[float]) -> Tuple[float, str]:
    """Pick a relevance floor from the score distribution itself.

    Returns (effective_floor, human_readable_reason).

    Rules:
      • Hard floor at +0.0 — LEADS aggregate < 0 means net-negative across PICO.
        Never include these.
      • If ≤ MIN_KEPT papers pass the hard floor, keep them all (corpus too
        small to detect a natural break).
      • Otherwise, sort the positive scores descending. Look for the largest
        gap between consecutive scores, but only count gaps that leave at
        least MIN_KEPT papers above them — this prevents a single high-scoring
        outlier from collapsing the kept set to one paper.
      • If a gap ≥ GAP_THRESHOLD exists in the eligible range, cut there.
      • Otherwise, keep the top half of positive scores with a soft floor of
        +0.10 (suppresses borderline papers when the distribution is uniformly
        mediocre).

    No max cap — if many papers clear the natural break, all of them stay.
    """
    MIN_KEPT = 20        # always keep at least this many papers
    GAP_THRESHOLD = 0.20  # gap must be this large to count as a natural break
    HARD_FLOOR = 0.0     # net-negative LEADS score → exclude

    if not scores:
        return HARD_FLOOR, "empty corpus"

    # Work from all papers above the hard floor, sorted best-first.
    eligible = sorted([s for s in scores if s >= HARD_FLOOR], reverse=True)
    if not eligible:
        return HARD_FLOOR, "no papers scored net-positive across PICO"

    # If fewer eligible papers than minimum, keep them all.
    if len(eligible) <= MIN_KEPT:
        return min(eligible), f"small eligible corpus ({len(eligible)} papers) — keep all"

    # Search for the largest gap that still leaves at least MIN_KEPT papers.
    best_gap = 0.0
    best_idx = -1
    for i in range(MIN_KEPT, len(eligible)):
        gap = eligible[i - 1] - eligible[i]
        if gap > best_gap:
            best_gap = gap
            best_idx = i

    if best_idx >= 0 and best_gap >= GAP_THRESHOLD:
        cut = eligible[best_idx - 1]
        return cut, (
            f"natural relevance break: gap of {best_gap:+.2f} between scores "
            f"{eligible[best_idx - 1]:+.2f} and {eligible[best_idx]:+.2f}, "
            f"keeping {best_idx} papers"
        )

    # No clear break — keep top MIN_KEPT papers as a guaranteed minimum, or all
    # eligible papers if they don't exceed 2× the minimum (corpus is small enough
    # that the LLM can handle all of them).
    if len(eligible) <= MIN_KEPT * 2:
        return min(eligible), f"uniform distribution — keeping all {len(eligible)} eligible papers"
    cut = eligible[MIN_KEPT - 1]
    return cut, f"uniform distribution — keeping top {MIN_KEPT} papers (score ≥ {cut:+.2f})"


class RerankRequest(BaseModel):
    """Score fetched papers for relevance against the PICO using LEADS, so the
    downstream summariser cites papers that pass a real screening pass rather
    than papers that merely keyword-matched the database query."""
    papers: List[PaperIn]
    pico: PicoIn
    inclusion: List[str] = Field(default_factory=list)
    exclusion: List[str] = Field(default_factory=list)
    model: Optional[str] = None
    # Auto-cutoff mode (default). The endpoint picks the cutoff itself based on
    # the score distribution — gap detection within the top half, hard floor at
    # 0.0, no max cap. Threshold / quantile_keep are honoured only when auto is
    # explicitly disabled (programmatic callers can still pin a specific cutoff).
    auto: bool = True
    # Manual overrides. Ignored when auto = True.
    threshold: float = -0.2
    quantile_keep: Optional[float] = None
    # Hard cap on output size after sorting. None = keep everything relevant.
    top_k: Optional[int] = None


@app.post("/api/papers/rerank")
def papers_rerank(req: RerankRequest):
    """Score each paper against PICO using LEADS-native, return ranked list
    with per-paper LEADS scores. Use LEADS unconditionally (this is its
    trained task) regardless of which model the user selected for thinking
    tasks elsewhere."""
    _ss["inclusion_list"] = list(req.inclusion or [])
    _ss["exclusion_list"] = list(req.exclusion or [])
    pico = _to_pico(req.pico)
    # Route to LEADS specifically: this is exactly the per-PICO relevance task
    # the model was fine-tuned for. Override whatever the caller asked for.
    model_name = resolve_model_name(req.model) or LEADS_MODEL_NAME

    def _score_one(p_in: PaperIn) -> Dict[str, Any]:
        bp = _to_backend_paper(p_in)
        raw = _screen_one(bp, pico, model_name, req.inclusion, req.exclusion)
        score = float(raw.get("_leads_score", 0.0))
        return {
            "paper": p_in.dict() if hasattr(p_in, "dict") else p_in.__dict__,
            "leads_score": score,
            "decision": str(raw.get("decision", "Exclude")),
            "reason": raw.get("reason", ""),
        }

    scored: List[Dict[str, Any]] = []
    workers = max(1, min(Config.PARALLEL_SCREENING_WORKERS, len(req.papers) or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_score_one, p) for p in req.papers]):
            try:
                scored.append(fut.result())
            except Exception as e:
                print(f"[papers_rerank] worker error: {e}")

    # Sort descending by LEADS score (most relevant first).
    scored.sort(key=lambda r: r["leads_score"], reverse=True)

    # ---- Decide the effective relevance floor -------------------------------
    cutoff_mode: str = "auto"
    quantile_cutoff: Optional[float] = None
    cutoff_reason: str = ""

    if req.auto:
        effective_floor, cutoff_reason = _auto_relevance_cutoff(
            [r["leads_score"] for r in scored]
        )
    else:
        cutoff_mode = "manual"
        effective_floor = float(req.threshold)
        if req.quantile_keep is not None and scored:
            q = max(0.0, min(1.0, float(req.quantile_keep)))
            n = len(scored)
            cutoff_idx = max(0, min(n - 1, int(round(n * q)) - 1))
            quantile_cutoff = scored[cutoff_idx]["leads_score"]
            effective_floor = max(effective_floor, quantile_cutoff)
        cutoff_reason = (
            f"manual: threshold={req.threshold:+.2f}"
            + (f", quantile_keep={req.quantile_keep:.2f}" if req.quantile_keep is not None else "")
        )

    kept = [r for r in scored if r["leads_score"] >= effective_floor]
    if req.top_k is not None:
        kept = kept[: req.top_k]

    return {
        "ranked": scored,
        "kept": kept,
        "cutoff_mode": cutoff_mode,
        "cutoff_reason": cutoff_reason,
        "threshold": req.threshold,
        "quantile_keep": req.quantile_keep,
        "quantile_cutoff": quantile_cutoff,
        "effective_floor": effective_floor,
        "total_scored": len(scored),
        "total_kept": len(kept),
        "model_used": model_name,
    }


class ScreenFullTextRequest(BaseModel):
    paper: PaperIn
    pico: PicoIn
    inclusion: List[str] = Field(default_factory=list)
    exclusion: List[str] = Field(default_factory=list)
    fullText: Optional[str] = None
    model: Optional[str] = None


def _pico_evidence_for_text(source_text: str, pico: PICOCriteria) -> Dict[str, Dict[str, Any]]:
    """For each PICO element, find the best-matching sentence in source_text via
    token overlap. Returns evidence + a coarse match label."""
    out: Dict[str, Dict[str, Any]] = {}
    sentences = re.split(r"(?<=[.!?])\s+", source_text or "")
    fields = [
        ("population", pico.population),
        ("intervention", pico.intervention),
        ("comparator", pico.comparator),
        ("outcome", pico.outcome),
    ]
    for field, value in fields:
        if not value:
            out[field] = {"evidence": "", "match": "no", "score": 0, "value": ""}
            continue
        toks = [t for t in re.split(r"\W+", value.lower()) if len(t) > 3]
        if not toks or not sentences:
            out[field] = {"evidence": "", "match": "no", "score": 0, "value": value}
            continue
        best_sent, best_score = "", 0
        for s in sentences:
            lo = s.lower()
            score = sum(1 for t in toks if t in lo)
            if score > best_score:
                best_sent, best_score = s, score
        threshold_yes = max(2, len(toks) // 2)
        if best_score >= threshold_yes:
            match = "yes"
        elif best_score > 0:
            match = "partial"
        else:
            match = "no"
        out[field] = {
            "evidence": (best_sent or (source_text or "")[:200]).strip(),
            "match": match,
            "score": best_score,
            "value": value,
        }
    return out


def _fulltext_reason(
    title: str, source_text: str, pico: PICOCriteria, decision: str,
    inclusion: List[str], exclusion: List[str], criteria_eval: Dict[str, str],
    has_full_text: bool, model_name: str,
) -> str:
    """LLM-written, study-specific eligibility justification for the Reason column.

    Explains WHY this paper was included/excluded in plain prose grounded in the
    actual text — not a generic "Population mismatch" template.
    """
    from langchain_core.messages import HumanMessage

    model = AIService.get_model(model_name)
    if not model:
        return ""
    text_block = (source_text or "").strip()[:8000] or "(no full text or abstract available)"
    inc_lines = "\n".join(f"  - {c}: {criteria_eval.get(c, '?')}" for c in inclusion) or "  (none specified)"
    exc_lines = "\n".join(f"  - {c}: {criteria_eval.get(c, '?')}" for c in exclusion) or "  (none specified)"

    prompt = f"""You are writing the eligibility justification for a systematic-review
FULL-TEXT screening decision. The decision already made is: {decision.upper()}.

Write 2-4 plain-prose sentences explaining SPECIFICALLY why this study was {decision}d:
  • Name what THIS paper actually is — its population, study design, intervention/
    exposure, and outcomes — using concrete terms from the text.
  • State which PICO element(s) and which inclusion/exclusion criteria drove the
    decision and WHY, referencing the paper's real content (e.g. "excluded because
    the sample was children with diabetes, not the target adult population").
  • If only an abstract is available (no full text), say the judgement is based on
    the abstract and flag the main uncertainty.
  • NEVER output a generic template like "Population mismatch; Intervention
    mismatch" or "lacks specificity". Be concrete, specific, and readable.

PICO frame:
  Population:   {pico.population or '(unspecified)'}
  Intervention: {pico.intervention or '(unspecified)'}
  Comparator:   {pico.comparator or '(unspecified)'}
  Outcome:      {pico.outcome or '(unspecified)'}

Inclusion criteria and verdicts:
{inc_lines}
Exclusion criteria and verdicts:
{exc_lines}

PAPER TITLE: {title}
{'FULL TEXT' if has_full_text else 'ABSTRACT (no full text was retrieved)'}:
{text_block}

Return ONLY the justification prose — no JSON, no headings, no bullet points."""

    try:
        r = model.invoke([HumanMessage(content=prompt)])
        return str(getattr(r, "content", "") or "").strip()[:1200]
    except Exception as e:
        print(f"[fulltext_reason] {e}")
        return ""


@app.post("/api/screen/fulltext")
def screen_fulltext(req: ScreenFullTextRequest):
    _ss["inclusion_list"] = list(req.inclusion or [])
    _ss["exclusion_list"] = list(req.exclusion or [])
    pico = _to_pico(req.pico)

    paper_dict = {
        "Title": req.paper.title,
        "Abstract": req.fullText or req.paper.abstract,
        "Source": req.paper.source,
        "URL": req.paper.url,
        "paper_id": req.paper.id,
    }
    raw = AIService.screen_full_text(paper_dict, pico, resolve_model_name(req.model) or resolve_model_name(_default_model()))

    decision_raw = str(raw.get("decision", "Exclude")).strip().lower()
    decision = "Include" if decision_raw.startswith("inc") else "Exclude"

    criteria_eval: Dict[str, str] = {}
    criteria_evidence: Dict[str, Dict[str, str]] = {}
    inclusion_score = 0
    exclusion_violations = 0
    source_text = req.fullText or req.paper.abstract or ""

    def _ev(criterion: str) -> str:
        toks = [t for t in re.split(r"\W+", criterion.lower()) if len(t) > 3]
        if not toks or not source_text:
            return source_text[:200]
        sentences = re.split(r"(?<=[.!?])\s+", source_text)
        best, best_score = "", 0
        for s in sentences:
            lo = s.lower()
            score = sum(1 for t in toks if t in lo)
            if score > best_score:
                best, best_score = s, score
        return best or source_text[:200]

    for crit in (req.inclusion or []):
        v = str(raw.get(crit, "INCLUDE")).upper()
        v = "INCLUDE" if v == "INCLUDE" else "EXCLUDE"
        criteria_eval[crit] = v
        criteria_evidence[crit] = {
            "decision": v,
            "evidence": _ev(crit),
            "reasoning": "Text supports this inclusion criterion." if v == "INCLUDE" else "Could not find supporting evidence.",
        }
        if v == "INCLUDE":
            inclusion_score += 1

    for crit in (req.exclusion or []):
        v = str(raw.get(crit, "INCLUDE")).upper()
        v = "EXCLUDE" if v == "EXCLUDE" else "INCLUDE"
        criteria_eval[crit] = v
        criteria_evidence[crit] = {
            "decision": v,
            "evidence": _ev(crit),
            "reasoning": "Paper violates this exclusion criterion." if v == "EXCLUDE" else "No exclusion violation detected.",
        }
        if v == "EXCLUDE":
            exclusion_violations += 1

    pico_evidence = _pico_evidence_for_text(source_text, pico)

    # Primary Reason: a study-specific, LLM-written justification explaining WHY
    # this paper was included/excluded. Falls back to a deterministic summary
    # (PICO match labels + criteria stats) if the model is unavailable/errors.
    inc_total = len(req.inclusion or [])
    exc_total = len(req.exclusion or [])

    reason_model = resolve_for_thinking(req.model)
    synthesised_reason = _fulltext_reason(
        req.paper.title, source_text, pico, decision,
        list(req.inclusion or []), list(req.exclusion or []), criteria_eval,
        bool(req.fullText), reason_model,
    )

    if not synthesised_reason:
        def _pico_label(elem: str) -> str:
            m = (pico_evidence.get(elem, {}) or {}).get("match", "no")
            if m == "yes":     return f"{elem} match"
            if m == "partial": return f"{elem} partial"
            return f"{elem} mismatch"

        pico_summary = "; ".join(_pico_label(e) for e in ("Population", "Intervention", "Comparator", "Outcome"))
        parts: List[str] = [f"PICO: {pico_summary}."]
        if inc_total > 0 or exc_total > 0:
            parts.append(
                f"Met {inclusion_score} of {inc_total} inclusion criteria; "
                f"{exclusion_violations} of {exc_total} exclusion violation"
                f"{'' if exclusion_violations == 1 else 's'}."
            )
        raw_reason = str(raw.get("reason", "")).strip()
        if raw_reason and raw_reason.lower() not in {"include", "exclude", "n/a", "na", "none", ""}:
            parts.append(raw_reason)
        synthesised_reason = " ".join(parts)

    return {
        "paper_id": req.paper.id,
        "Title": req.paper.title,
        "URL": req.paper.url,
        "Source": req.paper.source,
        "Abstract": req.paper.abstract,
        "Decision": decision,
        "Reason": synthesised_reason,
        "criteriaEval": criteria_eval,
        "criteriaEvidence": criteria_evidence,
        "picoEvidence": pico_evidence,
        "inclusion_score": inclusion_score,
        "exclusion_violations": exclusion_violations,
    }


# ---------------------------------------------------------------------------
# Agentic search optimization
# ---------------------------------------------------------------------------


class AgenticOptimizeRequest(BaseModel):
    base_query: str
    pico: PicoIn
    sources: List[str]
    model: Optional[str] = None
    task_id: Optional[str] = None


@app.post("/api/simulation/agentic/stream")
def simulation_agentic_stream(req: AgenticOptimizeRequest):
    """Streaming variant of /api/simulation/agentic.

    Runs the optimizer in a background thread and emits SSE events:
      event: progress   data: {iteration, total, source, count, relevance, reasoning}
      event: done       data: <full result object>
      event: error      data: {message}
    """
    pico = _to_pico(req.pico)
    model_name = resolve_for_thinking(req.model)
    cancel_event = _register_cancel(req.task_id)

    event_queue: "queue.Queue[Tuple[str, dict]]" = queue.Queue()

    def _cb(iteration: int, total: int, source: str, count: int, relevance: float, reasoning: str):
        # Check cancel BEFORE emitting progress so the next iteration won't start.
        if cancel_event and cancel_event.is_set():
            raise TaskCanceled()
        event_queue.put((
            "progress",
            {
                "iteration": int(iteration),
                "total": int(total),
                "source": str(source),
                "count": int(count),
                "relevance": float(relevance),
                "reasoning": str(reasoning or ""),
            },
        ))

    def _run():
        try:
            out = AIService.agentic_optimize_per_source(
                req.base_query, pico, model_name, req.sources,
                research_goal=req.base_query, progress_callback=_cb,
            )
            event_queue.put(("done", out))
        except TaskCanceled:
            event_queue.put(("canceled", {"message": "Canceled by user"}))
        except Exception as e:
            import traceback
            traceback.print_exc()
            event_queue.put(("error", {"message": str(e)}))
        finally:
            _unregister_cancel(req.task_id)

    threading.Thread(target=_run, daemon=True).start()

    def _gen():
        while True:
            try:
                event_type, data = event_queue.get(timeout=600)
            except queue.Empty:
                yield f"event: error\ndata: {_json.dumps({'message': 'timeout'})}\n\n"
                return
            yield f"event: {event_type}\ndata: {_json.dumps(data)}\n\n"
            if event_type in ("done", "error", "canceled"):
                return

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.post("/api/simulation/agentic")
def simulation_agentic(req: AgenticOptimizeRequest):
    pico = _to_pico(req.pico)
    model_name = resolve_for_thinking(req.model)
    try:
        # Python signature: (current_query, pico, model_name, active_sources, research_goal="", progress_callback=None)
        out = AIService.agentic_optimize_per_source(
            req.base_query, pico, model_name, req.sources
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"agentic_optimize_per_source failed: {e}")
    return out


# ---------------------------------------------------------------------------
# Snowballing / citations
# ---------------------------------------------------------------------------


class CitationsRequest(BaseModel):
    paper_id: str = ""
    source: str = ""
    title: str
    snowball_type: str = "Both"  # "Both" | "Backward (References)" | "Forward (Cited by)"
    max_per: int = 10
    sources: List[str] = Field(default_factory=lambda: ["PubMed", "Semantic Scholar", "Europe PMC"])


def _epmc_resolve(title: str, paper_id: str) -> Optional[Tuple[str, str]]:
    """Resolve a paper to (source, id) on Europe PMC. Falls back to a title search."""
    if paper_id and paper_id.isdigit():
        return ("MED", paper_id)
    try:
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        q = paper_id if (paper_id and not paper_id.isdigit()) else f'TITLE:"{title}"'
        params = {"query": q, "format": "json", "pageSize": 1, "resultType": "lite"}
        r = requests.get(url, params=params, timeout=10).json()
        for it in r.get("resultList", {}).get("result", []):
            return ((it.get("source") or "MED"), str(it.get("id") or ""))
    except Exception as e:
        print(f"[epmc_resolve] {e}")
    return None


def _epmc_links(source: str, pid: str, direction: str, max_per: int) -> List[Dict[str, Any]]:
    """direction: 'references' (backward) or 'citations' (forward)."""
    out: List[Dict[str, Any]] = []
    try:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{source}/{pid}/{direction}"
        params = {"pageSize": max_per, "format": "json"}
        r = requests.get(url, params=params, timeout=15).json()
        key = "referenceList" if direction == "references" else "citationList"
        items = (r.get(key) or {}).get("reference" if direction == "references" else "citation", []) or []
        ctype = "backward" if direction == "references" else "forward"
        for it in items[:max_per]:
            cid = str(it.get("id") or it.get("doi") or "")
            out.append({
                "id": cid,
                "title": (it.get("title") or it.get("source") or "Untitled").strip(),
                "abstract": (it.get("abstractText") or "").strip(),
                "url": (f"https://pubmed.ncbi.nlm.nih.gov/{cid}/" if cid.isdigit() else (f"https://doi.org/{cid}" if "/" in cid else "")),
                "source": f"Europe PMC ({'Reference' if ctype == 'backward' else 'Cited by'})",
                "citation_type": ctype,
            })
    except Exception as e:
        print(f"[epmc_links] {direction} {e}")
    return out


def _openalex_resolve_work(title: str, paper_id: str) -> Optional[Dict[str, Any]]:
    try:
        # DOI lookup
        if paper_id and "/" in paper_id and paper_id.startswith("10."):
            r = requests.get(
                f"https://api.openalex.org/works/doi:{paper_id}",
                params={"mailto": Config.ENTREZ_EMAIL}, timeout=10,
            )
            if r.status_code == 200:
                return r.json()
        # Title search
        r = requests.get(
            "https://api.openalex.org/works",
            params={"search": title, "per_page": 1, "mailto": Config.ENTREZ_EMAIL}, timeout=10,
        )
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                return results[0]
    except Exception as e:
        print(f"[openalex_resolve] {e}")
    return None


def _openalex_minimal(work: Dict[str, Any], ctype: str) -> Dict[str, Any]:
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    oa = work.get("open_access") or {}
    return {
        "id": (work.get("id") or "").split("/")[-1] or doi,
        "title": (work.get("display_name") or work.get("title") or "Untitled").strip(),
        "abstract": _reconstruct_oa_abstract(work.get("abstract_inverted_index")),
        "url": oa.get("oa_url") or (f"https://doi.org/{doi}" if doi else (work.get("id") or "")),
        "source": f"OpenAlex ({'Reference' if ctype == 'backward' else 'Cited by'})",
        "citation_type": ctype,
    }


def _reconstruct_oa_abstract(idx: Optional[dict]) -> str:
    if not idx:
        return ""
    positions = []
    for word, locs in idx.items():
        for loc in locs:
            positions.append((loc, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def _abstract_via_epmc(title: str, ext_id: str) -> str:
    """Fetch an abstract from Europe PMC's full-text search. EPMC has far wider
    abstract coverage than OpenAlex (which omits abstracts for publishers that
    forbid redistribution), and can be queried directly by PMID or DOI."""
    try:
        if ext_id and ext_id.isdigit():
            q = f"EXT_ID:{ext_id}"
        elif ext_id and "/" in ext_id:
            q = f'DOI:"{ext_id}"'
        elif title:
            q = f'TITLE:"{title}"'
        else:
            return ""
        r = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": q, "resultType": "core", "format": "json", "pageSize": 1},
            timeout=10,
        ).json()
        results = (r.get("resultList") or {}).get("result", []) or []
        if results:
            raw = (results[0].get("abstractText") or "").strip()
            # EPMC abstracts may carry light HTML/JATS markup — strip it.
            return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw)).strip()
    except Exception as e:
        print(f"[abstract_epmc] {e}")
    return ""


def _enrich_missing_abstracts(items: List[Dict[str, Any]], cap: int = 80) -> int:
    """Backfill empty abstracts for snowballed citations.

    Europe PMC reference/citation lists return title-only entries, which starves
    the downstream PICO screening. For each missing abstract we try Europe PMC
    search first (best coverage, direct PMID/DOI lookup), then fall back to
    OpenAlex. Lookups are deduplicated by title and bounded by `cap`.
    Returns the number of papers resolved.
    """
    seen: Dict[str, str] = {}
    resolved = 0
    looked = 0
    for it in items:
        if len((it.get("abstract") or "").strip()) >= 40:
            continue  # already has a usable abstract
        title = (it.get("title") or "").strip()
        ext_id = str(it.get("id") or "")
        if not title and not ext_id:
            continue
        key = title.lower() or ext_id
        if key in seen:                       # reuse a resolved duplicate
            if seen[key]:
                it["abstract"] = seen[key]
            continue
        if looked >= cap:
            continue
        looked += 1
        abstract = _abstract_via_epmc(title, ext_id)
        if not abstract:                      # EPMC miss → try OpenAlex
            work = _openalex_resolve_work(title, ext_id)
            abstract = _reconstruct_oa_abstract(work.get("abstract_inverted_index")) if work else ""
        seen[key] = abstract
        if abstract:
            it["abstract"] = abstract
            resolved += 1
    print(f"[enrich_abstracts] resolved {resolved}/{looked} lookups ({len(items)} citations)")
    return resolved


def _openalex_links(work: Dict[str, Any], direction: str, max_per: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        if direction == "backward":
            refs = (work.get("referenced_works") or [])[:max_per]
            for ref_url in refs:
                try:
                    rr = requests.get(ref_url, params={"mailto": Config.ENTREZ_EMAIL}, timeout=8)
                    if rr.status_code == 200:
                        out.append(_openalex_minimal(rr.json(), "backward"))
                except Exception:
                    continue
        else:
            cited_by = work.get("cited_by_api_url")
            if cited_by:
                rr = requests.get(cited_by, params={"per_page": max_per, "mailto": Config.ENTREZ_EMAIL}, timeout=12)
                if rr.status_code == 200:
                    for w in rr.json().get("results", [])[:max_per]:
                        out.append(_openalex_minimal(w, "forward"))
    except Exception as e:
        print(f"[openalex_links] {direction} {e}")
    return out


def _ss_resolve_id(title: str) -> Optional[str]:
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": title, "limit": 1, "fields": "paperId"}, timeout=10,
        )
        if r.status_code == 200:
            results = r.json().get("data", [])
            if results:
                return results[0].get("paperId")
    except Exception as e:
        print(f"[ss_resolve] {e}")
    return None


def _ss_links(pid: str, direction: str, max_per: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        endpoint = "references" if direction == "backward" else "citations"
        url = f"https://api.semanticscholar.org/graph/v1/paper/{pid}/{endpoint}"
        r = requests.get(url, params={"fields": "paperId,title,abstract,url", "limit": max_per}, timeout=12)
        if r.status_code != 200:
            return out
        outer = "citedPaper" if direction == "backward" else "citingPaper"
        for it in r.json().get("data", [])[:max_per]:
            paper = it.get(outer) or {}
            out.append({
                "id": paper.get("paperId", ""),
                "title": (paper.get("title") or "Untitled").strip(),
                "abstract": (paper.get("abstract") or "").strip(),
                "url": paper.get("url") or "",
                "source": f"Semantic Scholar ({'Reference' if direction == 'backward' else 'Cited by'})",
                "citation_type": "backward" if direction == "backward" else "forward",
            })
    except Exception as e:
        print(f"[ss_links] {direction} {e}")
    return out


@app.post("/api/citations")
def citations(req: CitationsRequest):
    want_back = req.snowball_type in {"Both", "Backward (References)"}
    want_fwd = req.snowball_type in {"Both", "Forward (Cited by)"}
    sources = set(s for s in req.sources)
    out: List[Dict[str, Any]] = []

    # Europe PMC (and via that any PubMed-indexed paper).
    if "Europe PMC" in sources or "PubMed" in sources:
        resolved = _epmc_resolve(req.title, req.paper_id)
        if resolved:
            src, pid = resolved
            if want_back:
                out.extend(_epmc_links(src, pid, "references", req.max_per))
            if want_fwd:
                out.extend(_epmc_links(src, pid, "citations", req.max_per))

    # OpenAlex.
    if "OpenAlex" in sources:
        work = _openalex_resolve_work(req.title, req.paper_id)
        if work:
            if want_back:
                out.extend(_openalex_links(work, "backward", req.max_per))
            if want_fwd:
                out.extend(_openalex_links(work, "forward", req.max_per))

    # Semantic Scholar.
    if "Semantic Scholar" in sources:
        ss_id = _ss_resolve_id(req.title)
        if ss_id:
            if want_back:
                out.extend(_ss_links(ss_id, "backward", req.max_per))
            if want_fwd:
                out.extend(_ss_links(ss_id, "forward", req.max_per))

    # Reference/citation lists often omit abstracts — backfill them so the
    # downstream screening has something to work with.
    _enrich_missing_abstracts(out)

    # Strip source markup (e.g. "<i>T. gondii</i>") from titles/abstracts.
    for it in out:
        it["title"] = clean_markup(it.get("title"))
        if it.get("abstract"):
            it["abstract"] = clean_markup(it["abstract"])

    return {"citations": out}


# ---------------------------------------------------------------------------
# Full-text fetch (Europe PMC + Unpaywall)
# ---------------------------------------------------------------------------


class FullTextRequest(BaseModel):
    Title: str
    URL: str
    Source: str
    paper_id: Optional[str] = None
    # Optional DOI for Unpaywall lookups. If the caller has it on hand we
    # use it directly; otherwise we try to mine one from the URL.
    doi: Optional[str] = None


# ---- helpers ---------------------------------------------------------------

def _extract_text_from_pdf(pdf_bytes: bytes) -> Optional[str]:
    """Parse a PDF byte string into plain text via pypdf. Returns None if the
    extraction is empty or too short to be useful."""
    if not pdf_bytes:
        return None
    try:
        from pypdf import PdfReader
        from io import BytesIO
        reader = PdfReader(BytesIO(pdf_bytes))
        parts: List[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                # Some pages (scanned / encrypted) can fail individually; skip them.
                continue
        text = "\n".join(parts).strip()
        return text if text and len(text) > 200 else None
    except Exception as e:
        print(f"[pdf_extract] {e}")
        return None


_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)

def _extract_doi(url: str, title: str = "") -> Optional[str]:
    """Pull a DOI out of a URL or other free text. Returns the canonical
    DOI string (`10.xxxx/yyy`) or None."""
    for candidate in (url or "", title or ""):
        m = _DOI_RE.search(candidate)
        if m:
            # Strip trailing punctuation that often follows a DOI in URLs.
            return re.sub(r"[).,;]+$", "", m.group(1))
    return None


_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([\w./-]+?)(?:v\d+)?(?:\.pdf)?(?:[/?#]|$)", re.I)

def _extract_arxiv_id(url: str) -> Optional[str]:
    m = _ARXIV_RE.search(url or "")
    return m.group(1) if m else None


def _fetch_pmc_pdf(pmcid: str) -> Optional[bytes]:
    """Download the PMC-hosted PDF for a given PMC ID. Tries the EuPMC
    rendered PDF URL first, then falls back to the NCBI PMC URL."""
    if not pmcid:
        return None
    if not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    pmcid = pmcid.upper()
    candidates = [
        f"https://europepmc.org/articles/{pmcid}/pdf",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/",
    ]
    for url in candidates:
        try:
            r = requests.get(
                url, timeout=30, allow_redirects=True,
                headers={"User-Agent": "EvidenceEngine/1.0 (haile.teshome@berkeley.edu)"},
            )
            ct = (r.headers.get("content-type") or "").lower()
            if r.status_code == 200 and (
                "pdf" in ct or r.content[:4] == b"%PDF"
            ):
                return r.content
        except Exception as e:
            print(f"[pmc_pdf] {url}: {e}")
            continue
    return None


def _fetch_unpaywall_pdf(doi: str) -> Optional[Tuple[bytes, str]]:
    """Resolve a DOI to an open-access PDF via the Unpaywall API and download
    the bytes. Returns (pdf_bytes, source_url) on success."""
    if not doi:
        return None
    try:
        email = getattr(Config, "ENTREZ_EMAIL", None) or "research@example.com"
        api = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        r = requests.get(api, timeout=15, headers={"User-Agent": "EvidenceEngine/1.0"})
        if r.status_code != 200:
            return None
        data = r.json() or {}
        if not data.get("is_oa"):
            return None
        # Try best_oa_location first; if it doesn't carry a PDF link, walk
        # the rest of oa_locations.
        candidates: List[str] = []
        best = data.get("best_oa_location") or {}
        if isinstance(best, dict):
            for k in ("url_for_pdf", "url"):
                if best.get(k):
                    candidates.append(best[k])
        for loc in (data.get("oa_locations") or []):
            if not isinstance(loc, dict):
                continue
            for k in ("url_for_pdf", "url"):
                if loc.get(k) and loc[k] not in candidates:
                    candidates.append(loc[k])

        for pdf_url in candidates:
            try:
                rr = requests.get(
                    pdf_url, timeout=30, allow_redirects=True,
                    headers={"User-Agent": "EvidenceEngine/1.0"},
                )
                ct = (rr.headers.get("content-type") or "").lower()
                if rr.status_code == 200 and (
                    "pdf" in ct or rr.content[:4] == b"%PDF"
                ):
                    return rr.content, pdf_url
            except Exception as e:
                print(f"[unpaywall_pdf {doi}] {pdf_url}: {e}")
                continue
    except Exception as e:
        print(f"[unpaywall {doi}] {e}")
    return None


def _fetch_arxiv_pdf(arxiv_id: str) -> Optional[bytes]:
    if not arxiv_id:
        return None
    try:
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        r = requests.get(
            url, timeout=30, allow_redirects=True,
            headers={"User-Agent": "EvidenceEngine/1.0"},
        )
        ct = (r.headers.get("content-type") or "").lower()
        if r.status_code == 200 and ("pdf" in ct or r.content[:4] == b"%PDF"):
            return r.content
    except Exception as e:
        print(f"[arxiv_pdf] {e}")
    return None


def _fetch_epmc_fulltext_text(paper_id: str) -> Optional[str]:
    """JATS-XML route via _fetch_epmc_full_text_xml (which already handles the
    PMID -> PMCID lookup). Returns plain text or None."""
    if not paper_id:
        return None
    xml = _fetch_epmc_full_text_xml(paper_id)
    if not xml:
        return None
    try:
        soup = BeautifulSoup(xml, "lxml-xml")
        body = soup.find("body") or soup
        text = body.get_text(separator="\n", strip=True)
        return text if text and len(text) > 200 else None
    except Exception as e:
        print(f"[epmc_fulltext parse] {e}")
        return None


# ---- endpoint --------------------------------------------------------------

@app.post("/api/fulltext/fetch")
def fulltext_fetch(req: FullTextRequest):
    """Strategy ladder, in order of decreasing reliability:
      1. Europe PMC JATS fullTextXML  (best — structured text)
      2. PMC PDF                       (parsed with pypdf)
      3. Unpaywall OA PDF              (any source, parsed with pypdf)
      4. arXiv PDF                     (for arXiv papers, parsed with pypdf)
      5. HTML scrape                   (last resort; skips useless landing pages)
    Returns `source` describing which tier supplied the text so the UI can
    show e.g. "PMC PDF (PMC1234567)" instead of just "Europe PMC".
    """
    pid = (req.paper_id or "").strip()
    if not pid and req.URL:
        m = re.search(r"/(PMC\d+|\d+)/?$", req.URL, re.IGNORECASE)
        if m:
            pid = m.group(1)

    debug_label = pid or (req.URL[:60] if req.URL else "(no id)")

    # One-shot lookup against EuPMC search: gives us BOTH the PMC ID (for the
    # XML / PMC-PDF tiers) and the DOI (for the Unpaywall tier). Saves a
    # second round-trip when both are needed.
    pmcid: Optional[str] = None
    lookup_doi: Optional[str] = None
    if pid:
        if pid.upper().startswith("PMC"):
            pmcid = pid.upper()
        elif pid.isdigit():
            meta = _lookup_pmc_metadata(pid)
            pmcid = meta.get("pmcid")
            lookup_doi = meta.get("doi")

    # Tier 1 — EuPMC structured XML.
    if pid:
        text = _fetch_epmc_fulltext_text(pid)
        if text:
            print(f"[fulltext_fetch] {debug_label} tier1 EuPMC XML -> {len(text)} chars")
            return {"status": "found", "text": text, "source": "Europe PMC (XML)"}

    # Tier 2 — PMC PDF (works for any PubMed paper with a PMC mirror).
    if pmcid:
        pdf = _fetch_pmc_pdf(pmcid)
        if pdf:
            text = _extract_text_from_pdf(pdf)
            if text:
                print(f"[fulltext_fetch] {debug_label} tier2 PMC PDF ({pmcid}) -> {len(text)} chars")
                return {"status": "found", "text": text, "source": f"PMC PDF ({pmcid})"}

    # Tier 3 — Unpaywall via DOI. Prefer the caller-supplied DOI, then the
    # one mined from URL, then the one returned by the EuPMC metadata lookup.
    doi = (req.doi or "").strip() or _extract_doi(req.URL, req.Title) or (lookup_doi or "")
    if doi:
        unpaywall = _fetch_unpaywall_pdf(doi)
        if unpaywall:
            pdf, src_url = unpaywall
            text = _extract_text_from_pdf(pdf)
            if text:
                host = src_url.split("/", 3)[2] if "://" in src_url else src_url
                print(f"[fulltext_fetch] {debug_label} tier3 Unpaywall PDF ({host}) -> {len(text)} chars")
                return {"status": "found", "text": text, "source": f"Unpaywall PDF ({host})"}

    # Tier 4 — arXiv PDF.
    if (req.Source or "").lower() == "arxiv" or "arxiv.org" in (req.URL or "").lower():
        arxiv_id = _extract_arxiv_id(req.URL or "") or (pid if pid and not pid.isdigit() else None)
        if arxiv_id:
            pdf = _fetch_arxiv_pdf(arxiv_id)
            if pdf:
                text = _extract_text_from_pdf(pdf)
                if text:
                    print(f"[fulltext_fetch] {debug_label} tier4 arXiv PDF ({arxiv_id}) -> {len(text)} chars")
                    return {"status": "found", "text": text, "source": f"arXiv PDF ({arxiv_id})"}

    # Tier 5 — HTML scrape, but only on hosts that might actually carry
    # real article content. Abstract landing pages are skipped because they
    # don't include the body text we'd need.
    if req.URL and req.URL.startswith("http"):
        host = req.URL.split("/", 3)[2].lower() if "://" in req.URL else ""
        skip_hosts = {"pubmed.ncbi.nlm.nih.gov", "europepmc.org", "www.ncbi.nlm.nih.gov"}
        if host and host not in skip_hosts:
            try:
                r = requests.get(req.URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                ct = (r.headers.get("content-type") or "").lower()
                if r.status_code == 200 and "pdf" in ct:
                    # Some publisher landing pages redirect to a PDF directly.
                    text = _extract_text_from_pdf(r.content)
                    if text:
                        print(f"[fulltext_fetch] {debug_label} tier5 publisher PDF ({host}) -> {len(text)} chars")
                        return {"status": "found", "text": text, "source": f"Publisher PDF ({host})"}
                if r.status_code == 200:
                    soup = BeautifulSoup(r.content, "html.parser")
                    for s in soup(["script", "style", "nav", "footer", "header"]):
                        s.decompose()
                    text = soup.get_text(separator="\n", strip=True)
                    if text and len(text) > 500:
                        print(f"[fulltext_fetch] {debug_label} tier5 HTML scrape ({host}) -> {len(text)} chars")
                        return {"status": "found", "text": text[:50000], "source": f"HTML scrape ({host})"}
            except Exception as e:
                print(f"[fulltext_fetch html] {e}")

    print(f"[fulltext_fetch] {debug_label} -> no full text "
          f"(pid={pid or '-'}, doi={doi or '-'}, source={req.Source}, url_host="
          f"{req.URL.split('/', 3)[2] if req.URL and '://' in req.URL else '-'})")
    return {"status": "missing", "reason": "Full text not retrievable from open-access sources."}


# ---------------------------------------------------------------------------
# Text extraction (heuristic + LLM-friendly)
# ---------------------------------------------------------------------------


class ExtractTextRequest(BaseModel):
    text: str
    query: str
    model: Optional[str] = None


# Regex that catches the typical section headings in a scientific paper.
# We use it to (a) label evidence by section ("Methods", "Results", ...) and
# (b) detect candidate section breakpoints for the section-aware preview.
_SECTION_HEADING_RE = re.compile(
    r"(?im)^\s*("
    r"abstract|background|introduction|methods?|materials?\s+and\s+methods?|"
    r"results?|findings?|discussion|conclusions?|limitations|"
    r"acknowledg(?:e)?ments|references|appendix|"
    r"supplementary|supporting\s+information|"
    r"funding|conflict[s]?\s+of\s+interest"
    r")\s*[:.\-]?\s*$"
)


def _section_at_offset(text: str, offset: int) -> str:
    """Return the most recent section heading at or before `offset`.

    Defaults to "Abstract" when no heading has been crossed yet. Returns
    "Other" when we can't tell.
    """
    if not text or offset < 0:
        return "Other"
    last_heading = ""
    for m in _SECTION_HEADING_RE.finditer(text[:max(0, offset)]):
        last_heading = m.group(1)
    if not last_heading:
        return "Abstract"
    h = last_heading.strip().lower()
    # Normalise to canonical labels.
    canon = {
        "abstract":           "Abstract",
        "background":         "Background",
        "introduction":       "Introduction",
        "method":             "Methods",
        "methods":            "Methods",
        "material and method":   "Methods",
        "materials and method":  "Methods",
        "material and methods":  "Methods",
        "materials and methods": "Methods",
        "result":             "Results",
        "results":            "Results",
        "findings":           "Results",
        "finding":            "Results",
        "discussion":         "Discussion",
        "conclusion":         "Conclusion",
        "conclusions":        "Conclusion",
        "limitations":        "Limitations",
        "references":         "References",
        "appendix":           "Appendix",
    }
    return canon.get(h, h.title())


def _anchor_quote_in_text(quote: str, text: str) -> Optional[Tuple[int, int]]:
    """Locate the model's quote in the source text. Returns (start, end) on
    success or None when no reasonable anchor exists.

    Tier 1 — direct substring match (case-insensitive).
    Tier 2 — whitespace-normalised substring (handles minor formatting drift).
    Tier 3 — best contiguous span matching ≥60 % of the quote's tokens.
    """
    if not quote or not text:
        return None
    q = quote.strip().strip('"').strip("'")
    if not q:
        return None

    lower_text = text.lower()
    lower_q = q.lower()
    idx = lower_text.find(lower_q)
    if idx >= 0:
        return (idx, idx + len(q))

    # Whitespace-collapsed search.
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()
    norm_text = _norm(lower_text)
    norm_q = _norm(lower_q)
    if norm_q and norm_q in norm_text:
        # Map the normalised offset back to the original text by counting
        # non-whitespace characters.
        target = norm_text.find(norm_q)
        consumed_norm = 0
        for i, ch in enumerate(text):
            if consumed_norm == target:
                # Walk forward in `text` until we've covered the quote length
                # ignoring excess whitespace.
                remaining = len(norm_q)
                j = i
                last_non_ws = i
                while j < len(text) and remaining > 0:
                    if not text[j].isspace():
                        remaining -= 1
                        last_non_ws = j
                    j += 1
                return (i, last_non_ws + 1)
            if not ch.isspace():
                consumed_norm += 1
        return None

    # Token-overlap fallback: find the window of `text` that contains the
    # highest number of ≥ 4-character tokens from the quote.
    tokens = [t for t in re.findall(r"\w{4,}", q.lower())]
    if not tokens:
        return None
    sentences = list(re.finditer(r"[^.!?\n]+[.!?\n]?", text))
    best_score = 0
    best_span: Optional[Tuple[int, int]] = None
    for m in sentences:
        s = m.group(0).lower()
        score = sum(1 for t in tokens if t in s)
        if score > best_score:
            best_score = score
            best_span = (m.start(), m.end())
    if best_score >= max(1, math.ceil(len(tokens) * 0.6)):
        return best_span
    return None


def _retrieve_relevant_chunks(text: str, query: str, top_k: int = 6, chunk_chars: int = 600) -> str:
    """Return the most query-relevant paragraphs from *text*, capped at
    top_k chunks (~3-5 k chars total).  Always prepends the opening ~800
    chars (abstract / intro) so the model has structural context.

    Scoring is a simple bag-of-words overlap — zero dependencies, fast."""
    import re as _re

    stop = {
        "the","a","an","is","are","was","were","what","which","how","why",
        "when","where","who","and","or","in","on","at","to","for","of",
        "with","by","from","that","this","be","has","have","had","do",
        "does","did","not","but","if","as","it","its","their","they",
    }
    query_words = {
        w.lower() for w in _re.findall(r"\w+", query)
        if w.lower() not in stop and len(w) > 2
    }

    # Split on blank lines; fall back to sentence grouping for dense PDFs.
    paragraphs = [p.strip() for p in _re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) <= 3:
        sentences = _re.split(r"(?<=[.!?])\s+", text)
        chunks, buf = [], ""
        for s in sentences:
            if len(buf) + len(s) > chunk_chars and buf:
                chunks.append(buf)
                buf = s
            else:
                buf += (" " if buf else "") + s
        if buf:
            chunks.append(buf)
        paragraphs = chunks or [text]

    # Group tiny paragraphs into ~chunk_chars blocks.
    merged, buf = [], ""
    for p in paragraphs:
        if len(buf) + len(p) > chunk_chars and buf:
            merged.append(buf)
            buf = p
        else:
            buf += ("\n\n" if buf else "") + p
    if buf:
        merged.append(buf)
    paragraphs = merged

    def score(chunk: str) -> float:
        words = {w.lower() for w in _re.findall(r"\w+", chunk)}
        overlap = len(query_words & words)
        # Bonus for statistical markers common in results sections
        stat_hit = bool(_re.search(
            r"\d+\.?\d*\s*(%|p\s*[<=]|mg|kg|n\s*=|ci|hr|or\b|rr\b|sd\b|sem\b)",
            chunk, _re.I,
        ))
        return overlap / (len(query_words) + 1) + (0.15 if stat_hit else 0.0)

    scored = sorted(enumerate(paragraphs), key=lambda x: -score(x[1]))
    top_indices = {i for i, _ in scored[:top_k]}

    # Always include the opening block for structural context.
    top_indices.add(0)

    selected = [paragraphs[i] for i in sorted(top_indices) if i < len(paragraphs)]
    return "\n\n---\n\n".join(selected)


@app.post("/api/extract/text")
def extract_text(req: ExtractTextRequest):
    """LLM-driven extraction that answers `query` against the supplied full
    text and returns:
      - `answer`: a 2-4 sentence natural-language answer to the question.
      - `evidence`: list of { quote, section, why, start, end } where the
        quote is a verbatim span from the text, `section` is the canonical
        section label inferred from headings, and start/end are character
        offsets in `text` so the UI can highlight precisely.
      - `values`: list of { field, value, quote, section, start, end } for
        structured key/value pairs the model extracted alongside the answer.

    Falls back to the legacy regex-based behaviour only when the model
    completely fails (no JSON, exception, etc.).
    """
    from langchain_core.messages import HumanMessage

    text = req.text or ""
    query = (req.query or "").strip()
    if not text:
        return {
            "answer": "No full text available for this paper.",
            "summary": "No full text available for this paper.",
            "evidence": [],
            "spans": [],
            "values": [],
        }
    if not query:
        return {
            "answer": "",
            "summary": "Enter a question to extract from the text.",
            "evidence": [],
            "spans": [],
            "values": [],
        }

    model_name = resolve_for_thinking(req.model)
    model = AIService.get_model(model_name)

    # Retrieve only the query-relevant chunks instead of the first 30 k chars.
    # Quotes are anchored against the FULL text afterwards, so this is safe.
    try:
        text_for_llm = _retrieve_relevant_chunks(text, query, top_k=6, chunk_chars=600)
        if not text_for_llm.strip():
            text_for_llm = text[:30000]
    except Exception:
        text_for_llm = text[:30000]

    prompt = f"""You are extracting evidence from a scientific paper to answer a researcher's
question. Use ONLY the paper text below. Do not bring in outside knowledge.

QUESTION: {query}

PAPER TEXT:
{text_for_llm}

Return ONLY a JSON object with this exact shape:
{{
  "answer": "<2-4 sentence natural-language answer; if the text does not contain enough information, say so explicitly>",
  "evidence": [
    {{
      "quote": "<VERBATIM excerpt copied from the paper, 10-300 characters>",
      "why":   "<one short sentence saying how this excerpt addresses the question>"
    }}
  ],
  "values": [
    {{
      "field": "<short label, e.g. 'Sample size', 'Primary outcome', 'HR (95% CI)', 'p-value'>",
      "value": "<the value as it appears in the text>",
      "quote": "<short verbatim sentence fragment containing the value>"
    }}
  ]
}}

RULES:
- Every "quote" must be a real verbatim string from the paper text — do not
  paraphrase. If you cannot find one, omit that evidence item.
- Return 1-5 evidence items, ordered by how directly they answer the question.
- Return 0-8 values; only include values that are actually present in the text.
- If the paper does not answer the question, set "evidence": [] and "values": []
  and explain in "answer" that the information is not in the paper.
"""

    answer = ""
    evidence_raw: List[Dict[str, Any]] = []
    values_raw: List[Dict[str, Any]] = []
    if model:
        try:
            r = model.invoke([HumanMessage(content=prompt)])
            data = AIService._extract_json(r.content) or {}
            answer = str(data.get("answer") or "").strip()
            ev = data.get("evidence") or []
            if isinstance(ev, list):
                evidence_raw = [e for e in ev if isinstance(e, dict)]
            vv = data.get("values") or []
            if isinstance(vv, list):
                values_raw = [v for v in vv if isinstance(v, dict)]
        except Exception as e:
            print(f"[extract_text] LLM call failed: {e}")

    # ---- Anchor evidence + values to the full text -------------------------
    evidence: List[Dict[str, Any]] = []
    seen_spans: set[Tuple[int, int]] = set()
    for e in evidence_raw:
        quote = str(e.get("quote") or "").strip().strip('"').strip("'")
        if not quote:
            continue
        anchored = _anchor_quote_in_text(quote, text)
        if not anchored:
            continue
        if anchored in seen_spans:
            continue
        seen_spans.add(anchored)
        evidence.append({
            "quote": text[anchored[0]:anchored[1]],
            "why":   str(e.get("why") or "").strip()[:300],
            "section": _section_at_offset(text, anchored[0]),
            "start": anchored[0],
            "end":   anchored[1],
        })

    values: List[Dict[str, Any]] = []
    for v in values_raw[:8]:
        field = str(v.get("field") or "").strip()
        value = str(v.get("value") or "").strip()
        if not field or not value:
            continue
        quote = str(v.get("quote") or "").strip().strip('"').strip("'") or value
        anchored = _anchor_quote_in_text(quote, text)
        item: Dict[str, Any] = {
            "field": field[:60],
            "value": value[:200],
            "quote": quote[:240],
        }
        if anchored:
            item["start"]   = anchored[0]
            item["end"]     = anchored[1]
            item["section"] = _section_at_offset(text, anchored[0])
        values.append(item)

    # ---- Regex fallback only when the LLM produced absolutely nothing ------
    if not answer and not evidence and not values:
        tokens = [t for t in re.split(r"\s+", query.lower()) if t]
        for m in re.finditer(r"[^.!?\n]+[.!?]", text):
            sent = m.group(0)
            lo = sent.lower()
            score = sum(1 for t in tokens if t in lo)
            if score >= max(1, math.ceil(len(tokens) / 3)):
                evidence.append({
                    "quote":  sent.strip(),
                    "why":    "Matched key terms from the question (regex fallback).",
                    "section": _section_at_offset(text, m.start()),
                    "start":   m.start(),
                    "end":     m.end(),
                })
                if len(evidence) >= 5:
                    break
        if evidence:
            answer = f"Found {len(evidence)} passage{'s' if len(evidence) > 1 else ''} matching key terms (LLM extraction unavailable)."
        else:
            answer = f"No relevant passages found for \"{query}\" in this paper."

    # Legacy fields kept so existing UI components (and the JSON export)
    # continue to work without breaking.
    spans = [{"start": e["start"], "end": e["end"], "label": e.get("section")} for e in evidence]
    return {
        "answer": answer,
        "summary": answer,    # alias for back-compat with the prior summary field
        "evidence": evidence,
        "spans": spans,
        "values": values,
    }


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------


class ExtractTablesRequest(BaseModel):
    Title: str
    URL: str
    Source: str
    paper_id: Optional[str] = None
    extraction_type: Optional[str] = "All"
    model: Optional[str] = None
    # Caller-supplied text used as input for the LLM fallback when neither the
    # JATS XML nor the HTML scrape produces tables. Without these the fallback
    # only sees the title and inevitably returns nothing.
    abstract: Optional[str] = None
    full_text: Optional[str] = None


def _classify_table(rows: List[List[str]], hint: str) -> str:
    blob = " ".join(c.lower() for r in rows[:3] for c in r)
    if any(k in blob for k in ("age", "sex", "bmi", "race", "demographic")):
        return "Demographics"
    if any(k in blob for k in ("p-value", "p =", "ci", "hr", "or ", "rr ")):
        return "Statistical Results"
    if any(k in blob for k in ("adverse", "events")):
        return "Adverse Events"
    if any(k in blob for k in ("outcome", "primary", "secondary")):
        return "Outcomes"
    return hint or "General"


def _extract_html_tables(url: str, extraction_type: str) -> List[Dict[str, Any]]:
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.content, "html.parser")
        out = []
        for i, t in enumerate(soup.find_all("table")):
            rows = []
            for tr in t.find_all("tr"):
                cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
            if rows:
                out.append({
                    "title": f"Table {i + 1}",
                    "type": _classify_table(rows, extraction_type or ""),
                    "data": rows,
                    "caption": "",
                })
        return out
    except Exception as e:
        print(f"[extract_html_tables] {e}")
        return []


def _lookup_pmc_metadata(pmid: str) -> Dict[str, Optional[str]]:
    """Resolve a PubMed PMID to PMC ID + DOI via the EuPMC search endpoint.

    Returns `{"pmcid": "PMC1234567" | None, "doi": "10.x/y" | None}`. Used by
    multiple downstream tiers (PMC PDF, Unpaywall DOI lookup).
    """
    out: Dict[str, Optional[str]] = {"pmcid": None, "doi": None}
    if not pmid or not str(pmid).strip().isdigit():
        return out
    try:
        r = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={
                "query": f"EXT_ID:{pmid} AND SRC:MED",
                "format": "json",
                "resultType": "lite",
                "pageSize": 1,
            },
            timeout=15,
            headers={"User-Agent": "EvidenceEngine/1.0"},
        )
        if r.status_code != 200:
            return out
        items = (r.json().get("resultList") or {}).get("result") or []
        if not items:
            return out
        item = items[0] or {}
        out["pmcid"] = item.get("pmcid") or None
        out["doi"] = item.get("doi") or None
        return out
    except Exception as e:
        print(f"[lookup_pmc_metadata] {pmid}: {e}")
        return out


def _lookup_pmcid_from_pmid(pmid: str) -> Optional[str]:
    """Back-compat wrapper around _lookup_pmc_metadata for the table-extraction
    code path that only needs the PMC ID."""
    return _lookup_pmc_metadata(pmid).get("pmcid")


def _fetch_epmc_full_text_xml(paper_id: str) -> Optional[bytes]:
    """Try several Europe PMC fullTextXML URL formats for the given ID.

    The fullTextXML endpoint accepts a PMC ID directly (`PMC1234567`), or a
    source-prefixed form (`MED/123/fullTextXML`). Plain PMIDs without source
    prefix do NOT work, so for those we look up the PMCID first.
    """
    if not paper_id:
        return None
    candidates: List[str] = []
    pid = str(paper_id).strip()
    if pid.upper().startswith("PMC"):
        candidates.append(pid.upper())
    elif pid.isdigit():
        # Probably a PMID — resolve to a PMC ID, otherwise we'll just 404.
        pmcid = _lookup_pmcid_from_pmid(pid)
        if pmcid:
            candidates.append(pmcid)
        candidates.append(f"MED/{pid}")
    else:
        candidates.append(pid)

    for cid in candidates:
        try:
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{cid}/fullTextXML"
            r = requests.get(url, timeout=20, headers={"User-Agent": "EvidenceEngine/1.0"})
            if r.status_code == 200 and r.content:
                return r.content
        except Exception as e:
            print(f"[fetch_epmc_full_text_xml] {cid}: {e}")
            continue
    return None


def _extract_epmc_tables(paper_id: str, extraction_type: str) -> List[Dict[str, Any]]:
    if not paper_id:
        return []
    xml = _fetch_epmc_full_text_xml(paper_id)
    if not xml:
        return []
    try:
        soup = BeautifulSoup(xml, "lxml-xml")
        out = []
        for i, tw in enumerate(soup.find_all("table-wrap")):
            label = tw.find("label")
            caption = tw.find("caption")
            rows = []
            for tr in tw.find_all("tr"):
                cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
                if cells:
                    rows.append(cells)
            if rows:
                out.append({
                    "title": label.get_text(strip=True) if label else f"Table {i + 1}",
                    "type": _classify_table(rows, extraction_type or ""),
                    "data": rows,
                    "caption": caption.get_text(strip=True) if caption else "",
                })
        return out
    except Exception as e:
        print(f"[extract_epmc_tables] parse error for {paper_id}: {e}")
        return []


@app.post("/api/extract/tables")
def extract_tables(req: ExtractTablesRequest):
    """Three-tier table extraction:
      1. Europe PMC JATS fullTextXML — open-access full-text, structured tables.
      2. HTML scrape of the supplied URL — works for journal landing pages
         that include rendered tables. Skipped for PubMed/EuPMC abstract URLs
         since those pages don't contain the article's tables.
      3. LLM extraction over title + abstract + (optional) caller-supplied
         full text. The caller's abstract/full_text fields are the difference
         between this tier finding anything and finding nothing.
    """
    pid = req.paper_id or ""
    if not pid and req.URL:
        m = re.search(r"/(PMC\d+|\d+)/?$", req.URL, re.IGNORECASE)
        if m:
            pid = m.group(1)

    tables: List[Dict[str, Any]] = []

    # Tier 1: Europe PMC JATS fullTextXML. Works for PMC and PubMed papers
    # that have a PMC mirror — _extract_epmc_tables handles the PMID -> PMCID
    # lookup transparently.
    if pid and req.Source in {"PubMed", "Europe PMC"}:
        tier1 = _extract_epmc_tables(pid, req.extraction_type or "")
        if tier1:
            tables.extend(tier1)
            print(f"[extract_tables] {pid} tier1 (EuPMC XML) -> {len(tier1)} tables")

    # Tier 2: HTML scrape — but only for URLs that are likely to render real
    # tables, not search/abstract landing pages. Skip pubmed.ncbi.nlm.nih.gov
    # and europepmc.org pages because those don't include the article's tables.
    if not tables and req.URL and req.URL.startswith("http"):
        host = req.URL.split("/", 3)[2].lower() if "://" in req.URL else ""
        skip_hosts = {"pubmed.ncbi.nlm.nih.gov", "europepmc.org", "www.ncbi.nlm.nih.gov"}
        if host not in skip_hosts:
            tier2 = _extract_html_tables(req.URL, req.extraction_type or "")
            if tier2:
                tables.extend(tier2)
                print(f"[extract_tables] {pid or req.URL[:60]} tier2 (HTML) -> {len(tier2)} tables")

    # Tier 3: LLM fallback over title + abstract + caller-supplied full text.
    # The caller-supplied content is what makes this tier useful — passing
    # only the title here was the original bug.
    if not tables:
        try:
            seed_parts: List[str] = [req.Title.strip()]
            if req.full_text:
                seed_parts.append(req.full_text[:14000].strip())
            elif req.abstract:
                seed_parts.append(req.abstract.strip())
            seed = "\n\n".join(p for p in seed_parts if p)
            ai_tables = AITableExtractor.extract_from_text(
                seed,
                resolve_for_thinking(req.model),
            )
            for i, t in enumerate(ai_tables or []):
                rows = [t.get("headers", [])] + t.get("rows", [])
                tables.append({
                    "title": t.get("label", f"Table {i + 1}"),
                    "type": _classify_table(rows, req.extraction_type or ""),
                    "data": rows,
                    "caption": t.get("caption", ""),
                })
            print(f"[extract_tables] {pid or 'no-id'} tier3 (LLM) -> {len(ai_tables or [])} tables (seed {len(seed)} chars)")
        except Exception as e:
            print(f"[extract_tables ai fallback] {e}")

    if not tables:
        print(f"[extract_tables] {pid or req.URL[:60]} -> no tables found (source={req.Source}, has_abstract={bool(req.abstract)}, has_fulltext={bool(req.full_text)})")

    return {"tables": tables}


# ---------------------------------------------------------------------------
# Quality assessment (heuristic, mirrors mockServices)
# ---------------------------------------------------------------------------


class QualityRequest(BaseModel):
    paper: PaperIn
    full_text: Optional[str] = None     # if available; falls back to abstract
    rubric_override: Optional[str] = None  # force a specific rubric ("RoB 2", "ROBINS-I", ...)
    model: Optional[str] = None


# ---------------------------------------------------------------------------
# Rubric registry — domain definitions per study-design rubric.
# Each domain pairs a signalling question with a one-line description so the
# LLM appraiser has enough structure to make a defensible judgment.
# ---------------------------------------------------------------------------

ROB2_DOMAINS = [
    {
        "id": "randomization",
        "name": "Bias arising from the randomization process",
        "signalling": (
            "Was the allocation sequence random and concealed? Were baseline "
            "differences between groups suggestive of a problem with the "
            "randomization?"
        ),
    },
    {
        "id": "deviations",
        "name": "Bias due to deviations from intended interventions",
        "signalling": (
            "Were participants and personnel aware of group assignment? Were "
            "there deviations from the intended intervention that affected "
            "outcomes? Was analysis appropriate (e.g., intention-to-treat)?"
        ),
    },
    {
        "id": "missing_data",
        "name": "Bias due to missing outcome data",
        "signalling": (
            "Were outcome data available for most participants? Was the "
            "proportion of missingness similar across groups? Was the missingness "
            "likely related to the true value of the outcome?"
        ),
    },
    {
        "id": "measurement",
        "name": "Bias in measurement of the outcome",
        "signalling": (
            "Was the outcome measurement method appropriate? Could measurement "
            "differ between intervention groups? Were outcome assessors blinded?"
        ),
    },
    {
        "id": "selection_reporting",
        "name": "Bias in selection of the reported result",
        "signalling": (
            "Was the analysis pre-specified (registered protocol, statistical "
            "analysis plan)? Were the reported results selected from multiple "
            "analyses or outcome measurements?"
        ),
    },
]

ROBINS_I_DOMAINS = [
    {
        "id": "confounding",
        "name": "Bias due to confounding",
        "signalling": (
            "Were important confounders identified and adjusted for? Was the "
            "method of adjustment appropriate (matching, regression, propensity "
            "scoring)? Were time-varying confounders handled?"
        ),
    },
    {
        "id": "selection",
        "name": "Bias in selection of participants",
        "signalling": (
            "Was selection into the study related to intervention or outcome? "
            "Was follow-up complete and did it start at intervention initiation?"
        ),
    },
    {
        "id": "classification",
        "name": "Bias in classification of interventions",
        "signalling": (
            "Were intervention groups clearly defined and consistently applied? "
            "Were misclassifications possible (e.g., from self-report)?"
        ),
    },
    {
        "id": "deviations",
        "name": "Bias due to deviations from intended interventions",
        "signalling": (
            "Were co-interventions balanced across groups? Did participants "
            "switch interventions in a way that biased the effect estimate?"
        ),
    },
    {
        "id": "missing_data",
        "name": "Bias due to missing data",
        "signalling": (
            "Were data on participants and outcomes reasonably complete? Were "
            "appropriate statistical methods used to handle missing data?"
        ),
    },
    {
        "id": "measurement",
        "name": "Bias in measurement of outcomes",
        "signalling": (
            "Was the outcome measure appropriate, applied consistently, and "
            "obtained blind to intervention status?"
        ),
    },
    {
        "id": "selection_reporting",
        "name": "Bias in selection of the reported result",
        "signalling": (
            "Were results selectively reported across multiple analyses, "
            "outcomes, or subgroups?"
        ),
    },
]

JBI_CROSS_SECTIONAL_DOMAINS = [
    {"id": "inclusion_criteria", "name": "Clear inclusion criteria",
     "signalling": "Were the criteria for inclusion in the sample clearly defined?"},
    {"id": "subjects_setting", "name": "Subjects and setting described in detail",
     "signalling": "Were the study subjects and the setting described in detail?"},
    {"id": "exposure_measurement", "name": "Valid and reliable exposure measurement",
     "signalling": "Was the exposure measured in a valid and reliable way?"},
    {"id": "outcome_measurement", "name": "Valid and reliable outcome measurement",
     "signalling": "Were the outcomes measured in a valid and reliable way?"},
    {"id": "confounding_identified", "name": "Confounders identified",
     "signalling": "Were confounding factors identified and strategies stated to deal with them?"},
    {"id": "statistical_analysis", "name": "Appropriate statistical analysis",
     "signalling": "Was the statistical analysis used appropriate to the data?"},
]

AMSTAR2_DOMAINS = [
    {"id": "pico_components", "name": "Research questions and inclusion criteria include the components of PICO",
     "signalling": "Did the SR's research questions and inclusion criteria include all PICO components?"},
    {"id": "protocol_registered", "name": "Protocol registered before review",
     "signalling": "Did the report contain explicit statement that review methods were established prior, and was the protocol registered?"},
    {"id": "study_designs_explained", "name": "Explanation for selection of study designs",
     "signalling": "Did the review authors explain their selection of the study designs for inclusion in the review?"},
    {"id": "search_comprehensive", "name": "Comprehensive literature search",
     "signalling": "Did the review authors use a comprehensive literature search strategy across multiple databases?"},
    {"id": "duplicate_screening", "name": "Duplicate study selection",
     "signalling": "Did the review authors perform study selection in duplicate?"},
    {"id": "duplicate_extraction", "name": "Duplicate data extraction",
     "signalling": "Did the review authors perform data extraction in duplicate?"},
    {"id": "excluded_studies_list", "name": "List of excluded studies with justification",
     "signalling": "Did the review authors provide a list of excluded studies and justify the exclusions?"},
    {"id": "included_studies_detail", "name": "Adequate description of included studies",
     "signalling": "Did the review authors describe the included studies in adequate detail?"},
    {"id": "rob_individual_assessed", "name": "Risk-of-bias assessed for individual studies",
     "signalling": "Did the review authors use a satisfactory technique for assessing the risk of bias in individual studies?"},
    {"id": "funding_sources", "name": "Funding sources reported for included studies",
     "signalling": "Did the review authors report on the sources of funding for the studies included in the review?"},
    {"id": "meta_analysis_appropriate", "name": "Appropriate statistical combination of results",
     "signalling": "If meta-analysis was performed, did the review authors use appropriate methods for statistical combination of results?"},
    {"id": "rob_in_interpretation", "name": "Risk-of-bias considered in interpretation",
     "signalling": "Did the review authors account for risk of bias in individual studies when interpreting/discussing the results of the review?"},
    {"id": "heterogeneity_discussed", "name": "Heterogeneity discussed",
     "signalling": "Did the review authors provide a satisfactory explanation for, and discussion of, any heterogeneity observed in the results?"},
    {"id": "publication_bias", "name": "Publication-bias investigated",
     "signalling": "Did the review authors investigate publication bias (small study bias)?"},
    {"id": "coi_reported", "name": "Conflicts of interest reported",
     "signalling": "Did the review authors report any potential sources of conflict of interest?"},
]

JBI_QUALITATIVE_DOMAINS = [
    {"id": "philosophical_congruity", "name": "Congruity between philosophical perspective and methodology",
     "signalling": "Is there congruity between the stated philosophical perspective and the research methodology?"},
    {"id": "methodology_objectives", "name": "Methodology aligned with objectives",
     "signalling": "Is there congruity between the methodology and the research question or objectives?"},
    {"id": "data_collection", "name": "Methodology aligned with data collection",
     "signalling": "Is there congruity between the methodology and the methods used to collect data?"},
    {"id": "representation_findings", "name": "Methodology aligned with findings representation",
     "signalling": "Is there congruity between the methodology and the representation and analysis of data?"},
    {"id": "researcher_position", "name": "Researcher's positionality stated",
     "signalling": "Has the researcher's influence on the research, and vice versa, been addressed?"},
    {"id": "participants_voice", "name": "Participants' voices represented",
     "signalling": "Are participants, and their voices, adequately represented?"},
    {"id": "ethical_approval", "name": "Ethical approval reported",
     "signalling": "Is the research ethical, according to current criteria, and is evidence of ethical approval provided?"},
]

RUBRIC_REGISTRY = {
    "RoB 2": {"applies_to": ["RCT"], "domains": ROB2_DOMAINS},
    "ROBINS-I": {"applies_to": ["Cohort", "Case-control", "Non-randomised"], "domains": ROBINS_I_DOMAINS},
    "JBI cross-sectional": {"applies_to": ["Cross-sectional"], "domains": JBI_CROSS_SECTIONAL_DOMAINS},
    "JBI qualitative": {"applies_to": ["Qualitative"], "domains": JBI_QUALITATIVE_DOMAINS},
    "AMSTAR 2": {"applies_to": ["Systematic review", "Meta-analysis"], "domains": AMSTAR2_DOMAINS},
}

# Map normalised study-design labels → rubric name.
DESIGN_TO_RUBRIC = {
    "rct": "RoB 2",
    "randomized controlled trial": "RoB 2",
    "randomised controlled trial": "RoB 2",
    "cluster rct": "RoB 2",
    "crossover rct": "RoB 2",
    "cohort": "ROBINS-I",
    "case-control": "ROBINS-I",
    "case control": "ROBINS-I",
    "non-randomised": "ROBINS-I",
    "non-randomized": "ROBINS-I",
    "quasi-experimental": "ROBINS-I",
    "cross-sectional": "JBI cross-sectional",
    "cross sectional": "JBI cross-sectional",
    "qualitative": "JBI qualitative",
    "mixed methods": "JBI qualitative",
    "systematic review": "AMSTAR 2",
    "meta-analysis": "AMSTAR 2",
    "meta analysis": "AMSTAR 2",
    "scoping review": "AMSTAR 2",
    "umbrella review": "AMSTAR 2",
}

JUDGMENT_VALUES = {"Low", "Some Concerns", "High", "No information", "Not applicable"}


def _detect_study_design(paper: PaperIn, full_text: str, model_name: str) -> Tuple[str, str]:
    """Return (design_label, raw_response) — design is one of the keys of
    DESIGN_TO_RUBRIC (or "Other" if unrecognised).
    """
    from langchain_core.messages import HumanMessage

    model = AIService.get_model(model_name)
    if not model:
        return "Other", ""

    text = full_text[:6000] if full_text else (paper.abstract or "")[:6000]
    prompt = f"""Classify the study DESIGN of this paper into ONE of these categories:

  - RCT (randomised / randomized controlled trial, including cluster and crossover variants)
  - Cohort (prospective or retrospective)
  - Case-control
  - Cross-sectional
  - Case series
  - Case report
  - Systematic review (with or without meta-analysis)
  - Meta-analysis
  - Qualitative (interview, focus group, ethnographic)
  - Mixed methods
  - Quasi-experimental (interrupted time series, controlled before-after)
  - Non-randomised (other intervention studies that are neither RCT nor observational)
  - Other (animal, in vitro, methodological, editorial, narrative review, opinion)

PAPER TITLE: {paper.title or "(untitled)"}

TEXT (abstract or available full text):
{text}

Return ONLY the category label, exactly as written above. No explanation, no JSON, no quotes.
"""
    try:
        r = model.invoke([HumanMessage(content=prompt)])
        raw = (r.content or "").strip()
        # Normalise: take the first non-empty line, strip quotes/markdown.
        first_line = (raw.splitlines() or [""])[0].strip().strip("'\"`*").strip()
        return first_line or "Other", raw
    except Exception as e:
        print(f"[quality] study-design detection error: {e}")
        return "Other", ""


def _resolve_rubric(design_label: str, override: Optional[str]) -> Tuple[str, List[Dict[str, str]]]:
    """Return (rubric_name, domains). If override is given and is a valid
    rubric name, use that; otherwise map from the detected study-design label.
    Falls back to JBI cross-sectional when nothing matches (safest broad rubric).
    """
    if override and override in RUBRIC_REGISTRY:
        rub = RUBRIC_REGISTRY[override]
        return override, rub["domains"]

    key = (design_label or "").strip().lower()
    rubric_name = DESIGN_TO_RUBRIC.get(key)
    if not rubric_name:
        # Fuzzy: try substring match against the keys.
        for k, v in DESIGN_TO_RUBRIC.items():
            if k in key or key in k:
                rubric_name = v
                break
    if not rubric_name:
        rubric_name = "JBI cross-sectional"
    return rubric_name, RUBRIC_REGISTRY[rubric_name]["domains"]


def _appraise_domains(
    paper: PaperIn,
    full_text: str,
    rubric_name: str,
    domains: List[Dict[str, str]],
    model_name: str,
) -> List[Dict[str, Any]]:
    """Run a batched per-domain LLM appraisal. Returns a list of domain judgments,
    each with judgment, rationale, supporting_quote, section.
    """
    from langchain_core.messages import HumanMessage

    model = AIService.get_model(model_name)
    if not model:
        return [{
            "id": d["id"], "name": d["name"],
            "judgment": "No information",
            "rationale": "Model unavailable.",
            "supporting_quote": "", "section": "",
        } for d in domains]

    text = full_text or (paper.abstract or "")
    has_full_text = bool(full_text and len(full_text) > len(paper.abstract or ""))

    domain_block = "\n".join(
        f"  - id: \"{d['id']}\"\n    name: \"{d['name']}\"\n    signalling_question: \"{d['signalling']}\""
        for d in domains
    )

    prompt = f"""You are a systematic-review methodologist performing a risk-of-bias appraisal
using the {rubric_name} rubric. For EACH of the domains below, render a judgment based ONLY on
what is stated in the paper text. Do not infer beyond what is written.

PAPER TITLE: {paper.title or "(untitled)"}

PAPER TEXT ({"full text available" if has_full_text else "abstract only — limited assessment"}):
{text[:14000]}

DOMAINS TO APPRAISE:
{domain_block}

For each domain, return:
  - "judgment": one of "Low" | "Some Concerns" | "High" | "No information"
       Use "No information" when the paper does not give you enough to decide
       (e.g. when only the abstract is available and methods detail is missing).
  - "rationale": 1-2 sentences explaining the judgment.
  - "supporting_quote": a SHORT exact quote from the paper that supports the
       judgment (≤ 200 chars). If no quote is available, return an empty string.
  - "section": where in the paper the quote was found
       ("Methods" | "Results" | "Discussion" | "Abstract" | "Other" | "" if no quote).

Return ONLY a JSON object with this shape:
{{
  "judgments": [
    {{
      "id": "<domain id>",
      "judgment": "Low" | "Some Concerns" | "High" | "No information",
      "rationale": "...",
      "supporting_quote": "...",
      "section": "Methods" | "Results" | "Discussion" | "Abstract" | "Other" | ""
    }},
    ...
  ]
}}

NEVER fabricate quotes that do not appear in the paper text. NEVER mark a domain
"Low" without a supporting quote unless the rubric explicitly allows it.
"""
    out: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    try:
        r = model.invoke([HumanMessage(content=prompt)])
        data = AIService._extract_json(r.content) or {}
        for j in (data.get("judgments") or []):
            if not isinstance(j, dict):
                continue
            did = str(j.get("id") or "").strip()
            judgment = str(j.get("judgment") or "No information").strip()
            if judgment not in JUDGMENT_VALUES:
                judgment = "No information"
            by_id[did] = {
                "id": did,
                "judgment": judgment,
                "rationale": str(j.get("rationale") or "").strip()[:600],
                "supporting_quote": str(j.get("supporting_quote") or "").strip()[:250],
                "section": str(j.get("section") or "").strip()[:30],
            }
    except Exception as e:
        print(f"[quality] domain appraisal error: {e}")

    # Always return one entry per requested domain, filling in "No information"
    # for any the model omitted.
    for d in domains:
        existing = by_id.get(d["id"])
        if existing:
            existing["name"] = d["name"]
            out.append(existing)
        else:
            out.append({
                "id": d["id"], "name": d["name"],
                "judgment": "No information",
                "rationale": "Domain not addressed in model response.",
                "supporting_quote": "", "section": "",
            })
    return out


def _aggregate_overall(domains: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Aggregate domain-level judgments to an overall RoB judgment.

    Follows the standard Cochrane logic:
      • Overall = High if ANY domain is High.
      • Overall = Some Concerns if ANY domain is Some Concerns (and none High).
      • Overall = Low only if EVERY domain is Low.
      • Overall = No information if every domain is No information.
    """
    judgments = [d["judgment"] for d in domains]
    if all(j == "No information" for j in judgments):
        return "No information", "All domains lacked sufficient information for an appraisal."
    if any(j == "High" for j in judgments):
        n = sum(1 for j in judgments if j == "High")
        return "High", f"{n} domain(s) judged High risk of bias."
    if any(j == "Some Concerns" for j in judgments):
        n = sum(1 for j in judgments if j == "Some Concerns")
        return "Some Concerns", f"{n} domain(s) raised some concerns."
    if all(j == "Low" for j in judgments):
        return "Low", "All domains judged Low risk of bias."
    return "Some Concerns", "Mixed domain judgments without any High risk."


@app.post("/api/quality/assess")
def quality_assess(req: QualityRequest):
    """Risk-of-bias appraisal using rubric appropriate to the detected study design.

    Pipeline:
      1. Detect study design from title + (full_text or abstract).
      2. Pick rubric: RCT → RoB 2, observational → ROBINS-I, cross-sectional →
         JBI, qualitative → JBI qualitative, SR/MA → AMSTAR 2 (override
         honoured if provided).
      3. Appraise each rubric domain via batched LLM call returning structured
         JSON with judgment, rationale, supporting_quote, section.
      4. Aggregate to an overall judgment per Cochrane rules.

    The legacy `score`/`rating`/`issues`/`highlightedAbstract` fields have
    been removed — the response shape is now centred on domain-level RoB
    judgments, which is what reviewers actually need.
    """
    paper = req.paper
    abs_text = paper.abstract or ""
    full_text = (req.full_text or "").strip() or abs_text

    model_name = resolve_for_thinking(req.model)

    design, _ = _detect_study_design(paper, full_text, model_name)
    rubric_name, domains_spec = _resolve_rubric(design, req.rubric_override)
    domain_results = _appraise_domains(paper, full_text, rubric_name, domains_spec, model_name)
    overall_judgment, overall_rationale = _aggregate_overall(domain_results)
    used_full_text = bool(req.full_text and len(req.full_text) > len(abs_text))

    return {
        "paper_id": paper.id,
        "title": paper.title,
        "source": paper.source,
        "url": paper.url,
        "abstract": abs_text,
        "study_design": design,
        "rubric": rubric_name,
        "domains": domain_results,
        "overall_judgment": overall_judgment,
        "overall_rationale": overall_rationale,
        "used_full_text": used_full_text,
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/api/models/local")
def list_local_models():
    """List models available in the local Ollama instance."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            return {"running": False, "models": [], "error": f"Ollama returned {r.status_code}"}
        data = r.json()
        models = [m.get("name") for m in data.get("models", []) if m.get("name")]
        return {"running": True, "models": models}
    except Exception as e:
        return {"running": False, "models": [], "error": str(e)}


@app.get("/api/health")
def health():
    # Surface any in-flight server-side tasks so the user can see what is
    # holding the Ollama queue (e.g., a long agentic-optimize run from a
    # previous click).
    with _cancel_lock:
        active = list(_cancel_events.keys())

    ollama: Dict[str, Any] = {"reachable": False, "loaded_models": []}
    try:
        r = requests.get("http://localhost:11434/api/ps", timeout=2)
        if r.status_code == 200:
            ollama["reachable"] = True
            ollama["loaded_models"] = [m.get("name") for m in r.json().get("models", [])]
    except Exception:
        pass

    return {
        "ok": True,
        "model_default": _default_model(),
        "providers": {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
        },
        "active_tasks": active,
        "ollama": ollama,
    }


@app.post("/api/tasks/cancel-all")
def cancel_all_tasks():
    """Signal every registered task to stop. Useful when something got
    orphaned by an old client (e.g., an AI Optimize started before the
    task-id wiring was in place)."""
    with _cancel_lock:
        ids = list(_cancel_events.keys())
        for ev in _cancel_events.values():
            ev.set()
    return {"canceled": ids, "count": len(ids)}


# ---------------------------------------------------------------------------
# Meta-analysis agent
# ---------------------------------------------------------------------------
from meta_analysis import (
    StudyEffect,
    compute_effect_size,
    pool as _ma_pool,
    extract_effect_size,
    subgroup_analysis as _ma_subgroup,
    leave_one_out as _ma_loo,
    cumulative_meta_analysis as _ma_cumulative,
    funnel_plot_data as _ma_funnel,
    egger_test as _ma_egger,
    begg_test as _ma_begg,
    trim_and_fill as _ma_trim_fill,
    meta_regression as _ma_metareg,
)
from dataclasses import asdict as _ma_asdict


class MetaExtractRequest(BaseModel):
    papers: List[PaperIn]
    outcome: str = ""                            # plain-English target outcome
    measure: str = ""                            # preferred effect measure hint
    model: Optional[str] = None                  # LLM for extraction
    # Per-paper full text (paper_id → text), opt-in. If absent, abstract only.
    full_texts: Dict[str, str] = Field(default_factory=dict)


@app.post("/api/meta/extract")
def meta_extract(req: MetaExtractRequest):
    """Run the meta-analysis extraction agent on a list of papers.

    Uses the platform's "thinking" model (Qwen/Claude/etc.) — NOT LEADS, which
    is fine-tuned for screening verdicts, not numerical extraction."""
    model_name = resolve_for_thinking(req.model)

    def _one(p: PaperIn) -> Dict[str, Any]:
        d = p.dict() if hasattr(p, "dict") else dict(p.__dict__)
        ft = req.full_texts.get(str(d.get("id") or "")) or None
        se = extract_effect_size(
            d, model_name=model_name,
            outcome_hint=req.outcome, measure_hint=req.measure,
            full_text=ft,
        )
        return _ma_asdict(se)

    out: List[Dict[str, Any]] = []
    workers = max(1, min(Config.PARALLEL_AGENT_WORKERS, len(req.papers) or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_one, p) for p in req.papers]):
            try:
                out.append(fut.result())
            except Exception as e:
                print(f"[meta_extract] worker error: {e}")
    return {"extractions": out, "model_used": model_name, "outcome": req.outcome}


class MetaPoolRequest(BaseModel):
    """Pool a set of pre-extracted (or user-edited) effect sizes.

    Accept the JSON shape returned by /api/meta/extract so the frontend can
    let the user edit individual numbers (or remove studies) and re-pool
    without re-running the LLM."""
    extractions: List[Dict[str, Any]]
    tau2_method: str = "DL"               # "DL" | "PM" | "REML"
    use_knapp_hartung: bool = False


def _hydrate_studies(rows: List[Dict[str, Any]]) -> List[StudyEffect]:
    """Convert API JSON rows back into StudyEffect dataclasses, re-computing
    yi/vi from raw inputs whenever the caller may have edited them."""
    out: List[StudyEffect] = []
    for d in rows:
        try:
            se = StudyEffect(**{k: v for k, v in d.items() if k in StudyEffect.__dataclass_fields__})
            # Always recompute from raw inputs if any of those changed.
            if se.effect_measure and se.effect_measure.upper() != "GENERIC":
                # Clear computed fields so they get re-derived from inputs.
                se.yi = None
                se.vi = None
                se.se = None
                se.ci_low = None
                se.ci_high = None
                compute_effect_size(se)
            elif se.yi is None or se.vi is None:
                compute_effect_size(se)
            out.append(se)
        except Exception as e:
            out.append(StudyEffect(
                paper_id=str(d.get("paper_id", "")),
                title=str(d.get("title", "")),
                error=f"row could not be parsed: {e}",
            ))
    return out


@app.post("/api/meta/pool")
def meta_pool(req: MetaPoolRequest):
    studies = _hydrate_studies(req.extractions)
    return _ma_pool(studies, tau2_method=req.tau2_method, use_knapp_hartung=req.use_knapp_hartung)


class MetaAnalysisRunRequest(BaseModel):
    """Run subgroup analysis, sensitivity analyses, publication-bias
    diagnostics, and meta-regression in a single call. The frontend can
    cache the result and switch tabs without re-hitting the backend."""
    extractions: List[Dict[str, Any]]
    tau2_method: str = "DL"
    use_knapp_hartung: bool = False


@app.post("/api/meta/run")
def meta_run(req: MetaAnalysisRunRequest):
    studies = _hydrate_studies(req.extractions)
    return {
        "pool": _ma_pool(studies, tau2_method=req.tau2_method, use_knapp_hartung=req.use_knapp_hartung),
        "subgroup": _ma_subgroup(studies, tau2_method=req.tau2_method),
        "leave_one_out": _ma_loo(studies, tau2_method=req.tau2_method),
        "cumulative": _ma_cumulative(studies, tau2_method=req.tau2_method),
        "funnel": _ma_funnel(studies, tau2_method=req.tau2_method),
        "egger": _ma_egger(studies),
        "begg": _ma_begg(studies),
        "trim_fill": _ma_trim_fill(studies, tau2_method=req.tau2_method),
        "meta_regression": _ma_metareg(studies, tau2_method=req.tau2_method),
    }


# ── Writing assistant ──────────────────────────────────────────────────────────

class WritingSummaryRequest(BaseModel):
    # Search configuration
    databases: List[str] = []
    unified_query: str = ""
    per_db_queries: Dict[str, str] = {}
    search_date: str = ""
    # Result funnel
    db_counts: Dict[str, int] = {}
    total_identified: Optional[int] = None
    duplicates_removed: Optional[int] = None
    after_dedup: Optional[int] = None
    screened_abstracts: Optional[int] = None
    included_abstracts: Optional[int] = None
    fulltext_assessed: Optional[int] = None
    included_final: Optional[int] = None
    # Review context
    pico: Dict[str, str] = {}
    inclusion_criteria: List[str] = []
    exclusion_criteria: List[str] = []
    goal: str = ""
    model: str = ""


@app.post("/api/writing/summary")
def writing_summary(req: WritingSummaryRequest):
    """Generate a PRISMA-compliant search strategy methods paragraph."""
    from langchain_core.messages import HumanMessage

    model = AIService.get_model(resolve_for_thinking(req.model))
    if not model:
        raise HTTPException(status_code=503, detail="No model available")

    # Build database + query block
    db_lines = []
    for db in req.databases:
        q = req.per_db_queries.get(db) or req.unified_query
        count = req.db_counts.get(db)
        count_str = f" ({count:,} records)" if count else ""
        db_lines.append(f"  • {db}{count_str}: {q[:200] if q else '(same as unified query)'}")
    db_block = "\n".join(db_lines) if db_lines else "  (no databases specified)"

    pico_block = ""
    if req.pico:
        pico_block = (
            f"PICO:\n"
            f"  P: {req.pico.get('population','')}\n"
            f"  I: {req.pico.get('intervention','')}\n"
            f"  C: {req.pico.get('comparator','')}\n"
            f"  O: {req.pico.get('outcome','')}"
        )

    ic_block = ""
    if req.inclusion_criteria:
        ic_block = "Inclusion criteria:\n" + "\n".join(f"  - {c}" for c in req.inclusion_criteria)
    ec_block = ""
    if req.exclusion_criteria:
        ec_block = "Exclusion criteria:\n" + "\n".join(f"  - {c}" for c in req.exclusion_criteria)

    # PRISMA funnel numbers
    funnel_parts = []
    if req.total_identified is not None:
        funnel_parts.append(f"Records identified: {req.total_identified:,}")
    if req.duplicates_removed:
        funnel_parts.append(f"Duplicates removed: {req.duplicates_removed:,}")
    if req.after_dedup is not None:
        funnel_parts.append(f"Records after deduplication: {req.after_dedup:,}")
    if req.screened_abstracts is not None:
        funnel_parts.append(f"Abstracts screened: {req.screened_abstracts:,}")
    if req.included_abstracts is not None:
        funnel_parts.append(f"Included after abstract screening: {req.included_abstracts:,}")
    if req.fulltext_assessed is not None:
        funnel_parts.append(f"Full texts assessed: {req.fulltext_assessed:,}")
    if req.included_final is not None:
        funnel_parts.append(f"Final included studies: {req.included_final:,}")
    funnel_block = " → ".join(funnel_parts) if funnel_parts else ""

    prompt = f"""You are writing the Information Sources and Search Strategy section of a systematic review methods section (PRISMA 2020 compliant).

Review question: {req.goal or '(not specified)'}

{pico_block}

{ic_block}

{ec_block}

Databases searched (with record counts and query strings):
{db_block}

Search date: {req.search_date or 'not recorded'}

PRISMA screening funnel:
{funnel_block or '(screening counts not yet available)'}

Write a single, well-structured paragraph (150–250 words) for the Methods section of a manuscript. It must:
1. Name every database searched and the date the search was conducted
2. Describe the search strategy (keywords, Boolean operators, MeSH/controlled vocabulary terms if identifiable from the query strings)
3. State any date or language restrictions if apparent from the queries
4. Briefly describe the screening process (title/abstract screening, full-text review) and the PRISMA record counts
5. Mention the inclusion and exclusion criteria applied

Write in past tense, third person, passive voice where appropriate. Do not invent details.
Output only the paragraph — no headings, no bullet points, no preamble."""

    resp = model.invoke([HumanMessage(content=prompt)])
    return {"summary": resp.content.strip()}
