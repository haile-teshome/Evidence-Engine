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

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import Config
from models import Paper as BackendPaper, PICOCriteria
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
    data = AIService.infer_pico_and_query(req.input, model_name, req.previous_goal or "")
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

    # Use a wider slice — the home rerank already auto-filtered to the relevant
    # set, so almost everything here should be useful.
    subset = papers[:25]
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
    """Remove citation markers that point outside the references range.

    The model occasionally hallucinates citation numbers beyond what was
    actually provided in the prompt. Strip those so the reader never sees a
    [N] that doesn't exist in the references list. All valid citations and
    the original numbering are preserved as-is so they line up with the
    full reference list shown in the UI.
    """
    def _repl(m: "re.Match[str]") -> str:
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= n_refs else ""
    return _CITE_RE.sub(_repl, summary)


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
    references = [
        {"title": (p.title or "").strip(), "url": p.url, "source": p.source, "id": str(p.id)}
        for p in grouped_papers
    ]
    summary = _strip_invalid_citations(summary, len(references))
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
# Search / data aggregation
# ---------------------------------------------------------------------------


class FetchAllRequest(BaseModel):
    query: str
    sources: List[str]
    max_per_source: int = 10
    limit: Optional[int] = None


@app.post("/api/papers/fetch")
def papers_fetch(req: FetchAllRequest):
    papers, counts = DataAggregator.fetch_all(
        req.query, req.sources, max_per_source=req.max_per_source, uploaded_files=None, limit=req.limit
    )
    return {
        "papers": [_paper_to_dict(p) for p in papers],
        "sourceCounts": counts,
    }


class SimulateYieldRequest(BaseModel):
    query: str
    sources: List[str]


@app.post("/api/simulation/yield")
def simulation_yield(req: SimulateYieldRequest):
    counts = DataAggregator.simulate_yield(req.query, req.sources)
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


def _normalize_abstract_decision(raw: Dict[str, Any], inclusion: List[str], exclusion: List[str], paper: PaperIn) -> Dict[str, Any]:
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

    return {
        "paper_id": paper.id,
        "Source": paper.source,
        "Title": paper.title,
        "URL": paper.url,
        "Abstract": abstract,
        "Decision": decision_upper,
        "Reason": reason or ("Meets inclusion criteria" if decision_upper == "INCLUDE" else "Excluded"),
        "Agent_Trace": agent_trace,
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
    return _normalize_abstract_decision(raw, req.inclusion, req.exclusion, req.paper)


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

    def _one(p_in: PaperIn) -> Dict[str, Any]:
        bp = _to_backend_paper(p_in)
        raw = _screen_one(bp, pico, model_name, req.inclusion, req.exclusion)
        return _normalize_abstract_decision(raw, req.inclusion, req.exclusion, p_in)

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
    MIN_KEPT = 5         # minimum number of papers above the cut, if available
    GAP_THRESHOLD = 0.10  # minimum gap size that counts as a natural break

    if not scores:
        return 0.0, "empty corpus"

    positive = sorted([s for s in scores if s >= 0.0], reverse=True)
    if not positive:
        return 0.0, "no papers scored net-positive across PICO"

    if len(positive) <= MIN_KEPT:
        return min(positive), f"small positive corpus ({len(positive)} papers) — keep all"

    # Search for the largest gap that leaves at least MIN_KEPT papers above it.
    # i is the index of the score *below* the gap; the gap separates positive[i-1]
    # (kept) from positive[i] (dropped). So we need i >= MIN_KEPT.
    max_search_idx = max(MIN_KEPT, len(positive) // 2) + 1
    best_gap = 0.0
    best_idx = -1
    for i in range(MIN_KEPT, min(max_search_idx, len(positive))):
        gap = positive[i - 1] - positive[i]
        if gap > best_gap:
            best_gap = gap
            best_idx = i

    if best_idx >= 0 and best_gap >= GAP_THRESHOLD:
        cut = positive[best_idx - 1]
        return cut, (
            f"natural relevance break: gap of {best_gap:+.2f} between scores "
            f"{positive[best_idx - 1]:+.2f} and {positive[best_idx]:+.2f}, "
            f"keeping {best_idx} papers"
        )

    # No significant gap — distribution is roughly uniform. Keep top half, with
    # a soft floor of +0.10 to drop borderline scores when nothing stands out.
    median_score = positive[len(positive) // 2]
    soft = max(0.10, median_score)
    return soft, f"uniform distribution — keep top half above {soft:+.2f}"


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

    return {
        "paper_id": req.paper.id,
        "Title": req.paper.title,
        "URL": req.paper.url,
        "Source": req.paper.source,
        "Abstract": req.paper.abstract,
        "Decision": decision,
        "Reason": str(raw.get("reason", "")),
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

    return {"citations": out}


# ---------------------------------------------------------------------------
# Full-text fetch (Europe PMC + Unpaywall)
# ---------------------------------------------------------------------------


class FullTextRequest(BaseModel):
    Title: str
    URL: str
    Source: str
    paper_id: Optional[str] = None


def _fetch_epmc_fulltext(paper_id: str) -> Optional[str]:
    if not paper_id:
        return None
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{paper_id}/fullTextXML"
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "EvidenceEngine/1.0"})
        if r.status_code == 200 and r.text and "<" in r.text:
            soup = BeautifulSoup(r.content, "lxml-xml")
            body = soup.find("body") or soup
            text = body.get_text(separator="\n", strip=True)
            return text if text and len(text) > 200 else None
    except Exception as e:
        print(f"[epmc_fulltext] {e}")
    return None


@app.post("/api/fulltext/fetch")
def fulltext_fetch(req: FullTextRequest):
    pid = req.paper_id or ""
    if not pid and req.URL:
        m = re.search(r"/(\d+)/?$", req.URL)
        if m:
            pid = m.group(1)
    text = _fetch_epmc_fulltext(pid) if pid else None
    if text:
        return {"status": "found", "text": text, "source": req.Source or "Europe PMC"}

    # Try HTML scrape fallback for any URL
    if req.URL and req.URL.startswith("http"):
        try:
            r = requests.get(req.URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                soup = BeautifulSoup(r.content, "html.parser")
                for s in soup(["script", "style", "nav", "footer", "header"]):
                    s.decompose()
                text = soup.get_text(separator="\n", strip=True)
                if text and len(text) > 500:
                    return {"status": "found", "text": text[:50000], "source": req.Source or "HTML"}
        except Exception as e:
            print(f"[fulltext_fetch html] {e}")

    return {"status": "missing", "reason": "Full text not retrievable from available sources."}


# ---------------------------------------------------------------------------
# Text extraction (heuristic + LLM-friendly)
# ---------------------------------------------------------------------------


class ExtractTextRequest(BaseModel):
    text: str
    query: str


@app.post("/api/extract/text")
def extract_text(req: ExtractTextRequest):
    text = req.text or ""
    query = req.query or ""
    if not text:
        return {"summary": "No full text available.", "spans": [], "values": []}

    tokens = [t for t in re.split(r"\s+", query.lower()) if t]
    spans: List[Dict[str, int]] = []
    for m in re.finditer(r"[^.!?\n]+[.!?]", text):
        sent = m.group(0)
        lo = sent.lower()
        score = sum(1 for t in tokens if t in lo)
        if score >= max(1, math.ceil(len(tokens) / 3)):
            spans.append({"start": m.start(), "end": m.end()})

    values: List[Dict[str, str]] = []
    patterns = [
        ("measurement", re.compile(r"([0-9]+\.?[0-9]*\s?(%|years|weeks|participants))", re.I)),
        ("p-value", re.compile(r"p\s*=\s*[0-9.]+", re.I)),
        ("confidence interval", re.compile(r"95%\s*CI\s*[0-9.\-,\s]+", re.I)),
        ("effect estimate", re.compile(r"(HR|RR|OR)\s*[0-9.]+", re.I)),
    ]
    for label, pat in patterns:
        for m in pat.finditer(text):
            if len(values) >= 12:
                break
            if spans and not any(s["start"] <= m.start() <= s["end"] for s in spans):
                continue
            values.append({
                "field": label,
                "value": m.group(0),
                "quote": text[max(0, m.start() - 30): m.end() + 30],
            })

    summary = (
        f"No matching evidence for \"{query}\" was found in this paper."
        if not spans
        else f"Found {len(spans)} relevant passage{'s' if len(spans) > 1 else ''} addressing \"{query}\"."
    )
    return {"summary": summary, "spans": spans, "values": values}


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


def _extract_epmc_tables(paper_id: str, extraction_type: str) -> List[Dict[str, Any]]:
    if not paper_id:
        return []
    try:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{paper_id}/fullTextXML"
        r = requests.get(url, timeout=20, headers={"User-Agent": "EvidenceEngine/1.0"})
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.content, "lxml-xml")
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
        print(f"[extract_epmc_tables] {e}")
        return []


@app.post("/api/extract/tables")
def extract_tables(req: ExtractTablesRequest):
    pid = req.paper_id or ""
    if not pid and req.URL:
        m = re.search(r"/(\d+)/?$", req.URL)
        if m:
            pid = m.group(1)

    tables: List[Dict[str, Any]] = []
    if pid and req.Source in {"PubMed", "Europe PMC"}:
        tables.extend(_extract_epmc_tables(pid, req.extraction_type or ""))
    if not tables and req.URL and req.URL.startswith("http"):
        tables.extend(_extract_html_tables(req.URL, req.extraction_type or ""))

    # LLM fallback over abstract / full-text if still empty
    if not tables:
        try:
            ai_tables = AITableExtractor.extract_from_text(
                req.Title + "\n",  # minimal seed; caller can resend with fulltext
                resolve_for_thinking(req.model),
            )
            for i, t in enumerate(ai_tables or []):
                tables.append({
                    "title": t.get("label", f"Table {i + 1}"),
                    "type": _classify_table([t.get("headers", [])] + t.get("rows", []), req.extraction_type or ""),
                    "data": [t.get("headers", [])] + t.get("rows", []),
                    "caption": t.get("caption", ""),
                })
        except Exception as e:
            print(f"[extract_tables ai fallback] {e}")

    return {"tables": tables}


# ---------------------------------------------------------------------------
# Quality assessment (heuristic, mirrors mockServices)
# ---------------------------------------------------------------------------


class QualityRequest(BaseModel):
    paper: PaperIn


FLAG_PATTERNS = [
    (re.compile(r"retract", re.I), "Retraction language"),
    (re.compile(r"predator", re.I), "Predatory venue"),
    (re.compile(r"conflict of interest|undisclosed", re.I), "Conflict of interest"),
    (re.compile(r"funded by|sponsored by", re.I), "Funding disclosure"),
    (re.compile(r"\bn\s*=\s*([1-9]|[1-2]\d)\b", re.I), "Very small sample size"),
    (re.compile(r"preliminary|pilot study", re.I), "Preliminary / pilot data"),
    (re.compile(r"no significant|not significant", re.I), "Null / underpowered finding"),
]

SEVERITY_WEIGHTS = {"high": 25, "medium": 12, "low": 5}


@app.post("/api/quality/assess")
def quality_assess(req: QualityRequest):
    p = req.paper
    abs_text = p.abstract or ""
    issues: List[Dict[str, str]] = []
    lo = abs_text.lower()

    if len(abs_text) < 200:
        issues.append({"severity": "high", "category": "Incomplete Abstract",
                       "message": "Abstract is unusually short — full methods/results may be missing.",
                       "evidence": abs_text[:80]})
    if "method" not in lo:
        issues.append({"severity": "medium", "category": "Missing Methods",
                       "message": "No explicit Methods section detected in the abstract."})
    if "result" not in lo and "conclus" not in lo:
        issues.append({"severity": "medium", "category": "Missing Results",
                       "message": "No Results or Conclusions language found."})
    if not re.search(r"(p\s*[<=>]\s*0\.\d+|95%\s*ci|confidence interval|n\s*=\s*\d+)", abs_text, re.I):
        issues.append({"severity": "low", "category": "Statistical Reporting",
                       "message": "No explicit statistical results (p-values, CIs, sample sizes) detected."})
    if not p.url or not re.match(r"^https?:", p.url):
        issues.append({"severity": "high", "category": "Missing Identifier",
                       "message": "No valid URL/DOI for this record."})
    if p.year and p.year < 2015:
        issues.append({"severity": "low", "category": "Older Publication",
                       "message": f"Published in {p.year} — may predate current guidelines."})

    # Highlighted abstract
    sentences = re.split(r"(?<=[.!?])\s+", abs_text)
    highlighted = []
    for s in sentences:
        hit = next(((pat, reason) for pat, reason in FLAG_PATTERNS if pat.search(s)), None)
        if hit:
            highlighted.append({"text": s + " ", "flagged": True, "reason": hit[1]})
        else:
            highlighted.append({"text": s + " ", "flagged": False})

    score = max(0, 100 - sum(SEVERITY_WEIGHTS[i["severity"]] for i in issues))
    rating = "Excellent" if score >= 85 else "Good" if score >= 70 else "Fair" if score >= 50 else "Poor"

    return {
        "paper_id": p.id,
        "title": p.title,
        "source": p.source,
        "url": p.url,
        "abstract": abs_text,
        "score": score,
        "rating": rating,
        "issues": issues,
        "highlightedAbstract": highlighted,
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
