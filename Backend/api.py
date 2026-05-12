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
    return os.getenv("DEFAULT_MODEL", Config.DEFAULT_MODEL or "claude-sonnet-4-6")


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


@app.post("/api/pico/infer", response_model=Analysis)
def pico_infer(req: InferRequest):
    model_name = req.model or _default_model()
    data = AIService.infer_pico_and_query(req.input, model_name, req.previous_goal or "")
    pico = PICOCriteria(
        population=data.get("p", ""),
        intervention=data.get("i", ""),
        comparator=data.get("c", ""),
        outcome=data.get("o", ""),
    )
    try:
        query = AIService.generate_mesh_query(pico, model_name)
    except Exception as e:
        print(f"[pico_infer] mesh query failed: {e}")
        query = ""
    return Analysis(
        p=data.get("p", ""),
        i=data.get("i", ""),
        c=data.get("c", ""),
        o=data.get("o", ""),
        inclusion=list(data.get("inclusion", []) or []),
        exclusion=list(data.get("exclusion", []) or []),
        query=query or "",
    )


class FormalQuestionRequest(BaseModel):
    pico: PicoIn
    model: Optional[str] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)


@app.post("/api/pico/formal-question")
def pico_formal_question(req: FormalQuestionRequest):
    q = AIService.generate_formal_question(_to_pico(req.pico), req.model or _default_model(), req.history)
    return {"question": q}


class SummaryRequest(BaseModel):
    goal: str
    papers: List[PaperIn] = Field(default_factory=list)
    model: Optional[str] = None


def _plain_summary(goal: str, papers: List[BackendPaper], model_name: str) -> str:
    """Plain-prose comprehensive summary (no HTML, no markdown fences).

    Instructs the model to cite ONLY papers that are directly relevant to the
    research goal — irrelevant hits from a broad MeSH search are dropped.
    """
    if not papers:
        return ""
    from langchain_core.messages import HumanMessage

    model = AIService.get_model(model_name)
    if not model:
        return ""

    subset = papers[:10]
    ctx = ""
    for idx, p in enumerate(subset):
        ctx += f"[{idx + 1}] {p.title}\n    Source: {p.source}\n    Abstract: {(p.abstract or '')[:400]}\n\n"

    prompt = f"""You are an expert systematic review analyst. Build a COHESIVE evaluation of the
research question, using only papers that are directly relevant. Ignore papers that are off-topic
or only tangentially related — do not mention them, do not cite them.

RESEARCH GOAL: {goal}

LITERATURE ({len(subset)} papers, numbered [1]-[{len(subset)}]):
{ctx}

Structure the response with exactly these three plain-text section headers, each followed by a
blank line, in this order:

Research landscape overview
Arguments supporting the research question
Arguments against or challenging the research question

Under each header write 2-5 sentences (or 3-5 bullet points for the supporting/against sections).
Cite ONLY directly relevant papers inline as [1], [2], etc. — do not invent citations.

If NONE of the papers are directly relevant, write a single short paragraph under "Research
landscape overview" stating that the initial search returned no directly relevant evidence and
recommending how to broaden or narrow the search. Leave the two argument sections empty (just the
header, no content).

Do NOT include a reference list. Do NOT use HTML, markdown bold/italics, or code fences. Plain
text with dashes for bullets is fine.
"""
    try:
        r = model.invoke([HumanMessage(content=prompt)])
        return (r.content or "").strip()
    except Exception as e:
        print(f"[plain_summary] {e}")
        return ""


_CITE_RE = re.compile(r"\[(\d+)\]")

_RELEVANCE_STOP = {
    "the", "and", "or", "in", "of", "a", "an", "is", "are", "was", "were", "on", "to", "for",
    "with", "between", "from", "how", "what", "why", "does", "do", "this", "that", "help", "me",
    "understand", "relationship", "study", "studies", "paper", "research", "patients", "effect",
    "effects", "result", "results", "based", "using", "using", "their", "have", "has", "had",
    "been", "into", "than", "more", "less", "such", "also", "but", "not", "can", "could", "may",
    "might", "would", "should", "us", "we", "you", "they", "its", "it's", "use", "used", "via",
}


def _goal_terms(goal: str) -> List[str]:
    return [t for t in re.findall(r"\w{4,}", (goal or "").lower()) if t not in _RELEVANCE_STOP]


def _filter_for_relevance(papers: List[BackendPaper], goal: str, top_n: int = 10) -> List[BackendPaper]:
    """Keep only papers whose title/abstract share key terms with the goal, ranked by overlap."""
    if not papers:
        return []
    terms = _goal_terms(goal)
    if not terms:
        return papers[:top_n]

    scored: List[Tuple[int, BackendPaper]] = []
    for p in papers:
        title_lower = (p.title or "").lower()
        abs_lower = (p.abstract or "").lower()
        title_hits = sum(1 for t in terms if t in title_lower)
        abs_hits = sum(1 for t in terms if t in abs_lower)
        score = title_hits * 3 + abs_hits
        if score > 0:
            scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_n]]


def _filter_and_renumber(summary: str, references: list) -> Tuple[str, list]:
    """Keep only references actually cited in the summary, renumbered 1..N in order of first use."""
    seen: List[int] = []
    for m in _CITE_RE.finditer(summary):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    if not seen:
        return summary, []

    mapping = {orig: new for new, orig in enumerate(seen, start=1)}

    def _repl(m: re.Match) -> str:
        n = int(m.group(1))
        return f"[{mapping[n]}]" if n in mapping else m.group(0)

    renumbered = _CITE_RE.sub(_repl, summary)
    kept = [references[n - 1] for n in seen if 1 <= n <= len(references)]
    return renumbered, kept


@app.post("/api/pico/summary")
def pico_summary(req: SummaryRequest):
    bps = [_to_backend_paper(p) for p in req.papers]
    if not bps:
        return {"summary": "", "references": []}
    # Drop off-topic papers before sending to the LLM — saves tokens and gives
    # the model only candidates worth reasoning about.
    relevant = _filter_for_relevance(bps, req.goal, top_n=10)
    if not relevant:
        return {
            "summary": (
                "Research landscape overview\n\nThe initial search returned papers that do not "
                "appear directly relevant to this question. Consider broadening the search terms, "
                "removing restrictive MeSH filters, or using a different combination of population "
                "and intervention keywords.\n\nArguments supporting the research question\n\n"
                "Arguments against or challenging the research question\n"
            ),
            "references": [],
        }
    summary = _plain_summary(req.goal, relevant, req.model or _default_model())
    references = [
        {"title": (p.title or "").strip(), "url": p.url, "source": p.source, "id": str(p.id)}
        for p in relevant
    ]
    summary, references = _filter_and_renumber(summary, references)
    return {"summary": summary, "references": references}


class RefinementRequest(BaseModel):
    goal: str
    papers: List[PaperIn] = Field(default_factory=list)
    model: Optional[str] = None


@app.post("/api/pico/suggestions")
def pico_suggestions(req: RefinementRequest):
    bps = [_to_backend_paper(p) for p in req.papers]
    suggs = AIService.get_refinement_suggestions(req.goal, bps, req.model or _default_model())
    return {"suggestions": list(suggs or [])}


class AdversarialRequest(BaseModel):
    pico: PicoIn
    model: Optional[str] = None


@app.post("/api/pico/adversarial")
def pico_adversarial(req: AdversarialRequest):
    q = AIService.generate_adversarial_query(_to_pico(req.pico), req.model or _default_model())
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
        model = AIService.get_model(req.model or _default_model())
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
    """Identify the single PICO element that is most under-specified and suggest a sharper value."""
    from langchain_core.messages import HumanMessage
    model = AIService.get_model(req.model or _default_model())
    if not model:
        return {"field": None, "current": "", "suggested": "", "reason": "Model unavailable."}

    prompt = f"""You are a clinical research methodologist reviewing a PICO breakdown for a
systematic review. Identify the ONE element that is most under-specified, ambiguous, or
methodologically weak, and propose a sharper replacement for that element only.

RESEARCH GOAL: {req.goal}

CURRENT PICO:
  Population: {req.pico.population}
  Intervention: {req.pico.intervention}
  Comparator: {req.pico.comparator}
  Outcome: {req.pico.outcome}

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
    try:
        r = model.invoke([HumanMessage(content=prompt)])
        raw = (r.content or "").strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {"field": None, "current": "", "suggested": "", "reason": "Could not parse model response."}
        data = _json.loads(m.group(0))
        field = str(data.get("field", "")).strip().lower()
        if field not in {"population", "intervention", "comparator", "outcome"}:
            return {"field": None, "current": "", "suggested": "", "reason": "Model returned an invalid field."}
        return {
            "field": field,
            "current": str(data.get("current", "")).strip(),
            "suggested": str(data.get("suggested", "")).strip(),
            "reason": str(data.get("reason", "")).strip(),
        }
    except Exception as e:
        print(f"[pico_refine] {e}")
        return {"field": None, "current": "", "suggested": "", "reason": f"Refine error: {e}"}


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


@app.post("/api/screen/abstract")
def screen_abstract(req: ScreenAbstractRequest):
    paper = _to_backend_paper(req.paper)
    pico = _to_pico(req.pico)
    # Make criteria available to legacy functions that read session_state
    _ss["inclusion_list"] = list(req.inclusion or [])
    _ss["exclusion_list"] = list(req.exclusion or [])
    raw = AIService.screen_paper(paper, pico, req.model or _default_model(), req.inclusion, req.exclusion)
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
    model_name = req.model or _default_model()

    def _one(p_in: PaperIn) -> Dict[str, Any]:
        bp = _to_backend_paper(p_in)
        raw = AIService.screen_paper(bp, pico, model_name, req.inclusion, req.exclusion)
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
    raw = AIService.screen_full_text(paper_dict, pico, req.model or _default_model())

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
    model_name = req.model or _default_model()
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
    model_name = req.model or _default_model()
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
                req.model or _default_model(),
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


# ---------------------------------------------------------------------------
# Benchmark dashboard endpoints
# ---------------------------------------------------------------------------
# The benchmark CLI (`benchmark/run_benchmark.py`) writes metrics.csv +
# predictions.csv + summary.md into `benchmark/reports/<run_id>/`. These
# endpoints expose those files to the dashboard so the user can browse
# comparisons in the UI without leaving the app.

import csv as _csv
from pathlib import Path as _Path

_BENCHMARK_REPORTS = _Path(__file__).resolve().parent.parent / "benchmark" / "reports"


def _list_run_ids() -> List[str]:
    if not _BENCHMARK_REPORTS.exists():
        return []
    return sorted(
        [p.name for p in _BENCHMARK_REPORTS.iterdir() if p.is_dir() and (p / "metrics.csv").exists()],
        reverse=True,
    )


def _read_csv(path: _Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        reader = _csv.DictReader(fh)
        rows: List[Dict[str, Any]] = []
        for row in reader:
            # Coerce numeric-looking fields
            out: Dict[str, Any] = {}
            for k, v in row.items():
                try:
                    if v == "" or v is None:
                        out[k] = None
                    elif v.lower() in {"true", "false"}:
                        out[k] = v.lower() == "true"
                    else:
                        f = float(v)
                        out[k] = int(f) if f.is_integer() and "." not in v else f
                except (ValueError, AttributeError):
                    out[k] = v
            rows.append(out)
        return rows


@app.get("/api/benchmark/runs")
def benchmark_runs():
    """List all available benchmark runs (newest first)."""
    runs = []
    for run_id in _list_run_ids():
        metrics_path = _BENCHMARK_REPORTS / run_id / "metrics.csv"
        try:
            rows = _read_csv(metrics_path)
            datasets = sorted({str(r.get("dataset")) for r in rows if r.get("dataset")})
            arches = sorted({str(r.get("architecture")) for r in rows if r.get("architecture")})
            models = sorted({str(r.get("model_tier")) for r in rows if r.get("model_tier")})
            runs.append({
                "id": run_id,
                "datasets": datasets,
                "architectures": arches,
                "model_tiers": models,
                "cell_count": len(rows),
            })
        except Exception as e:
            print(f"[benchmark_runs] skipping {run_id}: {e}")
    return {"runs": runs}


@app.get("/api/benchmark/runs/{run_id}")
def benchmark_run(run_id: str):
    """Return all metrics + (optional) predictions + summary for a single run."""
    folder = _BENCHMARK_REPORTS / run_id
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    metrics = _read_csv(folder / "metrics.csv")
    predictions = _read_csv(folder / "predictions.csv")
    summary_path = folder / "summary.md"
    summary = summary_path.read_text() if summary_path.exists() else ""
    return {
        "id": run_id,
        "metrics": metrics,
        "predictions": predictions,
        "summary": summary,
    }


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
