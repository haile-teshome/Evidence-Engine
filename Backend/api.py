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

# A source checkout keeps its own Backend/.env. An installed app cannot write
# inside itself, so its env file lives beside the database in ~/.evidence-engine.
# Loaded in that order because python-dotenv does not override what is already
# set, which keeps a developer's local file authoritative.
load_dotenv()
load_dotenv(os.path.join(os.path.expanduser("~"), ".evidence-engine", ".env"))

# IMPORTANT: install the headless shim BEFORE importing utils / data_services


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
from utils import AIService, Deduplicator, AITableExtractor, QueryGenerationError
import frameworks
from data_services import DataAggregator, OpenAlexService
from leads_screening import (
    LEADS_MODEL_NAME,
    LEADS_SCORE_THRESHOLD,
    NON_SCREENING_FALLBACK_MODEL,
    is_leads_model,
    resolve_for_thinking,
    resolve_model_name,
    screen_paper_leads,
)
import instruments as _inst
import pdf_text_repair

import time
import requests
from bs4 import BeautifulSoup
from html import unescape as html_unescape
from urllib.parse import quote, urljoin


# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

# The version is read from the build stamp (packaged) or package.json (source),
# so there is one place it is ever edited.
import self_update
import updates as _updates  # noqa: E402

app = FastAPI(title="Evidence Engine API", version=_updates.current_version())

# Local persistence (sessions + multi-reviewer projects), backed by SQLite, so the
# app needs no external service.
from store import router as store_router  # noqa: E402

# Screening prioritisation. Kept in its own module because it is the one
# component whose algorithm is pinned to a published benchmark figure.
from ranking import router as ranking_router  # noqa: E402

app.include_router(store_router)
app.include_router(ranking_router)


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


# Capture per-request LLM / database keys the user supplied from their browser
# into a request-scoped ContextVar. Keys travel as headers, are used only to
# construct model / data clients for this request, and are never persisted here.
from request_creds import set_request_creds, get_cred, HEADER_TO_CRED
from ratelimit import limited_get, cached, cache_stats


from store import resolve_auth as _resolve_auth


@app.middleware("http")
async def _capture_request_credentials(request, call_next):
    creds = {}
    for header, cred in HEADER_TO_CRED.items():
        val = request.headers.get(header)
        if val:
            creds[cred] = val.strip()
    set_request_creds(creds)
    # Resolve a bearer token (if any) to the authenticated account for this request.
    _resolve_auth(request.headers.get("authorization"))
    return await call_next(request)


# --- OS keychain storage for LLM keys (the "device keychain" storage mode) -----
import keychain as _keychain


class KeychainSetRequest(BaseModel):
    provider: str
    key: str


class KeychainProviderRequest(BaseModel):
    provider: str


@app.get("/api/keys/status")
def keys_status():
    """Report whether OS-keychain storage is available and which providers
    currently have a key stored there. No secrets are returned."""
    return {"available": _keychain.available(), "providers": _keychain.status()}


@app.post("/api/keys/set")
def keys_set(req: KeychainSetRequest):
    provider = (req.provider or "").strip().lower()
    if provider not in _keychain.PROVIDERS:
        raise HTTPException(status_code=400, detail="Unknown provider")
    key = (req.key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="Empty key")
    try:
        _keychain.set_key(provider, key)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Keychain unavailable: {e}")
    return {"ok": True, "providers": _keychain.status()}


@app.post("/api/keys/delete")
def keys_delete(req: KeychainProviderRequest):
    provider = (req.provider or "").strip().lower()
    if provider not in _keychain.PROVIDERS:
        raise HTTPException(status_code=400, detail="Unknown provider")
    _keychain.delete_key(provider)
    return {"ok": True, "providers": _keychain.status()}


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
    # PCC (scoping) elements; population is shared with the PICO fields above.
    concept: str = ""
    context: str = ""
    framework: str = "pico"


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
        concept=getattr(p, "concept", "") or "",
        context=getattr(p, "context", "") or "",
        framework=frameworks.normalize(getattr(p, "framework", "pico")),
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
    # "pico" (intervention review) or "pcc" (JBI scoping review). When omitted,
    # the frame is auto-detected from the goal text.
    framework: Optional[str] = None


class Analysis(BaseModel):
    p: str
    i: str
    c: str
    o: str
    inclusion: List[str]
    exclusion: List[str]
    query: str
    # Frame + PCC-native element values (empty for a PICO frame).
    framework: str = "pico"
    concept: str = ""
    context: str = ""


class FrameworkDetectRequest(BaseModel):
    input: str
    model: Optional[str] = None


@app.post("/api/framework/detect")
def framework_detect(req: FrameworkDetectRequest):
    """Pick the question frame (PICO vs PCC) that best fits the researcher's intended
    study design. Uses the model to classify the goal, with a keyword heuristic as a
    fast fallback. The UI lets the user override. Returns the detected id plus the
    registry so the client can render whichever frame without hardcoding elements."""
    text = (req.input or "").strip()
    heuristic = frameworks.detect_framework(text)
    fw = heuristic
    if text:
        model = AIService.get_model(resolve_for_thinking(req.model))
        if model:
            try:
                from langchain_core.messages import HumanMessage
                prompt = (
                    "Classify the intended review/study design for this research goal. Reply with ONE word only:\n"
                    "  PICO  = an intervention / effectiveness review that tests whether a specific intervention or\n"
                    "          exposure changes an outcome compared with a comparator (quantifies an effect).\n"
                    "  PCC   = a scoping review that MAPS the breadth, range, or nature of evidence on a topic\n"
                    "          (exploratory, 'what is known / what evidence exists', no comparator or measured effect).\n\n"
                    f'RESEARCH GOAL: "{text}"\n\n'
                    "Answer with exactly one token: PICO or PCC."
                )
                out = (model.invoke([HumanMessage(content=prompt)]).content or "").strip().lower()
                if "pcc" in out and "pico" not in out:
                    fw = "pcc"
                elif "pico" in out and "pcc" not in out:
                    fw = "pico"
                # otherwise keep the heuristic
            except Exception as e:
                print(f"[framework_detect] LLM classify failed, using heuristic: {e}")
    return {"framework": fw, "detected_by": ("model" if fw != heuristic or text else "heuristic"),
            "registry": frameworks.FRAMEWORKS}


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
    # Frame is explicit from the client (which runs /framework/detect and lets the
    # user override); default to PICO when unspecified so legacy callers are stable.
    fw = frameworks.normalize(req.framework) if req.framework else "pico"
    data = AIService.infer_pico_and_query(req.input, model_name, req.previous_goal or "",
                                          prior=req.prior, framework=fw)
    if fw == "pcc":
        # PCC (scoping): Population / Concept / Context. Map Concept onto the
        # intervention anchor slot and Context onto the population/context block
        # so the existing deterministic MeSH builder works unchanged.
        pop_str = _pico_value(data.get("population") or data.get("p", ""))
        concept_str = _pico_value(data.get("concept") or data.get("i", ""))
        context_str = _pico_value(data.get("context") or data.get("c", ""))
        pico = PICOCriteria(
            population=pop_str, intervention=concept_str, comparator="", outcome="",
            concept=concept_str, context=context_str, framework="pcc",
        )
        try:
            query = AIService.generate_mesh_query(pico, model_name, goal=req.input or "")
        except Exception as e:
            print(f"[pico_infer] mesh query failed: {e}")
            query = ""
        return Analysis(
            p=pop_str, i=concept_str, c=context_str, o="",
            inclusion=_coerce_str_list(data.get("inclusion")),
            exclusion=_coerce_str_list(data.get("exclusion")),
            query=query or "", framework="pcc", concept=concept_str, context=context_str,
        )
    p_str = _pico_value(data.get("p", ""))
    i_str = _pico_value(data.get("i", ""))
    c_str = _pico_value(data.get("c", ""))
    o_str = _pico_value(data.get("o", ""))
    pico = PICOCriteria(
        population=p_str,
        intervention=i_str,
        comparator=c_str,
        outcome=o_str,
        framework="pico",
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
        framework="pico",
    )


class SearchBuildRequest(BaseModel):
    """Input to /api/search/build: the current PICO/PCC elements to turn into a
    grounded, comprehensive PubMed query. Optional `seeds` are the reviewer's
    known/relevant studies used to broaden concepts (seed-study feedback)."""
    pico: PicoIn
    model: Optional[str] = None
    goal: Optional[str] = ""
    seeds: Optional[List[PaperIn]] = None


class ConceptBlock(BaseModel):
    name: str
    tiab: List[str] = []
    mesh: List[str] = []


class SearchBuildResponse(BaseModel):
    query: str
    concepts: List[ConceptBlock]


@app.post("/api/search/build", response_model=SearchBuildResponse)
def search_build(req: SearchBuildRequest):
    """Phase 1 grounded query builder: LLM concept synonyms -> real NCBI MeSH
    grounding -> deterministic PubMed assembler. Returns the query plus the
    per-concept breakdown so the reviewer can see and edit what the search does
    before it runs. Recall-oriented and validated on CLEF-TAR."""
    model_name = resolve_for_thinking(req.model)
    pico = _to_pico(req.pico)
    # Seed-study feedback: resolve MeSH for any PubMed seeds so their controlled
    # vocabulary can broaden the matching concept.
    seeds: List[Dict[str, Any]] = []
    if req.seeds:
        from data_services import PubMedService
        pmids = [str(s.id) for s in req.seeds if str(s.id or "").strip().isdigit()]
        mesh_by_pmid = PubMedService.fetch_mesh_for_pmids(pmids) if pmids else {}
        for s in req.seeds:
            rec = mesh_by_pmid.get(str(s.id), {})
            seeds.append({"mesh": rec.get("mesh", []), "title": s.title or rec.get("title", "")})
    try:
        query, concepts = AIService.generate_mesh_query(
            pico, model_name, goal=req.goal or "", ground_mesh=True, return_concepts=True,
            seeds=seeds or None)
    except QueryGenerationError as e:
        # The question could not be turned into a search. 422, not 500: nothing
        # is broken, the input could not be processed, and the message is
        # written for the reviewer because the UI shows it verbatim. There is
        # deliberately no fallback query here; a plausible wrong search costs
        # far more than a visible failure.
        print(f"[search_build] {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print(f"[search_build] {e}")
        raise HTTPException(status_code=500, detail=str(e))
    # generate_mesh_query returns the kept concept blocks as a list of
    # {name, tiab, mesh} (variable-N: whatever concept names the review yields).
    blocks: List[ConceptBlock] = []
    concept_list = concepts if isinstance(concepts, list) else []
    for c in concept_list:
        if not isinstance(c, dict):
            continue
        tiab = c.get("tiab", []) or []
        mesh = c.get("mesh", []) or []
        if tiab or mesh:
            blocks.append(ConceptBlock(name=c.get("name") or "Concept", tiab=tiab, mesh=mesh))
    return SearchBuildResponse(query=query or "", concepts=blocks)


class ResultsAskRequest(BaseModel):
    """Ask a question grounded in a set of captured records (RAG over results)."""
    question: str
    records: List[PaperIn]
    model: Optional[str] = None


class ResultsAskResponse(BaseModel):
    answer: str
    cited: List[str] = []


@app.post("/api/results/ask", response_model=ResultsAskResponse)
def results_ask(req: ResultsAskRequest):
    """Answer a natural-language question using ONLY the provided records
    (titles + abstracts), citing them by number. Grounds the answer in the
    reviewer's captured search results."""
    from langchain_core.messages import HumanMessage, SystemMessage
    model = AIService.get_model(resolve_for_thinking(req.model))
    recs = req.records[:60]
    if not recs:
        return ResultsAskResponse(answer="No records to search. Run a search first.", cited=[])
    corpus = "\n\n".join(
        f"[{i+1}] {r.title}\n{(r.abstract or '(no abstract)')[:1200]}"
        for i, r in enumerate(recs)
    )
    system = ("You are a systematic-review assistant. Answer the user's question using ONLY the "
              "numbered records provided. Cite the records you use as [N]. If the records do not "
              "contain the answer, say so plainly. Be concise and specific.")
    user = f"Records:\n{corpus}\n\nQuestion: {req.question}\n\nAnswer (cite with [N]):"
    try:
        resp = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        answer = (resp.content or "").strip()
    except Exception as e:
        print(f"[results_ask] {e}")
        raise HTTPException(status_code=500, detail=str(e))
    cited: List[str] = []
    for m in re.findall(r"\[(\d+)\]", answer):
        idx = int(m) - 1
        if 0 <= idx < len(recs):
            rid = str(recs[idx].id)
            if rid not in cited:
                cited.append(rid)
    return ResultsAskResponse(answer=answer, cited=cited)


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
    asked: List[str] = Field(default_factory=list)  # element ids already asked
    model: Optional[str] = None
    framework: str = "pico"


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

    # At most ONE clarifying question per frame element. Once all have been
    # asked (or the safety cap is hit) we're done — no re-asking.
    fw = frameworks.normalize(req.framework)
    frame_meta = frameworks.framework_of(fw)
    frame_label = frame_meta["label"]                     # "PICO" / "PCC"
    PICO_IDS = frameworks.element_ids(fw)                 # e.g. ["population","concept","context"]
    elem_labels = ", ".join(frameworks.label_for(fw, e) for e in PICO_IDS)
    asked = {str(a).strip().lower() for a in (req.asked or [])}
    remaining = [p for p in PICO_IDS if p not in asked]
    if req.round >= len(PICO_IDS) or not remaining:
        return {"done": True}

    pico_lines = (
        "\n".join(f"  {k.upper()}: {v}" for k, v in answered.items())
        or "  (nothing yet)"
    )
    remaining_label = ", ".join(frameworks.label_for(fw, p) for p in remaining)
    id_enum = " | ".join(f'"{e}"' for e in PICO_IDS)

    prompt = f"""You are a systematic-review librarian helping a researcher pin down their {frame_label} frame
before a database search.

RESEARCHER'S GOAL: "{goal}"

{frame_label} elements clarified so far:
{pico_lines}

TASK
────
Decide which {frame_label} elements ({elem_labels}) still
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
• Elements like Comparator/Context are frequently left unspecified on purpose;
  only ask about one when the question clearly hinges on it.
• When you do ask, pick the most important element that needs clarification and
  give EXACTLY 3 concrete, measurable options relevant to "{goal}".
  GOOD options: "adults 18–65 with major depressive disorder (DSM-5)",
  "CBT ≥12 sessions", "remission at 8 weeks (PHQ-9 < 5)".

Return ONLY one of:

{{"done": true}}

{{
  "done": false,
  "question": {{
    "id": {id_enum},
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
    goal: str = ""   # the researcher's stated input, so the question stays faithful to it


@app.post("/api/pico/formal-question")
def pico_formal_question(req: FormalQuestionRequest):
    q = AIService.generate_formal_question(_to_pico(req.pico), resolve_for_thinking(req.model), req.history, goal=req.goal or "")
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

    prompt = f"""You are an expert evidence synthesist writing the results narrative of a systematic
review. Produce a COMPREHENSIVE, critical plain-text synthesis of the evidence on the research
question: one a methodologist could read once and understand what the literature shows, how strong
that evidence is, and where studies disagree. Synthesize ACROSS studies; do not summarise them one
by one.

RESEARCH GOAL: {goal}

LITERATURE ({len(subset)} papers, numbered [1]-[{len(subset)}]):
{ctx}

Structure the response with exactly these THREE section headers, each on its own line followed by a
blank line, in this order (do NOT add any other section):

Research landscape overview
Arguments supporting the research question
Arguments against or challenging the research question

REQUIREMENTS:
1. "Research landscape overview" - 2 to 3 substantial paragraphs of connected prose, not a list. Give
   the reader the full lay of the land: the volume and maturity of the evidence; the populations,
   settings, and conditions studied; which study designs and analytic methods dominate (and which are
   missing); the data sources and sample sizes typically used; where findings converge into consensus
   and where they diverge or remain contested; and where the base is thin, fragmented, or absent. Weave
   in the overall strength and consistency of the evidence and how far it is likely to generalise. Cite
   the most representative papers inline as you make each point.

2. "Arguments supporting the research question" - 6 to 9 substantive bullets, ordered strongest
   evidence first. Each bullet makes ONE specific, evidence-backed claim and, wherever the abstracts
   supply them, includes the CONCRETE numbers: the finding and its direction, the effect size or
   performance metric (accuracy, AUC, mAP, sensitivity/specificity, hazard/odds ratio, percentage,
   p-value), the population/setting, the sample size, and the study design. Group studies that agree
   onto one bullet with a shared citation list rather than listing them separately. Never write a
   generic claim ("AI can help"): say what was shown, in whom, how well, and how strong the design was.

3. "Arguments against or challenging the research question" - 5 to 8 substantive bullets. Cover BOTH
   (a) genuinely contradictory or null findings, with their numbers and citations, AND (b) a critical
   appraisal of the supporting evidence: risk of bias, small or non-representative samples,
   single-centre or retrospective-only designs, overfitting or absent external/prospective validation,
   inconsistent data quality or reporting, confounding, and settings or subgroups where the effect
   weakens or does not hold. Cite specific evidence, and carefully distinguish "no evidence of an
   effect" from "evidence of no effect".

CITATION RULES:
  - Cite only papers actually relevant to the goal. Ignore off-topic ones entirely: do not mention or
    cite them.
  - Use inline citations like [3] or [5, 7]. Never invent a number outside [1]-[{len(subset)}].
  - Place each citation AFTER the claim it supports, at the END of the sentence or bullet (e.g.
    "... improving early detection and patient outcomes [4, 19]."). NEVER begin a bullet or sentence
    with a citation, and NEVER use "[N]:" as a label or prefix at the start of a line.
  - Always cite specific evidence: never write "[3] is relevant" without saying WHAT [3] showed.
  - Prefer synthesizing several studies per claim over one study per bullet.

FAILURE MODE:
If FEWER THAN 3 of the provided papers are directly relevant to the goal, do not pad. Write only the
"Research landscape overview" section (1 paragraph) stating that the directly relevant evidence base
is thin, name the closest-adjacent findings, and list 3 ways to broaden or refocus the search. Leave
the other sections out entirely.

FORMAT:
Plain text only. No HTML, no markdown bold or italics, no code fences. Dashes for bullets are fine.
Do NOT include a reference list - the UI renders one separately. Output ONLY the three sections above:
do NOT add an "evidence quality", "study characteristics", "open questions", "future research", "next
steps", "conclusion", or "follow-up" section of any kind.
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


_LEADING_CITE_RE = re.compile(r"^(\s*(?:[-*•]\s+)?)(\[\d+(?:\s*,\s*\d+)*\])\s*[:.–\-]?\s+(.+)$")


def _move_leading_citations(summary: str) -> str:
    """Move a citation that leads a line (e.g. "- [4, 19]: claim") to the END of
    the claim ("- claim [4, 19]."), matching the requested reading style where the
    citation follows the statement it supports."""
    out = []
    for line in summary.split("\n"):
        m = _LEADING_CITE_RE.match(line)
        if m:
            prefix, cite, rest = m.group(1), m.group(2), m.group(3).rstrip()
            if rest and rest[-1] in ".!?;:":
                rest = f"{rest[:-1]} {cite}{rest[-1]}"
            else:
                rest = f"{rest} {cite}"
            out.append(f"{prefix}{rest}")
        else:
            out.append(line)
    return "\n".join(out)


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
    # Move any citation that leads a bullet/sentence to the end of the claim.
    summary = _move_leading_citations(summary)
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


class SupplementaryQueryRequest(BaseModel):
    pico: PicoIn
    query: str = ""
    inclusion: List[str] = Field(default_factory=list)
    exclusion: List[str] = Field(default_factory=list)
    kept_titles: List[str] = Field(default_factory=list)
    model: Optional[str] = None


@app.post("/api/search/supplementary")
def search_supplementary(req: SupplementaryQueryRequest):
    """One adaptive round for the Home deep scan: propose a supplementary query
    from the relevant studies found so far, so the caller can fetch a second
    (separately logged) batch without touching the primary documented search."""
    return AIService.propose_supplementary_query(
        _to_pico(req.pico),
        req.query,
        req.inclusion,
        req.exclusion,
        req.kept_titles,
        resolve_for_thinking(req.model),
    )


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


_TITLE_LEADINS = re.compile(
    r"^(?:i\s+(?:want|would\s+like|need|wish|aim|plan|hope|am\s+looking)\s+to\s+"
    r"(?:know|understand|find\s+out|learn|explore|investigate|examine|study|review|assess|evaluate|research|determine|see)"
    r"(?:\s+(?:about|whether|if|how|what|the))?|"
    r"what\s+(?:is|are|was|were)|how\s+(?:does|do|did|can|to)|"
    r"can\s+you|could\s+you|please|tell\s+me\s+about|i'?m\s+interested\s+in|"
    r"(?:a\s+)?(?:study|systematic\s+review|review|analysis|investigation|meta[- ]analysis)\s+(?:of|on|about)|"
    r"the\s+(?:relationship|association|effect|effects|impact|role|link|correlation)\s+(?:between|of|on)|"
    r"explore|investigate|examine|assess|evaluate|determine|understand|"
    r"is\s+there\s+(?:a|an)?)\s+",
    re.IGNORECASE,
)


def _heuristic_title(goal: str) -> str:
    """Deterministic short summary of a research goal for the fallback title —
    strips conversational lead-ins ("I want to know about the relationship
    between …") rather than slicing the question verbatim."""
    g = goal.strip().strip("?.! ")
    for _ in range(3):                      # peel nested lead-ins in one pass each
        stripped = _TITLE_LEADINS.sub("", g).strip(" ?.!,:")
        if stripped == g or not stripped:
            break
        g = stripped
    if not g:
        g = goal.strip()
    short = " ".join(g.split()[:8]).strip(" ?.!,:")
    if len(short) > 60:
        short = short[:60].rsplit(" ", 1)[0]
    if not short:
        return "Untitled session"
    return short[0].upper() + short[1:]


@app.post("/api/sessions/title")
def session_title(req: TitleRequest):
    """Generate a short 3-6 word title from a research goal (LLM-driven, with a
    string-slice fallback so the frontend always gets something usable)."""
    goal = (req.goal or "").strip()
    if not goal:
        return {"title": "Untitled session"}
    fallback = _heuristic_title(goal)   # a summary, not a verbatim slice
    try:
        from langchain_core.messages import HumanMessage
        model = AIService.get_model(resolve_for_thinking(req.model))
        if not model:
            return {"title": fallback}
        prompt = (
            "Write a SHORT topic title (3 to 6 words) that summarizes this research goal, "
            "as a noun phrase like a paper's short title. Do NOT restate the question or use "
            "phrases like 'I want to know', 'the relationship between', 'a study of', 'what is'. "
            "No quotes, no end punctuation, no preamble — return only the title.\n\n"
            "Example goal: I want to know about the relationship between pet ownership and cardiovascular disease\n"
            "Example title: Pet Ownership and Cardiovascular Disease\n\n"
            f"GOAL: {goal}\n\nTITLE:"
        )
        r = model.invoke([HumanMessage(content=prompt)])
        title = (r.content or "").strip().split("\n")[0].strip().strip('"').strip("'").rstrip(".").strip()
        # Reject junk or an answer that just echoes the goal verbatim.
        if not title or len(title) > 70 or title.lower() == goal.lower():
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

    # Prioritise blanks, in the active frame's element order (Population first —
    # most load-bearing for retrieval relevance).
    fw = frameworks.normalize(getattr(req.pico, "framework", "pico"))
    frame_label = frameworks.framework_of(fw)["label"]
    PRIORITY = frameworks.element_ids(fw)
    values = {eid: (getattr(req.pico, eid, "") or "").strip() for eid in PRIORITY}
    elem_block = "\n".join(f"  {frameworks.label_for(fw, eid)}: {values[eid] or '(blank)'}" for eid in PRIORITY)
    field_enum = " | ".join(f'"{eid}"' for eid in PRIORITY)
    blanks = [f for f in PRIORITY if not values[f]]

    if blanks:
        target = blanks[0]
        prompt = f"""You are a research methodologist helping a researcher specify a
systematic-review {frame_label} frame. The researcher's stated goal is below. They did NOT specify the
{frameworks.label_for(fw, target).upper()} element. Your job is to ask ONE concise clarifying question and offer ONE
plausible starting value the researcher can accept, edit, or reject.

RESEARCH GOAL: {req.goal or "(not provided)"}

CURRENT {frame_label} (the blank field is the one we are asking about):
{elem_block}

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
        prompt = f"""You are a research methodologist reviewing a {frame_label} breakdown for a
systematic review. Identify the ONE element that is most under-specified, ambiguous, or
methodologically weak, and propose a sharper replacement for that element only.

RESEARCH GOAL: {req.goal}

CURRENT {frame_label}:
{elem_block}

Pick the single weakest element and propose a concrete improvement. Be specific — name a
subgroup, dose/duration, concept scope, context, or validated measure as appropriate. Do not
suggest changes to multiple elements; pick the most impactful one.

Return ONLY a JSON object with these exact keys:
{{
  "field": {field_enum},
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
        if field not in set(PRIORITY):
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
    # Selection strategy when a source matches more than its cap:
    # "relevance" (best match, default) or "recent" (newest).
    sort: str = "relevance"
    # Optional publication-year window (inclusive) applied at the source.
    year_from: Optional[int] = None
    year_to: Optional[int] = None


@app.post("/api/papers/fetch")
def papers_fetch(req: FetchAllRequest):
    papers, counts, source_status = DataAggregator.fetch_all(
        req.query, req.sources, max_per_source=req.max_per_source,
        uploaded_files=None, limit=req.limit, sort=req.sort,
        year_from=req.year_from, year_to=req.year_to,
    )
    return {
        "papers": [_paper_to_dict(p) for p in papers],
        "sourceCounts": counts,
        # Per-source diagnostics, so the UI can state why a source returned
        # little rather than offering the reviewer a list of possibilities.
        "sourceStatus": source_status,
    }


class PdfMetadataRequest(BaseModel):
    text: str = ""
    filename: str = ""
    model: str = ""


@app.post("/api/pdf/metadata")
def pdf_metadata(req: PdfMetadataRequest):
    """Extract clean bibliographic metadata (title / authors / year / abstract /
    DOI) from an uploaded PDF's extracted text, using the thinking model. This
    upgrades the client's filename-based heuristic. Best-effort — missing fields
    come back empty so the caller can keep its heuristic value."""
    from langchain_core.messages import HumanMessage
    text = (req.text or "")[:6000]   # title/abstract live in the opening pages
    if not text.strip():
        return {"title": "", "authors": "", "year": None, "abstract": "", "doi": ""}
    model = AIService.get_model(resolve_for_thinking(req.model))
    prompt = f"""Extract bibliographic metadata from the opening text of a research paper. Return ONLY a JSON object with these exact keys, no commentary or code fences:
{{
  "title": "<full paper title, or empty string>",
  "authors": "<comma-separated author names, or empty string>",
  "year": <4-digit publication year as a number, or null>,
  "abstract": "<the abstract verbatim if present, else a 2-3 sentence summary, or empty string>",
  "doi": "<DOI like 10.xxxx/yyyy, or empty string>"
}}
Filename hint: {req.filename}

PAPER TEXT:
{text}"""
    try:
        r = model.invoke([HumanMessage(content=prompt)])
        data = AIService._extract_json(r.content) or {}
    except Exception as e:
        print(f"[pdf_metadata] {e}")
        data = {}
    yr = data.get("year")
    try:
        yr = int(yr) if yr else None
    except Exception:
        yr = None
    return {
        "title": str(data.get("title") or "").strip(),
        "authors": str(data.get("authors") or "").strip(),
        "year": yr,
        "abstract": str(data.get("abstract") or "").strip(),
        "doi": str(data.get("doi") or "").strip(),
    }


class SimulateYieldRequest(BaseModel):
    query: str
    sources: List[str]


@app.post("/api/simulation/yield")
def simulation_yield(req: SimulateYieldRequest):
    # simulate_yield guards each source individually, but a failure in the
    # surrounding code (a bad query, an import error) used to surface as a 500
    # traceback in the browser. Report zeros and say so instead: the Planning
    # tab can render "0 for every source", it cannot render a stack trace.
    try:
        counts = DataAggregator.simulate_yield(req.query, req.sources)
        return {"counts": counts}
    except Exception as e:
        print(f"[simulation_yield] {e}")
        return {"counts": {s: 0 for s in (req.sources or [])},
                "error": "Could not estimate yield for this query."}


# ── Per-database syntax adaptation ──────────────────────────────────────────────
# Translate a PubMed-style base query into each search engine's native syntax,
# preserving the terms and Boolean logic. The base query is authored in PubMed
# field-tag syntax ([tiab], [Mesh], …) which most other engines reject.

# Concise rules so the LLM produces idiomatic, executable strings per engine.
_ENGINE_SYNTAX_RULES = {
    "PubMed": "Native PubMed/MEDLINE syntax. Keep field tags: [tiab], [ti], [Mesh], [Mesh:NoExp], date [dp] (e.g. 2017:2026[dp]), language [la], publication type [pt]. This is the canonical form.",
    "Europe PMC": "Europe PMC syntax. Boolean AND/OR/NOT in caps. Use fields TITLE:\"..\", ABSTRACT:\"..\", or no field for all-fields. MeSH via MESH:\"..\". Do NOT use PubMed bracket tags.",
    "Semantic Scholar": "Plain keyword/phrase search. No field tags. Quote phrases. Keep AND/OR/NOT and parentheses but no PubMed brackets.",
    "OpenAlex": "Free-text relevance search (no field tags, limited Boolean). Provide the key quoted phrases and terms; drop PubMed brackets, dates and publication-type filters.",
    "CrossRef": "Bibliographic free-text search. No field tags or Boolean operators; provide the key quoted phrases and terms only.",
    "arXiv": "arXiv API syntax. Use field prefixes ti:, abs:, all:, cat:. Phrases in quotes. Boolean AND/OR/ANDNOT. Map [ti]→ti:, [tiab]/[tw]→abs:, drop [Mesh]/[dp]/[la]. Combine concept blocks with AND.",
    "bioRxiv": "Preprint server with NO Boolean field search. Provide a short, focused keyword phrase of the 3-6 most important terms only.",
    "medRxiv": "Preprint server with NO Boolean field search. Provide a short, focused keyword phrase of the 3-6 most important terms only.",
    "DOAJ": "DOAJ Elasticsearch query-string syntax. Fields bibjson.title:\"..\", bibjson.abstract:\"..\". Boolean AND/OR in caps, quotes for phrases. Drop PubMed brackets.",
    "CORE": "CORE query syntax. Fields title:\"..\", abstract:\"..\", yearPublished:>=2017. Boolean AND/OR/NOT, quotes for phrases. Drop PubMed brackets.",
}


def _strip_pubmed_tags(q: str) -> str:
    """Deterministic fallback: remove PubMed bracket field tags and tidy whitespace."""
    out = re.sub(r"\[[^\]]+\]", "", q)
    out = re.sub(r"\bNOT\s*\(\s*\)", "", out)
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\(\s*\)", "", out).strip()
    return out


class AdaptQueriesRequest(BaseModel):
    base_query: str
    sources: List[str]
    model: str = ""


@app.post("/api/simulation/adapt")
def simulation_adapt(req: AdaptQueriesRequest):
    """Adapt the base query to each engine's native syntax (terms/logic preserved)."""
    from langchain_core.messages import HumanMessage

    sources = [s for s in req.sources if s != "Local PDFs"]
    base = (req.base_query or "").strip()
    if not base or not sources:
        return {"per_source_queries": {}}

    # Deterministic fallback for every source.
    fallback = {}
    for src in sources:
        fallback[src] = base if src == "PubMed" else _strip_pubmed_tags(base)

    model = AIService.get_model(resolve_for_thinking(req.model))
    if not model:
        return {"per_source_queries": fallback}

    rules = "\n".join(f"  - {src}: {_ENGINE_SYNTAX_RULES.get(src, 'General academic database; drop PubMed bracket tags, keep Boolean logic and quoted phrases.')}" for src in sources)
    prompt = f"""You are an expert systematic-review information specialist. Translate the following PubMed-style Boolean search query into the NATIVE syntax of each target database.

CRITICAL RULES:
- PRESERVE the search terms, synonyms, quoted phrases and Boolean logic (the AND/OR structure of the concept blocks) exactly. Do NOT add, remove, or reword search concepts.
- ONLY change field tags, operators, and formatting so the query is valid and idiomatic for each engine.
- Each engine's specific syntax rules:
{rules}

BASE QUERY (PubMed syntax):
{base}

Return ONLY a JSON object mapping each database name to its adapted query string, with these exact keys: {sources}. No commentary, no code fences."""

    try:
        resp = model.invoke([HumanMessage(content=prompt)])
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        content = content.strip()
        # Pull the JSON object out of the response.
        m = re.search(r"\{.*\}", content, re.DOTALL)
        data = _json.loads(m.group(0)) if m else {}
        out = {}
        for src in sources:
            v = data.get(src)
            out[src] = v.strip() if isinstance(v, str) and v.strip() else fallback[src]
        return {"per_source_queries": out}
    except Exception as e:
        print(f"simulation_adapt failed, using deterministic fallback: {e}")
        return {"per_source_queries": fallback}


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
    protocol: Optional[str] = None
    model: Optional[str] = None


_PICO_ASSESS_VOTES = {"PASS", "PARTIAL", "FAIL", "NA"}


# Decomposition depends only on the element text, not on the paper, so it is
# computed once per review rather than once per record. A duplicate call under
# threading is harmless, so no lock.
_ELEMENT_PARTS_CACHE: Dict[tuple, List[Dict[str, Any]]] = {}

# Connectives a decomposed part may keep from the sentence it was cut out of.
# Left in place they turn "Does this study involve X?" into a malformed question.
_LEADING_CONNECTIVE = re.compile(
    r"^(?:that\s+(?:contain|contains|include|includes|use|uses|hold|holds)?|"
    r"which\s+\w+|using|containing|include|including|involving|based\s+on|"
    r"with|for|to|in|on|of|and|or)\s+",
    re.I,
)


def _merge_or_alternatives(parts: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """Rejoin parts the model split across an "or", against instruction.

    "A or B" means either satisfies the element, so splitting it into two required
    parts silently demands both and drops every paper that has only one. The
    prompt forbids the split and the model still does it, so enforce it here:
    locate each part in the source text and merge any adjacent pair separated by
    exactly the word "or".
    """
    import difflib
    words = re.findall(r"[a-z0-9]+", text.lower())

    def span(p: str) -> Optional[tuple]:
        pw = re.findall(r"[a-z0-9]+", p.lower())
        if not pw or len(pw) > len(words):
            return None
        best, at = 0.0, None
        for i in range(len(words) - len(pw) + 1):
            r = difflib.SequenceMatcher(None, pw, words[i:i + len(pw)]).ratio()
            if r > best:
                best, at = r, i
        return (at, at + len(pw)) if best >= 0.7 and at is not None else None

    out: List[Dict[str, Any]] = []
    for p in parts:
        s = span(str(p.get("part") or ""))
        prev = out[-1] if out else None
        if prev and s and prev.get("_span") and words[prev["_span"][1]:s[0]] == ["or"]:
            prev["part"] = f'{prev["part"]} or {p["part"]}'[:120]
            prev["appears_as"] = (prev.get("appears_as") or [])[:4] + (p.get("appears_as") or [])[:4]
            # A near miss for one alternative may be satisfied by the other, so
            # the merged part cannot inherit either half's disqualifiers.
            prev["not_satisfied_by"] = []
            prev["_span"] = (prev["_span"][0], s[1])
            continue
        q = dict(p)
        q["_span"] = s
        out.append(q)
    for q in out:
        q.pop("_span", None)
    return out


def _decompose_element(text: str, model_name: str) -> List[Dict[str, Any]]:
    """Split an eligibility element into the separate things a paper must have.

    Returns [{"part": <short noun phrase>, "appears_as": [<wording>, ...]}, ...].

    ``appears_as`` is how that part could plausibly surface in an abstract, and
    it is generated FOR THE REVIEW BEING RUN rather than hardcoded, because an
    abstract almost never echoes an element's wording back and the vocabulary
    that bridges the gap is different in every field. Nothing domain-specific
    belongs in this prompt: the worked example is deliberately drawn from an
    unrelated field so the model learns the shape and not the content.

    Alternatives ("A or B") stay inside ONE part: either satisfies the element.
    Returns a single entry when it cannot decompose, which disables per-part
    checking and leaves the one-shot vote in place.
    """
    from langchain_core.messages import HumanMessage
    key = (text.strip(), model_name)
    if key in _ELEMENT_PARTS_CACHE:
        return _ELEMENT_PARTS_CACHE[key]

    model = AIService.get_model(model_name)
    parts: List[Dict[str, Any]] = []
    if model:
        prompt = f"""Break this eligibility requirement into the separate things a paper
must ALL have to satisfy it, and say how each one could appear in an abstract.

Splitting rules:
  • Split on "and", "using", "containing", "that contain" — these join things that
    must ALL be true.
  • NEVER split on "or". Alternatives stay inside ONE entry, because either one
    satisfies the requirement. Splitting them would force a paper to have both.
  • A phrase like "both X and Y" describes ONE combined requirement — keep it as a
    single entry rather than two, so the combination is checked as a combination.
  • 2 to 4 entries, each under 10 words.
  • Each entry must be a STANDALONE NOUN PHRASE that reads correctly in the
    sentence "Does this study involve ___ ?". Drop the connective you split on:
    write "electronic health records", never "using electronic health records";
    write "records holding both A and B", never "that contain both A and B".
    A fragment starting with using / that / with / to / for is malformed.

For each entry give TWO lists:
  "appears_as": 3 to 8 ways THAT ENTRY could be worded in a real abstract —
    synonyms, abbreviations and expansions, and specific instances.
    Stay strictly inside the entry. Do NOT list anything that merely sits in the
    same broad field, because that would make every paper in the field match.
    For a combined "both X and Y" entry, every wording must show BOTH halves
    together; a wording showing only X is not a way this entry appears.
  "not_satisfied_by": 2 to 5 NEAR MISSES — things close enough to be confused with
    the entry that nonetheless do NOT satisfy it. For a combination, name each
    half on its own. These are the cases that must be answered no.
    If the entry itself contains "or", NEVER list a near miss that has one of the
    alternatives but not the other: either alternative alone satisfies the entry,
    so "A without B" is a match, not a miss.

WORKED EXAMPLE (an unrelated field, shown only to fix the format)
REQUIREMENT: "satellite or drone imagery combined with both soil and weather
measurements to estimate crop yield"
CORRECT:
[{{"part": "satellite or drone imagery",
   "appears_as": ["remote sensing imagery", "Sentinel-2 scenes", "UAV aerial photographs",
                  "multispectral overflight data"],
   "not_satisfied_by": ["ground photographs only", "hand-drawn field maps"]}},
 {{"part": "both soil and weather measurements",
   "appears_as": ["soil moisture together with rainfall records",
                  "agronomic soil samples paired with local climate data",
                  "combined edaphic and meteorological variables"],
   "not_satisfied_by": ["soil samples with no weather data",
                        "rainfall records with no soil data",
                        "a generic mention of environmental conditions"]}},
 {{"part": "crop yield estimation",
   "appears_as": ["predicted harvest tonnage", "yield forecasting", "productivity per hectare"],
   "not_satisfied_by": ["mapping land cover without any yield outcome"]}}]
WRONG: splitting "satellite or drone imagery" into two entries — the requirement
says either one is enough, so splitting it would demand both. Equally wrong:
writing the second entry as "with both soil and weather measurements", which is a
fragment and not a standalone noun phrase.

REQUIREMENT: {text}

Return ONLY a JSON array of objects with keys "part", "appears_as" and
"not_satisfied_by"."""
        try:
            r = model.invoke([HumanMessage(content=prompt)])
            data = AIService._extract_json(r.content)
            def _strings(v: Any, n: int) -> List[str]:
                if not isinstance(v, list):
                    return []
                return [s for s in (str(x).strip()[:80] for x in v[:n]) if s]

            if isinstance(data, list):
                for item in data[:5]:
                    if isinstance(item, dict):
                        p = str(item.get("part") or "").strip().strip('"').strip("'")[:80]
                        syns = _strings(item.get("appears_as"), 8)
                        anti = _strings(item.get("not_satisfied_by"), 5)
                    else:
                        # Tolerate a bare array of strings from a model that
                        # ignored the object shape.
                        p, syns, anti = str(item).strip().strip('"').strip("'")[:80], [], []
                    # A part is interpolated into "Does this study involve ___?",
                    # so a leading connective the model failed to drop would make
                    # that question ungrammatical and reliably answered "yes".
                    p = _LEADING_CONNECTIVE.sub("", p).strip()
                    # For an "A or B" part either alternative is enough, so a near
                    # miss phrased as "A without B" is really a match. The prompt
                    # says so and the model still writes them, so drop them.
                    if re.search(r"\bor\b", p, re.I):
                        anti = [s for s in anti if not re.search(r"\bwithout\b", s, re.I)]
                    if p:
                        parts.append({"part": p, "appears_as": syns,
                                      "not_satisfied_by": anti})
        except Exception as e:
            print(f"[decompose_element] failed: {e}")

    if not parts:
        parts = [{"part": text.strip(), "appears_as": [], "not_satisfied_by": []}]
    else:
        parts = _merge_or_alternatives(parts, text)
    _ELEMENT_PARTS_CACHE[key] = parts
    return parts


def _part_present(part: Dict[str, Any], title: str, abstract: str, model_name: str) -> str:
    """Ask ONE focused question: is this single requirement present in the paper?

    ``part`` is one entry from :func:`_decompose_element`, carrying the required
    thing plus the wordings that review generated for it. The synonyms are
    interpolated per review rather than written into this prompt: a fixed
    glossary here would inject one review's subject matter into every other
    review, and would hand the model the answer for the review it was written
    for.

    Returns "yes" | "no" | "unclear". Defaults to "unclear" on any failure, so a
    broken call can never exclude a paper.
    """
    from langchain_core.messages import HumanMessage
    model = AIService.get_model(model_name)
    if not model:
        return "unclear"

    req = str(part.get("part") or "").strip()
    syns = [s for s in (part.get("appears_as") or []) if str(s).strip()]
    anti = [s for s in (part.get("not_satisfied_by") or []) if str(s).strip()]

    syn_block = (
        "\nCOUNTS AS YES, however the abstract words it: "
        + "; ".join(str(s) for s in syns[:8]) + "."
        if syns else
        "\nCount genuine synonyms and specific instances as yes: an abstract rarely"
        "\nechoes the requirement's own wording."
    )
    # The mirror of the synonym list. Without it the prompt says what a yes looks
    # like and nothing about what a no looks like, and that asymmetry alone drives
    # the answer toward yes on any paper in the neighbouring subject area.
    anti_block = (
        "\nCOUNTS AS NO, even though it is close: "
        + "; ".join(str(s) for s in anti[:5]) + "."
        if anti else ""
    )

    body = (abstract or "").strip()[:6000] or "(no abstract — judge from the title alone)"
    prompt = f"""TITLE: {title}

ABSTRACT: {body}

Judge ONLY the data this study actually analysed, or the intervention it actually
applied. Background, motivation, related work and future plans do NOT count.

QUESTION: Did this study use {req}?
{syn_block}{anti_block}

Being in the same broad field as the requirement is NOT enough on its own, and if
the requirement names a COMBINATION, having only one side of it is a no.

Answer with EXACTLY ONE WORD:
  yes      — present, or a specific instance of it is present
  no       — the study used something else, or only part of what is required,
             so the requirement is genuinely not met
  unclear  — the abstract truly does not say either way"""
    try:
        r = model.invoke([HumanMessage(content=prompt)])
        txt = str(getattr(r, "content", "") or "").lower()
        m = re.search(r"\b(yes|no|unclear)\b", txt)
        return m.group(1) if m else "unclear"
    except Exception as e:
        print(f"[part_present] failed: {e}")
        return "unclear"


def _pico_assess(paper: PaperIn, pico: PicoIn, model_name: str, protocol: str = "",
                 inclusion: Optional[List[str]] = None,
                 exclusion: Optional[List[str]] = None) -> Dict[str, Any]:
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
    # Framework-driven element set (PICO: P/I/C/O; PCC: Population/Concept/Context).
    fw = frameworks.normalize(getattr(pico, "framework", "pico"))
    frame_label = frameworks.framework_of(fw)["label"]
    elem_defs = frameworks.element_defs(fw)
    elem_vals = {e["id"]: (getattr(pico, e["id"], "") or "") for e in elem_defs}
    empty: Dict[str, Any] = {e["id"]: {"vote": "NA", "evidence": "", "reasoning": ""} for e in elem_defs}
    empty["overall_reasoning"] = ""
    empty["bucket"] = ""
    empty["failed_criteria"] = []
    if not model:
        return empty

    has_abstract = bool(abstract)
    abstract_block = abstract[:6000] if has_abstract else (
        "(NO ABSTRACT AVAILABLE — judge ONLY from the paper title above. "
        "Infer the likely population/intervention/outcome from the title, do NOT "
        "fabricate quotes, and explain your best-effort judgement.)"
    )

    disc_ids = set(frameworks.discriminating_ids(fw))
    elem_block = "\n".join(
        f"  {lbl}{' [DISCRIMINATING]' if eid in disc_ids else ''}: {elem_vals.get(eid) or '(unspecified)'}"
        for eid, lbl in [(e["id"], e["label"]) for e in elem_defs])
    json_shape = "\n".join(f'  "{e["id"]}":    {{ "vote": "...", "evidence": "...", "reasoning": "..." }},'
                           for e in elem_defs)

    proto_block = (
        f"\nFULL REVIEW QUESTION / PROTOCOL (authoritative — the element summaries\n"
        f"below may be abbreviated, so resolve any ambiguity against this):\n"
        f"{(protocol or '').strip()[:2500]}\n"
        if (protocol or "").strip() else ""
    )

    # Eligibility criteria are checked in THIS call rather than a second one.
    # Only the failures are requested back (as ids), because the exclusion-bucket
    # UI reads only the FAIL entries — asking for a verdict on all 14 criteria
    # would triple the output tokens for information nothing consumes.
    inc_list = [c for c in (inclusion or []) if str(c).strip()]
    exc_list = [c for c in (exclusion or []) if str(c).strip()]
    crit_ids: Dict[str, str] = {}
    crit_lines: List[str] = []
    for n, c in enumerate(inc_list, 1):
        crit_ids[f"I{n}"] = c
        crit_lines.append(f"  I{n}. {c}")
    for n, c in enumerate(exc_list, 1):
        crit_ids[f"E{n}"] = c
        crit_lines.append(f"  E{n}. {c}")
    crit_block = (
        "\nELIGIBILITY CRITERIA (I = inclusion, must be met; E = exclusion, must NOT apply):\n"
        + "\n".join(crit_lines)
        + "\n\nAlso return \"failed_criteria\": a list of the ids that count AGAINST this\n"
          "paper — an I-id the paper fails to meet, or an E-id that applies to it.\n"
          "Return [] when none. Ids only (e.g. [\"I2\",\"E1\"]), no prose.\n"
        if crit_lines else ""
    )

    prompt = f"""You are screening a paper against a {frame_label} frame for a systematic review.
For EACH of the {frame_label} elements below, decide a vote of PASS, PARTIAL, or FAIL.
Never use "NA" or "UNCERTAIN" — pick the closest of the three labels:

Work through this test IN ORDER for each element and stop at the first step that
matches. Do NOT skip to a verdict.

  STEP 1 — Does the paper's subject matter contain what the element asks for?
    Decide this by MEANING, never by wording. An abstract almost never quotes an
    element back: the same thing routinely appears under a different name, as a
    named instance of a general category, or as a field-specific abbreviation.
    Read for the underlying thing, not the label. Equally, a paper that shares
    the element's vocabulary while studying something else does not contain it.

  STEP 2 — If the answer to step 1 is NO, and the paper is plainly ABOUT A
    DIFFERENT SUBJECT so the missing part could not show up in the full text
    either  ->  FAIL.
    This is also the verdict when the element requires a COMBINATION and the
    paper has only one side of it, with nothing in its subject matter that the
    full text could supply for the other side.

  STEP 3 — If the answer to step 1 is YES and the abstract states it explicitly
    and completely  ->  PASS.

  STEP 4 — If the answer to step 1 is YES but the abstract is implicit, partial,
    broader or narrower  ->  PARTIAL. Send it to full text rather than discarding
    it. This is a high-sensitivity FIRST PASS: genuine uncertainty is resolved at
    full text, never here.

Never FAIL a paper merely because it does not echo the element's wording, and
never PASS one merely because it sits in the same broad field.

Elements marked [DISCRIMINATING] define what this review is actually about, and a
FAIL there will exclude the paper outright — so reserve FAIL there for papers that
are genuinely about a different subject (step 2), not for papers that are on topic
but vague. Unmarked elements are descriptive scope: judge them generously and
prefer PARTIAL over FAIL, because they never exclude anything on their own.

For a [DISCRIMINATING] element, do not agonise over the whole conjunction here.
Each of its required parts is verified separately afterwards, one focused question
per part, and that separate check is what decides the element's outcome.{proto_block}

For every vote also return:
  • evidence: a SHORT verbatim phrase or sentence copied directly from the
    abstract (≤ 200 characters) that best supports your vote. If there is no
    abstract, return an empty string for evidence — never invent a quote.
    A FAIL that rests on ABSENCE is the one case where empty evidence is correct
    and expected: you cannot quote a thing the abstract never mentions. Return ""
    rather than quoting an unrelated sentence to fill the field.
  • reasoning: one sentence, SPECIFIC to THIS paper — name what the paper
    actually studied (its real population/intervention/outcome as stated in the
    title or abstract) and why that earns the vote. For a FAIL, name the exact
    component that is missing. Never leave it blank and never write a generic
    template sentence.

ALWAYS write a 2-3 sentence "overall_reasoning" that (a) names in one clause what
THIS paper is actually about (use the title when there's no abstract), then
(b) states the specific PICO element(s) it matches or fails and why, referencing
concrete terms from the title/abstract. It must NEVER be blank. Do NOT write a
generic sentence such as "the paper does not address any of the PICO elements"
or "it lacks specificity regarding validated outcomes" — always ground it in
this study's actual topic. When only a title is available, say what the title
implies and flag what remains uncertain pending full text.

{frame_label}:
{elem_block}
{crit_block}
PAPER TITLE: {title}

PAPER ABSTRACT:
{abstract_block}

Also return "bucket": a 3-5 word label naming the single most decisive factor
(e.g. "Wrong study design", "Population out of scope", "All elements met"). It is
shown to the reviewer as the at-a-glance reason, so make it specific to this paper.

Return ONLY a JSON object with EXACTLY this shape:
{{
{json_shape}
  "overall_reasoning": "...",
  "bucket": "...",
  "failed_criteria": []
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

    def _clean_field(raw: Any, pico_seed: str) -> Dict[str, Any]:
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

        # For a discriminating element the per-part checklist is AUTHORITATIVE and
        # overrides the model's own one-shot verdict. Judging a four-part
        # conjunction in a single pass is where small models fail: qwen2.5:7b
        # called a Random Forest study "does not use AI/ML" and FAILed every paper
        # it was shown. Asking each part separately and combining them here is both
        # more accurate and reproducible from what the reviewer can see.
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
        #
        # Deliberately NOT run for FAIL. A FAIL usually rests on ABSENCE, and a
        # single shared token ("data", "clinical") is enough to drag back some
        # unrelated sentence, which then reads as evidence FOR the element it is
        # supposed to refute. Requiring 2 tokens elsewhere keeps the rescue from
        # firing on one generic word.
        if not evidence and vote != "FAIL":
            evidence = _best_sentence(pico_seed, min_overlap=2)[:240]

        reasoning = str(raw.get("reasoning") or "").strip()[:300]

        # Reclassification rules (no more silent NA when the abstract exists):
        #   • If the model said NA but we DID anchor a quote → PARTIAL.
        #     This is the "on par but not explicit" case.
        #   • If the model said NA and we anchored nothing → keep NA.
        #     (Abstract genuinely doesn't address this element.)
        #   • If the model said PASS or PARTIAL but we couldn't anchor a quote at
        #     all → downgrade to NA (we promised the user that every positive chip
        #     has a quote). FAIL is EXEMPT: "the abstract never mentions the
        #     required thing" is unquotable by construction, and downgrading was
        #     silently discarding exactly the votes that should exclude a paper.
        if vote == "NA" and evidence:
            vote = "PARTIAL"
            if not reasoning:
                reasoning = "Abstract content relates to this PICO element but does not match explicitly."
        elif vote == "NA":
            reasoning = ""
        if vote in {"PASS", "PARTIAL"} and not evidence:
            vote = "NA"
            reasoning = ""
        if vote == "FAIL" and not reasoning:
            reasoning = "The title/abstract does not provide what this element requires."

        return {"vote": vote, "evidence": evidence, "reasoning": reasoning}

    assessed = {eid: _clean_field(data.get(eid), elem_vals.get(eid) or "") for eid in elem_vals}

    # ---- Per-part verification of the discriminating element(s) --------------
    # One focused yes/no/unclear question per required part, each its own call.
    # A 7b model that gets a four-part conjunction wrong in a single pass answers
    # the same four parts correctly when asked one at a time; the cost is N small
    # calls whose prompts are short and whose output is a single word.
    for eid in elem_vals:
        if eid not in disc_ids or not (elem_vals.get(eid) or "").strip():
            continue
        parts = _decompose_element(elem_vals[eid], model_name)
        if len(parts) < 2:
            continue  # nothing to decompose; leave the one-shot vote alone
        components = [{"part": p.get("part", ""),
                       "present": _part_present(p, title, abstract, model_name)}
                      for p in parts]
        if any(c["present"] == "no" for c in components):
            derived = "FAIL"
        elif all(c["present"] == "yes" for c in components):
            derived = "PASS"
        else:
            derived = "PARTIAL"
        assessed[eid]["vote"] = derived
        assessed[eid]["components"] = components
        # The vote now comes from the per-part checks, so the reasoning has to as
        # well. Leaving the one-shot sentence in place next to a derived vote
        # produced panels that argued against their own verdict.
        missing = [c["part"] for c in components if c["present"] == "no"]
        unknown = [c["part"] for c in components if c["present"] == "unclear"]
        if missing:
            note = "Required part not present: " + "; ".join(missing[:3])
        elif unknown:
            note = "Abstract does not establish: " + "; ".join(unknown[:3])
        else:
            note = "All required parts present: " + "; ".join(
                c["part"] for c in components[:3])
        assessed[eid]["reasoning"] = note[:300]

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
            f'The abstract for "{title[:140]}" could not be mapped to the {frame_label} frame with '
            f"confidence — assess at full text."
        )

    bucket = str(data.get("bucket") or "").strip()[:60]

    # Map the returned ids back onto the criterion strings the UI knows about.
    # Unknown ids are dropped rather than guessed at.
    raw_failed = data.get("failed_criteria")
    failed_criteria: List[str] = []
    if isinstance(raw_failed, list):
        for item in raw_failed:
            key = str(item).strip().upper()
            if key in crit_ids and crit_ids[key] not in failed_criteria:
                failed_criteria.append(crit_ids[key])

    return {**assessed, "overall_reasoning": overall_reasoning,
            "bucket": bucket, "failed_criteria": failed_criteria}


def _decide_from_votes(assessment: Dict[str, Any], framework: str) -> tuple[str, str]:
    """Derive INCLUDE/EXCLUDE from the per-element votes, in code.

    Previously the decision came from a SEPARATE llm call that never saw these
    votes, so the same evidence could yield either answer (in one 906-record
    export, the pattern FAIL/PASS/PASS produced INCLUDE 394 times and EXCLUDE 86
    times). Deriving it here makes the decision reproducible from, and auditable
    against, the evidence the reviewer is shown.

    The rule, sensitivity-first as a first-pass abstract screen should be:
      • FAIL on any DISCRIMINATING element            -> EXCLUDE
      • no element assessable at all (every vote NA)  -> EXCLUDE (nothing to go on)
      • otherwise                                     -> INCLUDE
    PARTIAL and NA never exclude: an abstract that is merely vague belongs at
    full text, not in the discard pile.
    """
    disc = frameworks.discriminating_ids(framework)
    votes = {eid: str((assessment.get(eid) or {}).get("vote") or "NA").upper()
             for eid in frameworks.element_ids(framework)}

    failed = [eid for eid in disc if votes.get(eid) == "FAIL"]
    if failed:
        labels = ", ".join(frameworks.label_for(framework, eid) for eid in failed)
        return "EXCLUDE", f"Fails {labels}"

    if votes and all(v == "NA" for v in votes.values()):
        return "EXCLUDE", "No assessable evidence"

    return "INCLUDE", ""


def _normalize_abstract_decision(
    raw: Dict[str, Any],
    inclusion: List[str],
    exclusion: List[str],
    paper: PaperIn,
    pico: Optional[PicoIn] = None,
    model_name: Optional[str] = None,
    protocol: str = "",
    derive_decision: bool = True,
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

    # Per-element structured assessment with evidence quotes. This single call
    # now also checks the eligibility criteria and supplies the bucket, so the
    # generic path no longer needs a second screening call.
    fw = frameworks.normalize(getattr(pico, "framework", "pico") if pico is not None else "pico")
    pico_assessment: Dict[str, Any] = {
        eid: {"vote": "NA", "evidence": "", "reasoning": ""} for eid in frameworks.element_ids(fw)
    }
    pico_assessment.update({"overall_reasoning": "", "bucket": "", "failed_criteria": []})
    if pico is not None and model_name:
        try:
            pico_assessment = _pico_assess(
                paper, pico, model_name, protocol=protocol,
                inclusion=inclusion, exclusion=exclusion,
            )
        except Exception as e:
            print(f"[normalize_abstract] pico_assess failed: {e}")

    # The decision is DERIVED from the votes above (see _decide_from_votes), so
    # what the reviewer reads is what drove the call. A model that makes its own
    # verdict — LEADS is fine-tuned to do exactly that — keeps it via
    # derive_decision=False, which also preserves its benchmark behaviour.
    if derive_decision:
        decision_upper, derived_bucket = _decide_from_votes(pico_assessment, fw)
    else:
        derived_bucket = ""

    # Criterion-level trace, consumed by the exclusion-bucketing UI. The model
    # returns only the criteria that count against the paper; everything else is
    # a pass. When no assessment ran, leave every criterion unjudged.
    failed = set(pico_assessment.get("failed_criteria") or [])
    assessed_criteria = bool(pico_assessment.get("overall_reasoning"))
    for crit in list(inclusion) + list(exclusion):
        legacy = raw.get(crit)
        if isinstance(legacy, str):
            lv = legacy.strip().upper()
            vote = "PASS" if lv in {"INCLUDE", "PASS", "YES", "TRUE"} else (
                "FAIL" if lv in {"EXCLUDE", "FAIL", "NO", "FALSE"} else "N/A"
            )
        elif crit in failed:
            vote = "FAIL"
        elif assessed_criteria:
            vote = "PASS"
        else:
            vote = "N/A"
        agent_trace[crit] = {
            "vote": vote,
            "reasoning": f"Criterion evaluation: {vote}",
            "evidence": _evidence_for(crit),
        }

    # _pico_assess always returns a record-specific, non-blank overall_reasoning
    # (it judges from the title when no abstract is present), so use it directly.
    # Only fall back to the screener's own reason, then a minimal default.
    overall_reasoning = (
        str(pico_assessment.get("overall_reasoning") or "").strip()
        or reason.strip()
        or ("Meets inclusion criteria" if decision_upper == "INCLUDE" else "Excluded")
    )
    bucket = (
        derived_bucket
        or str(pico_assessment.get("bucket") or "").strip()
        or str(raw.get("bucket") or "").strip()
    )

    return {
        "Bucket": bucket,
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
                inclusion: List[str], exclusion: List[str], protocol: str = "") -> Dict[str, Any]:
    """Run the model's OWN verdict pass.

    Only LEADS needs this: it is fine-tuned to emit a screening verdict, and its
    benchmark numbers depend on that verdict being used as-is. For every other
    model the verdict is derived from the per-element votes instead, so calling
    the generic screener here would be a second LLM round-trip whose answer we
    then throw away.
    """
    if is_leads_model(model_name):
        return screen_paper_leads(paper, pico)
    return AIService.screen_paper(paper, pico, model_name, inclusion, exclusion, protocol)


def _screening_workers(model_name: str, n_papers: int) -> int:
    """Worker count sized to the backend actually serving the model.

    Ollama exposes a handful of inference slots per instance, so 16 workers at a
    local model just queue behind each other. Cloud APIs have no such ceiling.
    """
    local = not any(k in (model_name or "").lower() for k in ("gpt", "claude", "gemini"))
    cap = Config.PARALLEL_SCREENING_WORKERS_LOCAL if local else Config.PARALLEL_SCREENING_WORKERS
    return max(1, min(cap, n_papers or 1))


@app.post("/api/screen/abstract")
def screen_abstract(req: ScreenAbstractRequest):
    paper = _to_backend_paper(req.paper)
    pico = _to_pico(req.pico)
    # Make criteria available to legacy functions that read session_state
    model_name = resolve_model_name(req.model) or resolve_model_name(_default_model())
    leads = is_leads_model(model_name)
    raw = (_screen_one(paper, pico, model_name, req.inclusion, req.exclusion, req.protocol or "")
           if leads else {})
    # Per-PICO assessment uses the reasoning-tier model regardless of screening
    # model, because structured JSON output is its strength.
    pico_model = resolve_for_thinking(req.model)
    return _normalize_abstract_decision(
        raw, req.inclusion, req.exclusion, req.paper, pico=req.pico, model_name=pico_model,
        protocol=req.protocol or "", derive_decision=not leads,
    )


class ScreenAbstractBatchRequest(BaseModel):
    papers: List[PaperIn]
    pico: PicoIn
    inclusion: List[str] = Field(default_factory=list)
    exclusion: List[str] = Field(default_factory=list)
    protocol: Optional[str] = None
    model: Optional[str] = None


@app.post("/api/screen/abstract-batch")
def screen_abstract_batch(req: ScreenAbstractBatchRequest):
    pico = _to_pico(req.pico)
    model_name = resolve_model_name(req.model) or resolve_model_name(_default_model())

    pico_model = resolve_for_thinking(req.model)
    leads = is_leads_model(model_name)

    def _one(p_in: PaperIn) -> Dict[str, Any]:
        bp = _to_backend_paper(p_in)
        raw = (_screen_one(bp, pico, model_name, req.inclusion, req.exclusion, req.protocol or "")
               if leads else {})
        return _normalize_abstract_decision(
            raw, req.inclusion, req.exclusion, p_in, pico=req.pico, model_name=pico_model,
            protocol=req.protocol or "", derive_decision=not leads,
        )

    results: List[Dict[str, Any]] = []
    workers = _screening_workers(pico_model, len(req.papers))
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


def _rerank_finalize(req: RerankRequest, scored: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """Sort the scored papers, decide the relevance floor, and build the response
    payload. Shared by the blocking and streaming rerank endpoints."""
    # Sort descending by LEADS score (most relevant first).
    scored.sort(key=lambda r: r["leads_score"], reverse=True)

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


def _rerank_score_one_factory(req: RerankRequest, pico, model_name: str):
    """Build the per-paper scoring closure used by both rerank endpoints."""
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
    return _score_one


@app.post("/api/papers/rerank")
def papers_rerank(req: RerankRequest):
    """Score each paper against PICO using LEADS-native, return ranked list
    with per-paper LEADS scores. Use LEADS unconditionally (this is its
    trained task) regardless of which model the user selected for thinking
    tasks elsewhere."""
    pico = _to_pico(req.pico)
    # Route to LEADS specifically: this is exactly the per-PICO relevance task
    # the model was fine-tuned for. Override whatever the caller asked for.
    model_name = resolve_model_name(req.model) or LEADS_MODEL_NAME
    _score_one = _rerank_score_one_factory(req, pico, model_name)

    scored: List[Dict[str, Any]] = []
    workers = max(1, min(Config.PARALLEL_SCREENING_WORKERS, len(req.papers) or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_score_one, p) for p in req.papers]):
            try:
                scored.append(fut.result())
            except Exception as e:
                print(f"[papers_rerank] worker error: {e}")

    return _rerank_finalize(req, scored, model_name)


class SimilarRequest(BaseModel):
    seeds: List[Dict[str, Any]] = Field(default_factory=list)   # [{title, doi?, abstract?}]
    corpus: List[Dict[str, Any]] = Field(default_factory=list)  # optional extra pool to rank
    top_k: int = 40
    max_per_seed: int = 40
    model: Optional[str] = None                                  # embedding model override


@app.post("/api/papers/similar")
def papers_similar(req: SimilarRequest):
    """Semantic 'find similar' — the topical-similarity sibling of citation
    snowballing. For each seed paper we pull OpenAlex related_works (algorithmic
    topical similarity, NOT citations, so it reaches same-topic papers with no
    citation link), pool them, then rank by embedding cosine to the seed centroid
    using the local Ollama embedder by default (nothing leaves the machine).
    Returns new papers with a similarity score, ready to add to the screening
    queue exactly like snowball citations."""
    seeds = [s for s in (req.seeds or []) if (s.get("title") or s.get("doi"))]
    if not seeds:
        return {"papers": [], "note": "no seeds provided"}

    # 1. External discovery: OpenAlex related_works per seed (deduped pool).
    pool: Dict[str, Dict[str, Any]] = {}
    for s in seeds:
        for p in OpenAlexService.related(s.get("title", ""), s.get("doi", ""), req.max_per_seed):
            key = (p.id or p.title or "").lower().strip()
            if key and key not in pool:
                pool[key] = {"id": p.id, "title": p.title, "abstract": p.abstract,
                             "url": p.url, "source": p.source}
    # Allow the caller to also fold in a candidate corpus to rank (e.g. a
    # supplementary search) — dedupe against the discovered pool.
    for p in (req.corpus or []):
        key = str(p.get("id") or p.get("title") or "").lower().strip()
        if key and key not in pool:
            pool[key] = {**p, "source": p.get("source", "corpus")}
    candidates = list(pool.values())
    if not candidates:
        return {"papers": [], "note": "no related works found"}

    # 2. Semantic rank by embedding cosine to the seed centroid (local, no key).
    #    If the embedder is unavailable (e.g. Ollama down), keep source order.
    try:
        emb = AIService.get_embedder(req.model)
        if emb is not None:
            import numpy as np
            seed_text = [f"{s.get('title','')}\n{s.get('abstract','')}".strip() for s in seeds]
            cand_text = [f"{c.get('title','')}\n{c.get('abstract','')}".strip() for c in candidates]
            sv = np.asarray(emb.embed_documents(seed_text), dtype=float)
            cv = np.asarray(emb.embed_documents(cand_text), dtype=float)
            centroid = sv.mean(axis=0)
            sims = (cv @ centroid) / (np.linalg.norm(cv, axis=1) * np.linalg.norm(centroid) + 1e-9)
            order = list(np.argsort(-sims))
            ranked = [{**candidates[i], "similarity": round(float(sims[i]), 4)} for i in order]
        else:
            ranked = candidates
    except Exception as e:
        print(f"[papers_similar] embed-rank skipped, using source order: {e}")
        ranked = candidates

    return {"papers": ranked[:req.top_k]}


@app.post("/api/papers/rerank/stream")
def papers_rerank_stream(req: RerankRequest):
    """Same scoring as /papers/rerank, but streamed: emit a `progress` event as
    each paper finishes so the caller can show which article is being scored,
    then a final `done` event carrying the full ranked payload."""
    pico = _to_pico(req.pico)
    model_name = resolve_model_name(req.model) or LEADS_MODEL_NAME
    _score_one = _rerank_score_one_factory(req, pico, model_name)

    total = len(req.papers)
    event_queue: "queue.Queue" = queue.Queue()

    def _run():
        scored: List[Dict[str, Any]] = []
        try:
            workers = max(1, min(Config.PARALLEL_SCREENING_WORKERS, total or 1))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                done_n = 0
                for fut in as_completed([ex.submit(_score_one, p) for p in req.papers]):
                    try:
                        scored.append(fut.result())
                    except Exception as e:
                        print(f"[papers_rerank_stream] worker error: {e}")
                    done_n += 1
                    event_queue.put(("progress", {"done": done_n, "total": total}))
            event_queue.put(("done", _rerank_finalize(req, scored, model_name)))
        except Exception as e:
            event_queue.put(("error", {"message": str(e)}))

    threading.Thread(target=_run, daemon=True).start()

    def _gen():
        while True:
            try:
                event_type, data = event_queue.get(timeout=600)
            except queue.Empty:
                yield f"event: error\ndata: {_json.dumps({'message': 'timeout'})}\n\n"
                return
            yield f"event: {event_type}\ndata: {_json.dumps(data)}\n\n"
            if event_type in ("done", "error"):
                return

    return StreamingResponse(_gen(), media_type="text/event-stream")


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
    # Framework-driven element set (PICO P/I/C/O or PCC Population/Concept/Context).
    fields = [(eid, val) for eid, _lbl, val in pico.element_items()]
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
    pico = _to_pico(req.pico)

    paper_dict = {
        "Title": req.paper.title,
        "Abstract": req.fullText or req.paper.abstract,
        "Source": req.paper.source,
        "URL": req.paper.url,
        "paper_id": req.paper.id,
    }
    raw = AIService.screen_full_text(
        paper_dict, pico,
        resolve_model_name(req.model) or resolve_model_name(_default_model()),
        inclusion=req.inclusion, exclusion=req.exclusion,
    )

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

    # The full-text screener returns per-criterion verdicts under
    # "criteria_evaluations". Match each user criterion to its verdict by exact
    # then whitespace/case-normalised text, so the PRISMA exclusion "reasons" name
    # the specific criterion a paper failed rather than a generic PICO bucket.
    _ce = raw.get("criteria_evaluations")
    _ce = _ce if isinstance(_ce, dict) else {}
    _ce_norm = {re.sub(r"\s+", " ", str(k).strip().lower()): str(v).upper() for k, v in _ce.items()}

    def _crit_verdict(crit: str) -> str:
        if crit in _ce:
            return str(_ce[crit]).upper()
        return _ce_norm.get(re.sub(r"\s+", " ", str(crit).strip().lower()), str(raw.get(crit, "INCLUDE")).upper())

    for crit in (req.inclusion or []):
        v = _crit_verdict(crit)
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
        v = _crit_verdict(crit)
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


# How many citations we are willing to pull from one source and direction for
# one seed. Snowballing is meant to be exhaustive, so the default behaviour is
# to take the whole list; this is only a guard against a seminal paper with
# tens of thousands of citing works. Whatever it cuts off is REPORTED, never
# silently dropped.
_CITATION_CEILING = 500

_EPMC_PAGE = 100          # Europe PMC's maximum page size for these endpoints
_OA_BATCH = 50            # ids per openalex_id: filter, keeps the URL sane
_SS_PAGE = 100


class CitationsRequest(BaseModel):
    paper_id: str = ""
    source: str = ""
    title: str
    snowball_type: str = "Both"  # "Both" | "Backward (References)" | "Forward (Cited by)"
    max_per: int = 10
    sources: List[str] = Field(default_factory=lambda: ["PubMed", "Semantic Scholar", "Europe PMC"])
    # Fetch the whole reference / citation list and let the caller rank it,
    # rather than keeping whichever N the API happened to return first. The old
    # behaviour took page one in each source's own order, which for a
    # bibliography means the papers printed first, and those skew to
    # introduction and background rather than to the closest work.
    fetch_all: bool = True


def _epmc_resolve(title: str, paper_id: str) -> Optional[Tuple[str, str]]:
    """Resolve a paper to (source, id) on Europe PMC. Falls back to a title search.

    Memoised for the run: snowballing revisits the same papers constantly, and
    every repeat used to be a fresh request Europe PMC had to serve.
    """
    if paper_id and paper_id.isdigit():
        return ("MED", paper_id)

    def _lookup():
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        q = paper_id if (paper_id and not paper_id.isdigit()) else f'TITLE:"{title}"'
        params = {"query": q, "format": "json", "pageSize": 1, "resultType": "lite"}
        r = limited_get(url, params=params, timeout=6)
        if r is None or not r.ok:
            return None
        try:
            for it in r.json().get("resultList", {}).get("result", []):
                return ((it.get("source") or "MED"), str(it.get("id") or ""))
        except Exception as e:
            print(f"[epmc_resolve] {e}")
        return None

    return cached("epmc_resolve", paper_id or title, _lookup)


def _epmc_links(source: str, pid: str, direction: str, max_per: int,
                fetch_all: bool = True) -> Tuple[List[Dict[str, Any]], int]:
    """direction: 'references' (backward) or 'citations' (forward).

    Returns ``(citations, available)`` where `available` is the total the
    source reports holding, so a caller can say "85 available, 20 kept"
    instead of presenting a truncated list as the whole answer.
    """
    out: List[Dict[str, Any]] = []
    available = 0
    ctype = "backward" if direction == "references" else "forward"
    key = "referenceList" if direction == "references" else "citationList"
    inner = "reference" if direction == "references" else "citation"
    want = _CITATION_CEILING if fetch_all else max_per
    try:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{source}/{pid}/{direction}"
        page = 1
        while len(out) < want:
            params = {"pageSize": min(_EPMC_PAGE, want - len(out)), "page": page, "format": "json"}
            resp = limited_get(url, params=params, timeout=8)
            if resp is None or not resp.ok:
                break
            r = resp.json()
            available = int(r.get("hitCount") or 0) or available
            items = (r.get(key) or {}).get(inner, []) or []
            if not items:
                break
            for it in items:
                cid = str(it.get("id") or it.get("doi") or "")
                out.append({
                    "id": cid,
                    "title": (it.get("title") or it.get("source") or "Untitled").strip(),
                    "abstract": (it.get("abstractText") or "").strip(),
                    "url": (f"https://pubmed.ncbi.nlm.nih.gov/{cid}/" if cid.isdigit() else (f"https://doi.org/{cid}" if "/" in cid else "")),
                    "source": f"Europe PMC ({'Reference' if ctype == 'backward' else 'Cited by'})",
                    "citation_type": ctype,
                    "year": it.get("pubYear") or None,
                    "cited_by_count": it.get("citedByCount"),
                })
            if len(items) < params["pageSize"]:
                break
            page += 1
    except Exception as e:
        print(f"[epmc_links] {direction} {e}")
    return out, max(available, len(out))


def _openalex_resolve_work(title: str, paper_id: str) -> Optional[Dict[str, Any]]:
    """Memoised for the run: the same work is resolved under many seeds."""
    return cached("oa_work", paper_id or title, lambda: _openalex_resolve_work_uncached(title, paper_id))


def _openalex_resolve_work_uncached(title: str, paper_id: str) -> Optional[Dict[str, Any]]:
    try:
        # DOI lookup
        if paper_id and "/" in paper_id and paper_id.startswith("10."):
            r = limited_get(
                f"https://api.openalex.org/works/doi:{paper_id}",
                params=_polite_params(), timeout=6,
            )
            if r is not None and r.status_code == 200:
                return r.json()
        # Title search
        r = limited_get(
            "https://api.openalex.org/works",
            params=_polite_params({"search": title, "per_page": 1}), timeout=6,
        )
        if r is not None and r.status_code == 200:
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
        # Carried on the record because the full-text fetch needs it: mining a
        # DOI back out of the URL fails whenever the URL is a PMC or publisher
        # link, and the paper then came back "no_doi" though it was looked up
        # BY its DOI.
        "doi": doi,
        "source": f"OpenAlex ({'Reference' if ctype == 'backward' else 'Cited by'})",
        "year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count"),
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
    forbid redistribution), and can be queried directly by PMID or DOI.

    Memoised: the same citation appears under many seed papers.
    """
    if ext_id and ext_id.isdigit():
        q = f"EXT_ID:{ext_id}"
    elif ext_id and "/" in ext_id:
        q = f'DOI:"{ext_id}"'
    elif title:
        q = f'TITLE:"{title}"'
    else:
        return ""

    def _lookup() -> str:
        r = limited_get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": q, "resultType": "core", "format": "json", "pageSize": 1},
            timeout=6,
        )
        if r is None or not r.ok:
            return ""
        try:
            results = (r.json().get("resultList") or {}).get("result", []) or []
        except Exception as e:
            print(f"[abstract_epmc] {e}")
            return ""
        if not results:
            return ""
        raw = (results[0].get("abstractText") or "").strip()
        # EPMC abstracts may carry light HTML/JATS markup — strip it.
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw)).strip()

    return cached("epmc_abstract", q, _lookup)


# Enough workers to keep each host's bucket saturated while one call is in
# flight, and no more: the buckets, not this number, decide the request rate.
_ENRICH_WORKERS = 8


def _enrich_missing_abstracts(items: List[Dict[str, Any]], cap: int = 80) -> int:
    """Backfill empty abstracts for snowballed citations.

    Europe PMC reference/citation lists return title-only entries, which starves
    the downstream PICO screening. For each missing abstract we try Europe PMC
    search first (best coverage, direct PMID/DOI lookup), then fall back to
    OpenAlex. Lookups are deduplicated by title and bounded by `cap`.
    Returns the number of papers resolved.
    """
    # Which citations need a lookup, de-duplicated by title within this batch.
    todo: Dict[str, Tuple[str, str]] = {}          # key -> (title, ext_id)
    targets: Dict[str, List[Dict[str, Any]]] = {}  # key -> the rows to fill
    for it in items:
        if len((it.get("abstract") or "").strip()) >= 40:
            continue  # already has a usable abstract
        title = (it.get("title") or "").strip()
        ext_id = str(it.get("id") or "")
        if not title and not ext_id:
            continue
        key = title.lower() or ext_id
        targets.setdefault(key, []).append(it)
        if key not in todo and len(todo) < cap:
            todo[key] = (title, ext_id)

    def _resolve_one(key: str) -> Tuple[str, str]:
        title, ext_id = todo[key]
        abstract = _abstract_via_epmc(title, ext_id)
        if not abstract:                      # EPMC miss → try OpenAlex
            work = _openalex_resolve_work(title, ext_id)
            abstract = _reconstruct_oa_abstract(work.get("abstract_inverted_index")) if work else ""
        return key, abstract

    # Run the lookups concurrently. This does NOT raise the request rate: every
    # call still waits for a token from its host's bucket, which is set below
    # each service's published limit. What it removes is the dead time, where
    # the previous lookup's round trip had to finish before the next could
    # start and the achieved rate sat near 0.1 requests a second.
    resolved = 0
    if todo:
        with ThreadPoolExecutor(max_workers=_ENRICH_WORKERS) as pool:
            for key, abstract in pool.map(_resolve_one, list(todo)):
                if not abstract:
                    continue
                for row in targets.get(key, []):
                    row["abstract"] = abstract
                resolved += 1
    print(f"[enrich_abstracts] resolved {resolved}/{len(todo)} lookups "
          f"({len(items)} citations, cache {cache_stats()})")
    return resolved


def _openalex_links(work: Dict[str, Any], direction: str, max_per: int,
                    fetch_all: bool = True) -> Tuple[List[Dict[str, Any]], int]:
    """Returns ``(citations, available)``."""
    out: List[Dict[str, Any]] = []
    available = 0
    want = _CITATION_CEILING if fetch_all else max_per
    try:
        if direction == "backward":
            # referenced_works are canonical OpenAlex ids (openalex.org/W…).
            # Batch-resolve them through the API rather than dereferencing each
            # web URL (which returns HTML, not JSON).
            refs = [r.split("/")[-1] for r in (work.get("referenced_works") or []) if r]
            available = len(refs)
            for i in range(0, min(len(refs), want), _OA_BATCH):
                batch = refs[i:i + _OA_BATCH]
                rr = limited_get(
                    "https://api.openalex.org/works",
                    params=_polite_params({"filter": f"openalex_id:{'|'.join(batch)}",
                                           "per_page": len(batch)}),
                    timeout=12,
                )
                if rr is None or rr.status_code != 200:
                    break
                for w in rr.json().get("results", []):
                    out.append(_openalex_minimal(w, "backward"))
        else:
            # Forward citations via the `cites` filter (cited_by_api_url was
            # removed from the OpenAlex work schema).
            wid = (work.get("id") or "").split("/")[-1]
            cursor = "*"
            while wid and len(out) < want:
                rr = limited_get(
                    "https://api.openalex.org/works",
                    params=_polite_params({"filter": f"cites:{wid}",
                                           "per_page": min(200, want - len(out)),
                                           "cursor": cursor}),
                    timeout=12,
                )
                if rr is None or rr.status_code != 200:
                    break
                body = rr.json()
                available = int((body.get("meta") or {}).get("count") or 0) or available
                results = body.get("results", []) or []
                for w in results:
                    out.append(_openalex_minimal(w, "forward"))
                cursor = (body.get("meta") or {}).get("next_cursor") or ""
                if not results or not cursor:
                    break
    except Exception as e:
        print(f"[openalex_links] {direction} {e}")
    return out, max(available, len(out))


def _ss_resolve_id(title: str) -> Optional[str]:
    """Memoised. Semantic Scholar allows ONE request a second to a key holder,
    and unauthenticated callers share one pool worldwide, so a repeat lookup is
    the most expensive request this code can make."""
    return cached("ss_id", title, lambda: _ss_resolve_id_uncached(title))


def _ss_resolve_id_uncached(title: str) -> Optional[str]:
    try:
        r = limited_get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": title, "limit": 1, "fields": "paperId"}, timeout=8,
        )
        if r is not None and r.status_code == 200:
            results = r.json().get("data", [])
            if results:
                return results[0].get("paperId")
    except Exception as e:
        print(f"[ss_resolve] {e}")
    return None


def _ss_links(pid: str, direction: str, max_per: int,
              fetch_all: bool = True) -> Tuple[List[Dict[str, Any]], int]:
    """Returns ``(citations, available)``."""
    out: List[Dict[str, Any]] = []
    available = 0
    want = _CITATION_CEILING if fetch_all else max_per
    outer = "citedPaper" if direction == "backward" else "citingPaper"
    try:
        endpoint = "references" if direction == "backward" else "citations"
        url = f"https://api.semanticscholar.org/graph/v1/paper/{pid}/{endpoint}"
        offset = 0
        while len(out) < want:
            r = limited_get(url, params={
                "fields": "paperId,title,abstract,url,year,citationCount",
                "limit": min(_SS_PAGE, want - len(out)), "offset": offset,
            }, timeout=8)
            if r is None or not r.ok:
                break
            body = r.json()
            total = body.get("total")
            if isinstance(total, int):
                available = max(available, total)
            data = body.get("data", []) or []
            if not data:
                break
            for it in data:
                paper = it.get(outer) or {}
                out.append({
                    "id": paper.get("paperId", ""),
                    "title": (paper.get("title") or "Untitled").strip(),
                    "abstract": (paper.get("abstract") or "").strip(),
                    "url": paper.get("url") or "",
                    "source": f"Semantic Scholar ({'Reference' if direction == 'backward' else 'Cited by'})",
                    "citation_type": "backward" if direction == "backward" else "forward",
                    "year": paper.get("year"),
                    "cited_by_count": paper.get("citationCount"),
                })
            nxt = body.get("next")
            if not nxt:
                break
            offset = nxt
    except Exception as e:
        print(f"[ss_links] {direction} {e}")
    return out, max(available, len(out))


@app.post("/api/citations")
def citations(req: CitationsRequest):
    want_back = req.snowball_type in {"Both", "Backward (References)"}
    want_fwd = req.snowball_type in {"Both", "Forward (Cited by)"}
    sources = set(s for s in req.sources)

    # Every source reports how many it HOLDS as well as what it handed over,
    # so the caller can state "85 available, 20 kept" rather than presenting a
    # truncated list as the whole answer.
    Haul = Tuple[List[Dict[str, Any]], Dict[str, int]]

    def _from_epmc() -> Haul:
        got: List[Dict[str, Any]] = []
        avail: Dict[str, int] = {}
        resolved = _epmc_resolve(req.title, req.paper_id)
        if resolved:
            src, pid = resolved
            if want_back:
                rows, n = _epmc_links(src, pid, "references", req.max_per, req.fetch_all)
                got.extend(rows); avail["Europe PMC · backward"] = n
            if want_fwd:
                rows, n = _epmc_links(src, pid, "citations", req.max_per, req.fetch_all)
                got.extend(rows); avail["Europe PMC · forward"] = n
        return got, avail

    def _from_openalex() -> Haul:
        got: List[Dict[str, Any]] = []
        avail: Dict[str, int] = {}
        work = _openalex_resolve_work(req.title, req.paper_id)
        if work:
            if want_back:
                rows, n = _openalex_links(work, "backward", req.max_per, req.fetch_all)
                got.extend(rows); avail["OpenAlex · backward"] = n
            if want_fwd:
                rows, n = _openalex_links(work, "forward", req.max_per, req.fetch_all)
                got.extend(rows); avail["OpenAlex · forward"] = n
        return got, avail

    def _from_ss() -> Haul:
        got: List[Dict[str, Any]] = []
        avail: Dict[str, int] = {}
        ss_id = _ss_resolve_id(req.title)
        if ss_id:
            if want_back:
                rows, n = _ss_links(ss_id, "backward", req.max_per, req.fetch_all)
                got.extend(rows); avail["Semantic Scholar · backward"] = n
            if want_fwd:
                rows, n = _ss_links(ss_id, "forward", req.max_per, req.fetch_all)
                got.extend(rows); avail["Semantic Scholar · forward"] = n
        return got, avail

    # The three sources are independent and hit different hosts, so they wait
    # concurrently rather than in turn. Each still passes through its own host
    # bucket, so no service sees a higher rate than it did before; a seed
    # simply costs the slowest source instead of the sum of all three.
    jobs = []
    if "Europe PMC" in sources or "PubMed" in sources:
        jobs.append(_from_epmc)
    if "OpenAlex" in sources:
        jobs.append(_from_openalex)
    if "Semantic Scholar" in sources:
        jobs.append(_from_ss)

    out: List[Dict[str, Any]] = []
    available: Dict[str, int] = {}
    if jobs:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            for got, avail in pool.map(lambda fn: fn(), jobs):
                out.extend(got)
                available.update(avail)

    # Reference/citation lists often omit abstracts — backfill them so the
    # downstream screening has something to work with.
    _enrich_missing_abstracts(out)

    # Strip source markup (e.g. "<i>T. gondii</i>") from titles/abstracts.
    for it in out:
        it["title"] = clean_markup(it.get("title"))
        if it.get("abstract"):
            it["abstract"] = clean_markup(it["abstract"])

    total_available = sum(available.values())
    print(f"[citations] {req.title[:50]!r}: fetched {len(out)} of {total_available} available "
          f"{available}")
    return {
        "citations": out,
        # What each source and direction actually holds, so the reviewer can be
        # told what was left behind instead of discovering it later.
        "available": available,
        "total_available": total_available,
        "ceiling": _CITATION_CEILING,
    }


# ---------------------------------------------------------------------------
# Retraction / integrity check — Crossref carries the Retraction Watch data
# (keyless), so a review can flag retracted or withdrawn studies rather than
# silently including one.
# ---------------------------------------------------------------------------

_RETRACTION_TYPES = {"retraction", "withdrawal", "removal", "expression_of_concern", "partial_retraction"}


def _crossref_retraction(doi: str) -> Optional[Dict[str, Any]]:
    """Return retraction-notice info for a DOI (via Crossref `updated-by`), or None."""
    if not doi:
        return None
    try:
        r = requests.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": "EvidenceEngine/1.0"},
            timeout=15,
        )
        if not r.ok:
            return None
        msg = r.json().get("message", {}) or {}
    except Exception:
        return None
    hits = [u for u in (msg.get("updated-by") or [])
            if (u.get("type") or "").lower().replace("-", "_") in _RETRACTION_TYPES]
    if not hits:
        return None
    return {
        "types": sorted({(u.get("type") or "").lower().replace("-", "_") for u in hits}),
        "notice_doi": hits[0].get("DOI", ""),
    }


class RetractionCheckRequest(BaseModel):
    papers: List[Dict[str, Any]]   # [{id, doi?, url?, title?}]


@app.post("/api/papers/retractions")
def papers_retractions(req: RetractionCheckRequest):
    """Flag retracted / withdrawn papers using Crossref's Retraction Watch data.
    Returns the ids carrying a retraction notice so the UI can badge them."""
    out: List[Dict[str, Any]] = []
    checked = 0
    for p in (req.papers or [])[:120]:
        doi = (p.get("doi") or "").strip() or _extract_doi(p.get("url", "") or "", p.get("title", "") or "")
        if not doi:
            continue
        checked += 1
        info = _crossref_retraction(doi)
        if info:
            out.append({"id": p.get("id", ""), "doi": doi, "types": info["types"], "notice_doi": info.get("notice_doi", "")})
    return {"retracted": out, "checked": checked}


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
        if not text or len(text) <= 200:
            return None
        # Some publishers' fonts spell symbols as letters, so the text layer
        # says "10mM" where the paper says "10 µM" and "ratio ¼ 3.96" where it
        # says "=". Nothing downstream can recover either: a model reads the
        # wrong unit correctly, and the quote it cites matches the text, so
        # verifying against the document confirms the error.
        return pdf_text_repair.repair(text, pdf_bytes)
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
    """Download the PMC-hosted PDF for a given PMC ID.

    NO LONGER IN THE LADDER. Measured over a 411-record acquisition run it
    succeeded 0 times: NCBI answers the article PDF path with HTTP 200 and an
    HTML block page, and Europe PMC's rendered PDF returns 403 regardless of
    user agent. Kept only because the bulk-import path still calls it for
    records a reviewer supplies a PMC id for by hand. Prefer
    `_fetch_pmc_efetch_text`.
    """
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
                headers={"User-Agent": "EvidenceEngine/1.0 (research@example.com)"},
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


_OA_UA = {"User-Agent": "EvidenceEngine/1.0"}

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
# Addresses these APIs reject or that indicate a placeholder rather than a real
# person. Sending one of these is worse than sending nothing.
_PLACEHOLDER_EMAIL_DOMAINS = ("example.com", "example.org", "localhost", "test.com")


def _contact_email() -> str:
    """The user's OWN contact address for scholarly APIs that require one.

    Supplied per request from the browser profile. Returns "" when absent or
    obviously a placeholder, and every caller must degrade to an anonymous
    request in that case rather than inventing an address.
    """
    from request_creds import get_cred
    email = (get_cred("contact_email") or "").strip()
    if not email or not _EMAIL_RE.match(email):
        return ""
    if email.lower().rsplit("@", 1)[-1] in _PLACEHOLDER_EMAIL_DOMAINS:
        return ""
    return email


def _polite_params(extra: Optional[dict] = None) -> dict:
    """Query params for OpenAlex and Crossref.

    `mailto` is what Crossref asks for. OpenAlex asks for it too, but since
    February 2026 it is usage-based: an unkeyed caller shares a free budget of
    about a hundred requests a day WITH EVERY OTHER MACHINE ON THE SAME NETWORK,
    so on a university address it is usually spent before you start. A free key
    carries its own budget and is the only way to run a real search.
    """
    p = dict(extra or {})
    email = _contact_email()
    if email:
        p["mailto"] = email
    key = get_cred("openalex") or os.getenv("OPENALEX_API_KEY", "")
    if key:
        p["api_key"] = key
    return p

# OpenAlex asks for <= 3 requests/second from anonymous callers.
_OPENALEX_MIN_GAP = 0.35
_openalex_last = 0.0


def _openalex_oa_locations(doi: str, title: str = "") -> Tuple[List[str], str]:
    """Resolve a work to its open-access locations WITHOUT an API key or email.

    Returns ``(candidate_urls, oa_status)`` where oa_status is one of
    "pdf" (a direct PDF url exists), "landing" (open access but only a landing
    page), "closed" (not open access) or "unknown" (could not resolve).

    OpenAlex carries the same open-access index Unpaywall does but, unlike
    Unpaywall, does not require a contact email — Unpaywall rejects any call
    without a real address (HTTP 422), which silently disabled this whole tier.
    """
    global _openalex_last
    if not doi and not title:
        return [], "unknown"
    gap = time.time() - _openalex_last
    if gap < _OPENALEX_MIN_GAP:
        time.sleep(_OPENALEX_MIN_GAP - gap)
    try:
        if doi:
            url = f"https://api.openalex.org/works/doi:{quote(doi, safe='')}"
            r = requests.get(url, timeout=20, headers=_OA_UA, params=_polite_params())
        else:
            r = requests.get("https://api.openalex.org/works", timeout=20, headers=_OA_UA,
                             params=_polite_params({"filter": f"title.search:{title[:200]}",
                                                    "per_page": 1}))
        _openalex_last = time.time()
        if r.status_code != 200:
            print(f"[openalex_oa] {doi or title[:40]}: HTTP {r.status_code}")
            return [], "unknown"
        data = r.json() or {}
        if "results" in data:
            results = data.get("results") or []
            if not results:
                return [], "unknown"
            data = results[0]
    except Exception as e:
        print(f"[openalex_oa] {doi or title[:40]}: {e}")
        return [], "unknown"

    oa = data.get("open_access") or {}
    locs = [l for l in ([data.get("best_oa_location")] + (data.get("locations") or []))
            if isinstance(l, dict)]
    pdfs = [l["pdf_url"] for l in locs if l.get("pdf_url")]
    pages = [l["landing_page_url"] for l in locs
             if l.get("landing_page_url") and l.get("is_oa")]
    if oa.get("oa_url"):
        pages.append(oa["oa_url"])

    # De-duplicate while keeping PDFs ahead of landing pages.
    seen, ordered = set(), []
    for u in pdfs + pages:
        if u and u not in seen:
            seen.add(u)
            ordered.append(u)
    if pdfs:
        return ordered, "pdf"
    if not oa.get("is_oa"):
        return ordered, "closed"
    return ordered, "landing" if ordered else "closed"


_PDF_LINK_RE = re.compile(
    r'(?:citation_pdf_url"\s*content="([^"]+)")'
    r'|(?:content="([^"]+)"\s*name="citation_pdf_url")'
    r'|(?:href="([^"]+\.pdf(?:\?[^"]*)?)")',
    re.I,
)


def _pdf_from_landing_page(page_url: str) -> Optional[bytes]:
    """Follow an open-access landing page to the PDF it advertises.

    Publishers routinely expose the file through the ``citation_pdf_url`` meta
    tag rather than a direct link in the OA index, which is why a record can be
    open access and still come back with no downloadable PDF.
    """
    if not page_url or not page_url.startswith("http"):
        return None
    try:
        r = requests.get(page_url, timeout=25, allow_redirects=True, headers=_OA_UA)
        if r.status_code != 200:
            return None
        # The landing page may itself already be the PDF.
        if r.content[:5] == b"%PDF-":
            return r.content
        html = r.text[:400_000]
    except Exception as e:
        print(f"[landing_page] {page_url[:60]}: {e}")
        return None

    for m in _PDF_LINK_RE.finditer(html):
        href = m.group(1) or m.group(2) or m.group(3)
        if not href:
            continue
        target = urljoin(r.url, html_unescape(href))
        try:
            rr = requests.get(target, timeout=25, allow_redirects=True, headers=_OA_UA)
            ct = (rr.headers.get("content-type") or "").lower()
            if rr.status_code == 200 and ("pdf" in ct or rr.content[:5] == b"%PDF-"):
                return rr.content
        except Exception:
            continue
    return None


def _fetch_oa_pdf(doi: str, title: str = "") -> Tuple[Optional[bytes], str, str]:
    """Open-access PDF for a DOI. Returns (pdf_bytes, source_url, oa_status).

    OpenAlex first because it needs no credentials at all. Unpaywall is then
    tried as a SECOND, independent index, but only when the user has put their
    own contact address in their profile — without one Unpaywall returns 422,
    which is exactly how this tier came to be silently dead.
    """
    urls, status = _openalex_oa_locations(doi, title)

    if doi and _contact_email():
        found = _fetch_unpaywall_pdf(doi)
        if found:
            pdf, src = found
            return pdf, src, "pdf"

    for u in urls:
        try:
            r = requests.get(u, timeout=30, allow_redirects=True, headers=_OA_UA)
            ct = (r.headers.get("content-type") or "").lower()
            if r.status_code == 200 and ("pdf" in ct or r.content[:5] == b"%PDF-"):
                return r.content, u, status
        except Exception as e:
            print(f"[oa_pdf] {u[:60]}: {e}")
        # Not a PDF: it is probably the landing page, so follow it.
        pdf = _pdf_from_landing_page(u)
        if pdf:
            return pdf, u, status
    return None, "", status


def _fetch_unpaywall_pdf(doi: str) -> Optional[Tuple[bytes, str]]:
    """Resolve a DOI to an open-access PDF via the Unpaywall API and download
    the bytes. Returns (pdf_bytes, source_url) on success."""
    email = _contact_email()
    if not doi or not email:
        return None
    try:
        api = f"https://api.unpaywall.org/v2/{quote(doi, safe='')}"
        r = requests.get(api, timeout=15, params={"email": email},
                         headers={"User-Agent": "EvidenceEngine/1.0"})
        if r.status_code != 200:
            # 422 means the address was rejected. Say so, rather than letting a
            # config problem look exactly like a paywalled paper.
            print(f"[unpaywall {doi}] HTTP {r.status_code}: {r.text[:120]}")
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


_NCBI_IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"


def _ncbi_ids(ident: str) -> Dict[str, str]:
    """Map any of DOI / PMID / PMCID to the other two via NCBI's ID converter.

    The ladder used to give up on a record the moment it had no DOI, even when
    it was holding a PubMed id NCBI would happily convert. On a 411-record run
    that abandoned 16 records before a single open-access index was consulted.
    """
    ident = (ident or "").strip()
    if not ident:
        return {}
    params = {"ids": ident, "format": "json", "tool": "EvidenceEngine"}
    email = _contact_email()
    if email:
        params["email"] = email
    try:
        r = requests.get(_NCBI_IDCONV, params=params, headers=_OA_UA, timeout=25)
        if r.status_code != 200:
            return {}
        rec = ((r.json() or {}).get("records") or [{}])[0]
    except Exception as e:
        print(f"[ncbi_idconv] {ident[:40]}: {e}")
        return {}
    if rec.get("status") == "error":
        return {}
    return {k: v for k, v in (
        ("pmcid", rec.get("pmcid") or ""),
        ("pmid", rec.get("pmid") or ""),
        ("doi", rec.get("doi") or ""),
    ) if v}


def _xml_body_text(xml: str, min_chars: int = 2000) -> Optional[str]:
    """Plain text from a JATS document, but ONLY when it carries a real <body>.

    NCBI answers for every PMC record, including ones whose publisher withholds
    text-mining rights; those come back as front matter and an abstract. Without
    this check the ladder would accept a few hundred words of metadata as the
    full text and every downstream extraction would quietly work off an
    abstract.
    """
    if not xml or "<body" not in xml:
        return None
    try:
        soup = BeautifulSoup(xml, "lxml-xml")
        body = soup.find("body")
        if body is None:
            return None
        text = body.get_text(separator="\n", strip=True)
        return text if text and len(text) >= min_chars else None
    except Exception as e:
        print(f"[jats parse] {e}")
        return None


def _fetch_pmc_efetch_text(pmcid: str) -> Optional[str]:
    """Full text for a PMC record through NCBI E-utilities.

    This replaces the old PMC-PDF tier, which scored 0 out of 411 on a real
    run: NCBI answers the article PDF path with HTTP 200 and an HTML block
    page, so it read as a success and returned nothing. E-utilities is the
    interface NCBI actually supports for programmatic access and is not
    rate-blocked the same way.
    """
    if not pmcid:
        return None
    pmcid = pmcid.upper()
    if not pmcid.startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    params = {"db": "pmc", "id": pmcid, "retmode": "xml", "tool": "EvidenceEngine"}
    email = _contact_email()
    if email:
        params["email"] = email
    # An NCBI key lifts the shared limit from 3 to 10 requests/second, which is
    # the difference between minutes and an hour on a review-sized corpus.
    key = get_cred("ncbi") or os.getenv("NCBI_API_KEY", "")
    if key:
        params["api_key"] = key
    try:
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                         params=params, headers=_OA_UA, timeout=40)
        if r.status_code != 200:
            print(f"[efetch {pmcid}] HTTP {r.status_code}")
            return None
        return _xml_body_text(r.text)
    except Exception as e:
        print(f"[efetch {pmcid}] {e}")
        return None


def _fetch_s2_oa_pdf(doi: str) -> Optional[Tuple[bytes, str]]:
    """Semantic Scholar's own open-access PDF field, a separate index again."""
    if not doi:
        return None
    headers = dict(_OA_UA)
    key = get_cred("semantic_scholar") or Config.SEMANTIC_SCHOLAR_KEY
    if key:
        headers["x-api-key"] = key
    try:
        r = requests.get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi, safe='')}",
                         params={"fields": "openAccessPdf"}, headers=headers, timeout=30)
        if r.status_code != 200:
            return None
        url = ((r.json() or {}).get("openAccessPdf") or {}).get("url") or ""
    except Exception as e:
        print(f"[s2_oa {doi}] {e}")
        return None
    if not url:
        return None
    pdf = _download_pdf(url)
    return (pdf, url) if pdf else None


_OPENAIRE_PDF_RE = re.compile(r'"(https?://[^"]+?\.pdf[^"]*)"', re.I)


def _fetch_openaire_pdf(doi: str) -> Optional[Tuple[bytes, str]]:
    """Green open-access copies held in the OpenAIRE aggregator.

    Reaches institutional and national repositories that neither OpenAlex nor
    Unpaywall surfaces a downloadable link for.
    """
    if not doi:
        return None
    try:
        r = requests.get("https://api.openaire.eu/search/publications",
                         params={"doi": doi, "format": "json"}, headers=_OA_UA, timeout=30)
        if r.status_code != 200:
            return None
        body = r.text
    except Exception as e:
        print(f"[openaire {doi}] {e}")
        return None
    seen = set()
    for m in _OPENAIRE_PDF_RE.finditer(body):
        url = m.group(1)
        if url in seen:
            continue
        seen.add(url)
        if len(seen) > 4:
            break
        pdf = _download_pdf(url)
        if pdf:
            return pdf, url
    return None


def _download_pdf(url: str) -> Optional[bytes]:
    """GET a URL and return the bytes only if they really are a PDF.

    Several hosts answer a blocked request with HTTP 200 and an HTML notice, so
    status alone is not evidence of success; the magic number is.
    """
    if not url or not url.startswith("http"):
        return None
    try:
        r = requests.get(url, headers=_OA_UA, timeout=40, allow_redirects=True)
    except Exception as e:
        print(f"[download_pdf] {url[:70]}: {e}")
        return None
    if r.status_code != 200:
        return None
    ct = (r.headers.get("content-type") or "").lower()
    if r.content[:5] == b"%PDF-" or ("pdf" in ct and r.content[:5] == b"%PDF-"):
        return r.content
    return None


# ---- publisher text-and-data-mining APIs -----------------------------------
# Subscription content, accessed through the interface each publisher provides
# FOR text mining, with the user's own institutional credential. Nothing here
# works without a key the user supplied, and no request is made without one.

_LIBKEY_API = "https://public-api.thirdiron.com/public/v1/libraries"


def _libkey_lookup(doi: str) -> Dict[str, str]:
    """Ask LibKey where this library's entitled copy of a DOI lives.

    LibKey (Third Iron) maps a DOI to the copy the subscribing institution is
    entitled to, which is the same question a library link resolver answers but
    in one call and with a direct file link. Returns {} unless the user has
    configured their library's id and token, so a library without LibKey pays
    nothing for this.

    Note the download still requires entitlement: off the institution's network
    the link is for the READER to open, not for us to fetch. Both are returned
    so the caller can try the fetch and still hand the link over if it fails.
    """
    lib = get_cred("libkey_id")
    token = get_cred("libkey")
    if not (lib and token and doi):
        return {}
    try:
        r = requests.get(f"{_LIBKEY_API}/{quote(lib, safe='')}/articles/doi/{quote(doi, safe='')}",
                         params={"access_token": token}, headers=_OA_UA, timeout=25)
        if r.status_code != 200:
            print(f"[libkey {doi}] HTTP {r.status_code}")
            return {}
        data = (r.json() or {}).get("data") or {}
    except Exception as e:
        print(f"[libkey {doi}] {e}")
        return {}
    return {k: v for k, v in (
        ("pdf", data.get("fullTextFile") or ""),
        ("article", data.get("contentLocation") or ""),
        ("retraction", data.get("retractionNoticeUrl") or ""),
    ) if v}


def _fetch_libkey_pdf(doi: str) -> Optional[Tuple[bytes, str]]:
    """Try the entitled copy LibKey points at. Succeeds only where this machine
    already holds the entitlement (institutional network or VPN)."""
    link = _libkey_lookup(doi).get("pdf")
    if not link:
        return None
    pdf = _download_pdf(link)
    return (pdf, link) if pdf else None


def _fetch_elsevier_tdm(doi: str) -> Optional[Tuple[str, str]]:
    """Elsevier full-text article API. Key from dev.elsevier.com; entitlement
    comes from the institution's subscription."""
    key = get_cred("scopus") or os.getenv("SCOPUS_API_KEY", "")
    if not key or not doi:
        return None
    headers = {**_OA_UA, "X-ELS-APIKey": key, "Accept": "text/xml"}
    token = get_cred("elsevier_insttoken")
    if token:
        headers["X-ELS-Insttoken"] = token
    try:
        r = requests.get(f"https://api.elsevier.com/content/article/doi/{quote(doi, safe='')}",
                         params={"view": "FULL"}, headers=headers, timeout=45)
        if r.status_code != 200:
            print(f"[elsevier_tdm {doi}] HTTP {r.status_code}")
            return None
        soup = BeautifulSoup(r.text, "lxml-xml")
        node = soup.find("originalText") or soup.find("body") or soup
        text = node.get_text(separator="\n", strip=True)
        return (text, "Elsevier TDM") if text and len(text) > 2000 else None
    except Exception as e:
        print(f"[elsevier_tdm {doi}] {e}")
        return None


def _fetch_wiley_tdm(doi: str) -> Optional[Tuple[bytes, str]]:
    """Wiley Online Library TDM. Token requested by the institution's library."""
    token = get_cred("wiley")
    if not token or not doi:
        return None
    try:
        r = requests.get(f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{quote(doi, safe='')}",
                         headers={**_OA_UA, "Wiley-TDM-Client-Token": token},
                         timeout=45, allow_redirects=True)
        if r.status_code != 200:
            print(f"[wiley_tdm {doi}] HTTP {r.status_code}")
            return None
        return (r.content, "Wiley TDM") if r.content[:5] == b"%PDF-" else None
    except Exception as e:
        print(f"[wiley_tdm {doi}] {e}")
        return None


def _fetch_springer_tdm(doi: str) -> Optional[Tuple[str, str]]:
    """Springer Nature full-text API. Key from dev.springernature.com."""
    key = get_cred("springer") or os.getenv("SPRINGER_API_KEY", "")
    if not key or not doi:
        return None
    try:
        r = requests.get("https://api.springernature.com/openaccess/jats",
                         params={"q": f"doi:{doi}", "api_key": key}, headers=_OA_UA, timeout=45)
        if r.status_code != 200:
            print(f"[springer_tdm {doi}] HTTP {r.status_code}")
            return None
        text = _xml_body_text(r.text)
        return (text, "Springer Nature TDM") if text else None
    except Exception as e:
        print(f"[springer_tdm {doi}] {e}")
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

from pathlib import Path as _Path
from fastapi.responses import Response as _Response

# Acquired PDFs are cached on disk so the UI can display the actual file (not just
# extracted text). Keyed by the caller's paper_id, served from /api/fulltext/pdf.
_PDF_CACHE_DIR = _Path(os.path.expanduser("~")) / ".evidence-engine" / "pdfcache"


def _pdf_cache_key(paper_id: Optional[str], url: Optional[str]) -> str:
    base = (paper_id or url or "").strip()
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", base)[:80].strip("_.")
    return safe or hashlib.md5((url or paper_id or secrets.token_hex(8)).encode()).hexdigest()


def _cache_pdf(paper_id: Optional[str], url: Optional[str], pdf: bytes) -> Optional[str]:
    if not pdf:
        return None
    try:
        _PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = _pdf_cache_key(paper_id, url)
        (_PDF_CACHE_DIR / f"{key}.pdf").write_bytes(pdf)
        return key
    except Exception as e:
        print(f"[cache_pdf] {e}")
        return None


def _found(text: str, source: str, paper_id: Optional[str] = None,
           url: Optional[str] = None, pdf: Optional[bytes] = None) -> dict:
    resp = {"status": "found", "text": text, "source": source}
    key = _cache_pdf(paper_id, url, pdf) if pdf else None
    if key:
        resp["pdf_key"] = key
    return resp


# Extracted full text is cached on disk as well, keyed like the PDF cache. The
# frontend keeps only the key in its saved session and pulls the body back on
# load, so a review with a few hundred acquired papers does not rewrite tens of
# megabytes of article text into IndexedDB and SQLite on every autosave.
_TEXT_CACHE_DIR = _Path(os.path.expanduser("~")) / ".evidence-engine" / "textcache"


def _cache_text(paper_id: Optional[str], url: Optional[str], text: str) -> Optional[str]:
    if not text:
        return None
    try:
        _TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = _pdf_cache_key(paper_id, url)
        (_TEXT_CACHE_DIR / f"{key}.txt").write_text(text, encoding="utf-8")
        return key
    except Exception as e:
        print(f"[cache_text] {e}")
        return None


def _read_cached_text(key: str) -> Optional[str]:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key or "")[:80].strip("_.")
    if not safe:
        return None
    path = _TEXT_CACHE_DIR / f"{safe}.txt"
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else None
    except Exception as e:
        print(f"[read_cached_text] {e}")
        return None


@app.get("/api/fulltext/text/{key}")
def fulltext_text(key: str):
    """Return a cached full-text body by its key (from /api/fulltext/fetch or a PUT here)."""
    text = _read_cached_text(key)
    if text is None:
        raise HTTPException(status_code=404, detail="No cached text for that key")
    return {"key": key, "text": text}


class FullTextPutRequest(BaseModel):
    text: str


@app.put("/api/fulltext/text/{key}")
def fulltext_text_put(key: str, req: FullTextPutRequest):
    """Store a body the client extracted itself (an uploaded PDF, say) so the saved
    session can carry the key instead of the text. Returns the key actually used."""
    stored = _cache_text(key, None, req.text)
    if not stored:
        raise HTTPException(status_code=400, detail="Empty text or cache unavailable")
    return {"key": stored}


@app.post("/api/fulltext/fetch")
def fulltext_fetch(req: FullTextRequest):
    """Run the retrieval ladder, then cache any body found and hand back its key."""
    resp = _fulltext_fetch_impl(req)
    if isinstance(resp, dict) and resp.get("status") == "found" and resp.get("text"):
        key = _cache_text(req.paper_id, req.URL, resp["text"])
        if key:
            resp["text_key"] = key
    return resp


@app.post("/api/fulltext/pdf")
async def fulltext_pdf_upload(request: Request, paper_id: str = "", url: str = ""):
    """Take a PDF the reviewer uploaded in the browser into the cache the
    acquisition step writes to.

    Until this existed, an imported paper reached the server as text only, so
    every server-side reader that works on the FILE (table extraction above all)
    was skipped for exactly the papers a reviewer supplied themselves.
    """
    body = await request.body()
    if not body[:5].startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="That is not a PDF")
    if len(body) > 200 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF too large to cache")
    key = _cache_pdf(paper_id or None, url or None, body)
    if not key:
        raise HTTPException(status_code=500, detail="Could not write to the PDF cache")
    return {"key": key, "bytes": len(body)}


@app.post("/api/fulltext/read")
async def fulltext_read(request: Request, paper_id: str = "", url: str = ""):
    """Read the words out of a PDF the server holds, OCR included.

    The browser parses an imported PDF itself, which works for a born-digital
    file and returns nothing for a scanned one. Rather than hand that back to
    the reviewer as a chore ("run OCR on them"), the file is already here and
    Docling renders each page and reads it. Falls back to the text layer when
    docling is not installed, so the endpoint is always safe to call.
    """
    body = await request.body()
    pdf = body if body[:5].startswith(b"%PDF") else _read_cached_pdf(paper_id or None, url or None)
    if not pdf:
        raise HTTPException(status_code=404, detail="No PDF for that paper on this machine")

    # The text layer first: it is instant and exact where it exists.
    layer = (_extract_text_from_pdf(pdf) or "").strip()
    if len(re.sub(r"[^A-Za-z]", "", layer)) >= 200:
        return {"text": layer, "method": "text-layer", "chars": len(layer)}

    try:
        import docling_tables
    except Exception:
        docling_tables = None
    if docling_tables is not None and docling_tables.available():
        read = docling_tables.extract_text(pdf)
        if (read or "").strip():
            print(f"[fulltext_read] {paper_id or 'upload'} OCR -> {len(read)} chars")
            return {"text": read, "method": "ocr", "chars": len(read)}

    return {"text": layer, "method": "none", "chars": len(layer),
            "detail": "This file has no text layer, and no OCR reader is installed."}


@app.get("/api/fulltext/pdf/{key}")
def fulltext_pdf(key: str):
    """Serve a cached acquired PDF for inline display in the UI."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:80]
    path = _PDF_CACHE_DIR / f"{safe}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF not cached")
    return _Response(
        content=path.read_bytes(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{safe}.pdf"'},
    )


def _fulltext_fetch_impl(req: FullTextRequest) -> dict:
    """Strategy ladder, in order of decreasing reliability:
      1. Europe PMC JATS fullTextXML   (best — structured text)
      2. NCBI E-utilities PMC full text (structured, and not rate-blocked)
      3. Unpaywall / OpenAlex OA PDF
      4. Semantic Scholar open-access PDF
      5. OpenAIRE repository copy
      6. Publisher text-mining API      (only with the user's own key)
      7. arXiv PDF
      8. Direct open-access PDF link    (only when the record URL is itself a PDF)

    Tiers 1-5 and 7-8 use open-access content only. Tier 6 reaches subscription
    content through the interface each publisher provides FOR text mining, using
    a credential the user supplied; with no key it makes no request at all.

    The old PMC-PDF tier is gone: measured over a 411-record run it succeeded 0
    times, because NCBI answers that path with HTTP 200 and an HTML block page.

    Returns `source` describing which tier supplied the text so the UI can
    show e.g. "PMC full text (PMC1234567)" instead of just "Europe PMC".
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

    # Identifier backfill. Everything below is keyed on a DOI or a PMC id, so
    # resolve whichever is missing BEFORE deciding a record is unreachable.
    doi = (req.doi or "").strip() or _extract_doi(req.URL, req.Title) or (lookup_doi or "")
    if not doi and pid:
        doi = _ncbi_ids(pid).get("doi", "")
        if doi:
            print(f"[fulltext_fetch] {debug_label} recovered DOI {doi} from {pid}")
    if not pmcid and doi:
        pmcid = _ncbi_ids(doi).get("pmcid") or None

    # Tier 2 — NCBI E-utilities full text.
    if pmcid:
        text = _fetch_pmc_efetch_text(pmcid)
        if text:
            print(f"[fulltext_fetch] {debug_label} tier2 EFetch ({pmcid}) -> {len(text)} chars")
            return {"status": "found", "text": text, "source": f"PMC full text ({pmcid})"}

    # Tier 3 — open-access PDF via the OpenAlex OA index, including a
    # landing-page follow for records that advertise no direct PDF url.
    oa_status = "unknown"
    if doi or req.Title:
        pdf, src_url, oa_status = _fetch_oa_pdf(doi, req.Title or "")
        if pdf:
            text = _extract_text_from_pdf(pdf)
            if text:
                host = src_url.split("/", 3)[2] if "://" in src_url else src_url
                print(f"[fulltext_fetch] {debug_label} tier3 OA PDF ({host}) -> {len(text)} chars")
                return _found(text, f"Open access ({host})", req.paper_id, req.URL, pdf)

    # Tier 4 — Semantic Scholar's open-access PDF index.
    if doi:
        found = _fetch_s2_oa_pdf(doi)
        if found:
            pdf, src_url = found
            text = _extract_text_from_pdf(pdf)
            if text:
                print(f"[fulltext_fetch] {debug_label} tier4 Semantic Scholar OA -> {len(text)} chars")
                return _found(text, "Open access (Semantic Scholar)", req.paper_id, req.URL, pdf)

    # Tier 5 — OpenAIRE, which aggregates repository copies the other two miss.
    if doi:
        found = _fetch_openaire_pdf(doi)
        if found:
            pdf, src_url = found
            text = _extract_text_from_pdf(pdf)
            if text:
                host = src_url.split("/", 3)[2] if "://" in src_url else src_url
                print(f"[fulltext_fetch] {debug_label} tier5 OpenAIRE ({host}) -> {len(text)} chars")
                return _found(text, f"Repository ({host})", req.paper_id, req.URL, pdf)

    # Tier 6 — the reader's own institutional entitlement, via LibKey and then
    # the publisher's text-mining interface. Both are skipped entirely when the
    # corresponding credential is absent, so neither makes a request by default.
    if doi:
        got_pdf = _fetch_libkey_pdf(doi)
        if got_pdf:
            pdf, link = got_pdf
            text = _extract_text_from_pdf(pdf)
            if text:
                print(f"[fulltext_fetch] {debug_label} tier6 LibKey -> {len(text)} chars")
                return _found(text, "Library entitlement (LibKey)", req.paper_id, req.URL, pdf)
        for fn in (_fetch_elsevier_tdm, _fetch_springer_tdm):
            got = fn(doi)
            if got:
                text, label = got
                print(f"[fulltext_fetch] {debug_label} tier6 {label} -> {len(text)} chars")
                return {"status": "found", "text": text, "source": label}
        got_pdf = _fetch_wiley_tdm(doi)
        if got_pdf:
            pdf, label = got_pdf
            text = _extract_text_from_pdf(pdf)
            if text:
                print(f"[fulltext_fetch] {debug_label} tier6 {label} -> {len(text)} chars")
                return _found(text, label, req.paper_id, req.URL, pdf)

    # Tier 7 — arXiv PDF.
    if (req.Source or "").lower() == "arxiv" or "arxiv.org" in (req.URL or "").lower():
        arxiv_id = _extract_arxiv_id(req.URL or "") or (pid if pid and not pid.isdigit() else None)
        if arxiv_id:
            pdf = _fetch_arxiv_pdf(arxiv_id)
            if pdf:
                text = _extract_text_from_pdf(pdf)
                if text:
                    print(f"[fulltext_fetch] {debug_label} tier7 arXiv PDF ({arxiv_id}) -> {len(text)} chars")
                    return _found(text, f"arXiv PDF ({arxiv_id})", req.paper_id, req.URL, pdf)

    # Tier 8 — direct PDF link, only when the record's own URL is itself a PDF.
    if req.URL and req.URL.startswith("http") and req.URL.lower().split("?", 1)[0].rstrip("/").endswith(".pdf"):
        host = req.URL.split("/", 3)[2].lower() if "://" in req.URL else ""
        try:
            ua = "EvidenceEngine/1.0"
            r = requests.get(req.URL, timeout=25, allow_redirects=True, headers={"User-Agent": ua})
            ct = (r.headers.get("content-type") or "").lower()
            if r.status_code == 200 and ("pdf" in ct or r.content[:5] == b"%PDF-"):
                text = _extract_text_from_pdf(r.content)
                if text:
                    print(f"[fulltext_fetch] {debug_label} tier8 direct OA PDF ({host}) -> {len(text)} chars")
                    return _found(text, f"Open-access PDF ({host})", req.paper_id, req.URL, r.content)
        except Exception as e:
            print(f"[fulltext_fetch pdf] {e}")

    has_tdm = bool(get_cred("scopus") or get_cred("springer") or get_cred("wiley"))
    print(f"[fulltext_fetch] {debug_label} -> no full text "
          f"(pid={pid or '-'}, doi={doi or '-'}, pmcid={pmcid or '-'}, oa={oa_status}, "
          f"tdm_keys={'yes' if has_tdm else 'no'}, source={req.Source}, url_host="
          f"{req.URL.split('/', 3)[2] if req.URL and '://' in req.URL else '-'})")

    # A specific reason plus the exact links a human needs, so the reviewer can
    # finish the job by hand instead of being told only that it did not work.
    if not doi:
        code, reason = "no_doi", "No DOI found for this record, in the source or at NCBI."
    elif pmcid:
        # A PMC copy exists and E-utilities answered with metadata only, which
        # is how PMC represents an article whose publisher withholds
        # text-mining rights. Readable in a browser, not retrievable here.
        code, reason = ("pmc_restricted",
                        f"{pmcid} is in PMC but its publisher does not permit text mining. "
                        "Open it in a browser and upload the PDF.")
    elif oa_status == "closed":
        code, reason = ("paywalled",
                        "Paywalled. A publisher text-mining key, your library, or "
                        "interlibrary loan can reach it."
                        if not has_tdm else
                        "Paywalled, and your text-mining keys did not cover this publisher.")
    elif oa_status in ("pdf", "landing"):
        code, reason = "oa_blocked", "Listed as open access but the publisher blocked the download."
    else:
        code, reason = "unresolved", "Could not resolve this record in any open-access index."

    links: Dict[str, str] = {}
    if doi:
        # A one-click route to the copy this library is entitled to, for the
        # records we could not fetch ourselves. This is the whole value of
        # LibKey off-campus: we cannot download it, the reader can.
        for name, url in _libkey_lookup(doi).items():
            links[f"libkey_{name}" if name != "pdf" else "libkey"] = url
        links["doi"] = f"https://doi.org/{doi}"
        links["scholar"] = ("https://scholar.google.com/scholar?q="
                            + quote(f'"{doi}"', safe=""))
    if pid and pid.isdigit():
        links["pubmed"] = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
    if req.Title:
        links["search"] = ("https://scholar.google.com/scholar?q="
                           + quote(req.Title[:250], safe=""))
    if req.URL:
        links["record"] = req.URL

    return {
        "status": "missing",
        "reason": reason,
        "reason_code": code,
        "oa_status": oa_status,
        "doi": doi or "",
        "pmid": pid if pid and pid.isdigit() else "",
        "links": links,
    }


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


class ExtractFieldsRequest(BaseModel):
    text: str
    fields: List[Dict[str, Any]]           # [{id,label,type,options,description}]
    title: Optional[str] = None
    tables: Optional[str] = None           # extracted table text (numbers live here)
    model: Optional[str] = None


@app.post("/api/extract/fields")
def extract_fields(req: ExtractFieldsRequest):
    """AI-assisted data extraction into a user-defined form. For each field the
    model quotes the exact supporting snippet (locate-then-extract), so a reviewer
    can verify at a glance. Numbers are validated and always flagged for review
    (autonomous numeric exact-match is low; accuracy comes from human verification).

    Returns:
      values: {field_id: value}                      -- back-compat draft values
      fields: [{id,label,type,value,source_quote,confidence,needs_review}]
    Every value is a first draft; the reviewer confirms or corrects it."""
    import extraction_forms

    text = (req.text or "").strip()
    fields = [f for f in (req.fields or []) if f.get("id")]
    if not text or not fields:
        return {"values": {}, "fields": []}

    # Map the form fields to the extractor's schema (name = field id so the
    # returned values key by id, as the existing frontend expects).
    ef_fields = []
    for f in fields:
        ftype = (f.get("type") or "text").lower()
        ftype = "number" if ftype in ("number", "numeric") else ("categorical" if ftype == "category" else ftype)
        ef_fields.append({
            "name": f["id"],
            "type": ftype,
            "description": (f.get("label") or f["id"]) + (
                ". " + f["description"] if f.get("description") else ""),
            "options": f.get("options") or None,
        })

    model_name = resolve_for_thinking(req.model)
    if _local_model_missing(model_name):
        # Said once, plainly, instead of discovered after a run of empty rows.
        return {"values": {}, "fields": [],
                "error": f"The extraction model {model_name} is not installed on this machine. "
                         f"Install it with:  ollama pull {model_name}"}

    # Ask for the fields in groups. A whole imported workbook in one request is
    # not a hard question, it is a long answer: the reply is cut off part-way
    # and nothing at all can be parsed from it, so every field comes back empty.
    per_call = max(1, Config.EXTRACTION_FIELDS_PER_CALL)
    groups = [ef_fields[i:i + per_call] for i in range(0, len(ef_fields), per_call)]
    raw_fields: List[Dict[str, Any]] = []
    failures: List[str] = []
    for group in groups:
        # The whole text: extract_form sizes it to the context it opens. Slicing
        # here as well is how funding, conflicts of interest and data
        # availability came back blank on every paper longer than 30,000
        # characters, which is most of them.
        got = extraction_forms.extract_form(
            text=text, tables=(req.tables or ""), fields=group,
            model_name=model_name, ai_service=AIService,
            num_ctx=Config.EXTRACTION_NUM_CTX)
        raw_fields.extend(got.get("fields") or [])
        if got.get("error"):
            failures.append(got["error"])
    if len(groups) > 1:
        print(f"[extract_fields] {len(ef_fields)} fields in {len(groups)} groups, "
              f"{len(failures)} failed")

    # Ask again, one field at a time, for any field that came back holding
    # another field's answer. Asked twelve at a time a model that cannot find a
    # field sometimes copies a neighbour instead of returning nothing, and the
    # copy is indistinguishable from a real answer: it has a value and a quote,
    # and the quote is genuinely in the paper. Alone, there is nothing to copy.
    # Two ways a field comes back holding something that is not its answer, and
    # neither is visible in the value: it copied a neighbour's, or it cites a
    # sentence that does not contain it. Both get asked again on their own.
    # Three ways a field comes back holding something that is not its answer,
    # none of them visible in the value: it copied a neighbour's, it cites a
    # sentence that does not contain it, or it is one sentence of a declaration
    # the paper makes at greater length. All get asked again on their own.
    doubtful: Dict[str, str] = {}
    for name in extraction_forms.colliding_fields(raw_fields):
        doubtful.setdefault(name, "copied")
    for name in extraction_forms.unsupported_fields(raw_fields):
        doubtful.setdefault(name, "unsupported")
    for name in extraction_forms.partial_declarations(raw_fields, text):
        doubtful.setdefault(name, "partial")
    if doubtful:
        by_name = {f["name"]: f for f in ef_fields}
        print(f"[extract_fields] re-asking {len(doubtful)} doubtful field(s): "
              + ", ".join(f"{n} ({why})" for n, why in doubtful.items()))
        for name, why in doubtful.items():
            spec = by_name.get(name)
            if not spec:
                continue
            again = extraction_forms.extract_form(
                text=text, tables=(req.tables or ""), fields=[spec],
                model_name=model_name, ai_service=AIService,
                num_ctx=Config.EXTRACTION_NUM_CTX)
            got = (again.get("fields") or [None])[0]
            # Only on a clean answer: a failed re-ask must not turn a value the
            # reviewer could at least see and correct into an empty cell. And
            # only when the new answer is better supported than the old one,
            # because a second guess is not automatically a better one.
            if got and not again.get("error"):
                # A second guess is not automatically a better one, and
                # churning the value would lose whatever the reviewer had
                # already read, so the replacement has to be an improvement on
                # the terms the field was re-asked for.
                if why == "partial":
                    # A complete declaration runs over several sentences, so it
                    # can never match one quoted sentence. What is checked is
                    # that it is longer and that its words are in the paper.
                    old_len = len(str(
                        next((f.get("value") for f in raw_fields
                              if f["name"] == name), "") or ""))
                    better = (len(str(got.get("value") or "")) > old_len
                              and extraction_forms.grounded_in_text(got.get("value"), text))
                else:
                    better = not extraction_forms.unsupported_fields([got])
                if better:
                    raw_fields = [got if f["name"] == name else f for f in raw_fields]
    # A group that failed contributes no fields, so a partial answer is still an
    # answer; only say the whole thing failed when all of it did.
    result = {
        "fields": raw_fields,
        "error": ("" if not failures
                  else failures[0] if len(failures) == len(groups)
                  else f"{len(failures)} of {len(groups)} field groups failed: {failures[0]}"),
    }

    label_by_id = {f["id"]: (f.get("label") or f["id"]) for f in fields}
    out_fields, values = [], {}
    for ef in result.get("fields", []):
        fid = ef["name"]
        val = ef["value"]
        values[fid] = "" if val is None else val
        out_fields.append({
            "id": fid, "label": label_by_id.get(fid, fid), "type": ef["type"],
            "value": values[fid], "source_quote": ef["source_quote"],
            "confidence": ef["confidence"], "needs_review": ef["needs_review"],
            # "reported" / "not_reported" / "unread". An empty cell meant both
            # "this paper does not report it" and "nothing ever read this
            # field", and a review has to be able to write the first one down.
            "status": ef.get("status", "unread"),
            # Why it wants a human, when there is a specific reason: the
            # provenance dialog shows this instead of making the reviewer work
            # out what is wrong with a cell the app has flagged.
            "review_reason": ef.get("review_reason", ""),
        })
    # An empty row and a failed call look identical to the reviewer, so the
    # failure travels with the result instead of only reaching the server log.
    return {"values": values, "fields": out_fields, "model": model_name,
            "error": result.get("error") or ""}


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


def _table_sig(rows: List[List[str]]) -> str:
    """Content signature of a table (its first ~3 rows), for cross-source dedup."""
    flat = " ".join(str(c).strip().lower() for r in (rows or [])[:3] for c in r if str(c).strip())
    return re.sub(r"[^a-z0-9 ]+", "", flat).strip()[:180]


def _rows_similar(a: List[List[str]], b: List[List[str]]) -> bool:
    """True if two row-grids are the same underlying table (so the HTML, PDF, and
    image passes don't each add a copy of the same table)."""
    sa, sb = _table_sig(a), _table_sig(b)
    if not sa or not sb:
        return False
    if sa == sb:
        return True
    short, long = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    return len(short) > 24 and short in long


def _dedup_tables(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Union of tables from every source, keeping the first (highest-fidelity)
    version of any duplicate. Callers pass sources in fidelity order:
    HTML/JATS XML, then image-based recognition, then pdfplumber text."""
    out: List[Dict[str, Any]] = []
    for t in tables:
        rows = t.get("data") or []
        if not any(_rows_similar(rows, o.get("data") or []) for o in out):
            out.append(t)
    # renumber titles that were auto-generated ("Table N") so they stay sequential
    n = 0
    for t in out:
        if re.fullmatch(r"Table \d+", str(t.get("title", ""))):
            n += 1; t["title"] = f"Table {n}"
    return out


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
                    # JATS is structured markup, so there is no page to cite; the
                    # table label is the citable unit instead.
                    "page": None,
                    "tier": "jats",
                    "deterministic": True,
                })
        return out
    except Exception as e:
        print(f"[extract_epmc_tables] parse error for {paper_id}: {e}")
        return []


def _same_title(a: str, b: str) -> bool:
    """Titles match if their letters do, allowing for the punctuation and
    subtitle handling that differs between a PDF and a database record."""
    x = re.sub(r"[^a-z0-9]+", "", (a or "").lower())
    y = re.sub(r"[^a-z0-9]+", "", (b or "").lower())
    if not x or not y:
        return False
    return x == y or (len(x) > 40 and len(y) > 40 and (x.startswith(y[:40]) or y.startswith(x[:40])))


def _is_db_id(value: str) -> bool:
    """Whether this identifier means anything to a literature database.

    An imported paper carries whatever id the browser gave it, which no
    database has heard of. Asking Europe PMC about it returns nothing, and that
    nothing used to be read as "this paper has no tables"."""
    return bool(re.fullmatch(r"(?:PMC\d+|\d{5,9})", (value or "").strip(), re.IGNORECASE))


def _resolve_epmc_id(title: str) -> str:
    """A Europe PMC id for a paper known only by its title.

    An uploaded PDF carries no identifier, so the publisher's own structured
    full text is out of reach until one is found. The match has to be checked:
    a title search happily returns a review that merely cites the paper, and
    extracting that review's tables would be worse than finding none.
    """
    t = (title or "").strip()
    if len(t) < 12:
        return ""

    def _search() -> str:
        try:
            r = limited_get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={"query": f'TITLE:"{t[:300]}"', "resultType": "lite",
                        "format": "json", "pageSize": 3},
                timeout=8,
            )
            if r is None or not r.ok:
                return ""
            for hit in (r.json().get("resultList") or {}).get("result") or []:
                if _same_title(hit.get("title", ""), t):
                    return hit.get("pmcid") or hit.get("pmid") or ""
        except Exception as e:
            print(f"[resolve_epmc_id] {e}")
        return ""

    return cached("epmc_id", t, _search) or ""


def _read_cached_pdf(paper_id: Optional[str], url: Optional[str]) -> Optional[bytes]:
    """Read the open-access PDF cached during full-text acquisition, if present."""
    try:
        path = _PDF_CACHE_DIR / f"{_pdf_cache_key(paper_id, url)}.pdf"
        if path.exists():
            return path.read_bytes()
    except Exception as e:
        print(f"[read_cached_pdf] {e}")
    return None


def _table_is_real(rows: List[List[str]]) -> bool:
    """Filter out prose mis-detected as a table: require >=2 rows, >=2 columns,
    and at least half the cells populated."""
    if len(rows) < 2:
        return False
    ncol = max((len(r) for r in rows), default=0)
    if ncol < 2:
        return False
    cells = sum(len(r) for r in rows)
    filled = sum(1 for r in rows for c in r if c)
    return cells > 0 and filled >= 0.5 * cells


# Two patterns, because the two readers want different things from a caption.
# For deciding whether a page HAS a table, only a caption that starts a line
# counts: a page that merely cites "Supplementary Table 1" in a sentence is a
# page of prose, and treating it as captioned lets the layout back in. For
# deciding where to point a model, any mention will do: a two-column PDF's text
# layer interleaves the columns, so the caption that introduces Table 1 often
# lands in the middle of a sentence from the other column, and a window around
# an in-text reference costs one call and sometimes lands on the table.
_TABLE_CAPTION_LINE = re.compile(r"^[ \t]*(?:Supplementary\s+)?Table\s+\d+\b", re.M | re.I)
_TABLE_CAPTION = re.compile(r"\b(?:Supplementary\s+)?Table\s+\d+\b", re.I)


_PROSE_CELL = re.compile(r"[a-z],\s|\.\s+[A-Z]|\b(?:the|and|with|that|were|was|which)\b", re.I)


def _looks_like_page_layout(rows: List[List[str]], width_frac: float) -> bool:
    """True when the "table" is really the page.

    A two-column article read by the text-alignment strategy yields one grid per
    page: the gutter looks like a column boundary and every line looks like a
    row. Measured across journal PDFs, a genuine table sits inside one column
    (0.2-0.45 of the page width) while this artifact spans 0.85 and up, and its
    cells are clauses rather than values. Both tests have to fail it, so a wide
    table of short values in a single-column paper still gets through.
    """
    if width_frac < 0.7:
        return False
    cells = [c for r in rows for c in r if c]
    if not cells:
        return True
    wordy = sum(1 for c in cells
                if len(c) > 45 or len(c.split()) > 7 or _PROSE_CELL.search(c))
    return wordy / len(cells) > 0.25


def _extract_pdf_tables(pdf_bytes: bytes, extraction_type: str) -> List[Dict[str, Any]]:
    """Deterministically extract tables from a PDF with pdfplumber — no LLM, so
    the same PDF always yields the same tables (reproducible for a review).

    Tries ruling-line detection first (precise, for bordered tables); on pages
    with no bordered tables it falls back to text-alignment detection so
    borderless journal tables are still captured. Both strategies are
    deterministic; a density filter drops prose mis-read as a table."""
    if not pdf_bytes:
        return []
    text_settings = {
        "vertical_strategy": "text", "horizontal_strategy": "text",
        "snap_tolerance": 4, "join_tolerance": 4, "min_words_vertical": 2,
    }
    try:
        import io
        import pdfplumber
        out: List[Dict[str, Any]] = []
        idx = 0
        # Optional higher-fidelity fallback: UniTable image-based recognition for
        # pages where pdfplumber's text-layer heuristics find nothing (scanned
        # pages, complex spanning tables). Guarded + OFF by default, so this line
        # is a no-op unless EE_USE_UNITABLE=1 and the weights are present.
        try:
            import table_recognition as _tabrec
            _uni_on = _tabrec.available()
        except Exception:
            _tabrec, _uni_on = None, False

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                # Each entry carries the strategy that produced it, because the UI
                # states which tier a number came from and the two differ in fidelity.
                page_tables: List[Tuple[List[List[str]], str]] = []
                # Ruling lines first. The text-alignment fallback runs only where
                # the lines found nothing, and even then its results have to get
                # past _looks_like_page_layout: on a two-column page that strategy
                # always finds "a table", and the table it finds is the page.
                # A caption is what makes a text-alignment guess believable. Ruled
                # tables speak for themselves; without ruling lines, a page of a
                # journal article that never says "Table 1" does not contain one,
                # and what the strategy found is the figure labels or the layout.
                try:
                    captioned = bool(_TABLE_CAPTION_LINE.search(page.extract_text() or ""))
                except Exception:
                    captioned = False
                for settings in (None, text_settings):
                    try:
                        found = (page.find_tables() if settings is None
                                 else page.find_tables(text_settings)) or []
                    except Exception:
                        found = []
                    for t in found:
                        rows = [
                            [(c or "").strip() for c in row]
                            for row in (t.extract() or []) if any((c or "").strip() for c in row)
                        ]
                        # Three rows, not two: a bordered 2x2 in a journal PDF is
                        # a box around a figure panel, never a table.
                        if len(rows) < 3 or not _table_is_real(rows):
                            continue
                        if settings is not None:
                            if not captioned:
                                continue
                            span = (t.bbox[2] - t.bbox[0]) / float(page.width or 1)
                            if _looks_like_page_layout(rows, span):
                                continue
                        page_tables.append((rows, "pdfplumber"))
                    if page_tables:
                        break
                # Image-based recognition runs on EVERY page (when enabled), and its
                # result is added only if it isn't already one of pdfplumber's tables
                # for this page. This catches image-only / scanned / complex-spanning
                # tables that the text layer misses, without dropping the text ones.
                if _uni_on:
                    try:
                        page_img = page.to_image(resolution=150).original
                        irows = _tabrec.recognize_table_image(page_img)
                        if irows and _table_is_real(irows) and not any(_rows_similar(irows, r) for r, _ in page_tables):
                            page_tables.append((irows, "unitable"))
                    except Exception as _e:
                        print(f"[extract_pdf_tables] unitable pass: {_e}")
                for rows, tier in page_tables:
                    idx += 1
                    out.append({
                        "title": f"Table {idx}",
                        "type": _classify_table(rows, extraction_type or ""),
                        "data": rows,
                        "caption": "",
                        # Provenance. page_number is 1-based, which is what a
                        # reader sees on the page and what "p. 7" has to mean.
                        "page": page.page_number,
                        "tier": tier,
                        "deterministic": True,
                    })
        return out
    except Exception as e:
        print(f"[extract_pdf_tables] {e}")
        return []


class DocumentsAskRequest(BaseModel):
    question: str
    documents: List[Dict[str, Any]]     # [{id, title, text}] — the most-relevant slice of the library
    history: Optional[List[Dict[str, str]]] = None   # [{role, content}] recent chat turns, for follow-ups
    total_documents: Optional[int] = None            # size of the full library (documents may be a subset)
    model: Optional[str] = None


@app.post("/api/documents/ask")
def documents_ask(req: DocumentsAskRequest):
    """Answer a question across MULTIPLE documents (retrieved, uploaded, or cited
    in explanations). Returns a synthesised answer with [n] citations mapping to
    the supplied documents, plus the citation list so the UI can render sources."""
    from langchain_core.messages import HumanMessage

    question = (req.question or "").strip()
    docs = [d for d in (req.documents or []) if (d.get("text") or "").strip()]
    if not question or not docs:
        return {"answer": "", "documents": []}

    # Fit the documents into a character budget, spread evenly across them.
    docs = docs[:40]
    budget = 26000
    per = max(600, budget // max(1, len(docs)))
    blocks, cited = [], []
    for i, d in enumerate(docs, 1):
        title = (d.get("title") or "Untitled").strip()
        blocks.append(f"[{i}] {title}\n{(d.get('text') or '')[:per]}")
        cited.append({"n": i, "id": d.get("id", ""), "title": title})
    context = "\n\n".join(blocks)

    multi = len(docs) > 1
    total = req.total_documents if (req.total_documents and req.total_documents >= len(docs)) else len(docs)
    convo = ""
    for turn in (req.history or [])[-6:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        if content:
            convo += f"{role}: {content[:600]}\n"
    subset_note = (
        f" The {len(docs)} most relevant to this message are shown below."
        if total > len(docs) else ""
    )

    prompt = f"""You are the assistant inside a systematic-review platform, talking with a reviewer about the body of sources THEY have already collected. Their library holds {total} collected source{"s" if total != 1 else ""} (retrieved studies and uploaded files).{subset_note} These sources ARE available to you as the DOCUMENTS below — never tell the user you have no access to their documents or library.

Answer the user's latest MESSAGE:
- If it is about the sources (their content, methods, findings, which ones meet some criterion, a summary or comparison), answer using the DOCUMENTS as your source of truth and cite claims with the document number in square brackets, like [1] or [2][3]. If the DOCUMENTS genuinely do not cover it, say what is and isn't covered rather than refusing.
- If it is a general, how-to, or capability question, answer it directly and helpfully; you need not cite documents for that.

Write a genuinely useful, substantive answer in clean Markdown. Favour flowing prose over lists:
- Open with a short overview paragraph (2-4 sentences) that directly answers the message.
- Organise any detail under a few **bold section labels** written on their own line (for example **Methods**, **Findings**, **Limitations**), each followed by a short paragraph. Do NOT make the section label itself a bullet.
- Use bullet points ("- ") sparingly, and only for a genuine list of 3 or more parallel items. Never put a single fact on its own bullet.
- Prefer specifics over generalities: weave the actual figures, effect sizes, sample sizes, comparisons, and named methods into the sentences.
- Keep it tight and skimmable, not padded. Do not add a "References" section (the interface lists sources separately).
{"- When several documents are relevant, synthesise across them and note where they agree or differ, citing each." if multi else ""}

DOCUMENTS:
{context}
{("RECENT CONVERSATION:" + chr(10) + convo if convo else "")}
MESSAGE: {question}

Answer (Markdown, grounded in the documents where relevant, with [n] citations):"""

    model = AIService.get_model(resolve_for_thinking(req.model))
    answer = ""
    if model:
        try:
            r = model.invoke([HumanMessage(content=prompt)])
            answer = (getattr(r, "content", "") or "").strip()
        except Exception as e:
            print(f"[documents_ask] LLM call failed: {e}")
    if not answer:
        answer = "The model is unavailable or returned no answer. Please try again."
    return {"answer": answer, "documents": cited}


class RouteIntentRequest(BaseModel):
    message: str
    has_documents: Optional[bool] = None   # are documents (uploaded/retrieved) already in play?
    model: Optional[str] = None


@app.post("/api/route/intent")
def route_intent(req: RouteIntentRequest):
    """Route a home-chat message to the right tool so the single chat does the
    lightest thing that answers the user, instead of always running the full
    retrieval pipeline. Returns one of:
      - "documents": answer / summarise / compare / extract from documents ALREADY
                     in play (uploaded or retrieved). Handled by the doc-QA tool.
      - "search":    build, refine, or run a literature search (the heavy pipeline).
      - "chat":      general conversation, capability or how-to questions, greetings
                     — no retrieval and no documents needed.
    """
    from langchain_core.messages import HumanMessage

    msg = (req.message or "").strip()
    if not msg:
        return {"intent": "search"}
    has_docs = bool(req.has_documents)
    context_line = (
        "There ARE documents already available to answer from."
        if has_docs else
        "There are NO documents loaded yet."
    )
    # Bias: with no research topic and no docs, a bare command like "summarize" is
    # still a document action (it just has nothing to act on yet) — NOT a search.
    prompt = f"""You route each message for a systematic-review assistant to exactly one tool. Pick the tool that does the LEAST work needed to satisfy the user; only choose "search" when they genuinely want to find or refine literature.

Context: {context_line}

Tools:
- documents = the user wants to act on documents that already exist (or that they clearly expect to already exist): summarize, describe, explain, compare, list, extract data, or answer a question about "this/these/the paper(s)/the uploaded file(s)/the results". Examples: "summarize the uploaded article", "make the summary more descriptive", "what sample sizes did they use?", "which of these were RCTs?", "compare the interventions", "extract the outcomes", "explain the methods".
- search = the user wants to find NEW studies or build/refine a literature search or research question. This usually names a research TOPIC, population, intervention, or asks for a query/PICO/strategy. Examples: "find studies on statins and stroke", "I want to review the effect of vitamin D on falls in older adults", "generate a search strategy", "refine the query to include children", "add RCTs from the last 5 years".
- chat = general conversation, greetings, or questions about the assistant/how to use it that need neither documents nor a search. Examples: "hi", "what can you do?", "how does screening work here?", "thanks".

Rules:
- A command that summarizes/compares/extracts/explains is "documents" even if no documents are loaded yet.
- A message that states a research topic or clinical question to investigate is "search".
- When unsure between documents and search: choose "documents" if documents are available, otherwise "search".

Message: "{msg}"

Reply with ONLY one word: documents, search, or chat."""
    model = AIService.get_model(resolve_for_thinking(req.model))
    intent = "search"
    if model:
        try:
            r = model.invoke([HumanMessage(content=prompt)])
            out = (getattr(r, "content", "") or "").strip().lower()
            if "document" in out:
                intent = "documents"
            elif "chat" in out:
                intent = "chat"
            else:
                intent = "search"
        except Exception as e:
            print(f"[route_intent] {e}")
    return {"intent": intent}


class AssistantChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None   # [{role, content}] recent turns
    model: Optional[str] = None


@app.post("/api/assistant/chat")
def assistant_chat(req: AssistantChatRequest):
    """A lightweight general-purpose reply for messages that need neither a
    literature search nor the document corpus (greetings, capability / how-to
    questions). Keeps the single chat conversational without spinning up retrieval."""
    from langchain_core.messages import HumanMessage

    msg = (req.message or "").strip()
    if not msg:
        return {"answer": ""}
    convo = ""
    for turn in (req.history or [])[-6:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        convo += f"{role}: {turn.get('content', '')}\n"
    prompt = f"""You are the assistant inside a systematic-review platform. You help users search the literature, screen studies, extract data, appraise risk of bias, and write up reviews. Answer the user's message directly and concisely in clean Markdown.

If they ask what you can do or how to use the platform, explain briefly and, when useful, tell them they can attach their own PDFs/Word/Excel files or just describe a research goal to start a search. Do not invent findings or cite documents you have not been given.

{convo}User: {msg}

Assistant:"""
    model = AIService.get_model(resolve_for_thinking(req.model))
    answer = ""
    if model:
        try:
            r = model.invoke([HumanMessage(content=prompt)])
            answer = (getattr(r, "content", "") or "").strip()
        except Exception as e:
            print(f"[assistant_chat] {e}")
    if not answer:
        answer = "I can help you build a literature search, screen studies, extract data, and write up your review. Describe a research goal, or attach your own PDFs, Word, or Excel files to work from."
    return {"answer": answer}


# ---------------------------------------------------------------------------
# Home-chat agent: a single tool-calling loop over the user's collected library.
# The model decides when to search the library and when to read a source's full
# text, then answers grounded with [n] citations. Chat-only: its tools never
# reach into the other review stages. (Stage 2; the deep-research orchestrator
# will reuse these same tools across parallel subagents.)
# ---------------------------------------------------------------------------

def _supports_tools(model_name: str) -> bool:
    """Models we trust to call tools. Cloud chat models do; among local (Ollama)
    tags the llama3.1 / qwen2.5 / mistral-nemo families do. The default LEADS
    screening tag and small instruct-only tags do not."""
    m = (model_name or "").lower()
    if m.startswith(("claude", "gpt-4", "o1", "o3")) or "gemini" in m:
        return True
    return any(k in m for k in ("llama3.1", "llama-3.1", "qwen2.5", "qwen2", "mistral-nemo", "mistral-small", "command-r", "firefunction"))


class AssistantAgentRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None          # [{role, content}] recent turns
    library: List[Dict[str, Any]] = []                       # [{id, title, snippet, url, source, has_full_text}]
    full_texts: Optional[Dict[str, str]] = None             # id -> text the client already holds (e.g. uploads)
    model: Optional[str] = None


def _rank_library_for_agent(query: str, library: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    terms = [w for w in re.split(r"[^a-z0-9]+", (query or "").lower()) if len(w) >= 3]
    if not terms:
        return library
    scored = []
    for i, p in enumerate(library):
        title = (p.get("title") or "").lower()
        snip = (p.get("snippet") or "").lower()
        score = sum((3 if t in title else 0) + (1 if t in snip else 0) for t in terms)
        scored.append((-score, i, p))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [p for _, _, p in scored]


@app.post("/api/assistant/agent")
def assistant_agent(req: AssistantAgentRequest):
    """Answer a Home-chat message with a tool-calling loop over the user's library.
    The model calls search_library / read_full_text as needed, then answers grounded
    with [n] citations. Assumes a tool-capable model (the client picks one)."""
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

    message = (req.message or "").strip()
    if not message:
        return {"answer": "", "documents": [], "trace": []}

    library = req.library or []
    full_texts = req.full_texts or {}
    total = len(library)

    model = AIService.get_model(resolve_for_thinking(req.model))
    if model is None:
        return {"answer": "The model is unavailable. Please try again.", "documents": [], "trace": []}

    tool_schemas = [
        {"type": "function", "function": {
            "name": "search_library",
            "description": "Search the user's collected library of sources by topic/keywords. Returns the most relevant sources as id, title, a snippet, and whether full text is available. Call this first to find which sources bear on the question.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Topic or keywords to search the collected sources for."}}, "required": ["query"]},
        }},
        {"type": "function", "function": {
            "name": "read_full_text",
            "description": "Read the full text of one collected source, by the id from search_library. Use when a snippet is not enough. Returns the source text and the [n] number to cite it as.",
            "parameters": {"type": "object", "properties": {"paper_id": {"type": "string", "description": "id of the source to read."}}, "required": ["paper_id"]},
        }},
    ]
    try:
        bound = model.bind_tools(tool_schemas)
    except Exception as e:
        print(f"[assistant_agent] bind_tools failed: {e}")
        return {"answer": "", "documents": [], "trace": [], "unsupported": True}

    numbered: Dict[str, Dict[str, Any]] = {}   # id -> {n, id, title}
    order: List[str] = []                        # order[n-1] = id of the n-th source shown
    trace: List[Dict[str, Any]] = []

    def num_for(p: Dict[str, Any]) -> int:
        """Assign each source a stable citation number the first time it's shown."""
        pid = str(p.get("id"))
        if pid not in numbered:
            order.append(pid)
            numbered[pid] = {"n": len(order), "id": pid, "title": p.get("title", "Untitled")}
        return numbered[pid]["n"]

    def run_tool(name: str, args: Dict[str, Any]) -> str:
        if name == "search_library":
            q = (args.get("query") or "").strip()
            ranked = _rank_library_for_agent(q, library)[:8]
            trace.append({"tool": "search_library", "query": q, "n": len(ranked)})
            if not ranked:
                return f"The library has {len(library)} source(s), but none match '{q}'."
            out = [f"[{num_for(p)}] id={p.get('id')} | {p.get('title', 'Untitled')} | full_text={'yes' if p.get('has_full_text') else 'no'}\n  {(p.get('snippet') or '')[:400]}" for p in ranked]
            return f"The library has {len(library)} source(s). Most relevant to '{q}' (cite by the bracketed number):\n" + "\n".join(out)
        if name == "read_full_text":
            pid = (args.get("paper_id") or "").strip()
            p = next((x for x in library if str(x.get("id")) == pid), None)
            if not p:
                return f"No source with id {pid}."
            text = (full_texts.get(pid) or "").strip()
            if not text and (p.get("url") or "").strip():
                try:
                    fr = fulltext_fetch(FullTextRequest(Title=p.get("title", ""), URL=p.get("url", ""), Source=p.get("source", ""), paper_id=pid))
                    if isinstance(fr, dict) and fr.get("status") == "found":
                        text = (fr.get("text") or "").strip()
                except Exception as e:
                    print(f"[assistant_agent] read_full_text fetch failed: {e}")
            if not text:
                text = (p.get("snippet") or "").strip()
            n = num_for(p)
            trace.append({"tool": "read_full_text", "id": pid, "title": p.get("title", ""), "chars": len(text)})
            if not text:
                return f"No readable full text is available for source [{n}] (it may be paywalled or an unfetchable upload)."
            return f"Source [{n}] — {p.get('title', '')}\n{text[:8000]}"
        return f"Unknown tool: {name}"

    system = f"""You are the assistant inside a systematic-review platform, working with a reviewer over the library of {total} source{'s' if total != 1 else ''} THEY have collected. You have TOOLS to search that library and read a source's full text, so you DO have access to their documents. Never tell the user you lack access.

To answer a question about the sources: call search_library to find the relevant ones, then read_full_text on the ones you need, then answer grounded in what you read. Cite claims with the source's number in square brackets like [1] or [2][3] (search_library and read_full_text show each source's number in brackets). If, after searching and reading, the sources genuinely do not cover something, say what is and isn't covered rather than refusing.

If you cannot find or read a specific source the user names, say so plainly, tell them what the library does contain, and offer to search for it or invite them to attach the PDF. NEVER invent an answer or describe what such a document "typically", "usually", or "generally" contains from your own knowledge — only report what you actually read in the sources. For a genuine general or how-to question about using the platform you may answer directly without tools.

Write a substantive answer in clean Markdown: a short overview paragraph first, then a few **bold section labels** each followed by a short paragraph. Use bullets sparingly. Weave in specifics (figures, sample sizes, methods). Do not add a References section (sources are listed separately)."""

    messages: List[Any] = [SystemMessage(content=system)]
    for turn in (req.history or [])[-6:]:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        messages.append(HumanMessage(content=content) if turn.get("role") == "user" else AIMessage(content=content))
    messages.append(HumanMessage(content=message))

    answer = ""
    for _ in range(6):
        try:
            ai = bound.invoke(messages)
        except Exception as e:
            print(f"[assistant_agent] invoke failed: {e}")
            answer = "Sorry, that failed while reasoning over your library. Please try again."
            break
        messages.append(ai)
        tcs = getattr(ai, "tool_calls", None) or []
        if not tcs:
            answer = (getattr(ai, "content", "") or "").strip()
            break
        for tc in tcs:
            try:
                result = run_tool(tc.get("name", ""), tc.get("args", {}) or {})
            except Exception as e:
                result = f"Tool error: {e}"
            messages.append(ToolMessage(content=result, tool_call_id=tc.get("id", "") or ""))
    if not answer:
        # Out of tool steps: force a final grounded answer with no further tools.
        try:
            messages.append(HumanMessage(content="Now give your best final answer using what you gathered, with [n] citations. Do not call any more tools."))
            ai = bound.invoke(messages)
            answer = (getattr(ai, "content", "") or "").strip()
        except Exception as e:
            print(f"[assistant_agent] final invoke failed: {e}")
    if not answer:
        answer = "I searched your library but couldn't compose an answer. Please try rephrasing."

    # Sources list = exactly what the final answer cites with [n].
    refs = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer) if 1 <= int(x) <= len(order)})
    documents = [numbered[order[n - 1]] for n in refs]
    return {"answer": answer, "documents": documents, "trace": trace}


def _table_windows(text: str, size: int = 4000, cap: int = 8) -> List[str]:
    """The stretches of a paper where a table actually is.

    A caption is the anchor the typesetter leaves behind and the rows follow it,
    so these windows are where the numbers live. Sending them instead of the
    first N characters is the difference between a model seeing Table 3 of a
    40,000-character paper and never seeing a table at all.
    """
    if not text:
        return []
    spans: List[List[int]] = []
    for m in _TABLE_CAPTION.finditer(text):
        a, b = max(m.start() - 200, 0), min(m.start() + size, len(text))
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    if not spans:
        # No captions at all: the head of the paper, where an abstract often
        # reports its numbers inline.
        return [text[:size]]
    return [text[a:b] for a, b in spans[:cap]]


def _deterministic_pdf_tables(pdf_bytes: bytes, extraction_type: str,
                              tried: List[str]) -> List[Dict[str, Any]]:
    """The PDF readers, best first: Docling, then the text layer.

    Docling runs a layout model to find the table as a region and a structure
    model to read it, which is what a two-column page needs; pdfplumber infers
    the grid from where characters sit. They are not combined: when the better
    reader has answered, adding the weaker one's guesses only adds noise.
    """
    out: List[Dict[str, Any]] = []
    try:
        import docling_tables
    except Exception:
        docling_tables = None

    if docling_tables is not None and docling_tables.available():
        found = docling_tables.extract_tables(pdf_bytes)
        if found is None:
            tried.append("Docling could not read the file")
        else:
            for t in found:
                rows = t["data"]
                if not _table_is_real(rows):
                    continue
                out.append({
                    "title": t.get("title") or f"Table {len(out) + 1}",
                    "type": _classify_table(rows, extraction_type or ""),
                    "data": rows,
                    "caption": t.get("caption", ""),
                    "page": t.get("page"),
                    "tier": "docling",
                    "deterministic": True,
                })
            tried.append(f"Docling read the PDF: {len(out)} table(s)")
    else:
        tried.append("Docling not installed (pip install docling)")

    if not out:
        plumbed = _extract_pdf_tables(pdf_bytes, extraction_type) or []
        tried.append(f"PDF text layer: {len(plumbed)} table(s)")
        out.extend(plumbed)
    return out


@app.post("/api/extract/tables")
def extract_tables(req: ExtractTablesRequest):
    """Tables, by a ladder of readers that runs deterministic first.

      0. Identity. An uploaded PDF arrives with no identifier, so nothing below
         can be asked for the paper by name until a title search finds one.
      1. Europe PMC JATS fullTextXML: the publisher's own markup, where a table
         is already a table. No reader beats the typesetter's own file.
      2. The PDF itself, best reader first: Docling (layout + structure models),
         then the text layer, then image recognition where it is enabled. If no
         PDF is on this machine, the acquisition ladder is asked to fetch the
         open-access one first.
      3. A model over the windows of the text where a caption says a table is.
         Non-deterministic, so it runs only when every reader above came back
         empty.

    The response carries `tried`: which rungs ran and what each found. "This
    paper has no tables" and "we never managed to look" are different answers,
    and a reviewer has to be able to tell which one they got.
    """
    pid = req.paper_id or ""
    if not _is_db_id(pid) and req.URL:
        m = re.search(r"/(PMC\d+|\d+)/?$", req.URL, re.IGNORECASE)
        if m:
            pid = m.group(1)

    tried: List[str] = []
    tables: List[Dict[str, Any]] = []

    # Tier 0: find out what this paper is.
    if not _is_db_id(pid) and req.Title:
        found_id = _resolve_epmc_id(req.Title)
        if found_id:
            pid = found_id
            tried.append(f"matched the title to {found_id}")
        else:
            tried.append("no database record matched the title")

    # Tier 1: the publisher's structured full text.
    if _is_db_id(pid):
        tier1 = _extract_epmc_tables(pid, req.extraction_type or "") or []
        tried.append(f"Europe PMC full text: {len(tier1)} table(s)")
        tables.extend(tier1)
        if tier1:
            print(f"[extract_tables] {pid} JATS -> {len(tier1)} tables")

    # Tier 2: the file. Uploaded by the reviewer, cached by acquisition, or
    # fetched now from the open-access copy the identifier points at.
    pdf_bytes = _read_cached_pdf(req.paper_id, req.URL)
    if not pdf_bytes and not tables and (pid or req.URL):
        try:
            _fulltext_fetch_impl(FullTextRequest(
                Title=req.Title, URL=req.URL or "", Source=req.Source or "",
                paper_id=req.paper_id,
            ))
            pdf_bytes = _read_cached_pdf(req.paper_id, req.URL)
            tried.append("fetched the open-access PDF" if pdf_bytes
                         else "no open-access PDF to fetch")
        except Exception as e:
            tried.append(f"could not fetch a PDF ({str(e)[:80]})")

    if tables:
        # The publisher's own markup has answered. Reading the PDF as well would
        # only offer a worse copy of the same tables for the de-duplicator to
        # throw away, and a chance for a layout artifact to join them.
        tried.append("did not need to parse the PDF")
    elif pdf_bytes:
        tables = _dedup_tables(_deterministic_pdf_tables(pdf_bytes, req.extraction_type or "", tried))
    else:
        tried.append("no PDF on this machine to parse")

    # Tier 3: the model, on the parts of the text where a table is announced.
    if not tables:
        text = (req.full_text or "").strip() or (req.abstract or "").strip()
        windows = _table_windows(text)
        if not windows:
            tried.append("no text to read")
        else:
            found = 0
            for win in windows:
                seed = f"{req.Title}\n\n{win}"
                try:
                    ai_tables = AITableExtractor.extract_from_text(
                        seed,
                        resolve_for_thinking(req.model),
                        max_chars=len(seed) + 1,
                        num_ctx=Config.EXTRACTION_NUM_CTX,
                    ) or []
                except Exception as e:
                    print(f"[extract_tables ai fallback] {e}")
                    continue
                for t in ai_tables:
                    rows = [[str(c) for c in (t.get("headers") or [])]]
                    rows += [[str(c) for c in r] for r in (t.get("rows") or [])]
                    if not _table_is_real(rows):
                        continue
                    found += 1
                    tables.append({
                        "title": t.get("label", f"Table {found}"),
                        "type": _classify_table(rows, req.extraction_type or ""),
                        "data": rows,
                        "caption": t.get("caption", ""),
                        "page": None,
                        "tier": "llm",
                        "deterministic": False,
                    })
            tables = _dedup_tables(tables)
            tried.append(
                f"model over {len(windows)} region(s) of the text: {len(tables)} table(s)")
            print(f"[extract_tables] {pid or 'no-id'} tier3 (LLM) -> {len(tables)} tables "
                  f"from {len(windows)} windows of {len(text)} chars")

    if not tables:
        print(f"[extract_tables] {pid or req.URL[:60]} -> no tables found: {'; '.join(tried)}")

    return {"tables": tables, "tried": tried}


# ---------------------------------------------------------------------------
# Quality assessment (heuristic, mirrors mockServices)
# ---------------------------------------------------------------------------


class QualityRequest(BaseModel):
    paper: PaperIn
    full_text: Optional[str] = None     # if available; falls back to abstract
    rubric_override: Optional[str] = None  # legacy: force a rubric ("RoB 2", ...)
    study_design: Optional[str] = None  # captured at extraction; routes the instrument
    instrument_id: Optional[str] = None  # explicit override (else routed by design)
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


@app.get("/api/instruments")
def list_instruments():
    """Serve the instrument definitions (domains + signalling questions) so the
    frontend can render any tool generically. Backend is the single source of truth."""
    return {"instruments": [_inst.public_instrument(i) for i in _inst.INSTRUMENTS.values()]}


def _answer_signals(paper: PaperIn, text: str, inst: dict, model_name: str) -> Dict[str, Dict[str, dict]]:
    """Have the LLM ANSWER each signalling question (Y/PY/PN/N/NI) with a supporting
    quote — it never judges domains. Returns {domain_id: {signal_id: {answer, quote,
    section, rationale}}}. Roll-up to domain/overall judgments is done in code."""
    from langchain_core.messages import HumanMessage
    out: Dict[str, Dict[str, dict]] = {d["id"]: {} for d in inst["domains"]}
    model = AIService.get_model(model_name)
    if not model or not inst["domains"]:
        return out

    sig_lines = []
    for d in inst["domains"]:
        sig_lines.append(f'Domain "{d["id"]}" — {d["name"]}:')
        for s in d["signals"]:
            sig_lines.append(f'  - {s["id"]}: {s["text"]}')
    prompt = f"""You are a systematic-review methodologist applying the {inst['short_name']} tool.
Answer EACH signalling question below based ONLY on what the paper states. Do not judge the domain.

For each question return:
  - "answer": exactly one of "Y" (yes), "PY" (probably yes), "PN" (probably no), "N" (no), "NI" (no information)
  - "quote": a VERBATIM excerpt copied word-for-word from the PAPER TEXT below that supports your answer
    (<= 220 characters). Use "" only if the paper gives no explicit evidence. Never paraphrase or invent a quote.
  - "section": where that quote appears in the paper, e.g. "Methods", "Results", "Abstract", "Table 2" (or "")
  - "rationale": one clear sentence explaining the answer in terms of what the paper does or does not report

PAPER TITLE: {paper.title or "(untitled)"}
PAPER TEXT:
{text[:14000]}

SIGNALLING QUESTIONS:
{chr(10).join(sig_lines)}

Return ONLY JSON: {{"answers":[{{"signal_id":"1.1","answer":"Y","quote":"...","section":"Methods","rationale":"..."}}, ...]}}"""
    try:
        r = model.invoke([HumanMessage(content=prompt)])
        data = AIService._extract_json(r.content) or {}
        sig_to_domain = {s["id"]: d["id"] for d in inst["domains"] for s in d["signals"]}
        for it in (data.get("answers") or []):
            if not isinstance(it, dict):
                continue
            sid = str(it.get("signal_id") or "").strip()
            did = sig_to_domain.get(sid)
            if not did:
                continue
            ans = str(it.get("answer") or "NI").strip().upper()
            if ans not in _inst.YN:
                ans = "NI"
            out[did][sid] = {
                "answer": ans,
                "quote": str(it.get("quote") or "").strip()[:300],
                "section": str(it.get("section") or "").strip()[:120],
                "rationale": str(it.get("rationale") or "").strip()[:400],
            }
    except Exception as e:
        print(f"[quality] signalling error: {e}")
    return out


@app.post("/api/quality/assess")
def quality_assess(req: QualityRequest):
    """Instrument-driven risk-of-bias appraisal.

      1. Study design (from request, else detected) routes the instrument
         (RCT → RoB 2, NRSI → ROBINS-I, DTA → QUADAS-2, prediction/AI → PROBAST+AI, …).
      2. The LLM ANSWERS the instrument's signalling questions with evidence.
      3. Deterministic code rolls signalling answers up to per-domain judgments and
         an overall judgment (never a single holistic score).
    """
    paper = req.paper
    abs_text = paper.abstract or ""
    full_text = (req.full_text or "").strip() or abs_text
    model_name = resolve_for_thinking(req.model)

    design = (req.study_design or "").strip()
    if not design:
        design, _ = _detect_study_design(paper, full_text, model_name)

    instrument_id = req.instrument_id or _inst.instrument_for_design(design)
    if instrument_id not in _inst.INSTRUMENTS:
        instrument_id = _inst.instrument_for_design(design)
    inst = _inst.INSTRUMENTS[instrument_id]

    signals = _answer_signals(paper, full_text, inst, model_name)
    rolled = _inst.roll_up(instrument_id, {d: {s: v["answer"] for s, v in sig.items()} for d, sig in signals.items()})
    judg_by_domain = {d["id"]: d["judgment"] for d in rolled["domains"]}

    domains_out = []
    for d in inst["domains"]:
        sig_answers = signals.get(d["id"], {})
        sigs = [
            {**{"id": s["id"], "text": s["text"]}, **sig_answers.get(s["id"], {"answer": "NI", "quote": "", "section": "", "rationale": ""})}
            for s in d["signals"]
        ]
        # Domain-level rationale/quote synthesised from the signalling answers, so
        # existing UI (which reads domain.rationale/supporting_quote) keeps working
        # while the richer `signals` array powers the new generic renderer.
        rationale = " ".join(x["rationale"].strip() for x in sigs if x.get("rationale"))[:800]
        quoted = next((x for x in sigs if x.get("quote")), None)
        domains_out.append({
            "id": d["id"], "name": d["name"],
            "judgment": judg_by_domain.get(d["id"], inst["scale"][-1]),
            "rationale": rationale,
            "supporting_quote": quoted["quote"] if quoted else "",
            "section": quoted["section"] if quoted else "",
            "signals": sigs,
        })

    return {
        "paper_id": paper.id, "title": paper.title, "source": paper.source, "url": paper.url,
        "abstract": abs_text,
        "study_design": design,
        "instrument_id": instrument_id,
        "instrument": inst["short_name"],
        "axis": inst["axis"],
        "scale": inst["scale"],
        "reference": inst.get("reference", ""),
        "domains": domains_out,
        "overall_judgment": rolled["overall"],
        "overall_rationale": f"{inst['short_name']} roll-up of {len(domains_out)} domains.",
        "used_full_text": bool(req.full_text and len(req.full_text) > len(abs_text)),
        # Legacy alias so older frontends keep working.
        "rubric": inst["short_name"],
    }


class GradeOutcomeRequest(BaseModel):
    outcome: str
    starting: str = "randomized"                 # "randomized" | "observational"
    downgrades: Dict[str, int] = {}              # domain -> 0/-1/-2
    upgrades: Dict[str, int] = {}                # factor -> 0/+1/+2 (observational only)
    notes: Optional[Dict[str, str]] = None


@app.post("/api/grade")
def grade_outcome(req: GradeOutcomeRequest):
    """Deterministic GRADE certainty for one OUTCOME (outcome-level, not study-level)."""
    res = _inst.grade_certainty(req.starting, req.downgrades or {}, req.upgrades or {})
    return {
        "outcome": req.outcome,
        "starting": req.starting,
        "downgrade_domains": _inst.GRADE_DOWNGRADE,
        "upgrade_factors": _inst.GRADE_UPGRADE,
        "downgrades": req.downgrades or {},
        "upgrades": req.upgrades or {},
        "certainty": res["certainty"],
        "start_certainty": res["start"],
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


_LOCAL_TAGS: Dict[str, Any] = {"at": 0.0, "names": []}


def _local_model_missing(name: str) -> bool:
    """True when `name` is a local model this machine does not have.

    Extraction is routed to a general-purpose model regardless of what the
    reviewer picked for screening, so it can be a model they never downloaded.
    Two minutes of requests that were always going to fail, ending in "the
    model did not return usable JSON", is a bad way to learn that.
    """
    if not name or re.match(r"^(claude|gpt|gemini|o[13])", name, re.IGNORECASE):
        return False
    now = time.time()
    if now - _LOCAL_TAGS["at"] > 30:
        try:
            r = requests.get(f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags",
                             timeout=4)
            _LOCAL_TAGS["names"] = [m.get("name", "") for m in (r.json().get("models") or [])]
            _LOCAL_TAGS["at"] = now
        except Exception:
            return False          # cannot tell: let the call proceed and fail honestly
    return not any(t == name or t.startswith(name) for t in _LOCAL_TAGS["names"])


@app.get("/api/models/local")
def list_local_models():
    """List models available in the local Ollama instance."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            return {"running": False, "models": [], "error": f"Ollama returned {r.status_code}"}
        data = r.json()
        models = [m.get("name") for m in data.get("models", []) if m.get("name")]
        # The model a non-screening stage actually runs when a LEADS model is
        # selected for it. LEADS-Mistral is fine-tuned on screening verdicts and
        # returns nothing usable for other work, so those calls are redirected
        # here. The UI needs the name to say so instead of showing a model that
        # is not the one running.
        return {"running": True, "models": models,
                "non_screening_fallback": NON_SCREENING_FALLBACK_MODEL}
    except Exception as e:
        return {"running": False, "models": [], "error": str(e)}


@app.get("/api/reproducibility")
def reproducibility():
    """Run manifest for reproducibility. temperature=0 plus a fixed decoding seed
    make LOCAL (Ollama) and OpenAI runs deterministic, so a review re-runs to
    identical output; Anthropic/Gemini expose no seed API, so they are best-effort.
    Local model digests are pinned so a stored decision names the exact weights."""
    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    local_models: List[Dict[str, Any]] = []
    try:
        tags = requests.get(f"{ollama_base}/api/tags", timeout=4).json().get("models", [])
        for m in tags:
            local_models.append({
                "name": m.get("name"),
                "digest": (m.get("digest") or "")[:19],   # short pinned digest
                "modified": m.get("modified_at", ""),
            })
    except Exception:
        pass
    return {
        "seed": Config.RUN_SEED,
        "temperature": 0,
        "prompt_version": Config.PROMPT_VERSION,
        "deterministic_providers": ["ollama (local)", "openai (seeded)"],
        "best_effort_providers": ["anthropic", "google"],
        "local_models": local_models,
        "note": ("temperature=0 and a fixed seed make local Ollama and OpenAI runs "
                 "reproducible to identical output. Anthropic and Gemini expose no "
                 "seed, so those are best-effort. Hosted models can also change weights "
                 "silently; local pinned models (by digest) cannot."),
    }


class SheetColumn(BaseModel):
    header: str
    samples: List[str] = []


class SheetSpec(BaseModel):
    name: str = ""
    columns: List[SheetColumn] = []


class InferFormRequest(BaseModel):
    # One sheet, as before, or a whole workbook. A real extraction template is
    # several tabs: study characteristics for everything, then tabs that only
    # apply to one design, then outcomes for everything again.
    columns: List[SheetColumn] = []
    sheet_name: str = ""
    sheets: List[SheetSpec] = []
    model: Optional[str] = None


# Columns that describe the row rather than something to extract from a paper.
_NOT_A_VARIABLE = re.compile(
    r"^(article|articles|study|studies|paper|papers|title|citation|reference|author|authors|"
    r"year|doi|pmid|id|#|no\.?|row|reviewer|extractor|initials|date|notes?|comments?|status|"
    r"included?|exclude[d]?|screened)$", re.I)


def _infer_fields_heuristically(cols: List[SheetColumn]) -> List[Dict[str, Any]]:
    """What the sheet says on its own, with no model involved.

    Used when no model is configured and as the floor under the model's answer,
    so an unreachable model degrades to the behaviour this page already had
    rather than returning nothing.
    """
    out: List[Dict[str, Any]] = []
    for c in cols:
        header = (c.header or "").strip()
        if not header:
            continue
        vals = [v for v in (c.samples or []) if str(v).strip()]
        numeric = bool(vals) and all(
            re.fullmatch(r"[-+]?[\d,.]+\s*%?", str(v).strip()) for v in vals)
        uniq = {str(v).strip() for v in vals}
        # A handful of repeated short values is a category, not free text.
        categorical = bool(uniq) and len(uniq) <= 6 and len(vals) >= 3 and \
            all(len(u) <= 32 for u in uniq)
        out.append({
            "label": header,
            "type": "number" if numeric else ("select" if categorical else "text"),
            "options": sorted(uniq) if categorical and not numeric else [],
            "description": "",
            # Narrow and literal: a whole header that IS "id" or "reviewer".
            # Anything else stays in. A field that is wrongly present is one
            # the reviewer unticks; a field wrongly absent is one they never
            # find out about.
            "include": not bool(_NOT_A_VARIABLE.match(header)),
        })
    return out


def _classify_sheets(model, sheets: List[SheetSpec]) -> Dict[str, str]:
    """What each worksheet in a template is for.

    "extract"    - data recorded once per study: a real extraction tab
    "dictionary" - defines the variables (name, definition, allowed values)
    "other"      - instructions, a cover sheet, notes, a blank tab

    Matching the tab's NAME against a list of likely words only catches the
    obvious ones, and a dictionary named "Variables" or "Field guide" was being
    extracted as though its rows were studies. The reviewer can override every
    answer, so a wrong guess costs a click rather than a bad form.
    """
    roles = {sh.name: "extract" for sh in sheets}
    if not model or len(sheets) < 2:
        return roles
    listing = "\n".join(
        f'- "{sh.name}": columns = '
        + ", ".join(c.header for c in sh.columns[:14])
        + " | first values = "
        + "; ".join((c.samples or [""])[0] for c in sh.columns[:6] if c.samples)
        for sh in sheets
    )
    prompt = (
        "These are the worksheets in a systematic reviewer's data-extraction workbook.\n"
        "Say what each one is for:\n"
        '  "extract"    - data recorded once per included study\n'
        '  "dictionary" - defines the variables themselves (a variable name column\n'
        "                 and a definition column; its rows are field names, not studies)\n"
        '  "other"      - instructions, a cover sheet, notes, or an empty tab\n\n'
        f"WORKSHEETS:\n{listing}\n\n"
        'Reply with JSON only: {"sheets": [{"name": "...", "role": "extract"}]}'
    )
    try:
        from langchain_core.messages import HumanMessage
        r = model.invoke([HumanMessage(content=prompt)])
        data = AIService._extract_json(r.content) or {}
        for item in (data.get("sheets") or []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            role = str(item.get("role") or "").strip().lower()
            if name in roles and role in ("extract", "dictionary", "other"):
                roles[name] = role
    except Exception as e:
        print(f"[classify_sheets] {e}")
    # Never classify every sheet away: a workbook with nothing to extract is a
    # classification failure, not a real answer.
    if all(v != "extract" for v in roles.values()):
        return {sh.name: "extract" for sh in sheets}
    return roles


def _suggest_tab_rules(model, tabs: List[Dict[str, Any]]) -> None:
    """Ask which tabs cover every study and which only cover one kind.

    A template's tab names say this out loud ("RCTs", "Cohort studies",
    "Study characteristics"), and the deciding column is nearly always on one of
    the all-studies tabs. Mutates `tabs` in place, leaving the rule empty when
    the model does not answer usefully, because a wrong rule silently drops
    studies out of a tab and that is worse than no rule.
    """
    if len(tabs) < 2:
        return
    listing = "\n".join(
        f'- "{t["name"]}": ' + ", ".join(f["label"] for f in t["fields"][:12])
        for t in tabs
    )
    # Name the values each column can actually take. Without this the model
    # writes the rule in its own words ("Randomised controlled trial") while the
    # sheet, and so the extraction, says "RCT", and the tab then matches nothing.
    choices: Dict[str, List[str]] = {}
    for t in tabs:
        for f in t["fields"]:
            if f.get("options"):
                choices.setdefault(f["label"], list(f["options"]))
    choice_lines = "\n".join(f'- "{k}" is one of: {", ".join(v)}' for k, v in choices.items())
    every_field = sorted({f["label"] for t in tabs for f in t["fields"]})
    prompt = (
        "A systematic reviewer's data-extraction workbook has these tabs, each with\n"
        "its columns. Some tabs are filled in for EVERY included study; others only\n"
        "for studies of one kind (a design, a population, an outcome type).\n\n"
        f"TABS:\n{listing}\n\n"
        f"COLUMNS AVAILABLE ANYWHERE: {', '.join(every_field[:60])}\n\n"
        + (f"KNOWN COLUMN VALUES:\n{choice_lines}\n\n" if choice_lines else "")
        + "For each tab say whether it applies to all studies. If it does not, name the\n"
        "column whose value decides it and the values that qualify. The deciding\n"
        "column must be one of the columns listed above, and where that column's\n"
        "values are listed, use those values EXACTLY as written, not a longer name\n"
        "for the same thing.\n\n"
        'Reply with JSON only: {"tabs": [{"name": "...", "all_studies": true, '
        '"deciding_column": "", "values": []}]}'
    )
    try:
        from langchain_core.messages import HumanMessage
        r = model.invoke([HumanMessage(content=prompt)])
        data = AIService._extract_json(r.content) or {}
        by_name = {str(t.get("name", "")).strip().lower(): t
                   for t in (data.get("tabs") or []) if isinstance(t, dict)}
        labels = {f["label"].strip().lower() for t in tabs for f in t["fields"]}
        for tab in tabs:
            got = by_name.get(tab["name"].strip().lower())
            if not got or got.get("all_studies", True):
                continue
            col = str(got.get("deciding_column") or "").strip()
            vals = [str(v).strip() for v in (got.get("values") or []) if str(v).strip()]
            # Only accept a rule naming a column that actually exists, else the
            # tab would be gated on something nothing ever fills in.
            if not (col and vals and col.strip().lower() in labels):
                continue
            # If that column has a fixed set of values, the rule has to use them.
            # A value outside the set can never match, so a rule made entirely of
            # such values is dropped: no rule (the reviewer sets one) beats a rule
            # that silently empties the tab.
            opts = choices.get(col) or next(
                (list(f["options"]) for t2 in tabs for f in t2["fields"]
                 if f["label"].strip().lower() == col.strip().lower() and f.get("options")),
                [],
            )
            if opts:
                lower = {o.strip().lower(): o for o in opts}
                vals = [lower[v.strip().lower()] for v in vals if v.strip().lower() in lower]
                if not vals:
                    continue
            tab["when"] = {"label": col, "equals": vals[:12]}
    except Exception as e:
        print(f"[suggest_tab_rules] {e}")


class ClassifyTabsRequest(BaseModel):
    # Just enough of each tab to judge it: the name and its column labels.
    tabs: List[SheetSpec] = []
    model: Optional[str] = None


@app.post("/api/extraction/classify-tabs")
def classify_tabs(req: ClassifyTabsRequest):
    """Roles and rules for tabs whose fields have already been inferred.

    Separated from infer-form so the client can infer one sheet at a time and
    show real progress: a five-tab workbook is five model calls, and a single
    request that takes a minute with only a spinner tells the reviewer nothing
    about whether it is working or stuck.
    """
    sheets = [sh for sh in (req.tabs or []) if (sh.name or "").strip()]
    if not sheets:
        return {"roles": {}, "rules": {}}

    model = AIService.get_model(resolve_for_thinking(req.model))
    roles = _classify_sheets(model, sheets) if model else {sh.name: "extract" for sh in sheets}

    # _suggest_tab_rules works on the tab shape, so build it from what we were
    # given. Options travel as a column's samples, which is where it looks.
    tabs = [{
        "name": sh.name,
        "when": None,
        "fields": [{"label": c.header, "options": list(c.samples or []),
                    "type": "select" if c.samples else "text"}
                   for c in sh.columns],
    } for sh in sheets if roles.get(sh.name) == "extract"]
    if model:
        _suggest_tab_rules(model, tabs)
    return {"roles": roles, "rules": {t["name"]: t["when"] for t in tabs if t.get("when")}}


@app.post("/api/extraction/infer-form")
def infer_extraction_form(req: InferFormRequest):
    """Turn a reviewer's existing extraction workbook into a form.

    Taking the headers literally is what this page used to do, and it extracts
    badly: every field arrives as untyped text with no description, and the
    description is what tells the model where in a paper to look. So the sample
    values are read too, and a type and a description are inferred per column.

    A real template is several tabs, not one: study characteristics for every
    study, then tabs filled in only for one design, then outcomes for every
    study again. Each sheet becomes a tab, and the tabs that only apply to some
    studies get a rule naming the column and values that decide it.

    Sample values are examples of what the reviewer wants, never data to import:
    nothing here writes an extraction result.
    """
    sheets = [sh for sh in (req.sheets or []) if sh.columns]
    if not sheets and req.columns:
        sheets = [SheetSpec(name=req.sheet_name or "Form", columns=req.columns)]
    sheets = [sh for sh in sheets if [c for c in sh.columns if (c.header or "").strip()]]
    if not sheets:
        return {"tabs": [], "fields": [], "source": "empty"}

    model = AIService.get_model(resolve_for_thinking(req.model))
    tabs: List[Dict[str, Any]] = []
    used_model = False

    for sh in sheets:
        cols = [c for c in sh.columns if (c.header or "").strip()]
        fallback = _infer_fields_heuristically(cols)
        fields = fallback
        if model:
            preview = "\n".join(
                f'- "{c.header}": '
                + (", ".join(f'"{v}"' for v in (c.samples or [])[:5]) or "(no values yet)")
                for c in cols[:120]   # beyond this the fallback types still apply
            )
            prompt = (
                "A systematic reviewer records data extracted from papers in a spreadsheet.\n"
                "Turn this tab's columns into extraction fields.\n\n"
                "For each column decide:\n"
                "  include     - false if it is not something you read out of a paper\n"
                "                (a row number, study id, reviewer initials, a date stamp).\n"
                "  type        - number | text | select\n"
                "  options     - for select only: the allowed values, from the examples\n"
                "  description - an instruction naming what to look for and where in a\n"
                "                paper it usually appears. One sentence.\n\n"
                f"TAB: {sh.name or 'untitled'}\n"
                f"COLUMNS (header, then example values already recorded):\n{preview}\n\n"
                'Reply with JSON only: {"fields": [{"label": "...", "include": true, '
                '"type": "text", "options": [], "description": "..."}]}\n'
                "Keep the labels exactly as the headers are written."
            )
            try:
                from langchain_core.messages import HumanMessage
                r = model.invoke([HumanMessage(content=prompt)])
                data = AIService._extract_json(r.content) or {}
                by_label = {str(f.get("label", "")).strip().lower(): f
                            for f in (data.get("fields") or []) if isinstance(f, dict)}
                merged = []
                for base in fallback:
                    got = by_label.get(base["label"].strip().lower())
                    if not got:
                        merged.append(base)
                        continue
                    ftype = str(got.get("type") or base["type"]).lower()
                    if ftype not in ("number", "text", "select"):
                        ftype = base["type"]
                    opts = got.get("options")
                    merged.append({
                        "label": base["label"],
                        "type": ftype,
                        "options": [str(o) for o in opts][:20] if isinstance(opts, list) else base["options"],
                        "description": str(got.get("description") or "").strip(),
                        # The model does not decide this: it was unticking real
                        # variables on large templates, and an unticked field is
                        # dropped on apply, so columns went missing with nothing
                        # on screen to explain it.
                        "include": base["include"],
                    })
                fields = merged
                used_model = True
            except Exception as e:
                print(f"[infer_extraction_form] {sh.name}: {e}")

        tabs.append({"name": sh.name or f"Tab {len(tabs) + 1}", "fields": fields,
                     "when": None, "role": "extract"})

    roles = _classify_sheets(model, sheets) if model else {}
    for tab in tabs:
        tab["role"] = roles.get(tab["name"], "extract")
    if model:
        # Rules are only meaningful between the tabs that hold study data.
        _suggest_tab_rules(model, [t for t in tabs if t["role"] == "extract"])

    # `fields` is the flattened list, so a caller that only wants a single form
    # (and every existing test) keeps working.
    flat = [f for t in tabs for f in t["fields"]]
    return {"tabs": tabs, "fields": flat, "source": "model" if used_model else "heuristic"}


class LookupRequest(BaseModel):
    queries: List[str]
    cap: int = 500


def _classify_query(q: str) -> str:
    """A pasted line is a DOI, a PubMed ID, or a title."""
    t = q.strip()
    if re.match(r"^(https?://(dx\.)?doi\.org/)?10\.\d{4,9}/\S+$", t, re.I):
        return "doi"
    # PMIDs can technically be as short as one digit, but a short bare number in
    # a pasted list is far more often a stray year. The two mistakes are not
    # equally costly: reading "2020" as a PMID silently adds a wrong paper the
    # reviewer has no reason to doubt, while treating a genuine short PMID as a
    # title just fails the lookup and is reported back to them.
    if t.isdigit() and 5 <= len(t) <= 9:
        return "pmid"
    return "title"


def _lookup_one(q: str) -> Optional[Dict[str, Any]]:
    """Resolve one pasted line to a paper record, or None if nothing matched.

    Reuses the resolvers acquisition and snowballing already go through, so this
    inherits their per-host rate limiting and run cache rather than opening a
    second, unthrottled path to the same services.
    """
    kind = _classify_query(q)
    text = q.strip()
    doi = ""
    if kind == "doi":
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", text, flags=re.I)

    work = None
    if kind == "doi":
        work = _openalex_resolve_work("", doi)
    elif kind == "pmid":
        # OpenAlex indexes PubMed ids, and this keeps one code path for metadata.
        try:
            r = limited_get(
                "https://api.openalex.org/works",
                params=_polite_params({"filter": f"pmid:{text}", "per_page": 1}), timeout=6,
            )
            if r is not None and r.status_code == 200:
                results = r.json().get("results", [])
                work = results[0] if results else None
        except Exception as e:
            print(f"[lookup pmid] {e}")
    else:
        work = _openalex_resolve_work(text, "")

    if not work:
        return None

    rec = _openalex_minimal(work, "lookup")
    rec["source"] = "Imported list"
    rec["authors"] = ", ".join(
        (a.get("author") or {}).get("display_name", "")
        for a in (work.get("authorships") or [])[:8]
    ).strip(", ")
    # A title search can land on the wrong paper. Say what was asked for so the
    # reviewer can see the match rather than discovering it later in screening.
    rec["matched_from"] = text
    if not rec.get("abstract"):
        # OpenAlex often lacks the abstract; Europe PMC covers far more.
        ident = _openalex_minimal(work, "lookup").get("id", "")
        try:
            rec["abstract"] = _abstract_via_epmc(rec.get("title", ""), ident) or ""
        except Exception:
            rec["abstract"] = ""
    return rec


@app.post("/api/papers/lookup")
def papers_lookup(req: LookupRequest):
    """Resolve pasted titles, DOIs or PMIDs into papers that can be screened.

    A bare title cannot be screened: there is no abstract to read and no DOI to
    acquire a full text with. This fills both in where the services know them,
    and reports the ones it could not find rather than inventing records.
    """
    found: List[Dict[str, Any]] = []
    missing: List[str] = []
    for q in req.queries[: max(1, req.cap)]:
        if not (q or "").strip():
            continue
        try:
            rec = _lookup_one(q)
        except Exception as e:
            print(f"[papers_lookup] {q[:60]}: {e}")
            rec = None
        (found.append(rec) if rec else missing.append(q.strip()))
    return {"papers": found, "missing": missing,
            "requested": len([q for q in req.queries if (q or "").strip()])}


@app.get("/api/version")
def version():
    return {"version": _updates.current_version()}


@app.get("/api/update")
def update(force: bool = False):
    """Advisory: is a newer release published? Never fails the request.

    Answered from a once-a-day disk cache, so this is cheap enough for the app
    to ask on every load. `force` skips the cache for a manual re-check.
    """
    return _updates.check(force=force)


class UpdateInstallRequest(BaseModel):
    version: str


@app.post("/api/update/install")
def update_install(req: UpdateInstallRequest):
    """Download the named release and unpack it beside the reviewer's data.

    Nothing is replaced here. The launcher moves the staged copy into place
    once the app has quit, because a bundle cannot be swapped underneath the
    process running out of it.
    """
    if not _updates.is_newer(req.version, _updates.current_version()):
        raise HTTPException(400, "That version is not newer than the one running.")
    return self_update.start(req.version.strip().lstrip("vV"))


@app.get("/api/update/status")
def update_status():
    """How the download is getting on, for the banner to poll."""
    out = self_update.status()
    out["staged"] = self_update.staged()
    return out


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
    # Per-paper tables (paper_id → tables) the client already extracted. Effect
    # sizes live in results tables, and PDF-derived prose loses their column
    # structure, so passing these is the difference between reading the numbers
    # and guessing at a run-on sentence.
    tables: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


@app.post("/api/meta/extract")
def meta_extract(req: MetaExtractRequest):
    """Run the meta-analysis extraction agent on a list of papers.

    Uses the platform's "thinking" model (Qwen/Claude/etc.) — NOT LEADS, which
    is fine-tuned for screening verdicts, not numerical extraction."""
    model_name = resolve_for_thinking(req.model)

    def _one(p: PaperIn) -> Dict[str, Any]:
        d = p.dict() if hasattr(p, "dict") else dict(p.__dict__)
        ft = req.full_texts.get(str(d.get("id") or "")) or None
        pid = str(d.get("id") or "")
        se = extract_effect_size(
            d, model_name=model_name,
            outcome_hint=req.outcome, measure_hint=req.measure,
            full_text=ft, tables=req.tables.get(pid) or None,
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


class MetaAskRequest(BaseModel):
    """Ask one question about one paper and get a value back with its provenance."""
    paper: Dict[str, Any]
    question: str
    full_text: Optional[str] = None
    tables: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None


@app.post("/api/meta/ask")
def meta_ask(req: MetaAskRequest):
    """Grounded question answering over a single paper.

    Uses the thinking model, not LEADS: LEADS is fine-tuned for screening
    verdicts and does not do numerical extraction."""
    from meta_analysis import answer_grounded

    return answer_grounded(
        req.paper,
        req.question,
        model_name=resolve_for_thinking(req.model),
        full_text=req.full_text,
        tables=req.tables,
    )


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


def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats with None, recursively.

    Statistics legitimately produce NaN and infinity: a p-value with zero
    degrees of freedom, a z-score with zero standard error, a between-group Q
    with one group. FastAPI serialises with the strict JSON encoder, which
    rejects all three, so a single undefined statistic buried anywhere in the
    response turns the whole analysis into a 500 and the user sees "Analysis
    failed" with no hint which of a dozen sub-analyses was at fault.

    Undefined is information, not an error. It travels as null.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


@app.post("/api/meta/pool")
def meta_pool(req: MetaPoolRequest):
    studies = _hydrate_studies(req.extractions)
    return _json_safe(_ma_pool(studies, tau2_method=req.tau2_method,
                               use_knapp_hartung=req.use_knapp_hartung))


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
    # Every sub-analysis is wrapped: one undefined statistic anywhere in here
    # used to fail the serialisation of the entire response, so a run that
    # succeeded in nine of nine analyses still reached the user as a 500.
    return _json_safe({
        "pool": _ma_pool(studies, tau2_method=req.tau2_method, use_knapp_hartung=req.use_knapp_hartung),
        "subgroup": _ma_subgroup(studies, tau2_method=req.tau2_method),
        "leave_one_out": _ma_loo(studies, tau2_method=req.tau2_method),
        "cumulative": _ma_cumulative(studies, tau2_method=req.tau2_method),
        "funnel": _ma_funnel(studies, tau2_method=req.tau2_method),
        "egger": _ma_egger(studies),
        "begg": _ma_begg(studies),
        "trim_fill": _ma_trim_fill(studies, tau2_method=req.tau2_method),
        "meta_regression": _ma_metareg(studies, tau2_method=req.tau2_method),
    })


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
    # RAISE AI-use disclosure: stages where AI made or suggested judgements.
    # Each step carries the model that produced it: the reviewer chooses a model
    # per stage, so one name for the whole review would misreport the rest.
    ai_model: str = ""
    ai_models_by_stage: Dict[str, str] = {}
    ai_steps: List[Dict[str, str]] = []


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

    # RAISE AI-use block
    if req.ai_steps:
        # Name the model per stage. A review that screened with one model and
        # extracted with another has to say so, and the header lists every
        # model that made a judgement anywhere in the review.
        used: List[str] = []
        for st in req.ai_steps:
            for one in (x.strip() for x in (st.get("model") or "").split(",")):
                if one and one not in used:
                    used.append(one)
        header = ", ".join(used) or req.ai_model or "a large language model"
        ai_lines = [
            f"AI system{'s' if len(used) > 1 else ''}: {header}.",
            "Stages where AI made or suggested judgements (all reviewer-facing and overridable):",
        ]
        for st in req.ai_steps:
            # Not `model`: that name holds the LangChain client this endpoint
            # runs the prompt through, and shadowing it here cost a whole
            # endpoint once already.
            step_model = (st.get("model") or "").strip()
            ai_lines.append(
                f"  • {st.get('stage','')}: {st.get('purpose','')}"
                + (f" [model: {step_model}]" if step_model else "")
                + f" (Oversight: {st.get('oversight','')})"
            )
        ai_block = "\n".join(ai_lines)
    else:
        ai_block = "(No AI-assisted judgement stages were recorded.)"

    prompt = f"""You are writing the Search Strategy Methods appendix of a systematic review (PRISMA 2020 compliant). This is the prose appendix that accompanies the per-database search-string table; it must read like a polished journal methods appendix.

Review question: {req.goal or '(not specified)'}

{pico_block}

{ic_block}

{ec_block}

Databases searched (with record counts and query strings):
{db_block}

Search date: {req.search_date or 'not recorded'}

PRISMA screening funnel:
{funnel_block or '(screening counts not yet available)'}

AI and automation use (for the AI-use subsection):
{ai_block}

Write the appendix as SIX labeled subsections, in this exact order, each a flowing, substantive prose paragraph (no bullet points inside them). Put each subsection label in bold, followed by the paragraph. Match the depth, specificity and formal register of a published methods appendix in a high-quality journal: each paragraph should be 3-6 full sentences, concrete, and free of filler.

**Design and scope:** State the review type and aim, the clinical/topic domain (infer it specifically from the review question and PICO, naming the actual field), and how evidence was synthesized. If no effect estimates were pooled, state that it is a narrative synthesis and that formal risk-of-bias assessment or quantitative meta-analysis was not performed; otherwise describe the synthesis approach. Note adherence to the applicable items of the PRISMA 2020 statement.

**Data sources and search strategy:** Name every database searched and the search date / date range. Describe the concept blocks combined with the Boolean operator AND, deriving and naming the blocks from the PICO and the query strings (e.g. the population block, the intervention block, the outcome/application block), and give representative example terms from each. Note the use of controlled vocabulary (MeSH or database equivalents) alongside free-text title/abstract fields where identifiable from the strings, and mention any preprint or supplementary sources searched. State that the complete search strings and the date run for each source are provided in the accompanying appendix table.

**Eligibility criteria:** Describe, in prose, the study types and content that were included and excluded, grounded in the inclusion/exclusion criteria above and any date or language limits apparent from the queries. Be specific about what made a record eligible.

**Study selection:** Describe de-duplication and the screening workflow using the PRISMA counts (records retrieved, duplicates removed, unique records screened, and final included studies, only where provided). Describe title and abstract screening followed by full-text review, and that disagreements were resolved by consensus or adjudication by a senior reviewer. Note that the selection process is summarized in the PRISMA flow diagram.

**Data charting and synthesis:** Describe how each included study was charted (the key dimensions extracted), how studies were organized, and how findings were synthesized into the narrative.

**Use of AI and automation:** Provide a transparent declaration following the Responsible use of AI in evidence SynthEsis (RAISE) recommendations endorsed by Cochrane, the Campbell Collaboration, JBI and the Collaboration for Environmental Evidence. State that the review authors remain ultimately responsible for the content, methods and findings, including the decision to use AI. Name the AI system used and report, in prose, each stage at which AI made or suggested a judgement (drawn from the AI and automation use list above), making clear that all such use was conducted with human oversight and that every AI-generated judgement was reviewer-facing and could be overridden. Briefly note the principal limitations of large language models (potential for bias, overfitting, opaque decision-making and fabricated outputs) and that these were mitigated by human oversight, and state that ethical, legal and regulatory standards were observed. If no AI stages were recorded, state plainly that no AI or automation that makes or suggests judgements was used.

Rules:
- Write in past tense, third person, formal academic register, matching the quality of a published systematic-review methods appendix.
- Use ONLY the numbers, stages and facts provided above; do not invent counts, reviewer numbers, tool names, or details not given. If a number is missing, omit that claim rather than guessing.
- Do NOT name any specific commercial software, product, or company other than the AI system named above and the named bibliographic databases; for de-duplication/screening management refer generically (e.g. "a systematic review management tool") or simply state records were de-duplicated and screened.
- Output only the six labeled subsections — no overall title, no preamble, no closing remarks."""

    main_prompt = f"""You are writing the Methods section (main text) of a systematic review manuscript — concise and self-contained, as it appears in the body of the paper, NOT the appendix.

Review question: {req.goal or '(not specified)'}

{pico_block}

{ic_block}

{ec_block}

Databases searched (with record counts):
{db_block}

Search date: {req.search_date or 'not recorded'}

PRISMA screening funnel:
{funnel_block or '(screening counts not yet available)'}

AI and automation use:
{ai_block}

Write 2-3 tight paragraphs of flowing prose (no subsection labels, no bullet points) covering, in order: (1) the review type and aim, the topic domain named specifically, adherence to PRISMA 2020, and the synthesis approach (narrative vs. quantitative); (2) the databases searched and the search date, the concept blocks combined with AND (named from the PICO), the eligibility criteria, de-duplication and the two-stage screening (title/abstract then full text) with the actual PRISMA counts where provided, and how data were charted and synthesized; (3) a brief transparent statement of AI/automation use following the RAISE recommendations (name the AI system, note human oversight and that every AI-suggested judgement was reviewer-facing and could be overridden) — or state plainly that no judgement-making AI was used if none is recorded. Point the reader to the appendix for the full per-database search strings.

Rules:
- Past tense, third person, formal academic register. Concise — this is the body Methods, not the appendix.
- Use ONLY the numbers and facts provided; do not invent counts, reviewer numbers or tools. Omit any claim whose data is missing.
- Do NOT name commercial software or companies other than the AI system named above and the named bibliographic databases.
- Output only the prose paragraphs — no title, no preamble, no closing remarks."""

    # A model failure here used to escape the ThreadPoolExecutor as a 500 with a
    # traceback. The reviewer has already done the work this paragraph
    # describes; the right answer to a dead model is to say so, not to lose the
    # page. Each half is independent, so one failing does not sink the other.
    def _run(p: str) -> str:
        try:
            return (model.invoke([HumanMessage(content=p)]).content or "").strip()
        except Exception as e:
            print(f"[writing_summary] {e}")
            return ""

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_main = ex.submit(_run, main_prompt)
        fut_app = ex.submit(_run, prompt)
        main_text = fut_main.result()
        appendix = fut_app.result()

    out = {"main_text": main_text, "appendix": appendix, "summary": appendix}
    if not main_text and not appendix:
        out["error"] = "The model did not return a summary. Check that it is running."
    return out


class CharItem(BaseModel):
    id: str = ""
    title: str = ""
    abstract: str = ""
    full_text: str = ""


class CharacteristicsRequest(BaseModel):
    papers: List[CharItem] = []
    model: str = ""


@app.post("/api/writing/characteristics")
def writing_characteristics(req: CharacteristicsRequest):
    """Extract 'Characteristics of included studies' fields per paper (design,
    population, intervention, comparator, outcomes) from each study's text."""
    from langchain_core.messages import HumanMessage
    model = AIService.get_model(resolve_for_thinking(req.model))
    if not model:
        raise HTTPException(status_code=503, detail="No model available")

    def _one(p: CharItem) -> Dict[str, Any]:
        text = (p.full_text or p.abstract or "")[:6000]
        base = {"id": p.id, "design": "", "population": "", "intervention": "", "comparator": "", "outcomes": "", "key_finding": ""}
        if not text.strip():
            return base
        prompt = (
            "From this study report, fill a structured evidence-table row. "
            "Be concise (a short phrase per field), use the study's own terms, and return ONLY JSON with keys "
            '{"design","population","intervention","comparator","outcomes","key_finding"}. '
            '"key_finding" is the single most important result or conclusion, in one sentence. '
            "Use an empty string for anything not reported. No commentary, no code fences.\n\n"
            f"TITLE: {p.title}\n\nREPORT:\n{text}"
        )
        try:
            r = model.invoke([HumanMessage(content=prompt)])
            data = AIService._extract_json(r.content) or {}
        except Exception as e:
            print(f"[characteristics] {e}")
            data = {}
        for k in ("design", "population", "intervention", "comparator", "outcomes", "key_finding"):
            base[k] = str(data.get(k) or "").strip()
        return base

    workers = max(1, min(Config.PARALLEL_AGENT_WORKERS, len(req.papers) or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_one, req.papers))
    return {"characteristics": out}


class AskPaper(BaseModel):
    n: int = 0
    title: str = ""
    abstract: str = ""
    full_text: str = ""


class AskRequest(BaseModel):
    question: str = ""
    papers: List[AskPaper] = []
    model: str = ""


@app.post("/api/writing/ask")
def writing_ask(req: AskRequest):
    """Answer a question by synthesizing ONLY from the included papers, citing
    each claim with the study number(s) in brackets — grounded, no outside
    knowledge, no invented references."""
    from langchain_core.messages import HumanMessage
    model = AIService.get_model(resolve_for_thinking(req.model))
    if not model:
        raise HTTPException(status_code=503, detail="No model available")
    q = (req.question or "").strip()
    if not q:
        return {"answer": ""}
    blocks = []
    for p in req.papers[:40]:
        body = (p.full_text or p.abstract or "")[:2500]
        blocks.append(f"[{p.n}] {p.title}\n{body}")
    corpus = "\n\n".join(blocks) if blocks else "(no papers provided)"
    prompt = (
        "You are a systematic-review assistant. Answer the QUESTION using ONLY the numbered studies below. "
        "Cite every claim with the study number(s) in square brackets, e.g. [1] or [2,5]. "
        "Do NOT use outside knowledge and do NOT invent citations. If the studies do not address the question, "
        "say so plainly. Write a concise synthesised answer (a few sentences up to a short paragraph), grouping "
        "agreeing findings and noting disagreements.\n\n"
        f"QUESTION: {q}\n\nSTUDIES:\n{corpus}"
    )
    try:
        r = model.invoke([HumanMessage(content=prompt)])
        return {"answer": (r.content or "").strip()}
    except Exception as e:
        print(f"[writing_ask] {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ProtocolRequest(BaseModel):
    title: str = ""
    pico: Optional[Dict[str, Any]] = None
    inclusion: List[str] = []
    exclusion: List[str] = []
    sources: List[str] = []
    model: str = ""


@app.post("/api/writing/protocol")
def writing_protocol(req: ProtocolRequest):
    """Draft a register-first systematic-review protocol (PROSPERO-style
    sections) from the PICO and eligibility criteria. Original methods prose;
    the reviewer edits it before registering. Placeholders are labelled where a
    detail was not supplied; no results are invented."""
    from langchain_core.messages import HumanMessage
    model = AIService.get_model(resolve_for_thinking(req.model))
    if not model:
        raise HTTPException(status_code=503, detail="No model available")
    pico = req.pico or {}
    fw = frameworks.normalize(pico.get("framework"))
    is_pcc = fw == "pcc"
    elem_lines = "\n".join(f"{frameworks.label_for(fw, eid)}: {pico.get(eid, '')}"
                           for eid in frameworks.element_ids(fw))
    if is_pcc:
        review_kind = "scoping-review PROTOCOL (PRISMA-ScR / JBI methodology)"
        elig_label = "Eligibility criteria (Population, Concept, Context, and the types of evidence sources)"
        q_label = "Review objectives (what the scoping review will MAP — extent, range, nature of the evidence; not an effect question)"
        synth_label = "Data charting and presentation of results (narrative + tabular/mapping; no effect meta-analysis or GRADE)"
    else:
        review_kind = "systematic-review PROTOCOL"
        elig_label = "Eligibility criteria (Population, Intervention, Comparator, Outcomes, Study designs)"
        q_label = "Review question and objectives"
        synth_label = "Data synthesis and certainty of evidence (GRADE)"
    prompt = f"""You are drafting a {review_kind} to be registered BEFORE the
review begins. Use the details supplied; where a detail is missing, write a
clearly-labelled placeholder the author can complete (e.g. "[to be specified]").
Do not invent results or findings. Write clear methods prose using Markdown
headings, covering these sections in order:

1. Title
2. Background and rationale
3. {q_label}
4. {elig_label}
5. Information sources and search strategy
6. Study/source selection process (screening and dual review)
7. Data {"charting" if is_pcc else "extraction"}
8. {"Critical appraisal (optional in scoping reviews; state if performed)" if is_pcc else "Risk-of-bias assessment"}
9. {synth_label}
10. Timeline and planned amendments

DETAILS
Title: {req.title or "(untitled review)"}
{elem_lines}
Inclusion criteria: {'; '.join(req.inclusion) or '(none supplied)'}
Exclusion criteria: {'; '.join(req.exclusion) or '(none supplied)'}
Information sources: {', '.join(req.sources) or '(open-access databases)'}

Return ONLY the protocol text in Markdown."""
    try:
        r = model.invoke([HumanMessage(content=prompt)])
        return {"protocol": (r.content or "").strip()}
    except Exception as e:
        print(f"[writing_protocol] {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Research-integrity check: flag included studies that have been retracted or
# carry an expression of concern, using OpenAlex (is_retracted, sourced from
# Retraction Watch) and Crossref update relationships. Both are already queried
# elsewhere, so this needs no new credentials.
# ---------------------------------------------------------------------------

class IntegrityPaper(BaseModel):
    paper_id: str
    doi: Optional[str] = None
    title: Optional[str] = None


class IntegrityRequest(BaseModel):
    papers: List[IntegrityPaper] = []


def _norm_doi(s: Optional[str]) -> str:
    if not s:
        return ""
    m = re.search(r"10\.\d{4,9}/[^\s\"<>]+", s.strip().lower())
    return m.group(0) if m else ""


def _openalex_retracted(doi: str, title: Optional[str]) -> Optional[dict]:
    """Return {status, detail, doi} from OpenAlex, or None if not resolvable."""
    try:
        if doi:
            r = requests.get(f"https://api.openalex.org/works/doi:{doi}",
                             params={"select": "id,is_retracted,doi"}, timeout=8)
            if r.status_code == 200:
                w = r.json()
                return {"status": "retracted" if w.get("is_retracted") else "ok",
                        "detail": "Listed as retracted in OpenAlex" if w.get("is_retracted") else "",
                        "doi": doi, "source": "OpenAlex"}
        elif title:
            r = requests.get("https://api.openalex.org/works",
                             params={"search": title, "per_page": 1, "select": "id,is_retracted,doi,title"}, timeout=8)
            if r.status_code == 200:
                res = (r.json().get("results") or [])
                if res:
                    w = res[0]
                    return {"status": "retracted" if w.get("is_retracted") else "ok",
                            "detail": "Listed as retracted in OpenAlex (matched by title)" if w.get("is_retracted") else "",
                            "doi": _norm_doi(w.get("doi")), "source": "OpenAlex"}
    except Exception as e:
        print(f"[integrity/openalex] {e}")
    return None


def _crossref_update(doi: str) -> Optional[dict]:
    """Return a retraction/concern flag from Crossref update-to relations, with
    the linking notice's DOI and date so the reviewer can read the actual notice."""
    if not doi:
        return None
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=8)
        if r.status_code != 200:
            return None
        updates = r.json().get("message", {}).get("update-to") or []

        def _match(pred, status, label):
            for u in updates:
                t = (u.get("type") or "").lower()
                if pred(t):
                    parts = ((u.get("updated") or {}).get("date-parts") or [[]])[0]
                    date = "-".join(f"{x:02d}" if i else str(x) for i, x in enumerate(parts)) if parts else ""
                    notice_label = (u.get("label") or "").strip()
                    detail = label + (f" on {date}" if date else "")
                    if notice_label and notice_label.lower() not in label.lower():
                        detail += f" — {notice_label}"
                    return {"status": status, "detail": detail, "source": "Crossref",
                            "notice_doi": (u.get("DOI") or "").strip(), "notice_date": date}
            return None

        return (_match(lambda t: "retract" in t, "retracted", "Retraction notice issued")
                or _match(lambda t: "concern" in t, "concern", "Expression of concern issued")
                or _match(lambda t: ("correct" in t or "erratum" in t), "correction", "Correction/erratum issued"))
    except Exception as e:
        print(f"[integrity/crossref] {e}")
    return None


def _check_integrity_one(p: IntegrityPaper) -> dict:
    doi = _norm_doi(p.doi)
    out = {"paper_id": p.paper_id, "status": "unknown", "detail": "No DOI to check",
           "source": "", "doi": doi, "notice_doi": "", "notice_date": ""}
    oa = _openalex_retracted(doi, p.title)
    if oa:
        out.update(oa)
        doi = out.get("doi") or doi
    # A concern/correction is only added if OpenAlex did not already flag a retraction.
    if out["status"] in ("ok", "unknown") and doi:
        cr = _crossref_update(doi)
        if cr:
            out.update(cr)
        elif out["status"] == "unknown":
            out.update(status="ok", detail="No retraction found")
    return out


@app.post("/api/integrity/check")
def integrity_check(req: IntegrityRequest):
    papers = req.papers[:300]
    results: List[dict] = []
    if not papers:
        return {"results": results}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_check_integrity_one, p): p for p in papers}
        for f in as_completed(futs):
            p = futs[f]
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"paper_id": p.paper_id, "status": "unknown", "detail": str(e), "source": "", "doi": ""})
    return {"results": results}


# ---------------------------------------------------------------------------
# Built frontend
# ---------------------------------------------------------------------------
# Serving the Vite build from this process makes `uvicorn api:app` the whole
# application, with no Node server at runtime. This has to be the LAST thing the
# module registers: the fallback route matches any unclaimed path, so anything
# registered after it would be unreachable. Harmless when dist/ is absent, which
# is the case for backend-only development and the test suite.
import webapp  # noqa: E402

_SERVING_FRONTEND = webapp.mount_frontend(app)
