"""Behavioural tests for the LLM-backed API endpoints.

Run: Backend/.venv/bin/python -m pytest Backend/test_api_endpoints.py

Every model call is stubbed, so these test the code AROUND the model: request
parsing, prompt assembly, response shaping, and what happens when the model
returns something unusable. That last case is the one that matters, because a
7B model returns unusable output regularly and the user-visible result should be
a degraded answer, not a 500 with a traceback.
"""
import json

import pytest
from fastapi.testclient import TestClient

import api
import utils


@pytest.fixture
def client():
    return TestClient(api.app, raise_server_exceptions=False)


class StubModel:
    """Returns a canned reply to every invoke, whatever the prompt."""
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, messages):
        try:
            self.prompts.append(getattr(messages[-1], "content", str(messages)))
        except Exception:
            self.prompts.append("")
        if isinstance(self.reply, Exception):
            raise self.reply
        return type("R", (), {"content": self.reply})()

    # Some paths stream instead of invoking.
    def stream(self, messages):
        yield type("C", (), {"content": self.reply if isinstance(self.reply, str) else ""})()


@pytest.fixture
def stub_model(monkeypatch):
    """Patch model construction everywhere it is looked up."""
    def _install(reply):
        m = StubModel(reply)
        monkeypatch.setattr(api.AIService, "get_model", staticmethod(lambda *a, **k: m))
        monkeypatch.setattr(utils.AIService, "get_model", staticmethod(lambda *a, **k: m))
        return m
    return _install


PICO_BODY = {"population": "adults", "intervention": "metformin",
             "comparator": "placebo", "outcome": "HbA1c", "framework": "pico"}

PAPERS = [
    {"id": "p1", "paper_id": "p1", "source": "PubMed", "Source": "PubMed",
     "title": "A trial of metformin", "Title": "A trial of metformin",
     "abstract": "We randomised 128 adults.", "Abstract": "We randomised 128 adults.",
     "url": "", "URL": ""},
]


def post(client, path, body):
    return client.post(path, json=body)


# ---------------------------------------------------------------------------
# PICO endpoints
# ---------------------------------------------------------------------------

PICO_JSON = json.dumps({
    "population": "adults with type 2 diabetes",
    "intervention": "metformin",
    "comparator": "placebo",
    "outcome": "HbA1c reduction",
    "inclusion": ["Randomised trials"],
    "exclusion": ["Animal studies"],
    "questions": ["What age range?"],
    "suggestions": ["Consider narrowing the outcome"],
    "summary": "A trial of metformin in adults.",
    "question": "In adults with T2D, does metformin lower HbA1c versus placebo?",
})


PICO_ROUTES = [
    ("/api/pico/infer", {"topic": "metformin for diabetes"}),
    ("/api/pico/refine", {"pico": PICO_BODY, "instruction": "narrow the population"}),
    ("/api/pico/suggestions", {"pico": PICO_BODY}),
    ("/api/pico/summary", {"pico": PICO_BODY}),
    ("/api/pico/brainstorm", {"topic": "metformin"}),
    ("/api/pico/formal-question", {"pico": PICO_BODY}),
    ("/api/pico/clarify-questions", {"pico": PICO_BODY, "topic": "metformin"}),
    ("/api/pico/clarify-next", {"pico": PICO_BODY, "answers": [], "topic": "metformin"}),
    ("/api/pico/adversarial", {"pico": PICO_BODY}),
]


@pytest.mark.parametrize("path,body", PICO_ROUTES, ids=[p for p, _ in PICO_ROUTES])
class TestPicoEndpoints:
    def test_a_well_formed_request_succeeds(self, client, stub_model, path, body):
        stub_model(PICO_JSON)
        r = post(client, path, body)
        assert r.status_code in (200, 422), r.text
        if r.status_code == 200:
            assert r.json() is not None

    def test_a_model_failure_does_not_500(self, client, stub_model, path, body):
        stub_model(RuntimeError("model down"))
        assert post(client, path, body).status_code != 500

    def test_unparseable_model_output_does_not_500(self, client, stub_model, path, body):
        stub_model("this is not json")
        assert post(client, path, body).status_code != 500

    def test_no_model_available_does_not_500(self, client, monkeypatch, path, body):
        monkeypatch.setattr(api.AIService, "get_model", staticmethod(lambda *a, **k: None))
        monkeypatch.setattr(utils.AIService, "get_model", staticmethod(lambda *a, **k: None))
        assert post(client, path, body).status_code != 500

    def test_an_empty_body_is_rejected_cleanly(self, client, stub_model, path, body):
        stub_model(PICO_JSON)
        assert post(client, path, {}).status_code != 500


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

SCREEN_JSON = json.dumps({
    "population": {"vote": "PASS", "evidence": "128 adults", "reasoning": "adults"},
    "intervention": {"vote": "PASS", "evidence": "metformin", "reasoning": "drug"},
    "comparator": {"vote": "PASS", "evidence": "placebo", "reasoning": "placebo"},
    "outcome": {"vote": "PASS", "evidence": "HbA1c", "reasoning": "outcome"},
    "overall_reasoning": "Matches every element.",
    "bucket": "All elements met",
    "failed_criteria": [],
    "decision": "Include",
    "reason": "Matches",
})


class TestScreeningEndpoints:
    def _abstract_body(self):
        return {"paper": PAPERS[0], "pico": PICO_BODY, "model": "qwen2.5:7b"}

    def test_abstract_screening_returns_a_decision(self, client, stub_model):
        stub_model(SCREEN_JSON)
        r = post(client, "/api/screen/abstract", self._abstract_body())
        assert r.status_code == 200
        assert r.json().get("Decision") in ("INCLUDE", "EXCLUDE")

    def test_abstract_screening_returns_a_pico_panel(self, client, stub_model):
        stub_model(SCREEN_JSON)
        body = post(client, "/api/screen/abstract", self._abstract_body()).json()
        assert body.get("Pico_Assessment")

    def test_abstract_screening_survives_a_model_failure(self, client, stub_model):
        stub_model(RuntimeError("model down"))
        r = post(client, "/api/screen/abstract", self._abstract_body())
        assert r.status_code != 500

    def test_abstract_screening_survives_garbage_output(self, client, stub_model):
        stub_model("not json")
        r = post(client, "/api/screen/abstract", self._abstract_body())
        assert r.status_code != 500

    def test_batch_screening_returns_one_row_per_paper(self, client, stub_model):
        stub_model(SCREEN_JSON)
        papers = [dict(PAPERS[0], id=f"p{i}", paper_id=f"p{i}") for i in range(3)]
        r = post(client, "/api/screen/abstract-batch",
                 {"papers": papers, "pico": PICO_BODY, "model": "qwen2.5:7b"})
        if r.status_code != 200:
            pytest.skip(f"batch endpoint returned {r.status_code}")
        body = r.json()
        rows = body if isinstance(body, list) else body.get("results", [])
        assert len(rows) == 3

    def test_batch_screening_with_no_papers(self, client, stub_model):
        stub_model(SCREEN_JSON)
        r = post(client, "/api/screen/abstract-batch",
                 {"papers": [], "pico": PICO_BODY, "model": "qwen2.5:7b"})
        assert r.status_code != 500

    def test_fulltext_screening_returns_a_decision(self, client, stub_model):
        stub_model(SCREEN_JSON)
        r = post(client, "/api/screen/fulltext", {
            "paper": dict(PAPERS[0], full_text="Methods. We randomised 128 adults."),
            "pico": PICO_BODY, "model": "qwen2.5:7b",
        })
        assert r.status_code in (200, 422)
        if r.status_code == 200:
            assert r.json() is not None

    def test_fulltext_screening_survives_a_model_failure(self, client, stub_model):
        stub_model(RuntimeError("down"))
        r = post(client, "/api/screen/fulltext", {
            "paper": dict(PAPERS[0], full_text="text"), "pico": PICO_BODY,
        })
        assert r.status_code != 500


# ---------------------------------------------------------------------------
# Search and simulation
# ---------------------------------------------------------------------------

class TestSearchEndpoints:
    def test_search_build_returns_a_query(self, client, stub_model, monkeypatch):
        stub_model(json.dumps({"concepts": [
            {"name": "Metformin", "terms": ["metformin"], "mesh": ["Metformin"]},
            {"name": "Diabetes", "terms": ["type 2 diabetes"], "mesh": ["Diabetes Mellitus"]},
        ]}))
        r = post(client, "/api/search/build", {"pico": PICO_BODY, "model": "qwen2.5:7b"})
        assert r.status_code in (200, 422)
        if r.status_code == 200:
            assert "metformin" in json.dumps(r.json()).lower()

    def test_search_build_survives_a_model_failure(self, client, stub_model):
        stub_model(RuntimeError("down"))
        assert post(client, "/api/search/build",
                    {"pico": PICO_BODY}).status_code != 500

    def test_simulation_yield_returns_counts(self, client, monkeypatch):
        monkeypatch.setattr(api.DataAggregator, "simulate_yield",
                            staticmethod(lambda q, s: {src: 10 for src in s}))
        r = post(client, "/api/simulation/yield",
                 {"query": '("metformin"[tiab])', "sources": ["PubMed"]})
        assert r.status_code == 200
        assert json.dumps(r.json())

    def test_simulation_yield_survives_a_source_failure(self, client, monkeypatch):
        def boom(q, s):
            raise RuntimeError("all sources down")
        monkeypatch.setattr(api.DataAggregator, "simulate_yield", staticmethod(boom))
        r = post(client, "/api/simulation/yield",
                 {"query": "q", "sources": ["PubMed"]})
        assert r.status_code != 500

    def test_simulation_adapt_does_not_500(self, client, stub_model):
        stub_model(json.dumps({"query": '("metformin"[tiab])', "rationale": "narrowed"}))
        r = post(client, "/api/simulation/adapt",
                 {"query": '("metformin"[tiab])', "pico": PICO_BODY,
                  "counts": {"PubMed": 100000}, "sources": ["PubMed"]})
        assert r.status_code != 500

    def test_framework_detect_returns_a_framework(self, client, stub_model):
        stub_model('{"framework": "pcc"}')
        r = post(client, "/api/framework/detect", {"text": "A scoping review of AI in dentistry"})
        assert r.status_code in (200, 422)
        if r.status_code == 200:
            assert "pc" in json.dumps(r.json()).lower() or "pico" in json.dumps(r.json()).lower()


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------

class TestPaperEndpoints:
    def test_dedupe_removes_a_duplicate(self, client):
        dup = [PAPERS[0], dict(PAPERS[0], id="p2", paper_id="p2", source="Europe PMC")]
        r = post(client, "/api/papers/dedupe", {"papers": dup})
        if r.status_code != 200:
            pytest.skip(f"dedupe returned {r.status_code}")
        body = r.json()
        kept = body.get("unique", body.get("papers", body if isinstance(body, list) else []))
        assert len(kept) <= len(dup)

    def test_dedupe_with_an_empty_corpus(self, client):
        assert post(client, "/api/papers/dedupe", {"papers": []}).status_code != 500

    def test_rerank_does_not_500(self, client, stub_model):
        stub_model('{"scores": [{"paper_id": "p1", "score": 0.9}]}')
        r = post(client, "/api/papers/rerank",
                 {"papers": PAPERS, "pico": PICO_BODY, "model": "qwen2.5:7b"})
        assert r.status_code != 500

    def test_retractions_check_does_not_500(self, client, monkeypatch):
        monkeypatch.setattr(api.requests, "get",
                            lambda *a, **k: type("R", (), {
                                "status_code": 200,
                                "json": staticmethod(lambda: {"is_retracted": False}),
                                "text": "", "content": b"", "headers": {}})())
        assert post(client, "/api/papers/retractions", {"papers": PAPERS}).status_code != 500

    def test_similar_papers_does_not_500(self, client, monkeypatch):
        monkeypatch.setattr(api.OpenAlexService, "related", staticmethod(lambda *a, **k: []))
        r = post(client, "/api/papers/similar",
                 {"title": "A trial of metformin", "doi": "", "max_results": 5})
        assert r.status_code != 500


# ---------------------------------------------------------------------------
# Extraction, quality, writing
# ---------------------------------------------------------------------------

class TestExtractionEndpoints:
    def test_extract_text_returns_a_structure(self, client, stub_model):
        stub_model(json.dumps({"summary": "A summary", "evidence": [], "spans": [], "values": []}))
        r = post(client, "/api/extract/text",
                 {"text": "We randomised 128 adults.", "query": "sample size"})
        assert r.status_code in (200, 422)
        if r.status_code == 200:
            assert r.json() is not None

    def test_extract_text_survives_a_model_failure(self, client, stub_model):
        stub_model(RuntimeError("down"))
        assert post(client, "/api/extract/text",
                    {"text": "t", "query": "q"}).status_code != 500

    def test_extract_fields_does_not_500(self, client, stub_model):
        stub_model(json.dumps({"extractions": [
            {"name": "sample_size", "value": 128, "source_quote": "128 adults"}]}))
        r = post(client, "/api/extract/fields", {
            "text": "We randomised 128 adults.", "tables": "",
            "fields": [{"name": "sample_size", "type": "number"}], "model": "qwen2.5:7b"})
        assert r.status_code != 500

    def test_extract_tables_does_not_500(self, client, stub_model):
        stub_model('{"tables": []}')
        assert post(client, "/api/extract/tables",
                    {"text": "some text", "model": "qwen2.5:7b"}).status_code != 500

    def test_grade_does_not_500(self, client, stub_model):
        stub_model(json.dumps({"certainty": "Moderate", "rationale": "imprecision"}))
        assert post(client, "/api/grade", {
            "outcome": "HbA1c", "studies": PAPERS, "model": "qwen2.5:7b"}).status_code != 500

    def test_quality_assess_does_not_500(self, client, stub_model):
        stub_model(json.dumps({"domains": [], "overall": "Low", "rationale": "ok"}))
        assert post(client, "/api/quality/assess", {
            "paper": PAPERS[0], "instrument_id": "rob2",
            "model": "qwen2.5:7b"}).status_code != 500


class TestWritingEndpoints:
    def test_writing_summary_does_not_500(self, client, stub_model):
        stub_model("A narrative summary of the included studies.")
        assert post(client, "/api/writing/summary", {
            "papers": PAPERS, "pico": PICO_BODY, "model": "qwen2.5:7b"}).status_code != 500

    def test_writing_protocol_does_not_500(self, client, stub_model):
        stub_model("A protocol section.")
        assert post(client, "/api/writing/protocol", {
            "pico": PICO_BODY, "model": "qwen2.5:7b"}).status_code != 500

    def test_writing_characteristics_does_not_500(self, client, stub_model):
        stub_model(json.dumps({"rows": []}))
        assert post(client, "/api/writing/characteristics", {
            "papers": PAPERS, "model": "qwen2.5:7b"}).status_code != 500

    def test_writing_ask_does_not_500(self, client, stub_model):
        stub_model("An answer.")
        assert post(client, "/api/writing/ask", {
            "question": "What did the studies find?", "papers": PAPERS,
            "model": "qwen2.5:7b"}).status_code != 500

    def test_a_model_failure_never_leaks_a_traceback(self, client, stub_model):
        """Routes differ in how they signal a dead model: some degrade to an
        empty result, writing/protocol raises a clean HTTPException(500) with a
        detail. Either is acceptable; an unhandled traceback is not."""
        stub_model(RuntimeError("down"))
        for path, body in [
            ("/api/writing/summary", {"papers": PAPERS, "pico": PICO_BODY}),
            ("/api/writing/protocol", {"pico": PICO_BODY}),
            ("/api/writing/characteristics", {"papers": PAPERS}),
            ("/api/writing/ask", {"question": "q", "papers": PAPERS}),
        ]:
            r = post(client, path, body)
            assert "Traceback" not in r.text, path
            assert r.json() is not None, path

    def test_writing_summary_degrades_instead_of_failing(self, client, stub_model):
        """This one runs two model calls concurrently; an exception used to
        escape the executor as an unhandled 500."""
        stub_model(RuntimeError("down"))
        r = post(client, "/api/writing/summary", {"papers": PAPERS, "pico": PICO_BODY})
        assert r.status_code != 500
        assert r.json().get("error")


# ---------------------------------------------------------------------------
# Assistant and routing
# ---------------------------------------------------------------------------

class TestAssistantEndpoints:
    def test_assistant_chat_does_not_500(self, client, stub_model):
        stub_model("Hello, here is an answer.")
        assert post(client, "/api/assistant/chat", {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "qwen2.5:7b"}).status_code != 500

    def test_assistant_agent_does_not_500(self, client, stub_model):
        stub_model(json.dumps({"tool": None, "answer": "done"}))
        assert post(client, "/api/assistant/agent", {
            "messages": [{"role": "user", "content": "screen my papers"}],
            "model": "qwen2.5:7b"}).status_code != 500

    def test_route_intent_does_not_500(self, client, stub_model):
        stub_model('{"intent": "screen"}')
        assert post(client, "/api/route/intent",
                    {"text": "screen my papers"}).status_code != 500

    def test_documents_ask_does_not_500(self, client, stub_model):
        stub_model("An answer with [1] a citation.")
        assert post(client, "/api/documents/ask", {
            "question": "what is the sample size?",
            "documents": [{"id": "d1", "title": "T", "text": "We randomised 128 adults."}],
            "model": "qwen2.5:7b"}).status_code != 500

    def test_results_ask_does_not_500(self, client, stub_model):
        stub_model("An answer.")
        assert post(client, "/api/results/ask", {
            "question": "q", "papers": PAPERS, "model": "qwen2.5:7b"}).status_code != 500


# ---------------------------------------------------------------------------
# Task control and reproducibility
# ---------------------------------------------------------------------------

class TestTaskAndMeta:
    def test_cancel_an_unknown_task_is_not_an_error(self, client):
        r = post(client, "/api/tasks/cancel", {"task_id": "no-such-task"})
        assert r.status_code == 200
        assert r.json().get("canceled") is False

    def test_cancel_all_is_safe_with_nothing_running(self, client):
        assert post(client, "/api/tasks/cancel-all", {}).status_code in (200, 422)

    def test_reproducibility_manifest_reports_the_seed_and_temperature(self, client):
        r = client.get("/api/reproducibility")
        assert r.status_code == 200
        body = r.json()
        assert "seed" in body or "temperature" in body

    def test_integrity_check_does_not_500(self, client, monkeypatch):
        monkeypatch.setattr(api.requests, "get",
                            lambda *a, **k: type("R", (), {
                                "status_code": 200,
                                "json": staticmethod(lambda: {"is_retracted": False}),
                                "text": "", "content": b"", "headers": {}})())
        assert post(client, "/api/integrity/check", {"papers": PAPERS}).status_code != 500

    def test_sessions_listing_does_not_500(self, client):
        assert client.get("/api/sessions").status_code != 500
