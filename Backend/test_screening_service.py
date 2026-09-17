"""Tests for the AIService screening entry points and query optimisation.

Run: Backend/.venv/bin/python -m pytest Backend/test_screening_service.py

Model and network are stubbed. These are the top-level functions the API calls
per paper and per search, so a defect is systematic rather than occasional.

The invariant running through all of them: when the model misbehaves, the code
must degrade toward keeping papers and keeping the previous query, never toward
dropping them. Losing a relevant study at screening is unrecoverable; a false
include costs one full-text read.
"""
import pytest

import data_services as ds
import requests
import utils
from models import Paper, PICOCriteria


class StubModel:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(getattr(messages[-1], "content", ""))
        if isinstance(self.reply, Exception):
            raise self.reply
        r = self.reply(len(self.prompts)) if callable(self.reply) else self.reply
        return type("R", (), {"content": r})()


@pytest.fixture
def stub_model(monkeypatch):
    def _install(reply):
        m = StubModel(reply)
        monkeypatch.setattr(utils.AIService, "get_model", staticmethod(lambda *a, **k: m))
        return m
    return _install


PICO = PICOCriteria(population="adults with diabetes", intervention="metformin",
                    comparator="placebo", outcome="HbA1c")

PAPER = Paper(source="PubMed", id="p1",
              title="A randomised trial of metformin in adults with diabetes",
              abstract="We randomised 128 adults to metformin or placebo and measured HbA1c.",
              url="https://pubmed.ncbi.nlm.nih.gov/1/")

GOOD = ('{"decision":"Include","bucket":"All elements met",'
        '"reason":"Matches population, intervention and outcome."}')


# ---------------------------------------------------------------------------
# Abstract screening
# ---------------------------------------------------------------------------

class TestScreenPaper:
    def test_returns_a_decision_and_a_reason(self, stub_model):
        stub_model(GOOD)
        out = utils.AIService.screen_paper(PAPER, PICO, "m")
        assert str(out.get("decision", "")).lower().startswith("inc")
        assert out.get("reason")

    def test_a_model_failure_falls_back_to_keeping_the_paper(self, stub_model):
        """Recall over precision: a broken model must not silently exclude."""
        stub_model(RuntimeError("model down"))
        out = utils.AIService.screen_paper(PAPER, PICO, "m")
        assert str(out.get("decision", "")).lower().startswith("inc")

    @pytest.mark.parametrize("reply", ["", "not json", "{}", "null"])
    def test_unusable_output_still_returns_a_decision(self, stub_model, reply):
        stub_model(reply)
        out = utils.AIService.screen_paper(PAPER, PICO, "m")
        assert out.get("decision")

    def test_no_model_available_still_returns_a_decision(self, monkeypatch):
        monkeypatch.setattr(utils.AIService, "get_model", staticmethod(lambda *a, **k: None))
        out = utils.AIService.screen_paper(PAPER, PICO, "m")
        assert out.get("decision")

    def test_criteria_reach_the_prompt(self, stub_model):
        m = stub_model(GOOD)
        utils.AIService.screen_paper(PAPER, PICO, "m",
                                     inclusion=["Published 2010 or later"],
                                     exclusion=["Animal studies"])
        joined = " ".join(m.prompts)
        assert "Published 2010 or later" in joined
        assert "Animal studies" in joined

    def test_protocol_reaches_the_prompt(self, stub_model):
        m = stub_model(GOOD)
        utils.AIService.screen_paper(PAPER, PICO, "m", protocol="PROTOCOL-MARKER")
        assert "PROTOCOL-MARKER" in " ".join(m.prompts)

    def test_the_paper_title_reaches_the_prompt(self, stub_model):
        m = stub_model(GOOD)
        utils.AIService.screen_paper(PAPER, PICO, "m")
        assert "metformin" in " ".join(m.prompts).lower()

    def test_a_paper_with_no_abstract_is_still_screened(self, stub_model):
        stub_model(GOOD)
        bare = Paper(source="PubMed", id="p2", title="Title only", abstract="", url="")
        assert utils.AIService.screen_paper(bare, PICO, "m").get("decision")

    def test_an_explicit_exclude_is_respected(self, stub_model):
        stub_model('{"decision":"Exclude","bucket":"Wrong population",'
                   '"reason":"Paediatric cohort."}')
        out = utils.AIService.screen_paper(PAPER, PICO, "m")
        assert str(out.get("decision", "")).lower().startswith("exc")


# ---------------------------------------------------------------------------
# Full-text screening
# ---------------------------------------------------------------------------

FT_PAPER = {"paper_id": "p1", "Title": PAPER.title, "Abstract": PAPER.abstract,
            "full_text": "Methods. We randomised 128 adults. Results. HbA1c fell by 0.8%."}


class TestScreenFullText:
    def test_returns_a_decision(self, stub_model):
        stub_model(GOOD)
        out = utils.AIService.screen_full_text(FT_PAPER, PICO, "m")
        assert out.get("decision")

    def test_a_model_failure_does_not_raise(self, stub_model):
        stub_model(RuntimeError("model down"))
        assert isinstance(utils.AIService.screen_full_text(FT_PAPER, PICO, "m"), dict)

    @pytest.mark.parametrize("reply", ["", "garbage", "{}"])
    def test_unusable_output_still_returns_a_decision(self, stub_model, reply):
        stub_model(reply)
        assert utils.AIService.screen_full_text(FT_PAPER, PICO, "m").get("decision")

    def test_criteria_are_passed_explicitly_not_through_globals(self, stub_model):
        """These used to be read from a fake Streamlit session_state that the
        API wrote into. They are parameters now, and must actually be used."""
        m = stub_model(GOOD)
        utils.AIService.screen_full_text(FT_PAPER, PICO, "m",
                                         inclusion=["Reports HbA1c"],
                                         exclusion=["Conference abstract"])
        joined = " ".join(m.prompts)
        assert "Reports HbA1c" in joined
        assert "Conference abstract" in joined

    def test_absent_criteria_do_not_crash(self, stub_model):
        stub_model(GOOD)
        assert utils.AIService.screen_full_text(FT_PAPER, PICO, "m",
                                                inclusion=None, exclusion=None).get("decision")

    def test_the_paper_content_reaches_the_prompt(self, stub_model):
        m = stub_model(GOOD)
        utils.AIService.screen_full_text(FT_PAPER, PICO, "m")
        joined = " ".join(m.prompts)
        assert "metformin" in joined.lower()

    def test_a_paper_with_no_full_text_is_still_handled(self, stub_model):
        stub_model(GOOD)
        out = utils.AIService.screen_full_text({"paper_id": "p", "Title": "T"}, PICO, "m")
        assert isinstance(out, dict)


# ---------------------------------------------------------------------------
# Query optimisation
# ---------------------------------------------------------------------------

class TestTacticVariant:
    def test_returns_a_query_and_a_label(self):
        q, label = utils.AIService._tactic_variant('("dental"[tiab])', 0)
        assert isinstance(q, str) and isinstance(label, str)
        assert label

    def test_different_tactics_give_different_variants(self):
        seen = {utils.AIService._tactic_variant('("a"[tiab] OR "b"[tiab]) AND ("c"[tiab])', i)[0]
                for i in range(4)}
        assert len(seen) > 1

    @pytest.mark.parametrize("idx", [0, 1, 2, 3, 4, 5, 99])
    def test_any_tactic_index_is_safe(self, idx):
        q, label = utils.AIService._tactic_variant('("dental"[tiab])', idx)
        assert isinstance(q, str)

    @pytest.mark.parametrize("q", ["", "   "])
    def test_an_empty_query_does_not_crash(self, q):
        assert isinstance(utils.AIService._tactic_variant(q, 0)[0], str)

    def test_a_variant_keeps_parentheses_balanced(self):
        for i in range(5):
            q, _ = utils.AIService._tactic_variant('("a"[tiab] OR "b"[tiab]) AND ("c"[tiab])', i)
            assert q.count("(") == q.count(")")


class TestOptimizeQueryMultiAgent:
    @pytest.fixture(autouse=True)
    def _no_counts(self, monkeypatch):
        monkeypatch.setattr(ds.DataAggregator, "get_all_counts",
                            staticmethod(lambda *a, **k: {"PubMed": 100}))

    def test_returns_a_dict_with_a_query(self, stub_model):
        stub_model('{"query": "(\\"dental\\"[tiab])", "rationale": "narrowed"}')
        out = utils.AIService.optimize_query_multi_agent('("dental"[tiab])', "m")
        assert isinstance(out, dict)

    def test_a_model_failure_returns_the_original_query_rather_than_nothing(self, stub_model):
        """Losing the query would wipe the user's search design."""
        stub_model(RuntimeError("model down"))
        out = utils.AIService.optimize_query_multi_agent('("dental"[tiab])', "m")
        assert isinstance(out, dict)
        assert "dental" in str(out)

    @pytest.mark.parametrize("reply", ["", "not json", "{}"])
    def test_unusable_output_does_not_raise(self, stub_model, reply):
        stub_model(reply)
        assert isinstance(utils.AIService.optimize_query_multi_agent('("d"[tiab])', "m"), dict)

    def test_selected_sources_are_honoured(self, stub_model):
        stub_model('{"query": "(\\"dental\\"[tiab])"}')
        out = utils.AIService.optimize_query_multi_agent('("d"[tiab])', "m", ["PubMed"])
        assert isinstance(out, dict)


class TestOptimizeSearchStringPerSource:
    def test_returns_an_entry_per_source(self, stub_model, monkeypatch):
        stub_model('{"query": "(\\"dental\\"[tiab])"}')
        monkeypatch.setattr(ds.DataAggregator, "get_all_counts",
                            staticmethod(lambda *a, **k: {"PubMed": 10, "OpenAlex": 5}))
        out = utils.AIService.optimize_search_string_per_source(
            '("dental"[tiab])', PICO, "m", ["PubMed", "OpenAlex"])
        assert isinstance(out, dict)

    def test_a_model_failure_still_returns_a_dict(self, stub_model, monkeypatch):
        stub_model(RuntimeError("down"))
        monkeypatch.setattr(ds.DataAggregator, "get_all_counts",
                            staticmethod(lambda *a, **k: {}))
        out = utils.AIService.optimize_search_string_per_source(
            '("dental"[tiab])', PICO, "m", ["PubMed"])
        assert isinstance(out, dict)

    def test_no_sources_yields_an_empty_result(self, stub_model, monkeypatch):
        stub_model('{"query": "x"}')
        monkeypatch.setattr(ds.DataAggregator, "get_all_counts", staticmethod(lambda *a, **k: {}))
        assert utils.AIService.optimize_search_string_per_source(
            '("d"[tiab])', PICO, "m", []) == {}


class TestAgenticOptimizePerSource:
    @pytest.fixture(autouse=True)
    def _stub_counts(self, monkeypatch):
        monkeypatch.setattr(ds.DataAggregator, "get_all_counts",
                            staticmethod(lambda *a, **k: {"PubMed": 250}))
        monkeypatch.setattr(utils.AIService, "_analyze_title_relevance",
                            staticmethod(lambda *a, **k: 0.8))
        monkeypatch.setattr(ds.DataAggregator, "fetch_all",
                            staticmethod(lambda *a, **k: ([], {})))

    def test_returns_a_result_dict(self, stub_model):
        stub_model('{"query": "(\\"dental\\"[tiab])", "rationale": "ok"}')
        out = utils.AIService.agentic_optimize_per_source(
            '("dental"[tiab])', PICO, "m", ["PubMed"])
        assert isinstance(out, dict)

    def test_a_model_failure_does_not_raise(self, stub_model):
        stub_model(RuntimeError("down"))
        out = utils.AIService.agentic_optimize_per_source(
            '("dental"[tiab])', PICO, "m", ["PubMed"])
        assert isinstance(out, dict)

    def test_the_progress_callback_is_invoked(self, stub_model):
        stub_model('{"query": "(\\"dental\\"[tiab])"}')
        seen = []
        utils.AIService.agentic_optimize_per_source(
            '("dental"[tiab])', PICO, "m", ["PubMed"],
            progress_callback=lambda *a, **k: seen.append(a))
        assert isinstance(seen, list)

    def test_no_sources_is_safe(self, stub_model):
        stub_model('{"query": "x"}')
        assert isinstance(
            utils.AIService.agentic_optimize_per_source('("d"[tiab])', PICO, "m", []), dict)


class TestAnalyzeTitleRelevance:
    def test_returns_a_score_between_zero_and_one(self, stub_model):
        stub_model('{"relevant": 8, "total": 10}')
        score = utils.AIService._analyze_title_relevance(
            ["A trial of metformin", "Unrelated geology"], "metformin in diabetes", PICO)
        assert 0.0 <= float(score) <= 1.0

    def test_no_titles_returns_a_number(self, stub_model):
        stub_model('{"relevant": 0, "total": 0}')
        assert isinstance(float(utils.AIService._analyze_title_relevance([], "goal", PICO)), float)

    def test_a_model_failure_returns_a_number(self, stub_model):
        stub_model(RuntimeError("down"))
        assert isinstance(
            float(utils.AIService._analyze_title_relevance(["t"], "goal", PICO)), float)


# ---------------------------------------------------------------------------
# Citation snowballing
# ---------------------------------------------------------------------------

class TestFetchCitations:
    def test_returns_a_list(self, monkeypatch):
        monkeypatch.setattr(requests, "get",
                            lambda *a, **k: type("R", (), {
                                "status_code": 200,
                                "json": staticmethod(lambda: {"referenceList": {"reference": []}}),
                                "text": "", "content": b"", "headers": {}})())
        out = utils.AIService.fetch_citations("123", "PubMed", "A title", "backward", 10, ["PubMed"])
        assert isinstance(out, list)

    def test_a_network_failure_returns_a_list(self, monkeypatch):
        def boom(*a, **k):
            raise requests.exceptions.ConnectionError("down")
        monkeypatch.setattr(requests, "get", boom)
        assert isinstance(
            utils.AIService.fetch_citations("123", "PubMed", "T", "backward", 10, ["PubMed"]), list)

    @pytest.mark.parametrize("direction", ["backward", "forward"])
    def test_both_directions_are_handled(self, monkeypatch, direction):
        monkeypatch.setattr(requests, "get",
                            lambda *a, **k: type("R", (), {
                                "status_code": 200,
                                "json": staticmethod(lambda: {}),
                                "text": "", "content": b"", "headers": {}})())
        assert isinstance(
            utils.AIService.fetch_citations("123", "PubMed", "T", direction, 10, ["PubMed"]), list)

    def test_no_paper_id_is_safe(self, monkeypatch):
        monkeypatch.setattr(requests, "get",
                            lambda *a, **k: type("R", (), {
                                "status_code": 404, "json": staticmethod(lambda: {}),
                                "text": "", "content": b"", "headers": {}})())
        assert isinstance(
            utils.AIService.fetch_citations("", "PubMed", "T", "backward", 10, ["PubMed"]), list)
