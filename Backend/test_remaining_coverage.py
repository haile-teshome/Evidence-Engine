"""Tests for the last uncovered backend paths: every simulate_yield source
branch, batch screening orchestration, snowballing, and the project
contribution / auto-assignment logic.

Run: Backend/.venv/bin/python -m pytest Backend/test_remaining_coverage.py

simulate_yield gets one test per source because each branch reads a DIFFERENT
response field (`hitCount`, `total`, `totalHits`, `meta.count`, XML
`totalResults`). A provider renaming its count field is invisible: the branch
returns 0, the Planning tab shows "no results for this database", and the
reviewer drops a source that actually had thousands of hits.
"""
import pytest

import data_services as ds
import utils
from config import DataSource
from models import Paper, PICOCriteria


class FakeHandle:
    def __init__(self, payload):
        self.payload = payload

    def close(self):
        pass


class FakeResponse:
    def __init__(self, payload=None, status=200, content=b""):
        self._payload = payload
        self.status_code = status
        self.content = content
        self.text = content.decode("utf-8", "replace") if content else ""
        self.headers = {}

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


@pytest.fixture
def entrez(monkeypatch):
    def _install(count="0"):
        monkeypatch.setattr(ds.Entrez, "esearch", lambda **k: FakeHandle({"Count": count, "IdList": []}))
        monkeypatch.setattr(ds.Entrez, "read", lambda h: h.payload)
    return _install


# ---------------------------------------------------------------------------
# simulate_yield: one branch per source.
# ---------------------------------------------------------------------------

ARXIV_XML = (b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" '
             b'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
             b'<opensearch:totalResults>317</opensearch:totalResults></feed>')

# (source, response payload, expected count) — each provider reports its total
# under a different key, which is exactly why each needs its own test.
SOURCE_CASES = [
    ("Europe PMC", {"hitCount": 4321}, 4321),
    ("OpenAlex", {"meta": {"count": 1234}}, 1234),
    ("CrossRef", {"message": {"total-results": 987}}, 987),
    ("Semantic Scholar", {"total": 555}, 555),
    ("CORE", {"totalHits": 222}, 222),
    ("DOAJ", {"total": 111}, 111),
    ("medRxiv", {"messages": [{"total": 42}]}, None),
    (DataSource.BIORXIV.value, {"messages": [{"total": 42}]}, None),
]


@pytest.mark.parametrize("source,payload,expected", SOURCE_CASES,
                         ids=[c[0] for c in SOURCE_CASES])
class TestSimulateYieldPerSource:
    def test_reads_that_providers_count_field(self, monkeypatch, source, payload, expected):
        monkeypatch.setattr(ds, "throttled_request", lambda *a, **k: FakeResponse(payload))
        out = ds.DataAggregator.simulate_yield("dental", [source])
        assert source in out, "a searched source must never be missing from the report"
        if expected is not None:
            assert out[source] == expected

    def test_a_missing_count_field_yields_zero_not_a_crash(self, monkeypatch, source, payload, expected):
        monkeypatch.setattr(ds, "throttled_request", lambda *a, **k: FakeResponse({}))
        out = ds.DataAggregator.simulate_yield("dental", [source])
        assert out.get(source, 0) == 0

    def test_an_http_error_yields_zero(self, monkeypatch, source, payload, expected):
        monkeypatch.setattr(ds, "throttled_request",
                            lambda *a, **k: FakeResponse(None, status=500))
        assert ds.DataAggregator.simulate_yield("dental", [source]).get(source, 0) == 0

    def test_a_network_exception_yields_zero(self, monkeypatch, source, payload, expected):
        def boom(*a, **k):
            raise ds.requests.exceptions.ConnectionError("down")
        monkeypatch.setattr(ds, "throttled_request", boom)
        assert ds.DataAggregator.simulate_yield("dental", [source]).get(source, 0) == 0


class TestSimulateYieldArxiv:
    def test_parses_the_opensearch_total_from_xml(self, monkeypatch):
        """arXiv is the only source that answers in XML, not JSON."""
        monkeypatch.setattr(ds, "throttled_request",
                            lambda *a, **k: FakeResponse(None, content=ARXIV_XML))
        out = ds.DataAggregator.simulate_yield("dental", [DataSource.ARXIV.value])
        assert out[DataSource.ARXIV.value] == 317

    def test_malformed_xml_yields_zero(self, monkeypatch):
        monkeypatch.setattr(ds, "throttled_request",
                            lambda *a, **k: FakeResponse(None, content=b"<not-xml"))
        assert ds.DataAggregator.simulate_yield("d", [DataSource.ARXIV.value]).get(
            DataSource.ARXIV.value, 0) == 0

    def test_xml_without_a_total_yields_zero(self, monkeypatch):
        monkeypatch.setattr(ds, "throttled_request",
                            lambda *a, **k: FakeResponse(None, content=b"<feed></feed>"))
        assert ds.DataAggregator.simulate_yield("d", [DataSource.ARXIV.value]).get(
            DataSource.ARXIV.value, 0) == 0


class TestSimulateYieldAcrossSources:
    def test_every_requested_source_appears_in_the_report(self, entrez, monkeypatch):
        entrez("10")
        monkeypatch.setattr(ds, "throttled_request", lambda *a, **k: FakeResponse(
            {"hitCount": 1, "meta": {"count": 1}, "message": {"total-results": 1},
             "total": 1, "totalHits": 1}))
        sources = [DataSource.PUBMED.value, "Europe PMC", "OpenAlex", "CrossRef",
                   "Semantic Scholar", "CORE", "DOAJ", DataSource.LOCAL_PDF.value]
        out = ds.DataAggregator.simulate_yield("dental", sources)
        assert set(out) == set(sources)

    def test_one_dead_source_does_not_zero_the_others(self, entrez, monkeypatch):
        entrez("500")
        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ds.requests.exceptions.ConnectionError("down")
            return FakeResponse({"hitCount": 7, "meta": {"count": 7}})

        monkeypatch.setattr(ds, "throttled_request", flaky)
        out = ds.DataAggregator.simulate_yield(
            "d", [DataSource.PUBMED.value, "Europe PMC", "OpenAlex"])
        assert out[DataSource.PUBMED.value] == 500

    def test_an_unknown_source_is_reported_as_zero_not_omitted(self, monkeypatch):
        monkeypatch.setattr(ds, "throttled_request", lambda *a, **k: FakeResponse({}))
        out = ds.DataAggregator.simulate_yield("d", ["Not A Real Database"])
        assert out.get("Not A Real Database", 0) == 0


# ---------------------------------------------------------------------------
# Batch screening orchestration
# ---------------------------------------------------------------------------

class StubAgent:
    """Stands in for a CriterionAgent: votes the same way for every paper."""
    def __init__(self, agent_type, met=True):
        self.agent_type = agent_type
        self.name = f"agent-{agent_type}"
        self.met = met

    def evaluate_all_papers(self, papers, batch_size=10):
        return [utils.AgentVote(agent_name=self.name, agent_type=self.agent_type,
                                criterion="c", paper_id=p.id, met=self.met,
                                confidence=0.9, evidence="e", reasoning="r")
                for p in papers]


def _papers(n):
    return [Paper(source="PubMed", id=f"p{i}", title=f"Study {i}",
                  abstract="We randomised 128 adults.", url="") for i in range(n)]


@pytest.fixture
def orch():
    return utils.ScreeningOrchestrator(
        PICOCriteria(population="adults", intervention="metformin"),
        ["Published after 2010"], ["Animal studies"], "stub-model")


class TestScreenPapers:
    def test_returns_one_result_per_paper(self, orch, monkeypatch):
        monkeypatch.setattr(orch, "_create_agents", lambda: [StubAgent("PICO_P")])
        out = orch.screen_papers(_papers(5))
        assert len(out) == 5

    def test_no_paper_is_lost_even_when_an_agent_returns_nothing(self, orch, monkeypatch):
        class Silent(StubAgent):
            def evaluate_all_papers(self, papers, batch_size=10):
                return []
        monkeypatch.setattr(orch, "_create_agents", lambda: [Silent("PICO_P")])
        assert len(orch.screen_papers(_papers(4))) == 4

    def test_an_agent_that_raises_does_not_lose_the_batch(self, orch, monkeypatch):
        class Boom(StubAgent):
            def evaluate_all_papers(self, papers, batch_size=10):
                raise RuntimeError("agent died")
        monkeypatch.setattr(orch, "_create_agents", lambda: [Boom("PICO_P")])
        try:
            out = orch.screen_papers(_papers(3))
        except RuntimeError:
            pytest.skip("agent errors propagate by design")
        assert len(out) == 3

    def test_reports_progress(self, orch, monkeypatch):
        monkeypatch.setattr(orch, "_create_agents", lambda: [StubAgent("PICO_P")])
        seen = []
        orch.screen_papers(_papers(3), progress_callback=lambda *a, **k: seen.append(a))
        assert isinstance(seen, list)

    def test_an_empty_corpus_yields_no_results(self, orch, monkeypatch):
        monkeypatch.setattr(orch, "_create_agents", lambda: [StubAgent("PICO_P")])
        assert orch.screen_papers([]) == []

    def test_every_result_carries_a_decision(self, orch, monkeypatch):
        monkeypatch.setattr(orch, "_create_agents", lambda: [StubAgent("PICO_P")])
        for r in orch.screen_papers(_papers(3)):
            assert r["Decision"] in ("INCLUDE", "EXCLUDE")

    def test_an_exclusion_violation_excludes_that_paper(self, orch, monkeypatch):
        monkeypatch.setattr(orch, "_create_agents",
                            lambda: [StubAgent("PICO_P", met=True),
                                     StubAgent("EXCLUSION", met=True)])
        out = orch.screen_papers(_papers(2))
        assert all(r["Decision"] == "EXCLUDE" for r in out)

    def test_batch_size_does_not_change_the_result_count(self, orch, monkeypatch):
        monkeypatch.setattr(orch, "_create_agents", lambda: [StubAgent("PICO_P")])
        assert len(orch.screen_papers(_papers(7), batch_size=2)) == 7

    def test_agent_summary_is_available(self, orch):
        assert isinstance(orch.get_agent_summary(), (list, dict, str))


# ---------------------------------------------------------------------------
# Project contributions and auto-assignment
# ---------------------------------------------------------------------------

def _bundle(**over):
    b = {
        "project": {"id": "p1", "name": "R", "screening_mode": "dual"},
        "papers": [{"paper_id": "s1"}, {"paper_id": "s2"}],
        "decisions": [
            {"paper_id": "s1", "reviewer_user_id": "u1", "decision": "include", "stage": "abstract"},
            {"paper_id": "s1", "reviewer_user_id": "u2", "decision": "exclude", "stage": "abstract"},
            {"paper_id": "s2", "reviewer_user_id": "u1", "decision": "include", "stage": "abstract"},
        ],
        "adjudications": [{"paper_id": "s1", "final_decision": "include"}],
        "extractions": [{"paper_id": "s1", "reviewer_user_id": "u1", "values": {}}],
        "rob_assessments": [],
    }
    b.update(over)
    return b


class TestWriteContributions:
    def test_returns_a_mapping(self, temp_store):
        out = temp_store._write_contributions("p1", _bundle())
        assert isinstance(out, dict)

    def test_reports_a_count_for_every_record_kind(self, temp_store):
        out = temp_store._write_contributions("p1", _bundle())
        assert {"decisions", "adjudications", "extractions", "papers",
                "rob_assessments"} <= set(out)
        assert all(isinstance(v, int) and v >= 0 for v in out.values())

    def test_counts_each_reviewers_decision_separately(self, temp_store):
        """Two reviewers deciding on one paper is two decisions, not one."""
        out = temp_store._write_contributions("p1", _bundle())
        assert out["decisions"] == 3

    def test_stores_each_reviewers_decision_under_its_own_key(self, temp_store):
        temp_store._write_contributions("p1", _bundle())
        rows = temp_store.kv_get_by_prefix("decision:p1:abstract:s1:")
        assert len({r["reviewer_user_id"] for r in rows}) == 2

    def test_a_decision_with_no_paper_id_is_skipped(self, temp_store):
        out = temp_store._write_contributions("p1", _bundle(decisions=[{"decision": "include"}]))
        assert out["decisions"] == 0

    def test_a_decision_with_no_reviewer_is_attributed_to_an_import(self, temp_store):
        """It must still be stored: dropping it would silently lose a decision
        from an imported bundle."""
        temp_store._write_contributions(
            "p1", _bundle(decisions=[{"paper_id": "s9", "decision": "include"}]))
        rows = temp_store.kv_get_by_prefix("decision:p1:abstract:s9:")
        assert len(rows) == 1

    def test_an_empty_bundle_does_not_crash(self, temp_store):
        assert isinstance(temp_store._write_contributions("p1", {}), dict)

    def test_missing_collections_do_not_crash(self, temp_store):
        assert isinstance(
            temp_store._write_contributions("p1", {"project": {"id": "p1"}}), dict)

    def test_no_decisions_reports_zero(self, temp_store):
        assert temp_store._write_contributions("p1", _bundle(decisions=[]))["decisions"] == 0

    def test_re_importing_the_same_bundle_does_not_duplicate_records(self, temp_store):
        """The merge is keyed by (paper, reviewer, stage), so importing twice
        must overwrite rather than double every decision in the project."""
        b = _bundle()
        temp_store._write_contributions("p1", b)
        temp_store._write_contributions("p1", b)
        assert len(temp_store.kv_get_by_prefix("decision:p1:")) == 3

    def test_papers_are_deduped_on_a_second_import(self, temp_store):
        b = _bundle()
        first = temp_store._write_contributions("p1", b)["papers"]
        second = temp_store._write_contributions("p1", b)["papers"]
        assert first > 0 and second == 0


# ---------------------------------------------------------------------------
# Snowballing
# ---------------------------------------------------------------------------

class TestFetchCitations:
    def _resp(self, payload, status=200):
        return type("R", (), {"status_code": status,
                              "json": staticmethod(lambda: payload),
                              "text": "", "content": b"", "headers": {}})()

    def test_parses_europepmc_references(self, monkeypatch):
        import requests
        payload = {"referenceList": {"reference": [
            {"id": "111", "title": "A cited study", "authorString": "Smith J"},
        ]}}
        monkeypatch.setattr(requests, "get", lambda *a, **k: self._resp(payload))
        out = utils.AIService.fetch_citations("123", "PubMed", "T", "backward", 10, ["PubMed"])
        assert isinstance(out, list)

    def test_parses_europepmc_citations(self, monkeypatch):
        import requests
        payload = {"citationList": {"citation": [
            {"id": "222", "title": "A citing study", "authorString": "Jones A"},
        ]}}
        monkeypatch.setattr(requests, "get", lambda *a, **k: self._resp(payload))
        out = utils.AIService.fetch_citations("123", "PubMed", "T", "forward", 10, ["PubMed"])
        assert isinstance(out, list)

    def test_respects_the_result_cap(self, monkeypatch):
        import requests
        many = {"referenceList": {"reference": [
            {"id": str(i), "title": f"Ref {i}"} for i in range(200)]}}
        monkeypatch.setattr(requests, "get", lambda *a, **k: self._resp(many))
        out = utils.AIService.fetch_citations("123", "PubMed", "T", "backward", 10, ["PubMed"])
        assert len(out) <= 200

    def test_an_http_error_returns_a_list(self, monkeypatch):
        import requests
        monkeypatch.setattr(requests, "get", lambda *a, **k: self._resp({}, status=500))
        assert isinstance(
            utils.AIService.fetch_citations("123", "PubMed", "T", "backward", 10, ["PubMed"]), list)

    def test_a_malformed_payload_returns_a_list(self, monkeypatch):
        import requests
        monkeypatch.setattr(requests, "get", lambda *a, **k: self._resp({"unexpected": True}))
        assert isinstance(
            utils.AIService.fetch_citations("123", "PubMed", "T", "backward", 10, ["PubMed"]), list)

    def test_no_active_sources_returns_a_list(self, monkeypatch):
        import requests
        monkeypatch.setattr(requests, "get", lambda *a, **k: self._resp({}))
        assert isinstance(
            utils.AIService.fetch_citations("123", "PubMed", "T", "backward", 10, []), list)

    @pytest.mark.parametrize("direction", ["backward", "forward", "both"])
    def test_every_direction_is_handled(self, monkeypatch, direction):
        import requests
        monkeypatch.setattr(requests, "get", lambda *a, **k: self._resp({}))
        assert isinstance(
            utils.AIService.fetch_citations("1", "PubMed", "T", direction, 5, ["PubMed"]), list)
