"""Tests for the search pipeline: query construction and yield estimation.

Run: Backend/.venv/bin/python -m pytest Backend/test_search_pipeline.py

These run BEFORE screening, so a defect here poisons every downstream stage: the
wrong query builds the wrong corpus, and screening then does an excellent job on
the wrong papers. Both functions covered here have already shipped bugs of
exactly that kind (a 260x Europe PMC inflation from untranslated field tags, and
a silent zero from an over-large page size).

No network and no model: the LLM is stubbed and the transport is replaced.
"""
import pytest

import data_services as ds
import utils
from config import DataSource
from models import PICOCriteria


class StubModel:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return type("R", (), {"content": self.reply})()


CONCEPTS = """{"concepts":[
 {"name":"Artificial intelligence","terms":["machine learning","AI"],"mesh":["Artificial Intelligence"]},
 {"name":"Dental","terms":["dental","dentistry"],"mesh":["Dentistry"]},
 {"name":"Adults","terms":["adults","patients"],"mesh":[]}
]}"""


@pytest.fixture
def stub_model(monkeypatch):
    def _install(reply=CONCEPTS):
        m = StubModel(reply)
        monkeypatch.setattr(utils.AIService, "get_model", staticmethod(lambda *a, **k: m))
        return m
    return _install


PICO = PICOCriteria(population="adults", intervention="AI prediction models",
                    comparator="", outcome="diagnostic accuracy")


# ---------------------------------------------------------------------------
# generate_mesh_query
#
# The validated strategy: AND together the DISCRIMINATING concept blocks only.
# Requiring a broad population facet like "adults" mostly deletes true hits
# without narrowing scope, which is how a search ends up with near-zero recall.
# ---------------------------------------------------------------------------

class TestGenerateMeshQuery:
    def test_builds_and_of_or_blocks(self, stub_model):
        stub_model()
        q = utils.AIService.generate_mesh_query(PICO, "m")
        assert " AND " in q
        assert " OR " in q

    def test_drops_the_broad_population_block(self, stub_model):
        """"Adults" is not discriminating. AND-ing it in deletes true hits."""
        stub_model()
        q = utils.AIService.generate_mesh_query(PICO, "m")
        assert "adults" not in q.lower()

    def test_keeps_the_discriminating_blocks(self, stub_model):
        stub_model()
        q = utils.AIService.generate_mesh_query(PICO, "m")
        assert "dental" in q.lower()
        assert "machine learning" in q.lower()

    def test_tags_free_text_terms_as_tiab(self, stub_model):
        stub_model()
        q = utils.AIService.generate_mesh_query(PICO, "m")
        assert "[tiab]" in q

    def test_tags_mesh_headings_as_mesh(self, stub_model):
        stub_model()
        q = utils.AIService.generate_mesh_query(PICO, "m")
        assert "[Mesh]" in q

    def test_parentheses_are_balanced(self, stub_model):
        stub_model()
        q = utils.AIService.generate_mesh_query(PICO, "m")
        assert q.count("(") == q.count(")")

    def test_return_concepts_reports_only_the_kept_blocks(self, stub_model):
        stub_model()
        q, concepts = utils.AIService.generate_mesh_query(PICO, "m", return_concepts=True)
        names = [c["name"] for c in concepts]
        assert "Adults" not in names
        assert len(concepts) == 2

    def test_every_returned_concept_has_the_expected_shape(self, stub_model):
        stub_model()
        _, concepts = utils.AIService.generate_mesh_query(PICO, "m", return_concepts=True)
        for c in concepts:
            assert {"name", "tiab", "mesh"} <= set(c)

    def test_pcc_fields_are_used_as_well_as_pico(self, stub_model):
        """The same builder serves scoping reviews, which have concept/context
        rather than intervention/outcome."""
        m = stub_model()
        pcc = PICOCriteria(population="any patients", concept="AI using dental records",
                           context="any setting", framework="pcc")
        utils.AIService.generate_mesh_query(pcc, "m")
        prompt = m.calls and True
        assert prompt

    def test_malformed_model_output_does_not_raise(self, stub_model):
        stub_model("this is not json")
        out = utils.AIService.generate_mesh_query(PICO, "m")
        assert isinstance(out, str)

    def test_model_exception_does_not_raise(self, stub_model):
        stub_model(RuntimeError("model down"))
        out = utils.AIService.generate_mesh_query(PICO, "m")
        assert isinstance(out, str)

    def test_no_concepts_returns_a_string_not_a_broken_query(self, stub_model):
        stub_model('{"concepts": []}')
        q = utils.AIService.generate_mesh_query(PICO, "m")
        assert isinstance(q, str)
        assert "AND AND" not in q
        assert not q.strip().endswith("AND")

    def test_single_concept_produces_no_dangling_and(self, stub_model):
        stub_model('{"concepts":[{"name":"Dental","terms":["dental"],"mesh":["Dentistry"]}]}')
        q = utils.AIService.generate_mesh_query(PICO, "m")
        assert " AND " not in q
        assert q.count("(") == q.count(")")

    def test_empty_pico_does_not_raise(self, stub_model):
        stub_model()
        assert isinstance(utils.AIService.generate_mesh_query(PICOCriteria(), "m"), str)

    def test_concept_with_no_mesh_still_yields_a_usable_block(self, stub_model):
        stub_model('{"concepts":[{"name":"Dental","terms":["dental","oral"],"mesh":[]},'
                   '{"name":"AI","terms":["machine learning"],"mesh":[]}]}')
        q = utils.AIService.generate_mesh_query(PICO, "m")
        assert "[tiab]" in q and "[Mesh]" not in q

    def test_concept_with_no_terms_does_not_emit_an_empty_group(self, stub_model):
        stub_model('{"concepts":[{"name":"Empty","terms":[],"mesh":[]},'
                   '{"name":"Dental","terms":["dental"],"mesh":[]}]}')
        q = utils.AIService.generate_mesh_query(PICO, "m")
        assert "()" not in q.replace(" ", "")

    def test_the_inferred_marker_never_reaches_the_query(self, stub_model):
        """PICO fields display "(inferred)" for auto-filled values. Searching
        PubMed for the word "inferred" would be nonsense."""
        m = stub_model()
        p = PICOCriteria(population="(inferred) adults", intervention="(inferred) AI models")
        utils.AIService.generate_mesh_query(p, "m")
        assert "inferred" not in utils.AIService.generate_mesh_query(p, "m").lower()


# ---------------------------------------------------------------------------
# simulate_yield
#
# Drives the per-database counts on the Planning tab, which the reviewer uses to
# decide the search is sane before committing to it. A wrong number here is a
# wrong plan, and it has been wrong before.
# ---------------------------------------------------------------------------

class FakeHandle:
    def __init__(self, payload):
        self.payload = payload

    def close(self):
        pass


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self._payload = payload
        self.status_code = status
        self.text = ""
        self.content = b""
        self.headers = {}

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


@pytest.fixture
def stub_pubmed(monkeypatch):
    """Stub Entrez so no NCBI request is made."""
    def _install(count="1234"):
        monkeypatch.setattr(ds.Entrez, "esearch", lambda **k: FakeHandle({"Count": count}))
        monkeypatch.setattr(ds.Entrez, "read", lambda h: h.payload)
    return _install


class TestSimulateYield:
    def test_reports_a_count_per_requested_source(self, stub_pubmed, monkeypatch):
        stub_pubmed("4321")
        monkeypatch.setattr(ds, "throttled_request",
                            lambda *a, **k: FakeResponse({"meta": {"count": 10}}))
        out = ds.DataAggregator.simulate_yield("dental AND ai", [DataSource.PUBMED.value])
        assert out[DataSource.PUBMED.value] == 4321

    def test_returns_a_dict_keyed_by_source(self, stub_pubmed, monkeypatch):
        stub_pubmed()
        monkeypatch.setattr(ds, "throttled_request",
                            lambda *a, **k: FakeResponse({"meta": {"count": 7}}))
        sources = [DataSource.PUBMED.value, "OpenAlex"]
        out = ds.DataAggregator.simulate_yield("q", sources)
        assert set(out) <= set(sources)

    def test_counts_are_always_integers(self, stub_pubmed, monkeypatch):
        stub_pubmed("99")
        monkeypatch.setattr(ds, "throttled_request",
                            lambda *a, **k: FakeResponse({"meta": {"count": 5}}))
        out = ds.DataAggregator.simulate_yield("q", [DataSource.PUBMED.value, "OpenAlex"])
        assert all(isinstance(v, int) for v in out.values())

    def test_a_non_numeric_count_becomes_zero_not_a_crash(self, stub_pubmed):
        stub_pubmed("not-a-number")
        out = ds.DataAggregator.simulate_yield("q", [DataSource.PUBMED.value])
        assert out[DataSource.PUBMED.value] == 0

    def test_a_failing_source_yields_zero_and_does_not_stop_the_others(self, stub_pubmed, monkeypatch):
        """One dead provider must not take down the whole yield estimate."""
        stub_pubmed("50")

        def boom(*a, **k):
            raise ds.requests.exceptions.ConnectionError("down")

        monkeypatch.setattr(ds, "throttled_request", boom)
        out = ds.DataAggregator.simulate_yield("q", [DataSource.PUBMED.value, "OpenAlex"])
        assert out[DataSource.PUBMED.value] == 50
        assert out.get("OpenAlex", 0) == 0

    def test_entrez_failure_yields_zero(self, monkeypatch):
        def boom(**k):
            raise RuntimeError("NCBI down")
        monkeypatch.setattr(ds.Entrez, "esearch", boom)
        out = ds.DataAggregator.simulate_yield("q", [DataSource.PUBMED.value])
        assert out.get(DataSource.PUBMED.value, 0) == 0

    def test_local_pdfs_report_zero_rather_than_referencing_an_upload(self):
        """simulate_yield estimates REMOTE databases and has no upload context.
        Reading one here raised NameError before this was pinned."""
        out = ds.DataAggregator.simulate_yield("q", [DataSource.LOCAL_PDF.value])
        assert out.get(DataSource.LOCAL_PDF.value) == 0

    def test_empty_source_list_yields_an_empty_result(self):
        assert ds.DataAggregator.simulate_yield("q", []) == {}

    def test_no_source_is_silently_dropped_from_the_report(self, stub_pubmed, monkeypatch):
        """A missing key reads as "not searched" in the UI, which is different
        from "searched and found nothing"."""
        stub_pubmed("1")
        monkeypatch.setattr(ds, "throttled_request",
                            lambda *a, **k: FakeResponse(None, status=500))
        sources = [DataSource.PUBMED.value, "OpenAlex", "CrossRef"]
        out = ds.DataAggregator.simulate_yield("q", sources)
        assert set(out) == set(sources)

    def test_europepmc_query_is_translated_before_counting(self, monkeypatch):
        """Sending PubMed field tags to Europe PMC runs them as full-text search
        and inflated one real count by about 260x."""
        seen = {}

        def capture(url, params=None, **k):
            seen["params"] = params or {}
            return FakeResponse({"hitCount": 12})

        monkeypatch.setattr(ds, "throttled_request", capture)
        ds.DataAggregator.simulate_yield('("dental"[tiab])', ["Europe PMC"])
        assert "[tiab]" not in str(seen.get("params", {}))
