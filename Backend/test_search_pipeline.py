"""Tests for the search pipeline: query construction and yield estimation.

Run: Backend/.venv/bin/python -m pytest Backend/test_search_pipeline.py

These run BEFORE screening, so a defect here poisons every downstream stage: the
wrong query builds the wrong corpus, and screening then does an excellent job on
the wrong papers. Both functions covered here have already shipped bugs of
exactly that kind (a 260x Europe PMC inflation from untranslated field tags, and
a silent zero from an over-large page size).

No network and no model: the LLM is stubbed and the transport is replaced.
"""
import json
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

    def test_a_dropped_discriminating_term_is_added_back(self, monkeypatch):
        """The failure this guard exists for.

        A real question ("AI/ML models ... using EHRs containing both medical and
        dental data") decomposed into AI/ML, EHR and *medical* data, silently
        dropping "dental". The query stayed well-formed and returned 15,580
        PubMed hits about prediction models in general instead of the 43 about
        linked dental records. Nothing in the result revealed the omission, so
        the check is made in code rather than left to the prompt.
        """
        calls = {"n": 0}

        class Stub:
            def invoke(self, _msgs):
                calls["n"] += 1
                # Always omits "dental", however many times it is asked.
                return type("R", (), {"content": json.dumps({"concepts": [
                    {"name": "AI/ML", "terms": ["machine learning"], "mesh": []},
                    {"name": "Medical data", "terms": ["medical data"], "mesh": []},
                ]})})()

        monkeypatch.setattr(utils.AIService, "get_model", staticmethod(lambda *a, **k: Stub()))
        pico = PICOCriteria(population="", intervention="", comparator="", outcome="")
        pico.concept = "AI/ML models using records containing both medical and dental data"
        q, concepts = utils.AIService.generate_mesh_query(pico, "m", return_concepts=True)

        assert "dental" in q.lower(), "a term the user wrote must survive into the query"
        assert calls["n"] > 1, "a dropped discriminating term should trigger a retry"

    def test_a_term_buried_in_another_concept_does_not_count_as_covered(self, monkeypatch):
        """Being present is not the same as being required.

        Concepts are AND-ed; the terms inside one are OR-ed. So "dental" listed
        among the electronic-health-records synonyms is satisfied by any paper
        mentioning "EHR", and constrains nothing. That is exactly what shipped:
        the query contained the word "dental" and still returned general EHR
        papers, because it was never its own block.
        """
        calls = {"n": 0}

        class Stub:
            def invoke(self, _msgs):
                calls["n"] += 1
                if calls["n"] == 1:
                    # "dental" present, but only as a synonym of another concept.
                    return type("R", (), {"content": json.dumps({"concepts": [
                        {"name": "Electronic health records",
                         "terms": ["electronic health record", "EHR", "dental record"],
                         "mesh": []},
                    ]})})()
                return type("R", (), {"content": json.dumps({"concepts": [
                    {"name": "Electronic health records", "terms": ["EHR"], "mesh": []},
                    {"name": "Dental data", "terms": ["dental data"], "mesh": []},
                ]})})()

        monkeypatch.setattr(utils.AIService, "get_model", staticmethod(lambda *a, **k: Stub()))
        pico = PICOCriteria(population="", intervention="", comparator="", outcome="")
        pico.concept = "records containing both medical and dental data"
        q, concepts = utils.AIService.generate_mesh_query(pico, "m", return_concepts=True)

        assert calls["n"] > 1, "a term with no block of its own should trigger a retry"
        assert any("dental" in (c["name"] or "").lower() for c in concepts), \
            "the term must end up as its own AND-ed block, not a buried synonym"

    def test_concepts_are_salvaged_from_a_truncated_reply(self, monkeypatch):
        """A cut-off reply must not throw away the concepts it did contain.

        The real failure: asked for synonyms of "Healthcare settings, global
        scope, 2010-present", the model had nothing real to list and looped on
        permutations ("clinical settings", "hospital settings", "clinics
        settings", ...) until it hit the 4096-token output cap. The JSON ended
        mid-string, json.loads rejected all 21,000 characters, and four correct
        concepts sitting at the top of the reply were discarded with it.
        """
        truncated = (
            '{"concepts": ['
            '{"name": "AI/ML", "terms": ["machine learning", "AI"], "mesh": ["Artificial Intelligence"]},'
            '{"name": "Dental data", "terms": ["dental data", "oral health"], "mesh": []},'
            '{"name": "Settings", "terms": ["clinical settings", "hospital settings", "clinics sett'
        )

        class Stub:
            def invoke(self, _msgs):
                return type("R", (), {"content": truncated})()

        monkeypatch.setattr(utils.AIService, "get_model", staticmethod(lambda *a, **k: Stub()))
        pico = PICOCriteria(population="", intervention="", comparator="", outcome="")
        pico.concept = "AI/ML models using records containing both medical and dental data"
        q, concepts = utils.AIService.generate_mesh_query(pico, "m", return_concepts=True)

        names = " ".join(c["name"] for c in concepts).lower()
        assert "ai/ml" in names, "the complete concepts before the break must survive"
        assert "dental" in q.lower(), "including the discriminating one"

    def test_the_setting_facet_is_not_sent_for_decomposition(self, stub_model):
        """Context is excluded as a source facet.

        The prompt already says not to invent concepts for setting, country or
        year; passing the setting in anyway contradicted that and is what sent
        the model looping over "settings" synonyms until it ran out of tokens.
        """
        captured = {}

        class Stub:
            def invoke(self, msgs):
                captured["prompt"] = msgs[0].content
                return type("R", (), {"content": json.dumps({"concepts": [
                    {"name": "Dental", "terms": ["dental"], "mesh": []}]})})()

        import pytest as _pytest
        pico = PICOCriteria(population="", intervention="", comparator="", outcome="")
        pico.concept = "records containing dental data"
        pico.context = "Healthcare settings, global scope, 2010-present"
        with _pytest.MonkeyPatch.context() as mp:
            mp.setattr(utils.AIService, "get_model", staticmethod(lambda *a, **k: Stub()))
            utils.AIService.generate_mesh_query(pico, "m")
        assert "global scope" not in captured["prompt"]

    def test_outcome_terms_are_not_forced_into_the_query(self, stub_model):
        """Coverage applies to what the search is ABOUT, not to outcomes.

        Broad outcome and study-design facets are dropped on purpose: requiring
        them deletes true hits without narrowing scope. Repairing their absence
        would undo the strategy the block decomposition exists to implement.
        """
        stub_model()
        q = utils.AIService.generate_mesh_query(PICO, "m")
        assert "diagnostic" not in q.lower()
        assert "accuracy" not in q.lower()

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

    # These three used to assert the opposite: that a failure returned a string
    # rather than raising. That contract was the problem. There is no generic
    # query that answers an arbitrary review question, so the string it returned
    # was assembled from leftover words and searched for something else, while
    # looking entirely normal. A visible failure is recoverable; a plausible
    # wrong search gets screened, acquired and written up.

    def test_malformed_model_output_raises_rather_than_inventing_a_query(self, stub_model):
        stub_model("this is not json")
        with pytest.raises(utils.QueryGenerationError):
            utils.AIService.generate_mesh_query(PICO, "m")

    def test_a_model_failure_raises(self, stub_model):
        stub_model(RuntimeError("model down"))
        with pytest.raises(utils.QueryGenerationError):
            utils.AIService.generate_mesh_query(PICO, "m")

    def test_an_empty_concept_list_raises(self, stub_model):
        stub_model('{"concepts": []}')
        with pytest.raises(utils.QueryGenerationError):
            utils.AIService.generate_mesh_query(PICO, "m")

    def test_the_error_tells_the_reviewer_what_to_do(self, stub_model):
        """The message reaches the UI verbatim, so it is written for a person."""
        stub_model("this is not json")
        with pytest.raises(utils.QueryGenerationError) as e:
            utils.AIService.generate_mesh_query(PICO, "m")
        assert "reword" in str(e.value).lower() or "shorten" in str(e.value).lower()

    def test_every_retry_uses_a_different_seed(self, monkeypatch):
        """Sampling is greedy, so an identical prompt and seed return identical
        bytes. A retry that varies neither is guaranteed to fail the same way,
        which is how one bad question cost three minutes of retries."""
        seeds = []

        class Stub:
            def invoke(self, _msgs):
                return type("R", (), {"content": "not json"})()

        def fake_get_model(name, seed=None):
            seeds.append(seed)
            return Stub()

        monkeypatch.setattr(utils.AIService, "get_model", staticmethod(fake_get_model))
        with pytest.raises(utils.QueryGenerationError):
            utils.AIService.generate_mesh_query(PICO, "m")
        real = [x for x in seeds if x is not None]
        assert len(real) > 1, "it should retry"
        assert len(set(real)) == len(real), "each attempt needs its own seed"

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
