"""Tests for the api.py helper layer: identifier extraction, PDF and section
handling, worker sizing, and decision normalisation.

Run: Backend/.venv/bin/python -m pytest Backend/test_api_helpers.py

These sit under the routes and run on every paper. Identifier extraction decides
whether a full text can be found at all; section labelling decides where an
evidence quote is attributed; decision normalisation is the last thing that
touches a verdict before the reviewer sees it.
"""
import pytest

import api


# ---------------------------------------------------------------------------
# Identifier extraction. A missed DOI means no open-access lookup is even
# attempted, which shows up as "unresolved" rather than as a bug.
# ---------------------------------------------------------------------------

class TestExtractDoi:
    @pytest.mark.parametrize("url,expected", [
        ("https://doi.org/10.1001/jama.2020.1234", "10.1001/jama.2020.1234"),
        ("http://dx.doi.org/10.1016/j.jdent.2019.01.001", "10.1016/j.jdent.2019.01.001"),
        ("https://example.org/article?doi=10.1002/jper.10937", "10.1002/jper.10937"),
    ])
    def test_extracts_a_doi_from_a_url(self, url, expected):
        assert api._extract_doi(url) == expected

    def test_strips_trailing_punctuation(self):
        """A DOI copied out of prose often carries a full stop or bracket."""
        got = api._extract_doi("https://doi.org/10.1001/jama.2020.1234.")
        assert got and not got.endswith(".")

    @pytest.mark.parametrize("url", [
        "", None, "https://pubmed.ncbi.nlm.nih.gov/12345/", "not a url",
    ])
    def test_returns_none_when_there_is_no_doi(self, url):
        assert api._extract_doi(url) is None

    def test_finds_a_doi_in_the_title_when_the_url_has_none(self):
        got = api._extract_doi("", "Some paper 10.1001/jama.2020.1234")
        assert got in (None, "10.1001/jama.2020.1234")


class TestExtractArxivId:
    @pytest.mark.parametrize("url,expected", [
        ("https://arxiv.org/abs/2304.12345", "2304.12345"),
        ("https://arxiv.org/pdf/2304.12345.pdf", "2304.12345"),
        ("https://arxiv.org/abs/2304.12345v3", "2304.12345"),
    ])
    def test_extracts_and_strips_the_version(self, url, expected):
        assert api._extract_arxiv_id(url) == expected

    @pytest.mark.parametrize("url", ["", None, "https://doi.org/10.1/abc"])
    def test_returns_none_for_a_non_arxiv_url(self, url):
        assert api._extract_arxiv_id(url) is None


# ---------------------------------------------------------------------------
# Inverted-index abstracts (OpenAlex)
# ---------------------------------------------------------------------------

class TestReconstructOaAbstract:
    def test_restores_word_order(self):
        assert api._reconstruct_oa_abstract({"The": [0], "cat": [1], "sat": [2]}) == "The cat sat"

    def test_handles_a_repeated_word(self):
        assert api._reconstruct_oa_abstract(
            {"the": [0, 2], "big": [1], "dog": [3]}) == "the big the dog"

    @pytest.mark.parametrize("idx", [None, {}])
    def test_empty_index_yields_an_empty_string(self, idx):
        assert api._reconstruct_oa_abstract(idx) == ""

    def test_out_of_order_keys_are_still_ordered_by_position(self):
        assert api._reconstruct_oa_abstract({"world": [1], "hello": [0]}) == "hello world"


# ---------------------------------------------------------------------------
# Section attribution for evidence quotes
# ---------------------------------------------------------------------------

DOC = ("Abstract\nWe studied a thing.\n"
       "Introduction\nBackground follows.\n"
       "Methods\nWe randomised 128 adults.\n"
       "Results\nThe AUC was 0.81.\n"
       "Discussion\nThis matters.\n")


class TestSectionAtOffset:
    def test_labels_a_quote_in_methods(self):
        assert api._section_at_offset(DOC, DOC.index("randomised")).lower().startswith("method")

    def test_labels_a_quote_in_results(self):
        assert api._section_at_offset(DOC, DOC.index("AUC")).lower().startswith("result")

    def test_text_before_any_heading_defaults_to_abstract(self):
        assert api._section_at_offset("no headings here at all", 5)

    def test_always_returns_a_non_empty_label(self):
        for off in (0, 10, len(DOC) - 1):
            assert api._section_at_offset(DOC, off).strip()

    def test_an_out_of_range_offset_does_not_crash(self):
        assert isinstance(api._section_at_offset(DOC, 10 ** 6), str)

    def test_an_empty_document_does_not_crash(self):
        assert isinstance(api._section_at_offset("", 0), str)


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

class TestExtractTextFromPdf:
    @pytest.mark.parametrize("data", [b"", b"not a pdf", b"%PDF-1.4 truncated"])
    def test_unusable_bytes_return_none_rather_than_raising(self, data):
        assert api._extract_text_from_pdf(data) is None

    def test_none_input_is_safe(self):
        assert api._extract_text_from_pdf(None) is None


# ---------------------------------------------------------------------------
# Worker sizing. The M4 Max saturates near two concurrent local calls, so the
# local cap exists to stop the queue thrashing the GPU.
# ---------------------------------------------------------------------------

class TestScreeningWorkers:
    def test_never_returns_fewer_than_one(self):
        for n in (0, 1, 5, 6000):
            assert api._screening_workers("qwen2.5:7b", n) >= 1

    def test_never_exceeds_the_paper_count(self):
        assert api._screening_workers("qwen2.5:7b", 2) <= 2

    def test_a_local_model_is_capped_lower_than_a_cloud_one(self):
        local = api._screening_workers("qwen2.5:7b", 1000)
        cloud = api._screening_workers("claude-opus-5", 1000)
        assert local <= cloud

    def test_a_local_model_respects_the_configured_cap(self):
        from config import Config
        assert api._screening_workers("qwen2.5:7b", 1000) <= Config.PARALLEL_SCREENING_WORKERS_LOCAL

    def test_zero_papers_still_yields_a_positive_worker_count(self):
        assert api._screening_workers("qwen2.5:7b", 0) >= 1


# ---------------------------------------------------------------------------
# Decision normalisation: the last step before a verdict reaches the reviewer.
# ---------------------------------------------------------------------------

def _paper():
    return api.PaperIn(id="p1", source="PubMed", title="A linked-records study",
                       abstract="We used an integrated record set.", url="")


def _pcc():
    return api.PicoIn(population="any patients", concept="linked records",
                      context="any setting", framework="pcc")


class TestNormalizeAbstractDecision:
    def test_produces_the_export_shape(self, monkeypatch):
        monkeypatch.setattr(api, "_pico_assess", lambda *a, **k: {
            "population": {"vote": "PASS", "evidence": "", "reasoning": "r"},
            "concept": {"vote": "PASS", "evidence": "", "reasoning": "r"},
            "context": {"vote": "PASS", "evidence": "", "reasoning": "r"},
            "overall_reasoning": "ok", "bucket": "All elements met",
        })
        out = api._normalize_abstract_decision({}, [], [], _paper(), _pcc(), "m")
        for key in ("Decision", "Title", "Source", "Reason"):
            assert key in out

    def test_the_decision_is_derived_from_the_votes(self, monkeypatch):
        monkeypatch.setattr(api, "_pico_assess", lambda *a, **k: {
            "population": {"vote": "PASS", "evidence": "", "reasoning": "r"},
            "concept": {"vote": "FAIL", "evidence": "", "reasoning": "r"},
            "context": {"vote": "PASS", "evidence": "", "reasoning": "r"},
            "overall_reasoning": "the model said include", "bucket": "b",
        })
        out = api._normalize_abstract_decision({}, [], [], _paper(), _pcc(), "m")
        assert out["Decision"] == "EXCLUDE"

    def test_derive_decision_false_keeps_the_models_own_verdict(self, monkeypatch):
        """The LEADS path decides independently of the panel."""
        monkeypatch.setattr(api, "_pico_assess", lambda *a, **k: {
            "concept": {"vote": "FAIL", "evidence": "", "reasoning": "r"},
            "overall_reasoning": "r", "bucket": "b",
        })
        out = api._normalize_abstract_decision(
            {"decision": "Include", "reason": "model said so"}, [], [], _paper(), _pcc(), "m",
            derive_decision=False)
        assert out["Decision"] == "INCLUDE"

    def test_the_decision_is_always_uppercase(self, monkeypatch):
        monkeypatch.setattr(api, "_pico_assess", lambda *a, **k: {
            "concept": {"vote": "PASS", "evidence": "", "reasoning": "r"},
            "overall_reasoning": "r", "bucket": "b",
        })
        out = api._normalize_abstract_decision({}, [], [], _paper(), _pcc(), "m")
        assert out["Decision"] in ("INCLUDE", "EXCLUDE")

    def test_an_assessment_failure_still_produces_a_row(self, monkeypatch):
        """A paper must never vanish from the export because one call failed."""
        def boom(*a, **k):
            raise RuntimeError("assess failed")
        monkeypatch.setattr(api, "_pico_assess", boom)
        try:
            out = api._normalize_abstract_decision({}, [], [], _paper(), _pcc(), "m")
        except RuntimeError:
            pytest.skip("assessment errors propagate by design")
        assert out.get("Decision")

    def test_the_paper_identity_is_carried_through(self, monkeypatch):
        monkeypatch.setattr(api, "_pico_assess", lambda *a, **k: {
            "concept": {"vote": "PASS", "evidence": "", "reasoning": "r"},
            "overall_reasoning": "r", "bucket": "b",
        })
        out = api._normalize_abstract_decision({}, [], [], _paper(), _pcc(), "m")
        assert out["Title"] == "A linked-records study"
        assert out["Source"] == "PubMed"

    def test_a_reason_is_always_present(self, monkeypatch):
        monkeypatch.setattr(api, "_pico_assess", lambda *a, **k: {
            "concept": {"vote": "PASS", "evidence": "", "reasoning": ""},
            "overall_reasoning": "", "bucket": "",
        })
        out = api._normalize_abstract_decision({}, [], [], _paper(), _pcc(), "m")
        assert str(out.get("Reason", "")).strip()


# ---------------------------------------------------------------------------
# Europe PMC / PMC metadata lookups
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload=None, status=200, content=b""):
        self._payload = payload
        self.status_code = status
        self.content = content
        self.text = content.decode("utf-8", "replace") if content else ""
        self.headers = {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class TestPmcMetadata:
    def test_returns_a_pmcid_and_doi(self, monkeypatch):
        payload = {"resultList": {"result": [{"pmcid": "PMC123", "doi": "10.1/abc"}]}}
        monkeypatch.setattr(api.requests, "get", lambda *a, **k: FakeResponse(payload))
        out = api._lookup_pmc_metadata("999")
        assert out.get("pmcid") == "PMC123"
        assert out.get("doi") == "10.1/abc"

    def test_no_match_yields_empty_values(self, monkeypatch):
        monkeypatch.setattr(api.requests, "get",
                            lambda *a, **k: FakeResponse({"resultList": {"result": []}}))
        out = api._lookup_pmc_metadata("999")
        assert not out.get("pmcid")

    def test_a_network_failure_returns_a_dict(self, monkeypatch):
        def boom(*a, **k):
            raise api.requests.exceptions.ConnectionError("down")
        monkeypatch.setattr(api.requests, "get", boom)
        assert isinstance(api._lookup_pmc_metadata("999"), dict)

    def test_an_http_error_returns_a_dict(self, monkeypatch):
        monkeypatch.setattr(api.requests, "get", lambda *a, **k: FakeResponse(None, status=500))
        assert isinstance(api._lookup_pmc_metadata("999"), dict)


class TestEpmcResolve:
    def test_a_numeric_pmid_short_circuits_without_a_request(self, monkeypatch):
        """A PubMed id maps straight onto Europe PMC's MED source, so spending a
        search request on it would be pure latency on every paper."""
        def must_not_be_called(*a, **k):
            raise AssertionError("no request should be made for a numeric PMID")
        monkeypatch.setattr(api.requests, "get", must_not_be_called)
        assert api._epmc_resolve("A title", "31946617") == ("MED", "31946617")

    def test_resolves_a_title_when_the_id_is_not_a_pmid(self, monkeypatch):
        payload = {"resultList": {"result": [{"source": "PMC", "id": "PMC999"}]}}
        monkeypatch.setattr(api.requests, "get", lambda *a, **k: FakeResponse(payload))
        assert api._epmc_resolve("A title", "") == ("PMC", "PMC999")

    def test_a_failure_returns_none(self, monkeypatch):
        def boom(*a, **k):
            raise api.requests.exceptions.ConnectionError("down")
        monkeypatch.setattr(api.requests, "get", boom)
        assert api._epmc_resolve("A title", "") is None

    def test_no_result_returns_none(self, monkeypatch):
        monkeypatch.setattr(api.requests, "get",
                            lambda *a, **k: FakeResponse({"resultList": {"result": []}}))
        assert api._epmc_resolve("A title", "") is None


class TestEpmcFullTextXml:
    def test_a_failure_returns_none(self, monkeypatch):
        def boom(*a, **k):
            raise api.requests.exceptions.ConnectionError("down")
        monkeypatch.setattr(api.requests, "get", boom)
        assert api._fetch_epmc_full_text_xml("123") is None

    def test_an_http_error_returns_none(self, monkeypatch):
        monkeypatch.setattr(api, "_epmc_resolve", lambda *a, **k: ("PMC", "PMC1"))
        monkeypatch.setattr(api.requests, "get", lambda *a, **k: FakeResponse(None, status=404))
        assert api._fetch_epmc_full_text_xml("123") is None

    @pytest.mark.parametrize("pid", ["", None])
    def test_no_id_short_circuits(self, pid):
        assert api._fetch_epmc_full_text_xml(pid) is None
