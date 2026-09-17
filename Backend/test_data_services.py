"""Tests for the literature-source services. No network: every response is a
recorded shape fed through a stubbed transport.

Run: Backend/.venv/bin/python -m pytest Backend/test_data_services.py

What these protect against: a provider changes a field name, the parser silently
yields zero papers or papers with empty abstracts, and the review is built on a
corpus that is quietly wrong. Nothing raises. That is the same failure shape as
the PubMed abstract truncation, which cost half of every structured abstract.
"""
import pytest

import data_services as ds
from models import Paper
from utils import Deduplicator


class FakeResponse:
    def __init__(self, payload=None, text="", status=200):
        self._payload = payload
        self.text = text
        self.status_code = status
        self.content = text.encode() if text else b""
        self.headers = {}

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


@pytest.fixture
def stub(monkeypatch):
    """Replace the shared transport and hand back the captured calls."""
    calls = []

    def _install(payload=None, text="", status=200):
        def fake(url, params=None, headers=None, method="GET", **kw):
            calls.append({"url": url, "params": params or {}, "headers": headers or {}})
            return FakeResponse(payload, text, status)
        monkeypatch.setattr(ds, "throttled_request", fake)
        return calls

    return _install


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

class TestOpenAlex:
    def test_parses_results_into_papers(self, stub):
        stub({"results": [{
            "id": "https://openalex.org/W123",
            "title": "A dental prediction model",
            "doi": "https://doi.org/10.1/abc",
            "abstract_inverted_index": {"Hello": [0], "world": [1]},
            "open_access": {"oa_url": "https://x.org/a.pdf"},
            "publication_year": 2024,
        }]})
        papers = ds.OpenAlexService.fetch("dental", 10)
        assert len(papers) == 1
        p = papers[0]
        assert p.source == "OpenAlex"
        assert p.title == "A dental prediction model"
        assert p.abstract == "Hello world"

    def test_sends_no_contact_email(self, stub):
        """Addresses are per-user and per-request. A hardcoded or placeholder
        mailto is rejected by the very APIs it is meant to help with."""
        calls = stub({"results": []})
        ds.OpenAlexService.fetch("dental", 5)
        assert "mailto" not in calls[0]["params"]

    def test_empty_results_yield_no_papers(self, stub):
        stub({"results": []})
        assert ds.OpenAlexService.fetch("dental", 10) == []

    def test_malformed_payload_does_not_raise(self, stub):
        stub({"unexpected": "shape"})
        assert ds.OpenAlexService.fetch("dental", 10) == []

    def test_strips_field_tags_from_the_query(self, stub):
        """PubMed syntax sent verbatim to OpenAlex matches nothing useful."""
        calls = stub({"results": []})
        ds.OpenAlexService.fetch('("dental"[tiab])', 5)
        assert "[tiab]" not in str(calls[0]["params"])


class TestInvertedAbstract:
    def test_reconstructs_word_order(self):
        idx = {"The": [0], "cat": [1], "sat": [2]}
        assert ds._reconstruct_inverted(idx) == "The cat sat"

    def test_handles_a_repeated_word_at_several_positions(self):
        idx = {"the": [0, 2], "big": [1], "dog": [3]}
        assert ds._reconstruct_inverted(idx) == "the big the dog"

    @pytest.mark.parametrize("idx", [{}, None])
    def test_empty_index_yields_empty_string(self, idx):
        assert ds._reconstruct_inverted(idx) == ""


# ---------------------------------------------------------------------------
# Crossref, Semantic Scholar, DOAJ, ClinicalTrials
# ---------------------------------------------------------------------------

class TestCrossRef:
    def test_parses_items(self, stub):
        stub({"message": {"items": [{
            "DOI": "10.1/abc",
            "title": ["A linked records study"],
            "abstract": "<jats:p>Some abstract</jats:p>",
            "URL": "https://doi.org/10.1/abc",
        }]}})
        papers = ds.CrossRefService.fetch("dental", 10)
        assert len(papers) == 1
        assert "linked records" in papers[0].title

    def test_sends_no_email_in_the_user_agent(self, stub):
        calls = stub({"message": {"items": []}})
        ds.CrossRefService.fetch("dental", 5)
        assert "mailto" not in str(calls[0]["headers"])

    def test_missing_message_key_is_survivable(self, stub):
        stub({})
        assert ds.CrossRefService.fetch("dental", 10) == []


class TestSemanticScholar:
    def test_parses_data(self, stub):
        stub({"data": [{
            "paperId": "abc123",
            "title": "Machine learning in dentistry",
            "abstract": "We trained a model.",
            "url": "https://semanticscholar.org/abc123",
        }]})
        papers = ds.SemanticScholarService.fetch("dental", 10)
        assert len(papers) == 1
        assert papers[0].title == "Machine learning in dentistry"

    def test_missing_data_key_is_survivable(self, stub):
        stub({})
        assert ds.SemanticScholarService.fetch("dental", 10) == []


class TestDOAJ:
    def test_parses_bibjson(self, stub):
        stub({"results": [{
            "id": "doaj123",
            "bibjson": {
                "title": "Open access dental study",
                "abstract": "An abstract.",
                "identifier": [{"type": "doi", "id": "10.1/xyz"}],
                "link": [{"type": "fulltext", "url": "https://x.org/full"}],
            },
        }]})
        papers = ds.DOAJService.fetch("dental", 10)
        assert len(papers) == 1
        assert papers[0].url == "https://x.org/full"

    def test_falls_back_to_a_doi_url_when_no_fulltext_link(self, stub):
        stub({"results": [{
            "id": "doaj123",
            "bibjson": {"title": "T", "abstract": "A",
                        "identifier": [{"type": "doi", "id": "10.1/xyz"}], "link": []},
        }]})
        assert "10.1/xyz" in ds.DOAJService.fetch("dental", 10)[0].url

    def test_missing_bibjson_is_survivable(self, stub):
        stub({"results": [{"id": "x"}]})
        assert isinstance(ds.DOAJService.fetch("dental", 10), list)


class TestClinicalTrials:
    def test_returns_a_list_for_a_well_formed_response(self, stub):
        stub({"studies": []})
        assert ds.ClinicalTrialsService.fetch("dental", 10) == []

    def test_unexpected_shape_is_survivable(self, stub):
        stub({"nope": 1})
        assert isinstance(ds.ClinicalTrialsService.fetch("dental", 10), list)


# ---------------------------------------------------------------------------
# Every service must fail soft. A dead provider may not take the search down.
# ---------------------------------------------------------------------------

SERVICES = [
    ds.OpenAlexService, ds.CrossRefService, ds.SemanticScholarService,
    ds.DOAJService, ds.ClinicalTrialsService, ds.COREService,
    ds.SpringerService, ds.IEEEService, ds.ScopusService,
    ds.WebOfScienceService, ds.MedRxivService, ds.BioRxivService,
    ds.ArXivService,
]


@pytest.mark.parametrize("service", SERVICES, ids=lambda s: s.__name__)
class TestEveryServiceFailsSoft:
    def test_http_error_returns_a_list_not_an_exception(self, monkeypatch, service):
        monkeypatch.setattr(ds, "throttled_request",
                            lambda *a, **k: FakeResponse({}, status=500))
        assert isinstance(service.fetch("query", 5), list)

    def test_network_exception_returns_a_list(self, monkeypatch, service):
        def boom(*a, **k):
            raise ds.requests.exceptions.ConnectionError("down")
        monkeypatch.setattr(ds, "throttled_request", boom)
        monkeypatch.setattr(ds.requests, "get", boom)
        assert isinstance(service.fetch("query", 5), list)

    def test_garbage_body_returns_a_list(self, monkeypatch, service):
        monkeypatch.setattr(ds, "throttled_request",
                            lambda *a, **k: FakeResponse(None, text="<html>nope</html>"))
        assert isinstance(service.fetch("query", 5), list)

    def test_returns_paper_objects_only(self, monkeypatch, service):
        monkeypatch.setattr(ds, "throttled_request",
                            lambda *a, **k: FakeResponse({}, status=200))
        assert all(isinstance(p, Paper) for p in service.fetch("query", 5))


# ---------------------------------------------------------------------------
# Deduplication runs across every source, so a bug here changes the corpus.
# ---------------------------------------------------------------------------

def _paper(title, source="PubMed", pid="1"):
    return Paper(source=source, id=pid, title=title, abstract="", url="")


class TestDeduplicator:
    def test_identical_titles_from_different_sources_collapse(self):
        kept, dropped = Deduplicator.run([
            _paper("A study of periodontal disease", "PubMed", "1"),
            _paper("A study of periodontal disease", "Europe PMC", "2"),
        ])
        assert len(kept) == 1
        assert len(dropped) == 1

    def test_different_titles_are_both_kept(self):
        kept, _ = Deduplicator.run([_paper("Study A", pid="1"), _paper("Study B", pid="2")])
        assert len(kept) == 2

    def test_case_and_punctuation_differences_still_collapse(self):
        kept, _ = Deduplicator.run([
            _paper("A Study of Periodontal Disease.", "PubMed", "1"),
            _paper("a study of periodontal disease", "OpenAlex", "2"),
        ])
        assert len(kept) == 1

    def test_empty_input(self):
        kept, dropped = Deduplicator.run([])
        assert kept == [] and dropped == []

    def test_single_paper_is_never_dropped(self):
        kept, dropped = Deduplicator.run([_paper("Only one")])
        assert len(kept) == 1 and dropped == []

    def test_normalise_is_stable_and_case_folding(self):
        a = Deduplicator.normalize_text("The  Study, of Disease.")
        b = Deduplicator.normalize_text("the study of disease")
        assert a == b

    def test_kept_plus_dropped_accounts_for_every_input(self):
        """No paper may vanish silently during deduplication."""
        papers = [_paper("A", pid="1"), _paper("A", pid="2"), _paper("B", pid="3")]
        kept, dropped = Deduplicator.run(papers)
        assert len(kept) + len(dropped) == len(papers)


# ---------------------------------------------------------------------------
# Throttling. Providers ban clients that ignore their rate limits, and one ban
# takes out the whole search for everyone using the app.
# ---------------------------------------------------------------------------

class TestThrottle:
    def test_consecutive_requests_are_spaced_out(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(ds.time, "sleep", lambda s: sleeps.append(s))
        monkeypatch.setattr(ds.requests, "get", lambda *a, **k: FakeResponse({}))
        monkeypatch.setattr(ds, "_last_request_time", ds.time.time())
        ds.throttled_request("https://example.org/a")
        assert sleeps and sleeps[0] > 0

    def test_retries_on_timeout_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(ds.time, "sleep", lambda s: None)
        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ds.requests.exceptions.Timeout("slow")
            return FakeResponse({"ok": True})

        monkeypatch.setattr(ds.requests, "get", flaky)
        assert ds.throttled_request("https://example.org/a").json() == {"ok": True}
        assert calls["n"] == 2

    def test_gives_up_after_the_retry_budget(self, monkeypatch):
        monkeypatch.setattr(ds.time, "sleep", lambda s: None)

        def always_timeout(*a, **k):
            raise ds.requests.exceptions.Timeout("slow")

        monkeypatch.setattr(ds.requests, "get", always_timeout)
        with pytest.raises(ds.requests.exceptions.Timeout):
            ds.throttled_request("https://example.org/a", max_retries=2)
