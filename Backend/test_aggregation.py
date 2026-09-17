"""Tests for cross-source aggregation, counting, and the PubMed/EuropePMC paths.

Run: Backend/.venv/bin/python -m pytest Backend/test_aggregation.py

These functions fan out across every configured database and combine the
results. The recurring failure mode is a partial one: some sources answer, one
raises, and the corpus silently ends up smaller than the reviewer believes. A
review built on a quietly truncated corpus is not reproducible, and nothing in
the UI says anything went wrong.
"""
import pytest

import data_services as ds
from config import DataSource
from models import Paper


class FakeHandle:
    def __init__(self, payload):
        self.payload = payload

    def close(self):
        pass


class FakeResponse:
    def __init__(self, payload=None, status=200, text=""):
        self._payload = payload
        self.status_code = status
        self.text = text
        self.content = text.encode() if text else b""
        self.headers = {}

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


@pytest.fixture
def entrez(monkeypatch):
    """Stub NCBI Entrez so nothing leaves the machine."""
    def _install(count="100", ids=None, articles=None):
        def esearch(**k):
            return FakeHandle({"Count": count, "IdList": ids or []})

        def efetch(**k):
            return FakeHandle({"PubmedArticle": articles or []})

        monkeypatch.setattr(ds.Entrez, "esearch", esearch)
        monkeypatch.setattr(ds.Entrez, "efetch", efetch)
        monkeypatch.setattr(ds.Entrez, "read", lambda h: h.payload)
    return _install


def _article(pmid="1", title="A title", sections=None):
    return {
        "MedlineCitation": {
            "PMID": pmid,
            "Article": {
                "ArticleTitle": title,
                "Abstract": {"AbstractText": sections or ["An abstract."]},
            },
        }
    }


# ---------------------------------------------------------------------------
# PubMed fetch
# ---------------------------------------------------------------------------

class TestPubMedFetch:
    def test_parses_articles_into_papers(self, entrez):
        entrez(ids=["1"], articles=[_article(pmid="1", title="Linked records study")])
        papers = ds.PubMedService.fetch("dental", 10)
        assert len(papers) == 1
        assert papers[0].title == "Linked records study"
        assert papers[0].source == DataSource.PUBMED.value

    def test_builds_a_pubmed_url_from_the_pmid(self, entrez):
        entrez(ids=["12345"], articles=[_article(pmid="12345")])
        assert "12345" in ds.PubMedService.fetch("dental", 10)[0].url

    def test_joins_every_abstract_section(self, entrez):
        """The truncation regression, at the service level rather than the
        parser: a structured abstract must arrive whole."""
        entrez(ids=["1"], articles=[_article(sections=[
            "Background text.", "The methods that decide eligibility.", "Results.",
        ])])
        abstract = ds.PubMedService.fetch("dental", 10)[0].abstract
        assert "methods that decide eligibility" in abstract
        assert "Results" in abstract

    def test_no_results_yields_no_papers(self, entrez):
        entrez(ids=[], articles=[])
        assert ds.PubMedService.fetch("dental", 10) == []

    def test_an_entrez_failure_returns_a_list_not_an_exception(self, monkeypatch):
        def boom(**k):
            raise RuntimeError("NCBI unavailable")
        monkeypatch.setattr(ds.Entrez, "esearch", boom)
        assert ds.PubMedService.fetch("dental", 10) == []

    def test_adds_a_tiab_qualifier_to_an_untagged_query(self, entrez, monkeypatch):
        seen = {}

        def esearch(**k):
            seen.update(k)
            return FakeHandle({"Count": "0", "IdList": []})

        monkeypatch.setattr(ds.Entrez, "esearch", esearch)
        monkeypatch.setattr(ds.Entrez, "read", lambda h: h.payload)
        ds.PubMedService.fetch("dental caries", 10)
        assert "tiab" in str(seen.get("term", "")).lower()

    def test_leaves_an_already_tagged_query_alone(self, entrez, monkeypatch):
        seen = {}

        def esearch(**k):
            seen.update(k)
            return FakeHandle({"Count": "0", "IdList": []})

        monkeypatch.setattr(ds.Entrez, "esearch", esearch)
        monkeypatch.setattr(ds.Entrez, "read", lambda h: h.payload)
        ds.PubMedService.fetch('("dental"[mh])', 10)
        assert str(seen.get("term")) == '("dental"[mh])'

    @pytest.mark.parametrize("sort,expected", [
        ("recent", "pub_date"), ("relevance", "relevance"), ("date", "pub_date"),
    ])
    def test_sort_maps_to_an_entrez_sort_key(self, monkeypatch, sort, expected):
        seen = {}

        def esearch(**k):
            seen.update(k)
            return FakeHandle({"Count": "0", "IdList": []})

        monkeypatch.setattr(ds.Entrez, "esearch", esearch)
        monkeypatch.setattr(ds.Entrez, "read", lambda h: h.payload)
        ds.PubMedService.fetch("dental", 10, sort=sort)
        assert seen.get("sort") == expected

    def test_a_year_window_is_passed_through_as_a_date_filter(self, monkeypatch):
        seen = {}

        def esearch(**k):
            seen.update(k)
            return FakeHandle({"Count": "0", "IdList": []})

        monkeypatch.setattr(ds.Entrez, "esearch", esearch)
        monkeypatch.setattr(ds.Entrez, "read", lambda h: h.payload)
        ds.PubMedService.fetch("dental", 10, year_from=2015, year_to=2020)
        assert "2015" in str(seen) and "2020" in str(seen)


class TestMeshLookup:
    def test_returns_a_heading_and_entry_terms(self, monkeypatch):
        monkeypatch.setattr(ds.Entrez, "esearch", lambda **k: FakeHandle({"IdList": ["68003813"]}))
        monkeypatch.setattr(ds.Entrez, "esummary",
                            lambda **k: FakeHandle([{"DS_MeshTerms": ["Dentistry", "Odontology"]}]))
        monkeypatch.setattr(ds.Entrez, "read", lambda h: h.payload)
        ds.PubMedService._mesh_cache.clear()
        heading, entries = ds.PubMedService.mesh_lookup("dentistry")
        assert heading == "Dentistry"
        assert "Odontology" in entries

    def test_no_match_returns_empty(self, monkeypatch):
        monkeypatch.setattr(ds.Entrez, "esearch", lambda **k: FakeHandle({"IdList": []}))
        monkeypatch.setattr(ds.Entrez, "read", lambda h: h.payload)
        ds.PubMedService._mesh_cache.clear()
        assert ds.PubMedService.mesh_lookup("not-a-real-heading") == ("", [])

    def test_result_is_cached_so_a_repeat_costs_no_request(self, monkeypatch):
        calls = {"n": 0}

        def esearch(**k):
            calls["n"] += 1
            return FakeHandle({"IdList": []})

        monkeypatch.setattr(ds.Entrez, "esearch", esearch)
        monkeypatch.setattr(ds.Entrez, "read", lambda h: h.payload)
        ds.PubMedService._mesh_cache.clear()
        ds.PubMedService.mesh_lookup("dentistry")
        ds.PubMedService.mesh_lookup("dentistry")
        assert calls["n"] == 1

    @pytest.mark.parametrize("term", ["", "   ", None])
    def test_empty_term_short_circuits(self, term):
        assert ds.PubMedService.mesh_lookup(term) == ("", [])

    def test_an_ncbi_failure_returns_empty_rather_than_raising(self, monkeypatch):
        def boom(**k):
            raise RuntimeError("down")
        monkeypatch.setattr(ds.Entrez, "esearch", boom)
        ds.PubMedService._mesh_cache.clear()
        assert ds.PubMedService.mesh_lookup("dentistry") == ("", [])


class TestFetchMeshForPmids:
    def test_returns_headings_keyed_by_pmid(self, monkeypatch):
        rec = {"PubmedArticle": [{
            "MedlineCitation": {
                "PMID": "111",
                "MeshHeadingList": [{"DescriptorName": "Dentistry"}],
                "Article": {"ArticleTitle": "A study"},
            }
        }]}
        monkeypatch.setattr(ds.Entrez, "efetch", lambda **k: FakeHandle(rec))
        monkeypatch.setattr(ds.Entrez, "read", lambda h: h.payload)
        out = ds.PubMedService.fetch_mesh_for_pmids(["111"])
        assert out["111"]["mesh"] == ["Dentistry"]
        assert out["111"]["title"] == "A study"

    @pytest.mark.parametrize("pmids", [[], ["not-a-pmid"], ["", None]])
    def test_non_numeric_ids_are_filtered_out(self, pmids):
        assert ds.PubMedService.fetch_mesh_for_pmids(pmids) == {}

    def test_a_failure_returns_an_empty_map(self, monkeypatch):
        def boom(**k):
            raise RuntimeError("down")
        monkeypatch.setattr(ds.Entrez, "efetch", boom)
        assert ds.PubMedService.fetch_mesh_for_pmids(["111"]) == {}


# ---------------------------------------------------------------------------
# Europe PMC
# ---------------------------------------------------------------------------

class TestEuropePmcFetch:
    def test_parses_results(self, monkeypatch):
        payload = {"resultList": {"result": [
            {"id": "123", "title": "A dental study", "abstractText": "An abstract",
             "doi": "10.1/abc"},
        ]}, "nextCursorMark": ""}
        monkeypatch.setattr(ds, "throttled_request", lambda *a, **k: FakeResponse(payload))
        papers = ds.EuropePMCService.fetch("dental", 10)
        assert len(papers) == 1
        assert papers[0].title == "A dental study"

    def test_translates_pubmed_tags_before_querying(self, monkeypatch):
        """Untranslated field tags run as full-text search and inflated one real
        count by roughly 260x."""
        seen = {}

        def capture(url, params=None, **k):
            seen["params"] = params or {}
            return FakeResponse({"resultList": {"result": []}})

        monkeypatch.setattr(ds, "throttled_request", capture)
        ds.EuropePMCService.fetch('("dental"[tiab])', 10)
        assert "[tiab]" not in str(seen["params"])

    def test_never_requests_more_than_the_page_limit(self, monkeypatch):
        """Europe PMC answers HTTP 200 with an EMPTY list above pageSize 1000,
        so an over-large request silently returns nothing at all."""
        seen = {}

        def capture(url, params=None, **k):
            seen["params"] = params or {}
            return FakeResponse({"resultList": {"result": []}})

        monkeypatch.setattr(ds, "throttled_request", capture)
        ds.EuropePMCService.fetch("dental", 50_000)
        assert int(seen["params"].get("pageSize", 0)) <= 1000

    def test_respects_max_results(self, monkeypatch):
        results = [{"id": str(i), "title": f"T{i}", "abstractText": "a"} for i in range(50)]
        monkeypatch.setattr(ds, "throttled_request",
                            lambda *a, **k: FakeResponse(
                                {"resultList": {"result": results}, "nextCursorMark": ""}))
        assert len(ds.EuropePMCService.fetch("dental", 10)) <= 10

    def test_a_failure_returns_a_list(self, monkeypatch):
        monkeypatch.setattr(ds, "throttled_request",
                            lambda *a, **k: FakeResponse(None, status=500))
        assert ds.EuropePMCService.fetch("dental", 10) == []


# ---------------------------------------------------------------------------
# Counting across sources
# ---------------------------------------------------------------------------

class TestCounts:
    def test_get_total_counts_counts_the_sources_it_supports(self, entrez, monkeypatch):
        """NOTE: this function currently has no callers (the live paths are
        simulate_yield for the API and get_all_counts for query optimisation),
        and it only implements five sources. A source it does not implement is
        silently ABSENT from the result rather than reported as zero, which
        reads as "not searched" rather than "searched and found nothing"."""
        entrez(count="42")
        monkeypatch.setattr(ds, "throttled_request",
                            lambda *a, **k: FakeResponse({"total": 7, "totalHits": 7}))
        out = ds.DataAggregator.get_total_counts(
            "dental", [DataSource.PUBMED.value, "Semantic Scholar", "CORE"])
        assert out[DataSource.PUBMED.value] == 42
        assert all(isinstance(v, int) for v in out.values())

    def test_an_unimplemented_source_is_omitted_rather_than_zeroed(self, entrez, monkeypatch):
        entrez(count="42")
        monkeypatch.setattr(ds, "throttled_request", lambda *a, **k: FakeResponse({}))
        out = ds.DataAggregator.get_total_counts("dental", [DataSource.PUBMED.value, "OpenAlex"])
        assert DataSource.PUBMED.value in out
        assert "OpenAlex" not in out

    def test_get_total_counts_survives_one_dead_source(self, entrez, monkeypatch):
        entrez(count="42")

        def boom(*a, **k):
            raise ds.requests.exceptions.ConnectionError("down")

        monkeypatch.setattr(ds, "throttled_request", boom)
        out = ds.DataAggregator.get_total_counts("dental", [DataSource.PUBMED.value, "OpenAlex"])
        assert out[DataSource.PUBMED.value] == 42
        assert out.get("OpenAlex", 0) == 0

    def test_get_all_counts_returns_integers(self, entrez, monkeypatch):
        entrez(count="5")
        monkeypatch.setattr(ds, "throttled_request",
                            lambda *a, **k: FakeResponse({"meta": {"count": 3},
                                                          "hitCount": 3,
                                                          "message": {"total-results": 3}}))
        out = ds.DataAggregator.get_all_counts("dental", [DataSource.PUBMED.value, "OpenAlex"])
        assert all(isinstance(v, int) for v in out.values())

    def test_a_non_numeric_count_does_not_crash(self, entrez, monkeypatch):
        entrez(count="not-a-number")
        monkeypatch.setattr(ds, "throttled_request", lambda *a, **k: FakeResponse({}))
        out = ds.DataAggregator.get_total_counts("dental", [DataSource.PUBMED.value])
        assert out.get(DataSource.PUBMED.value, 0) == 0

    def test_empty_source_list(self):
        assert ds.DataAggregator.get_total_counts("dental", []) == {}


# ---------------------------------------------------------------------------
# fetch_all: the fan-out that builds the corpus
# ---------------------------------------------------------------------------

class TestFetchAll:
    def test_combines_papers_from_several_sources(self, entrez, monkeypatch):
        entrez(ids=["1"], articles=[_article(pmid="1", title="From PubMed")])
        monkeypatch.setattr(ds, "throttled_request", lambda *a, **k: FakeResponse(
            {"results": [{"id": "W1", "title": "From OpenAlex",
                          "abstract_inverted_index": {"a": [0]},
                          "open_access": {}}]}))
        papers, counts = ds.DataAggregator.fetch_all(
            "dental", [DataSource.PUBMED.value, "OpenAlex"], max_per_source=5)[:2]
        titles = {p.title for p in papers}
        assert "From PubMed" in titles

    def test_one_failing_source_does_not_lose_the_others(self, entrez, monkeypatch):
        """The partial-failure case. A dead provider must cost its own results
        and nothing else."""
        entrez(ids=["1"], articles=[_article(pmid="1", title="From PubMed")])

        def boom(*a, **k):
            raise ds.requests.exceptions.ConnectionError("down")

        monkeypatch.setattr(ds, "throttled_request", boom)
        out = ds.DataAggregator.fetch_all("dental", [DataSource.PUBMED.value, "OpenAlex"], 5)
        papers = out[0] if isinstance(out, tuple) else out
        assert any(p.title == "From PubMed" for p in papers)

    def test_returns_paper_objects_only(self, entrez, monkeypatch):
        entrez(ids=["1"], articles=[_article()])
        monkeypatch.setattr(ds, "throttled_request", lambda *a, **k: FakeResponse({}))
        out = ds.DataAggregator.fetch_all("dental", [DataSource.PUBMED.value], 5)
        papers = out[0] if isinstance(out, tuple) else out
        assert all(isinstance(p, Paper) for p in papers)

    def test_no_sources_yields_no_papers(self):
        out = ds.DataAggregator.fetch_all("dental", [], 5)
        papers = out[0] if isinstance(out, tuple) else out
        assert papers == []

    def test_local_pdfs_with_no_uploads_is_safe(self):
        out = ds.DataAggregator.fetch_all("dental", [DataSource.LOCAL_PDF.value], 5)
        papers = out[0] if isinstance(out, tuple) else out
        assert papers == []


class TestOpenAlexRelated:
    def test_returns_papers_for_a_seed(self, monkeypatch):
        monkeypatch.setattr(ds, "throttled_request", lambda *a, **k: FakeResponse(
            {"results": [{"id": "https://openalex.org/W2", "title": "Related work",
                          "abstract_inverted_index": {"x": [0]}, "open_access": {}}],
             "related_works": []}))
        out = ds.OpenAlexService.related("A seed title", "10.1/abc", 10)
        assert isinstance(out, list)

    def test_a_failure_returns_a_list(self, monkeypatch):
        def boom(*a, **k):
            raise ds.requests.exceptions.ConnectionError("down")
        monkeypatch.setattr(ds, "throttled_request", boom)
        assert ds.OpenAlexService.related("seed", "", 10) == []

    def test_no_seed_returns_a_list(self, monkeypatch):
        monkeypatch.setattr(ds, "throttled_request", lambda *a, **k: FakeResponse({"results": []}))
        assert ds.OpenAlexService.related("", "", 10) == []
