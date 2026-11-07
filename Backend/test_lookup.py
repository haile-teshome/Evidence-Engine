"""Tests for resolving a pasted list of papers. No network: every call is stubbed.

Run: Backend/.venv/bin/python -m pytest Backend/test_lookup.py

A pasted line is the least structured input the app takes, so the cases that
matter are the ones where getting it wrong is quiet: a DOI misread as a title
searches for the literal string and returns the wrong paper, and a lookup that
fails has to be reported rather than dropped, because the reviewer has to go and
find those by hand.
"""
import pytest

import api


class TestClassify:
    @pytest.mark.parametrize("value", [
        "10.1001/jama.2020.1234",
        "10.1016/S0140-6736(20)30183-5",
        "https://doi.org/10.1001/jama.2020.1234",
        "http://dx.doi.org/10.1001/jama.2020.1234",
        "10.1001/JAMA.2020.1234",
    ])
    def test_dois(self, value):
        assert api._classify_query(value) == "doi"

    @pytest.mark.parametrize("value", ["32150360", "12345", "123456789"])
    def test_pmids(self, value):
        assert api._classify_query(value) == "pmid"

    @pytest.mark.parametrize("value", [
        "Effect of X on Y: a randomised trial",
        "10 reasons trials fail",          # starts with digits, is not a DOI
        "2020",                            # a stray year, not a 4-digit PMID
        "1234567890123",                   # too long to be a PMID
        "Smith et al. 2020",
    ])
    def test_titles(self, value):
        assert api._classify_query(value) == "title"

    def test_surrounding_whitespace_does_not_change_the_kind(self):
        assert api._classify_query("  10.1001/jama.2020.1234  ") == "doi"


class TestLookupOne:
    def _work(self, **over):
        w = {
            "id": "https://openalex.org/W123",
            "display_name": "A trial of something",
            "doi": "https://doi.org/10.1001/jama.2020.1234",
            "publication_year": 2020,
            "abstract_inverted_index": {"Background": [0], "text": [1]},
            "authorships": [{"author": {"display_name": "A Smith"}},
                            {"author": {"display_name": "B Jones"}}],
            "open_access": {"oa_url": "https://example.test/paper.pdf"},
        }
        w.update(over)
        return w

    def test_title_is_resolved_and_mapped(self, monkeypatch):
        monkeypatch.setattr(api, "_openalex_resolve_work", lambda t, i: self._work())
        rec = api._lookup_one("A trial of something")
        assert rec["title"] == "A trial of something"
        assert rec["abstract"] == "Background text"
        assert rec["authors"] == "A Smith, B Jones"
        assert rec["year"] == 2020
        assert rec["source"] == "Imported list"

    def test_what_the_reviewer_typed_is_kept(self, monkeypatch):
        """A title search can match the wrong paper; show what was asked for."""
        monkeypatch.setattr(api, "_openalex_resolve_work", lambda t, i: self._work())
        assert api._lookup_one("  my typed title  ")["matched_from"] == "my typed title"

    def test_a_doi_is_looked_up_as_a_doi_not_searched_as_text(self, monkeypatch):
        seen = {}

        def fake(title, paper_id):
            seen["title"], seen["id"] = title, paper_id
            return self._work()

        monkeypatch.setattr(api, "_openalex_resolve_work", fake)
        api._lookup_one("https://doi.org/10.1001/jama.2020.1234")
        assert seen["id"] == "10.1001/jama.2020.1234"
        assert seen["title"] == ""

    def test_missing_abstract_falls_back_to_europe_pmc(self, monkeypatch):
        monkeypatch.setattr(api, "_openalex_resolve_work",
                            lambda t, i: self._work(abstract_inverted_index=None))
        monkeypatch.setattr(api, "_abstract_via_epmc", lambda title, ident: "From EPMC")
        assert api._lookup_one("something")["abstract"] == "From EPMC"

    def test_a_failing_fallback_is_not_fatal(self, monkeypatch):
        monkeypatch.setattr(api, "_openalex_resolve_work",
                            lambda t, i: self._work(abstract_inverted_index=None))

        def boom(title, ident):
            raise RuntimeError("epmc down")

        monkeypatch.setattr(api, "_abstract_via_epmc", boom)
        assert api._lookup_one("something")["abstract"] == ""

    def test_nothing_found_is_none_not_an_invented_record(self, monkeypatch):
        monkeypatch.setattr(api, "_openalex_resolve_work", lambda t, i: None)
        assert api._lookup_one("no such paper anywhere") is None


class TestEndpoint:
    def test_found_and_missing_are_both_reported(self, monkeypatch):
        def fake(q):
            return {"id": "W1", "title": q} if q.startswith("good") else None

        monkeypatch.setattr(api, "_lookup_one", fake)
        from fastapi.testclient import TestClient
        c = TestClient(api.app)
        r = c.post("/api/papers/lookup", json={"queries": ["good one", "bad one", "good two"]})
        assert r.status_code == 200
        body = r.json()
        assert len(body["papers"]) == 2
        assert body["missing"] == ["bad one"]
        assert body["requested"] == 3

    def test_blank_lines_are_ignored_not_counted(self, monkeypatch):
        monkeypatch.setattr(api, "_lookup_one", lambda q: {"id": "W1", "title": q})
        from fastapi.testclient import TestClient
        c = TestClient(api.app)
        body = c.post("/api/papers/lookup", json={"queries": ["a", "", "   ", "b"]}).json()
        assert body["requested"] == 2
        assert len(body["papers"]) == 2

    def test_one_bad_lookup_does_not_sink_the_batch(self, monkeypatch):
        def fake(q):
            if q == "explodes":
                raise RuntimeError("upstream broke")
            return {"id": "W1", "title": q}

        monkeypatch.setattr(api, "_lookup_one", fake)
        from fastapi.testclient import TestClient
        c = TestClient(api.app)
        body = c.post("/api/papers/lookup", json={"queries": ["fine", "explodes", "also fine"]}).json()
        assert len(body["papers"]) == 2
        assert body["missing"] == ["explodes"]

    def test_the_cap_bounds_the_work(self, monkeypatch):
        monkeypatch.setattr(api, "_lookup_one", lambda q: {"id": "W1", "title": q})
        from fastapi.testclient import TestClient
        c = TestClient(api.app)
        body = c.post("/api/papers/lookup",
                      json={"queries": [f"q{i}" for i in range(50)], "cap": 10}).json()
        assert len(body["papers"]) == 10
