"""Tests for the open-access acquisition ladder. No network.

Run: Backend/.venv/bin/python -m pytest Backend/test_acquisition.py

The governing rule these pin: a retrieval failure must say WHY, and a
configuration failure must never be indistinguishable from a legitimately
paywalled paper. Violating that rule hid a totally dead acquisition tier.
"""
import pytest

import api
from request_creds import set_request_creds


class FakeResponse:
    def __init__(self, status=200, content=b"", headers=None, json_data=None, url=""):
        self.status_code = status
        self.content = content
        self.headers = headers or {}
        self.text = content.decode("utf-8", "replace") if isinstance(content, bytes) else str(content)
        self._json = json_data
        self.url = url

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


PDF_BYTES = b"%PDF-1.4 fake pdf body"


def _openalex(is_oa, pdf_url=None, landing=None):
    """A minimal OpenAlex work payload."""
    best = {}
    if pdf_url:
        best["pdf_url"] = pdf_url
    if landing:
        best["landing_page_url"] = landing
        best["is_oa"] = is_oa
    return {"open_access": {"is_oa": is_oa, "oa_url": landing}, "best_oa_location": best or None,
            "locations": []}


# ---------------------------------------------------------------------------
# OA classification. The status drives the reason the reviewer is shown, so a
# wrong label sends them to the wrong place to get the paper by hand.
# ---------------------------------------------------------------------------

class TestOpenAlexClassification:
    def teardown_method(self):
        set_request_creds({})
        api._openalex_last = 0.0

    def _patch(self, monkeypatch, payload, status=200):
        monkeypatch.setattr(api.requests, "get",
                            lambda *a, **k: FakeResponse(status=status, json_data=payload))

    def test_direct_pdf_is_classified_pdf(self, monkeypatch):
        self._patch(monkeypatch, _openalex(True, pdf_url="https://x.org/a.pdf"))
        urls, status = api._openalex_oa_locations("10.1/abc")
        assert status == "pdf"
        assert "https://x.org/a.pdf" in urls

    def test_landing_page_only_is_classified_landing(self, monkeypatch):
        self._patch(monkeypatch, _openalex(True, landing="https://x.org/article"))
        _, status = api._openalex_oa_locations("10.1/abc")
        assert status == "landing"

    def test_not_open_access_is_classified_closed(self, monkeypatch):
        self._patch(monkeypatch, _openalex(False))
        _, status = api._openalex_oa_locations("10.1/abc")
        assert status == "closed"

    def test_http_error_is_unknown_not_closed(self, monkeypatch):
        """An API outage must not be reported to the user as 'paywalled'."""
        self._patch(monkeypatch, None, status=503)
        _, status = api._openalex_oa_locations("10.1/abc")
        assert status == "unknown"

    def test_network_exception_is_unknown(self, monkeypatch):
        def boom(*a, **k):
            raise api.requests.exceptions.Timeout("timed out")
        monkeypatch.setattr(api.requests, "get", boom)
        _, status = api._openalex_oa_locations("10.1/abc")
        assert status == "unknown"

    def test_no_doi_and_no_title_short_circuits(self):
        assert api._openalex_oa_locations("", "") == ([], "unknown")

    def test_pdf_urls_are_ordered_before_landing_pages(self, monkeypatch):
        payload = {"open_access": {"is_oa": True, "oa_url": "https://x.org/land"},
                   "best_oa_location": {"pdf_url": "https://x.org/a.pdf",
                                        "landing_page_url": "https://x.org/land",
                                        "is_oa": True},
                   "locations": []}
        self._patch(monkeypatch, payload)
        urls, _ = api._openalex_oa_locations("10.1/abc")
        assert urls[0].endswith(".pdf")


# ---------------------------------------------------------------------------
# Unpaywall. The incident in full: ENTREZ_EMAIL was a placeholder, Unpaywall
# answered 422 to every request, and the caller returned None on any non-200.
# A dead tier and a paywalled paper produced byte-identical results.
# ---------------------------------------------------------------------------

class TestUnpaywallRequiresAnEmail:
    def teardown_method(self):
        set_request_creds({})

    def test_is_skipped_entirely_without_a_contact_email(self, monkeypatch):
        """No address means no call at all, rather than a guaranteed 422."""
        called = []
        monkeypatch.setattr(api.requests, "get",
                            lambda *a, **k: called.append(a) or FakeResponse(422))
        set_request_creds({})
        assert api._fetch_unpaywall_pdf("10.1/abc") is None
        assert called == [], "must not call Unpaywall without a usable address"

    def test_is_skipped_for_a_placeholder_address(self, monkeypatch):
        called = []
        monkeypatch.setattr(api.requests, "get",
                            lambda *a, **k: called.append(a) or FakeResponse(422))
        set_request_creds({"contact_email": "someone@example.com"})
        assert api._fetch_unpaywall_pdf("10.1/abc") is None
        assert called == []

    def test_422_is_logged_not_silently_swallowed(self, monkeypatch, capsys):
        """The failure must leave a trace. Silence is what hid this for months."""
        set_request_creds({"contact_email": "j.smith@ucsf.edu"})
        monkeypatch.setattr(api.requests, "get",
                            lambda *a, **k: FakeResponse(422, content=b"bad email"))
        assert api._fetch_unpaywall_pdf("10.1/abc") is None
        assert "422" in capsys.readouterr().out

    def test_no_doi_short_circuits(self):
        set_request_creds({"contact_email": "j.smith@ucsf.edu"})
        assert api._fetch_unpaywall_pdf("") is None


# ---------------------------------------------------------------------------
# Landing-page follow. Publishers routinely advertise the file only through the
# citation_pdf_url meta tag, which is why a record can be open access and still
# yield no downloadable PDF from the index alone.
# ---------------------------------------------------------------------------

class TestLandingPageFollow:
    def test_finds_pdf_via_citation_pdf_url_meta_tag(self, monkeypatch):
        html = b'<html><head><meta name="citation_pdf_url" content="https://x.org/full.pdf"></head></html>'
        def fake_get(url, **k):
            if url.endswith(".pdf"):
                return FakeResponse(200, PDF_BYTES, {"content-type": "application/pdf"})
            return FakeResponse(200, html, {"content-type": "text/html"}, url="https://x.org/article")
        monkeypatch.setattr(api.requests, "get", fake_get)
        assert api._pdf_from_landing_page("https://x.org/article") == PDF_BYTES

    def test_returns_the_page_when_it_is_already_a_pdf(self, monkeypatch):
        monkeypatch.setattr(api.requests, "get",
                            lambda *a, **k: FakeResponse(200, PDF_BYTES,
                                                         {"content-type": "application/pdf"}))
        assert api._pdf_from_landing_page("https://x.org/a.pdf") == PDF_BYTES

    def test_403_returns_none(self, monkeypatch):
        """Publishers that block automated download are a wall, not an error."""
        monkeypatch.setattr(api.requests, "get", lambda *a, **k: FakeResponse(403, b"denied"))
        assert api._pdf_from_landing_page("https://x.org/article") is None

    def test_html_with_no_pdf_link_returns_none(self, monkeypatch):
        monkeypatch.setattr(api.requests, "get",
                            lambda *a, **k: FakeResponse(200, b"<html>nothing here</html>",
                                                         {"content-type": "text/html"},
                                                         url="https://x.org/article"))
        assert api._pdf_from_landing_page("https://x.org/article") is None

    @pytest.mark.parametrize("url", ["", "not-a-url", "ftp://x.org/a.pdf"])
    def test_non_http_urls_short_circuit(self, url):
        assert api._pdf_from_landing_page(url) is None

    def test_network_exception_returns_none(self, monkeypatch):
        def boom(*a, **k):
            raise api.requests.exceptions.ConnectionError("no route")
        monkeypatch.setattr(api.requests, "get", boom)
        assert api._pdf_from_landing_page("https://x.org/article") is None
