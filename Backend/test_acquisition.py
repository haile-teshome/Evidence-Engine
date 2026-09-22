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


# ---------------------------------------------------------------------------
# The acquisition ladder after the open-access rebuild.
#
# The old PMC-PDF tier scored 0 out of 411 on a real run because NCBI answers
# that path with HTTP 200 and an HTML block page. These tests pin the shape of
# what replaced it: E-utilities full text, two more open-access indexes, an
# identifier backfill, and publisher text-mining tiers that stay dormant
# without a credential.
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, status=200, text="", content=b"", headers=None):
        self.status_code = status
        self.text = text
        self.content = content
        self.headers = headers or {}

    def json(self):
        import json as _j
        return _j.loads(self.text or "{}")


def test_efetch_rejects_metadata_only_records():
    """PMC answers for every record it holds, including ones whose publisher
    withholds text-mining rights; those come back as front matter and an
    abstract. Accepting them would feed an abstract to every downstream
    extraction while reporting the full text as acquired."""
    import api
    front_matter = "<article><front><article-meta><title>T</title></article-meta></front></article>"
    assert api._xml_body_text(front_matter) is None


def test_efetch_accepts_a_real_body():
    import api
    body = "<article><body><p>" + ("Methods and results. " * 200) + "</p></body></article>"
    text = api._xml_body_text(body)
    assert text and len(text) > 2000


def test_xml_body_text_requires_a_minimum_length():
    """A <body> holding one sentence is a stub, not a full text."""
    import api
    assert api._xml_body_text("<article><body><p>Short.</p></body></article>") is None


def test_download_pdf_rejects_an_html_block_page_served_as_200(monkeypatch):
    """The exact failure that made the old PMC tier useless: HTTP 200, HTML
    body. Status alone is not evidence of success, the magic number is."""
    import api
    monkeypatch.setattr(api.requests, "get", lambda *a, **k: _Resp(
        200, content=b"<!DOCTYPE html><title>Access blocked</title>",
        headers={"content-type": "text/html"}))
    assert api._download_pdf("https://example.org/x.pdf") is None


def test_download_pdf_accepts_a_real_pdf(monkeypatch):
    import api
    monkeypatch.setattr(api.requests, "get", lambda *a, **k: _Resp(
        200, content=b"%PDF-1.7 ...", headers={"content-type": "application/pdf"}))
    assert api._download_pdf("https://example.org/x.pdf") == b"%PDF-1.7 ..."


def test_publisher_tdm_tiers_make_no_request_without_a_credential(monkeypatch):
    """These tiers reach subscription content under the reader's institutional
    entitlement. With no credential stored they must not call out at all."""
    import api
    from request_creds import set_request_creds
    set_request_creds({})
    monkeypatch.delenv("SCOPUS_API_KEY", raising=False)
    monkeypatch.delenv("SPRINGER_API_KEY", raising=False)

    called = []
    monkeypatch.setattr(api.requests, "get", lambda *a, **k: called.append(a) or _Resp(200))
    assert api._fetch_elsevier_tdm("10.1/x") is None
    assert api._fetch_wiley_tdm("10.1/x") is None
    assert api._fetch_springer_tdm("10.1/x") is None
    assert called == []


def test_wiley_tdm_sends_the_token_it_was_given(monkeypatch):
    import api
    from request_creds import set_request_creds
    set_request_creds({"wiley": "tok-123"})
    seen = {}

    def fake_get(url, **kw):
        seen.update(kw.get("headers") or {})
        return _Resp(200, content=b"%PDF-1.4 body")
    monkeypatch.setattr(api.requests, "get", fake_get)
    got = api._fetch_wiley_tdm("10.1/x")
    set_request_creds({})
    assert got and got[0].startswith(b"%PDF")
    assert seen.get("Wiley-TDM-Client-Token") == "tok-123"


def test_ncbi_id_backfill_reads_the_converter(monkeypatch):
    """A record with a PubMed id but no DOI used to be abandoned before any
    open-access index was consulted."""
    import api
    monkeypatch.setattr(api.requests, "get", lambda *a, **k: _Resp(
        200, text='{"records": [{"pmid": "123", "doi": "10.1/x", "pmcid": "PMC9"}]}'))
    ids = api._ncbi_ids("123")
    assert ids["doi"] == "10.1/x"
    assert ids["pmcid"] == "PMC9"


def test_ncbi_id_backfill_returns_nothing_on_an_error_record(monkeypatch):
    import api
    monkeypatch.setattr(api.requests, "get", lambda *a, **k: _Resp(
        200, text='{"records": [{"status": "error", "errmsg": "invalid article id"}]}'))
    assert api._ncbi_ids("nope") == {}


def test_libkey_makes_no_request_without_both_credentials(monkeypatch):
    """The id alone or the token alone resolves nothing, so neither should
    produce a call. A library without LibKey pays nothing for this tier."""
    import api
    from request_creds import set_request_creds
    called = []
    monkeypatch.setattr(api.requests, "get", lambda *a, **k: called.append(a) or _Resp(200))

    for creds in ({}, {"libkey_id": "42"}, {"libkey": "tok"}):
        set_request_creds(creds)
        assert api._libkey_lookup("10.1/x") == {}
    set_request_creds({})
    assert called == []


def test_libkey_returns_the_entitled_links(monkeypatch):
    import api
    from request_creds import set_request_creds
    set_request_creds({"libkey_id": "42", "libkey": "tok"})
    seen = {}

    def fake_get(url, **kw):
        seen["url"] = url
        seen["params"] = kw.get("params")
        return _Resp(200, text='{"data": {"fullTextFile": "https://x/a.pdf",'
                               ' "contentLocation": "https://x/article"}}')
    monkeypatch.setattr(api.requests, "get", fake_get)
    got = api._libkey_lookup("10.1/x")
    set_request_creds({})
    assert got["pdf"] == "https://x/a.pdf"
    assert got["article"] == "https://x/article"
    assert "/libraries/42/articles/doi/" in seen["url"]
    assert seen["params"]["access_token"] == "tok"


def test_libkey_survives_a_library_that_does_not_hold_the_article(monkeypatch):
    """A 404 means this library has no entitled copy, which is an answer, not
    a failure: the ladder must fall through rather than raise."""
    import api
    from request_creds import set_request_creds
    set_request_creds({"libkey_id": "42", "libkey": "tok"})
    monkeypatch.setattr(api.requests, "get", lambda *a, **k: _Resp(404, text="{}"))
    assert api._libkey_lookup("10.1/x") == {}
    assert api._fetch_libkey_pdf("10.1/x") is None
    set_request_creds({})


def test_libkey_link_is_offered_even_when_the_download_fails(monkeypatch):
    """Off the institution's network we cannot fetch the entitled file, but the
    reader can open it. Losing the link there would waste the whole lookup."""
    import api
    from request_creds import set_request_creds
    set_request_creds({"libkey_id": "42", "libkey": "tok"})
    monkeypatch.setattr(api, "_libkey_lookup",
                        lambda doi: {"pdf": "https://x/a.pdf", "article": "https://x/art"})
    monkeypatch.setattr(api, "_download_pdf", lambda url: None)          # entitlement refused
    monkeypatch.setattr(api, "_fetch_epmc_fulltext_text", lambda pid: None)
    monkeypatch.setattr(api, "_fetch_pmc_efetch_text", lambda pmcid: None)
    monkeypatch.setattr(api, "_ncbi_ids", lambda ident: {})
    monkeypatch.setattr(api, "_fetch_oa_pdf", lambda doi, title="": (None, "", "closed"))
    monkeypatch.setattr(api, "_fetch_s2_oa_pdf", lambda doi: None)
    monkeypatch.setattr(api, "_fetch_openaire_pdf", lambda doi: None)

    out = api.fulltext_fetch(api.FullTextRequest(
        Title="T", URL="", Source="PubMed", paper_id="1", doi="10.1/x"))
    set_request_creds({})
    assert out["status"] == "missing"
    assert out["links"]["libkey"] == "https://x/a.pdf"
    assert out["links"]["libkey_article"] == "https://x/art"
