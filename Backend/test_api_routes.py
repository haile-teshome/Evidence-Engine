"""HTTP contract tests for the API surface. No network, no model, throwaway DB.

Run: Backend/.venv/bin/python -m pytest Backend/test_api_routes.py

These do not test whether screening is *correct* — that lives in
test_screening_logic.py. They test that the HTTP layer behaves: validation
rejects bad input with 422 rather than 500, auth denies before it reads, and a
handler that hits a dead dependency degrades instead of returning a stack trace
to the browser.

The credential-leak cases matter most. Keys arrive as headers on every request,
so a handler that echoed one back, or logged it, would expose it broadly.
"""
import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Liveness and read-only endpoints
# ---------------------------------------------------------------------------

class TestReadOnlyEndpoints:
    def test_health_is_ok(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_instruments_lists_the_rob_tools(self, client):
        r = client.get("/api/instruments")
        assert r.status_code == 200
        assert isinstance(r.json(), (list, dict))

    def test_keys_status_reports_without_revealing_anything(self, client):
        r = client.get("/api/keys/status", headers={"x-db-core-key": "SECRET-VALUE"})
        assert r.status_code == 200
        assert "SECRET-VALUE" not in r.text, "a status endpoint must never echo a key"

    def test_unknown_route_is_404(self, client):
        assert client.get("/api/definitely-not-a-route").status_code == 404

    def test_missing_cached_pdf_is_404_not_500(self, client):
        assert client.get("/api/fulltext/pdf/no-such-key").status_code == 404

    def test_pdf_key_cannot_escape_the_cache_directory(self, client):
        """Path traversal in a user-supplied key must not read arbitrary files."""
        r = client.get("/api/fulltext/pdf/..%2F..%2F..%2Fetc%2Fpasswd")
        assert r.status_code in (400, 404)
        assert "root:" not in r.text


# ---------------------------------------------------------------------------
# Request validation. A malformed body is the client's fault (422), never a
# server crash (500).
# ---------------------------------------------------------------------------

VALIDATED_POSTS = [
    "/api/screen/abstract",
    "/api/fulltext/fetch",
    "/api/papers/dedupe",
    "/api/extract/text",
    "/api/framework/detect",
    "/api/meta/pool",
]


@pytest.mark.parametrize("path", VALIDATED_POSTS)
class TestRequestValidation:
    def test_empty_body_is_rejected_cleanly(self, client, path):
        assert client.post(path, json={}).status_code in (400, 422)

    def test_wrong_types_are_rejected_cleanly(self, client, path):
        r = client.post(path, json={"paper": "not-an-object", "papers": 42})
        assert r.status_code in (400, 422)

    def test_no_body_at_all_is_rejected_cleanly(self, client, path):
        assert client.post(path).status_code in (400, 422)

    def test_validation_failure_never_returns_500(self, client, path):
        for body in ({}, {"x": None}, {"papers": None}):
            assert client.post(path, json=body).status_code != 500


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestAuth:
    def test_me_without_a_token_does_not_leak_a_user(self, client):
        r = client.get("/api/auth/me")
        assert r.status_code in (200, 401, 403)
        if r.status_code == 200:
            assert not (r.json() or {}).get("email")

    def test_login_with_no_credentials_is_rejected(self, client):
        assert client.post("/api/auth/login", json={}).status_code in (400, 401, 422)

    def test_login_with_wrong_credentials_is_rejected(self, client):
        r = client.post("/api/auth/login",
                        json={"email": "nobody@example.com", "password": "wrong"})
        assert r.status_code in (400, 401, 403, 422)

    def test_a_rejected_login_does_not_return_a_token(self, client):
        r = client.post("/api/auth/login",
                        json={"email": "nobody@example.com", "password": "wrong"})
        assert "token" not in r.text.lower() or r.status_code >= 400

    def test_garbage_bearer_token_is_not_accepted(self, client):
        r = client.get("/api/auth/me", headers={"Authorization": "Bearer garbage"})
        assert r.status_code in (200, 401, 403)
        if r.status_code == 200:
            assert not (r.json() or {}).get("email")

    def test_signup_rejects_a_malformed_body(self, client):
        assert client.post("/api/auth/signup", json={}).status_code in (400, 422)


# ---------------------------------------------------------------------------
# Behaviour when a dependency is unavailable. Ollama being down, or a provider
# timing out, must not surface as a 500 with a traceback.
# ---------------------------------------------------------------------------

class TestDegradesWithoutDependencies:
    def test_screening_without_a_model_does_not_500(self, client, monkeypatch):
        monkeypatch.setattr(api.AIService, "get_model", lambda *a, **k: None)
        r = client.post("/api/screen/abstract", json={
            "paper": {"id": "1", "source": "PubMed", "title": "T",
                      "abstract": "A", "url": ""},
            "pico": {"population": "p", "concept": "c", "context": "x",
                     "framework": "pcc"},
            "model": "nonexistent-model",
        })
        assert r.status_code != 500

    def test_fulltext_fetch_for_an_unresolvable_record_reports_a_reason(self, client, monkeypatch):
        """A miss must always say why, so the reviewer knows where to go next."""
        monkeypatch.setattr(api, "_fetch_epmc_fulltext_text", lambda *a, **k: None)
        monkeypatch.setattr(api, "_lookup_pmc_metadata", lambda *a, **k: {})
        monkeypatch.setattr(api, "_fetch_oa_pdf", lambda *a, **k: (None, "", "closed"))
        r = client.post("/api/fulltext/fetch", json={
            "Title": "A paywalled paper", "URL": "", "Source": "PubMed", "paper_id": "999",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "missing"
        assert body["reason"], "a miss with no reason is what hid a dead tier"
        assert body["reason_code"] in ("no_doi", "paywalled", "oa_blocked", "unresolved")

    def test_dedupe_handles_an_empty_corpus(self, client):
        r = client.post("/api/papers/dedupe", json={"papers": []})
        assert r.status_code in (200, 400, 422)
        if r.status_code == 200:
            assert r.json() is not None

    def test_models_local_survives_ollama_being_down(self, client, monkeypatch):
        def boom(*a, **k):
            raise api.requests.exceptions.ConnectionError("ollama is not running")
        monkeypatch.setattr(api.requests, "get", boom)
        assert client.get("/api/models/local").status_code != 500


# ---------------------------------------------------------------------------
# Credentials must not leak back out through any response.
# ---------------------------------------------------------------------------

SECRET = "sk-super-secret-value-12345"


class TestCredentialsDoNotLeak:
    @pytest.mark.parametrize("header", [
        "x-llm-openai-key", "x-llm-anthropic-key", "x-db-core-key",
        "x-db-ncbi-key", "x-user-contact-email",
    ])
    def test_a_supplied_credential_is_never_echoed(self, client, header):
        for path in ("/api/health", "/api/keys/status", "/api/instruments"):
            r = client.get(path, headers={header: SECRET})
            assert SECRET not in r.text, f"{path} echoed the {header} value"

    def test_a_validation_error_does_not_echo_credentials(self, client):
        r = client.post("/api/screen/abstract", json={},
                        headers={"x-llm-openai-key": SECRET})
        assert SECRET not in r.text
