"""Serving the built frontend from the API process.

The point of this mount is that one process answers everything, so the tests cover the
three ways that can go wrong: an API route shadowed by the catch-all, a deep link that
does not reach the single-page app, and a path that escapes the build directory.
"""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import webapp


@pytest.fixture
def built(tmp_path):
    d = tmp_path / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text("<!doctype html><title>Evidence Engine</title><div id=root></div>")
    (d / "assets" / "app-abc123.js").write_text("console.log('app')")
    (d / "favicon.svg").write_text("<svg/>")
    (tmp_path / "secret.txt").write_text("do not serve me")
    return d


@pytest.fixture
def client(built):
    app = FastAPI()

    @app.get("/api/ping")
    def ping():
        return {"ok": True}

    assert webapp.mount_frontend(app, built) is True
    return TestClient(app)


def test_api_routes_still_win(client):
    r = client.get("/api/ping")
    assert r.status_code == 200 and r.json() == {"ok": True}


def test_missing_api_route_is_a_404_not_the_app(client):
    """A 200 page in place of a 404 turns a typo into a silent bug."""
    r = client.get("/api/not-a-route")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")


def test_root_serves_the_app(client):
    r = client.get("/")
    assert r.status_code == 200 and "Evidence Engine" in r.text
    assert r.headers["cache-control"] == "no-store"      # it names hashed assets


def test_hashed_assets_are_cached_hard(client):
    r = client.get("/assets/app-abc123.js")
    assert r.status_code == 200 and "console.log" in r.text
    assert "immutable" in r.headers.get("cache-control", "")


def test_real_files_at_the_root_are_served(client):
    assert client.get("/favicon.svg").status_code == 200


def test_deep_links_reach_the_client_router(client):
    """An invitation link is /?invite=TOKEN, and /projects is a client route."""
    for path in ("/projects", "/quality/rob", "/?invite=TOKEN"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "<div id=root>" in r.text, path


def test_paths_cannot_escape_the_build_directory(client):
    for path in ("/../secret.txt", "/assets/../../secret.txt", "/%2e%2e%2fsecret.txt"):
        r = client.get(path)
        assert "do not serve me" not in r.text, path


def test_absent_build_is_not_fatal(tmp_path):
    """Backend-only development and the test suite must keep working with no dist/."""
    app = FastAPI()
    assert webapp.mount_frontend(app, tmp_path / "nothing") is False
    assert TestClient(app).get("/").status_code == 404


def test_status_reports_whether_it_is_serving(built, monkeypatch):
    monkeypatch.setenv("EE_DIST_DIR", str(built))
    st = webapp.frontend_status()
    assert st["serving_frontend"] is True and st["hint"] is None
    monkeypatch.setenv("EE_DIST_DIR", str(built.parent / "absent"))
    st = webapp.frontend_status()
    assert st["serving_frontend"] is False and "pnpm build" in st["hint"]


def test_unknown_api_route_is_404_for_any_method(client):
    """A POST to a route that does not exist must say so, not blame the method.

    The catch-all was registered for GET only, so Starlette matched the path,
    found no POST, and answered 405 "Method Not Allowed" before this handler
    ran. That reads as a method mismatch and sends you hunting for the wrong
    bug; the usual real cause is a backend running code from before the route
    existed.
    """
    for method in ("post", "put", "patch", "delete"):
        r = getattr(client, method)("/api/not/a/real/route")
        assert r.status_code == 404, f"{method} gave {r.status_code}"
        assert "No such endpoint" in r.json()["detail"]


def test_posting_to_a_page_path_is_still_405(client):
    """A static page genuinely cannot be posted to, so 405 is right there."""
    r = client.post("/some/client/route")
    assert r.status_code == 405


def test_deep_link_still_serves_the_app_on_get(client):
    assert client.get("/some/client/route").status_code == 200
