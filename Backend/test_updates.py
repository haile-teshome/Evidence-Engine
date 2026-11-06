"""Tests for the release check. No network: every fetch is stubbed.

Run: Backend/.venv/bin/python -m pytest Backend/test_updates.py

The cases worth having are the ones that are invisible in normal use. A check
that raises would stop a page load; a check that treats a non-version tag as
newer would nag every user forever; and a cached verdict that is not recomputed
would keep claiming an update after the user installed it.
"""
import json
import time

import pytest
import requests

import updates


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Redirect the cache and clear the environment this module reads."""
    monkeypatch.setenv("EE_UPDATE_CACHE", str(tmp_path / "update-check.json"))
    monkeypatch.delenv("EE_VERSION", raising=False)
    monkeypatch.delenv("EE_UPDATE_CHECK", raising=False)
    monkeypatch.delenv("EE_UPDATE_REPO", raising=False)


def _release(tag="v2.0.0", **over):
    out = {"tag": tag, "name": f"Evidence Engine {tag}", "url": "https://example.test/r",
           "notes": "notes", "published_at": "2026-11-05T00:00:00Z"}
    out.update(over)
    return out


def _stub_fetch(monkeypatch, value=None, exc=None):
    calls = {"n": 0}

    def fake():
        calls["n"] += 1
        if exc is not None:
            raise exc
        return value

    monkeypatch.setattr(updates, "fetch_latest", fake)
    return calls


# ---------------------------------------------------------------------------
# Version resolution
# ---------------------------------------------------------------------------

class TestCurrentVersion:
    def test_env_wins(self, monkeypatch):
        monkeypatch.setenv("EE_VERSION", "9.9.9")
        assert updates.current_version() == "9.9.9"

    def test_leading_v_is_stripped(self, monkeypatch):
        """A build stamped "v1.2.3" must compare equal to the tag "v1.2.3"."""
        monkeypatch.setenv("EE_VERSION", "v1.2.3")
        assert updates.current_version() == "1.2.3"

    def test_version_stamp_beats_package_json(self, monkeypatch, tmp_path):
        (tmp_path / "VERSION").write_text("3.1.4\n", encoding="utf-8")
        (tmp_path / "package.json").write_text(json.dumps({"version": "0.0.1"}), encoding="utf-8")
        monkeypatch.setattr(updates, "_ROOT", tmp_path)
        assert updates.current_version() == "3.1.4"

    def test_falls_back_to_package_json(self, monkeypatch, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({"version": "2.5.0"}), encoding="utf-8")
        monkeypatch.setattr(updates, "_ROOT", tmp_path)
        assert updates.current_version() == "2.5.0"

    def test_no_sources_is_zero_not_a_crash(self, monkeypatch, tmp_path):
        monkeypatch.setattr(updates, "_ROOT", tmp_path)
        assert updates.current_version() == "0.0.0"

    def test_unreadable_package_json_is_not_fatal(self, monkeypatch, tmp_path):
        (tmp_path / "package.json").write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(updates, "_ROOT", tmp_path)
        assert updates.current_version() == "0.0.0"

    def test_the_shipped_version_is_a_real_version(self):
        """Guards against an edit that leaves package.json unparseable."""
        assert updates.parse(updates.current_version()) is not None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class TestIsNewer:
    @pytest.mark.parametrize("latest,current,expected", [
        ("v1.1.0", "1.0.0", True),
        ("1.0.1", "1.0.0", True),
        ("v2.0.0", "1.9.9", True),
        ("v1.0.0", "1.0.0", False),      # same release, no banner
        ("v0.9.0", "1.0.0", False),      # a downgrade is not an update
        ("v1.0.0", "1.0.0-rc1", True),   # a final release beats its own rc
    ])
    def test_ordering(self, latest, current, expected):
        assert updates.is_newer(latest, current) is expected

    @pytest.mark.parametrize("tag", ["nightly", "", "latest", "release-candidate", None])
    def test_unparseable_tag_is_never_newer(self, tag):
        """Otherwise one oddly named release nags every user permanently."""
        assert updates.is_newer(tag, "1.0.0") is False

    def test_unparseable_current_is_never_newer(self):
        assert updates.is_newer("v2.0.0", "dev") is False


# ---------------------------------------------------------------------------
# check(): caching, recomputation, failure
# ---------------------------------------------------------------------------

class TestCheck:
    def test_newer_release_is_reported(self, monkeypatch):
        monkeypatch.setenv("EE_VERSION", "1.0.0")
        _stub_fetch(monkeypatch, _release("v2.0.0"))
        out = updates.check()
        assert out["newer"] is True
        assert out["latest"] == "v2.0.0"
        assert out["url"] == "https://example.test/r"
        assert out["error"] == ""

    def test_same_version_is_not_an_update(self, monkeypatch):
        monkeypatch.setenv("EE_VERSION", "2.0.0")
        _stub_fetch(monkeypatch, _release("v2.0.0"))
        assert updates.check()["newer"] is False

    def test_second_call_is_served_from_cache(self, monkeypatch):
        monkeypatch.setenv("EE_VERSION", "1.0.0")
        calls = _stub_fetch(monkeypatch, _release("v2.0.0"))
        updates.check()
        updates.check()
        assert calls["n"] == 1

    def test_force_skips_the_cache(self, monkeypatch):
        monkeypatch.setenv("EE_VERSION", "1.0.0")
        calls = _stub_fetch(monkeypatch, _release("v2.0.0"))
        updates.check()
        updates.check(force=True)
        assert calls["n"] == 2

    def test_stale_cache_is_refetched(self, monkeypatch):
        monkeypatch.setenv("EE_VERSION", "1.0.0")
        _stub_fetch(monkeypatch, _release("v2.0.0"))
        updates.check()
        cached = json.loads(updates.cache_path().read_text(encoding="utf-8"))
        cached["checked_at"] = time.time() - (updates.CACHE_TTL_SECONDS + 10)
        updates.cache_path().write_text(json.dumps(cached), encoding="utf-8")
        calls = _stub_fetch(monkeypatch, _release("v3.0.0"))
        assert updates.check()["latest"] == "v3.0.0"
        assert calls["n"] == 1

    def test_cache_hit_recomputes_against_the_current_version(self, monkeypatch):
        """Installing the update must clear the banner without waiting a day."""
        monkeypatch.setenv("EE_VERSION", "1.0.0")
        _stub_fetch(monkeypatch, _release("v2.0.0"))
        assert updates.check()["newer"] is True
        monkeypatch.setenv("EE_VERSION", "2.0.0")      # user installed it
        out = updates.check()                          # same warm cache
        assert out["newer"] is False

    def test_network_failure_reports_no_update_and_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("EE_VERSION", "1.0.0")
        _stub_fetch(monkeypatch, exc=requests.ConnectionError("offline"))
        out = updates.check()
        assert out["newer"] is False
        assert out["error"] == "ConnectionError"
        assert out["current"] == "1.0.0"

    def test_failure_falls_back_to_the_last_known_release(self, monkeypatch):
        monkeypatch.setenv("EE_VERSION", "1.0.0")
        _stub_fetch(monkeypatch, _release("v2.0.0"))
        updates.check()
        _stub_fetch(monkeypatch, exc=requests.Timeout("slow"))
        out = updates.check(force=True)
        assert out["latest"] == "v2.0.0"
        assert out["newer"] is True
        assert out["error"] == "Timeout"

    def test_corrupt_cache_is_ignored(self, monkeypatch):
        monkeypatch.setenv("EE_VERSION", "1.0.0")
        updates.cache_path().parent.mkdir(parents=True, exist_ok=True)
        updates.cache_path().write_text("{not json", encoding="utf-8")
        _stub_fetch(monkeypatch, _release("v2.0.0"))
        assert updates.check()["newer"] is True

    def test_disabled_makes_no_request(self, monkeypatch):
        monkeypatch.setenv("EE_UPDATE_CHECK", "0")
        calls = _stub_fetch(monkeypatch, _release("v2.0.0"))
        out = updates.check()
        assert out["enabled"] is False
        assert out["newer"] is False
        assert calls["n"] == 0

    def test_unwritable_cache_still_answers(self, monkeypatch):
        """A read-only install directory must not break the check."""
        monkeypatch.setenv("EE_VERSION", "1.0.0")
        monkeypatch.setenv("EE_UPDATE_CACHE", "/proc/nope/update.json")
        _stub_fetch(monkeypatch, _release("v2.0.0"))
        assert updates.check()["newer"] is True


# ---------------------------------------------------------------------------
# fetch_latest(): the one place that talks to GitHub
# ---------------------------------------------------------------------------

class TestFetchLatest:
    def test_sends_a_user_agent_and_maps_the_fields(self, monkeypatch):
        seen = {}

        class Res:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"tag_name": "v4.2.0", "name": "Four", "html_url": "https://x.test",
                        "body": "b" * (updates.NOTES_LIMIT + 500), "published_at": "2026-01-01"}

        def fake_get(url, headers=None, timeout=None):
            seen.update(url=url, headers=headers or {}, timeout=timeout)
            return Res()

        monkeypatch.setattr(requests, "get", fake_get)
        out = updates.fetch_latest()
        assert "haile-teshome/Evidence-Engine" in seen["url"]
        assert seen["headers"].get("User-Agent")      # GitHub rejects callers without one
        assert seen["timeout"] == updates._TIMEOUT
        assert out["tag"] == "v4.2.0"
        assert len(out["notes"]) == updates.NOTES_LIMIT

    def test_repo_is_overridable(self, monkeypatch):
        monkeypatch.setenv("EE_UPDATE_REPO", "someone/else")
        assert updates.repo() == "someone/else"

    def test_missing_fields_become_empty_strings(self, monkeypatch):
        class Res:
            def raise_for_status(self):
                pass

            def json(self):
                return {}

        monkeypatch.setattr(requests, "get", lambda *a, **k: Res())
        out = updates.fetch_latest()
        assert out == {"tag": "", "name": "", "url": "", "notes": "", "published_at": ""}
