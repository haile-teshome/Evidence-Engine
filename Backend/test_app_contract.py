"""Application-level guards. No network, no model.

Run: Backend/.venv/bin/python -m pytest Backend/test_app_contract.py

Two jobs:

1. Prove the app still assembles. During the Streamlit-shim removal a module was
   deleted while `api.py` still imported it. `py_compile` reported success
   because it only checks syntax, so the break would have surfaced at runtime for
   a user. A one-second import test catches that class of mistake outright.

2. Pin the safety invariants at the LLM boundary. The model is stubbed: what is
   under test is how the code behaves when a model misbehaves, which is the case
   that actually loses papers.
"""
import pytest

import api


# ---------------------------------------------------------------------------
# 1. The app assembles
# ---------------------------------------------------------------------------

class TestAppAssembles:
    def test_every_module_imports(self):
        import importlib
        for mod in ("config", "models", "frameworks", "instruments", "store",
                    "request_creds", "leads_screening", "data_services", "utils", "api"):
            importlib.import_module(mod)

    def test_routes_are_registered(self):
        assert len(api.app.routes) > 50

    def test_core_endpoints_exist(self):
        paths = {getattr(r, "path", "") for r in api.app.routes}
        for p in ("/api/screen/abstract", "/api/fulltext/fetch", "/api/models/local"):
            assert p in paths, f"missing route {p}"

    def test_streamlit_is_not_imported_anywhere(self):
        """The shim is gone. Nothing may reintroduce a Streamlit dependency into
        what is now a plain FastAPI service."""
        import sys
        assert "streamlit" not in sys.modules

    def test_no_contact_email_is_baked_into_the_source(self):
        """Addresses come from the user's profile per request. A hardcoded one
        gets rejected by the very APIs it is meant to unlock."""
        import pathlib
        import re
        for name in ("api.py", "data_services.py", "utils.py"):
            src = (pathlib.Path(__file__).parent / name).read_text()
            for line in src.splitlines():
                if line.strip().startswith("#"):
                    continue
                assert not re.search(r'"[\w.+-]+@[\w-]+\.\w{2,}"', line), \
                    f"hardcoded email literal in {name}: {line.strip()[:80]}"


# ---------------------------------------------------------------------------
# 2. LLM-boundary invariants
# ---------------------------------------------------------------------------

class StubModel:
    """Stands in for a chat model. `reply` may be a string or an exception."""
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return type("R", (), {"content": self.reply})()


PART = {"part": "electronic health records", "appears_as": ["EHR"],
        "not_satisfied_by": ["paper charts"]}


class TestPartPresentNeverExcludesOnFailure:
    """The single most important invariant in screening: a broken call must not
    be able to drop a paper. _part_present returns "unclear" on any failure, and
    only "no" produces a FAIL, so a malfunctioning model loses precision but
    never recall."""

    @pytest.mark.parametrize("reply", [
        RuntimeError("model exploded"),
        "",
        "I am not going to answer that",
        "{}",
        "maybe?",
    ])
    def test_unusable_replies_yield_unclear(self, monkeypatch, reply):
        monkeypatch.setattr(api.AIService, "get_model", lambda *_: StubModel(reply))
        assert api._part_present(PART, "T", "A", "m") == "unclear"

    def test_missing_model_yields_unclear(self, monkeypatch):
        monkeypatch.setattr(api.AIService, "get_model", lambda *_: None)
        assert api._part_present(PART, "T", "A", "m") == "unclear"

    @pytest.mark.parametrize("reply,expected", [
        ("yes", "yes"), ("no", "no"), ("unclear", "unclear"),
        ("YES", "yes"), ("  no  ", "no"), ("Answer: yes", "yes"),
    ])
    def test_parses_the_three_valid_answers(self, monkeypatch, reply, expected):
        monkeypatch.setattr(api.AIService, "get_model", lambda *_: StubModel(reply))
        assert api._part_present(PART, "T", "A", "m") == expected

    def test_prompt_carries_the_reviews_own_vocabulary(self, monkeypatch):
        """Synonyms are generated per review, never hardcoded, so a page about
        one subject cannot leak into a review about another."""
        captured = {}

        class Capturing(StubModel):
            def invoke(self, messages):
                captured["prompt"] = messages[0].content
                return super().invoke(messages)

        monkeypatch.setattr(api.AIService, "get_model", lambda *_: Capturing("yes"))
        api._part_present(PART, "Title", "Abstract", "m")
        assert "EHR" in captured["prompt"]
        assert "paper charts" in captured["prompt"]
        for leaked in ("periodontal", "caries", "dental data"):
            assert leaked not in captured["prompt"].lower()


class TestDecomposeElement:
    def teardown_method(self):
        api._ELEMENT_PARTS_CACHE.clear()

    def test_result_is_cached_per_element_and_model(self, monkeypatch):
        """Decomposition depends only on the element text, so it must cost one
        call per review rather than one per paper."""
        stub = StubModel('[{"part": "electronic health records", '
                         '"appears_as": ["EHR"], "not_satisfied_by": ["paper"]}]')
        monkeypatch.setattr(api.AIService, "get_model", lambda *_: stub)
        api._decompose_element("some element", "m")
        api._decompose_element("some element", "m")
        assert stub.calls == 1

    def test_malformed_json_falls_back_to_the_whole_element(self, monkeypatch):
        """A failed decomposition must disable per-part checking, not invent
        parts that would exclude papers."""
        monkeypatch.setattr(api.AIService, "get_model", lambda *_: StubModel("not json"))
        parts = api._decompose_element("the whole element text", "m")
        assert len(parts) == 1
        assert parts[0]["part"] == "the whole element text"

    def test_every_part_has_the_full_shape(self, monkeypatch):
        monkeypatch.setattr(api.AIService, "get_model", lambda *_: StubModel("garbage"))
        for p in api._decompose_element("x", "m"):
            assert set(p) >= {"part", "appears_as", "not_satisfied_by"}

    def test_bare_string_array_is_tolerated(self, monkeypatch):
        """Small models sometimes ignore the object shape and return strings."""
        monkeypatch.setattr(api.AIService, "get_model",
                            lambda *_: StubModel('["electronic health records", "AI models"]'))
        parts = api._decompose_element("a and b", "m")
        assert [p["part"] for p in parts] == ["electronic health records", "AI models"]

    def test_or_near_misses_are_stripped(self, monkeypatch):
        """For an "A or B" part, "A without B" is a match, not a miss."""
        monkeypatch.setattr(api.AIService, "get_model", lambda *_: StubModel(
            '[{"part": "satellite or aerial imagery", "appears_as": [],'
            ' "not_satisfied_by": ["satellite without aerial", "ground photos only"]}]'))
        parts = api._decompose_element("satellite or aerial imagery", "m")
        assert parts[0]["not_satisfied_by"] == ["ground photos only"]
