"""Tests for structured-form extraction, LEADS screening, and the OS keychain.

Run: Backend/.venv/bin/python -m pytest Backend/test_extraction_forms.py

Extraction is where a hallucinated number becomes a data point in a
meta-analysis. The guard is provenance: a numeric value only earns "high"
confidence if it can actually be found in the source text. These tests pin that
guard and the behaviour when the model returns nothing usable.
"""
import pytest

import extraction_forms as EF
import keychain
import leads_screening as L


class StubModel:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        if isinstance(self.reply, Exception):
            raise self.reply
        return type("R", (), {"content": self.reply})()


class StubAI:
    """Stands in for AIService, which is injected to avoid a circular import."""
    def __init__(self, model):
        self._model = model

    def get_model(self, _name):
        return self._model

    @staticmethod
    def _extract_json(text):
        import json
        try:
            return json.loads(text)
        except Exception:
            return None


TEXT = ("We enrolled 128 patients between 2019 and 2023. "
        "The mean age was 42.5 years. The trial was randomised and double blind.")

FIELDS = [
    {"name": "sample_size", "type": "number"},
    {"name": "mean_age", "type": "number"},
    {"name": "design", "type": "text"},
]


def _extract(reply, fields=FIELDS, text=TEXT, tables=""):
    return EF.extract_form(text, tables, fields, "m", StubAI(StubModel(reply)))


class TestBuildPrompt:
    def test_includes_every_requested_field(self):
        p = EF.build_prompt(TEXT, "", FIELDS)
        for f in FIELDS:
            assert f["name"] in p

    def test_includes_the_source_text(self):
        assert "128 patients" in EF.build_prompt(TEXT, "", FIELDS)

    def test_includes_tables_when_supplied(self):
        assert "TABLE-MARKER" in EF.build_prompt(TEXT, "TABLE-MARKER", FIELDS)

    def test_truncates_a_very_long_text(self):
        """Prompts have to fit in the context window; an over-long body would be
        silently cut by the model instead, losing the end of the paper."""
        p = EF.build_prompt("x" * 100_000, "", FIELDS, max_text=500)
        assert len(p) < 20_000

    def test_truncates_very_long_tables(self):
        p = EF.build_prompt(TEXT, "y" * 100_000, FIELDS, max_tables=500)
        assert len(p) < 20_000

    def test_empty_field_list_still_produces_a_prompt(self):
        assert isinstance(EF.build_prompt(TEXT, "", []), str)


class TestExtractForm:
    def test_returns_one_entry_per_requested_field(self):
        out = _extract('{"extractions":[{"name":"sample_size","value":128,"source_quote":"128 patients"}]}')
        assert len(out["fields"]) == len(FIELDS)
        assert {f["name"] for f in out["fields"]} == {f["name"] for f in FIELDS}

    def test_a_grounded_number_earns_high_confidence(self):
        out = _extract('{"extractions":[{"name":"sample_size","value":128,'
                       '"source_quote":"We enrolled 128 patients"}]}')
        f = next(f for f in out["fields"] if f["name"] == "sample_size")
        assert f["confidence"] == "high"

    def test_every_numeric_field_needs_review_however_confident(self):
        """Deliberate policy, not a bug: extracted numbers feed a meta-analysis,
        so a human verifies each one even when it is grounded in the source."""
        out = _extract('{"extractions":[{"name":"sample_size","value":128,'
                       '"source_quote":"We enrolled 128 patients"}]}')
        f = next(f for f in out["fields"] if f["name"] == "sample_size")
        assert f["confidence"] == "high"
        assert f["needs_review"] is True

    def test_a_confident_text_field_does_not_need_review(self):
        out = _extract('{"extractions":[{"name":"design","value":"randomised",'
                       '"source_quote":"The trial was randomised"}]}')
        f = next(f for f in out["fields"] if f["name"] == "design")
        assert f["needs_review"] is False

    def test_a_number_absent_from_the_source_does_not_earn_high_confidence(self):
        """The fabrication guard: 999 appears nowhere in the text."""
        out = _extract('{"extractions":[{"name":"sample_size","value":999,'
                       '"source_quote":"We enrolled 128 patients"}]}')
        f = next(f for f in out["fields"] if f["name"] == "sample_size")
        assert f["confidence"] != "high"
        assert f["needs_review"] is True

    def test_a_field_the_model_omitted_is_flagged_for_review(self):
        out = _extract('{"extractions":[]}')
        assert all(f["needs_review"] for f in out["fields"])

    def test_no_model_flags_every_field_rather_than_failing(self):
        out = EF.extract_form(TEXT, "", FIELDS, "m", StubAI(None))
        assert len(out["fields"]) == len(FIELDS)
        assert all(f["needs_review"] for f in out["fields"])
        assert all(f.get("error") for f in out["fields"])

    @pytest.mark.parametrize("reply", ["", "not json", "{}", '{"extractions": null}'])
    def test_unusable_model_output_still_returns_every_field(self, reply):
        out = _extract(reply)
        assert len(out["fields"]) == len(FIELDS)

    def test_a_model_exception_does_not_propagate(self):
        out = _extract(RuntimeError("model down"))
        assert len(out["fields"]) == len(FIELDS)

    def test_the_model_name_is_reported_back(self):
        assert _extract('{"extractions":[]}')["model"] == "m"

    def test_every_field_carries_the_full_shape(self):
        out = _extract('{"extractions":[{"name":"design","value":"randomised",'
                       '"source_quote":"the trial was randomised"}]}')
        for f in out["fields"]:
            assert {"name", "type", "value", "source_quote",
                    "confidence", "needs_review"} <= set(f)

    def test_a_quoted_text_field_earns_high_confidence(self):
        out = _extract('{"extractions":[{"name":"design","value":"randomised",'
                       '"source_quote":"The trial was randomised"}]}')
        f = next(f for f in out["fields"] if f["name"] == "design")
        assert f["confidence"] == "high"

    def test_an_unquoted_text_field_is_flagged(self):
        out = _extract('{"extractions":[{"name":"design","value":"randomised","source_quote":""}]}')
        f = next(f for f in out["fields"] if f["name"] == "design")
        assert f["confidence"] == "low"

    def test_a_number_found_only_in_the_tables_is_still_grounded(self):
        out = EF.extract_form("No numbers in prose.", "Total N = 250", FIELDS, "m",
                              StubAI(StubModel('{"extractions":[{"name":"sample_size",'
                                               '"value":250,"source_quote":"Total N = 250"}]}')))
        f = next(f for f in out["fields"] if f["name"] == "sample_size")
        assert f["confidence"] == "high"

    def test_an_extraction_for_an_unrequested_field_is_ignored(self):
        out = _extract('{"extractions":[{"name":"not_requested","value":1,"source_quote":"x"}]}')
        assert {f["name"] for f in out["fields"]} == {f["name"] for f in FIELDS}


# ---------------------------------------------------------------------------
# LEADS native screening
# ---------------------------------------------------------------------------

class FakePico:
    population = "adults"
    intervention = "metformin"
    comparator = "placebo"
    outcome = "HbA1c"


class FakePaper:
    title = "A randomised trial of metformin"
    abstract = "We randomised 128 adults to metformin or placebo and measured HbA1c."


class TestScreenPaperLeads:
    @pytest.fixture
    def stub_ollama(self, monkeypatch):
        def _install(reply):
            class FakeChat:
                def __init__(self, **kw):
                    pass

                def invoke(self, _msgs):
                    if isinstance(reply, Exception):
                        raise reply
                    return type("R", (), {"content": reply})()

            import langchain_ollama
            monkeypatch.setattr(langchain_ollama, "ChatOllama", FakeChat)
        return _install

    ALL_YES = ('{"evaluations":[{"eligibility":"YES","rationale":"a"},'
               '{"eligibility":"YES","rationale":"b"},'
               '{"eligibility":"YES","rationale":"c"},'
               '{"eligibility":"YES","rationale":"d"}]}')
    ALL_NO = ('{"evaluations":[{"eligibility":"NO","rationale":"a"},'
              '{"eligibility":"NO","rationale":"b"},'
              '{"eligibility":"NO","rationale":"c"},'
              '{"eligibility":"NO","rationale":"d"}]}')

    def test_all_criteria_met_includes(self, stub_ollama):
        stub_ollama(self.ALL_YES)
        out = L.screen_paper_leads(FakePaper(), FakePico())
        assert out["decision"] == "Include"

    def test_all_criteria_failed_excludes(self, stub_ollama):
        stub_ollama(self.ALL_NO)
        out = L.screen_paper_leads(FakePaper(), FakePico())
        assert out["decision"] == "Exclude"

    def test_result_carries_the_score_and_threshold(self, stub_ollama):
        stub_ollama(self.ALL_YES)
        out = L.screen_paper_leads(FakePaper(), FakePico())
        assert "_leads_score" in out and "_leads_threshold" in out
        assert isinstance(out["_leads_score"], float)

    def test_result_carries_a_reason_and_bucket(self, stub_ollama):
        stub_ollama(self.ALL_NO)
        out = L.screen_paper_leads(FakePaper(), FakePico())
        assert out.get("reason")
        assert out.get("bucket")

    def test_a_model_failure_does_not_raise(self, stub_ollama):
        stub_ollama(RuntimeError("ollama down"))
        out = L.screen_paper_leads(FakePaper(), FakePico())
        assert out["decision"] in ("Include", "Exclude")

    def test_unparseable_output_does_not_raise(self, stub_ollama):
        stub_ollama("not json at all")
        assert isinstance(L.screen_paper_leads(FakePaper(), FakePico()), dict)

    def test_an_empty_pico_does_not_raise(self, stub_ollama):
        stub_ollama(self.ALL_YES)

        class Empty:
            population = intervention = comparator = outcome = ""

        assert isinstance(L.screen_paper_leads(FakePaper(), Empty()), dict)


class TestResolveForThinking:
    def test_returns_a_string_or_none(self):
        out = L.resolve_for_thinking("qwen2.5:7b")
        assert out is None or isinstance(out, str)


# ---------------------------------------------------------------------------
# OS keychain. Never raise on read, and never leave a stale cached value.
# ---------------------------------------------------------------------------

class TestKeychain:
    def teardown_method(self):
        keychain._cache.clear()

    def test_available_reports_a_boolean(self):
        assert isinstance(keychain.available(), bool)

    def test_get_key_returns_empty_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(keychain, "_AVAILABLE", False)
        assert keychain.get_key("openai") == ""

    def test_set_key_raises_a_clear_error_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(keychain, "_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="not available"):
            keychain.set_key("openai", "sk-x")

    def test_delete_key_is_a_no_op_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(keychain, "_AVAILABLE", False)
        keychain.delete_key("openai")

    def test_get_key_returns_empty_when_the_backend_raises(self, monkeypatch):
        """A locked or broken keychain must not take the app down."""
        monkeypatch.setattr(keychain, "_AVAILABLE", True)

        class Boom:
            @staticmethod
            def get_password(*a):
                raise RuntimeError("keychain locked")

        monkeypatch.setattr(keychain, "keyring", Boom)
        assert keychain.get_key("openai") == ""

    def test_set_then_get_round_trips_through_the_cache(self, monkeypatch):
        monkeypatch.setattr(keychain, "_AVAILABLE", True)
        store = {}

        class Fake:
            @staticmethod
            def set_password(s, p, k):
                store[p] = k

            @staticmethod
            def get_password(s, p):
                return store.get(p)

            @staticmethod
            def delete_password(s, p):
                store.pop(p, None)

        monkeypatch.setattr(keychain, "keyring", Fake)
        keychain.set_key("openai", "sk-secret")
        assert keychain.get_key("openai") == "sk-secret"

    def test_delete_clears_the_cache_so_a_deleted_key_is_not_still_served(self, monkeypatch):
        """A stale cache entry would keep sending a revoked key."""
        monkeypatch.setattr(keychain, "_AVAILABLE", True)
        store = {"openai": "sk-secret"}

        class Fake:
            @staticmethod
            def get_password(s, p):
                return store.get(p)

            @staticmethod
            def delete_password(s, p):
                store.pop(p, None)

        monkeypatch.setattr(keychain, "keyring", Fake)
        assert keychain.get_key("openai") == "sk-secret"
        keychain.delete_key("openai")
        assert keychain.get_key("openai") == ""

    def test_delete_survives_a_backend_error(self, monkeypatch):
        monkeypatch.setattr(keychain, "_AVAILABLE", True)

        class Boom:
            @staticmethod
            def delete_password(*a):
                raise RuntimeError("no such item")

        monkeypatch.setattr(keychain, "keyring", Boom)
        keychain.delete_key("openai")
