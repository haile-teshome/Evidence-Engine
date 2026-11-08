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

    def get_model(self, _name, num_ctx=None):
        self.num_ctx = num_ctx
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


class TestTextBudget:
    """How much of the article gets sent.

    A fixed 30,000-character prefix was the reason funding, conflicts of
    interest, grant numbers and data availability came back blank: in two real
    papers those statements begin at 90 % and 80 % of the way through, so they
    were never in the prompt. The budget comes from the context now.
    """

    def test_it_scales_with_the_context(self):
        assert EF.text_budget(32768) > EF.text_budget(8192) > EF.text_budget(4096)

    def test_the_app_s_context_holds_a_whole_journal_article(self):
        # The two papers this was measured on are 41.6k and 45.1k characters.
        assert EF.text_budget(32768) >= 60_000

    def test_the_tables_come_out_of_the_same_budget(self):
        assert EF.text_budget(32768, tables_chars=8000) == EF.text_budget(32768) - 8000

    def test_it_never_goes_below_a_usable_floor(self):
        # A context too small for a paper should still get a prompt that asks a
        # real question rather than one holding two sentences.
        assert EF.text_budget(1) == EF.MIN_TEXT_CHARS
        assert EF.text_budget(32768, tables_chars=10**9) == EF.MIN_TEXT_CHARS

    def test_a_caller_that_names_no_context_gets_a_cautious_budget(self):
        """Ollama's default window is 2048 or 4096 tokens when nobody asks.

        Sending more than the window holds is cut by the server instead, which
        is the same loss as truncating here and harder to notice.
        """
        assert EF.text_budget(None) == EF.text_budget(EF.ASSUMED_NUM_CTX)
        assert EF.text_budget(None) < EF.text_budget(32768)


class TestWindow:
    """Which part of the article survives when all of it will not fit."""

    BACK_MATTER = "Funding: this work was supported by grant ABC123."

    def _paper(self, size=200_000):
        return ("INTRO " + "a" * (size // 2) + " RESULTS " + "b" * (size // 2)
                + " " + self.BACK_MATTER)

    def test_a_paper_that_fits_is_sent_whole(self):
        assert EF.window(TEXT, 100_000) == TEXT

    def test_it_keeps_the_front(self):
        assert EF.window(self._paper(), 40_000).startswith("INTRO")

    def test_it_keeps_the_back_matter(self):
        """The failure this exists for: funding and conflicts of interest sit in
        the last tenth of a paper, and a prefix drops that whole class of
        field."""
        assert self.BACK_MATTER in EF.window(self._paper(), 40_000)

    def test_it_stays_within_the_budget(self):
        assert len(EF.window(self._paper(), 40_000)) <= 40_000

    def test_it_says_where_the_gap_is(self):
        # A reviewer reading a quote needs to know the middle is not there.
        assert EF.OMITTED_MARKER in EF.window(self._paper(), 40_000)

    def test_a_budget_too_small_to_split_still_returns_text(self):
        for budget in (0, -5, 10, 300, EF.MIN_SEGMENT_CHARS):
            got = EF.window(self._paper(), budget)
            assert isinstance(got, str)
            if budget > 0:
                assert len(got) <= max(budget, len(self._paper()))

    def test_the_old_prefix_would_have_lost_it(self):
        # Pins the regression rather than just the fix.
        paper = self._paper()
        assert self.BACK_MATTER not in paper[:30_000]


class TestFieldStatus:
    """Why a cell is empty.

    An empty value meant two different things: the paper does not report this,
    and nothing ever read it. A review writes the first one down as a finding
    and has to go back and do the second.
    """

    def _status(self, out, name):
        return next(f["status"] for f in out["fields"] if f["name"] == name)

    def test_a_value_is_reported(self):
        out = _extract('{"extractions": [{"name": "sample_size", "value": 128, '
                       '"source_quote": "We enrolled 128 patients"}]}')
        assert self._status(out, "sample_size") == "reported"

    def test_a_null_the_model_returned_is_not_reported(self):
        # The model read the article and says the value is not in it.
        out = _extract('{"extractions": [{"name": "sample_size", "value": null, '
                       '"source_quote": ""}]}')
        assert self._status(out, "sample_size") == "not_reported"

    def test_a_field_the_model_never_mentioned_is_unread(self):
        out = _extract('{"extractions": [{"name": "sample_size", "value": 128, '
                       '"source_quote": "We enrolled 128 patients"}]}')
        assert self._status(out, "mean_age") == "unread"
        assert self._status(out, "design") == "unread"

    def test_a_failed_group_is_unread_and_not_mistaken_for_an_answer(self):
        """The important one. A reply cut off mid-JSON returns every field
        empty, and reading that as "the paper reports none of this" would write
        a whole column of findings that nobody established."""
        out = _extract("not json at all")
        assert out["error"]
        for name in ("sample_size", "mean_age", "design"):
            assert self._status(out, name) == "unread"

    def test_no_model_is_unread(self):
        out = EF.extract_form(TEXT, "", FIELDS, "m", StubAI(None))
        for f in out["fields"]:
            assert f["status"] == "unread"

    def test_an_empty_string_counts_as_not_reported(self):
        out = _extract('{"extractions": [{"name": "design", "value": "", '
                       '"source_quote": ""}]}')
        assert self._status(out, "design") == "not_reported"


class TestTheBackMatterReachesTheModel:
    """End to end over extract_form: the article's tail is in the prompt."""

    def test_a_field_stated_only_at_the_end_is_sent(self):
        sentinel = "Funding: supported by grant R01CA200234."
        paper = "Methods. " + ("filler sentence. " * 2500) + sentinel
        assert len(paper) > 40_000
        assert sentinel not in paper[:30_000]      # the old behaviour lost it

        model = StubModel('{"extractions": []}')
        EF.extract_form(paper, "", FIELDS, "m", StubAI(model), num_ctx=32768)

        assert sentinel in model.prompts[0]

    def test_the_context_it_was_told_about_is_the_one_it_opens(self):
        ai = StubAI(StubModel('{"extractions": []}'))
        EF.extract_form(TEXT, "", FIELDS, "m", ai, num_ctx=32768)
        assert ai.num_ctx == 32768


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


class TestFieldGroups:
    """Asking for a whole imported workbook in one request.

    Measured on a real paper with the local 7B: 40 fields filled 31 of them;
    183, which is what a Covidence workbook looks like, returned no usable JSON
    and filled none. The failure is in the answer, not the question: every field
    costs a value, a source quote and a confidence in the reply, and a reply cut
    off part-way parses to nothing at all. So the endpoint asks in groups.
    """

    def _fields(self, n):
        return [{"id": f"f{i}", "label": f"Field {i}", "type": "text"} for i in range(n)]

    def _client(self):
        from fastapi.testclient import TestClient
        import api
        return TestClient(api.app), api

    def test_a_large_form_is_split_into_several_requests(self, monkeypatch):
        c, api = self._client()
        seen = []

        def fake(*, text, tables, fields, model_name, ai_service, num_ctx=None):
            seen.append(len(fields))
            return {"fields": [{"name": f["name"], "value": "v", "type": "text",
                                "source_quote": "", "confidence": "low",
                                "needs_review": True} for f in fields]}

        import extraction_forms
        monkeypatch.setattr(extraction_forms, "extract_form", fake)
        body = c.post("/api/extract/fields",
                      json={"text": "Methods. " * 100, "fields": self._fields(60)}).json()

        assert len(seen) > 1, "60 fields went out as one request"
        assert max(seen) <= api.Config.EXTRACTION_FIELDS_PER_CALL
        assert len(body["values"]) == 60, "every field has to come back"

    def test_one_failed_group_does_not_empty_the_row(self, monkeypatch):
        """A partial answer is still an answer: the fields that were read are
        worth more to the reviewer than a blank row."""
        c, api = self._client()
        calls = {"n": 0}

        def fake(*, text, tables, fields, model_name, ai_service, num_ctx=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"fields": [], "error": "the model did not return usable JSON"}
            return {"fields": [{"name": f["name"], "value": "v", "type": "text",
                                "source_quote": "", "confidence": "low",
                                "needs_review": True} for f in fields]}

        import extraction_forms
        monkeypatch.setattr(extraction_forms, "extract_form", fake)
        body = c.post("/api/extract/fields",
                      json={"text": "Methods. " * 100, "fields": self._fields(60)}).json()

        filled = [v for v in body["values"].values() if str(v).strip()]
        assert filled, "the groups that worked were thrown away"
        assert "of" in body["error"] and "groups failed" in body["error"]

    def test_every_group_failing_is_reported_as_the_failure_it_is(self, monkeypatch):
        c, api = self._client()

        def fake(*, text, tables, fields, model_name, ai_service, num_ctx=None):
            return {"fields": [], "error": "the model did not return usable JSON"}

        import extraction_forms
        monkeypatch.setattr(extraction_forms, "extract_form", fake)
        body = c.post("/api/extract/fields",
                      json={"text": "Methods. " * 100, "fields": self._fields(60)}).json()

        assert body["error"] == "the model did not return usable JSON"
        assert not [v for v in body["values"].values() if str(v).strip()]

    def test_a_small_form_is_still_one_request(self, monkeypatch):
        c, api = self._client()
        seen = []

        def fake(*, text, tables, fields, model_name, ai_service, num_ctx=None):
            seen.append(len(fields))
            return {"fields": []}

        import extraction_forms
        monkeypatch.setattr(extraction_forms, "extract_form", fake)
        c.post("/api/extract/fields", json={"text": "Methods. " * 100, "fields": self._fields(6)})
        assert seen == [6]


class TestSupportedByQuote:
    """Does the cited sentence actually say the value?

    The contract of locate-then-extract is that the quote is the sentence the
    value was read from, and that was never checked. The quote was verified to
    exist somewhere in the paper, which it always does, and a number was
    checked against the whole paper rather than against its own quote. So a
    value lifted from a different sentence arrived marked high confidence and
    not flagged for review.
    """

    def test_a_value_its_sentence_does_not_mention_is_unsupported(self):
        assert not EF.supported_by_quote(
            "300 and 800 \u03bcM",
            "Interestingly, better results were obtained with drug concentrations "
            "exceeding the doses typically administered for seizure control.")

    def test_a_paraphrase_of_its_sentence_is_supported(self):
        # An extraction is normally a paraphrase, and must not be punished.
        assert EF.supported_by_quote(
            "retrospective cohort",
            "An observational, retrospective single-center study was conducted")

    def test_a_verbatim_value_is_supported(self):
        assert EF.supported_by_quote(
            "15.64 months", "The median OS of our cohort (n = 272) was 15.64 months.")

    def test_a_one_word_value_in_its_sentence_is_supported(self):
        assert EF.supported_by_quote(
            "Italy", "Department of Neuroscience, University of Turin, Turin, Italy")

    def test_a_long_value_from_another_sentence_is_unsupported(self):
        assert not EF.supported_by_quote(
            "TFC is cofounder, major stock holder and board member of Katmai",
            "Notes: Role of the funder: The National Cancer Institute played no role "
            "in the design and conduct of the study.")

    def test_nothing_to_check_is_not_a_failure(self):
        assert EF.supported_by_quote("", "any sentence")
        assert EF.supported_by_quote(None, "any sentence")

    def test_a_value_with_no_sentence_is_unsupported(self):
        assert not EF.supported_by_quote("15.64 months", "")


class TestAttribution:
    """Whose finding is this?

    A clinical study of two anti-seizure drugs returned "300 and 800 uM" as its
    in vitro concentration. The number really is in the sentence it cites, so
    the support check passes; the sentence is somebody else's cell-culture work,
    discussed, and the study did no in vitro work at all. A review that pooled
    it would have pooled a number from a paper it never screened.
    """

    def test_a_cited_finding_is_attributed_elsewhere(self):
        assert EF.attributed_elsewhere(
            "The inhibitory effect was observed at concentrations between 300 and "
            "800 \u03bcM (Rizzo et al., 2017).")

    def test_a_numbered_citation_counts(self):
        assert EF.attributed_elsewhere("Previous studies reported a benefit [26].")

    def test_a_methods_sentence_citing_its_own_protocol_is_not(self):
        """The false positive that would matter: this is the study's own
        treatment, and flagging it would teach the reviewer to ignore flags."""
        assert not EF.attributed_elsewhere(
            "Patients received temozolomide according to the Stupp protocol "
            "(Stupp et al., 2005).")

    def test_the_study_s_own_result_is_not(self):
        assert not EF.attributed_elsewhere(
            "The median OS of our cohort (n = 272) was 15.64 months.")

    def test_a_citation_without_a_finding_is_not(self):
        assert not EF.attributed_elsewhere("Glioblastoma is the commonest glioma (3).")


class TestConfidenceReadsTheQuote:
    def _field(self, out, name="sample_size"):
        return next(f for f in out["fields"] if f["name"] == name)

    def test_a_number_elsewhere_in_the_paper_is_not_grounded_in_this_quote(self):
        """272 is in the text, but not in the sentence offered as its source."""
        out = EF.extract_form(
            "We enrolled 272 patients. The mean age was 42.5 years.", "", FIELDS, "m",
            StubAI(StubModel('{"extractions":[{"name":"sample_size","value":272,'
                             '"source_quote":"The mean age was 42.5 years."}]}')))
        f = self._field(out)
        assert f["confidence"] == "low"
        assert f["needs_review"] is True
        assert "does not appear in the sentence" in f["review_reason"]

    def test_a_text_value_its_quote_does_not_mention_is_flagged(self):
        out = EF.extract_form(
            "The trial was randomised. Patients were followed for a year.", "",
            [{"name": "design", "type": "text"}], "m",
            StubAI(StubModel('{"extractions":[{"name":"design","value":"double blind crossover",'
                             '"source_quote":"Patients were followed for a year."}]}')))
        f = self._field(out, "design")
        assert f["confidence"] == "low"
        assert "does not mention the value" in f["review_reason"]

    def test_a_value_credited_to_another_paper_is_flagged(self):
        text = ("The inhibitory effect of LAC on cell cycle progression was observed "
                "at concentrations between 300 and 800 uM (Rizzo et al., 2017).")
        out = EF.extract_form(
            text, "", [{"name": "conc", "type": "text"}], "m",
            StubAI(StubModel('{"extractions":[{"name":"conc","value":"300 and 800 uM",'
                             f'"source_quote":"{text}"}}]}}')))
        f = self._field(out, "conc")
        assert f["confidence"] == "low", f
        assert f["needs_review"] is True
        assert "another paper" in f["review_reason"]

    def test_a_sound_extraction_keeps_its_confidence_and_says_nothing(self):
        out = EF.extract_form(
            "We enrolled 128 patients.", "", [{"name": "design", "type": "text"}], "m",
            StubAI(StubModel('{"extractions":[{"name":"design","value":"enrolled 128 patients",'
                             '"source_quote":"We enrolled 128 patients."}]}')))
        f = self._field(out, "design")
        assert f["confidence"] == "high"
        assert f["needs_review"] is False
        assert f["review_reason"] == ""


class TestUnsupportedFields:
    def test_it_names_a_field_whose_quote_does_not_support_it(self):
        assert EF.unsupported_fields([
            {"name": "a", "type": "text", "value": "double blind crossover",
             "source_quote": "Patients were followed for a year in this study."},
        ]) == ["a"]

    def test_it_leaves_a_supported_field_alone(self):
        assert EF.unsupported_fields([
            {"name": "a", "type": "text", "value": "randomised",
             "source_quote": "The trial was randomised and double blind throughout."},
        ]) == []

    def test_a_number_has_to_be_in_its_own_quote(self):
        assert EF.unsupported_fields([
            {"name": "n", "type": "number", "value": 272,
             "source_quote": "The mean age was 42.5 years in this cohort."},
        ]) == ["n"]

    def test_an_empty_field_is_not_unsupported(self):
        assert EF.unsupported_fields([
            {"name": "a", "type": "text", "value": None, "source_quote": ""}]) == []

    def test_a_field_with_no_real_quote_is_left_to_the_confidence_flag(self):
        assert EF.unsupported_fields([
            {"name": "a", "type": "text", "value": "x", "source_quote": "n=5"}]) == []

    def test_it_survives_junk(self):
        assert EF.unsupported_fields([]) == []
        assert EF.unsupported_fields(None) == []


class TestCollidingFields:
    """A field holding another field's answer.

    Asked for twelve at once, a model that cannot find a field sometimes hands
    back a neighbour's answer rather than nothing, and the copy is
    indistinguishable from a real extraction: it has a value, and a quote that
    really is in the paper. It is how "study period" came back as "15.64
    months" from the sentence giving the median overall survival.
    """

    def test_two_fields_with_the_same_value_and_quote_are_copies(self):
        quote = "The median OS of our cohort (n = 272) was 15.64 months."
        got = EF.colliding_fields([
            {"name": "os", "value": "15.64 months", "source_quote": quote},
            {"name": "followup", "value": "15.64 months", "source_quote": quote},
        ])
        assert sorted(got) == ["followup", "os"]

    def test_one_sentence_supporting_several_fields_is_not_a_copy(self):
        """The ordinary case, which must not be re-asked: a methods sentence is
        the source for the design, the sample size and the comparator at once."""
        quote = "We randomised 128 adults to metformin or placebo in a double-blind trial."
        assert EF.colliding_fields([
            {"name": "n", "value": "128", "source_quote": quote},
            {"name": "drug", "value": "metformin", "source_quote": quote},
            {"name": "design", "value": "randomised", "source_quote": quote},
        ]) == []

    def test_the_same_value_from_different_sentences_is_not_a_copy(self):
        # Two fields can legitimately share a number: a trial's randomised and
        # analysed counts are often equal, and each has its own source.
        assert EF.colliding_fields([
            {"name": "randomised", "value": "128", "source_quote": "We randomised 128 adults."},
            {"name": "analysed", "value": "128", "source_quote": "All 128 were analysed."},
        ]) == []

    def test_empty_fields_are_not_copies_of_each_other(self):
        assert EF.colliding_fields([
            {"name": "a", "value": None, "source_quote": ""},
            {"name": "b", "value": "", "source_quote": ""},
        ]) == []

    def test_a_value_with_no_real_quote_is_left_alone(self):
        # Nothing to have copied: without a source there is no shared sentence,
        # and the field is already flagged as unverified.
        assert EF.colliding_fields([
            {"name": "a", "value": "5", "source_quote": ""},
            {"name": "b", "value": "5", "source_quote": "n=5"},
        ]) == []

    def test_it_survives_junk(self):
        assert EF.colliding_fields([]) == []
        assert EF.colliding_fields(None) == []


class TestReAskingCopiedFields:
    """The endpoint asks a copied field again, on its own."""

    def _fields(self):
        return [{"id": "os", "label": "Median OS", "type": "text"},
                {"id": "followup", "label": "Study period", "type": "text"}]

    def _client(self):
        from fastapi.testclient import TestClient
        import api
        return TestClient(api.app), api

    def test_a_copied_field_is_asked_again_alone_and_replaced(self, monkeypatch):
        c, _ = self._client()
        quote = "The median OS of our cohort (n = 272) was 15.64 months."
        calls = []

        def fake(*, text, tables, fields, model_name, ai_service, num_ctx=None):
            calls.append([f["name"] for f in fields])
            if len(fields) == 1 and fields[0]["name"] == "followup":
                # Alone, with no neighbour to copy from, it finds the real one.
                return {"fields": [{"name": "followup", "value": "2015-2021",
                                    "type": "text", "status": "reported",
                                    "source_quote": "During the study period (2015-2021)",
                                    "confidence": "high", "needs_review": False}]}
            return {"fields": [{"name": n, "value": "15.64 months", "type": "text",
                                "status": "reported", "source_quote": quote,
                                "confidence": "high", "needs_review": False}
                               for n in [f["name"] for f in fields]]}

        import extraction_forms
        monkeypatch.setattr(extraction_forms, "colliding_fields",
                            EF.colliding_fields)      # the real detector
        monkeypatch.setattr(extraction_forms, "extract_form", fake)
        body = c.post("/api/extract/fields",
                      json={"text": "Methods. " * 200, "fields": self._fields()}).json()

        assert ["os", "followup"] in calls, "the first pass asks them together"
        assert ["followup"] in calls or ["os"] in calls, "no field was re-asked alone"
        assert body["values"]["followup"] == "2015-2021"

    def test_a_failed_re_ask_keeps_the_value_the_reviewer_could_correct(self, monkeypatch):
        """An empty cell is worse than a wrong one here: the reviewer can see a
        wrong value against its quote and fix it, and cannot see nothing."""
        c, _ = self._client()
        quote = "The median OS of our cohort (n = 272) was 15.64 months."

        def fake(*, text, tables, fields, model_name, ai_service, num_ctx=None):
            if len(fields) == 1:
                return {"fields": [], "error": "the model did not return usable JSON"}
            return {"fields": [{"name": f["name"], "value": "15.64 months", "type": "text",
                                "status": "reported", "source_quote": quote,
                                "confidence": "high", "needs_review": False}
                               for f in fields]}

        import extraction_forms
        monkeypatch.setattr(extraction_forms, "colliding_fields", EF.colliding_fields)
        monkeypatch.setattr(extraction_forms, "extract_form", fake)
        body = c.post("/api/extract/fields",
                      json={"text": "Methods. " * 200, "fields": self._fields()}).json()

        assert body["values"]["followup"] == "15.64 months"

    def test_an_unsupported_value_is_also_re_asked(self, monkeypatch):
        """The commoner failure: the model finds a value somewhere and attaches
        a sentence that reads as if it were the source."""
        c, _ = self._client()
        calls = []

        def fake(*, text, tables, fields, model_name, ai_service, num_ctx=None):
            names = [f["name"] for f in fields]
            calls.append(names)
            if names == ["os"]:
                return {"fields": [{"name": "os", "value": "15.64 months", "type": "text",
                                    "status": "reported", "confidence": "high",
                                    "source_quote": "The median OS of our cohort was 15.64 months.",
                                    "needs_review": False}]}
            return {"fields": [{"name": n, "value": "15.64 months", "type": "text",
                                "status": "reported", "confidence": "high",
                                "source_quote": "Death was considered as a failure event in the analysis.",
                                "needs_review": False} for n in names]}

        import extraction_forms
        monkeypatch.setattr(extraction_forms, "unsupported_fields", EF.unsupported_fields)
        monkeypatch.setattr(extraction_forms, "colliding_fields", EF.colliding_fields)
        monkeypatch.setattr(extraction_forms, "extract_form", fake)
        body = c.post("/api/extract/fields",
                      json={"text": "Methods. " * 200, "fields": self._fields()}).json()

        assert ["os"] in calls, f"the unsupported field was never re-asked: {calls}"
        quote = next(f["source_quote"] for f in body["fields"] if f["id"] == "os")
        assert "median OS" in quote, "the better-supported answer was not kept"

    def test_a_re_ask_that_is_no_better_supported_is_not_kept(self, monkeypatch):
        """A second guess is not automatically a better one, and churning the
        value would lose whatever the reviewer had already read."""
        c, _ = self._client()

        def fake(*, text, tables, fields, model_name, ai_service, num_ctx=None):
            names = [f["name"] for f in fields]
            bad = "Death was considered as a failure event in the analysis."
            if names == ["os"] or names == ["followup"]:
                return {"fields": [{"name": names[0], "value": "a different wrong answer",
                                    "type": "text", "status": "reported",
                                    "confidence": "high", "source_quote": bad,
                                    "needs_review": False}]}
            return {"fields": [{"name": n, "value": "15.64 months", "type": "text",
                                "status": "reported", "confidence": "high",
                                "source_quote": bad, "needs_review": False} for n in names]}

        import extraction_forms
        monkeypatch.setattr(extraction_forms, "unsupported_fields", EF.unsupported_fields)
        monkeypatch.setattr(extraction_forms, "colliding_fields", EF.colliding_fields)
        monkeypatch.setattr(extraction_forms, "extract_form", fake)
        body = c.post("/api/extract/fields",
                      json={"text": "Methods. " * 200, "fields": self._fields()}).json()

        assert body["values"]["os"] == "15.64 months"

    def test_a_field_the_paper_does_not_report_is_not_re_asked(self, monkeypatch):
        """Both detectors only name a field that has a value. Re-asking every
        empty one would double the calls for a paper that reports less."""
        c, _ = self._client()
        calls = []

        def fake(*, text, tables, fields, model_name, ai_service, num_ctx=None):
            names = [f["name"] for f in fields]
            calls.append(names)
            return {"fields": [{"name": n, "value": None, "type": "text",
                                "status": "not_reported", "confidence": "none",
                                "source_quote": "", "needs_review": True} for n in names]}

        import extraction_forms
        monkeypatch.setattr(extraction_forms, "unsupported_fields", EF.unsupported_fields)
        monkeypatch.setattr(extraction_forms, "colliding_fields", EF.colliding_fields)
        monkeypatch.setattr(extraction_forms, "extract_form", fake)
        c.post("/api/extract/fields",
               json={"text": "Methods. " * 200, "fields": self._fields()})

        assert calls == [["os", "followup"]], f"an empty field was re-asked: {calls}"

    def test_nothing_is_re_asked_when_nothing_collided(self, monkeypatch):
        c, _ = self._client()
        calls = []

        def fake(*, text, tables, fields, model_name, ai_service, num_ctx=None):
            calls.append([f["name"] for f in fields])
            return {"fields": [{"name": f["name"], "value": f["name"], "type": "text",
                                "status": "reported", "source_quote": f"a sentence about {f['name']} in the paper",
                                "confidence": "high", "needs_review": False}
                               for f in fields]}

        import extraction_forms
        monkeypatch.setattr(extraction_forms, "colliding_fields", EF.colliding_fields)
        monkeypatch.setattr(extraction_forms, "extract_form", fake)
        c.post("/api/extract/fields",
               json={"text": "Methods. " * 200, "fields": self._fields()}).json()

        assert calls == [["os", "followup"]], f"a second pass ran for nothing: {calls}"


class TestTheEndpointSendsTheWholeArticle:
    """The endpoint used to slice the text to 30,000 characters as well.

    Two truncations, the same number, and the inner one is sized to the
    context. The outer one simply deleted the last third of every long paper
    before the extractor could decide anything, which is why funding,
    conflicts of interest and data availability came back blank.
    """

    def _fields(self, n):
        return [{"id": f"f{i}", "label": f"Field {i}", "type": "text"} for i in range(n)]

    def _client(self):
        from fastapi.testclient import TestClient
        import api
        return TestClient(api.app), api

    def test_the_article_reaches_the_extractor_whole(self, monkeypatch):
        c, _ = self._client()
        sentinel = "Funding: supported by grant R01CA200234."
        paper = "Methods. " + ("filler sentence. " * 2500) + sentinel
        assert len(paper) > 40_000
        assert sentinel not in paper[:30_000]
        seen = {}

        def fake(*, text, tables, fields, model_name, ai_service, num_ctx=None):
            seen["text"] = text
            return {"fields": []}

        import extraction_forms
        monkeypatch.setattr(extraction_forms, "extract_form", fake)
        c.post("/api/extract/fields", json={"text": paper, "fields": self._fields(4)})

        assert seen["text"] == paper
        assert sentinel in seen["text"]

    def test_the_status_of_each_field_reaches_the_reviewer(self, monkeypatch):
        c, _ = self._client()

        def fake(*, text, tables, fields, model_name, ai_service, num_ctx=None):
            return {"fields": [
                {"name": "f0", "value": "v", "type": "text", "source_quote": "q",
                 "confidence": "high", "needs_review": False, "status": "reported"},
                {"name": "f1", "value": None, "type": "text", "source_quote": "",
                 "confidence": "none", "needs_review": True, "status": "not_reported"},
                {"name": "f2", "value": None, "type": "text", "source_quote": "",
                 "confidence": "none", "needs_review": True, "status": "unread"},
            ]}

        import extraction_forms
        monkeypatch.setattr(extraction_forms, "extract_form", fake)
        body = c.post("/api/extract/fields",
                      json={"text": "Methods. " * 100, "fields": self._fields(3)}).json()

        assert {f["id"]: f["status"] for f in body["fields"]} == {
            "f0": "reported", "f1": "not_reported", "f2": "unread"}

    def test_a_field_with_no_status_is_treated_as_unread(self, monkeypatch):
        """Never "not_reported" by default: that would put a finding nobody
        established into the review."""
        c, _ = self._client()

        def fake(*, text, tables, fields, model_name, ai_service, num_ctx=None):
            return {"fields": [{"name": "f0", "value": None, "type": "text",
                                "source_quote": "", "confidence": "none",
                                "needs_review": True}]}

        import extraction_forms
        monkeypatch.setattr(extraction_forms, "extract_form", fake)
        body = c.post("/api/extract/fields",
                      json={"text": "Methods. " * 100, "fields": self._fields(1)}).json()
        assert body["fields"][0]["status"] == "unread"


class TestMissingModel:
    """Extraction runs on a general-purpose model whatever the reviewer picked
    for screening, so it can be one they never downloaded. Saying so costs one
    request; finding out costs a run of empty rows."""

    def _client(self):
        from fastapi.testclient import TestClient
        import api
        return TestClient(api.app), api

    def test_it_says_which_model_to_install(self, monkeypatch):
        c, api = self._client()
        api._LOCAL_TAGS.update({"at": 0.0, "names": []})
        monkeypatch.setattr(api, "resolve_for_thinking", lambda m: "qwen2.5:7b")

        class R:
            def json(self_inner):
                return {"models": [{"name": "some-other-model:latest"}]}

        monkeypatch.setattr(api.requests, "get", lambda *a, **k: R())
        body = c.post("/api/extract/fields",
                      json={"text": "Methods. " * 60,
                            "fields": [{"id": "n", "label": "N", "type": "number"}]}).json()

        assert "not installed" in body["error"]
        assert "ollama pull qwen2.5:7b" in body["error"]
        assert body["values"] == {}

    def test_an_installed_model_is_not_complained_about(self, monkeypatch):
        c, api = self._client()
        api._LOCAL_TAGS.update({"at": 0.0, "names": []})
        monkeypatch.setattr(api, "resolve_for_thinking", lambda m: "qwen2.5:7b")

        class R:
            def json(self_inner):
                return {"models": [{"name": "qwen2.5:7b"}]}

        monkeypatch.setattr(api.requests, "get", lambda *a, **k: R())
        import extraction_forms
        monkeypatch.setattr(extraction_forms, "extract_form",
                            lambda **k: {"fields": [{"name": "n", "value": 272, "type": "number",
                                                     "source_quote": "", "confidence": "low",
                                                     "needs_review": True}]})
        body = c.post("/api/extract/fields",
                      json={"text": "Methods. " * 60,
                            "fields": [{"id": "n", "label": "N", "type": "number"}]}).json()

        assert body["error"] == ""
        assert body["values"]["n"] == 272

    def test_a_cloud_model_is_never_called_missing(self, monkeypatch):
        """Nothing local to check, and the keys are a separate question."""
        c, api = self._client()
        api._LOCAL_TAGS.update({"at": 0.0, "names": []})
        monkeypatch.setattr(api, "resolve_for_thinking", lambda m: "claude-3-5-sonnet")

        def boom(*a, **k):
            raise AssertionError("asked Ollama about a cloud model")

        monkeypatch.setattr(api.requests, "get", boom)
        import extraction_forms
        monkeypatch.setattr(extraction_forms, "extract_form", lambda **k: {"fields": []})
        body = c.post("/api/extract/fields",
                      json={"text": "Methods. " * 60, "model": "claude-3-5-sonnet",
                            "fields": [{"id": "n", "label": "N", "type": "number"}]}).json()
        assert body["error"] == ""

    def test_an_unreachable_ollama_does_not_block_the_run(self, monkeypatch):
        """If we cannot tell, let the call proceed and fail honestly."""
        c, api = self._client()
        api._LOCAL_TAGS.update({"at": 0.0, "names": []})
        monkeypatch.setattr(api, "resolve_for_thinking", lambda m: "qwen2.5:7b")

        def down(*a, **k):
            raise OSError("connection refused")

        monkeypatch.setattr(api.requests, "get", down)
        import extraction_forms
        monkeypatch.setattr(extraction_forms, "extract_form", lambda **k: {"fields": []})
        body = c.post("/api/extract/fields",
                      json={"text": "Methods. " * 60,
                            "fields": [{"id": "n", "label": "N", "type": "number"}]}).json()
        assert "not installed" not in (body.get("error") or "")
