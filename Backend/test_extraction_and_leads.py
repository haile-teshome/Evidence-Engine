"""Tests for LEADS parsing, structured-form extraction helpers, and table parsing.

Run: Backend/.venv/bin/python -m pytest Backend/test_extraction_and_leads.py

These are the functions that turn a model's loose output into structured data a
reviewer will trust. The failure mode throughout is quiet: a number parsed from
the wrong place, or a value marked "high confidence" when nothing in the paper
supports it, both look completely normal on screen.
"""
import pytest

import extraction_forms as EF
import leads_screening as L
import table_recognition as TR


# ---------------------------------------------------------------------------
# LEADS model identification and tag resolution
# ---------------------------------------------------------------------------

class TestLeadsModelDetection:
    @pytest.mark.parametrize("name", [
        "leads",
        "LEADS",
        "hf.co/mradermacher/leads-mistral-7b-v1-GGUF:latest",
        "leads-mistral-7b",
        "some/leads_mistral/path",
    ])
    def test_recognises_leads_models(self, name):
        assert L.is_leads_model(name)

    @pytest.mark.parametrize("name", [
        None, "", "qwen2.5:7b", "llama3.1:8b", "medgemma", "mistral:7b",
    ])
    def test_does_not_misidentify_other_models(self, name):
        """A false positive here silently switches the whole screening strategy
        to the LEADS path, which decides independently of the PICO panel."""
        assert not L.is_leads_model(name)

    def test_short_alias_resolves_to_the_full_ollama_tag(self):
        assert L.resolve_model_name("leads") == L.LEADS_MODEL_NAME

    def test_other_names_pass_through_untouched(self):
        assert L.resolve_model_name("qwen2.5:7b") == "qwen2.5:7b"

    def test_none_passes_through(self):
        assert L.resolve_model_name(None) is None


# ---------------------------------------------------------------------------
# LEADS output parsing. A 7B model emits JSON in four different shapes; each
# fallback exists because a real response arrived that way.
# ---------------------------------------------------------------------------

class TestParseEvaluations:
    def test_plain_json(self):
        out = L._parse_evaluations('{"evaluations":[{"eligibility":"YES","rationale":"ok"}]}')
        assert out[0]["eligibility"] == "YES"

    def test_fenced_json_block(self):
        text = '```json\n{"evaluations":[{"eligibility":"NO","rationale":"nope"}]}\n```'
        assert L._parse_evaluations(text)[0]["eligibility"] == "NO"

    def test_json_embedded_in_prose(self):
        text = 'Sure! Here is my answer:\n{"evaluations":[{"eligibility":"PARTIAL","rationale":"maybe"}]}\nHope that helps.'
        assert L._parse_evaluations(text)[0]["eligibility"] == "PARTIAL"

    def test_regex_scrape_when_json_is_unparseable(self):
        text = 'broken { "eligibility": "YES", "rationale": "found it" ,,, '
        out = L._parse_evaluations(text)
        assert out and out[0]["eligibility"] == "YES"

    @pytest.mark.parametrize("text", ["", None, "no json at all", "{not json}"])
    def test_unusable_output_returns_empty_not_a_crash(self, text):
        assert L._parse_evaluations(text) == []

    def test_multiple_evaluations_are_all_returned(self):
        text = ('{"evaluations":[{"eligibility":"YES","rationale":"a"},'
                '{"eligibility":"NO","rationale":"b"}]}')
        assert len(L._parse_evaluations(text)) == 2


class TestLeadsScore:
    def test_all_yes_scores_one(self):
        assert L._score([{"eligibility": "YES"}, {"eligibility": "YES"}]) == 1.0

    def test_all_no_scores_minus_one(self):
        assert L._score([{"eligibility": "NO"}]) == -1.0

    def test_partial_is_half(self):
        assert L._score([{"eligibility": "PARTIAL"}]) == 0.5

    def test_uncertain_is_neutral(self):
        assert L._score([{"eligibility": "UNCERTAIN"}]) == 0.0

    def test_empty_scores_zero_rather_than_dividing_by_zero(self):
        assert L._score([]) == 0.0

    def test_is_a_mean_not_a_sum(self):
        assert L._score([{"eligibility": "YES"}, {"eligibility": "NO"}]) == 0.0

    def test_case_insensitive(self):
        assert L._score([{"eligibility": "yes"}]) == 1.0

    def test_unknown_label_is_ignored_but_still_counted(self):
        assert L._score([{"eligibility": "MAYBE"}]) == 0.0

    def test_missing_key_does_not_crash(self):
        assert L._score([{}]) == 0.0


# ---------------------------------------------------------------------------
# Numeric parsing for extraction forms
# ---------------------------------------------------------------------------

class TestNumericParsing:
    @pytest.mark.parametrize("raw,expected", [
        (42, 42.0), (4.2, 4.2), ("42", 42.0), ("4.2", 4.2),
        ("1,234", 1234.0), ("  7  ", 7.0), ("-3.5", -3.5),
        ("n=128 patients", 128.0), ("mean 12.5 (SD 3)", 12.5),
    ])
    def test_parses_numbers_from_messy_values(self, raw, expected):
        assert EF._num(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "NA", "n/a", "none", "NULL", "nan", "no digits"])
    def test_non_numeric_values_return_none(self, raw):
        assert EF._num(raw) is None

    def test_zero_is_a_value_not_a_missing_marker(self):
        """Returning None for 0 would silently drop real zero counts."""
        assert EF._num(0) == 0.0
        assert EF._num("0") == 0.0


class TestNumberGroundedInText:
    @pytest.mark.parametrize("val,text", [
        (128, "We enrolled 128 patients."),
        (1234, "A total of 1,234 records were screened."),
        (12.5, "The mean was 12.5 years."),
        (12.5, "The mean was 12.50 years."),
    ])
    def test_finds_the_number_however_it_is_written(self, val, text):
        assert EF._num_in_text(val, text)

    def test_absent_number_is_not_found(self):
        assert not EF._num_in_text(999, "We enrolled 128 patients.")

    def test_does_not_match_a_number_embedded_in_a_longer_one(self):
        """12 must not be "found" inside 128, or a fabricated value would be
        marked as grounded in the source."""
        assert not EF._num_in_text(12, "We enrolled 128 patients.")

    def test_does_not_match_across_a_decimal_point(self):
        assert not EF._num_in_text(5, "The value was 12.53 units.")


class TestConfidence:
    def test_empty_value_has_no_confidence(self):
        assert EF._confidence("number", "", "", "text") == "none"
        assert EF._confidence("number", None, "", "text") == "none"

    def test_numeric_value_present_in_the_source_is_high(self):
        t = next(iter(EF.NUMERIC_TYPES))
        assert EF._confidence(t, 128, "we enrolled 128 patients", "we enrolled 128 patients") == "high"

    def test_numeric_value_absent_from_the_source_is_low(self):
        """The exact fabrication guard: a number the paper never states must not
        be presented to the reviewer as high confidence."""
        t = next(iter(EF.NUMERIC_TYPES))
        assert EF._confidence(t, 999, "we enrolled 128 patients", "we enrolled 128 patients") == "low"

    def test_numeric_value_without_a_quote_is_low(self):
        t = next(iter(EF.NUMERIC_TYPES))
        assert EF._confidence(t, 128, "", "we enrolled 128 patients") == "low"

    def test_text_value_with_a_quote_is_high(self):
        assert EF._confidence("text", "randomised", "the trial was randomised", "...") == "high"

    def test_text_value_without_a_quote_is_low(self):
        assert EF._confidence("text", "randomised", "", "...") == "low"


class TestFieldLines:
    def test_renders_one_line_per_field(self):
        out = EF._field_lines([{"name": "sample_size", "type": "number"},
                               {"name": "design", "type": "text"}])
        assert "sample_size" in out and "design" in out

    def test_empty_field_list_does_not_crash(self):
        assert isinstance(EF._field_lines([]), str)


# ---------------------------------------------------------------------------
# Table parsing
# ---------------------------------------------------------------------------

class TestHtmlToRows:
    def test_parses_a_simple_table(self):
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
        assert TR._html_to_rows(html) == [["A", "B"], ["1", "2"]]

    def test_header_cells_are_included(self):
        html = "<table><tr><th>H1</th><th>H2</th></tr><tr><td>1</td><td>2</td></tr></table>"
        rows = TR._html_to_rows(html)
        assert rows[0] == ["H1", "H2"]

    def test_cell_text_is_stripped(self):
        html = "<table><tr><td>  padded  </td></tr></table>"
        assert TR._html_to_rows(html) == [["padded"]]

    @pytest.mark.parametrize("html", ["", "not html", "<table></table>", None])
    def test_unusable_input_returns_empty_not_a_crash(self, html):
        assert TR._html_to_rows(html) == []

    def test_available_reports_a_boolean_without_loading_models(self):
        """Called on a UI path, so it must be cheap and must never raise when the
        optional recognition dependencies are absent."""
        assert isinstance(TR.available(), bool)
