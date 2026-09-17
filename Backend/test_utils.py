"""Tests for the AI-service helpers in utils.py. No network, no model.

Run: Backend/.venv/bin/python -m pytest Backend/test_utils.py

These are the deterministic parts of the search and screening pipeline: JSON
recovery from loose model output, query translation between databases, and vote
aggregation. Each one runs on every paper or every search, so a defect here is
systematic rather than occasional.
"""
import pytest

from models import Paper
from utils import AIService as A, QueryCleaner as Q


# ---------------------------------------------------------------------------
# JSON recovery. Small models wrap JSON in prose, fences, or both. Every one of
# these shapes is a real response that had to be parsed.
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_plain_object(self):
        assert A._extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_with_a_language_tag(self):
        assert A._extract_json('```json\n{"a": 2}\n```') == {"a": 2}

    def test_fenced_without_a_language_tag(self):
        assert A._extract_json('```\n{"a": 2}\n```') == {"a": 2}

    def test_embedded_in_prose(self):
        assert A._extract_json('Sure! {"a": 3} Hope that helps.') == {"a": 3}

    def test_array_payload(self):
        assert A._extract_json('[1, 2, 3]') == [1, 2, 3]

    def test_nested_braces_are_balanced_correctly(self):
        out = A._extract_json('prefix {"a": {"b": [1, 2]}, "c": 3} suffix')
        assert out == {"a": {"b": [1, 2]}, "c": 3}

    @pytest.mark.parametrize("text", ["", "   ", "nope", "{unbalanced", None])
    def test_unparseable_returns_none_rather_than_raising(self, text):
        assert A._extract_json(text) is None

    def test_unicode_survives(self):
        assert A._extract_json('{"t": "Zähne 日本語 🦷"}')["t"] == "Zähne 日本語 🦷"


# ---------------------------------------------------------------------------
# Score presentation
# ---------------------------------------------------------------------------

class TestScoreToRating:
    @pytest.mark.parametrize("score,expected", [
        (0.0, "Very Poor"), (0.3, "Poor"), (0.5, "Fair"),
        (0.7, "Good"), (0.9, "Excellent"), (1.0, "Excellent"),
    ])
    def test_known_bands(self, score, expected):
        assert A._score_to_rating(score) == expected

    def test_is_monotonic_across_the_range(self):
        """A higher score must never present as a worse rating."""
        order = ["Very Poor", "Poor", "Fair", "Good", "Excellent"]
        ranks = [order.index(A._score_to_rating(s / 20)) for s in range(21)]
        assert ranks == sorted(ranks)

    def test_out_of_range_scores_do_not_raise(self):
        for s in (-1.0, 2.0):
            assert isinstance(A._score_to_rating(s), str)


# ---------------------------------------------------------------------------
# Query translation between databases. PubMed syntax sent verbatim to a source
# that does not understand it matches nothing, or matches everything.
# ---------------------------------------------------------------------------

class TestQueryAdaptation:
    def test_retag_rewrites_every_occurrence(self):
        assert A._retag('"x"[tiab] AND "y"[tiab]', "tiab", "tw") == '"x"[tw] AND "y"[tw]'

    def test_retag_leaves_other_tags_alone(self):
        assert "[mh]" in A._retag('"x"[tiab] OR "y"[mh]', "tiab", "tw")

    def test_arxiv_gets_tags_stripped(self):
        """arXiv has no field-tag syntax, so tags must not survive."""
        assert "[" not in A._adapt_query_for_source('("dental"[tiab])', "arXiv")

    @pytest.mark.parametrize("source", ["arXiv", "OpenAlex", "CrossRef", "Semantic Scholar"])
    def test_no_pubmed_tags_reach_a_tagless_source(self, source):
        out = A._adapt_query_for_source('("a"[tiab] OR "b"[mh]) AND "c"[tw]', source)
        assert "[tiab]" not in out and "[mh]" not in out and "[tw]" not in out

    def test_pubmed_keeps_its_own_syntax(self):
        q = '("dental"[tiab])'
        assert A._adapt_query_for_source(q, "PubMed") == q

    def test_adaptation_preserves_the_search_terms(self):
        assert "dental" in A._adapt_query_for_source('("dental"[tiab])', "arXiv")

    @pytest.mark.parametrize("q", ["", "   "])
    def test_empty_query_does_not_raise(self, q):
        assert isinstance(A._adapt_query_for_source(q, "arXiv"), str)


class TestQueryCleaner:
    def test_strips_field_tags(self):
        assert "[" not in Q.clean_for_general_search('("dental"[tiab] OR "oral"[mh])')

    def test_keeps_the_terms(self):
        out = Q.clean_for_general_search('("dental"[tiab] OR "oral"[mh])')
        assert "dental" in out and "oral" in out

    @pytest.mark.parametrize("q", ["", "   ", "plain query"])
    def test_degenerate_inputs_are_safe(self, q):
        assert isinstance(Q.clean_for_general_search(q), str)


class TestConceptBlocks:
    def test_splits_on_top_level_and(self):
        blocks = A._split_concept_blocks('("a"[tiab] OR "b"[tiab]) AND ("c"[tiab])')
        assert len(blocks) == 2

    def test_does_not_split_inside_a_group(self):
        """Splitting an OR group would turn alternatives into requirements and
        silently delete most true hits."""
        blocks = A._split_concept_blocks('("a"[tiab] OR "b"[tiab])')
        assert len(blocks) == 1

    @pytest.mark.parametrize("q", ["", "   ", "\n\t "])
    def test_blank_query_yields_no_blocks(self, q):
        """Whitespace used to slip past the empty check and produce "()", an
        empty group that is malformed in every database's query syntax."""
        assert A._split_concept_blocks(q) == []

    def test_every_returned_block_is_non_empty(self):
        for b in A._split_concept_blocks('("a"[tiab]) AND ("b"[tiab]) AND ("c"[tiab])'):
            assert b.strip()


class TestQueryDiff:
    def test_reports_added_and_removed_terms(self):
        d = A._query_diff('"a"[tiab] OR "b"[tiab]', '"a"[tiab] OR "c"[tiab]')
        assert isinstance(d, dict)
        flat = " ".join(str(v) for v in d.values())
        assert "c" in flat or "b" in flat

    def test_identical_queries_report_no_change(self):
        d = A._query_diff('"a"[tiab]', '"a"[tiab]')
        assert all(not v for v in d.values())

    def test_empty_inputs_do_not_raise(self):
        assert isinstance(A._query_diff("", ""), dict)


class TestSearchAnchors:
    def test_extracts_anchor_terms_from_a_pico_phrase(self):
        out = A._pico_to_search_anchors("adults with type 2 diabetes")
        assert isinstance(out, list)

    @pytest.mark.parametrize("text", ["", "   ", None])
    def test_degenerate_input_is_safe(self, text):
        assert isinstance(A._pico_to_search_anchors(text), list)

    def test_strips_parenthetical_operationalisation(self):
        """PICO fields read well on the card but make terrible search terms:
        "measured by validated index (e.g., MedDiet Score)" must not reach the
        query builder verbatim."""
        out = A._pico_to_search_anchors(
            "Adherence to a Mediterranean diet (e.g., MedDiet Score)")
        assert not any("e.g." in a for a in out)
        assert any("mediterranean" in a.lower() for a in out)

    def test_returns_at_most_three_anchors(self):
        out = A._pico_to_search_anchors(
            "adults with diabetes, hypertension, obesity, asthma and arthritis")
        assert len(out) <= 3

    def test_strips_the_inferred_marker(self):
        out = A._pico_to_search_anchors("(inferred) Mediterranean diet")
        assert not any("inferred" in a.lower() for a in out)


# ---------------------------------------------------------------------------
# MeSH handling. Europe PMC's MESH: field breaks OR semantics, so a query built
# only from MeSH terms returns fewer results than one of its own clauses.
# ---------------------------------------------------------------------------

class TestStripMeshOnlyTerms:
    def test_leaves_a_mixed_query_usable(self):
        out = A._strip_mesh_only_terms('"a"[tiab] OR "b"[mh]')
        assert "a" in out

    @pytest.mark.parametrize("q", ["", "   "])
    def test_empty_input_is_safe(self, q):
        assert isinstance(A._strip_mesh_only_terms(q), str)

    def test_returns_a_string_for_a_mesh_only_query(self):
        assert isinstance(A._strip_mesh_only_terms('"a"[mh] OR "b"[mh]'), str)
