"""Tests for the deterministic parsing and query-translation logic.

Run: Backend/.venv/bin/python -m pytest Backend/test_data_parsing.py

No network. Every case pins a defect that shipped and corrupted data silently:
nothing raised, nothing logged, the review simply got worse. That is what makes
these functions worth locking down rather than the code around them.
"""
import pytest

from data_services import entrez_abstract, to_europepmc_query


class _Labelled(str):
    """Stand-in for Biopython's StringElement, which carries XML attributes."""
    def __new__(cls, text, label=None):
        s = super().__new__(cls, text)
        s.attributes = {"Label": label} if label else {}
        return s


# ---------------------------------------------------------------------------
# entrez_abstract
#
# Incident: the parser took AbstractText[0], which on a structured abstract is
# the BACKGROUND section alone — precisely the part that never says what data or
# methods a study used. Measured against PubMed, one abstract went from 1981
# characters to 158, and about half the corpus was affected. Nothing errored;
# screening quality just quietly dropped, because the model was being asked what
# data a study used while shown only its introduction.
# ---------------------------------------------------------------------------

class TestEntrezAbstract:
    def test_joins_every_section_of_a_structured_abstract(self):
        article = {"Abstract": {"AbstractText": [
            _Labelled("Periodontal disease is common.", "BACKGROUND"),
            _Labelled("We linked dental and medical records.", "METHODS"),
            _Labelled("AUC was 0.81.", "RESULTS"),
            _Labelled("The model is usable.", "CONCLUSIONS"),
        ]}}
        out = entrez_abstract(article)
        assert "METHODS" in out and "RESULTS" in out
        assert "linked dental and medical records" in out
        assert "AUC was 0.81" in out

    def test_does_not_return_only_the_first_section(self):
        """The exact regression. A background-only abstract is worse than useless
        for screening, because the discriminating detail lives in Methods."""
        article = {"Abstract": {"AbstractText": [
            _Labelled("Background text.", "BACKGROUND"),
            _Labelled("The methods that actually matter.", "METHODS"),
        ]}}
        assert "methods that actually matter" in entrez_abstract(article)

    def test_unlabelled_sections_are_joined_without_a_label_prefix(self):
        article = {"Abstract": {"AbstractText": ["First part.", "Second part."]}}
        out = entrez_abstract(article)
        assert out == "First part. Second part."
        assert ":" not in out

    def test_plain_string_abstract_passes_through(self):
        article = {"Abstract": {"AbstractText": "A single unstructured abstract."}}
        assert entrez_abstract(article) == "A single unstructured abstract."

    @pytest.mark.parametrize("article", [
        {},
        {"Abstract": {}},
        {"Abstract": {"AbstractText": []}},
        {"Abstract": None},
    ])
    def test_missing_abstract_returns_empty_string_not_a_crash(self, article):
        assert entrez_abstract(article) == ""

    def test_blank_sections_are_skipped(self):
        article = {"Abstract": {"AbstractText": ["Real text.", "", "   "]}}
        assert entrez_abstract(article) == "Real text."

    def test_longer_than_any_single_section(self):
        """Property form of the regression: the join must never be shorter than
        its longest input section."""
        sections = ["a" * 40, "b" * 900, "c" * 120]
        article = {"Abstract": {"AbstractText": sections}}
        assert len(entrez_abstract(article)) >= max(len(s) for s in sections)


# ---------------------------------------------------------------------------
# to_europepmc_query
#
# Incident: PubMed field tags were passed through untranslated, so Europe PMC ran
# them as full-text search and returned ~260x more records than intended. A
# separate issue: Europe PMC's MESH: field breaks OR semantics (MESH:"Dentistry"
# OR MESH:"Chronic Disease" returned FEWER hits than MESH:"Dentistry" alone), so
# those clauses are dropped rather than trusted.
# ---------------------------------------------------------------------------

class TestEuropePmcQuery:
    def test_translates_tiab_to_title_abs(self):
        out = to_europepmc_query('("dental"[tiab])')
        assert "TITLE_ABS:" in out
        assert "[tiab]" not in out

    @pytest.mark.parametrize("tag", ["[tiab]", "[tw]", "[ti]"])
    def test_all_supported_tags_are_translated(self, tag):
        out = to_europepmc_query(f'("dental"{tag})')
        assert tag not in out

    def test_no_pubmed_tags_survive(self):
        out = to_europepmc_query('("a"[tiab] OR "b"[mh] OR "c"[tw]) AND "d"[pt]')
        assert "[" not in out

    def test_drops_mesh_clauses(self):
        out = to_europepmc_query('"Dentistry"[mh] OR "Chronic Disease"[mh]')
        assert "MESH:" not in out

    def test_leaves_no_dangling_boolean_operators(self):
        out = to_europepmc_query('("dental"[tiab] OR "x"[mh]) AND "y"[mh]').strip()
        assert not out.upper().endswith((" AND", " OR", " NOT"))
        assert not out.upper().startswith(("AND ", "OR ", "NOT "))

    def test_leaves_no_empty_groups(self):
        out = to_europepmc_query('("a"[mh]) AND ("dental"[tiab])')
        assert "()" not in out.replace(" ", "")

    def test_balanced_parentheses(self):
        out = to_europepmc_query('(("a"[tiab] OR "b"[mh]) AND ("c"[tiab] OR "d"[mh]))')
        assert out.count("(") == out.count(")")

    def test_preserves_the_search_terms_themselves(self):
        out = to_europepmc_query('("periodontal disease"[tiab])')
        assert "periodontal disease" in out

    @pytest.mark.parametrize("q", ["", "   "])
    def test_empty_input_does_not_crash(self, q):
        assert isinstance(to_europepmc_query(q), str)

    def test_plain_query_without_tags_is_left_usable(self):
        out = to_europepmc_query("dental AND machine learning")
        assert "dental" in out and "machine learning" in out

    def test_is_idempotent(self):
        """Translating an already-translated query must not corrupt it."""
        once = to_europepmc_query('("dental"[tiab] OR "x"[mh])')
        assert to_europepmc_query(once) == once
