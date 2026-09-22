"""Tests for per-database query translation.

Run: pytest test_query_translation.py

The bug these guard against does not look like a bug. Every one of these engines
accepts a malformed query and returns results, so a mistranslation shows up as a
search that ran fine and answered a slightly different question. That is why the
assertions are about STRUCTURE (is the AND still there, is the field scoping
still there, did every block survive) rather than about exact strings.

The query used throughout is the real one that exposed this: four AND-ed blocks
where the last, "dental", is what makes the question specific. If a translation
drops a block, the search silently becomes a different review.
"""
import pytest

import query_translation as qt
from query_translation import to_openreview_query, Block, parse_pubmed_query, translate_query

# Four blocks; the fourth carries the discriminating term.
Q = ('("Artificial Intelligence"[Mesh] OR "machine learning"[tiab] OR AI[tiab]) '
     'AND ("Electronic Health Records"[Mesh] OR "electronic health record"[tiab] OR EHR[tiab]) '
     'AND ("medical data"[tiab] OR "clinical data"[tiab]) '
     'AND ("Dental Records"[Mesh] OR "dental data"[tiab] OR "oral health"[tiab])')

ALL_SOURCES = ["Scopus", "Web of Science", "IEEE Xplore", "Springer Nature", "OpenAlex",
               "arXiv", "DOAJ", "CORE", "ClinicalTrials.gov", "CrossRef",
               "Semantic Scholar", "bioRxiv", "medRxiv"]
BOOLEAN_SOURCES = ["Scopus", "Web of Science", "IEEE Xplore", "Springer Nature",
                   "OpenAlex", "arXiv", "DOAJ", "CORE", "ClinicalTrials.gov"]
KEYWORD_SOURCES = ["CrossRef", "Semantic Scholar", "bioRxiv", "medRxiv"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParse:
    def test_finds_every_and_block(self):
        assert len(parse_pubmed_query(Q)) == 4

    def test_separates_free_text_from_controlled_headings(self):
        b = parse_pubmed_query(Q)[0]
        assert "machine learning" in b.terms
        assert "Artificial Intelligence" in b.mesh

    def test_strips_field_tags_from_values(self):
        assert all("[" not in t for b in parse_pubmed_query(Q) for t in b.all_terms)

    def test_headings_are_usable_as_free_text(self):
        """Engines without a controlled vocabulary still benefit: "Dental
        Records" is also a phrase people write in abstracts."""
        assert "Dental Records" in parse_pubmed_query(Q)[3].all_terms

    def test_does_not_split_inside_a_nested_group(self):
        q = '(a[tiab] OR (b[tiab] AND c[tiab])) AND d[tiab]'
        assert len(parse_pubmed_query(q)) == 2

    def test_deduplicates_case_insensitively(self):
        b = Block(terms=["EHR", "ehr", "Ehr"])
        assert len(b.all_terms) == 1

    def test_an_untagged_term_is_kept_as_free_text(self):
        """Dropping a term changes the question, so unknown shapes are kept."""
        assert "dental" in parse_pubmed_query("dental AND x[tiab]")[0].all_terms

    def test_an_unknown_tag_keeps_its_value(self):
        assert "review" in parse_pubmed_query('"review"[pt] AND x[tiab]')[0].all_terms

    @pytest.mark.parametrize("bad", ["", "   ", "()", "AND", "((("])
    def test_degenerate_input_does_not_raise(self, bad):
        assert isinstance(parse_pubmed_query(bad), list)


# ---------------------------------------------------------------------------
# The property every dialect must hold
# ---------------------------------------------------------------------------

class TestAllDialects:
    @pytest.mark.parametrize("src", ALL_SOURCES)
    def test_the_discriminating_term_survives(self, src):
        """THE test. "dental" is the one word that makes this question specific;
        a translation that loses it returns a different review's results."""
        assert "dental" in translate_query(Q, src).lower()

    @pytest.mark.parametrize("src", ALL_SOURCES)
    def test_no_pubmed_tags_leak_through(self, src):
        """A keyword engine matches "[Mesh]" as literal text."""
        out = translate_query(Q, src).lower()
        assert "[mesh]" not in out and "[tiab]" not in out

    @pytest.mark.parametrize("src", ALL_SOURCES)
    def test_output_is_never_empty(self, src):
        assert translate_query(Q, src).strip()

    @pytest.mark.parametrize("src", BOOLEAN_SOURCES)
    def test_every_block_stays_required(self, src):
        """Blocks are AND-ed. Losing one widens the question silently."""
        assert translate_query(Q, src).count(" AND ") == 3

    @pytest.mark.parametrize("src", BOOLEAN_SOURCES)
    def test_parentheses_are_balanced(self, src):
        out = translate_query(Q, src)
        assert out.count("(") == out.count(")")

    @pytest.mark.parametrize("src", KEYWORD_SOURCES)
    def test_keyword_engines_get_no_boolean_operators(self, src):
        """These score a bag of words, so an AND becomes a term to match."""
        assert " AND " not in translate_query(Q, src)

    @pytest.mark.parametrize("src", KEYWORD_SOURCES)
    def test_keyword_flattening_represents_every_block(self, src):
        """Taking terms in order would spend the budget on block one and drop
        exactly the terms that make the question specific."""
        out = translate_query(Q, src).lower()
        assert "dental" in out
        assert any(w in out for w in ("ehr", "electronic health"))


# ---------------------------------------------------------------------------
# Per-dialect syntax
# ---------------------------------------------------------------------------

class TestDialectSyntax:
    def test_scopus_scopes_to_title_abstract_keyword(self):
        assert "TITLE-ABS-KEY(" in qt.to_scopus_query(Q)

    def test_web_of_science_uses_topic_search(self):
        out = qt.to_wos_query(Q)
        assert out.startswith("TS=(")
        assert out.count("TS=(") == 4      # one per block, not one per term

    def test_arxiv_scopes_to_the_abstract(self):
        assert "abs:" in qt.to_arxiv_query(Q)

    def test_doaj_uses_bibjson_dot_notation(self):
        out = qt.to_doaj_query(Q)
        assert "bibjson.title:" in out and "bibjson.abstract:" in out

    def test_doaj_emits_no_wildcards(self):
        """DOAJ disables wildcard and regex queries for performance."""
        assert "*" not in qt.to_doaj_query(Q)

    def test_core_scopes_to_title_and_abstract(self):
        out = qt.to_core_query(Q)
        assert "title:" in out and "abstract:" in out

    def test_openalex_booleans_are_uppercase(self):
        """OpenAlex only treats AND/OR/NOT as operators when capitalised."""
        out = qt.to_openalex_query(Q)
        assert " OR " in out and " or " not in out

    def test_ieee_emits_plain_boolean_without_field_tags(self):
        """The Metadata API documents no field-qualified syntax."""
        out = qt.to_ieee_query(Q)
        assert " AND " in out and ":" not in out

    def test_ieee_query_stays_short_enough_for_a_url(self):
        assert len(qt.to_ieee_query(Q)) < 2000

    def test_ctgov_emits_essie_compatible_boolean(self):
        out = qt.to_ctgov_query(Q)
        assert " AND " in out and "AREA[" not in out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_pubmed_is_passed_through_unchanged(self):
        assert translate_query(Q, "PubMed") == Q

    def test_europe_pmc_uses_its_own_translator(self):
        assert "TITLE_ABS:" in translate_query(Q, "Europe PMC")

    def test_an_unknown_source_is_left_alone(self):
        """Better PubMed-ish than a guess at an engine we have not modelled."""
        assert translate_query(Q, "Some New Database") == Q

    def test_empty_query_stays_empty(self):
        assert translate_query("", "Scopus") == ""

    def test_a_translator_error_falls_back_instead_of_failing_the_search(self, monkeypatch):
        monkeypatch.setitem(qt._TRANSLATORS, "Scopus",
                            lambda q: (_ for _ in ()).throw(RuntimeError("boom")))
        assert translate_query(Q, "Scopus") == Q

    def test_every_ui_source_is_covered(self):
        """A source added to the UI without a translator silently gets PubMed
        syntax, which is how this class of bug arrived in the first place."""
        ui_sources = set(ALL_SOURCES) | {"PubMed", "Europe PMC"}
        handled = set(qt._TRANSLATORS) | {"PubMed", "Europe PMC"}
        assert ui_sources <= handled


# ---------------------------------------------------------------------------
# OpenReview.
#
# A term search, not a boolean one: no field tags, no AND / OR. A PubMed string
# sent verbatim would be searched for its own brackets and [tiab] markers.
# ---------------------------------------------------------------------------

class TestOpenReview:
    Q = ('("artificial intelligence"[tiab] OR "machine learning"[tiab]) '
         'AND ("clinical notes"[tiab] OR "electronic health records"[tiab])')

    def test_strips_every_quotation_mark(self):
        """Measured against the live API: a term of several quoted phrases
        returns ZERO results, while the same words unquoted return matches."""
        out = to_openreview_query(self.Q)
        assert '"' not in out and "'" not in out

    def test_keeps_no_pubmed_syntax(self):
        out = to_openreview_query(self.Q)
        for marker in ("[tiab]", "[Mesh]", " AND ", " OR ", "(", ")"):
            assert marker not in out, f"{marker!r} survived into the term"

    def test_represents_every_block_of_the_question(self):
        """Taking the first terms overall would spend the budget on the first
        concept and drop the one that makes the question specific."""
        out = to_openreview_query(self.Q).lower()
        assert "artificial intelligence" in out or "machine learning" in out
        assert "clinical notes" in out or "electronic health records" in out

    def test_collapses_whitespace(self):
        assert "  " not in to_openreview_query(self.Q)

    def test_passes_a_plain_query_through(self):
        assert to_openreview_query("machine learning") == "machine learning"

    def test_is_registered_for_the_source(self):
        assert translate_query(self.Q, "OpenReview") == to_openreview_query(self.Q)
