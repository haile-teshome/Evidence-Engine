"""Tests for the benchmarked duplicate-detection rules (dedup_rules.py). Run: pytest test_dedup_rules.py"""
from dedup_rules import flag_duplicates, norm_doi, norm_title, norm_first_page, doi_from_text

R = lambda title, **k: dict(title=title, **k)


class TestNormalisation:
    def test_doi_prefixes_case_and_url_encoding(self):
        assert norm_doi("https://doi.org/10.1016/S0140-6736%2818%2932091-1") == "10.1016/s0140-6736(18)32091-1"
        assert norm_doi("doi: 10.1000/ABC.") == "10.1000/abc"

    def test_title_ignores_spacing_hyphens_tags_and_language_tag(self):
        assert norm_title("17beta-Estradiol prevents dysfunction") == norm_title("17 beta Estradiol prevents dysfunction")
        assert norm_title("Alpha<inf>2A</inf> receptors") == norm_title("Alpha2A receptors")
        assert norm_title("Protective effects of incretin. [Japanese]") == norm_title("Protective effects of incretin")

    def test_first_page_ignores_junk(self):
        assert norm_first_page("643-645", 1992) == "643" and norm_first_page("643-5", 1992) == "643"
        assert norm_first_page("2010", 2010) == "" and norm_first_page("Sep-20", 2010) == "" and norm_first_page("e29", 2019) == "29"

    def test_doi_from_id_or_url(self):
        assert doi_from_text("", "W123", "https://doi.org/10.1002/abc") == "10.1002/abc"
        assert doi_from_text("", "12345678", "https://pubmed.ncbi.nlm.nih.gov/12345678/") == ""


class TestRules:
    def test_same_doi_with_agreeing_titles(self):
        a = R("Erratum: Tissue engineering in animal models", doi="10.1371/journal.pone.0105484")
        b = R("Tissue engineering in animal models for urinary diversion", doi="10.1371/journal.pone.0105484")
        assert flag_duplicates([a, b]) == [False, True]

    def test_shared_supplement_doi_with_unrelated_titles_is_not_a_duplicate(self):
        a = R("Atorvastatin at the onset of reperfusion", doi="10.1016/j.yjmcc.2010.06.001", year="2010")
        b = R("Akt1 is essential for ischaemic preconditioning", doi="10.1016/j.yjmcc.2010.06.001", year="2010")
        assert flag_duplicates([a, b]) == [False, False]

    def test_fuzzy_title_same_year_and_pages(self):
        a = R("A polymeric prostaglandin (PGBx) attenuates adenine nucleotide loss", year="1985", pages="775-83", authors="Pieper G. M.")
        b = R("A polymeric prostaglandin (PGB(x)) attenuates adenine nucleotide loss", year="1985", pages="775-783", authors="Pieper GM")
        assert flag_duplicates([a, b]) == [False, True]

    def test_same_title_but_conflicting_pages_are_distinct(self):
        a = R("Adrenergic reactions of sheep rumen", year="1980", pages="349-356", journal="Acta Physiol Pol")
        b = R("Adrenergic reactions of sheep rumen", year="1980", pages="341-348", journal="Acta Physiol Pol")
        assert flag_duplicates([a, b]) == [False, False]

    def test_cochrane_versions_are_distinct(self):
        a = R("Omega-3 fatty acids for the primary and secondary prevention", doi="10.1002/14651858.CD003177.pub3", year="2018", pages="CD003177")
        b = R("Omega-3 fatty acids for the primary and secondary prevention", doi="10.1002/14651858.CD003177.pub4", year="2018", pages="CD003177")
        assert flag_duplicates([a, b]) == [False, False]

    def test_years_more_than_one_apart_are_distinct(self):
        a = R("NTP research reports", year="2016"); b = R("NTP research reports", year="2019")
        assert flag_duplicates([a, b]) == [False, False]

    def test_epub_year_one_apart_with_same_pages(self):
        a = R("Behavioral and neurochemical characterization of mice", year="2007", pages="113-124", authors="Smith J")
        b = R("Behavioral and neurochemical characterization of mice", year="2008", pages="113-124", authors="Smith J")
        assert flag_duplicates([a, b]) == [False, True]

    def test_title_and_doi_only_records_still_collapse(self):
        a = R("A study of periodontal disease"); b = R("A Study, of Periodontal Disease.")
        assert flag_duplicates([a, b]) == [False, True]

    def test_different_titles_are_kept(self):
        assert flag_duplicates([R("Study A"), R("Study B")]) == [False, False]

    def test_keeps_the_first_occurrence(self):
        flags = flag_duplicates([R("Same work", year="2001"), R("Same work", year="2001"), R("Same work", year="2001")])
        assert flags == [False, True, True]
