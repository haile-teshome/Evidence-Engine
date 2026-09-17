"""Tests for the domain models, framework detection, and credential plumbing.

Run: Backend/.venv/bin/python -m pytest Backend/test_models_and_frameworks.py

Small surface, high traffic. Framework detection picks the question frame for
the whole review, and `to_dict` shapes what the frontend renders, so a defect
here is broad rather than deep.
"""
import pytest

import frameworks
from models import PICOCriteria, Paper, clean_markup
from request_creds import get_cred, set_request_creds


# ---------------------------------------------------------------------------
# Framework detection. Picking PICO for a scoping review means screening
# against intervention/comparator/outcome, which a scoping review does not have.
# ---------------------------------------------------------------------------

class TestDetectFramework:
    @pytest.mark.parametrize("question", [
        "What is the effect of metformin versus placebo on HbA1c in adults with type 2 diabetes?",
        "Does exercise reduce blood pressure compared with usual care?",
        "Efficacy of drug A versus drug B on mortality",
    ])
    def test_comparative_effectiveness_questions_are_pico(self, question):
        assert frameworks.detect_framework(question) == "pico"

    @pytest.mark.parametrize("question", [
        "What is known about AI applied to dental records? A scoping review of the literature.",
        "A scoping review of digital health tools in dentistry",
    ])
    def test_scoping_questions_are_pcc(self, question):
        assert frameworks.detect_framework(question) == "pcc"

    @pytest.mark.parametrize("question", ["", "   ", None, "asdf"])
    def test_ambiguous_input_still_returns_a_valid_framework(self, question):
        """Never None and never a crash: the whole pipeline keys off this."""
        assert frameworks.detect_framework(question) in ("pico", "pcc")

    def test_detection_is_deterministic(self):
        q = "A scoping review of AI in dentistry"
        assert len({frameworks.detect_framework(q) for _ in range(5)}) == 1


class TestFrameworkLabels:
    def test_known_elements_get_their_proper_label(self):
        assert frameworks.label_for("pico", "population") == "Population"
        assert frameworks.label_for("pcc", "context") == "Context"

    def test_an_unknown_element_still_yields_a_readable_label(self):
        assert frameworks.label_for("pcc", "nonsense")

    def test_every_element_of_every_framework_has_a_label(self):
        for fw in ("pico", "pcc"):
            for eid in frameworks.element_ids(fw):
                assert frameworks.label_for(fw, eid).strip()

    def test_discriminating_ids_are_a_subset_of_element_ids(self):
        """A discriminating element that is not in the frame could never be
        voted on, so it would silently never exclude anything."""
        for fw in ("pico", "pcc"):
            assert set(frameworks.discriminating_ids(fw)) <= set(frameworks.element_ids(fw))


# ---------------------------------------------------------------------------
# PICOCriteria
# ---------------------------------------------------------------------------

class TestPicoCriteria:
    def test_to_dict_carries_both_short_and_long_keys(self):
        d = PICOCriteria(population="adults", intervention="drug").to_dict()
        assert d["p"] == "adults" and d["population"] == "adults"
        assert d["i"] == "drug"

    def test_to_dict_includes_the_framework(self):
        assert PICOCriteria(framework="pcc").to_dict()["framework"] == "pcc"

    def test_to_dict_never_omits_a_key_for_an_empty_field(self):
        """A missing key and an empty value render differently in the UI."""
        d = PICOCriteria().to_dict()
        for k in ("p", "i", "c", "o", "population", "framework"):
            assert k in d

    def test_element_items_follow_the_framework(self):
        items = PICOCriteria(population="adults", concept="AI", framework="pcc").element_items()
        assert [i[0] for i in items] == ["population", "concept", "context"]

    def test_element_items_carry_id_label_and_value(self):
        for eid, label, value in PICOCriteria(population="adults").element_items():
            assert isinstance(eid, str) and isinstance(label, str)
            assert value is not None

    def test_pico_framework_yields_four_elements(self):
        assert len(PICOCriteria(framework="pico").element_items()) == 4

    def test_defaults_are_empty_strings_not_none(self):
        p = PICOCriteria()
        assert p.population == "" and p.intervention == ""


class TestPaper:
    def test_to_dict_uses_the_titlecase_keys_the_frontend_and_csv_expect(self):
        p = Paper(source="PubMed", id="1", title="T", abstract="A", url="u")
        d = p.to_dict()
        assert d == {"Source": "PubMed", "ID": "1", "Title": "T",
                     "Abstract": "A", "URL": "u"}

    def test_score_is_omitted_when_absent_and_present_when_set(self):
        base = Paper(source="PubMed", id="1", title="T", abstract="A", url="u")
        assert "Score" not in base.to_dict() and "score" not in base.to_dict()
        scored = Paper(source="PubMed", id="1", title="T", abstract="A", url="u", score=7)
        assert 7 in scored.to_dict().values()

    def test_optional_fields_default_safely(self):
        p = Paper(source="PubMed", id="1", title="T", abstract="A")
        assert p.url == ""
        assert p.score is None


class TestCleanMarkup:
    def test_strips_html_tags(self):
        assert clean_markup("<b>Bold</b> text") == "Bold text"

    def test_decodes_entities(self):
        assert "&" in clean_markup("A &amp; B")

    @pytest.mark.parametrize("raw", ["", None, "   "])
    def test_degenerate_input_is_safe(self, raw):
        assert isinstance(clean_markup(raw), str)

    def test_plain_text_is_unchanged(self):
        assert clean_markup("Just plain text") == "Just plain text"

    def test_jats_wrapped_abstract_is_unwrapped(self):
        """Crossref returns abstracts wrapped in JATS tags."""
        assert clean_markup("<jats:p>An abstract.</jats:p>") == "An abstract."


# ---------------------------------------------------------------------------
# Per-request credentials
# ---------------------------------------------------------------------------

class TestRequestCreds:
    def teardown_method(self):
        set_request_creds({})

    def test_a_set_credential_is_readable(self):
        set_request_creds({"core": "abc"})
        assert get_cred("core") == "abc"

    def test_an_unset_credential_is_empty_not_none(self):
        set_request_creds({"core": "abc"})
        assert get_cred("openai") == ""

    def test_clearing_removes_everything(self):
        set_request_creds({"core": "abc"})
        set_request_creds({})
        assert get_cred("core") == ""

    def test_none_payload_is_treated_as_empty(self):
        set_request_creds(None)
        assert get_cred("core") == ""

    def test_credentials_do_not_leak_between_requests(self):
        """Each request installs its own map. One user's key must never be
        visible to the next request."""
        set_request_creds({"openai": "user-one-key"})
        set_request_creds({"openai": "user-two-key"})
        assert get_cred("openai") == "user-two-key"
