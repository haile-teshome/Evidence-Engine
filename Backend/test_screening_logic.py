"""Tests for the deterministic screening logic.

Run: Backend/.venv/bin/python -m pytest Backend/test_screening_logic.py

Every case here pins a defect that actually shipped. The LLM is not under test:
these are the pure functions that decide what its answers MEAN, and each one
failed silently in production at some point, which is exactly why they are worth
locking down. Where a test corresponds to a real incident, the docstring says so.
"""
import re

import pytest

import api
import frameworks
from request_creds import set_request_creds


# ---------------------------------------------------------------------------
# _decide_from_votes — the verdict must follow the panel the reviewer is shown.
#
# Incident: an export had 41/120 rows whose Decision contradicted the votes
# displayed beside it, because the decision came from a second, independent LLM
# call instead of from the votes.
# ---------------------------------------------------------------------------

def _panel(**votes):
    """Build an assessment dict in the shape _pico_assess returns."""
    return {eid: {"vote": v, "evidence": "", "reasoning": ""} for eid, v in votes.items()}


class TestDecideFromVotes:
    def test_fail_on_discriminating_element_excludes(self):
        decision, reason = api._decide_from_votes(
            _panel(population="PASS", concept="FAIL", context="PASS"), "pcc")
        assert decision == "EXCLUDE"
        assert "Concept" in reason

    def test_pass_on_all_includes(self):
        decision, reason = api._decide_from_votes(
            _panel(population="PASS", concept="PASS", context="PASS"), "pcc")
        assert decision == "INCLUDE"
        assert reason == ""

    def test_partial_on_discriminating_includes(self):
        """PARTIAL is deliberately NOT an exclusion: abstract-level uncertainty
        goes to full text. Locking this so the sensitivity/workload tradeoff is
        never changed by accident."""
        decision, _ = api._decide_from_votes(_panel(concept="PARTIAL"), "pcc")
        assert decision == "INCLUDE"

    def test_fail_on_non_discriminating_element_does_not_exclude(self):
        """Context is descriptive scope in PCC. A FAIL there must not exclude."""
        decision, _ = api._decide_from_votes(
            _panel(population="PASS", concept="PASS", context="FAIL"), "pcc")
        assert decision == "INCLUDE"

    def test_all_na_excludes_as_unassessable(self):
        decision, reason = api._decide_from_votes(
            _panel(population="NA", concept="NA", context="NA"), "pcc")
        assert decision == "EXCLUDE"
        assert "assessable" in reason.lower()

    def test_missing_element_is_treated_as_na_not_a_crash(self):
        decision, _ = api._decide_from_votes({"concept": {"vote": "PASS"}}, "pcc")
        assert decision in ("INCLUDE", "EXCLUDE")

    def test_none_vote_does_not_crash(self):
        decision, _ = api._decide_from_votes({"concept": {"vote": None}}, "pcc")
        assert decision == "EXCLUDE"

    def test_lowercase_votes_are_normalised(self):
        decision, _ = api._decide_from_votes(_panel(concept="fail"), "pcc")
        assert decision == "EXCLUDE"

    def test_pico_uses_its_own_discriminating_elements(self):
        """PICO discriminates on population+intervention, so a FAIL on outcome
        must not exclude."""
        panel = _panel(population="PASS", intervention="PASS",
                       comparator="NA", outcome="FAIL")
        decision, _ = api._decide_from_votes(panel, "pico")
        assert decision == "INCLUDE"
        decision, _ = api._decide_from_votes(_panel(population="FAIL"), "pico")
        assert decision == "EXCLUDE"


# ---------------------------------------------------------------------------
# Element decomposition — "or" means either, "and" means both.
#
# Incident: the model split "prediction models OR decision support tools" into
# two REQUIRED parts, which silently demanded both and dropped papers that had
# only one. The prompt forbids it; this guard enforces it.
# ---------------------------------------------------------------------------

SRC = ("AI/ML prediction models or clinical decision support tools using electronic "
       "health records that contain both medical and dental data")


def _part(name, appears=(), anti=()):
    return {"part": name, "appears_as": list(appears), "not_satisfied_by": list(anti)}


class TestMergeOrAlternatives:
    def test_merges_parts_split_across_or(self):
        merged = api._merge_or_alternatives([
            _part("AI/ML prediction models", anti=["statistics only"]),
            _part("clinical decision support tools", anti=["info systems"]),
            _part("electronic health records", anti=["paper charts"]),
        ], SRC)
        assert len(merged) == 2
        assert merged[0]["part"] == "AI/ML prediction models or clinical decision support tools"

    def test_merged_part_drops_inherited_near_misses(self):
        """A near miss for one alternative is satisfied by the other, so the
        merged part must not inherit either half's disqualifiers."""
        merged = api._merge_or_alternatives([
            _part("AI/ML prediction models", anti=["statistics only"]),
            _part("clinical decision support tools", anti=["info systems"]),
        ], SRC)
        assert merged[0]["not_satisfied_by"] == []

    def test_does_not_merge_when_there_is_no_or(self):
        src = "school-based mindfulness programmes delivered by trained teachers"
        parts = [_part("school-based mindfulness programmes"), _part("trained teachers")]
        assert len(api._merge_or_alternatives(parts, src)) == 2

    def test_unmatchable_part_is_preserved_not_dropped(self):
        parts = [_part("something absent from the source text entirely")]
        assert len(api._merge_or_alternatives(parts, SRC)) == 1

    def test_empty_input(self):
        assert api._merge_or_alternatives([], SRC) == []


class TestLeadingConnective:
    """Incident: decomposition returned the fragment "that contain both medical
    and dental data", which rendered as "Does this study involve that contain
    both medical and dental data?" — a malformed question the model answered
    "yes" to on every paper, so the discriminating part never discriminated."""

    @pytest.mark.parametrize("raw,expected", [
        ("that contain both medical and dental data", "both medical and dental data"),
        ("using electronic health records", "electronic health records"),
        ("containing soil measurements", "soil measurements"),
        ("with trained teachers", "trained teachers"),
        ("involving human participants", "human participants"),
    ])
    def test_strips_leading_connectives(self, raw, expected):
        assert api._LEADING_CONNECTIVE.sub("", raw).strip() == expected

    @pytest.mark.parametrize("raw", [
        "records holding both A and B",
        "satellite imagery",
        "electronic health records",
        "usingivity",           # must not strip a prefix that is part of a word
    ])
    def test_leaves_wellformed_noun_phrases_alone(self, raw):
        assert api._LEADING_CONNECTIVE.sub("", raw).strip() == raw


# ---------------------------------------------------------------------------
# Contact email — a placeholder is worse than nothing.
#
# Incident: ENTREZ_EMAIL was "researcher@example.com". Unpaywall rejects
# example.com with HTTP 422, and the caller discarded the error, so an entire
# acquisition tier was dead for every paper without anything reporting it.
# ---------------------------------------------------------------------------

class TestContactEmail:
    def teardown_method(self):
        set_request_creds({})

    @pytest.mark.parametrize("bad", [
        "", "   ", "notanemail", "a@b", "@nodomain.com", "no-at-sign.com",
        "someone@example.com", "someone@example.org", "test@localhost",
    ])
    def test_rejects_unusable_addresses(self, bad):
        set_request_creds({"contact_email": bad})
        assert api._contact_email() == ""

    @pytest.mark.parametrize("good", [
        "j.smith@ucsf.edu", "R.Patel@lab.uni-koeln.de", "a.b+tag@sub.domain.org",
    ])
    def test_accepts_real_addresses(self, good):
        set_request_creds({"contact_email": good})
        assert api._contact_email() == good

    def test_absent_credential_is_empty_not_an_error(self):
        set_request_creds({})
        assert api._contact_email() == ""

    def test_polite_params_omits_mailto_when_unset(self):
        """Sending a fabricated address is worse than sending none."""
        set_request_creds({})
        assert "mailto" not in api._polite_params({"per_page": 1})

    def test_polite_params_adds_mailto_when_set(self):
        set_request_creds({"contact_email": "j.smith@ucsf.edu"})
        assert api._polite_params({"per_page": 1})["mailto"] == "j.smith@ucsf.edu"

    def test_polite_params_preserves_caller_params(self):
        set_request_creds({})
        assert api._polite_params({"per_page": 7})["per_page"] == 7


# ---------------------------------------------------------------------------
# Framework registry
# ---------------------------------------------------------------------------

class TestFrameworks:
    def test_pcc_discriminates_on_concept_only(self):
        assert frameworks.discriminating_ids("pcc") == ["concept"]

    def test_pico_discriminates_on_population_and_intervention(self):
        assert frameworks.discriminating_ids("pico") == ["population", "intervention"]

    def test_pcc_element_order(self):
        assert frameworks.element_ids("pcc") == ["population", "concept", "context"]

    def test_labels_are_human_readable(self):
        assert frameworks.label_for("pcc", "concept") == "Concept"

    def test_unknown_framework_falls_back_rather_than_raising(self):
        assert frameworks.discriminating_ids("nonsense")

    def test_none_framework_falls_back(self):
        assert frameworks.discriminating_ids(None)
