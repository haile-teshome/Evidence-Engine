"""Tests for screening orchestration and evidence anchoring.

Run: Backend/.venv/bin/python -m pytest Backend/test_screening_pipeline.py

The model is stubbed throughout. What is under test is the code around it: how a
loose response becomes a structured verdict, and whether a quote the model
supplies can actually be found in the paper.

Anchoring matters more than it looks. It is what separates a vote backed by the
source from one the model invented, and a mis-anchor puts a highlight on text
that does not support the claim beside it.
"""
import pytest

import api
from models import PICOCriteria


ABSTRACT = (
    "We enrolled 128 patients from the dental clinic between 2019 and 2023. "
    "A machine learning model was trained on linked medical and dental records. "
    "Results showed an AUC of 0.81 overall."
)


# ---------------------------------------------------------------------------
# Quote anchoring
# ---------------------------------------------------------------------------

class TestAnchorQuote:
    def test_exact_substring_anchors(self):
        span = api._anchor_quote_in_text("We enrolled 128 patients", ABSTRACT)
        assert span is not None
        s, e = span
        assert ABSTRACT[s:e] == "We enrolled 128 patients"

    def test_matching_is_case_insensitive(self):
        assert api._anchor_quote_in_text("we enrolled 128 PATIENTS", ABSTRACT) is not None

    def test_whitespace_drift_still_anchors(self):
        """Models reflow whitespace when quoting. That must not lose the anchor."""
        assert api._anchor_quote_in_text("We  enrolled   128\npatients", ABSTRACT) is not None

    def test_partial_token_overlap_anchors(self):
        assert api._anchor_quote_in_text(
            "enrolled 128 patients from the dental", ABSTRACT) is not None

    def test_unrelated_quote_does_not_anchor(self):
        """The fabrication guard. A quote that is not in the paper must return
        None so the vote gets downgraded rather than shown as evidence."""
        assert api._anchor_quote_in_text(
            "completely unrelated sentence about geology", ABSTRACT) is None

    @pytest.mark.parametrize("quote", ["", "   ", None])
    def test_empty_quote_does_not_anchor(self, quote):
        assert api._anchor_quote_in_text(quote, ABSTRACT) is None

    def test_empty_text_does_not_anchor(self):
        assert api._anchor_quote_in_text("anything", "") is None

    def test_returned_span_is_within_bounds_and_ordered(self):
        for q in ["We enrolled 128 patients", "AUC of 0.81", "machine learning model"]:
            span = api._anchor_quote_in_text(q, ABSTRACT)
            assert span is not None
            s, e = span
            assert 0 <= s < e <= len(ABSTRACT)

    def test_span_actually_covers_related_text(self):
        span = api._anchor_quote_in_text("AUC of 0.81", ABSTRACT)
        s, e = span
        assert "0.81" in ABSTRACT[s:e]


# ---------------------------------------------------------------------------
# _pico_assess: turning one model response into a per-element panel
# ---------------------------------------------------------------------------

class StubModel:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        if isinstance(self.reply, Exception):
            raise self.reply
        r = self.reply(len(self.prompts)) if callable(self.reply) else self.reply
        return type("R", (), {"content": r})()


PANEL = """{
  "population": {"vote":"PASS","evidence":"We enrolled 128 patients","reasoning":"patients"},
  "concept": {"vote":"PASS","evidence":"linked medical and dental records","reasoning":"linked"},
  "context": {"vote":"PASS","evidence":"between 2019 and 2023","reasoning":"in range"},
  "overall_reasoning": "Matches all three elements.",
  "bucket": "All elements met",
  "failed_criteria": []
}"""


def _paper():
    return api.PaperIn(id="1", source="PubMed", title="A linked-records study",
                       abstract=ABSTRACT, url="")


def _pcc():
    return api.PicoIn(population="any patients",
                      concept="machine learning using records with both medical and dental data",
                      context="any setting 2010 to present", framework="pcc")


@pytest.fixture
def stub(monkeypatch):
    def _install(reply=PANEL):
        m = StubModel(reply)
        monkeypatch.setattr(api.AIService, "get_model", lambda *a, **k: m)
        # Keep decomposition deterministic and free. TWO parts, because the
        # per-part loop deliberately skips an element it cannot decompose
        # (see test_single_part_elements_keep_the_one_shot_vote).
        monkeypatch.setattr(api, "_decompose_element", lambda text, model: [
            {"part": "machine learning", "appears_as": [], "not_satisfied_by": []},
            {"part": "medical and dental records", "appears_as": [], "not_satisfied_by": []},
        ])
        return m
    return _install


class TestPicoAssess:
    def test_returns_a_vote_for_every_framework_element(self, stub):
        stub()
        out = api._pico_assess(_paper(), _pcc(), "m")
        for eid in ("population", "concept", "context"):
            assert eid in out
            assert out[eid]["vote"] in ("PASS", "PARTIAL", "FAIL", "NA")

    def test_carries_overall_reasoning_and_bucket(self, stub):
        stub()
        out = api._pico_assess(_paper(), _pcc(), "m")
        assert out.get("overall_reasoning")
        assert out.get("bucket")

    def test_malformed_json_does_not_raise(self, stub):
        stub("this is not json at all")
        out = api._pico_assess(_paper(), _pcc(), "m")
        assert isinstance(out, dict)

    def test_model_exception_does_not_raise(self, stub):
        stub(RuntimeError("model down"))
        assert isinstance(api._pico_assess(_paper(), _pcc(), "m"), dict)

    def test_missing_elements_are_filled_in_rather_than_absent(self, stub):
        stub('{"population": {"vote":"PASS","evidence":"","reasoning":"x"}}')
        out = api._pico_assess(_paper(), _pcc(), "m")
        for eid in ("population", "concept", "context"):
            assert eid in out

    def test_overall_reasoning_is_never_blank(self, stub):
        """A record with no explanation is useless to a reviewer, so a blank
        must be replaced by a title-grounded fallback."""
        stub('{"population":{"vote":"NA","evidence":"","reasoning":""},'
             '"overall_reasoning":"","bucket":""}')
        out = api._pico_assess(_paper(), _pcc(), "m")
        assert out["overall_reasoning"].strip()

    def test_a_fabricated_quote_is_not_kept_as_evidence(self, stub):
        """The model returns a quote that appears nowhere in the abstract. It
        must not be presented to the reviewer as supporting evidence."""
        stub('{"population":{"vote":"PASS","evidence":"a sentence about geology",'
             '"reasoning":"x"},"overall_reasoning":"y","bucket":"z"}')
        out = api._pico_assess(_paper(), _pcc(), "m")
        ev = (out["population"].get("evidence") or "")
        assert "geology" not in ev

    def test_criteria_are_included_in_the_prompt_when_supplied(self, stub):
        m = stub()
        api._pico_assess(_paper(), _pcc(), "m",
                         inclusion=["Published 2010 or later"],
                         exclusion=["Animal studies"])
        prompt = m.prompts[0]
        assert "Published 2010 or later" in prompt
        assert "Animal studies" in prompt

    def test_protocol_text_is_included_when_supplied(self, stub):
        m = stub()
        api._pico_assess(_paper(), _pcc(), "m", protocol="PROTOCOL-MARKER-123")
        assert "PROTOCOL-MARKER-123" in m.prompts[0]

    def test_discriminating_elements_are_marked_in_the_prompt(self, stub):
        m = stub()
        api._pico_assess(_paper(), _pcc(), "m")
        assert "[DISCRIMINATING]" in m.prompts[0]

    def test_per_part_results_overwrite_both_vote_and_reasoning(self, stub, monkeypatch):
        """A vote derived from the part checks must carry reasoning derived from
        the same checks. Leaving the one-shot sentence produced panels that
        argued against their own verdict on 88 of 120 rows."""
        stub()
        monkeypatch.setattr(api, "_decompose_element", lambda t, m: [
            {"part": "machine learning", "appears_as": [], "not_satisfied_by": []},
            {"part": "unicorn husbandry", "appears_as": [], "not_satisfied_by": []},
        ])
        monkeypatch.setattr(api, "_part_present",
                            lambda part, t, a, m: "no" if "unicorn" in part["part"] else "yes")
        out = api._pico_assess(_paper(), _pcc(), "m")
        assert out["concept"]["vote"] == "FAIL"
        assert "unicorn husbandry" in out["concept"]["reasoning"]

    def test_all_parts_present_yields_pass_with_matching_reasoning(self, stub, monkeypatch):
        stub()
        monkeypatch.setattr(api, "_part_present", lambda *a, **k: "yes")
        out = api._pico_assess(_paper(), _pcc(), "m")
        assert out["concept"]["vote"] == "PASS"
        assert "All required parts present" in out["concept"]["reasoning"]

    def test_unclear_parts_yield_partial_with_matching_reasoning(self, stub, monkeypatch):
        stub()
        monkeypatch.setattr(api, "_decompose_element", lambda t, m: [
            {"part": "a", "appears_as": [], "not_satisfied_by": []},
            {"part": "b", "appears_as": [], "not_satisfied_by": []},
        ])
        monkeypatch.setattr(api, "_part_present", lambda *a, **k: "unclear")
        out = api._pico_assess(_paper(), _pcc(), "m")
        assert out["concept"]["vote"] == "PARTIAL"
        assert "does not establish" in out["concept"]["reasoning"].lower()

    def test_single_part_elements_keep_the_one_shot_vote(self, stub, monkeypatch):
        """Nothing to decompose means nothing to verify, so the per-part loop
        must leave the original vote alone rather than inventing one."""
        stub()
        monkeypatch.setattr(api, "_decompose_element",
                            lambda t, m: [{"part": "whole thing", "appears_as": [],
                                           "not_satisfied_by": []}])
        out = api._pico_assess(_paper(), _pcc(), "m")
        assert out["concept"]["vote"] == "PASS"

    def test_a_paper_with_no_abstract_is_still_assessed(self, stub):
        stub()
        p = api.PaperIn(id="1", source="PubMed", title="Title only", abstract="", url="")
        out = api._pico_assess(p, _pcc(), "m")
        assert isinstance(out, dict)
        assert out.get("overall_reasoning")


# ---------------------------------------------------------------------------
# The decision must always be derivable, whatever the model returned.
# ---------------------------------------------------------------------------

class TestEndToEndVerdict:
    def test_a_fail_on_the_discriminating_element_excludes(self, stub, monkeypatch):
        stub()
        monkeypatch.setattr(api, "_part_present", lambda *a, **k: "no")
        assessed = api._pico_assess(_paper(), _pcc(), "m")
        decision, reason = api._decide_from_votes(assessed, "pcc")
        assert decision == "EXCLUDE"
        assert reason

    def test_all_parts_present_includes(self, stub, monkeypatch):
        stub()
        monkeypatch.setattr(api, "_part_present", lambda *a, **k: "yes")
        assessed = api._pico_assess(_paper(), _pcc(), "m")
        assert api._decide_from_votes(assessed, "pcc")[0] == "INCLUDE"

    def test_a_broken_model_never_silently_excludes(self, stub):
        """Recall is the property worth protecting: a malfunctioning model may
        cost precision, but it must not drop papers on its own."""
        stub(RuntimeError("model down"))
        assessed = api._pico_assess(_paper(), _pcc(), "m")
        decision, _ = api._decide_from_votes(assessed, "pcc")
        assert decision in ("INCLUDE", "EXCLUDE")
        assert all((assessed.get(e) or {}).get("vote") != "FAIL"
                   for e in ("population", "concept", "context"))
