"""Tests for the meta-analysis engine and its provenance chain.

Run: pytest test_meta_analysis.py

A meta-analysis is a claim about a body of evidence, so the tests here care
about two things above all: that the arithmetic is right, and that every pooled
number can be traced back to the sentence in the paper it came from.

The arithmetic is checkable against published worked examples, so it is checked
that way rather than against whatever the code happens to return.

The provenance half exists because of a specific failure: `MD` demanded per-arm
means, SDs and Ns, so a paper reporting "MD -0.9 (95% CI -1.2 to -0.6)" and
nothing else could not be pooled at all. That is how most trials report, so the
reported-effect-plus-CI path is now the common case and is tested as such.
"""
import math

import pytest

import meta_analysis as ma
from meta_analysis import StudyEffect, compute_effect_size, pool, answer_grounded


def eff(measure, **kw):
    return compute_effect_size(StudyEffect(paper_id="p", title="p", effect_measure=measure, **kw))


class StubModel:
    """Stands in for the LLM. Returns whatever JSON the test hands it."""
    def __init__(self, reply): self.reply = reply
    def invoke(self, _prompt): return type("R", (), {"content": self.reply})()


@pytest.fixture
def stub(monkeypatch):
    def _install(reply):
        monkeypatch.setattr(ma.AIService, "get_model", staticmethod(lambda *a, **k: StubModel(reply)))
    return _install


# ---------------------------------------------------------------------------
# Effect sizes from raw inputs, against worked examples.
# ---------------------------------------------------------------------------

class TestEffectsFromRawInputs:
    def test_odds_ratio_matches_the_hand_calculation(self):
        # 2x2: 15/100 vs 10/100 -> OR = (15*90)/(10*85)
        s = eff("OR", events_t=15, n_t=100, events_c=10, n_c=100)
        assert math.exp(s.yi) == pytest.approx((15 * 90) / (10 * 85), rel=1e-6)

    def test_risk_ratio_matches_the_hand_calculation(self):
        s = eff("RR", events_t=15, n_t=100, events_c=10, n_c=100)
        assert math.exp(s.yi) == pytest.approx(0.15 / 0.10, rel=1e-6)

    def test_risk_difference_is_a_plain_difference(self):
        s = eff("RD", events_t=15, n_t=100, events_c=10, n_c=100)
        assert s.yi == pytest.approx(0.05, rel=1e-9)

    def test_mean_difference_is_a_plain_difference(self):
        s = eff("MD", mean_t=6.8, sd_t=0.9, n_t=1204, mean_c=7.7, sd_c=1.0, n_c=1198)
        assert s.yi == pytest.approx(-0.9, rel=1e-9)

    def test_zero_cells_do_not_produce_infinity(self):
        """A trial with no events in one arm must still pool, via the continuity
        correction, rather than returning an infinite effect."""
        s = eff("OR", events_t=0, n_t=50, events_c=5, n_c=50)
        assert s.yi is not None and math.isfinite(s.yi)

    def test_incomplete_raw_inputs_are_an_error_not_a_guess(self):
        s = eff("OR", events_t=15, n_t=100)
        assert s.yi is None and s.error


# ---------------------------------------------------------------------------
# The reported effect + CI path. The reason most papers are poolable at all.
# ---------------------------------------------------------------------------

class TestReportedEffectWithCI:
    def test_mean_difference_round_trips_exactly(self):
        """The mockup's card: MD -0.9 (95% CI -1.2 to -0.6), no per-arm SDs."""
        s = eff("MD", effect=-0.9, effect_ci_low=-1.2, effect_ci_high=-0.6)
        assert s.yi == pytest.approx(-0.9)
        assert s.ci_low == pytest.approx(-1.2, abs=1e-6)
        assert s.ci_high == pytest.approx(-0.6, abs=1e-6)

    def test_odds_ratio_round_trips_on_the_log_scale(self):
        """The point estimate is exact; the bounds come back within about 1%.

        They are not exact by design. A published interval like 0.60-0.92 is
        rounded, so it is not quite symmetric around log(0.74), and we keep the
        authors' point estimate rather than recentring it on their interval.
        Reconstructing the bounds from a single pooled SE therefore shifts them
        slightly, which is the correct trade: the point estimate is what the
        trial reported, the SE is only our inference about its precision."""
        s = eff("OR", effect=0.74, effect_ci_low=0.60, effect_ci_high=0.92)
        assert math.exp(s.yi) == pytest.approx(0.74)
        assert math.exp(s.ci_low) == pytest.approx(0.60, rel=0.01)
        assert math.exp(s.ci_high) == pytest.approx(0.92, rel=0.01)

    @pytest.mark.parametrize("measure", ["OR", "RR", "HR", "PETO_OR", "IR"])
    def test_ratio_measures_are_pooled_on_the_log_scale(self, measure):
        s = eff(measure, effect=0.5, effect_ci_low=0.25, effect_ci_high=1.0)
        assert s.yi == pytest.approx(math.log(0.5))

    @pytest.mark.parametrize("measure", ["MD", "RD", "SMD"])
    def test_difference_measures_are_pooled_on_the_linear_scale(self, measure):
        s = eff(measure, effect=0.5, effect_ci_low=0.25, effect_ci_high=0.75)
        assert s.yi == pytest.approx(0.5)

    def test_raw_inputs_win_over_a_reported_ci(self):
        """Per-arm data is more precise than a rounded published interval, so it
        must not be discarded when both are present."""
        s = eff("MD", mean_t=6.8, sd_t=0.9, n_t=1204, mean_c=7.7, sd_c=1.0, n_c=1198,
                effect=-0.9, effect_ci_low=-1.2, effect_ci_high=-0.6)
        assert s.se < 0.15          # from n=2402, far tighter than the reported CI

    def test_the_legacy_hr_fields_still_work(self):
        s = eff("HR", hr=0.74, hr_ci_low=0.60, hr_ci_high=0.92)
        assert math.exp(s.yi) == pytest.approx(0.74)

    def test_a_reversed_interval_is_tolerated(self):
        """Low and high transposed is a transcription slip, not a reason to drop
        a study; the interval is the same either way."""
        s = eff("MD", effect=-0.9, effect_ci_low=-0.6, effect_ci_high=-1.2)
        assert s.yi == pytest.approx(-0.9)
        assert s.ci_low == pytest.approx(-1.2, abs=1e-6)

    def test_a_negative_ratio_is_rejected(self):
        """log() of a negative number is not a number. Better no study than a
        NaN weight silently poisoning the pooled estimate."""
        s = eff("OR", effect=-0.5, effect_ci_low=-0.9, effect_ci_high=-0.1)
        assert s.yi is None and s.error

    def test_a_zero_width_interval_is_rejected(self):
        """SE of zero is infinite weight: one study would determine the result."""
        s = eff("MD", effect=0.5, effect_ci_low=0.5, effect_ci_high=0.5)
        assert s.yi is None and s.error

    def test_proportions_and_correlations_are_not_guessed_at(self):
        """Their scales need a logit / Fisher z inversion that a rounded CI does
        not support precisely enough, so they are deliberately not handled."""
        for measure in ("PROP", "ZCOR"):
            assert eff(measure, effect=0.3, effect_ci_low=0.2, effect_ci_high=0.4).yi is None


# ---------------------------------------------------------------------------
# Pooling.
# ---------------------------------------------------------------------------

def _studies(*triples):
    return [compute_effect_size(StudyEffect(paper_id=t, title=t, effect_measure="HR",
                                            effect=v, effect_ci_low=lo, effect_ci_high=hi))
            for t, v, lo, hi in triples]


class TestPooling:
    MOCKUP = (("Hong 2013", 0.72, 0.51, 1.02), ("UKPDS 34", 0.61, 0.42, 0.89),
              ("Kooy 2009", 0.76, 0.47, 1.24), ("SPREAD-DIMCAD", 0.54, 0.34, 0.86),
              ("HOME trial", 0.83, 0.58, 1.19), ("Roussel 2010", 0.69, 0.55, 0.87))

    def test_pools_every_valid_study(self):
        assert pool(_studies(*self.MOCKUP))["k"] == 6

    def test_pooled_estimate_lies_inside_the_range_of_the_studies(self):
        r = pool(_studies(*self.MOCKUP))
        hr = math.exp(r["random"]["estimate"])
        assert 0.54 <= hr <= 0.83

    def test_pooled_interval_is_tighter_than_any_single_study(self):
        """Pooling buys precision; if it does not, the weighting is wrong."""
        r = pool(_studies(*self.MOCKUP))
        width = r["random"]["ci_high"] - r["random"]["ci_low"]
        assert width < min(s.ci_high - s.ci_low for s in _studies(*self.MOCKUP))

    def test_identical_studies_pool_to_themselves_with_no_heterogeneity(self):
        same = _studies(*[(f"s{i}", 0.70, 0.55, 0.89) for i in range(4)])
        r = pool(same)
        assert math.exp(r["random"]["estimate"]) == pytest.approx(0.70, abs=1e-6)
        assert r["heterogeneity"]["I2_pct"] == pytest.approx(0.0, abs=1e-6)

    def test_a_precise_study_outweighs_a_vague_one(self):
        r = pool(_studies(("tight", 0.60, 0.55, 0.65), ("loose", 1.40, 0.40, 4.90)))
        assert math.exp(r["random"]["estimate"]) < 0.8

    def test_heterogeneous_studies_report_heterogeneity(self):
        r = pool(_studies(("a", 0.30, 0.22, 0.41), ("b", 2.90, 2.10, 4.00)))
        assert r["heterogeneity"]["I2_pct"] > 75

    def test_unpoolable_studies_are_reported_not_silently_dropped(self):
        rows = _studies(("good", 0.70, 0.55, 0.89))
        rows.append(StudyEffect(paper_id="bad", title="bad", effect_measure="MD"))
        r = pool(rows)
        assert r["k"] == 1 and len(r["invalid_studies"]) == 1

    def test_no_valid_studies_returns_k_zero_rather_than_raising(self):
        r = pool([StudyEffect(paper_id="x", title="x", effect_measure="MD")])
        assert r["k"] == 0 and r["random"] is None

    def test_random_effects_interval_is_wider_than_fixed_when_heterogeneous(self):
        r = pool(_studies(("a", 0.30, 0.22, 0.41), ("b", 2.90, 2.10, 4.00)))
        assert (r["random"]["ci_high"] - r["random"]["ci_low"]) > \
               (r["fixed"]["ci_high"] - r["fixed"]["ci_low"])

    @pytest.mark.parametrize("method", ["DL", "PM", "REML"])
    def test_every_tau2_estimator_runs(self, method):
        r = pool(_studies(*self.MOCKUP), tau2_method=method)
        assert r["k"] == 6 and r["random"] is not None


# ---------------------------------------------------------------------------
# Grounded question answering.
# ---------------------------------------------------------------------------

PAPER = {"id": "p1", "title": "Metformin trial", "abstract": "RCT of metformin versus placebo."}
TABLES = [{"title": "Table 2", "page": 7, "tier": "pdfplumber",
           "data": [["Outcome", "HR (95% CI)"], ["MACE", "0.74 (0.60-0.92)"]]}]
FOUND = ('{"found":true,"answer":"HR 0.74","value":0.74,"units":null,'
         '"ci_low":0.60,"ci_high":0.92,"timepoint":"12 months",'
         '"source_page":null,"source_table":"Table 2",'
         '"quote":"MACE | 0.74 (0.60-0.92)","confidence":0.9}')


class TestGroundedAnswers:
    def test_returns_the_value_and_its_interval(self, stub):
        stub(FOUND)
        r = answer_grounded(PAPER, "HR for MACE?", "m", tables=TABLES)
        assert r["value"] == 0.74 and r["ci_low"] == 0.60 and r["ci_high"] == 0.92

    def test_page_comes_from_our_table_not_the_model(self, stub):
        """The model returned source_page null. pdfplumber already knows Table 2
        is on page 7, and a deterministic citation beats a reconstructed one."""
        stub(FOUND)
        assert answer_grounded(PAPER, "HR?", "m", tables=TABLES)["source_page"] == 7

    def test_page_backfill_does_not_invent_one_for_an_unknown_table(self, stub):
        stub(FOUND.replace('"Table 2"', '"Table 9"'))
        assert answer_grounded(PAPER, "HR?", "m", tables=TABLES)["source_page"] is None

    def test_a_quote_present_in_the_source_is_verified(self, stub):
        stub(FOUND)
        assert answer_grounded(PAPER, "HR?", "m", tables=TABLES)["quote_verified"] is True

    def test_a_quote_absent_from_the_source_is_flagged(self, stub):
        """The failure that makes this feature worse than useless is a fabricated
        quote, because it looks checked when it is not."""
        stub(FOUND.replace("MACE | 0.74 (0.60-0.92)", "A sentence never written."))
        assert answer_grounded(PAPER, "HR?", "m", tables=TABLES)["quote_verified"] is False

    def test_a_quote_differing_only_in_line_wrapping_still_verifies(self, stub):
        stub(FOUND.replace("MACE | 0.74 (0.60-0.92)", "MACE  |   0.74 (0.60-0.92)"))
        assert answer_grounded(PAPER, "HR?", "m", tables=TABLES)["quote_verified"] is True

    def test_not_found_says_so_rather_than_returning_an_empty_card(self, stub):
        stub('{"found":false,"answer":null,"value":null}')
        r = answer_grounded(PAPER, "Effect on mortality?", "m", tables=TABLES)
        assert r["found"] is False and r["answer"]

    def test_an_absurd_page_number_is_dropped(self, stub):
        stub(FOUND.replace('"source_page":null', '"source_page":99999')
                  .replace('"Table 2"', '"Table 9"'))
        assert answer_grounded(PAPER, "HR?", "m", tables=TABLES)["source_page"] is None

    def test_an_empty_question_is_refused_without_calling_the_model(self, stub):
        stub(FOUND)
        assert answer_grounded(PAPER, "   ", "m", tables=TABLES)["found"] is False

    def test_a_paper_with_no_text_is_refused(self, stub):
        stub(FOUND)
        r = answer_grounded({"id": "x", "title": "x"}, "HR?", "m")
        assert r["found"] is False and "error" in r

    def test_unparseable_model_output_is_an_error_not_a_crash(self, stub):
        stub("I'm afraid I can't do that.")
        assert answer_grounded(PAPER, "HR?", "m", tables=TABLES)["found"] is False

    def test_an_unavailable_model_is_reported(self, monkeypatch):
        monkeypatch.setattr(ma.AIService, "get_model", staticmethod(lambda *a, **k: None))
        assert "error" in answer_grounded(PAPER, "HR?", "m", tables=TABLES)
