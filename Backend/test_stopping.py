"""Tests for the stop indicators (stopping.py). Run: pytest test_stopping.py"""
import math
import pytest

import stopping
from stopping import hypergeom_cdf, likely_complete, order_history, safe_to_stop, since_last_include, stop_indicators


class TestHypergeometric:
    def test_matches_a_hand_computed_value(self):
        # Population 10 with 3 successes, draw 4: P[X <= 0] = C(7,4)/C(10,4) = 35/210
        assert hypergeom_cdf(0, 10, 3, 4) == pytest.approx(35 / 210)

    def test_is_one_at_the_upper_support(self):
        assert hypergeom_cdf(3, 10, 3, 4) == 1.0

    def test_agrees_with_scipy_when_available(self):
        scipy = pytest.importorskip("scipy.stats")
        for (k, M, K, n) in [(2, 500, 30, 60), (0, 4878, 11, 900), (5, 48370, 763, 20000)]:
            assert hypergeom_cdf(k, M, K, n) == pytest.approx(float(scipy.hypergeom.cdf(k, M, K, n)), rel=1e-6, abs=1e-12)


class TestHistory:
    def test_unordered_labels_are_treated_as_screened_first(self):
        assert order_history({"a": 1, "b": 0, "c": 0}, ["b", "c"]) == [1, 0, 0]

    def test_history_order_wins_over_dict_order(self):
        assert order_history({"a": 1, "b": 0}, ["b", "a"]) == [0, 1]

    def test_since_last_include(self):
        assert since_last_include([1, 0, 0, 0]) == 3
        assert since_last_include([0, 0, 1]) == 0
        assert since_last_include([0, 0]) == 2
        assert since_last_include([]) == 0


class TestLikelyComplete:
    def test_needs_the_screened_floor(self):
        dec = [1] + [0] * 250                        # 251 screened of 5,000: below 20%
        assert likely_complete(dec, 5000)["likely_complete"] is False

    def test_needs_an_empty_tail_of_at_least_the_window(self):
        dec = [1] + [0] * 1100                       # 22% screened, 1,100 excludes after the include
        assert likely_complete(dec, 5000)["likely_complete"] is True
        dec = [0] * 1000 + [1] + [0] * 100           # only 100 since the last include
        assert likely_complete(dec, 5000)["likely_complete"] is False

    def test_window_never_below_200_records(self):
        assert likely_complete([1] + [0] * 199, 300)["window"] == 200

    def test_needs_at_least_one_include(self):
        assert likely_complete([0] * 1200, 5000)["likely_complete"] is False


class TestSafeToStop:
    def test_nothing_found_is_never_safe(self):
        assert safe_to_stop([0] * 400, 1000)["safe_to_stop"] is False

    def test_all_screened_is_safe(self):
        assert safe_to_stop([1, 0, 0], 3)["safe_to_stop"] is True

    def test_two_labels_are_never_safe(self):
        assert safe_to_stop([1, 0], 1000)["safe_to_stop"] is False

    def test_long_empty_tail_with_many_includes_becomes_safe(self):
        # 60 includes found early in a 4,000-record corpus, then nothing for 2,500 records:
        # under H0 at least 4 includes remain among 3,900, and 2,500 empty draws are then very unlikely.
        dec = [1] * 60 + [0] * 2500
        res = safe_to_stop(dec, 4000)
        assert res["safe_to_stop"] is True and res["p_value"] < 0.05

    def test_few_includes_need_almost_everything_screened(self):
        # 10 includes: H0 says one more remains. Even 90% screened leaves p = 0.10.
        dec = [1] * 10 + [0] * 8990
        assert safe_to_stop(dec, 10000)["safe_to_stop"] is False

    def test_p_value_decreases_as_the_empty_tail_grows(self):
        p = [safe_to_stop([1] * 20 + [0] * k, 5000)["p_value"] for k in (500, 1000, 2000, 3000)]
        assert p == sorted(p, reverse=True)


class TestIndicators:
    def test_detail_names_the_state(self):
        res = stop_indicators({"a": 1, "b": 0}, ["a", "b"], 1000)
        assert res["safe_to_stop"] is False and "start after" in res["stop_detail"]

    def test_likely_before_safe(self):
        labels = {f"r{i}": (1 if i < 3 else 0) for i in range(1500)}
        res = stop_indicators(labels, [f"r{i}" for i in range(1500)], 5000)
        assert res["likely_complete"] is True and res["safe_to_stop"] is False
