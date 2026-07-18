"""Tests for the deterministic instrument roll-up algorithms.

Run: python -m pytest Backend/test_instruments.py   (or: python Backend/test_instruments.py)

These lock the signalling-answer → domain-judgment → overall logic against the
published tools, so the roll-up can be refactored without silently changing a
judgment.
"""
from instruments import (
    rob2_domain, rob2_overall, robins_domain, robins_overall,
    lhu_domain, lhu_overall, grade_certainty, roll_up, instrument_for_design,
    INSTRUMENTS,
)


# ---- RoB 2 domains ---------------------------------------------------------
def test_rob2_d1_low_when_randomized_concealed_no_imbalance():
    assert rob2_domain("d1", {"1.1": "Y", "1.2": "Y", "1.3": "N"}) == "Low"

def test_rob2_d1_high_on_baseline_imbalance():
    assert rob2_domain("d1", {"1.1": "Y", "1.2": "Y", "1.3": "Y"}) == "High"

def test_rob2_d1_some_concerns_when_not_concealed():
    assert rob2_domain("d1", {"1.1": "Y", "1.2": "N", "1.3": "N"}) == "Some concerns"

def test_rob2_d5_high_on_selective_reporting():
    assert rob2_domain("d5", {"5.1": "N", "5.2": "PY", "5.3": "PY"}) == "High"

def test_rob2_d5_low_when_prespecified():
    # Pre-specified plan (5.1=Y) and result NOT cherry-picked (5.2/5.3=No).
    assert rob2_domain("d5", {"5.1": "Y", "5.2": "N", "5.3": "N"}) == "Low"


# ---- RoB 2 overall ---------------------------------------------------------
def test_rob2_overall_low_all_low():
    assert rob2_overall(["Low", "Low", "Low", "Low", "Low"]) == "Low"

def test_rob2_overall_high_if_any_high():
    assert rob2_overall(["Low", "Low", "High", "Low", "Low"]) == "High"

def test_rob2_overall_some_concerns():
    assert rob2_overall(["Low", "Some concerns", "Low", "Low", "Low"]) == "Some concerns"

def test_rob2_overall_multiple_concerns_becomes_high():
    assert rob2_overall(["Some concerns", "Some concerns", "Some concerns", "Low", "Low"]) == "High"


# ---- ROBINS-I overall (worst domain) --------------------------------------
def test_robins_overall_critical_dominates():
    assert robins_overall(["Low", "Serious", "Critical", "Moderate"]) == "Critical"

def test_robins_overall_serious():
    assert robins_overall(["Low", "Moderate", "Serious", "Low"]) == "Serious"

def test_robins_overall_low_all_low():
    assert robins_overall(["Low", "Low", "Low"]) == "Low"

def test_robins_domain_no_information_when_all_ni():
    assert robins_domain("d1", {"1.1c": "NI", "1.2": "NI"}) == "No information"

def test_robins_domain_critical_signal():
    assert robins_domain("d1", {"1.1c": "N", "1.2": "Y"}) == "Critical"


# ---- QUADAS-2 / PROBAST (Low/High/Unclear) --------------------------------
def test_lhu_low_all_yes():
    assert lhu_domain("d1", {"a": "Y", "b": "PY"}) == "Low"

def test_lhu_high_any_no():
    assert lhu_domain("d1", {"a": "Y", "b": "N"}) == "High"

def test_lhu_unclear_on_ni():
    assert lhu_domain("d1", {"a": "Y", "b": "NI"}) == "Unclear"

def test_lhu_overall_high_beats_unclear():
    assert lhu_overall(["Low", "Unclear", "High"]) == "High"

def test_lhu_overall_unclear():
    assert lhu_overall(["Low", "Unclear", "Low"]) == "Unclear"


# ---- GRADE certainty -------------------------------------------------------
def test_grade_rct_starts_high_no_downgrade():
    assert grade_certainty("randomized", {}, {})["certainty"] == "High"

def test_grade_rct_two_serious_downgrades():
    r = grade_certainty("randomized", {"risk_of_bias": -1, "imprecision": -1}, {})
    assert r["certainty"] == "Low"

def test_grade_rct_downgrade_floor_is_very_low():
    r = grade_certainty("randomized", {"risk_of_bias": -2, "inconsistency": -2, "imprecision": -2}, {})
    assert r["certainty"] == "Very low"

def test_grade_observational_starts_low():
    assert grade_certainty("observational", {}, {})["certainty"] == "Low"

def test_grade_observational_upgrade_large_effect():
    r = grade_certainty("observational", {}, {"large_effect": 1})
    assert r["certainty"] == "Moderate"

def test_grade_upgrades_ignored_for_rct():
    r = grade_certainty("randomized", {}, {"large_effect": 2})
    assert r["certainty"] == "High"   # already capped at High; upgrades not applied


# ---- routing + engine ------------------------------------------------------
def test_routing_by_design():
    assert instrument_for_design("Randomized controlled trial") == "rob2"
    assert instrument_for_design("prospective cohort study") == "robins_i"
    assert instrument_for_design("diagnostic test accuracy study") == "quadas_2"
    assert instrument_for_design("clinical prediction model (machine learning)") == "probast_ai"
    assert instrument_for_design("something unusual") == "jbi_xsectional"

def test_roll_up_engine_end_to_end_rob2():
    answers = {
        "d1": {"1.1": "Y", "1.2": "Y", "1.3": "N"},
        "d2": {"2.1": "N", "2.2": "N", "2.6": "Y", "2.7": "N"},
        "d3": {"3.1": "Y"},
        "d4": {"4.1": "N", "4.2": "N", "4.3": "N", "4.4": "N", "4.5": "N"},
        "d5": {"5.1": "Y", "5.2": "N", "5.3": "N"},
    }
    res = roll_up("rob2", answers)
    assert len(res["domains"]) == 5
    assert res["overall"] in INSTRUMENTS["rob2"]["scale"]


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
