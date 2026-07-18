"""Tests for the deterministic instrument roll-up algorithms.

Run: python -m pytest Backend/test_instruments.py   (or: python Backend/test_instruments.py)

These lock the signalling-answer → domain-judgment → overall logic against the
published tools, so the roll-up can be refactored without silently changing a
judgment. The domain roll-up is polarity-aware: each signalling question carries
a `concern` direction ("no" = No/Probably-no flags a problem; "yes" = Yes flags
a problem), so these tests pin both the polarity handling and the algorithms.
"""
from instruments import (
    rob2_domain, rob2_overall, robins_domain, robins_overall,
    lhu_domain, lhu_overall, grade_certainty, roll_up, instrument_for_design,
    INSTRUMENTS, _sig,
)


def dom(instrument_id, domain_id):
    """Fetch a real domain definition from an instrument."""
    for d in INSTRUMENTS[instrument_id]["domains"]:
        if d["id"] == domain_id:
            return d
    raise KeyError(domain_id)


# ---- RoB 2 domains ---------------------------------------------------------
def test_rob2_d1_low_when_randomized_concealed_no_imbalance():
    assert rob2_domain(dom("rob2", "d1"), {"1.1": "Y", "1.2": "Y", "1.3": "N"}) == "Low"

def test_rob2_d1_high_on_baseline_imbalance():
    assert rob2_domain(dom("rob2", "d1"), {"1.1": "Y", "1.2": "Y", "1.3": "Y"}) == "High"

def test_rob2_d1_some_concerns_when_not_concealed():
    assert rob2_domain(dom("rob2", "d1"), {"1.1": "Y", "1.2": "N", "1.3": "N"}) == "Some concerns"

def test_rob2_d5_high_on_selective_reporting():
    assert rob2_domain(dom("rob2", "d5"), {"5.1": "N", "5.2": "PY", "5.3": "PY"}) == "High"

def test_rob2_d5_low_when_prespecified():
    # Pre-specified plan (5.1=Y) and result NOT cherry-picked (5.2/5.3=No).
    assert rob2_domain(dom("rob2", "d5"), {"5.1": "Y", "5.2": "N", "5.3": "N"}) == "Low"


# ---- RoB 2 overall ---------------------------------------------------------
def test_rob2_overall_low_all_low():
    assert rob2_overall(["Low", "Low", "Low", "Low", "Low"]) == "Low"

def test_rob2_overall_high_if_any_high():
    assert rob2_overall(["Low", "Low", "High", "Low", "Low"]) == "High"

def test_rob2_overall_some_concerns():
    assert rob2_overall(["Low", "Some concerns", "Low", "Low", "Low"]) == "Some concerns"

def test_rob2_overall_multiple_concerns_becomes_high():
    assert rob2_overall(["Some concerns", "Some concerns", "Some concerns", "Low", "Low"]) == "High"


# ---- polarity-aware ROBINS domain -----------------------------------------
# Build synthetic domains so we exercise concern="no", concern="yes", and crit.
def _robins_dom(signals):
    return {"id": "dx", "name": "x", "signals": signals}

def test_robins_domain_no_information_when_all_ni():
    d = _robins_dom([_sig("a", "controlled well?", "no"), _sig("b", "measured well?", "no")])
    assert robins_domain(d, {"a": "NI", "b": "NI"}) == "No information"

def test_robins_domain_critical_signal():
    d = _robins_dom([_sig("a", "controlled for all confounders?", "no", crit=True), _sig("b", "x", "no")])
    assert robins_domain(d, {"a": "N", "b": "Y"}) == "Critical"

def test_robins_domain_moderate_one_concern():
    d = _robins_dom([_sig("a", "done well?", "no"), _sig("b", "done well?", "no")])
    assert robins_domain(d, {"a": "Y", "b": "N"}) == "Moderate"

def test_robins_domain_serious_two_concerns():
    d = _robins_dom([_sig("a", "done well?", "no"), _sig("b", "done well?", "no"), _sig("c", "done well?", "no")])
    assert robins_domain(d, {"a": "N", "b": "N", "c": "Y"}) == "Serious"

def test_robins_domain_reverse_polarity_yes_is_concern():
    # concern="yes": answering Yes is the problem (e.g. "could selection bias the result?").
    d = _robins_dom([_sig("a", "could it bias?", "yes")])
    assert robins_domain(d, {"a": "Y"}) == "Moderate"
    assert robins_domain(d, {"a": "N"}) == "Low"


# ---- ROBINS-I overall (worst domain) --------------------------------------
def test_robins_overall_critical_dominates():
    assert robins_overall(["Low", "Serious", "Critical", "Moderate"]) == "Critical"

def test_robins_overall_serious():
    assert robins_overall(["Low", "Moderate", "Serious", "Low"]) == "Serious"

def test_robins_overall_low_all_low():
    assert robins_overall(["Low", "Low", "Low"]) == "Low"


# ---- QUADAS-2 / PROBAST / JBI (Low/High/Unclear) --------------------------
def _lhu_dom(signals):
    return {"id": "dx", "name": "x", "signals": signals}

def test_lhu_low_all_yes():
    d = _lhu_dom([_sig("a", "done well?", "no"), _sig("b", "done well?", "no")])
    assert lhu_domain(d, {"a": "Y", "b": "PY"}) == "Low"

def test_lhu_high_any_no():
    d = _lhu_dom([_sig("a", "done well?", "no"), _sig("b", "done well?", "no")])
    assert lhu_domain(d, {"a": "Y", "b": "N"}) == "High"

def test_lhu_unclear_on_ni():
    d = _lhu_dom([_sig("a", "done well?", "no"), _sig("b", "done well?", "no")])
    assert lhu_domain(d, {"a": "Y", "b": "NI"}) == "Unclear"

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
    assert instrument_for_design("systematic review with meta-analysis") == "amstar_2"
    assert instrument_for_design("analytical cross-sectional survey") == "jbi_xsectional"
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

def test_roll_up_engine_end_to_end_robins_i():
    # A clean cohort: all "done well" signals Yes, all "could it bias" signals No.
    answers = {}
    for d in INSTRUMENTS["robins_i"]["domains"]:
        answers[d["id"]] = {s["id"]: ("N" if s.get("concern") == "yes" else "Y") for s in d["signals"]}
    res = roll_up("robins_i", answers)
    assert len(res["domains"]) == 7
    assert res["overall"] == "Low"

def test_every_nonscaffold_instrument_rolls_up():
    # Smoke test: each fully-defined instrument produces a valid overall from all-NI.
    for iid, inst in INSTRUMENTS.items():
        if inst.get("scaffold"):
            continue
        answers = {d["id"]: {} for d in inst["domains"]}
        res = roll_up(iid, answers)
        assert res["overall"] in inst["scale"], f"{iid} overall {res['overall']} not in scale"


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
