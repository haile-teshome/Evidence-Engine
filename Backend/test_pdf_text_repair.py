"""Tests for repairing a PDF text layer that mis-encodes its own symbols.

Run: Backend/.venv/bin/python -m pytest Backend/test_pdf_text_repair.py

This is the one class of extraction error that verification cannot catch. A
model reading "quetiapine (0, 5, or 10 mM)" extracts 10 mM correctly, and the
quote it cites is genuinely in the text, so checking the value against the
document confirms it. The paper says 10 µM. The error is a factor of a
thousand, and it arrives already verified.

So the repairs have to be right in both directions: fix a document that is
wrong, and leave alone a document that is not.
"""
import sys
import types

import pytest

import pdf_text_repair as R

MICRO = "µ"
FRACTION = "¼"       # what a broken font gives instead of "="
THORN = "þ"          # ... and instead of "+"


class TestOperators:
    def test_it_repairs_a_document_that_shows_the_fault(self):
        text = (f"hazard ratio{FRACTION}3.96, 95% CI{FRACTION}0.29 to 12.40; "
                f"P{FRACTION}.02, log-rank test")
        got = R.repair_operators(text)
        assert FRACTION not in got
        assert "hazard ratio=3.96" in got
        assert "P=.02" in got

    def test_it_leaves_a_genuine_fraction_alone(self):
        """The direction that matters more: a recipe is not a broken PDF."""
        text = f"Add 1{FRACTION} cups of flour and {FRACTION} teaspoon of salt."
        assert R.repair_operators(text) == text

    def test_one_occurrence_is_not_evidence_of_a_broken_font(self):
        # A single oddity could be anything; the fault shows up dozens of times.
        text = f"The dose was reduced by {FRACTION} in two patients."
        assert R.repair_operators(text) == text

    def test_it_repairs_the_plus_sign(self):
        text = f"4 Gy {THORN}QTP, or 4 Gy {THORN}QTP{THORN}ATR, or 4 Gy {THORN}ATR"
        got = R.repair_operators(text)
        assert THORN not in got
        assert "4 Gy +QTP" in got

    def test_it_survives_an_empty_or_clean_document(self):
        assert R.repair_operators("") == ""
        assert R.repair_operators("A perfectly ordinary sentence.") \
            == "A perfectly ordinary sentence."


class TestLigatures:
    def test_it_spells_out_the_letters(self):
        # Not cosmetic: a quote saying "confidence interval" cannot be found in
        # a document spelling it "con<fi>dence", so its value reads unverified.
        assert R.expand_ligatures("95% conﬁdence interval") == "95% confidence interval"
        assert R.expand_ligatures("inﬂammation") == "inflammation"
        assert R.expand_ligatures("diﬃcult") == "difficult"

    def test_it_needs_no_evidence(self):
        # Unlike the operators, the decomposition is unambiguous.
        assert R.expand_ligatures("one ﬁx") == "one fix"


class TestSymbolFixes:
    def test_it_applies_a_fix_where_the_token_appears(self):
        got = R.apply_symbol_fixes("treated with quetiapine (0, 5, or 10mM).",
                                   [("10mM", f"10{MICRO}M")])
        assert f"10{MICRO}M" in got

    def test_it_applies_a_fix_across_the_spaces_another_extractor_inserted(self):
        """The fix is found with one library and applied to text from another,
        and they disagree about spaces inside a token."""
        got = R.apply_symbol_fixes("quetiapine (0, 5, or 10 mM).",
                                   [("or10mM", f"or10{MICRO}M")])
        assert MICRO in got

    def test_a_token_that_is_not_there_changes_nothing(self):
        assert R.apply_symbol_fixes("no units here", [("10mM", f"10{MICRO}M")]) \
            == "no units here"


def _fake_pdfplumber(chars, monkeypatch):
    """Stand in for pdfplumber with one page of the given characters."""
    page = types.SimpleNamespace(chars=chars)
    pdf = types.SimpleNamespace(
        pages=[page], __enter__=lambda self: self, __exit__=lambda *a: False)

    class Ctx:
        def __enter__(self): return pdf
        def __exit__(self, *a): return False

    module = types.ModuleType("pdfplumber")
    module.open = lambda *a, **k: Ctx()
    monkeypatch.setitem(sys.modules, "pdfplumber", module)


def _char(text, x0, font="OGMCFC+AdvOTbf7bbdaa", top=100.0, width=4.0):
    return {"text": text, "x0": x0, "x1": x0 + width, "top": top,
            "width": width, "fontname": font}


class TestFontDrivenDetection:
    """Which characters came from a symbol font, and what they really are."""

    def _token(self, spelled, symbol_index, symbol_font="OGMDNO+AdvPi1"):
        chars = []
        for i, ch in enumerate(spelled):
            font = symbol_font if i == symbol_index else "OGMCFC+AdvOTbf7bbdaa"
            chars.append(_char(ch, 10.0 + i * 4.0, font=font))
        return chars

    def test_it_finds_a_micro_sign_drawn_as_an_m(self, monkeypatch):
        _fake_pdfplumber(self._token("10mM)", 2), monkeypatch)
        assert R.symbol_font_fixes(b"%PDF-fake") == [("10mM)", f"10{MICRO}M)")]

    def test_it_ignores_an_m_in_the_body_font(self, monkeypatch):
        # 10 mM really is millimolar when the text font drew it.
        _fake_pdfplumber(self._token("10mM)", symbol_index=-1), monkeypatch)
        assert R.symbol_font_fixes(b"%PDF-fake") == []

    def test_it_ignores_a_symbol_font_glyph_it_has_no_mapping_for(self, monkeypatch):
        # Guessing here would turn a readable value into a wrong one.
        _fake_pdfplumber(self._token("10zM)", 2), monkeypatch)
        assert R.symbol_font_fixes(b"%PDF-fake") == []

    def test_it_will_not_return_a_token_too_short_to_place(self, monkeypatch):
        # A bare "m" appears everywhere; replacing it would corrupt the text.
        _fake_pdfplumber([_char("m", 10.0, font="OGMDNO+AdvPi1")], monkeypatch)
        assert R.symbol_font_fixes(b"%PDF-fake") == []

    def test_a_token_stops_at_a_gap(self, monkeypatch):
        chars = self._token("10mM)", 2)
        chars += [_char("w", 400.0), _char("i", 404.0), _char("de", 408.0)]
        _fake_pdfplumber(chars, monkeypatch)
        spelled, _ = R.symbol_font_fixes(b"%PDF-fake")[0]
        assert spelled == "10mM)", "the token ran past a space"

    def test_no_bytes_means_no_inspection(self):
        assert R.symbol_font_fixes(b"") == []


class TestRepair:
    def test_it_does_both_jobs(self, monkeypatch):
        _fake_pdfplumber(
            [_char(c, 10.0 + i * 4.0,
                   font="OGMDNO+AdvPi1" if i == 2 else "OGMCFC+AdvOTbf7bbdaa")
             for i, c in enumerate("10mM)")], monkeypatch)
        text = (f"quetiapine (0, 5, or 10mM). The hazard ratio{FRACTION}3.96, "
                f"95% conﬁdence interval{FRACTION}0.29 to 12.40; P{FRACTION}.02.")
        got = R.repair(text, b"%PDF-fake")
        assert f"10{MICRO}M" in got
        assert FRACTION not in got
        assert "confidence interval=0.29" in got

    def test_it_leaves_a_sound_document_untouched(self):
        text = "The hazard ratio was 3.96 (95% CI 0.29 to 12.40; P = .02)."
        assert R.repair(text, None) == text

    @pytest.mark.parametrize("empty", [None, ""])
    def test_it_survives_nothing(self, empty):
        assert R.repair(empty, b"%PDF-fake") == empty
