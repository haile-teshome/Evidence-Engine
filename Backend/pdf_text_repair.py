"""Repairing a PDF text layer that mis-encodes its own symbols.

Some publishers' PDFs carry fonts whose character map is wrong, so the text
layer spells symbols as letters. Two kinds show up, and both change the meaning
of a number:

  - An operator drawn from the body font with a broken map. In one JNCI paper
    every "=" reads as "¼": "hazard ratio ¼ 3.96, 95% CI ¼ 0.29 to 12.40".
    45 of them, plus 3 "þ" where "+" belongs.
  - A symbol drawn from a separate "Pi" font, which has no map at all, so its
    glyph falls back to whatever the base encoding gives. In the same paper
    "quetiapine (0, 5, or 10 µM)" reads as "10mM": micromolar rendered as
    millimolar, a thousand-fold error that looks like a plausible dose.

Neither is recoverable downstream. A model reading "10mM" extracts 10 mM,
correctly, from a document that says the wrong thing, and no amount of
verification against that text will catch it: the quote matches.

The two are repaired differently because the evidence differs. A Pi-font glyph
is identified by its font, which is certain, so it is repaired exactly where the
font says to. A broken operator in the body font has no such marker, so it is
repaired only in a document that demonstrates the fault, in the shape no
genuine character takes.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# Fonts that hold symbols rather than text. Names are substring-matched because
# a PDF prefixes its subset tag: "OGMDNO+AdvPi1".
PI_FONTS = ("advpi", "mathematicalpi", "symbol", "universalstd", "esstixsix")

# What a Pi-font glyph means when it decodes to a plain character. Only entries
# evidenced in real documents are listed: guessing here would turn a readable
# value into a wrong one, which is worse than leaving it alone.
PI_GLYPH: Dict[str, str] = {
    "m": "µ",     # micro. AdvPi1 draws µ and the fallback map gives "m".
}

# Operators the body font spells as something else. The replacement is applied
# only to a document that shows the fault (see OPERATOR_EVIDENCE).
OPERATOR_GLYPH: Dict[str, str] = {
    "¼": "=",     # ¼ for "="
    "þ": "+",     # þ for "+"
}

# What the fault looks like: the character pressed against a number, which is
# how an equals sign is typeset and how a fraction never is. "1¼ cups" has a
# space after it; "ratio¼3.96" and "P¼.02" do not.
OPERATOR_EVIDENCE = {
    "¼": re.compile(r"[A-Za-z0-9\])]\s?¼\s?[-+.\d]"),
    # Spaces around it vary by extractor: pypdf gives "4 Gy þQTP"
    # where pdfplumber gives "4GyþQTP".
    "þ": re.compile(r"[A-Za-z0-9]\s?þ\s?[A-Za-z0-9]"),
}
# How many times the shape has to appear before the document is believed to be
# faulty. Three is past coincidence and far below the 45 in the paper measured.
OPERATOR_MIN_EVIDENCE = 3


# Typographic ligatures. Unlike the glyphs above these are not a fault, but
# they break every string comparison made against the text: a quote saying
# "confidence interval" cannot be found in a document that spells it
# "conﬁdence", so the value it supports shows as unverified and a reviewer
# is sent to check a sentence that is really there. The decomposition is
# unambiguous, so this needs no evidence to apply.
LIGATURES: Dict[str, str] = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
    "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
}


def expand_ligatures(text: str) -> str:
    """Spell ligatures as the letters they stand for."""
    for ligature, letters in LIGATURES.items():
        if ligature in text:
            text = text.replace(ligature, letters)
    return text


def repair_operators(text: str) -> str:
    """Fix operators the body font mis-encodes, in a document that shows it."""
    if not text:
        return text
    for glyph, real in OPERATOR_GLYPH.items():
        if glyph not in text:
            continue
        hits = OPERATOR_EVIDENCE[glyph].findall(text)
        if len(hits) >= OPERATOR_MIN_EVIDENCE:
            text = text.replace(glyph, real)
    return text


def _token_around(chars: List[dict], index: int) -> Tuple[int, int]:
    """The run of adjacent characters the character at `index` belongs to.

    A "token" here is what a reader would call a word: the characters touching
    each other, which is how the mis-encoded glyph is found again in text
    assembled by a different library.
    """
    def touching(left: dict, right: dict) -> bool:
        gap = (right.get("x0") or 0) - (left.get("x1") or 0)
        width = max(right.get("width") or 0, left.get("width") or 0, 1)
        return abs((left.get("top") or 0) - (right.get("top") or 0)) < 2 and gap < width * 0.6

    start = index
    while start > 0 and touching(chars[start - 1], chars[start]):
        start -= 1
    end = index
    while end + 1 < len(chars) and touching(chars[end], chars[end + 1]):
        end += 1
    return start, end


def symbol_font_fixes(pdf_bytes: bytes) -> List[Tuple[str, str]]:
    """(token as the text layer spells it, token as it should read) pairs.

    Driven entirely by the font: a character from a Pi font that decoded to a
    plain letter is a symbol, and which symbol is known. Returns the containing
    token so the fix can be applied to text produced by any extractor, rather
    than by a pattern that might match a number that was never wrong.
    """
    if not pdf_bytes:
        return []
    try:
        import pdfplumber
        from io import BytesIO
    except Exception:
        return []

    fixes: List[Tuple[str, str]] = []
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                chars = page.chars
                for i, char in enumerate(chars):
                    font = (char.get("fontname") or "").lower()
                    if not any(marker in font for marker in PI_FONTS):
                        continue
                    real = PI_GLYPH.get(char.get("text") or "")
                    if not real:
                        continue
                    start, end = _token_around(chars, i)
                    spelled = "".join(c.get("text") or "" for c in chars[start:end + 1])
                    if len(spelled) < 3:
                        # Too short to find again without matching the wrong
                        # thing: a bare "m" appears everywhere.
                        continue
                    corrected = (spelled[:i - start] + real + spelled[i - start + 1:])
                    if spelled != corrected:
                        fixes.append((spelled, corrected))
    except Exception as e:  # noqa: BLE001
        print(f"[pdf_text_repair] font inspection failed: {e}")
        return []
    # Deduplicated, longest first: a token that contains another must be
    # replaced before it, or the shorter replacement breaks the longer match.
    return sorted(set(fixes), key=lambda pair: -len(pair[0]))


def apply_symbol_fixes(text: str, fixes: List[Tuple[str, str]]) -> str:
    """Apply the font-driven fixes wherever their token appears in `text`."""
    for spelled, corrected in fixes:
        if spelled in text:
            text = text.replace(spelled, corrected)
            continue
        # Extractors disagree about spaces inside a token, so try again with
        # the spaces the other library may have inserted.
        loose = re.compile(r"\s*".join(re.escape(ch) for ch in spelled))
        text = loose.sub(lambda _m, c=corrected: c, text)
    return text


def repair(text: Optional[str], pdf_bytes: Optional[bytes] = None) -> Optional[str]:
    """The text as the document means it, as far as that can be established."""
    if not text:
        return text
    if pdf_bytes:
        text = apply_symbol_fixes(text, symbol_font_fixes(pdf_bytes))
    return expand_ligatures(repair_operators(text))
