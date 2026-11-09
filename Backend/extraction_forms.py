"""AI-assisted data extraction into user-defined extraction forms.

A form is a list of fields, each with a name, a type (text/number/categorical/
date/boolean), and an optional instruction. Given a study's text (plus any
extracted tables), the model fills each field AND quotes the exact source snippet
that supports it, so a human can verify quickly.

Design decisions grounded in our extraction benchmarking:
- Autonomous numeric exact-match is low (~0.33 even for purpose-built models);
  the accuracy comes from human verification. So NUMBER fields are flagged
  needs_review by default and every value carries a source_quote for one-glance
  checking (locate-then-extract).
- Text fields are where the LLM is strong (soft-match ~0.7-0.9); they get a
  lighter-touch review.
- Numbers are validated (parseable, and checked for presence in the source) and
  never returned as percentages.

This module is self-contained: it reuses AIService.get_model / _extract_json and
langchain's HumanMessage, and adds no state.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_core.messages import HumanMessage
except Exception:  # pragma: no cover - fall back to the langchain path used elsewhere
    from langchain.schema import HumanMessage

NUMERIC_TYPES = {"number", "numeric", "integer", "float"}

# What a model writes when it means "nothing here". Stored verbatim these look
# like data: a workbook cell reading "null" is worse than an empty one, because
# it exports, sorts and gets counted.
NOT_REPORTED = {"null", "nan", "n/a", "na", "none reported", "not reported",
                "not stated", "not applicable", "not specified", "undefined",
                "-", "--", "---"}

# The same thing in the words a model actually writes: "Not explicitly
# reported", "Not reported in the article", "NR". Listing every spelling is
# hopeless, so the shape is matched instead, and anything that gets past both
# lands in the table as text, which is the one thing a review table must not
# show. Narrow on purpose: only an adverb may sit between the two halves, so
# "no adverse events reported" stays the finding it is.
_NOT_REPORTED_RE = re.compile(
    r"^(?:n\.?r\.?"
    r"|(?:data\s+)?(?:not|none)\s+(?:\w+ly\s+)?"
    r"(?:reported|stated|specified|mentioned|described|given|provided"
    r"|available|applicable|shown|discussed|assessed|listed)"
    r"(?:\s+(?:in|by|for|within|anywhere)\b.*)?)$",
    re.IGNORECASE,
)

SYSTEM = (
    "You are a meticulous systematic-review data extractor. You fill a data "
    "extraction form from a study report. For EACH field you must (1) quote the "
    "exact sentence or table cell that supports the value, then (2) give the value. "
    "Report numbers as the RAW value exactly as printed (a count or statistic, "
    "never a percentage or a derived/computed value). Use null when the field is "
    "not reported in the text."
)


def _blank(v: Any) -> bool:
    """True for a value that carries no information, however it was spelled."""
    if v is None:
        return True
    if isinstance(v, str):
        # Stripped of the quotes and the full stop a model puts round it, so
        # '"Not reported."' is read as the silence it is.
        s = v.strip().strip('"\u201c\u201d\'').strip().rstrip(".").strip()
        return (s == "" or s.lower() in NOT_REPORTED
                or bool(_NOT_REPORTED_RE.match(s)))
    return False


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _letters(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s)


def quote_in_text(quote: str, text: str) -> bool:
    """Is this quote actually in the paper?

    "Verbatim" is the claim a source quote makes, so the check is on characters,
    not meaning. PDF text layers break words across lines and sprinkle stray
    spaces, so a second pass compares letters and digits only. A very short
    quote is not evidence of anything and never passes.
    """
    q = _norm(quote)
    if len(q) < 12 or q in NOT_REPORTED:
        return False
    if q in _norm(text):
        return True
    ql, hay = _letters(q), _letters(_norm(text))
    return len(ql) >= 10 and ql[:160] in hay


def _num(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if s == "" or s.lower() in ("na", "n/a", "none", "null", "nan"):
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def _num_in_text(val: float, text: str) -> bool:
    cands = set()
    if float(val).is_integer():
        cands.add(str(int(val)))
        if abs(val) >= 1000:
            cands.add(f"{int(val):,}")
    for dp in (0, 1, 2):
        cands.add(f"{val:.{dp}f}")
    return any(re.search(r"(?<![\d.])" + re.escape(c) + r"(?![\d])", text) for c in cands)


def _field_lines(fields: List[Dict]) -> str:
    lines = []
    for f in fields:
        t = (f.get("type") or "text").lower()
        desc = (f.get("description") or "").strip()
        opts = f.get("options")
        extra = f" (one of: {', '.join(opts)})" if opts else ""
        lines.append(f'- "{f["name"]}" [{t}]{extra}: {desc}'.rstrip(": ").rstrip())
    return "\n".join(lines)


# How much of the article to send.
#
# A fixed prefix was the reason extraction returned blanks for funding,
# conflicts of interest, grant numbers and data availability: those sit in a
# paper's back matter, 80 to 90 % of the way in, so a 30,000-character prefix
# never showed them to the model at all. Measured on two papers (41.6k and
# 45.1k characters), 8 such fields: 3 filled at 30,000 characters, 8 of 8 on
# the whole text, with no other change.
#
# So the limit is the context, which is the real limit, and it is computed
# rather than guessed. Characters per token were measured with Ollama's own
# prompt_eval_count on those two papers: 3.33 for a clean two-column PDF and
# 2.65 for one whose text layer lost its spaces, because run-together words
# tokenise far worse. The constant sits well below the worse of the two,
# because the two costs are not symmetric: windowing a little early loses the
# middle of an unusually long paper, while underestimating by a few per cent
# overruns the context and the reply comes back cut mid-JSON, which loses every
# field in the group. 2.2 still holds a whole typical article, and leaves room
# for text that tokenises down to about 2.05 characters per token.
CHARS_PER_TOKEN = 2.2
# Held back for the system prompt, the field list, the JSON schema and the
# model's own answer, which for a 12-field group with quotes runs to about 900
# tokens. Generous, because overrunning the context costs the whole reply: it
# is cut mid-JSON and every field in the group comes back empty.
RESERVE_TOKENS = 4000
# A caller that does not say what context it opened gets a budget that fits the
# small default Ollama uses when nobody asks (2048 or 4096 tokens). Sending more
# than the window holds is silently truncated by the server, which is the same
# failure as truncating here but harder to see. The app always passes num_ctx.
ASSUMED_NUM_CTX = 8192
# Fewer characters than this cannot answer an extraction form, so the budget
# never drops below it even if the arithmetic says so.
MIN_TEXT_CHARS = 4000
# The shortest head or tail worth sending if the article has to be windowed.
MIN_SEGMENT_CHARS = 1500
# What a reviewer sees in the quote when the middle was dropped.
OMITTED_MARKER = "\n\n[... middle of the article omitted ...]\n\n"


def text_budget(num_ctx: Optional[int] = None, tables_chars: int = 0) -> int:
    """How many characters of article text fit alongside everything else."""
    usable = max(0, (num_ctx or ASSUMED_NUM_CTX) - RESERVE_TOKENS)
    return max(MIN_TEXT_CHARS, int(usable * CHARS_PER_TOKEN) - max(0, tables_chars))


def window(text: str, budget: int) -> str:
    """The part of the article to send when all of it does not fit.

    Head AND tail, never a prefix. The front holds the design, population and
    methods; the back holds funding, conflicts of interest, data availability
    and the author contributions. A prefix drops that whole class of field, and
    a review reports funding and conflicts whether or not a model found them.
    """
    if budget <= 0 or len(text) <= budget:
        return text
    if budget <= len(OMITTED_MARKER) + 2 * MIN_SEGMENT_CHARS:
        return text[:budget]
    # A quarter to the tail: in the two papers measured, the funding statement
    # began at 90 % and 80 % of the way through, so a fifth was not enough.
    room = budget - len(OMITTED_MARKER)
    tail = max(MIN_SEGMENT_CHARS, int(room * 0.25))
    head = room - tail
    return text[:head] + OMITTED_MARKER + text[-tail:]


def build_prompt(text: str, tables: str, fields: List[Dict],
                 max_text: Optional[int] = None, max_tables: int = 8000) -> str:
    body = window(text, max_text) if (text and max_text) else (text or "(no study text provided)")
    tbl = f"\n\nTABLES (extracted from the full text):\n{tables[:max_tables]}\n" if tables else ""
    schema = (
        '{"extractions": [{"name": "<field name>", '
        '"value": <value or null>, "source_quote": "<verbatim snippet supporting it>"}]}'
    )
    return (
        f"STUDY TEXT:\n{body}{tbl}\n\n"
        f"EXTRACTION FORM (fill every field):\n{_field_lines(fields)}\n\n"
        "For each field: first find the exact supporting sentence or table cell, "
        "then record the value. Match the requested arm/outcome/timepoint precisely. "
        "The source_quote must be the sentence the value appears in, copied "
        "verbatim; if you cannot find the value in a sentence, the field is not "
        "reported.\n"
        "Take only what THIS article reports about its own study. Its "
        "introduction and discussion describe other studies' findings, and "
        "those are not this article's data.\n"
        "If the article reports a field in more than one place, for more than "
        "one group, or in separate statements covering different parties (for "
        "example one author's conflicts and the remaining authors'), give every "
        "one of them in the value, separated by '; '. A partial answer to a "
        "field the article answers fully is wrong.\n"
        "If a field is not reported, use null and an empty source_quote.\n"
        f"Return ONLY this JSON object:\n{schema}\n"
    )


# Words that carry no evidence, so matching on them would mean nothing.
_EMPTY_WORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "by", "for",
    "from", "with", "was", "were", "is", "are", "be", "been", "has", "have",
    "had", "this", "that", "these", "those", "it", "its", "as", "not", "no",
    "there", "their", "which", "when", "than", "then", "also", "both",
}


def _content_words(s: str) -> List[str]:
    """The words in `s` that could be matched against a source sentence."""
    words = re.sub(r"[^0-9A-Za-z\u00b5\u0370-\u03ff.]+", " ", (s or "").lower()).split()
    out = []
    for w in words:
        w = w.strip(".")
        if not w or w in _EMPTY_WORDS:
            continue
        # Short words carry no evidence unless they are numbers: "5" does, "se"
        # does not.
        if len(w) < 3 and not any(ch.isdigit() for ch in w):
            continue
        out.append(w)
    return out


# A sentence that credits its finding to another paper. Both halves are needed:
# a citation alone is ordinary in a methods sentence ("dosed per the Stupp
# protocol (Stupp et al., 2005)"), and a reporting verb alone is how a paper
# describes its own results.
_CITATION = re.compile(
    r"\bet\s?al\b"                                  # (Rizzo et al., 2017)
    r"|\([A-Z][A-Za-z\u00c0-\u024f'`-]+,?\s+\d{4}\)"      # (Rizzo, 2017)
    r"|\[\d{1,3}(?:\s?[,\u2013-]\s?\d{1,3})*\]"             # [26], [14-16]
)
# Verbs that report a finding. Deliberately not "according to": a methods
# sentence citing the protocol it followed ("dosed according to the Stupp
# protocol (Stupp et al., 2005)") describes this study's own treatment, and
# flagging it would train the reviewer to ignore the flag.
_ATTRIBUTION = re.compile(
    r"\b(showed|shown|reported|reports|demonstrated|found|observed|describes"
    r"|described|suggested|suggests|concluded)\b",
    re.IGNORECASE,
)


def attributed_elsewhere(quote: str) -> bool:
    """Does this sentence credit its finding to a different paper?

    A clinical study of two anti-seizure drugs returned "300 and 800 µM" as its
    in vitro concentration. The value really is in the sentence it cites, so
    checking the value against its quote passes. The sentence reads "the
    inhibitory effect ... was observed at concentrations between 300 and 800 µM
    (Rizzo et al., 2017)": it is somebody else's cell-culture work, discussed,
    and the study did no in vitro work at all.

    Nothing about the value says so, and a review that pools it has pooled a
    number from a paper it never screened. This does not reject the value,
    because a methods sentence legitimately cites the protocol it followed; it
    marks it for a human, which is what the flag is for.
    """
    text = quote or ""
    return bool(_CITATION.search(text)) and bool(_ATTRIBUTION.search(text))


def supported_by_quote(value: Any, quote: str) -> bool:
    """Does the cited sentence actually say the value?

    The whole contract of locate-then-extract is that the quote is the sentence
    the value was read from. That was never checked: the quote was verified to
    exist somewhere in the paper, which it always does, and the value was
    checked against the whole paper rather than against its own quote.

    So a value lifted from a different sentence arrived pre-verified. A clinical
    paper's "in vitro concentration" came back as "300 and 800 µM" -- a figure
    from work its discussion cites, not from the study -- quoting a sentence
    containing neither number, and was marked high confidence and not flagged.

    Half the value's content words have to appear in the quote, which allows the
    paraphrase an extraction normally is ("retrospective cohort" for "an
    observational, retrospective single-center study") while rejecting a value
    the sentence does not mention at all.
    """
    want = _content_words(str(value if value is not None else ""))
    if not want:
        return True                                   # nothing to check
    have = set(_content_words(quote))
    if not have:
        return False
    hits = sum(1 for w in want if w in have)
    return hits >= max(1, (len(want) + 1) // 2)


def _confidence(field_type: str, value: Any, quote: str, text: str) -> str:
    if _blank(value):
        return "none"
    if field_type in NUMERIC_TYPES:
        n = _num(value)
        if n is None:
            return "low"
        # In its own quote, not merely somewhere in the paper. A paper contains
        # hundreds of numbers; that one of them matches says nothing about
        # whether this sentence is where this field's number came from.
        grounded = bool(quote) and _num_in_text(n, quote)
        return "high" if grounded else "low"
    # text/categorical/date: the quote IS the evidence, so it has to be in the
    # paper. Without this a model that echoes the field's own description back as
    # its source scores high and arrives pre-accepted, which is how a whole
    # risk-of-bias column can be invented and look verified.
    if not quote_in_text(quote, text):
        return "low"
    # A verdict with no words to check. "No" shares nothing with any sentence,
    # so its quote can neither support nor contradict it, and counting that as
    # verified is how a cohort study arrived with "immortal time bias: No" and
    # an animal study with a risk-of-bias judgement on every domain of an
    # instrument written for human trials, all pre-accepted.
    if not _content_words(str(value)):
        return "low"
    # The quote is in the paper. Whether it is the sentence THIS value came
    # from is a separate question, and the one that was never asked.
    if not supported_by_quote(value, quote):
        return "low"
    # Supported, but by a sentence reporting another paper's result.
    return "low" if attributed_elsewhere(quote) else "high"


# The labelled sections a paper makes its declarations in. These are the fields
# a review must report whether or not a model found them, and the ones where a
# partial answer is a reporting failure rather than a rough edge.
_DECLARATION_NAMES = (
    r"disclosures?|conflicts? of interest|competing interests?"
    r"|declarations? of (?:competing )?interest|funding|financial support"
    r"|acknowledge?ments?|data availability|ethics(?: approval| statement)?"
    r"|notes?|author contributions?|references"
)
# Either on a line of its own ("Funding") or inline with a colon
# ("Disclosures: TFC is ..."), which is how the two journals measured do it.
_DECLARATION_HEADING = re.compile(
    r"(?:^[ \t]*(?:" + _DECLARATION_NAMES + r")\b[:.]?[ \t]*$)"
    r"|(?:\b(?:" + _DECLARATION_NAMES + r")[ \t]*:)",
    re.IGNORECASE | re.MULTILINE,
)
# Below this share of its section, the answer is a sentence out of a
# declaration rather than the declaration. Measured on two papers: a complete
# answer covered 90 % and 92 % of its section, a partial one 8 % and 41 %.
_WHOLE_DECLARATION = 0.5


def _find_loose(text: str, quote: str) -> int:
    """Where `quote` starts in `text`, allowing for the line breaks a PDF adds.

    A quote is compared word by word, not character by character: the text
    layer breaks a sentence wherever the column ended, so "disclosed by\nthe
    other authors" has a newline exactly where the quote has a space.
    """
    if not text or not quote:
        return -1
    head = quote[:160]
    at = text.find(head)
    if at >= 0:
        return at
    words = head.split()
    if len(words) < 3:
        return -1
    hit = re.search(r"\s+".join(re.escape(w) for w in words), text)
    return hit.start() if hit else -1


def declaration_span(text: str, at: int) -> Optional[Tuple[int, int]]:
    """The labelled declaration section containing offset `at`, if any."""
    if not text or at < 0:
        return None
    marks = [m.start() for m in _DECLARATION_HEADING.finditer(text)]
    if not marks:
        return None
    before = [m for m in marks if m <= at]
    if not before:
        return None
    start = max(before)
    after = [m for m in marks if m > at]
    return (start, min(after) if after else len(text))


def partial_declaration(value: Any, quote: str, text: str) -> bool:
    """Is this one sentence of a declaration the paper makes at greater length?

    A paper's funding and conflicts are mandatory reporting items, and the
    model answers them with a single quotable sentence. One journal's
    disclosure section names a specific author's industry positions and then
    closes with "No potential conflicts of interests are disclosed by the other
    authors"; asked for the conflicts, the model returned that closing
    sentence. It is true, it is verbatim, its quote checks out, and as the
    conflicts of interest for that paper it is wrong. The same section-closing
    effect dropped a second grant number from the funding field.

    So: the quote sits inside a labelled declaration section, and the value
    covers less than half of it. A complete answer covers nearly all of its
    section, which is what separates the two.
    """
    text = text or ""
    quote = (quote or "").strip()
    answer = str(value if value is not None else "").strip()
    if len(quote) < 20 or len(answer) < 20:
        return False
    at = _find_loose(text, quote)
    if at < 0:
        return False
    span = declaration_span(text, at)
    if not span:
        return False
    section = text[span[0]:span[1]]
    if len(section) < 120:
        return False
    return (len(answer) / len(section)) < _WHOLE_DECLARATION


def grounded_in_text(value: Any, text: str, ratio: float = 0.8) -> bool:
    """Are the value's own words in the document at all?

    Used to accept a longer answer to a declaration without checking it
    against a single sentence, which a multi-sentence answer can never match.
    It is an anti-fabrication check, not a provenance one.
    """
    want = _content_words(str(value if value is not None else ""))
    if not want:
        return False
    have = set(_content_words(text or ""))
    hits = sum(1 for w in want if w in have)
    return hits >= ratio * len(want)


def partial_declarations(fields: List[Dict], text: str) -> List[str]:
    """Names of fields holding one sentence of a longer declaration."""
    return [str(f.get("name") or "") for f in (fields or [])
            if partial_declaration(f.get("value"), f.get("source_quote") or "", text)]


def colliding_fields(fields: List[Dict]) -> List[str]:
    """Names of fields that hold another field's answer rather than their own.

    Asked for twelve fields at once, a model that cannot find one of them will
    sometimes hand back a neighbour's answer instead of nothing. It is how
    "median follow-up" came back holding the median overall survival, quoting
    the sentence that gives the overall survival, and how "study period" came
    back as "15.64 months" when the study ran from 2015 to 2021.

    The detection is narrow on purpose. One sentence supporting several fields
    is ordinary: "we randomised 128 adults to metformin or placebo" is the
    source for the design, the sample size and the comparator. So the value has
    to be identical as well before this is a copy rather than a shared source,
    and when it is, at most one of the fields can be right.

    Every field in the group is returned, because which of them owns the answer
    cannot be told from here: the caller asks each one again on its own, where
    there is no neighbour to copy from.
    """
    groups: Dict[Tuple[str, str], List[str]] = {}
    for f in fields or []:
        value = str(f.get("value") or "").strip().lower()
        quote = str(f.get("source_quote") or "").strip().lower()
        # Both halves have to be there: a dozen empty fields are not copies of
        # each other, and a value with no quote has no shared source to share.
        if not value or len(quote) < 20:
            continue
        groups.setdefault((value, quote), []).append(str(f.get("name") or ""))
    return [name for names in groups.values() if len(names) > 1 for name in names]


def unsupported_fields(fields: List[Dict]) -> List[str]:
    """Names of fields whose value is not in the sentence they cite.

    A different failure from a copied answer, and a more common one: the model
    finds a value somewhere, then attaches a sentence that reads as if it were
    the source. Both have a value and a real quote, and neither can be told
    from a sound extraction by looking at the value alone.
    """
    out = []
    for f in fields or []:
        value = f.get("value")
        quote = str(f.get("source_quote") or "").strip()
        if _blank(value) or len(quote) < 20:
            continue
        ftype = (f.get("type") or "text").lower()
        if ftype in NUMERIC_TYPES:
            n = _num(value)
            ok = n is not None and _num_in_text(n, quote)
        else:
            ok = supported_by_quote(value, quote)
        if not ok:
            out.append(str(f.get("name") or ""))
    return out


def extract_form(text: str, tables: str, fields: List[Dict], model_name: str,
                 ai_service, num_ctx: Optional[int] = None) -> Dict[str, Any]:
    """Return per-field extractions with provenance and review flags.

    ai_service: the AIService class (passed in to avoid a circular import); we use
    ai_service.get_model and ai_service._extract_json.

    Returns {"fields": [{name, type, value, source_quote, confidence, needs_review}],
             "model": model_name}.
    """
    # Extraction reads a whole paper, so it asks for a bigger context than the
    # app's default, which is sized for an abstract.
    model = (ai_service.get_model(model_name, num_ctx=num_ctx) if num_ctx
             else ai_service.get_model(model_name))
    if not model:
        return {"fields": [{"name": f["name"], "type": f.get("type", "text"),
                            "value": None, "source_quote": "", "confidence": "none",
                            "needs_review": True, "status": "unread",
                            "error": "model unavailable"} for f in fields],
                "model": model_name, "error": "model unavailable"}

    budget = text_budget(num_ctx, len(tables or ""))
    prompt = build_prompt(text, tables, fields, max_text=budget)
    raw_by_name: Dict[str, Dict] = {}
    error = ""
    try:
        resp = model.invoke([HumanMessage(content=f"{SYSTEM}\n\n{prompt}")])
        data = ai_service._extract_json(getattr(resp, "content", "") or "") or {}
        if not data:
            # Usually the answer ran past the context and was cut mid-JSON. The
            # caller has to hear about it: every field comes back empty, which
            # otherwise looks like a paper that reports nothing.
            error = "the model did not return usable JSON (its answer may have been cut short)"
        for item in (data.get("extractions") or []):
            if isinstance(item, dict) and item.get("name"):
                raw_by_name[str(item["name"]).strip()] = item
    except Exception as e:  # noqa: BLE001
        error = str(e)
        print(f"[extraction_forms] extract failed: {e}")

    src = (text or "") + " " + (tables or "")
    out_fields = []
    for f in fields:
        name, ftype = f["name"], (f.get("type") or "text").lower()
        item = raw_by_name.get(name, {})
        value = item.get("value")
        quote = str(item.get("source_quote") or "").strip()
        if _blank(quote):
            quote = ""
        if ftype in NUMERIC_TYPES:
            value = _num(value)                       # coerce; None if unparseable
        elif _blank(value):
            value = None
        conf = _confidence(ftype, value, quote, src)
        if conf == "high" and partial_declaration(value, quote, text):
            conf = "low"
        # Human-in-the-loop policy: numbers always verified; text verified when
        # the model gave no source or low confidence.
        needs_review = (ftype in NUMERIC_TYPES) or conf in ("low", "none")
        # Why a cell is empty, which the caller already knows here and used to
        # throw away. A field the model answered with null is the model saying
        # the paper does not report it; a field missing from the reply, or a
        # whole group that failed, was never read. The reviewer has to tell
        # those apart: the first belongs in the review as "not reported", the
        # second is work still to do.
        if not _blank(value):
            status = "reported"
        elif error or name not in raw_by_name:
            status = "unread"
        else:
            status = "not_reported"
        reason = ""
        if not _blank(value) and quote:
            if attributed_elsewhere(quote):
                reason = ("the sentence this was read from credits it to another "
                          "paper, so it may not be this study's own result")
            elif ftype in NUMERIC_TYPES and not _num_in_text(_num(value) or 0, quote):
                reason = "the number does not appear in the sentence quoted for it"
            elif ftype not in NUMERIC_TYPES and not _content_words(str(value)):
                reason = ("a one-word verdict cannot be checked against the sentence "
                          "quoted for it, so it needs a human")
            elif ftype not in NUMERIC_TYPES and not supported_by_quote(value, quote):
                reason = "the sentence quoted for this does not mention the value"
            elif partial_declaration(value, quote, text):
                reason = ("the paper's own declaration on this runs longer than the "
                          "sentence quoted, so this may be part of the answer")
        out_fields.append({
            "name": name, "type": ftype, "value": value,
            "source_quote": quote, "confidence": conf, "needs_review": needs_review,
            "status": status, "review_reason": reason,
        })
    return {"fields": out_fields, "model": model_name, "error": error}
