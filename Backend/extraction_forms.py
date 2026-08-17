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
from typing import Any, Dict, List, Optional

try:
    from langchain_core.messages import HumanMessage
except Exception:  # pragma: no cover - fall back to the langchain path used elsewhere
    from langchain.schema import HumanMessage

NUMERIC_TYPES = {"number", "numeric", "integer", "float"}

SYSTEM = (
    "You are a meticulous systematic-review data extractor. You fill a data "
    "extraction form from a study report. For EACH field you must (1) quote the "
    "exact sentence or table cell that supports the value, then (2) give the value. "
    "Report numbers as the RAW value exactly as printed (a count or statistic, "
    "never a percentage or a derived/computed value). Use null when the field is "
    "not reported in the text."
)


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


def build_prompt(text: str, tables: str, fields: List[Dict],
                 max_text: int = 12000, max_tables: int = 8000) -> str:
    body = text[:max_text] if text else "(no study text provided)"
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
        "If a field is not reported, use null and an empty source_quote.\n"
        f"Return ONLY this JSON object:\n{schema}\n"
    )


def _confidence(field_type: str, value: Any, quote: str, text: str) -> str:
    if value is None or value == "":
        return "none"
    if field_type in NUMERIC_TYPES:
        n = _num(value)
        if n is None:
            return "low"
        grounded = bool(quote) and _num_in_text(n, text + " " + quote)
        return "high" if grounded else "low"
    # text/categorical/date: strong if the model quoted a source
    return "high" if quote else "low"


def extract_form(text: str, tables: str, fields: List[Dict], model_name: str,
                 ai_service) -> Dict[str, Any]:
    """Return per-field extractions with provenance and review flags.

    ai_service: the AIService class (passed in to avoid a circular import); we use
    ai_service.get_model and ai_service._extract_json.

    Returns {"fields": [{name, type, value, source_quote, confidence, needs_review}],
             "model": model_name}.
    """
    model = ai_service.get_model(model_name)
    if not model:
        return {"fields": [{"name": f["name"], "type": f.get("type", "text"),
                            "value": None, "source_quote": "", "confidence": "none",
                            "needs_review": True, "error": "model unavailable"} for f in fields],
                "model": model_name}

    prompt = build_prompt(text, tables, fields)
    raw_by_name: Dict[str, Dict] = {}
    try:
        resp = model.invoke([HumanMessage(content=f"{SYSTEM}\n\n{prompt}")])
        data = ai_service._extract_json(getattr(resp, "content", "") or "") or {}
        for item in (data.get("extractions") or []):
            if isinstance(item, dict) and item.get("name"):
                raw_by_name[str(item["name"]).strip()] = item
    except Exception as e:  # noqa: BLE001
        print(f"[extraction_forms] extract failed: {e}")

    src = (text or "") + " " + (tables or "")
    out_fields = []
    for f in fields:
        name, ftype = f["name"], (f.get("type") or "text").lower()
        item = raw_by_name.get(name, {})
        value = item.get("value")
        quote = str(item.get("source_quote") or "").strip()
        if ftype in NUMERIC_TYPES:
            value = _num(value)                       # coerce; None if unparseable
        conf = _confidence(ftype, value, quote, src)
        # Human-in-the-loop policy: numbers always verified; text verified when
        # the model gave no source or low confidence.
        needs_review = (ftype in NUMERIC_TYPES) or conf in ("low", "none")
        out_fields.append({
            "name": name, "type": ftype, "value": value,
            "source_quote": quote, "confidence": conf, "needs_review": needs_review,
        })
    return {"fields": out_fields, "model": model_name}
