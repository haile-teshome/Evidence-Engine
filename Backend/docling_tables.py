"""Layout-aware table extraction with Docling, the strongest deterministic tier.

pdfplumber reads the PDF's own text layer and infers a grid from where the
characters sit. A two-column journal page defeats that outright: the page
itself looks like a two-column table, and the real table sits inside one of
those columns. Docling runs a layout model over the rendered page to find the
table as a region first, then a table-structure model (TableFormer) to recover
its rows, columns and spans. Same input, same output, every time.

Everything here is lazy and guarded, the same shape as table_recognition.py: if
docling is not installed, `available()` is False and the caller falls through to
the next tier. Nothing in the packaged app depends on it.

    pip install docling          # ~1 GB with its models, downloaded on first use

Set EE_USE_DOCLING=0 to switch the tier off even when it is installed.
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import List, Optional


def _enabled() -> bool:
    return os.getenv("EE_USE_DOCLING", "1") != "0"


@lru_cache(maxsize=1)
def available() -> bool:
    """True when docling can be imported. Cached: the import is not cheap."""
    if not _enabled():
        return False
    try:
        import docling.document_converter  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _converter():
    from docling.document_converter import DocumentConverter
    return DocumentConverter()


def _rows_from_table(table, doc) -> List[List[str]]:
    """Rows of strings from a docling table, whichever shape this version gives.

    The cell grid comes first because it is what the model produced: a header
    spanning two columns appears twice, which is true. `export_to_dataframe`
    runs the same table through pandas, which renames the second copy to
    "Seizures recurrences.I" to keep its column names unique, and that lands in
    the reviewer's spreadsheet as a value the paper never printed.
    """
    try:
        grid = table.data.grid
        rows = [[(getattr(c, "text", "") or "").strip() for c in row] for row in grid]
        if any(any(c for c in r) for r in rows):
            return rows
    except Exception:
        pass
    try:
        df = table.export_to_dataframe(doc)
        header = [str(c) for c in df.columns]
        body = [[("" if v is None else str(v)).strip() for v in row]
                for row in df.astype(object).values.tolist()]
        return ([header] if any(h and not h.startswith("Unnamed") for h in header) else []) + body
    except Exception:
        return []


def _label_number(caption: str) -> Optional[int]:
    """The N in "Table N", when the caption starts with one."""
    m = re.match(r"\s*table\s+(\d+)\b", caption or "", re.IGNORECASE)
    return int(m.group(1)) if m else None


def name_tables(tables: List[dict]) -> List[dict]:
    """Give each table its number, and keep a caption only if it is plausible.

    Captions are linked to tables by the layout model, and on a crowded page it
    gets that wrong: two tables both claiming "Table 4". The labels are believed
    only when they read as a sequence; otherwise the tables are numbered by the
    order they appear in, which is reliable, and the captions are dropped,
    because a caption on the wrong table is worse than no caption at all.
    """
    labels = [_label_number(t.get("caption", "")) for t in tables]
    believable = (bool(labels) and all(n is not None for n in labels)
                  and len(set(labels)) == len(labels)
                  and labels == sorted(labels))
    for i, t in enumerate(tables):
        t["title"] = f"Table {labels[i] if believable else i + 1}"
        if not believable:
            t["caption"] = ""
    return tables


_READ_CACHE: "dict[str, dict]" = {}
_READ_CACHE_MAX = 8


def read_pdf(pdf_bytes: bytes) -> Optional[dict]:
    """Read a PDF properly: layout, tables, and OCR for pages that are images.

    Returns {"text": str, "tables": [...]} or None when docling is not here.

    Docling renders each page and runs OCR on anything with no text layer, so a
    scanned paper comes back with words. That is the whole reason this is worth
    the dependency: the alternative was telling the reviewer to go and OCR their
    own files.

    The result is memoised by content hash, because the import reads a document
    for its text and table extraction reads the same one later for its tables,
    and a conversion is seconds per page.
    """
    if not pdf_bytes or not available():
        return None
    key = hashlib.sha1(pdf_bytes).hexdigest()
    hit = _READ_CACHE.get(key)
    if hit is not None:
        return hit

    tmp: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            fh.write(pdf_bytes)
            tmp = Path(fh.name)
        result = _converter().convert(str(tmp))
        doc = getattr(result, "document", None)
        if doc is None:
            return None
        out = {"text": _document_text(doc), "tables": _document_tables(doc)}
        if len(_READ_CACHE) >= _READ_CACHE_MAX:
            _READ_CACHE.pop(next(iter(_READ_CACHE)))
        _READ_CACHE[key] = out
        return out
    except Exception as e:
        print(f"[docling read_pdf] {e}")
        return None
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except Exception:
                pass


def _document_text(doc) -> str:
    """The document as text. Markdown, because it keeps the tables readable as
    tables, which is what the extraction prompts are reading."""
    for export in ("export_to_markdown", "export_to_text"):
        try:
            text = getattr(doc, export)()
            if (text or "").strip():
                return text
        except Exception:
            continue
    return ""


def _document_tables(doc) -> List[dict]:
    out: List[dict] = []
    for table in (getattr(doc, "tables", []) or []):
        rows = [r for r in _rows_from_table(table, doc) if any(c for c in r)]
        if len(rows) < 2:
            continue
        caption = ""
        try:
            caption = (table.caption_text(doc) or "").strip()
        except Exception:
            pass
        page = None
        try:
            page = table.prov[0].page_no
        except Exception:
            pass
        out.append({"title": "", "data": rows, "caption": caption, "page": page})
    return name_tables(out)


def extract_text(pdf_bytes: bytes) -> Optional[str]:
    """The words in a PDF, OCR included. None when docling is not installed."""
    got = read_pdf(pdf_bytes)
    return None if got is None else got["text"]


def extract_tables(pdf_bytes: bytes) -> Optional[List[dict]]:
    """Tables in a PDF, or None when this tier cannot run.

    None and [] mean different things to the caller: None is "docling is not
    here, try the next reader", [] is "docling read the file and it has no
    tables", which is an answer.
    """
    got = read_pdf(pdf_bytes)
    return None if got is None else got["tables"]
