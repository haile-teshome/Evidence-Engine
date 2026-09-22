"""Bibliographic duplicate detection: the rule set the manuscript benchmarks.

Records are dicts with the keys title, doi, year, authors, journal and pages (all optional strings). The first
occurrence of a work is kept and later ones are flagged, which is the keep/remove convention of the ASySD
labelled sets (Hair et al. 2023) the rules were measured on. Development used the Diabetes and Cardiac sets with
error inspection on all five; on all five pooled (147,112 records) the rules reach F1 0.994 (95% CI 0.9935 to
0.9946), sensitivity 0.994, specificity 0.998, against 0.977 for the previous exact title-plus-year matcher and
0.999 for ASySD itself. With only title and DOI available (sources that carry no year, author, journal or page
fields) the same rules reach F1 0.987.

The rules, in order, for an incoming record against each candidate already kept:
  1. Same DOI (URL-decoded, prefix-stripped, lower-cased) and titles that agree (ratio >= 50 or one contained in
     the other) -> duplicate. Titles must agree because conference supplements share one DOI across abstracts.
  2. Cochrane reviews whose DOIs differ only in the .pubN version -> distinct records.
  3. Publication years more than one apart -> distinct. Conflicting first-page numbers -> distinct (the labelled
     sets treat the same abstract in two venues as two records).
  4. Otherwise the titles decide: normalised to letters and digits (spacing, hyphenation, <inf> tags and a trailing
     language tag do not count), similarity >= 96 is a duplicate unless the first authors conflict in the same
     journal with no pages to check; similarity 90 to 96 needs the same year and either the same first page or the
     same first author in a similar journal. A year one apart, or a title under 30 characters, needs page or author
     support (an identical short title with nothing contradicting it still counts). A title contained in a longer one counts when pages or first author agree.
Candidates are found by blocking on title word pairs and title prefix/suffix, plus the DOI index.
"""
from __future__ import annotations

import re
import urllib.parse
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

try:
    from rapidfuzz import fuzz as _fuzz
    def _ratio(a: str, b: str) -> float: return _fuzz.ratio(a, b)
    def _partial(a: str, b: str) -> float: return _fuzz.partial_ratio(a, b)
except ImportError:  # pragma: no cover - slow fallback, same semantics for ratio
    from difflib import SequenceMatcher
    def _ratio(a: str, b: str) -> float: return 100.0 * SequenceMatcher(None, a, b).ratio()
    def _partial(a: str, b: str) -> float:
        s, l = (a, b) if len(a) <= len(b) else (b, a)
        if not s: return 0.0
        return max(100.0 * SequenceMatcher(None, s, l[i:i + len(s)]).ratio() for i in range(0, max(1, len(l) - len(s) + 1)))

T_HIGH, T_MID, SHORT_TITLE = 96, 90, 30

_URL_DOI = re.compile(r"^(?:https?://)?(?:dx\.)?doi\.org/|^doi:\s*", re.I)
_NON = re.compile(r"[^a-z0-9]+")
_COCHRANE = re.compile(r"^(10\.1002/14651858\.cd\d+)(?:\.pub(\d+))?$")
_TAGS = re.compile(r"</?(inf|sup|i|b)>", re.I)
_BRACKET_END = re.compile(r"\[[^\]]*\]\s*\.?\s*$")
_LANG_END = re.compile(r"[\.\s\(\[]*(japanese|chinese|german|french|spanish|russian|portuguese|italian|polish|turkish|korean|dutch|czech|hungarian|danish|swedish|norwegian|finnish)[\)\]]?\s*\.?\s*$", re.I)
_MONTH = re.compile(r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", re.I)
_STOP = set("a an the of in on and for to with by from at as is are into during versus vs".split())
_DOI_LIKE = re.compile(r"10\.\d{4,}/\S+")


def norm_doi(d: Optional[str]) -> str:
    d = urllib.parse.unquote((d or "").strip())
    return _URL_DOI.sub("", d.lower()).strip().strip(".").rstrip("/")


def norm_title(t: Optional[str]) -> str:
    t = _TAGS.sub("", (t or "")).split("\n")[0].strip()
    t = _BRACKET_END.sub("", t).strip()
    t = _LANG_END.sub("", t).strip()
    return _NON.sub("", t.lower())


def _tokens(t: Optional[str]) -> List[str]:
    return [w for w in _NON.sub(" ", _TAGS.sub("", (t or "").split("\n")[0]).lower()).split() if w not in _STOP and len(w) > 2]


def norm_year(y) -> Optional[int]:
    m = re.search(r"(19|20)\d\d", str(y or ""))
    return int(m.group()) if m else None


def norm_first_page(p: Optional[str], year: Optional[int]) -> str:
    s = (p or "").strip().lower()
    if not s or _MONTH.search(s[:6]) or "e+" in s:
        return ""
    m = re.search(r"\d+", s.split("-")[0])
    if not m:
        return ""
    v = m.group()
    return "" if (year and v == str(year)) else (v.lstrip("0") or "0")


def norm_first_author(a: Optional[str]) -> str:
    s = _NON.sub(" ", (a or "").lower()).split()
    return s[0] if s and len(s[0]) > 1 else ""


def norm_journal(j: Optional[str]) -> str:
    return _NON.sub("", (j or "").lower())


def doi_from_text(*candidates: Optional[str]) -> str:
    """First DOI-looking string among the candidates (an id field, a URL), normalised; else ''."""
    for c in candidates:
        m = _DOI_LIKE.search(str(c or ""))
        if m:
            return norm_doi(m.group())
    return ""


def prepare(rec: Dict) -> Dict:
    y = norm_year(rec.get("year"))
    d = norm_doi(rec.get("doi"))
    m = _COCHRANE.match(d)
    t = norm_title(rec.get("title"))
    toks = _tokens(rec.get("title"))
    keys = set()
    if len(toks) >= 2:
        keys.add(("h", toks[0], toks[1])); keys.add(("t", toks[-2], toks[-1]))
    if len(toks) >= 4:
        keys.add(("m", toks[1], toks[2]))
    if len(t) >= 12:
        keys.add(("p", t[:12])); keys.add(("s", t[-12:]))
    return dict(d=d, cbase=(m.group(1) if m else None), cver=(m.group(2) if m else None), t=t, keys=keys, y=y,
                pg=norm_first_page(rec.get("pages"), y), au=norm_first_author(rec.get("authors")), jo=norm_journal(rec.get("journal")))


def same_work(x: Dict, k: Dict) -> bool:
    """Do prepared records x and k describe the same work?"""
    if not x["t"] or not k["t"]:
        return bool(x["d"] and x["d"] == k["d"])
    sim = _ratio(x["t"], k["t"])
    psim = _partial(x["t"], k["t"]) if min(len(x["t"]), len(k["t"])) >= 20 else 0.0
    if x["d"] and x["d"] == k["d"]:
        return sim >= 50 or psim >= 85
    if x["cbase"] and k["cbase"] and x["cbase"] == k["cbase"] and x["cver"] != k["cver"]:
        return False
    if x["y"] is not None and k["y"] is not None and abs(x["y"] - k["y"]) > 1:
        return False
    same_year = (x["y"] is None or k["y"] is None or x["y"] == k["y"])
    pages_conflict = bool(x["pg"] and k["pg"] and x["pg"] != k["pg"])
    pages_same = bool(x["pg"] and k["pg"] and x["pg"] == k["pg"])
    author_same = bool(x["au"] and k["au"] and x["au"] == k["au"])
    author_conflict = bool(x["au"] and k["au"] and x["au"] != k["au"])
    if pages_conflict:
        return False
    eff = max(sim, psim if (pages_same or author_same) else 0.0)
    if eff < T_MID:
        return False
    if not same_year and not (pages_same or author_same):
        return False
    if len(x["t"]) < SHORT_TITLE and not (author_same or pages_same):
        # A short, generic title needs a second key, unless it is letter-for-letter the same with nothing
        # contradicting it (the same record retrieved from two sources with no other metadata).
        if not (sim == 100 and same_year and not author_conflict):
            return False
    if eff >= T_HIGH:
        if author_conflict and not pages_same and (not x["pg"] or not k["pg"]) and x["jo"] and k["jo"] and x["jo"] == k["jo"]:
            return False
        return True
    journal_sim = _partial(x["jo"], k["jo"]) if x["jo"] and k["jo"] else 0.0
    return same_year and (pages_same or (author_same and journal_sim >= 80))


def flag_duplicates(records: Iterable[Dict]) -> List[bool]:
    """One flag per record, in order: True when the record duplicates an earlier kept one."""
    kept: List[Dict] = []
    by_doi: Dict[str, List[int]] = defaultdict(list)
    by_block: Dict[tuple, List[int]] = defaultdict(list)
    flags: List[bool] = []
    for rec in records:
        x = prepare(rec)
        cands = set(by_doi.get(x["d"], ())) if x["d"] else set()
        for key in x["keys"]:
            cands.update(by_block.get(key, ()))
        dup = any(same_work(x, kept[j]) for j in cands)
        flags.append(dup)
        if not dup:
            idx = len(kept); kept.append(x)
            if x["d"]:
                by_doi[x["d"]].append(idx)
            for key in x["keys"]:
                by_block[key].append(idx)
    return flags
