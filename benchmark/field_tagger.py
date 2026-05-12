"""One-shot LLM classifier that tags each paper into a paper-type bucket.

Runs once per paper at dataset-load time, cached to <dataset_dir>/field_tags.csv
so repeat benchmark runs don't pay this cost again.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable

CATEGORIES = ["RCT", "observational", "qualitative", "review", "preclinical", "other"]


def _heuristic_tag(title: str, abstract: str) -> str:
    blob = (title + " " + abstract).lower()
    if any(k in blob for k in ("randomized", "randomised", "placebo-controlled", "double-blind", " rct")):
        return "RCT"
    if any(k in blob for k in ("systematic review", "meta-analysis", "scoping review", "narrative review", "literature review")):
        return "review"
    if any(k in blob for k in ("in vitro", "in vivo", "mouse model", "rat model", "cell line", "knockout", "transgenic")):
        return "preclinical"
    if any(k in blob for k in ("interview", "focus group", "thematic analysis", "qualitative", "phenomenolog")):
        return "qualitative"
    if any(k in blob for k in ("cohort", "case-control", "case control", "cross-sectional", "registry", "observational")):
        return "observational"
    return "other"


def _llm_tag(title: str, abstract: str, model) -> str:
    from architectures.base import invoke, extract_json
    prompt = f"""Classify the study design of this paper into ONE category:
- RCT (any randomized trial)
- observational (cohort, case-control, cross-sectional, registry)
- qualitative (interviews, thematic analysis)
- review (systematic, narrative, scoping, meta-analysis)
- preclinical (in vitro, animal models)
- other

Return ONLY JSON: {{"category": "<one of above>"}}

Title: {title}
Abstract: {abstract[:600]}
"""
    try:
        raw = invoke(model, prompt)
        data = extract_json(raw) or {}
        cat = str(data.get("category", "")).strip().lower()
        # Normalize
        for c in CATEGORIES:
            if cat.startswith(c.lower()):
                return c
    except Exception:
        pass
    return _heuristic_tag(title, abstract)


def tag_papers(papers: Iterable, dataset_dir: Path, model=None, force: bool = False) -> Dict[str, str]:
    """Return paper_id -> category. Caches to <dataset_dir>/field_tags.csv.

    If `model` is None, uses purely heuristic tagging (fast, free, decent).
    """
    cache_path = Path(dataset_dir) / "field_tags.csv"
    cache: Dict[str, str] = {}
    if cache_path.exists() and not force:
        with open(cache_path, newline="") as fh:
            for row in csv.DictReader(fh):
                cache[row["paper_id"]] = row["category"]

    out: Dict[str, str] = {}
    new_rows = []
    for p in papers:
        if p.paper_id in cache:
            out[p.paper_id] = cache[p.paper_id]
            continue
        tag = _llm_tag(p.title, p.abstract, model) if model else _heuristic_tag(p.title, p.abstract)
        out[p.paper_id] = tag
        new_rows.append({"paper_id": p.paper_id, "category": tag})

    if new_rows:
        existing = []
        if cache_path.exists():
            with open(cache_path, newline="") as fh:
                existing = list(csv.DictReader(fh))
        with open(cache_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["paper_id", "category"])
            writer.writeheader()
            for row in existing + new_rows:
                writer.writerow(row)
    return out
