"""Download a SYNERGY systematic review dataset, stratified-sample to N papers,
and fetch title+abstract from OpenAlex.

Usage:
    python scripts/download_synergy.py van_Dis_2020 --sample 300 --output data/synergy/van_Dis_2020
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import requests


SYNERGY_BASE = "https://raw.githubusercontent.com/asreview/synergy-dataset/master/datasets"


def fetch_ids(review: str) -> list[dict]:
    url = f"{SYNERGY_BASE}/{review}/{review}_ids.csv"
    print(f"Fetching {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    rows = []
    reader = csv.DictReader(r.text.splitlines())
    for row in reader:
        rows.append({
            "openalex_id": row.get("openalex_id") or row.get("id") or "",
            "doi": row.get("doi", ""),
            "label_included": int(row.get("label_included", 0)),
            "label_abstract_included": int(row.get("label_abstract_included", row.get("label_included", 0)) or 0),
        })
    return rows


def stratified_sample(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    pos = [r for r in rows if r["label_included"] == 1]
    neg = [r for r in rows if r["label_included"] == 0]
    if len(pos) + len(neg) <= n:
        return rows
    base_rate = len(pos) / (len(pos) + len(neg))
    n_pos = max(min(int(round(n * base_rate)), len(pos)), min(10, len(pos)))
    n_neg = n - n_pos
    sample = rng.sample(pos, n_pos) + rng.sample(neg, min(n_neg, len(neg)))
    rng.shuffle(sample)
    print(f"Stratified sample: {n_pos} positives + {n - n_pos} negatives (base rate {base_rate:.1%})")
    return sample


def fetch_openalex(openalex_id: str, throttle: float = 0.1, mailto: str = "researcher@example.com") -> dict:
    """OpenAlex lookup; throttled to be a polite citizen."""
    if not openalex_id:
        return {}
    # SYNERGY stores full URLs like 'https://openalex.org/W2049085090'.
    # The API expects the short id; hit it directly.
    short = openalex_id.rsplit("/", 1)[-1]
    if not short.startswith("W"):
        return {}
    url = f"https://api.openalex.org/works/{short}"
    time.sleep(throttle)
    try:
        r = requests.get(url, params={"mailto": mailto}, timeout=20)
        if r.status_code != 200:
            return {}
        return r.json()
    except Exception:
        return {}


def _reconstruct(inv_idx: dict | None) -> str:
    if not inv_idx:
        return ""
    pos = []
    for word, locs in inv_idx.items():
        for loc in locs:
            pos.append((loc, word))
    pos.sort()
    return " ".join(w for _, w in pos)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("review", help="SYNERGY review name (e.g. van_Dis_2020)")
    ap.add_argument("--sample", type=int, default=300, help="Stratified sample size (0 = all)")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--mailto", default=os.getenv("ENTREZ_EMAIL", "researcher@example.com"))
    ap.add_argument("--throttle", type=float, default=0.1, help="Seconds between OpenAlex calls")
    args = ap.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rows = fetch_ids(args.review)
    print(f"Loaded {len(rows)} ids ({sum(r['label_included'] for r in rows)} positives)")
    if args.sample and args.sample < len(rows):
        rows = stratified_sample(rows, args.sample)

    records = []
    fail = 0
    for i, row in enumerate(rows):
        meta = fetch_openalex(row["openalex_id"], throttle=args.throttle, mailto=args.mailto)
        if not meta:
            fail += 1
            continue
        title = meta.get("title") or meta.get("display_name") or ""
        abstract = _reconstruct(meta.get("abstract_inverted_index"))
        if not title and not abstract:
            fail += 1
            continue
        records.append({
            "paper_id": (meta.get("id") or row["openalex_id"]).split("/")[-1] or f"P{i:04d}",
            "title": title.strip(),
            "abstract": abstract.strip(),
            "label": row["label_included"],
            "doi": (meta.get("doi") or row["doi"] or "").replace("https://doi.org/", ""),
        })
        if (i + 1) % 25 == 0:
            print(f"  fetched {i + 1}/{len(rows)} (failures: {fail})")

    print(f"Done: {len(records)} usable records ({fail} failed lookups)")

    rec_path = out / "records.csv"
    with open(rec_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["paper_id", "title", "abstract", "label", "doi"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
