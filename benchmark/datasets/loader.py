"""Dataset loader. Each dataset lives at `data/<group>/<name>/` containing:
  - records.csv   (paper_id, title, abstract, label)
  - criteria.yaml (pico + inclusion + exclusion)

The CLI accepts either `<group>/<name>` paths or top-level names that map to
`data/<name>/`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import yaml

from architectures.base import Paper, ScreeningContext


@dataclass
class Dataset:
    name: str
    context: ScreeningContext
    papers: List[Paper]

    def __iter__(self) -> Iterator[Paper]:
        return iter(self.papers)


def _find_dir(root: Path, name: str) -> Path:
    """Resolve a dataset name to its directory. Supports nested names like 'synergy/foo'."""
    direct = root / name
    if direct.is_dir():
        return direct
    raise FileNotFoundError(f"No dataset directory found at {direct}")


def load_dataset(name: str, data_root: Path | str = "data", limit: int | None = None) -> Dataset:
    root = Path(data_root)
    folder = _find_dir(root, name)
    crit_path = folder / "criteria.yaml"
    rec_path = folder / "records.csv"
    if not crit_path.exists():
        raise FileNotFoundError(f"Missing criteria.yaml in {folder}")
    if not rec_path.exists():
        raise FileNotFoundError(f"Missing records.csv in {folder}")

    with open(crit_path) as fh:
        crit = yaml.safe_load(fh) or {}
    ctx = ScreeningContext(
        pico={
            "population": crit.get("pico", {}).get("population", ""),
            "intervention": crit.get("pico", {}).get("intervention", ""),
            "comparator": crit.get("pico", {}).get("comparator", ""),
            "outcome": crit.get("pico", {}).get("outcome", ""),
        },
        inclusion=list(crit.get("inclusion", []) or []),
        exclusion=list(crit.get("exclusion", []) or []),
    )

    papers: List[Paper] = []
    with open(rec_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label_raw = (row.get("label") or row.get("included") or "").strip()
            try:
                label = int(label_raw) if label_raw != "" else None
            except ValueError:
                label = 1 if label_raw.lower() in ("yes", "true", "include") else 0
            papers.append(
                Paper(
                    paper_id=row.get("paper_id") or row.get("id") or str(len(papers) + 1),
                    title=row.get("title", "").strip(),
                    abstract=row.get("abstract", "").strip(),
                    label=label,
                )
            )
            if limit and len(papers) >= limit:
                break

    return Dataset(name=name, context=ctx, papers=papers)
