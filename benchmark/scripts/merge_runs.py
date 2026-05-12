"""Merge several per-tier benchmark run folders into a single consolidated
report folder. Each input folder is expected to have metrics.csv,
predictions.csv and the optional supplementary CSVs (bootstrap_ci.csv etc).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from tabulate import tabulate


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def merge(out_name: str, run_dirs: list[Path]) -> Path:
    out = Path("reports") / f"{out_name}-merged"
    out.mkdir(parents=True, exist_ok=True)

    combined: dict[str, list[dict]] = {}
    for rd in run_dirs:
        for csv_name in (
            "metrics.csv", "metrics_aggregated.csv", "predictions.csv",
            "bootstrap_ci.csv", "pairwise_mcnemar.csv", "interrater_panel.csv",
            "field_stratified.csv", "stability.csv",
        ):
            rows = _read_csv(rd / csv_name)
            if rows:
                combined.setdefault(csv_name, []).extend(rows)

    for csv_name, rows in combined.items():
        _write_csv(out / csv_name, rows)

    # Concatenate summary.md bodies
    summaries = []
    for rd in run_dirs:
        path = rd / "summary.md"
        if path.exists():
            summaries.append(f"## From {rd.name}\n\n" + path.read_text())
    if summaries:
        (out / "summary.md").write_text(
            f"# Merged benchmark report — {out_name}\n\n" + "\n\n---\n\n".join(summaries)
        )

    return out


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("Usage: merge_runs.py <out_run_id> <run_dir> [<run_dir> …]", file=sys.stderr)
        return 2
    out_name = argv[1]
    run_dirs = [Path(p.rstrip("/")) for p in argv[2:]]
    out = merge(out_name, run_dirs)
    print(f"Merged {len(run_dirs)} run dirs → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
