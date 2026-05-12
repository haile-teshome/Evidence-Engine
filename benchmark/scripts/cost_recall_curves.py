"""Reframe existing predictions as operational SR-screening metrics.

Pure post-processing on existing predictions. Computes:
- Screening burden: % of papers a human still has to review if they trust
  the model's includes + a confidence-ranked safety net for excludes
- WSS@95 / WSS@100: work saved at 95% / 100% recall
- Time-to-recall: for each recall floor in {0.80, 0.90, 0.95, 1.00}, the
  fraction of papers that must be reviewed when sorting by confidence
- Estimated wall-clock cost: s/paper × papers needed to reach recall target

Usage:
    python scripts/cost_recall_curves.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


def load_all(root: Path):
    """Auto-discover every reports/<run_id>/predictions.csv under benchmark/."""
    table = defaultdict(list)
    for path in sorted((root / "reports").glob("*/predictions.csv")):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                arch = r.get("architecture", "")
                tier = r.get("model_tier", "")
                pid = r.get("paper_id", "")
                label = r.get("label", "")
                pred = r.get("prediction", "")
                conf = r.get("confidence", "")
                t = r.get("wall_time_s", "")
                if not arch or not tier or label in ("", None):
                    continue
                try:
                    table[(arch, tier)].append({
                        "paper_id": pid,
                        "label": int(label),
                        "prediction": int(pred),
                        "confidence": float(conf) if conf else 0.5,
                        "wall_time_s": float(t) if t else 0.0,
                    })
                except ValueError:
                    continue
    return table


def cost_recall(rows: list[dict]) -> dict:
    """Sort by confidence (highest first), then for each recall target report
    how many papers reviewed to reach it (i.e., where in the ranked list the
    target-th true positive lives)."""
    n = len(rows)
    total_pos = sum(r["label"] for r in rows)
    if total_pos == 0:
        return {"n": n, "total_pos": 0}

    ranked = sorted(rows, key=lambda r: -r["confidence"])
    found = 0
    reviewed = 0
    targets = [0.50, 0.80, 0.90, 0.95, 1.00]
    target_counts = {t: int(round(t * total_pos)) for t in targets}
    target_hits: dict[float, int] = {}
    for r in ranked:
        reviewed += 1
        if r["label"] == 1:
            found += 1
        for t, k in list(target_counts.items()):
            if found >= k and t not in target_hits:
                target_hits[t] = reviewed
        if len(target_hits) == len(targets):
            break

    out = {"n": n, "total_pos": total_pos}
    for t in targets:
        hits = target_hits.get(t, n)
        out[f"reviewed_at_{int(t*100)}"] = hits
        out[f"burden_at_{int(t*100)}"] = hits / n
        # WSS = (baseline=0.95n) - reviewed)/n at recall=95
        baseline = t * n
        out[f"wss_at_{int(t*100)}"] = max(0.0, (baseline - hits) / n)

    # If we trust model includes and review only those: capture rate vs effort
    inc_idx = [r for r in rows if r["prediction"] == 1]
    inc_tp = sum(1 for r in inc_idx if r["label"] == 1)
    out["model_includes"] = len(inc_idx)
    out["model_includes_burden"] = len(inc_idx) / n
    out["model_includes_recall"] = inc_tp / total_pos
    out["model_includes_precision"] = inc_tp / len(inc_idx) if inc_idx else 0.0

    # Time per paper
    times = [r["wall_time_s"] for r in rows if r["wall_time_s"] > 0]
    out["mean_s_per_paper"] = sum(times) / len(times) if times else 0.0
    return out


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    table = load_all(root)

    # Per-cell cost-recall (one row per architecture x tier)
    print("=" * 130)
    print("PER-CELL COST-RECALL CURVES (rank by confidence; find papers reviewed to hit recall floors)")
    print("=" * 130)
    print(f"{'architecture':28s} {'tier':14s} {'n':>4} {'P':>3} {'inc_burden':>11} {'inc_recall':>11} "
          f"{'b@50':>5} {'b@80':>5} {'b@90':>5} {'b@95':>5} {'b@100':>6} {'WSS95':>6} {'s/p':>5}")
    print("-" * 130)

    all_rows = []
    for key in sorted(table.keys()):
        arch, tier = key
        m = cost_recall(table[key])
        if not m.get("total_pos"):
            continue
        print(f"{arch:28s} {tier:14s} {m['n']:4d} {m['total_pos']:3d} "
              f"{m['model_includes_burden']:11.3f} {m['model_includes_recall']:11.3f} "
              f"{m['burden_at_50']:5.2f} {m['burden_at_80']:5.2f} {m['burden_at_90']:5.2f} "
              f"{m['burden_at_95']:5.2f} {m['burden_at_100']:6.2f} "
              f"{m['wss_at_95']:6.3f} {m['mean_s_per_paper']:5.1f}")
        all_rows.append({"architecture": arch, "tier": tier, **m})

    # Headline interpretation rows
    print()
    print("=" * 130)
    print("OPERATIONAL TRANSLATION (assume 1000-paper screen; how many papers a reviewer must actually read)")
    print("=" * 130)
    print(f"{'architecture':28s} {'tier':14s}  burden@95 → papers/1000      time-to-95% recall (min)")
    print("-" * 130)
    for r in all_rows:
        papers_at_95 = int(round(r["burden_at_95"] * 1000))
        time_min = (r["burden_at_95"] * 1000 * r["mean_s_per_paper"]) / 60
        # Also report papers a human reviewer would have to verify (model includes mode)
        inc_papers = int(round(r["model_includes_burden"] * 1000))
        inc_recall = r["model_includes_recall"]
        print(f"{r['architecture']:28s} {r['tier']:14s}  {papers_at_95:5d}/1000 papers to 95% recall    "
              f"({time_min:6.1f} LLM-min)   |  trust-includes: {inc_papers:4d}/1000 → "
              f"recall={inc_recall:.2f}")

    # Write CSV
    out_csv = root / "reports" / "cost_recall.csv"
    keys = list(all_rows[0].keys()) if all_rows else []
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nWrote {out_csv} ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
