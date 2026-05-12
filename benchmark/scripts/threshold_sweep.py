"""Threshold sweep over an existing LEADS run's predictions.

Takes the per-paper `confidence` field (which maps the LEADS aggregate score
from [-1, +1] to [0, 1]) and recomputes recall/specificity/F1/MCC at a range
of thresholds. Output: a Pareto curve + the threshold that maximizes
the operator-supplied utility function (default: 2×recall + specificity).

Usage:
    python scripts/threshold_sweep.py reports/<run_id>/predictions.csv \
        --architecture leads_native --model-tier leads
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

sys.path.append(str(Path(__file__).resolve().parent.parent))

from metrics import compute  # noqa: E402


def load_predictions(path: Path, arch: str, tier: str) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["architecture"] != arch or r["model_tier"] != tier:
                continue
            rows.append(r)
    return rows


def sweep(rows: list[dict], thresholds: Sequence[float]) -> list[dict]:
    """For each threshold (applied to confidence ∈ [0, 1], i.e. LEADS score
    mapped from [-1, +1]), recompute predictions and metrics."""
    out = []
    y_true = [int(r["label"]) for r in rows if r["label"] not in ("", None)]
    confidences = [float(r["confidence"]) for r in rows if r["label"] not in ("", None)]

    for t in thresholds:
        y_pred = [1 if c >= t else 0 for c in confidences]
        m = compute(y_true, y_pred)
        # Map back to the underlying LEADS score (just for display)
        leads_score_thresh = 2 * t - 1
        out.append({
            "threshold_confidence": round(t, 3),
            "threshold_leads_score": round(leads_score_thresh, 3),
            "tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn,
            "recall": round(m.recall, 3),
            "specificity": round(m.specificity, 3),
            "precision": round(m.precision, 3),
            "f1": round(m.f1, 3),
            "mcc": round(m.mcc, 3),
        })
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions_csv")
    ap.add_argument("--architecture", default="leads_native")
    ap.add_argument("--model-tier", default="leads")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--utility", default="2*recall + specificity",
                    help="Expression in `r` (recall) and `s` (specificity) to maximize")
    args = ap.parse_args(argv)

    rows = load_predictions(Path(args.predictions_csv), args.architecture, args.model_tier)
    if not rows:
        print(f"No rows for {args.architecture} × {args.model_tier}", file=sys.stderr)
        return 1

    thresholds = [round(i * args.step, 3) for i in range(int(1 / args.step) + 1)]
    sweep_results = sweep(rows, thresholds)

    print(f"Threshold sweep over {len(rows)} predictions ({args.architecture} × {args.model_tier})")
    print(f"{'thresh':>7} {'l_score':>7} {'TP':>3} {'FP':>4} {'TN':>4} {'FN':>3} "
          f"{'recall':>7} {'spec':>6} {'prec':>6} {'F1':>6} {'MCC':>6}")
    print("-" * 75)
    for r in sweep_results:
        print(f"{r['threshold_confidence']:7.2f} {r['threshold_leads_score']:+7.2f} "
              f"{r['tp']:3d} {r['fp']:4d} {r['tn']:4d} {r['fn']:3d}  "
              f"{r['recall']:.3f}  {r['specificity']:.3f}  "
              f"{r['precision']:.3f}  {r['f1']:.3f}  {r['mcc']:+.3f}")

    # Best by user-supplied utility — exposed scalars: r=recall, s=specificity, recall, specificity, f1, mcc
    def util(row):
        return eval(args.utility, {
            "r": row["recall"], "recall": row["recall"],
            "s": row["specificity"], "specificity": row["specificity"],
            "f1": row["f1"], "mcc": row["mcc"], "precision": row["precision"],
        })

    best = max(sweep_results, key=util)
    print(f"\nBest by '{args.utility}': threshold={best['threshold_confidence']} "
          f"(LEADS score >= {best['threshold_leads_score']}) → "
          f"recall={best['recall']}, specificity={best['specificity']}, F1={best['f1']}")

    # Also surface a few canonical operating points
    print("\nCanonical operating points (highest specificity that still maintains recall):")
    for floor in [1.00, 0.95, 0.90, 0.80]:
        eligible = [r for r in sweep_results if r["recall"] >= floor]
        if not eligible:
            continue
        b = max(eligible, key=lambda r: r["specificity"])
        print(f"  recall ≥ {floor:.2f} → threshold {b['threshold_confidence']} | "
              f"recall={b['recall']}, specificity={b['specificity']}, F1={b['f1']}, MCC={b['mcc']:+.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
