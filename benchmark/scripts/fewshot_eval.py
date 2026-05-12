"""Compare few-shot LEADS vs plain LEADS on the 282-paper holdout
(excluding the 6 exemplars baked into the few-shot prompt).

Reports recall/specificity/F1/MCC + threshold sweep on each, side-by-side.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from architectures.leads_native_fewshot import EXEMPLAR_IDS  # noqa: E402
from metrics import compute  # noqa: E402


def load(path: Path, arch: str, tier: str) -> list[dict]:
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["architecture"] != arch or r["model_tier"] != tier:
                continue
            try:
                r["label_i"] = int(r["label"]) if r["label"] not in ("", None) else None
                r["pred_i"] = int(r["prediction"])
                r["conf_f"] = float(r["confidence"])
            except (ValueError, TypeError):
                continue
            out.append(r)
    return out


def summarize(rows: list[dict], label: str, thresholds: list[float]) -> None:
    print(f"\n=== {label} ===")
    print(f"  N papers: {len(rows)}   labels: {sum(r['label_i'] for r in rows)} positives")
    # Default-threshold metrics (uses model's own prediction column)
    y_true = [r["label_i"] for r in rows]
    y_pred_default = [r["pred_i"] for r in rows]
    y_conf = [r["conf_f"] for r in rows]
    m = compute(y_true, y_pred_default, y_conf)
    print(f"  Default (threshold=0.0): TP={m.tp} FP={m.fp} TN={m.tn} FN={m.fn} "
          f"recall={m.recall:.3f} spec={m.specificity:.3f} F1={m.f1:.3f} MCC={m.mcc:+.3f} WSS@95={m.wss_at_95:.3f}")
    print(f"  Threshold sweep (confidence ≥ t → INCLUDE):")
    print(f"    {'t':>5} {'leads':>7} {'TP':>3} {'FP':>4} {'TN':>4} {'FN':>3} {'recall':>7} {'spec':>6} {'F1':>6} {'MCC':>7} {'util':>6}")
    for t in thresholds:
        y_pred = [1 if c >= t else 0 for c in y_conf]
        mm = compute(y_true, y_pred, y_conf)
        leads_t = 2 * t - 1
        util = 2 * mm.recall + mm.specificity
        print(f"    {t:5.2f} {leads_t:+7.2f} {mm.tp:3d} {mm.fp:4d} {mm.tn:4d} {mm.fn:3d} "
              f"{mm.recall:7.3f} {mm.specificity:6.3f} {mm.f1:6.3f} {mm.mcc:+7.3f} {util:6.3f}")


def _latest(root: Path, prefix: str) -> Path | None:
    matches = sorted((root / "reports").glob(f"{prefix}*/predictions.csv"))
    return matches[-1] if matches else None


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    fs_path = _latest(root, "leads-fewshot-")
    bl_path = _latest(root, "leads-local-")
    if fs_path is None or bl_path is None:
        print("Need both reports/leads-fewshot-*/ and reports/leads-local-*/ runs.", file=sys.stderr)
        return 2
    fewshot = load(fs_path, "leads_native_fewshot", "leads")
    baseline = load(bl_path, "leads_native", "leads")

    # Filter out the 6 exemplars from BOTH cells, using the same holdout set
    fs_holdout = [r for r in fewshot if r["paper_id"] not in EXEMPLAR_IDS]
    bl_holdout = [r for r in baseline if r["paper_id"] not in EXEMPLAR_IDS]

    # Sanity check: same papers in both
    fs_ids = {r["paper_id"] for r in fs_holdout}
    bl_ids = {r["paper_id"] for r in bl_holdout}
    assert fs_ids == bl_ids, f"Paper-id mismatch: {len(fs_ids ^ bl_ids)} differences"

    thresholds = [round(0.05 * i, 2) for i in range(0, 21)]

    summarize(bl_holdout, f"BASELINE — leads_native × leads (holdout n={len(bl_holdout)})", thresholds)
    summarize(fs_holdout, f"FEW-SHOT — leads_native_fewshot × leads (holdout n={len(fs_holdout)})", thresholds)

    # Direct per-paper comparison
    by_id_bl = {r["paper_id"]: r for r in bl_holdout}
    by_id_fs = {r["paper_id"]: r for r in fs_holdout}
    flipped_to_yes = []
    flipped_to_no = []
    agree = 0
    for pid in sorted(fs_ids):
        if by_id_bl[pid]["pred_i"] == by_id_fs[pid]["pred_i"]:
            agree += 1
        elif by_id_fs[pid]["pred_i"] == 1 and by_id_bl[pid]["pred_i"] == 0:
            flipped_to_yes.append((pid, by_id_bl[pid]["label_i"]))
        elif by_id_fs[pid]["pred_i"] == 0 and by_id_bl[pid]["pred_i"] == 1:
            flipped_to_no.append((pid, by_id_bl[pid]["label_i"]))

    print(f"\n=== Pairwise comparison (n={len(fs_ids)} holdout papers) ===")
    print(f"  Agreement: {agree}/{len(fs_ids)} = {agree/len(fs_ids):.3f}")
    print(f"  Flipped EXCLUDE→INCLUDE in few-shot: {len(flipped_to_yes)}  "
          f"(of which true-positives: {sum(1 for _, lb in flipped_to_yes if lb == 1)})")
    print(f"  Flipped INCLUDE→EXCLUDE in few-shot: {len(flipped_to_no)}  "
          f"(of which true-positives: {sum(1 for _, lb in flipped_to_no if lb == 1)})")

    # Confidence-distribution shift
    print(f"\n=== Confidence distribution on holdout (n={len(fs_holdout)}) ===")
    print(f"  Baseline mean confidence: {sum(r['conf_f'] for r in bl_holdout)/len(bl_holdout):.3f}")
    print(f"  Few-shot mean confidence: {sum(r['conf_f'] for r in fs_holdout)/len(fs_holdout):.3f}")
    print(f"  Baseline positives' mean conf: "
          f"{sum(r['conf_f'] for r in bl_holdout if r['label_i']==1)/max(1,sum(1 for r in bl_holdout if r['label_i']==1)):.3f}")
    print(f"  Few-shot positives' mean conf: "
          f"{sum(r['conf_f'] for r in fs_holdout if r['label_i']==1)/max(1,sum(1 for r in fs_holdout if r['label_i']==1)):.3f}")
    print(f"  Baseline negatives' mean conf: "
          f"{sum(r['conf_f'] for r in bl_holdout if r['label_i']==0)/max(1,sum(1 for r in bl_holdout if r['label_i']==0)):.3f}")
    print(f"  Few-shot negatives' mean conf: "
          f"{sum(r['conf_f'] for r in fs_holdout if r['label_i']==0)/max(1,sum(1 for r in fs_holdout if r['label_i']==0)):.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
