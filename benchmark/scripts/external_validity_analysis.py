"""External validity analysis: threshold sweep on LEADS-native applied to
3 fresh SYNERGY datasets (Sep_2021, Bannach-Brown_2019, Muthu_2021).

Reports per-dataset:
  - Default threshold metrics
  - Threshold sweep
  - Best operating point under "2*recall + specificity" utility
  - Canonical floors: recall ≥ {1.00, 0.95, 0.90, 0.80}

Compared side-by-side with van_Dis_2020 baseline at threshold +0.2 (confidence ≥ 0.6).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from metrics import compute  # noqa: E402


def load(path: Path, dataset: str = None, arch: str = "leads_native", tier: str = "leads") -> list[dict]:
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("architecture") != arch or r.get("model_tier") != tier:
                continue
            if dataset and r.get("dataset") != dataset:
                continue
            if r.get("label") in ("", None):
                continue
            try:
                r["label_i"] = int(r["label"])
                r["pred_i"] = int(r["prediction"])
                r["conf_f"] = float(r["confidence"])
            except (ValueError, TypeError):
                continue
            out.append(r)
    return out


def analyze(rows: list[dict], dataset_name: str) -> dict:
    if not rows:
        return {}
    y_true = [r["label_i"] for r in rows]
    confs = [r["conf_f"] for r in rows]
    total_pos = sum(y_true)

    print(f"\n{'='*100}")
    print(f"DATASET: {dataset_name}   (n={len(rows)}, positives={total_pos}, base rate={total_pos/len(rows):.1%})")
    print(f"{'='*100}")

    # Default threshold (model's own prediction)
    y_pred = [r["pred_i"] for r in rows]
    m = compute(y_true, y_pred, confs)
    print(f"Default (LEADS threshold=0.0): TP={m.tp} FP={m.fp} TN={m.tn} FN={m.fn} "
          f"recall={m.recall:.3f} spec={m.specificity:.3f} F1={m.f1:.3f} MCC={m.mcc:+.3f} WSS@95={m.wss_at_95:.3f}")
    print()
    print(f"Threshold sweep (confidence ≥ t → INCLUDE):")
    print(f"  {'t':>5} {'leads':>7} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'recall':>7} {'spec':>6} {'F1':>6} {'MCC':>7} {'util':>6}")
    sweep = []
    for i in range(0, 21):
        t = round(i * 0.05, 2)
        y_pred = [1 if c >= t else 0 for c in confs]
        mm = compute(y_true, y_pred, confs)
        leads_t = 2 * t - 1
        util = 2 * mm.recall + mm.specificity
        sweep.append({"t": t, "leads_t": leads_t, "tp": mm.tp, "fp": mm.fp, "tn": mm.tn, "fn": mm.fn,
                       "recall": mm.recall, "specificity": mm.specificity, "f1": mm.f1, "mcc": mm.mcc, "util": util})
        print(f"  {t:5.2f} {leads_t:+7.2f} {mm.tp:4d} {mm.fp:4d} {mm.tn:4d} {mm.fn:4d} "
              f"{mm.recall:7.3f} {mm.specificity:6.3f} {mm.f1:6.3f} {mm.mcc:+7.3f} {util:6.3f}")

    # Sweet spot
    best = max(sweep, key=lambda r: r["util"])
    print(f"\nBest by '2*recall + specificity': threshold={best['t']} (LEADS ≥ {best['leads_t']:+.2f}) → "
          f"recall={best['recall']:.3f} spec={best['specificity']:.3f} util={best['util']:.3f}")

    # Canonical recall floors
    print(f"\nCanonical operating points (highest specificity while maintaining recall ≥ floor):")
    for floor in [1.00, 0.95, 0.90, 0.80]:
        eligible = [r for r in sweep if r["recall"] >= floor]
        if not eligible:
            print(f"  recall ≥ {floor:.2f}: NOT ACHIEVABLE (max recall = {max(r['recall'] for r in sweep):.2f})")
            continue
        b = max(eligible, key=lambda r: r["specificity"])
        print(f"  recall ≥ {floor:.2f} → t={b['t']} (LEADS ≥ {b['leads_t']:+.2f}) → "
              f"recall={b['recall']:.3f} spec={b['specificity']:.3f} F1={b['f1']:.3f} MCC={b['mcc']:+.3f}")

    # Specifically: the van_Dis sweet-spot threshold (confidence ≥ 0.6, LEADS ≥ +0.2)
    at_06 = next(r for r in sweep if r["t"] == 0.60)
    print(f"\nAt van_Dis sweet-spot threshold (conf ≥ 0.6 / LEADS ≥ +0.20): "
          f"recall={at_06['recall']:.3f} spec={at_06['specificity']:.3f} "
          f"F1={at_06['f1']:.3f} MCC={at_06['mcc']:+.3f}")
    return {"dataset": dataset_name, "n": len(rows), "total_pos": total_pos,
            "default_metrics": m, "sweep": sweep, "best": best, "at_t_06": at_06}


def _latest(root: Path, prefix: str) -> Path | None:
    matches = sorted((root / "reports").glob(f"{prefix}*/predictions.csv"))
    return matches[-1] if matches else None


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ext_csv = _latest(root, "external-validity-")
    van_dis_csv = _latest(root, "leads-local-")
    if ext_csv is None or van_dis_csv is None:
        print("Need both reports/external-validity-*/ and reports/leads-local-*/ runs.", file=sys.stderr)
        return 2

    print("EXTERNAL VALIDITY: leads_native × leads on 3 new SYNERGY systematic-review datasets")
    print("vs. baseline (van_Dis_2020 same architecture × tier)\n")

    # Baseline first
    van_dis_rows = load(van_dis_csv)
    results = []
    if van_dis_rows:
        results.append(analyze(van_dis_rows, "van_Dis_2020 (baseline)"))

    # External datasets
    for ds in ["synergy/Sep_2021", "synergy/Bannach-Brown_2019", "synergy/Muthu_2021"]:
        rows = load(ext_csv, dataset=ds)
        if not rows:
            print(f"\n(missing rows for {ds})")
            continue
        results.append(analyze(rows, ds))

    # Combined summary table — sweet-spot threshold across all
    print(f"\n{'='*100}")
    print("EXTERNAL VALIDITY SUMMARY (at LEADS ≥ +0.20 sweet-spot threshold)")
    print(f"{'='*100}")
    print(f"{'Dataset':35s} {'n':>4} {'P':>3} {'recall':>7} {'spec':>6} {'F1':>6} {'MCC':>7} {'util':>6}  "
          f"{'best_t':>7} {'best_recall':>11} {'best_spec':>9}")
    for r in results:
        b = r["best"]
        at = r["at_t_06"]
        print(f"{r['dataset']:35s} {r['n']:4d} {r['total_pos']:3d} "
              f"{at['recall']:7.3f} {at['specificity']:6.3f} {at['f1']:6.3f} {at['mcc']:+7.3f} {at['util']:6.3f}  "
              f"{b['leads_t']:+7.2f} {b['recall']:11.3f} {b['specificity']:9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
