"""Cross-model voting ensembles over existing predictions (zero new LLM calls).

For each architecture, combine predictions across multiple model tiers via
several voting rules (any-include, majority, all-include, soft-average) and
recompute metrics. This is pure post-processing — we use predictions already
written to disk from prior runs.

Reads every predictions.csv under `reports/`, groups by (architecture, tier),
and builds ensembles. Designed to be re-run any time new benchmark runs land
in `reports/`.

Usage:
    python scripts/cross_model_ensemble.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from metrics import compute  # noqa: E402


def load_all(root: Path) -> dict:
    """Returns {(arch, tier): {paper_id: (label, prediction, confidence)}}.

    Auto-discovers every reports/<run_id>/predictions.csv under the benchmark dir."""
    table: dict[tuple, dict[str, tuple]] = defaultdict(dict)
    for path in sorted((root / "reports").glob("*/predictions.csv")):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                arch = r.get("architecture", "")
                tier = r.get("model_tier", "")
                pid = r.get("paper_id", "")
                label = r.get("label", "")
                pred = r.get("prediction", "")
                conf = r.get("confidence", "")
                if not arch or not tier or not pid or label in ("", None):
                    continue
                try:
                    label_i = int(label)
                    pred_i = int(pred)
                    conf_f = float(conf) if conf else 0.5
                except ValueError:
                    continue
                table[(arch, tier)][pid] = (label_i, pred_i, conf_f)
    return table


def ensemble_metrics(rows: list[tuple[int, list[int], list[float]]], rule: str) -> dict:
    """rows = [(label, [pred_per_tier], [conf_per_tier])]; rule ∈ {any, majority, all, avg}."""
    y_true = [r[0] for r in rows]
    y_pred: list[int] = []
    y_conf: list[float] = []
    for _, preds, confs in rows:
        if rule == "any":
            y_pred.append(1 if any(p == 1 for p in preds) else 0)
            y_conf.append(max(confs))
        elif rule == "all":
            y_pred.append(1 if all(p == 1 for p in preds) else 0)
            y_conf.append(min(confs))
        elif rule == "majority":
            yes = sum(1 for p in preds if p == 1)
            y_pred.append(1 if yes > len(preds) / 2 else 0)
            y_conf.append(sum(confs) / len(confs))
        elif rule == "avg":
            avg = sum(confs) / len(confs)
            y_pred.append(1 if avg >= 0.5 else 0)
            y_conf.append(avg)
        else:
            raise ValueError(rule)
    m = compute(y_true, y_pred, y_conf)
    return {
        "n": m.n, "tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn,
        "recall": m.recall, "specificity": m.specificity, "precision": m.precision,
        "f1": m.f1, "mcc": m.mcc, "wss_at_95": m.wss_at_95,
    }


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    table = load_all(root)

    # Group by architecture: which tiers do we have for each?
    by_arch: dict[str, list[str]] = defaultdict(list)
    for (arch, tier) in table.keys():
        by_arch[arch].append(tier)

    print("Available cells:")
    for arch, tiers in sorted(by_arch.items()):
        print(f"  {arch:30s} → {sorted(tiers)}")
    print()

    # Per-architecture cross-tier ensembles
    print("=" * 100)
    print("PER-ARCHITECTURE CROSS-TIER ENSEMBLES (combine all available tiers for each architecture)")
    print("=" * 100)
    header = f"{'architecture':32s} {'rule':10s} {'tiers':30s} {'n':>4} {'TP':>3} {'FP':>4} {'FN':>3} {'recall':>7} {'spec':>6} {'F1':>6} {'MCC':>7} {'WSS95':>6}"
    print(header)
    print("-" * len(header))

    all_results: list[dict] = []
    for arch in sorted(by_arch):
        tiers = sorted(by_arch[arch])
        if len(tiers) < 2:
            continue
        # Find paper_ids common to ALL tiers for this arch
        common: set[str] | None = None
        for t in tiers:
            ids = set(table[(arch, t)].keys())
            common = ids if common is None else common & ids
        if not common:
            continue
        common_sorted = sorted(common)
        rows: list[tuple[int, list[int], list[float]]] = []
        for pid in common_sorted:
            label = table[(arch, tiers[0])][pid][0]
            preds = [table[(arch, t)][pid][1] for t in tiers]
            confs = [table[(arch, t)][pid][2] for t in tiers]
            rows.append((label, preds, confs))
        for rule in ("any", "majority", "all", "avg"):
            m = ensemble_metrics(rows, rule)
            tier_label = "+".join(tiers)
            print(f"{arch:32s} {rule:10s} {tier_label:30s} {m['n']:4d} {m['tp']:3d} {m['fp']:4d} {m['fn']:3d} "
                  f"{m['recall']:7.3f} {m['specificity']:6.3f} {m['f1']:6.3f} {m['mcc']:+7.3f} {m['wss_at_95']:6.3f}")
            all_results.append({"scope": "per-arch", "architecture": arch, "rule": rule, "tiers": tier_label, **m})
        print()

    # Cross-architecture ensembles at the same tier (where multiple architectures share a tier)
    print("=" * 100)
    print("CROSS-ARCHITECTURE ENSEMBLES (per tier, combine all available architectures)")
    print("=" * 100)
    print(header)
    print("-" * len(header))

    by_tier: dict[str, list[str]] = defaultdict(list)
    for (arch, tier) in table.keys():
        by_tier[tier].append(arch)
    for tier in sorted(by_tier):
        archs = sorted(set(by_tier[tier]))
        if len(archs) < 2:
            continue
        common: set[str] | None = None
        for a in archs:
            ids = set(table[(a, tier)].keys())
            common = ids if common is None else common & ids
        if not common:
            continue
        common_sorted = sorted(common)
        rows = []
        for pid in common_sorted:
            label = table[(archs[0], tier)][pid][0]
            preds = [table[(a, tier)][pid][1] for a in archs]
            confs = [table[(a, tier)][pid][2] for a in archs]
            rows.append((label, preds, confs))
        for rule in ("any", "majority", "all", "avg"):
            m = ensemble_metrics(rows, rule)
            arch_label = "+".join(archs)
            print(f"{'(all archs)':32s} {rule:10s} {tier:30s} {m['n']:4d} {m['tp']:3d} {m['fp']:4d} {m['fn']:3d} "
                  f"{m['recall']:7.3f} {m['specificity']:6.3f} {m['f1']:6.3f} {m['mcc']:+7.3f} {m['wss_at_95']:6.3f}")
            all_results.append({"scope": "per-tier", "architecture": arch_label, "rule": rule, "tiers": tier, **m})
        print()

    # Curated "best of breed" ensembles — combine the strongest cells across runs
    print("=" * 100)
    print("CURATED ENSEMBLES (handpicked combinations of strong cells)")
    print("=" * 100)
    print(header)
    print("-" * len(header))

    bundles = [
        ("high-recall trio (LEADS native ×3)", [
            ("leads_native", "leads"),
            ("leads_multi_persona", "leads"),
            ("leads_native", "medium"),
        ]),
        ("LEADS + strict", [
            ("leads_native", "leads"),
            ("single_combined", "leads"),
        ]),
        ("LEADS + decompose_match medgemma", [
            ("leads_native", "leads"),
            ("decompose_match", "specialized"),
        ]),
        ("LEADS + decompose_match medgemma + cascade_leads_strict", [
            ("leads_native", "leads"),
            ("decompose_match", "specialized"),
            ("cascade_leads_strict", "leads"),
        ]),
        ("leads + medgemma + qwen2.5:7b (single_combined)", [
            ("single_combined", "leads"),
            ("single_combined", "specialized"),
            ("single_combined", "medium"),
        ]),
        ("decompose_match × 3 tiers", [
            ("decompose_match", "small"),
            ("decompose_match", "medium"),
            ("decompose_match", "specialized"),
        ]),
    ]
    for name, cells in bundles:
        avail = [(a, t) for a, t in cells if (a, t) in table]
        if len(avail) < 2:
            print(f"{name}: skipped — only {len(avail)} cells available")
            continue
        common: set[str] | None = None
        for c in avail:
            ids = set(table[c].keys())
            common = ids if common is None else common & ids
        if not common:
            print(f"{name}: skipped — no common paper_ids")
            continue
        common_sorted = sorted(common)
        rows = []
        for pid in common_sorted:
            label = table[avail[0]][pid][0]
            preds = [table[c][pid][1] for c in avail]
            confs = [table[c][pid][2] for c in avail]
            rows.append((label, preds, confs))
        cell_label = "+".join(f"{a}/{t}" for a, t in avail)
        print(f"\n  {name}: {cell_label}  (n={len(common_sorted)})")
        for rule in ("any", "majority", "all", "avg"):
            m = ensemble_metrics(rows, rule)
            print(f"  {'':>30}{rule:10s} {'':30s} {m['n']:4d} {m['tp']:3d} {m['fp']:4d} {m['fn']:3d} "
                  f"{m['recall']:7.3f} {m['specificity']:6.3f} {m['f1']:6.3f} {m['mcc']:+7.3f} {m['wss_at_95']:6.3f}")
            all_results.append({"scope": "curated", "architecture": name, "rule": rule, "tiers": cell_label, **m})

    # Write CSV
    out_csv = root / "reports" / "ensemble_results.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        for r in all_results:
            w.writerow(r)
    print(f"\nWrote {out_csv} ({len(all_results)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
