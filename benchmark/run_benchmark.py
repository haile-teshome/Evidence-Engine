"""Run the architecture × model × dataset matrix and write a comprehensive report.

Examples:
  # Smoke test
  python run_benchmark.py --datasets sample \
      --architectures single_combined cascade_triage \
      --models small medium

  # Real run with stability + stratification + stats
  python run_benchmark.py --datasets synergy/van_Dis_2020 \
      --architectures all --models small medium \
      --repeat 3 --workers 4 --field-stratify
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

from tabulate import tabulate
from tqdm import tqdm

from architectures import REGISTRY, ScreeningArchitecture
from architectures.base import Paper, ScreeningContext, ScreeningResult
from datasets.loader import Dataset, load_dataset
from field_tagger import tag_papers, CATEGORIES
from metrics import (
    Metrics, compute, cohens_kappa, fleiss_kappa, krippendorff_alpha_binary,
    pairwise_kappa, stability_score, mcnemar_test, holm_bonferroni,
    bootstrap_ci, metrics_by_stratum,
)
from models import DEFAULT_TIERS, build, is_available, ModelSpec


def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM screening architecture benchmark")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--architectures", nargs="+", required=True,
                    help=f"Architecture slugs or 'all'. Available: {', '.join(REGISTRY)}")
    ap.add_argument("--models", nargs="+", required=True,
                    help=f"Model tiers or 'all'. Available: {', '.join(DEFAULT_TIERS)}")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="Per-cell paper-level parallelism")
    ap.add_argument("--repeat", type=int, default=1,
                    help="How many times to re-run each cell for stability measurement")
    ap.add_argument("--field-stratify", action="store_true",
                    help="Tag each paper into a paper-type bucket and report metrics per bucket")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--run-id", default=None, help="Override timestamp run id (useful when launching multiple parallel processes that should write to the same folder)")
    return ap.parse_args(argv)


def _resolve(slugs: List[str], all_keys: Iterable[str]) -> List[str]:
    if "all" in slugs:
        return list(all_keys)
    return slugs


def run_cell(
    arch: ScreeningArchitecture, model, ctx: ScreeningContext, papers: List[Paper],
    workers: int, label: str,
) -> List[ScreeningResult]:
    results: List[ScreeningResult | None] = [None] * len(papers)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(arch.screen, p, ctx, model): i for i, p in enumerate(papers)}
            for fut in tqdm(as_completed(futures), total=len(futures), leave=False, desc=label[:40]):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    results[i] = ScreeningResult(paper_id=papers[i].paper_id, prediction=0, reasoning=f"ERROR: {e}")
    else:
        for i, p in enumerate(tqdm(papers, leave=False, desc=label[:40])):
            try:
                results[i] = arch.screen(p, ctx, model)
            except Exception as e:
                results[i] = ScreeningResult(paper_id=p.paper_id, prediction=0, reasoning=f"ERROR: {e}")
    return [r for r in results if r is not None]


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    arch_keys = _resolve(args.architectures, REGISTRY.keys())
    tier_keys = _resolve(args.models, DEFAULT_TIERS.keys())
    unknown = [k for k in arch_keys if k not in REGISTRY]
    if unknown:
        print(f"Unknown architectures: {unknown}. Available: {list(REGISTRY)}", file=sys.stderr)
        return 2

    # Datasets
    datasets: List[Dataset] = []
    for name in args.datasets:
        try:
            datasets.append(load_dataset(name, data_root=args.data_root, limit=args.limit))
            print(f"Loaded {name}: {len(datasets[-1].papers)} papers")
        except Exception as e:
            print(f"Skipping {name}: {e}", file=sys.stderr)
    if not datasets:
        return 2

    # Field tagging (per dataset, cached)
    field_tags: dict[str, dict[str, str]] = {}
    if args.field_stratify:
        for ds in datasets:
            dataset_dir = Path(args.data_root) / ds.name
            tags = tag_papers(ds.papers, dataset_dir, model=None)  # heuristic-only; cheap
            field_tags[ds.name] = tags
            counts: dict[str, int] = {}
            for v in tags.values():
                counts[v] = counts.get(v, 0) + 1
            print(f"  {ds.name} field tags: {counts}")

    # Models
    model_objs: dict[str, tuple[ModelSpec, object]] = {}
    for tier_key in tier_keys:
        spec = DEFAULT_TIERS.get(tier_key)
        if spec is None:
            continue
        ok, why = is_available(spec)
        if not ok:
            print(f"Skipping tier {tier_key} ({spec.name} via {spec.provider}): {why}")
            continue
        try:
            model_objs[tier_key] = (spec, build(spec))
        except Exception as e:
            print(f"Failed to build model {spec.name}: {e}", file=sys.stderr)
    if not model_objs:
        return 2

    # Output
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {out_dir}")

    preds_rows: list[dict] = []
    metric_rows: list[dict] = []
    # Predictions keyed by (dataset, arch, tier, repeat_index) for stability + interrater
    by_cell: dict[tuple[str, str, str, int], list[int]] = {}

    for ds in datasets:
        for arch_key in arch_keys:
            arch_cls = REGISTRY[arch_key]
            for tier_key, (spec, model) in model_objs.items():
                cell_label = f"{ds.name} | {arch_key} | {tier_key} ({spec.name})"
                for rep in range(args.repeat):
                    print(f"\n▶ {cell_label} | rep {rep + 1}/{args.repeat}")
                    arch = arch_cls()
                    t0 = time.time()
                    results = run_cell(arch, model, ds.context, ds.papers, args.workers, cell_label)
                    elapsed = time.time() - t0

                    y_pred = [r.prediction for r in results]
                    by_cell[(ds.name, arch_key, tier_key, rep)] = y_pred

                    for paper, res in zip(ds.papers, results):
                        preds_rows.append({
                            "dataset": ds.name,
                            "architecture": arch_key,
                            "model_tier": tier_key,
                            "model_name": spec.name,
                            "repeat": rep + 1,
                            "paper_id": paper.paper_id,
                            "title": paper.title,
                            "field": field_tags.get(ds.name, {}).get(paper.paper_id, ""),
                            "label": paper.label,
                            "prediction": res.prediction,
                            "confidence": res.confidence,
                            "llm_calls": res.llm_calls,
                            "wall_time_s": round(res.wall_time_s, 3),
                            "reasoning": (res.reasoning or "")[:500],
                        })

                    y_true = [int(p.label) if p.label is not None else 0 for p in ds.papers]
                    m = compute(y_true, y_pred, [r.confidence for r in results])
                    wt = sorted(r.wall_time_s for r in results)
                    p95 = wt[int(0.95 * len(wt))] if wt else 0.0
                    metric_rows.append({
                        "dataset": ds.name,
                        "architecture": arch_key,
                        "model_tier": tier_key,
                        "model_name": spec.name,
                        "repeat": rep + 1,
                        **m.as_dict(),
                        "avg_llm_calls_per_paper": round(sum(r.llm_calls for r in results) / max(len(results), 1), 2),
                        "avg_seconds_per_paper": round(sum(r.wall_time_s for r in results) / max(len(results), 1), 2),
                        "p95_seconds_per_paper": round(p95, 2),
                        "total_seconds": round(elapsed, 2),
                    })

    _write_report(out_dir, preds_rows, metric_rows, by_cell, datasets, field_tags, args)
    print(f"\n✅ Done. See {out_dir}/summary.md")
    return 0


def _write_report(
    out_dir: Path, preds_rows: list[dict], metric_rows: list[dict],
    by_cell: dict, datasets: List[Dataset], field_tags: dict, args,
) -> None:
    if not metric_rows:
        return

    # CSVs
    with open(out_dir / "predictions.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(preds_rows[0].keys()))
        writer.writeheader()
        writer.writerows(preds_rows)
    with open(out_dir / "metrics.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    # --- Aggregate across repeats for the headline table ---
    # Mean of each metric per (dataset, arch, tier).
    agg: dict[tuple, list[dict]] = {}
    for row in metric_rows:
        key = (row["dataset"], row["architecture"], row["model_tier"], row["model_name"])
        agg.setdefault(key, []).append(row)

    headline_rows = []
    NUMERIC_FIELDS = [
        "f1", "recall", "specificity", "precision", "mcc", "kappa",
        "accuracy", "wss_at_95", "avg_llm_calls_per_paper",
        "avg_seconds_per_paper", "p95_seconds_per_paper",
    ]
    for (dataset, arch, tier, model_name), rows in agg.items():
        rec = {"dataset": dataset, "architecture": arch, "model_tier": tier, "model_name": model_name, "repeats": len(rows)}
        for f in NUMERIC_FIELDS:
            vals = [r[f] for r in rows if r.get(f) is not None]
            rec[f] = sum(vals) / len(vals) if vals else 0.0
        headline_rows.append(rec)

    with open(out_dir / "metrics_aggregated.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(headline_rows[0].keys()))
        writer.writeheader()
        writer.writerows(headline_rows)

    # --- Stability per (dataset, arch, tier) when --repeat > 1 ---
    stability: dict[tuple, dict] = {}
    repeats_by_cell: dict[tuple, list[list[int]]] = {}
    for (dataset, arch, tier, _rep), preds in by_cell.items():
        repeats_by_cell.setdefault((dataset, arch, tier), []).append(preds)
    for key, reps in repeats_by_cell.items():
        if len(reps) > 1:
            stability[key] = stability_score(reps)

    # --- Bootstrap CIs + Pairwise McNemar across architectures (within tier × dataset) ---
    # Use the first repeat for these tests to keep it deterministic.
    boot_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    kappa_panel_rows: list[dict] = []
    field_rows: list[dict] = []

    for dataset in {row["dataset"] for row in metric_rows}:
        ds = next(d for d in datasets if d.name == dataset)
        y_true = [int(p.label) if p.label is not None else 0 for p in ds.papers]

        # Group preds by tier
        tiers_seen = {row["model_tier"] for row in metric_rows if row["dataset"] == dataset}
        for tier in tiers_seen:
            preds_per_arch: dict[str, list[int]] = {}
            for arch in {row["architecture"] for row in metric_rows if row["dataset"] == dataset and row["model_tier"] == tier}:
                # Use repeat=1
                preds = by_cell.get((dataset, arch, tier, 0))
                if preds:
                    preds_per_arch[arch] = preds

            # Bootstrap CIs on F1 + recall per arch
            for arch, preds in preds_per_arch.items():
                f1_p, f1_lo, f1_hi = bootstrap_ci(y_true, preds, metric_fn=lambda yt, yp: compute(yt, yp).f1)
                r_p, r_lo, r_hi = bootstrap_ci(y_true, preds, metric_fn=lambda yt, yp: compute(yt, yp).recall)
                boot_rows.append({
                    "dataset": dataset, "architecture": arch, "model_tier": tier,
                    "f1": round(f1_p, 3), "f1_ci_low": round(f1_lo, 3), "f1_ci_high": round(f1_hi, 3),
                    "recall": round(r_p, 3), "recall_ci_low": round(r_lo, 3), "recall_ci_high": round(r_hi, 3),
                })

            # Pairwise McNemar
            if len(preds_per_arch) >= 2:
                pvals: dict[tuple[str, str], float] = {}
                for a, ap in preds_per_arch.items():
                    for b, bp in preds_per_arch.items():
                        if a >= b:
                            continue
                        chi2, p = mcnemar_test(ap, bp, y_true)
                        pvals[(a, b)] = p
                rejected = holm_bonferroni(pvals)
                for (a, b), p in pvals.items():
                    pairwise_rows.append({
                        "dataset": dataset, "model_tier": tier, "a": a, "b": b,
                        "p_value": round(p, 5), "significant_holm": rejected[(a, b)],
                    })

                # Fleiss + Krippendorff across architectures
                items = list(zip(*[preds_per_arch[k] for k in sorted(preds_per_arch)]))
                kappa_panel_rows.append({
                    "dataset": dataset, "model_tier": tier,
                    "n_raters": len(preds_per_arch),
                    "fleiss_kappa": round(fleiss_kappa([list(r) for r in items]), 4),
                    "krippendorff_alpha": round(krippendorff_alpha_binary([list(r) for r in items]), 4),
                })

            # Field stratification
            tags = field_tags.get(dataset, {})
            if tags:
                strata = [tags.get(p.paper_id, "other") for p in ds.papers]
                for arch, preds in preds_per_arch.items():
                    bucket_metrics = metrics_by_stratum(y_true, preds, strata)
                    for stratum, m in bucket_metrics.items():
                        field_rows.append({
                            "dataset": dataset, "model_tier": tier, "architecture": arch,
                            "field": stratum, **m.as_dict(),
                        })

    if boot_rows:
        with open(out_dir / "bootstrap_ci.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(boot_rows[0].keys())); writer.writeheader(); writer.writerows(boot_rows)
    if pairwise_rows:
        with open(out_dir / "pairwise_mcnemar.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(pairwise_rows[0].keys())); writer.writeheader(); writer.writerows(pairwise_rows)
    if kappa_panel_rows:
        with open(out_dir / "interrater_panel.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(kappa_panel_rows[0].keys())); writer.writeheader(); writer.writerows(kappa_panel_rows)
    if field_rows:
        with open(out_dir / "field_stratified.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(field_rows[0].keys())); writer.writeheader(); writer.writerows(field_rows)
    if stability:
        rows = [{
            "dataset": k[0], "architecture": k[1], "model_tier": k[2],
            "consistency": round(v["consistency"], 4),
            "mean_pairwise_kappa": round(v["mean_pairwise_kappa"], 4),
        } for k, v in stability.items()]
        with open(out_dir / "stability.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)

    # --- Markdown summary ---
    lines = [f"# Benchmark report — {out_dir.name}", "", f"Args: `{vars(args)}`", ""]

    for ds_name in sorted({r["dataset"] for r in headline_rows}):
        lines.append(f"## Dataset: {ds_name}")
        lines.append("")
        dsrows = [r for r in headline_rows if r["dataset"] == ds_name]
        tbl = [
            {
                "Architecture": r["architecture"],
                "Model": f"{r['model_tier']} ({r['model_name']})",
                "F1": f"{r['f1']:.3f}",
                "Recall": f"{r['recall']:.3f}",
                "Spec.": f"{r['specificity']:.3f}",
                "Prec.": f"{r['precision']:.3f}",
                "MCC": f"{r['mcc']:.3f}",
                "Kappa": f"{r['kappa']:.3f}",
                "Acc.": f"{r['accuracy']:.3f}",
                "WSS@95": f"{r['wss_at_95']:.3f}",
                "LLM/p": r["avg_llm_calls_per_paper"],
                "s/p": r["avg_seconds_per_paper"],
                "p95 s/p": r["p95_seconds_per_paper"],
                "n_rep": r["repeats"],
            }
            for r in sorted(dsrows, key=lambda x: (-x["recall"], -x["f1"]))
        ]
        lines.append(tabulate(tbl, headers="keys", tablefmt="github")); lines.append("")

        if field_rows:
            lines.append("### Field stratification")
            field_df = [r for r in field_rows if r["dataset"] == ds_name]
            if field_df:
                tbl = [
                    {
                        "Architecture": r["architecture"], "Tier": r["model_tier"], "Field": r["field"],
                        "n": r["n"], "F1": f"{r['f1']:.3f}", "Recall": f"{r['recall']:.3f}",
                        "Spec.": f"{r['specificity']:.3f}", "MCC": f"{r['mcc']:.3f}",
                    }
                    for r in sorted(field_df, key=lambda x: (x["architecture"], x["model_tier"], x["field"]))
                ]
                lines.append(tabulate(tbl, headers="keys", tablefmt="github")); lines.append("")

        if pairwise_rows:
            lines.append("### Pairwise McNemar (Holm-Bonferroni corrected)")
            pw = [r for r in pairwise_rows if r["dataset"] == ds_name]
            if pw:
                tbl = [
                    {
                        "Tier": r["model_tier"], "A": r["a"], "B": r["b"],
                        "p": r["p_value"],
                        "sig (α=0.05)": "✓" if r["significant_holm"] else "—",
                    }
                    for r in sorted(pw, key=lambda x: (x["model_tier"], x["p_value"]))
                ]
                lines.append(tabulate(tbl, headers="keys", tablefmt="github")); lines.append("")

        if kappa_panel_rows:
            lines.append("### Interrater agreement across architectures")
            kp = [r for r in kappa_panel_rows if r["dataset"] == ds_name]
            if kp:
                tbl = [{"Tier": r["model_tier"], "Raters": r["n_raters"],
                        "Fleiss κ": r["fleiss_kappa"], "Krippendorff α": r["krippendorff_alpha"]}
                       for r in sorted(kp, key=lambda x: x["model_tier"])]
                lines.append(tabulate(tbl, headers="keys", tablefmt="github")); lines.append("")

        if stability:
            stab = [{**{"dataset": k[0], "arch": k[1], "tier": k[2]}, **v} for k, v in stability.items() if k[0] == ds_name]
            if stab:
                lines.append("### Stability across repeats")
                tbl = [{"Architecture": r["arch"], "Tier": r["tier"],
                        "Consistency": f"{r['consistency']:.3f}", "Mean pairwise κ": f"{r['mean_pairwise_kappa']:.3f}"}
                       for r in sorted(stab, key=lambda x: (x["arch"], x["tier"]))]
                lines.append(tabulate(tbl, headers="keys", tablefmt="github")); lines.append("")

        if boot_rows:
            lines.append("### Bootstrap 95% CIs (F1, Recall)")
            bs = [r for r in boot_rows if r["dataset"] == ds_name]
            if bs:
                tbl = [{
                    "Architecture": r["architecture"], "Tier": r["model_tier"],
                    "F1 (95% CI)": f"{r['f1']:.3f} [{r['f1_ci_low']:.3f}–{r['f1_ci_high']:.3f}]",
                    "Recall (95% CI)": f"{r['recall']:.3f} [{r['recall_ci_low']:.3f}–{r['recall_ci_high']:.3f}]",
                } for r in sorted(bs, key=lambda x: (x["model_tier"], x["architecture"]))]
                lines.append(tabulate(tbl, headers="keys", tablefmt="github")); lines.append("")

    with open(out_dir / "summary.md", "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
