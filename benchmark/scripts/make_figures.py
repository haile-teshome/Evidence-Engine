"""Generate all visualizations for the 5-experiment report.

Reads existing CSVs / predictions and writes PNGs to reports/figures/.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from architectures.leads_native_fewshot import EXEMPLAR_IDS  # noqa: E402
from metrics import compute  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def find_latest(prefix: str) -> Path | None:
    """Return the most recent reports/<prefix>*/predictions.csv path, or None."""
    matches = sorted((ROOT / "reports").glob(f"{prefix}*/predictions.csv"))
    return matches[-1] if matches else None


# ============================================================================
# Common loaders
# ============================================================================

def load_predictions(path: Path) -> list[dict]:
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("label") in ("", None):
                continue
            try:
                r["label_i"] = int(r["label"])
                r["pred_i"] = int(r["prediction"])
                r["conf_f"] = float(r["confidence"])
                out.append(r)
            except (ValueError, TypeError):
                continue
    return out


def sweep_metrics(rows: list[dict], thresholds: np.ndarray) -> dict:
    y_true = [r["label_i"] for r in rows]
    confs = [r["conf_f"] for r in rows]
    recalls, specs, f1s, mccs, utils, tps, fps = [], [], [], [], [], [], []
    for t in thresholds:
        y_pred = [1 if c >= t else 0 for c in confs]
        m = compute(y_true, y_pred, confs)
        recalls.append(m.recall); specs.append(m.specificity)
        f1s.append(m.f1); mccs.append(m.mcc)
        utils.append(2 * m.recall + m.specificity)
        tps.append(m.tp); fps.append(m.fp)
    return {"recall": recalls, "spec": specs, "f1": f1s, "mcc": mccs,
            "util": utils, "tp": tps, "fp": fps, "thresholds": thresholds}


# ============================================================================
# Figure 1 — Threshold sweep on van_Dis_2020 (the foundational result)
# ============================================================================

def fig_threshold_sweep_van_dis():
    base = find_latest("leads-local-")
    if base is None:
        print("(skip fig 1) no reports/leads-local-* run found"); return
    rows = [r for r in load_predictions(base)
            if r["architecture"] == "leads_native" and r["model_tier"] == "leads"]
    ts = np.linspace(0, 1, 21)
    s = sweep_metrics(rows, ts)
    leads_scores = 2 * ts - 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(leads_scores, s["recall"], "-o", color="#1f77b4", label="Recall", lw=2, ms=4)
    ax.plot(leads_scores, s["spec"], "-s", color="#d62728", label="Specificity", lw=2, ms=4)
    ax.plot(leads_scores, s["f1"], "-^", color="#2ca02c", label="F1", lw=2, ms=4)
    ax.axvline(0.20, color="black", ls="--", alpha=0.5, lw=1)
    ax.text(0.21, 0.05, "sweet spot\n(LEADS ≥ +0.20)", fontsize=9, va="bottom")
    ax.set_xlabel("LEADS aggregate score threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("van_Dis_2020 — threshold sweep (leads_native × leads, n=288)")
    ax.legend(loc="center left")
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1]
    ax.plot(leads_scores, s["tp"], "-o", color="#2ca02c", label="True positives", lw=2, ms=4)
    ax.plot(leads_scores, s["fp"], "-s", color="#d62728", label="False positives", lw=2, ms=4)
    ax.axhline(10, color="#2ca02c", ls=":", alpha=0.4, lw=1)
    ax.text(-0.95, 12, "n_positives = 10", fontsize=8, color="#2ca02c")
    ax.axvline(0.20, color="black", ls="--", alpha=0.5, lw=1)
    ax.set_xlabel("LEADS aggregate score threshold")
    ax.set_ylabel("Count (papers, n=288)")
    ax.set_title("Confusion-matrix counts vs. threshold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_threshold_sweep_van_dis.png")
    plt.close()
    print(f"wrote {FIG_DIR.name}/01_threshold_sweep_van_dis.png")


# ============================================================================
# Figure 2 — Cross-model ensembling: recall × specificity scatter
# ============================================================================

def fig_ensemble_scatter():
    """Horizontal-bar ranked view: top ensembles by utility (2*recall + spec),
    with recall/spec/F1 broken out side by side so the trade-offs are visible
    at a glance. Adds the threshold-tuned baseline as a reference row."""
    rows = []
    with open(ROOT / "reports/ensemble_results.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            for k in ("recall", "specificity", "f1", "mcc"):
                r[k] = float(r[k])
            r["util"] = 2 * r["recall"] + r["specificity"]
            # Filter out the silly "rule=avg, tier=specialized" types where confidence is meaningless
            rows.append(r)

    # Inject the threshold-tuned reference as a synthetic row
    reference = {"scope": "REF", "architecture": "leads_native @ +0.20", "rule": "—",
                 "tiers": "leads", "recall": 1.000, "specificity": 0.676, "f1": 0.100, "mcc": 0.260,
                 "util": 2.676}

    # Pick top 12 by recall ≥ 0.7 AND specificity ≥ 0.3 (i.e., points that aren't degenerate corners)
    ranked = sorted([r for r in rows if r["recall"] >= 0.7 and r["specificity"] >= 0.3],
                    key=lambda r: -r["util"])[:12]
    ranked.insert(0, reference)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # === LEFT PANEL: horizontal bars, three metrics side-by-side ===
    ax = axes[0]
    n = len(ranked)
    y = np.arange(n)[::-1]   # reverse so highest util is on top
    h = 0.27

    def short_label(r):
        scope = r["scope"]
        arch = r["architecture"][:40] + ("…" if len(r["architecture"]) > 40 else "")
        tier = r["tiers"]
        rule = r["rule"]
        if scope == "REF":
            return f"★ {arch}"
        if scope == "per-arch":
            return f"{arch} × cross-tier  ({rule})"
        if scope == "per-tier":
            return f"all archs × {tier}  ({rule})"
        return f"{arch}  ({rule})"

    labels = [short_label(r) for r in ranked]
    recalls = [r["recall"] for r in ranked]
    specs = [r["specificity"] for r in ranked]
    f1s = [r["f1"] for r in ranked]

    ax.barh(y + h, recalls, h, color="#1f77b4", label="Recall", edgecolor="black", linewidth=0.4)
    ax.barh(y, specs, h, color="#d62728", label="Specificity", edgecolor="black", linewidth=0.4)
    ax.barh(y - h, f1s, h, color="#2ca02c", label="F1", edgecolor="black", linewidth=0.4)

    # Highlight reference row
    ax.axhspan(y[0] - 1.5*h, y[0] + 1.5*h, color="gold", alpha=0.15, zorder=0)

    # Annotate values at bar end
    for yi, v in zip(y + h, recalls):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8)
    for yi, v in zip(y, specs):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8)
    for yi, v in zip(y - h, f1s):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Metric value")
    ax.set_xlim(0, 1.15)
    ax.set_title("Top 12 ensembles by utility (2·recall + spec) vs. reference (★)\nfiltered to non-degenerate operating points")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3); ax.grid(axis="y", visible=False)
    # y already runs top→best because we used np.arange(n)[::-1] descending

    # === RIGHT PANEL: scatter with only the labeled points + Pareto frontier ===
    ax = axes[1]
    all_pts = [(r["recall"], r["specificity"], short_label(r), r) for r in rows]
    # All other points (low-importance) in gray
    ax.scatter([p[0] for p in all_pts], [p[1] for p in all_pts],
               s=24, color="lightgray", alpha=0.6, zorder=1)

    # Highlighted ranked points
    colors_top = plt.cm.viridis(np.linspace(0, 1, len(ranked)))
    for i, r in enumerate(ranked):
        if r["scope"] == "REF":
            ax.scatter(r["recall"], r["specificity"], s=300, marker="*",
                       color="gold", edgecolor="black", linewidth=1.5, zorder=10,
                       label=f"★ ref: {r['architecture']}")
        else:
            ax.scatter(r["recall"], r["specificity"], s=120,
                       color=colors_top[i], edgecolor="black", linewidth=0.6, zorder=5)
            ax.annotate(f"{i}", (r["recall"], r["specificity"]),
                        fontsize=8, fontweight="bold", color="white",
                        ha="center", va="center", zorder=11)

    # Pareto frontier (non-dominated): sort by recall asc, keep running max spec
    pts_sorted = sorted([(p["recall"], p["specificity"]) for p in rows], key=lambda x: -x[0])
    frontier = []
    best_spec = -1
    for rec, sp in pts_sorted:
        if sp > best_spec:
            frontier.append((rec, sp))
            best_spec = sp
    frontier.sort()
    ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
            color="black", lw=1.2, ls="--", alpha=0.4, label="Pareto frontier")

    ax.set_xlabel("Recall"); ax.set_ylabel("Specificity")
    ax.set_xlim(-0.02, 1.05); ax.set_ylim(-0.02, 1.05)
    ax.set_title("Same 12 ensembles in (recall, specificity) space\n(numbered, color-coded by rank)")
    ax.legend(loc="lower left", fontsize=9)

    plt.suptitle("Experiment 2 — Cross-model ensembling (80 configurations summarized)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_ensemble_recall_spec_scatter.png", bbox_inches="tight")
    plt.close()
    print(f"wrote {FIG_DIR.name}/02_ensemble_recall_spec_scatter.png")


# ============================================================================
# Figure 3 — Cost-recall curves (papers reviewed vs recall achieved)
# ============================================================================

def fig_cost_recall_curves():
    """Aggregate every (arch, tier) cell found under reports/ and plot the
    cost-recall curve for each. Auto-discovers all predictions.csv files."""
    cells_data: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted((ROOT / "reports").glob("*/predictions.csv")):
        for r in load_predictions(path):
            cells_data[(r["architecture"], r["model_tier"])].append(r)

    cells_of_interest = [(arch, tier, f"{arch} × {tier}") for (arch, tier) in sorted(cells_data)]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(cells_of_interest), 1)))

    for (arch, tier, label), col in zip(cells_of_interest, palette):
        rows = cells_data[(arch, tier)]
        if not rows:
            continue
        ranked = sorted(rows, key=lambda r: -r["conf_f"])
        n = len(ranked)
        total_pos = sum(r["label_i"] for r in ranked)
        if not total_pos:
            continue
        xs, ys = [0], [0]
        found = 0
        for i, r in enumerate(ranked, 1):
            if r["label_i"] == 1:
                found += 1
            xs.append(i / n)
            ys.append(found / total_pos)
        ax.plot(xs, ys, lw=2, color=col, label=label, alpha=0.9)

    # Reference: random
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1, label="random ordering")
    # Reference: perfect
    ax.plot([0, total_pos / n if n else 0, 1], [0, 1, 1],
            color="black", ls="--", lw=1, alpha=0.4, label="perfect oracle")

    ax.axhline(0.95, color="red", ls=":", lw=1, alpha=0.4)
    ax.text(0.01, 0.96, "95% recall floor", fontsize=8, color="red")
    ax.set_xlabel("Fraction of papers reviewed (descending confidence)")
    ax.set_ylabel("Recall (fraction of true positives captured)")
    ax.set_title("Cost-recall curves on van_Dis_2020 (n=288, P=10)\nlower-and-leftward = better (less reading to capture more positives)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_cost_recall_curves.png")
    plt.close()
    print(f"wrote {FIG_DIR.name}/03_cost_recall_curves.png")


# ============================================================================
# Figure 4 — Few-shot LEADS vs baseline LEADS
# ============================================================================

def fig_fewshot_vs_baseline():
    base = find_latest("leads-local-")
    fs_path = find_latest("leads-fewshot-")
    if base is None or fs_path is None:
        print("(skip fig 4) need both leads-local-* and leads-fewshot-* runs"); return
    baseline = [r for r in load_predictions(base)
                if r["architecture"] == "leads_native" and r["model_tier"] == "leads"
                and r["paper_id"] not in EXEMPLAR_IDS]
    fewshot = [r for r in load_predictions(fs_path)
               if r["architecture"] == "leads_native_fewshot" and r["model_tier"] == "leads"
               and r["paper_id"] not in EXEMPLAR_IDS]

    ts = np.linspace(0, 1, 21)
    bs = sweep_metrics(baseline, ts)
    fs = sweep_metrics(fewshot, ts)
    leads_scores = 2 * ts - 1

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    ax.plot(leads_scores, bs["recall"], "-o", label="baseline", color="#1f77b4", lw=2, ms=4)
    ax.plot(leads_scores, fs["recall"], "-s", label="few-shot", color="#ff7f0e", lw=2, ms=4)
    ax.set_xlabel("LEADS threshold"); ax.set_ylabel("Recall")
    ax.set_title("Recall (282-paper holdout)")
    ax.set_ylim(-0.05, 1.05); ax.legend()

    ax = axes[1]
    ax.plot(leads_scores, bs["spec"], "-o", label="baseline", color="#1f77b4", lw=2, ms=4)
    ax.plot(leads_scores, fs["spec"], "-s", label="few-shot", color="#ff7f0e", lw=2, ms=4)
    ax.set_xlabel("LEADS threshold"); ax.set_ylabel("Specificity")
    ax.set_title("Specificity (282-paper holdout)")
    ax.set_ylim(-0.05, 1.05); ax.legend()

    # Confidence histogram by gold class
    ax = axes[2]
    bins = np.linspace(0, 1, 21)
    bl_pos = [r["conf_f"] for r in baseline if r["label_i"] == 1]
    bl_neg = [r["conf_f"] for r in baseline if r["label_i"] == 0]
    fs_pos = [r["conf_f"] for r in fewshot if r["label_i"] == 1]
    fs_neg = [r["conf_f"] for r in fewshot if r["label_i"] == 0]
    ax.hist(bl_neg, bins=bins, alpha=0.45, label="baseline negatives", color="#1f77b4")
    ax.hist(fs_neg, bins=bins, alpha=0.45, label="few-shot negatives", color="#ff7f0e")
    ax.scatter(bl_pos, [-2]*len(bl_pos), marker="^", color="#1f77b4", s=80, label="baseline positives")
    ax.scatter(fs_pos, [-4]*len(fs_pos), marker="v", color="#ff7f0e", s=80, label="few-shot positives")
    ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Count")
    ax.set_title("Confidence distribution by gold class")
    ax.legend(fontsize=8, loc="upper right")

    plt.suptitle("Experiment 4 — few-shot LEADS vs baseline LEADS\n(no statistically meaningful improvement; 22 flips, all on gold-negatives)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_fewshot_vs_baseline.png", bbox_inches="tight")
    plt.close()
    print(f"wrote {FIG_DIR.name}/04_fewshot_vs_baseline.png")


# ============================================================================
# Figure 5 — External validity: per-dataset threshold sweeps + summary
# ============================================================================

def fig_external_validity():
    base = find_latest("leads-local-")
    ext = find_latest("external-validity-")
    if base is None or ext is None:
        print("(skip fig 5) need both leads-local-* and external-validity-* runs"); return
    sources = {
        "van_Dis_2020\n(baseline)":                         (base, "leads_native", "leads", None),
        "Sep_2021\n(rat object recognition)":               (ext,  "leads_native", "leads", "synergy/Sep_2021"),
        "Bannach-Brown_2019\n(rodent depression models)":   (ext,  "leads_native", "leads", "synergy/Bannach-Brown_2019"),
        "Muthu_2021\n(lumbar spine surgery)":               (ext,  "leads_native", "leads", "synergy/Muthu_2021"),
    }
    ts = np.linspace(0, 1, 21)
    leads_scores = 2 * ts - 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors_for = {"Recall": "#1f77b4", "Specificity": "#d62728", "F1": "#2ca02c", "MCC": "#9467bd"}

    for ax, (name, (path, arch, tier, ds)) in zip(axes.flat, sources.items()):
        rows = [r for r in load_predictions(path)
                if r["architecture"] == arch and r["model_tier"] == tier
                and (ds is None or r.get("dataset") == ds)]
        s = sweep_metrics(rows, ts)
        total_pos = sum(1 for r in rows if r["label_i"] == 1)
        ax.plot(leads_scores, s["recall"], "-o", color=colors_for["Recall"], label="Recall", lw=2, ms=3)
        ax.plot(leads_scores, s["spec"], "-s", color=colors_for["Specificity"], label="Specificity", lw=2, ms=3)
        ax.plot(leads_scores, s["f1"], "-^", color=colors_for["F1"], label="F1", lw=2, ms=3)
        ax.plot(leads_scores, s["mcc"], "-v", color=colors_for["MCC"], label="MCC", lw=2, ms=3)
        # Best threshold by util
        best_i = int(np.argmax(s["util"]))
        ax.axvline(leads_scores[best_i], color="black", ls="--", alpha=0.4)
        ax.text(leads_scores[best_i] + 0.02, -0.15,
                f"best t = {leads_scores[best_i]:+.2f}",
                fontsize=8, ha="left")
        ax.set_xlabel("LEADS aggregate score threshold")
        ax.set_ylabel("Metric value")
        ax.set_title(f"{name}  (n={len(rows)}, P={total_pos})")
        ax.set_ylim(-0.25, 1.05)
        ax.legend(fontsize=8, loc="lower left", ncol=2)

    plt.suptitle("Experiment 1 — External validity: LEADS-native × leads on 4 SR domains\nOptimal threshold varies +0.20 to +0.60 across topics",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_external_validity_sweeps.png", bbox_inches="tight")
    plt.close()
    print(f"wrote {FIG_DIR.name}/05_external_validity_sweeps.png")

    # Companion summary bar chart
    rows_summary = []
    for name, (path, arch, tier, ds) in sources.items():
        rows = [r for r in load_predictions(path)
                if r["architecture"] == arch and r["model_tier"] == tier
                and (ds is None or r.get("dataset") == ds)]
        s = sweep_metrics(rows, ts)
        # @ +0.20 → t=0.60 in confidence
        idx_020 = int(np.argmin(np.abs(leads_scores - 0.20)))
        best_i = int(np.argmax(s["util"]))
        rows_summary.append({
            "name": name, "n": len(rows),
            "P": sum(1 for r in rows if r["label_i"] == 1),
            "recall_020": s["recall"][idx_020], "spec_020": s["spec"][idx_020],
            "f1_020": s["f1"][idx_020], "mcc_020": s["mcc"][idx_020],
            "best_t": leads_scores[best_i],
            "best_recall": s["recall"][best_i], "best_spec": s["spec"][best_i],
            "best_f1": s["f1"][best_i],
        })

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(rows_summary))
    w = 0.18
    ax.bar(x - 1.5*w, [r["recall_020"] for r in rows_summary], w, label="recall @ +0.20", color="#1f77b4")
    ax.bar(x - 0.5*w, [r["spec_020"]   for r in rows_summary], w, label="spec @ +0.20",   color="#d62728")
    ax.bar(x + 0.5*w, [r["f1_020"]     for r in rows_summary], w, label="F1 @ +0.20",     color="#2ca02c")
    ax.bar(x + 1.5*w, [r["best_t"]/1.0  for r in rows_summary], w, label="best threshold (LEADS-score units)", color="#ff7f0e")
    # second axis for threshold value
    for i, r in enumerate(rows_summary):
        ax.text(i + 1.5*w, r["best_t"] + 0.01, f"{r['best_t']:+.2f}", ha="center", fontsize=9, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in rows_summary], fontsize=9)
    ax.set_ylabel("Value")
    ax.set_title("External validity summary: performance at +0.20 threshold + per-dataset best threshold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05b_external_validity_summary_bars.png", bbox_inches="tight")
    plt.close()
    print(f"wrote {FIG_DIR.name}/05b_external_validity_summary_bars.png")


# ============================================================================
# Figure 6 — Stability heatmap (rep × paper)
# ============================================================================

def fig_stability_heatmap():
    """Three-panel stability evidence:
       (a) Per-rep confusion-matrix counts as identical stacked bars
       (b) 4×4 pairwise Cohen's κ heatmap (all = +1.00)
       (c) Confidence distribution overlaid for all 4 reps (identical)"""
    base = find_latest("leads-local-")
    stab = find_latest("stability-leads-")
    if base is None or stab is None:
        print("(skip fig 6) need both leads-local-* and stability-leads-* runs"); return
    sources = [(base, 0), (stab, 10)]
    # Build per-rep prediction arrays
    by_rep: dict[int, list[dict]] = defaultdict(list)
    for path, offset in sources:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("architecture") != "leads_native" or r.get("model_tier") != "leads":
                    continue
                if r.get("label") in ("", None):
                    continue
                try:
                    rep = int(r["repeat"]) + offset
                    by_rep[rep].append({
                        "paper_id": r["paper_id"],
                        "label": int(r["label"]),
                        "pred": int(r["prediction"]),
                        "conf": float(r["confidence"]),
                    })
                except (ValueError, TypeError):
                    continue
    reps = sorted(by_rep)
    rep_labels = ["rep 1\n(original)"] + [f"rep {i + 2}\n(stability rerun)" for i in range(len(reps) - 1)]
    rep_short = ["rep 1 (orig)"] + [f"rep {i + 2} (rerun)" for i in range(len(reps) - 1)]

    # Per-rep metrics
    metrics_per_rep = []
    for r in reps:
        rows = by_rep[r]
        y_true = [x["label"] for x in rows]
        y_pred = [x["pred"] for x in rows]
        m = compute(y_true, y_pred, [x["conf"] for x in rows])
        metrics_per_rep.append({"rep": r, "tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn,
                                "recall": m.recall, "spec": m.specificity, "f1": m.f1, "mcc": m.mcc})

    fig = plt.figure(figsize=(15, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.85, 1.2])

    # ============ Panel A: per-rep confusion-matrix stacked bars ============
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(reps))
    tp = np.array([m["tp"] for m in metrics_per_rep])
    fn = np.array([m["fn"] for m in metrics_per_rep])
    fp = np.array([m["fp"] for m in metrics_per_rep])
    tn = np.array([m["tn"] for m in metrics_per_rep])
    # Stacked: TP (green) + FN (orange) + FP (red) + TN (gray)
    ax.bar(x, tp, color="#2ca02c", label=f"TP (recall-good): {tp[0]}", edgecolor="black", lw=0.5)
    ax.bar(x, fn, bottom=tp, color="#ff7f0e", label=f"FN (missed): {fn[0]}", edgecolor="black", lw=0.5)
    ax.bar(x, fp, bottom=tp + fn, color="#d62728", label=f"FP (false alarm): {fp[0]}", edgecolor="black", lw=0.5)
    ax.bar(x, tn, bottom=tp + fn + fp, color="#cccccc", label=f"TN (spec-good): {tn[0]}", edgecolor="black", lw=0.5)
    # Annotate identical totals
    for xi in x:
        ax.text(xi, tp[xi]/2, str(tp[xi]), ha="center", va="center", fontsize=9, fontweight="bold")
        if fn[xi] > 0:
            ax.text(xi, tp[xi] + fn[xi]/2, str(fn[xi]), ha="center", va="center", fontsize=9)
        ax.text(xi, tp[xi] + fn[xi] + fp[xi]/2, str(fp[xi]), ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        ax.text(xi, tp[xi] + fn[xi] + fp[xi] + tn[xi]/2, str(tn[xi]), ha="center", va="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(rep_labels, fontsize=9)
    ax.set_ylabel("Papers (n=288)")
    ax.set_title("A — Confusion counts per replicate\n(all 4 bars identical → bit-for-bit deterministic)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 320)

    # ============ Panel B: pairwise Cohen's κ heatmap ============
    ax = fig.add_subplot(gs[0, 1])
    # Compute pairwise kappa
    from metrics import cohens_kappa
    rep_preds = {r: [x["pred"] for x in by_rep[r]] for r in reps}
    K = np.eye(len(reps))
    for i, ri in enumerate(reps):
        for j, rj in enumerate(reps):
            if i != j:
                K[i, j] = cohens_kappa(rep_preds[ri], rep_preds[rj])
    im = ax.imshow(K, cmap="RdYlGn", vmin=0.5, vmax=1.0)
    for i in range(len(reps)):
        for j in range(len(reps)):
            ax.text(j, i, f"{K[i,j]:+.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="black")
    ax.set_xticks(np.arange(len(reps))); ax.set_xticklabels(rep_short, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(reps))); ax.set_yticklabels(rep_short, fontsize=9)
    ax.set_title("B — Pairwise Cohen's κ\nall pairs = +1.00 (perfect agreement)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="κ")
    ax.grid(False)

    # ============ Panel C: confidence histogram overlaid ============
    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(0, 1, 21)
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(reps)))
    for r, col, name in zip(reps, palette, rep_short):
        confs = [x["conf"] for x in by_rep[r]]
        ax.hist(confs, bins=bins, histtype="step", lw=2, color=col,
                label=name)
    ax.axvline(0.6, color="black", ls="--", alpha=0.4)
    ax.text(0.61, ax.get_ylim()[1] * 0.92, "+0.20 threshold", fontsize=8, va="top")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Paper count")
    ax.set_title("C — Confidence distribution per replicate\n(all 4 histograms superimpose exactly)")
    ax.legend(fontsize=8, loc="upper left")

    plt.suptitle("Experiment 3 — Stability of leads_native × leads on van_Dis_2020  "
                 f"(n=288, 4 reps, Fleiss κ = +1.00, 100% per-paper agreement)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_stability_heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"wrote {FIG_DIR.name}/06_stability_heatmap.png")


# ============================================================================
# Figure 7 — Overall summary: per-cell recall × specificity scatter
# ============================================================================

def fig_overall_scatter():
    """All measured cells on van_Dis at default threshold.

    Two-panel: (left) scatter with smart label offsets via adjustText-style
    manual placement; (right) horizontal-bar leaderboard ranked by utility.
    """
    cells = defaultdict(list)
    for path in sorted((ROOT / "reports").glob("*/predictions.csv")):
        for r in load_predictions(path):
            cells[(r["architecture"], r["model_tier"])].append(r)

    points = []
    for (arch, tier), rows in cells.items():
        y_true = [r["label_i"] for r in rows]
        y_pred = [r["pred_i"] for r in rows]
        m = compute(y_true, y_pred, [r["conf_f"] for r in rows])
        points.append({
            "arch": arch, "tier": tier,
            "recall": m.recall, "spec": m.specificity, "f1": m.f1, "mcc": m.mcc,
            "util": 2 * m.recall + m.specificity, "n": len(rows),
        })
    # Synthetic reference: threshold-tuned LEADS
    ref = {"arch": "leads_native @ +0.20", "tier": "leads",
           "recall": 1.000, "spec": 0.676, "f1": 0.100, "mcc": 0.260, "util": 2.676, "n": 288}

    tier_colors = {"leads": "#d62728", "small": "#1f77b4", "medium": "#ff7f0e",
                   "specialized": "#2ca02c", "large": "#9467bd", "leading": "#8c564b"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # === LEFT: scatter with numbered points + numbered legend (no label overlap) ===
    ax = axes[0]
    # Only label top-12 by utility; rest just get colored dots
    points_ranked = sorted(points, key=lambda p: -p["util"])
    top_n = 12
    top = points_ranked[:top_n]
    rest = points_ranked[top_n:]

    # Background dots (un-labeled, faded)
    for p in rest:
        ax.scatter(p["recall"], p["spec"], s=70,
                   c=tier_colors.get(p["tier"], "gray"),
                   alpha=0.35, edgecolor="black", linewidth=0.4, zorder=2)
    # Top-12 dots: bigger, numbered
    for i, p in enumerate(top, start=1):
        ax.scatter(p["recall"], p["spec"], s=170,
                   c=tier_colors.get(p["tier"], "gray"),
                   alpha=0.92, edgecolor="black", linewidth=0.8, zorder=4)
        ax.annotate(str(i), (p["recall"], p["spec"]),
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white", zorder=11)
    # Reference star
    ax.scatter(ref["recall"], ref["spec"], s=520, marker="*", c="gold",
               edgecolor="black", linewidth=1.6, zorder=12,
               label="★ leads_native × leads @ +0.20 (threshold-tuned)")

    # Pareto frontier among all points (including ref)
    all_for_pareto = points + [ref]
    pts_sorted = sorted([(p["recall"], p["spec"]) for p in all_for_pareto], key=lambda x: -x[0])
    frontier = []
    best_spec = -1
    for rec, sp in pts_sorted:
        if sp > best_spec:
            frontier.append((rec, sp)); best_spec = sp
    frontier.sort()
    ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
            color="black", lw=1.2, ls="--", alpha=0.5, zorder=3, label="Pareto frontier")

    # Tier color legend
    from matplotlib.lines import Line2D
    tier_handles = [Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=tier_colors[t], markersize=10,
                           markeredgecolor="black", label=f"tier = {t}")
                    for t in sorted(tier_colors) if any(p["tier"] == t for p in points)]
    leg_t = ax.legend(handles=tier_handles, loc="lower left", fontsize=9,
                      framealpha=0.95, title="model tier")
    ax.add_artist(leg_t)
    # Reference + Pareto legend
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.plot([0, 1], [1, 0], color="lightgray", ls=":", lw=1, zorder=0)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=tier_colors[t], markersize=10,
                      markeredgecolor="black", label=f"tier = {t}")
               for t in sorted(tier_colors) if any(p["tier"] == t for p in points)]
    handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                          markersize=18, markeredgecolor="black",
                          label="threshold-tuned ref"))
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.95)
    ax.plot([0, 1], [1, 0], color="lightgray", ls=":", lw=1, zorder=0)
    ax.set_xlabel("Recall"); ax.set_ylabel("Specificity")
    ax.set_xlim(-0.02, 1.05); ax.set_ylim(-0.02, 1.05)
    ax.set_title("A — All measured cells on van_Dis_2020 (default threshold = 0)")

    # === RIGHT: leaderboard horizontal bars, top 12 by utility ===
    ax = axes[1]
    ranked = sorted(points + [ref], key=lambda p: -p["util"])[:12]
    n = len(ranked)
    y = np.arange(n)[::-1]
    h = 0.27
    labels = [f"{'★ ' if p is ref else ''}{p['arch']} × {p['tier']}" for p in ranked]
    recalls = [p["recall"] for p in ranked]
    specs = [p["spec"] for p in ranked]
    f1s = [p["f1"] for p in ranked]
    ax.barh(y + h, recalls, h, color="#1f77b4", label="Recall", edgecolor="black", lw=0.4)
    ax.barh(y, specs, h, color="#d62728", label="Specificity", edgecolor="black", lw=0.4)
    ax.barh(y - h, f1s, h, color="#2ca02c", label="F1", edgecolor="black", lw=0.4)
    # Highlight reference
    ref_y = [yi for yi, p in zip(y, ranked) if p is ref]
    if ref_y:
        ax.axhspan(ref_y[0] - 1.5*h, ref_y[0] + 1.5*h, color="gold", alpha=0.18, zorder=0)
    # Value labels
    for yi, v in zip(y + h, recalls):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8)
    for yi, v in zip(y, specs):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8)
    for yi, v in zip(y - h, f1s):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8)

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    # No invert: y already descends so best (rank 0) sits at top of chart
    ax.set_xlim(0, 1.15); ax.set_xlabel("Metric value")
    ax.set_title("B — Top 12 cells by utility (2·recall + spec) — threshold-tuned LEADS at the top")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3); ax.grid(axis="y", visible=False)

    plt.suptitle("Overall comparison — single-cell architectures + threshold-tuned reference",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_overall_recall_spec_scatter.png", bbox_inches="tight")
    plt.close()
    print(f"wrote {FIG_DIR.name}/07_overall_recall_spec_scatter.png")


# ============================================================================
# Run all
# ============================================================================

def main() -> int:
    fig_threshold_sweep_van_dis()
    fig_ensemble_scatter()
    fig_cost_recall_curves()
    fig_fewshot_vs_baseline()
    fig_external_validity()
    fig_stability_heatmap()
    fig_overall_scatter()
    print(f"\nAll figures written to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
