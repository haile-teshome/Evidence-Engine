"""Stability analysis across repeats of leads_native × leads.

Combines repeats from one new --repeat 3 run with the original rep=1 from the
latest reports/leads-local-* run (auto-discovered) to get 4 independent passes
over the same papers. Reports:

  - Per-rep metrics (F1/recall/spec/MCC)
  - Pairwise Cohen's kappa across reps + mean
  - Per-paper consistency (% of reps that agreed with the modal prediction)
  - Where the disagreements live (positive vs negative gold class)
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from metrics import compute, cohens_kappa, fleiss_kappa  # noqa: E402


def load_runs(paths: list[tuple[Path, int]]) -> dict[str, list[tuple[int, int, float]]]:
    """Returns {paper_id: [(rep_index, prediction, label, confidence), ...]}.
    Each `paths` element is (csv_path, rep_offset)."""
    by_paper: dict[str, list] = defaultdict(list)
    for path, offset in paths:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("architecture") != "leads_native" or r.get("model_tier") != "leads":
                    continue
                if r.get("label") in ("", None):
                    continue
                try:
                    label = int(r["label"])
                    pred = int(r["prediction"])
                    conf = float(r["confidence"])
                    rep = int(r.get("repeat", "1")) + offset
                except (ValueError, TypeError):
                    continue
                by_paper[r["paper_id"]].append((rep, pred, label, conf))
    return by_paper


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    new_run = sys.argv[1] if len(sys.argv) > 1 else None
    if new_run is None:
        # Auto-find
        candidates = sorted(root.glob("reports/stability-leads-*/predictions.csv"))
        if not candidates:
            print("Pass new-run predictions.csv path as argv[1]", file=sys.stderr)
            return 2
        new_run = str(candidates[-1])

    baseline_candidates = sorted(root.glob("reports/leads-local-*/predictions.csv"))
    if not baseline_candidates:
        print("No reports/leads-local-* baseline run found.", file=sys.stderr)
        return 2
    sources = [
        (baseline_candidates[-1], 0),   # rep1 → 1
        (Path(new_run), 10),            # rep1, rep2, rep3 → 11, 12, 13
    ]
    by_paper = load_runs(sources)

    # Build per-rep prediction matrices, sorted by rep, indexed by paper_id
    paper_ids = sorted(by_paper.keys())
    reps_seen: set = set()
    for pid in paper_ids:
        for rep, _, _, _ in by_paper[pid]:
            reps_seen.add(rep)
    reps = sorted(reps_seen)
    print(f"Loaded {len(paper_ids)} papers × {len(reps)} repetitions (rep ids: {reps})")

    # Per-rep metrics
    print(f"\n=== Per-rep metrics ===")
    print(f"  {'rep':>4} {'TP':>3} {'FP':>4} {'TN':>4} {'FN':>3} {'recall':>7} {'spec':>6} {'F1':>6} {'MCC':>7}")
    rep_preds: dict[int, list[int]] = {}
    rep_labels: list[int] = []
    for r in reps:
        preds = []
        labels = []
        confs = []
        for pid in paper_ids:
            match = [t for t in by_paper[pid] if t[0] == r]
            if not match:
                continue
            _, p, lb, c = match[0]
            preds.append(p)
            labels.append(lb)
            confs.append(c)
        m = compute(labels, preds, confs)
        rep_preds[r] = preds
        if not rep_labels:
            rep_labels = labels
        print(f"  {r:4d} {m.tp:3d} {m.fp:4d} {m.tn:4d} {m.fn:3d} "
              f"{m.recall:7.3f} {m.specificity:6.3f} {m.f1:6.3f} {m.mcc:+7.3f}")

    # Pairwise kappa
    print(f"\n=== Pairwise Cohen's kappa across reps ===")
    kappas: list[float] = []
    for i, r1 in enumerate(reps):
        for r2 in reps[i+1:]:
            if r1 in rep_preds and r2 in rep_preds and len(rep_preds[r1]) == len(rep_preds[r2]):
                k = cohens_kappa(rep_preds[r1], rep_preds[r2])
                kappas.append(k)
                print(f"  rep {r1} vs rep {r2}: κ = {k:+.4f}")
    if kappas:
        print(f"  Mean pairwise κ: {sum(kappas)/len(kappas):+.4f}  (range: {min(kappas):+.4f} to {max(kappas):+.4f})")

    # Fleiss across all reps
    if len(reps) >= 2:
        ratings = []
        for i in range(len(rep_preds[reps[0]])):
            ratings.append([rep_preds[r][i] for r in reps])
        fk = fleiss_kappa(ratings)
        print(f"\n  Fleiss κ ({len(reps)}-rater): {fk:+.4f}")

    # Per-paper consistency
    print(f"\n=== Per-paper consistency ===")
    full_agree = 0
    partial_disagree = 0
    disagreement_papers_pos = []
    disagreement_papers_neg = []
    for i, pid in enumerate(paper_ids):
        votes = [rep_preds[r][i] for r in reps if i < len(rep_preds[r])]
        if not votes:
            continue
        if all(v == votes[0] for v in votes):
            full_agree += 1
        else:
            partial_disagree += 1
            label = rep_labels[i]
            if label == 1:
                disagreement_papers_pos.append((pid, votes))
            else:
                disagreement_papers_neg.append((pid, votes))
    total = full_agree + partial_disagree
    print(f"  Full agreement across all {len(reps)} reps: {full_agree}/{total} = {full_agree/total:.3f}")
    print(f"  At least one disagreement:                  {partial_disagree}/{total} = {partial_disagree/total:.3f}")
    print(f"    of which gold positives: {len(disagreement_papers_pos)}  (label=1, flickering)")
    print(f"    of which gold negatives: {len(disagreement_papers_neg)}")
    if disagreement_papers_pos:
        print(f"\n  Positive papers with at least one disagreement (potential FNs):")
        for pid, votes in disagreement_papers_pos[:5]:
            print(f"    {pid}: votes {votes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
