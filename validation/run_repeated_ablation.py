"""
validation/run_repeated_ablation.py

Runs the full ablation study (V1-V5) N times and reports mean ± std
per variant per metric — suitable for a conference paper.

Calls run_all_variants() directly from ablation_study.py.

Usage (from project root):
    python -m validation.run_repeated_ablation
    python -m validation.run_repeated_ablation --runs 5
    python -m validation.run_repeated_ablation --runs 5 --only-high
    python -m validation.run_repeated_ablation --runs 5 --variant V1 V3
"""

import os, sys, json, time, math, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.ablation_study import (
    run_all_variants,
    FEATURE_LABELS,
    LABEL_DISPLAY,
)

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")
OUT_PATH          = os.path.join(os.path.dirname(__file__), "repeated_ablation_results.json")

VARIANTS     = ["V1", "V2", "V3", "V4", "V5"]
DESCRIPTIONS = {
    "V1": "Full system",
    "V2": "w/o JSON mode",
    "V3": "w/o few-shot",
    "V4": "w/o enrichment",
    "V5": "Weaker model (phi3:mini)",
}


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

def fmt(m, s, pct=False):
    if pct:
        return f"{m*100:.1f} ± {s*100:.1f}%"
    return f"{m:.4f} ± {s:.4f}"


# ---------------------------------------------------------------------------
# Aggregate across runs
# ---------------------------------------------------------------------------

def aggregate(all_runs: list[dict], variants: list[str]) -> dict:
    """
    all_runs: list of dicts, each dict is one run's output from run_all_variants()
              i.e. {V1: {accuracy, macro_f1, ais, fallback_rate, per_class, ...}, V2: ...}
    Returns: {V1: {metric: {mean, std, values}}, V2: ...}
    """
    summary = {}
    for v in variants:
        runs_for_v = [r[v] for r in all_runs if v in r]
        if not runs_for_v:
            continue

        summary[v] = {
            "accuracy":      _agg(runs_for_v, "accuracy"),
            "macro_f1":      _agg(runs_for_v, "macro_f1"),
            "fallback_rate": _agg(runs_for_v, "fallback_rate"),
            "avg_time_s":    _agg(runs_for_v, "avg_time_s"),
            "ais":           _agg(runs_for_v, "ais"),
            "per_class_f1":  {
                feat: _agg([r["per_class"][feat]["f1"] for r in runs_for_v])
                for feat in FEATURE_LABELS
            },
        }
    return summary


def _agg(data, key=None):
    """If key given, extract from list of dicts. Otherwise data is already a list."""
    vals = [d[key] for d in data] if key else data
    return {"mean": mean(vals), "std": std(vals), "values": vals}


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(summary: dict, n: int, variants: list[str]):
    w = 78
    print("\n" + "=" * w)
    print(f"  REPEATED ABLATION RESULTS  (N={n})   format: mean ± std")
    print("=" * w)

    # Main metrics table
    metrics = [
        ("Accuracy",      "accuracy",      True),
        ("Macro F1",      "macro_f1",      False),
        ("AIS",           "ais",           True),
        ("Fallback Rate", "fallback_rate", True),
        ("Avg Time (s)",  "avg_time_s",    False),
    ]

    print(f"\n  {'Variant':<6} {'Description':<24}", end="")
    for label, _, _ in metrics:
        print(f"  {label:>22}", end="")
    print()
    print("  " + "-" * (w - 2))

    for v in variants:
        if v not in summary:
            continue
        s = summary[v]
        print(f"  {v:<6} {DESCRIPTIONS[v]:<24}", end="")
        for _, key, pct in metrics:
            agg = s[key]
            print(f"  {fmt(agg['mean'], agg['std'], pct):>22}", end="")
        print()

    # Per-class F1 table
    print(f"\n  Per-class F1:")
    col_w = 22
    print(f"  {'Variant':<6}", end="")
    for feat in FEATURE_LABELS:
        print(f"  {LABEL_DISPLAY[feat][:18]:>{col_w}}", end="")
    print()
    print("  " + "-" * (6 + (col_w + 2) * len(FEATURE_LABELS)))

    for v in variants:
        if v not in summary:
            continue
        print(f"  {v:<6}", end="")
        for feat in FEATURE_LABELS:
            agg = summary[v]["per_class_f1"][feat]
            print(f"  {fmt(agg['mean'], agg['std']):>{col_w}}", end="")
        print()

    # Delta vs V1
    if "V1" in summary:
        print(f"\n  Delta vs V1 (positive = V1 is better):")
        print(f"  {'Variant':<6} {'Acc':>22} {'Macro F1':>22} {'AIS':>22}")
        print("  " + "-" * 76)
        v1 = summary["V1"]
        for v in variants:
            if v == "V1" or v not in summary:
                continue
            s = summary[v]
            d_acc = v1["accuracy"]["mean"]      - s["accuracy"]["mean"]
            d_f1  = v1["macro_f1"]["mean"]      - s["macro_f1"]["mean"]
            d_ais = v1["ais"]["mean"]            - s["ais"]["mean"]
            print(f"  {v:<6} {d_acc:>+22.4f} {d_f1:>+22.4f} {d_ais*100:>+21.1f}%")

    print("\n" + "=" * w)


def print_latex(summary: dict, n: int, variants: list[str]):
    print(f"\n  --- LaTeX snippet (N={n}) ---\n")
    print(r"  \begin{tabular}{llcccc}")
    print(r"  \hline")
    print(r"  \textbf{Variant} & \textbf{Description} & \textbf{Accuracy} "
          r"& \textbf{Macro F1} & \textbf{AIS} & \textbf{Fallback\%} \\")
    print(r"  \hline")

    for v in variants:
        if v not in summary:
            continue
        s   = summary[v]
        acc = s["accuracy"];      f1 = s["macro_f1"]
        ais = s["ais"];           fb = s["fallback_rate"]
        print(f"  {v} & {DESCRIPTIONS[v]} "
              f"& ${acc['mean']*100:.1f} \\pm {acc['std']*100:.1f}\\%$ "
              f"& ${f1['mean']:.3f} \\pm {f1['std']:.3f}$ "
              f"& ${ais['mean']*100:.1f} \\pm {ais['std']*100:.1f}\\%$ "
              f"& ${fb['mean']*100:.1f} \\pm {fb['std']*100:.1f}\\%$ \\\\")
        if v == "V1":
            print(r"  \hline")

    print(r"  \hline")
    print(r"  \end{tabular}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",      type=int, default=5,
                        help="Number of times to run the full ablation (default: 5).")
    parser.add_argument("--only-high", action="store_true",
                        help="Only evaluate high-confidence GT entries.")
    parser.add_argument("--variant",   nargs="+", choices=VARIANTS, default=None,
                        help="Run only specific variants e.g. --variant V1 V3.")
    args = parser.parse_args()

    with open(GROUND_TRUTH_PATH) as f:
        gt_entries = json.load(f)
    if args.only_high:
        gt_entries = [e for e in gt_entries if e["confidence"] == "high"]

    variants = args.variant or VARIANTS
    n        = args.runs

    print(f"\n{'='*60}")
    print(f"  REPEATED ABLATION  —  {n} runs  —  {len(gt_entries)} tables")
    print(f"  Variants: {' '.join(variants)}")
    print(f"{'='*60}\n")

    all_runs = []

    for i in range(1, n + 1):
        print(f"\n{'='*40}")
        print(f"  ABLATION RUN {i}/{n}")
        print(f"{'='*40}")
        t0      = time.time()
        results = run_all_variants(gt_entries, selected=None if len(variants) == 5
                                               else variants[0] if len(variants) == 1
                                               else None)
        # run_all_variants only accepts a single variant string or None
        # if multiple specific variants requested, run each
        if args.variant and len(args.variant) > 1:
            results = {}
            for v in variants:
                results.update(run_all_variants(gt_entries, selected=v))

        elapsed = time.time() - t0
        all_runs.append(results)

        print(f"\n  Run {i} done in {elapsed:.0f}s")
        for v in variants:
            if v in results:
                m = results[v]
                print(f"    {v}: acc={m['accuracy']:.3f}  f1={m['macro_f1']:.3f}  "
                      f"ais={m.get('ais', 0):.3f}")

    summary = aggregate(all_runs, variants)
    print_summary(summary, n, variants)
    print_latex(summary, n, variants)

    out = {
        "n_runs":    n,
        "n_tables":  len(gt_entries),
        "variants":  variants,
        "all_runs":  all_runs,
        "summary":   summary,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved -> {OUT_PATH}\n")


if __name__ == "__main__":
    main()