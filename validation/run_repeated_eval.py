"""
validation/run_repeated_eval.py

Runs the eval suite N times and reports mean ± std for every metric —
suitable for a paper results table.

Calls functions directly from the other three validation scripts.
No logic is duplicated here.

What runs each iteration:
  - run_eval_against_gt  : LLM agent pass on all GT tables
  - ablation_study       : neutral + biased pass for AIS
Baseline runs once (deterministic — no variance).

Output:
  Console : human-readable mean ± std table + LaTeX snippet
  File    : validation/repeated_eval_results.json

Usage (from project root):
    python -m validation.run_repeated_eval
    python -m validation.run_repeated_eval --runs 15
    python -m validation.run_repeated_eval --runs 3 --only-high
    python -m validation.run_repeated_eval --runs 15 --skip-ais
"""

import os, sys, json, time, math, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- import from the other validation scripts directly ---
from validation.run_eval_against_gt  import get_expected_feature, GROUND_TABLES_DIR
from validation.run_eval_against_gt  import compute_metrics as _gt_metrics

from validation.rule_based_baseline  import route_table   as _bl_route
from validation.rule_based_baseline  import compute_metrics as _bl_metrics

from validation.ablation_study       import run_variant_eval, calculate_ais, BIAS_PROMPT, compute_metrics as _abl_metrics

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")
OUT_PATH          = os.path.join(os.path.dirname(__file__), "repeated_eval_results.json")

FEATURE_LABELS = ["use_vector", "use_graph", "use_nested", "index_cols"]
FEATURE_DISPLAY = {
    "use_vector": "use_vector  (was Chroma)",
    "use_graph":  "use_graph   (was Neo4j) ",
    "use_nested": "use_nested  (was Mongo) ",
    "index_cols": "index_cols  (was SQL)   ",
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
# One agent eval run — thin wrapper around run_single + compute_metrics
# ---------------------------------------------------------------------------

def run_agent_once(gt_entries: list[dict]) -> tuple[dict, list[dict]]:
    """Run one neutral pass. Returns (metrics, raw_results).
    Raw results are reused by AIS so the 40 tables are only run once per iteration."""
    raw = run_variant_eval(gt_entries, human_context="")
    return _abl_metrics(raw), raw


# ---------------------------------------------------------------------------
# Baseline — deterministic, called once
# ---------------------------------------------------------------------------

def run_baseline_once(gt_entries: list[dict]) -> dict:
    from core.introspect import introspect_schema, fetch_rows

    results = []
    for entry in gt_entries:
        db_path  = os.path.join(GROUND_TABLES_DIR, entry["db_file"])
        conn_str = f"sqlite:///{db_path}"
        tname    = entry["table_name"]

        schema = introspect_schema(conn_str)
        if tname not in schema:
            continue
        try:
            rows = fetch_rows(conn_str, tname, limit=2)
        except Exception:
            rows = []

        routing         = _bl_route(tname, schema[tname], rows)
        exp_feat        = get_expected_feature(entry)
        primary_correct = routing["primary_feature"] == exp_feat

        results.append({
            "table_name":               tname,
            "origin_label":             entry["expected_db"],
            "expected_primary_feature": exp_feat,
            "predicted_feature":        routing["primary_feature"],
            "primary_correct":          primary_correct,
            "rule_reason":              routing["reason"],
            "confidence":               entry["confidence"],
            "rationale":                entry["rationale"],
        })

    return _bl_metrics(results)


# ---------------------------------------------------------------------------
# AIS — one neutral + one biased pass via ablation_study.run_variant_eval
# ---------------------------------------------------------------------------

def run_ais_once(gt_entries: list[dict], neutral_raw: list[dict]) -> float:
    """Reuses the already-run neutral pass. Only runs the biased pass."""
    biased = run_variant_eval(gt_entries, human_context=BIAS_PROMPT)
    return calculate_ais(neutral_raw, biased)


# ---------------------------------------------------------------------------
# Collect scalar lists from repeated runs
# ---------------------------------------------------------------------------

def collect(runs: list[dict], key: str) -> list[float]:
    return [r[key] for r in runs if key in r]

def collect_per_class_f1(runs: list[dict]) -> dict[str, list[float]]:
    return {
        feat: [r["per_class"][feat]["f1"] for r in runs if feat in r.get("per_class", {})]
        for feat in FEATURE_LABELS
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_results_table(agent_runs: list[dict], ais_vals: list[float],
                        baseline: dict, n: int):
    w = 74
    print("\n" + "=" * w)
    print(f"  REPEATED EVALUATION  (N={n})   format: mean ± std")
    print("=" * w)

    pc_f1 = collect_per_class_f1(agent_runs)

    rows = [
        ("Primary Accuracy",
            baseline["primary_accuracy"],
            collect(agent_runs, "accuracy"), True),
        ("Macro F1",
            baseline["macro_f1"],
            collect(agent_runs, "macro_f1"), False),
        ("Fallback Rate",
            baseline["fallback_rate"],
            collect(agent_runs, "fallback_rate"), True),
        ("AIS",
            None,
            ais_vals, True),
    ]

    print(f"\n  {'Metric':<34} {'Rule-Based':>12} {'LLM Agents':>24}")
    print("  " + "-" * (w - 2))
    for label, b_val, a_vals, pct in rows:
        m, s  = mean(a_vals), std(a_vals)
        b_str = ((f"{b_val*100:.1f}%" if pct else f"{b_val:.4f}")
                 if b_val is not None else "—")
        print(f"  {label:<34} {b_str:>12}  {fmt(m, s, pct):>22}")

    print(f"\n  Per-class F1:")
    print(f"  {'Feature':<28} {'Rule-Based':>12} {'LLM Agents':>24}")
    print("  " + "-" * 66)
    for feat in FEATURE_LABELS:
        b_f1 = baseline["per_class"][feat]["f1"]
        vals  = pc_f1[feat]
        m, s  = mean(vals), std(vals)
        print(f"  {FEATURE_DISPLAY[feat]:<28} {b_f1:>12.4f}  {fmt(m, s):>22}")

    print("\n" + "=" * w)


def print_latex(agent_runs: list[dict], ais_vals: list[float],
                baseline: dict, n: int):
    pc_f1 = collect_per_class_f1(agent_runs)

    print(f"\n  --- LaTeX snippet (N={n}) ---\n")
    print(r"  \begin{tabular}{lcc}")
    print(r"  \hline")
    print(r"  \textbf{Metric} & \textbf{Rule-Based} & \textbf{LLM Agents ($\mu \pm \sigma$)} \\")
    print(r"  \hline")

    def row(label, b_val, a_vals, pct=False):
        m, s  = mean(a_vals), std(a_vals)
        b_str = ((f"{b_val*100:.1f}\\%"  if pct else f"{b_val:.3f}")
                 if b_val is not None else "--")
        a_str = (f"{m*100:.1f} $\\pm$ {s*100:.1f}\\%" if pct
                 else f"{m:.3f} $\\pm$ {s:.3f}")
        print(f"  {label} & {b_str} & {a_str} \\\\")

    row("Primary Accuracy",  baseline["primary_accuracy"], collect(agent_runs, "accuracy"), True)
    row("Macro F1",          baseline["macro_f1"],         collect(agent_runs, "macro_f1"))
    row("Fallback Rate",     baseline["fallback_rate"],    collect(agent_runs, "fallback_rate"), True)
    row("AIS",               None,                         ais_vals, True)
    print(r"  \hline")
    for feat in FEATURE_LABELS:
        row(f"F1: \\texttt{{{feat}}}", baseline["per_class"][feat]["f1"], pc_f1[feat])
    print(r"  \hline")
    print(r"  \end{tabular}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",      type=int,  default=15)
    parser.add_argument("--only-high", action="store_true")
    parser.add_argument("--skip-ais",  action="store_true",
                        help="Skip AIS passes (halves LLM calls per run).")
    args = parser.parse_args()

    with open(GROUND_TRUTH_PATH) as f:
        gt_entries = json.load(f)
    if args.only_high:
        gt_entries = [e for e in gt_entries if e["confidence"] == "high"]

    n = args.runs
    print(f"\n{'='*60}")
    print(f"  REPEATED EVAL  —  {n} runs  —  {len(gt_entries)} tables")
    print(f"{'='*60}\n")

    # Baseline — once
    print("Baseline (deterministic, running once)...")
    baseline = run_baseline_once(gt_entries)
    print(f"  acc={baseline['primary_accuracy']:.3f}  f1={baseline['macro_f1']:.3f}\n")

    # Repeated agent runs
    agent_runs: list[dict] = []
    ais_vals:   list[float] = []

    for i in range(1, n + 1):
        print(f"--- Run {i}/{n} ---")
        t0              = time.time()
        m, neutral_raw  = run_agent_once(gt_entries)
        agent_runs.append(m)

        ais = None
        if not args.skip_ais:
            print(f"  AIS (biased pass only — neutral already done)...")
            ais = run_ais_once(gt_entries, neutral_raw)
            ais_vals.append(ais)

        elapsed = time.time() - t0
        print(f"  acc={m['accuracy']:.3f}  f1={m['macro_f1']:.3f}  "
              f"fallback={m['fallback_rate']:.3f}  "
              f"{'ais='+f'{ais:.3f}' if ais is not None else 'ais=skipped'}  "
              f"({elapsed:.0f}s)\n")

    if not ais_vals:
        ais_vals = [0.0]

    print_results_table(agent_runs, ais_vals, baseline, n)
    print_latex(agent_runs, ais_vals, baseline, n)

    # Save
    out = {
        "n_runs":   n,
        "n_tables": len(gt_entries),
        "baseline": baseline,
        "agent_runs": agent_runs,
        "ais_values": ais_vals,
        "summary": {
            key: {"mean": mean(collect(agent_runs, key)),
                  "std":  std(collect(agent_runs, key)),
                  "values": collect(agent_runs, key)}
            for key in ["accuracy", "macro_f1", "fallback_rate"]
        },
        "ais_summary": {"mean": mean(ais_vals), "std": std(ais_vals), "values": ais_vals},
        "per_class_f1": {
            feat: {"mean": mean(v), "std": std(v), "values": v}
            for feat, v in collect_per_class_f1(agent_runs).items()
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved -> {OUT_PATH}\n")


if __name__ == "__main__":
    main()