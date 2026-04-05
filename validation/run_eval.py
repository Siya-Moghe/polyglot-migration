"""
validation/run_eval.py

Runs the migration pipeline on every .db in validation/ground_tables/,
compares routing decisions against ground_truth.json, and prints a full
evaluation report suitable for a research paper.

Run from your project root:
    python -m validation.run_eval

Optional flags:
    --only-high       skip medium-confidence cases
    --table TABLE     run only cases whose table_name matches TABLE
    --no-export       skip Stage 3 (DB export); only measures routing accuracy
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict

# Add project root to path so core.* imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.introspect import introspect_schema, fetch_rows
from core.orchestrator import plan_embeddings

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")
GROUND_TABLES_DIR = os.path.join(os.path.dirname(__file__), "ground_tables")

LABELS = ["chroma", "mongo", "neo4j", "relational"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(only_high: bool = False) -> list[dict]:
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)
    if only_high:
        gt = [e for e in gt if e["confidence"] == "high"]
    return gt


def run_single(db_file: str, table_name: str) -> dict:
    """
    Run orchestrator on a single .db file and return the routing result.
    Returns dict with: target_db, routing_reason, fallback_used, retries, elapsed_s
    """
    db_path = os.path.join(GROUND_TABLES_DIR, db_file)
    conn_str = f"sqlite:///{db_path}"

    t0 = time.time()

    schema = introspect_schema(conn_str)
    if table_name not in schema:
        return {"error": f"Table '{table_name}' not found in {db_file}"}

    # Fetch sample rows for enrichment (same as pipeline.py)
    try:
        sample_rows = fetch_rows(conn_str, table_name, limit=2)
    except Exception:
        sample_rows = []

    sample_rows_map = {table_name: sample_rows}

    strategies = plan_embeddings(
        {table_name: schema[table_name]},
        sample_rows_map=sample_rows_map,
    )

    elapsed = time.time() - t0
    final = strategies.get(table_name, {})

    return {
        "predicted_db":   final.get("target_db", "unknown"),
        "routing_reason": final.get("routing_reason", final.get("reasoning", "N/A")),
        "fallback_used":  final.get("fallback_used", False),
        "elapsed_s":      round(elapsed, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """
    results: list of {expected_db, predicted_db, fallback_used, confidence, ...}
    Returns overall accuracy, per-class precision/recall/F1, fallback rate.
    """
    total  = len(results)
    correct = sum(1 for r in results if r["predicted_db"] == r["expected_db"])
    fallbacks = sum(1 for r in results if r["fallback_used"])

    # Per-class TP/FP/FN
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for r in results:
        pred = r["predicted_db"]
        exp  = r["expected_db"]
        if pred == exp:
            tp[exp] += 1
        else:
            fp[pred] += 1
            fn[exp]  += 1

    per_class = {}
    for label in LABELS:
        p  = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        r  = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[label] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
                             "tp": tp[label], "fp": fp[label], "fn": fn[label]}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(LABELS)

    return {
        "total":         total,
        "correct":       correct,
        "accuracy":      round(correct / total, 3) if total else 0,
        "fallback_rate": round(fallbacks / total, 3) if total else 0,
        "fallbacks":     fallbacks,
        "macro_f1":      round(macro_f1, 3),
        "per_class":     per_class,
    }


def confusion_matrix(results: list[dict]) -> dict[str, dict[str, int]]:
    """Returns cm[expected][predicted] = count"""
    cm = {exp: {pred: 0 for pred in LABELS} for exp in LABELS}
    for r in results:
        exp  = r["expected_db"]
        pred = r["predicted_db"]
        if exp in cm and pred in cm[exp]:
            cm[exp][pred] += 1
    return cm


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def print_banner(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_per_table(results: list[dict]):
    print_banner("PER-TABLE ROUTING RESULTS")
    header = f"{'#':>3}  {'Table':<28} {'Expected':<12} {'Predicted':<12} {'Match':<6} {'Fallback':<9} {'Time':>6}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results, 1):
        match_sym = "✓" if r["predicted_db"] == r["expected_db"] else "✗"
        fb_sym    = "YES" if r["fallback_used"] else "no"
        conf_tag  = "" if r["confidence"] == "high" else " ~"
        print(
            f"{i:>3}  {r['table_name']:<28} {r['expected_db']:<12} {r['predicted_db']:<12} "
            f"{match_sym:<6} {fb_sym:<9} {r['elapsed_s']:>5.1f}s{conf_tag}"
        )

    print("\n  ~ = medium-confidence case (debatable ground truth)")


def print_confusion_matrix(cm: dict):
    print_banner("CONFUSION MATRIX  (rows=expected, cols=predicted)")
    header_cols = "".join(f"{l:>12}" for l in LABELS)
    print(f"{'':>14}{header_cols}")
    print(f"{'':>14}" + "-" * (12 * len(LABELS)))

    for exp in LABELS:
        row = "".join(f"{cm[exp][pred]:>12}" for pred in LABELS)
        print(f"  {exp:<12}{row}")


def print_per_class_metrics(per_class: dict):
    print_banner("PER-CLASS METRICS")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("  " + "-" * 56)
    for label in LABELS:
        m = per_class[label]
        print(
            f"  {label:<12} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} "
            f"{m['tp']:>5} {m['fp']:>5} {m['fn']:>5}"
        )


def print_summary(metrics: dict, total_time: float, args):
    print_banner("EVALUATION SUMMARY")
    print(f"  Model            : {_get_model()}")
    print(f"  Dataset          : {'high-confidence only' if args.only_high else 'full (high + medium)'}")
    print(f"  Tables evaluated : {metrics['total']}")
    print(f"  Correct routing  : {metrics['correct']} / {metrics['total']}")
    print(f"  Accuracy         : {metrics['accuracy']:.1%}")
    print(f"  Macro F1         : {metrics['macro_f1']:.3f}")
    print(f"  Fallback rate    : {metrics['fallback_rate']:.1%}  ({metrics['fallbacks']} tables)")
    print(f"  Total eval time  : {total_time:.1f}s")
    print(f"  Avg time/table   : {total_time / metrics['total']:.1f}s" if metrics["total"] else "")


def print_error_analysis(results: list[dict]):
    wrong = [r for r in results if r["predicted_db"] != r["expected_db"]]
    if not wrong:
        print_banner("ERROR ANALYSIS — No misclassifications! ✓")
        return

    print_banner(f"ERROR ANALYSIS — {len(wrong)} misclassification(s)")
    for r in wrong:
        print(f"\n  Table     : {r['table_name']}")
        print(f"  Expected  : {r['expected_db'].upper()}")
        print(f"  Predicted : {r['predicted_db'].upper()}")
        print(f"  Confidence: {r['confidence']}")
        print(f"  GT Reason : {r['rationale']}")
        print(f"  LLM Reason: {r['routing_reason'][:120]}...")


def _get_model() -> str:
    try:
        from core.llm_utils import MODEL
        return MODEL
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate polyglot routing accuracy.")
    parser.add_argument("--only-high", action="store_true",
                        help="Evaluate only high-confidence ground-truth cases.")
    parser.add_argument("--table", type=str, default=None,
                        help="Run only this table_name (substring match).")
    args = parser.parse_args()

    gt_entries = load_ground_truth(only_high=args.only_high)

    if args.table:
        gt_entries = [e for e in gt_entries if args.table.lower() in e["table_name"].lower()]

    if not gt_entries:
        print("No ground-truth entries matched the filters.")
        sys.exit(1)

    print(f"\nEvaluating {len(gt_entries)} tables against ground truth...\n")

    results   = []
    total_t0  = time.time()

    for i, entry in enumerate(gt_entries, 1):
        db_file    = entry["db_file"]
        table_name = entry["table_name"]

        print(f"[{i:>2}/{len(gt_entries)}] {table_name} ({db_file})")

        result = run_single(db_file, table_name)

        if "error" in result:
            print(f"       ERROR: {result['error']}")
            continue

        result.update({
            "table_name":  table_name,
            "expected_db": entry["expected_db"],
            "confidence":  entry["confidence"],
            "rationale":   entry["rationale"],
        })
        results.append(result)

        match = "✓" if result["predicted_db"] == result["expected_db"] else "✗"
        print(f"       {match} expected={entry['expected_db'].upper():<12} "
              f"predicted={result['predicted_db'].upper():<12} ({result['elapsed_s']}s)")

    total_time = time.time() - total_t0

    if not results:
        print("No results collected — check errors above.")
        sys.exit(1)

    metrics = compute_metrics(results)
    cm      = confusion_matrix(results)

    print_per_table(results)
    print_confusion_matrix(cm)
    print_per_class_metrics(metrics["per_class"])
    print_error_analysis(results)
    print_summary(metrics, total_time, args)

    # Save results JSON for further analysis / paper tables
    out_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "confusion_matrix": cm,
            "results": results,
        }, f, indent=2)
    print(f"\n  Full results saved to: {out_path}\n")


if __name__ == "__main__":
    main()