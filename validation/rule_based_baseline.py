"""
validation/rule_based_baseline.py

A deterministic rule-based router that classifies each table using
hand-crafted heuristics — no LLM involved, zero fallbacks.

Used as a comparison baseline against the LangGraph multi-agent system.

Run from your project root:
    python -m validation.rule_based_baseline

Results are saved to validation/baseline_results.json.
Run AFTER run_eval.py so you can compare side-by-side.
"""

import os
import sys
import json
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.introspect import introspect_schema, fetch_rows

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")
GROUND_TABLES_DIR = os.path.join(os.path.dirname(__file__), "ground_tables")
BASELINE_OUT_PATH = os.path.join(os.path.dirname(__file__), "baseline_results.json")
AGENT_OUT_PATH    = os.path.join(os.path.dirname(__file__), "eval_results.json")

LABELS = ["chroma", "mongo", "neo4j", "relational"]

# ─────────────────────────────────────────────────────────────────────────────
# Heuristic thresholds (tunable)
# ─────────────────────────────────────────────────────────────────────────────
LONG_TEXT_SAMPLE_CHARS = 80   # sample value longer than this → "long text"
MIN_FK_FOR_NEO4J       = 2    # tables with >= this many FK cols → neo4j candidate
MIN_TEXT_FOR_CHROMA    = 1    # tables with >= this many long-text cols → chroma candidate


# ─────────────────────────────────────────────────────────────────────────────
# Core routing logic
# ─────────────────────────────────────────────────────────────────────────────

def route_table(table_name: str, info: dict, sample_rows: list[dict]) -> dict:
    """
    Deterministic routing using ordered heuristic rules.

    Priority order (first match wins):
      1. NEO4J   — 2+ FK columns  (junction/mapping table)
      2. CHROMA  — 1+ long TEXT column with sample value > 80 chars
                   OR 2+ TEXT columns and no FKs at all
      3. MONGO   — no FKs, mix of text + other types, short text fields
      4. RELATIONAL — everything else (numeric-heavy, temporal, single FK)
    """
    columns      = info.get("columns", [])
    foreign_keys = info.get("foreign_keys", [])
    primary_keys = set(info.get("primary_keys", []))

    # Build column lookup
    fk_cols = set()
    for fk in foreign_keys:
        for col in fk.get("columns", fk.get("constrained_columns", [])):
            fk_cols.add(col)

    fk_count       = len(fk_cols)
    text_cols      = []
    long_text_cols = []
    numeric_cols   = []
    temporal_cols  = []

    for col in columns:
        name  = col["name"]
        ctype = str(col["type"]).upper()

        if name in primary_keys or name in fk_cols:
            continue   # skip structural columns for content analysis

        if "TEXT" in ctype or "VARCHAR" in ctype or "CHAR" in ctype:
            text_cols.append(name)
            # Check actual sample length
            if sample_rows:
                val = sample_rows[0].get(name, "")
                if isinstance(val, str) and len(val) > LONG_TEXT_SAMPLE_CHARS:
                    long_text_cols.append(name)
        elif "INT" in ctype:
            numeric_cols.append(name)
        elif "FLOAT" in ctype or "REAL" in ctype or "DOUBLE" in ctype or "DECIMAL" in ctype or "NUMERIC" in ctype:
            numeric_cols.append(name)
        elif "DATE" in ctype or "TIME" in ctype or "TIMESTAMP" in ctype:
            temporal_cols.append(name)

    # ── Rule 1: NEO4J ──────────────────────────────────────────────
    if fk_count >= MIN_FK_FOR_NEO4J:
        return {
            "target_db": "neo4j",
            "reasoning": (
                f"Rule 1: {fk_count} FK columns detected (>= {MIN_FK_FOR_NEO4J}). "
                f"Junction/mapping table — graph relationship."
            ),
        }

    # ── Rule 2: CHROMA ─────────────────────────────────────────────
    if long_text_cols:
        return {
            "target_db": "chroma",
            "reasoning": (
                f"Rule 2a: Long text detected in column(s) {long_text_cols} "
                f"(sample > {LONG_TEXT_SAMPLE_CHARS} chars). Semantic embedding target."
            ),
        }

    # Fallback chroma signal: 2+ text cols, no FKs (likely content table)
    if len(text_cols) >= 2 and fk_count == 0 and not numeric_cols and not temporal_cols:
        return {
            "target_db": "chroma",
            "reasoning": (
                f"Rule 2b: {len(text_cols)} text columns, no FKs, no numeric/temporal cols. "
                f"Content-oriented — semantic embedding."
            ),
        }

    # ── Rule 3: MONGO ──────────────────────────────────────────────
    if fk_count == 0:
        return {
            "target_db": "mongo",
            "reasoning": (
                f"Rule 3: No FK columns. Self-contained entity with "
                f"{len(text_cols)} text, {len(numeric_cols)} numeric cols — document store."
            ),
        }

    # ── Rule 4: RELATIONAL (default) ───────────────────────────────
    return {
        "target_db": "relational",
        "reasoning": (
            f"Rule 4 (default): {fk_count} FK(s), {len(numeric_cols)} numeric, "
            f"{len(temporal_cols)} temporal cols — structured/transactional table."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (same functions as run_eval.py — kept self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    total     = len(results)
    correct   = sum(1 for r in results if r["predicted_db"] == r["expected_db"])

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
        per_class[label] = {
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "tp": tp[label], "fp": fp[label], "fn": fn[label],
        }

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(LABELS)

    return {
        "total":         total,
        "correct":       correct,
        "accuracy":      round(correct / total, 3) if total else 0,
        "fallback_rate": 0.0,   # rule-based never falls back
        "fallbacks":     0,
        "macro_f1":      round(macro_f1, 3),
        "per_class":     per_class,
    }


def confusion_matrix(results: list[dict]) -> dict:
    cm = {exp: {pred: 0 for pred in LABELS} for exp in LABELS}
    for r in results:
        exp  = r["expected_db"]
        pred = r["predicted_db"]
        if exp in cm and pred in cm[exp]:
            cm[exp][pred] += 1
    return cm


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────

def print_banner(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_per_table(results):
    print_banner("PER-TABLE ROUTING RESULTS  [RULE-BASED BASELINE]")
    header = f"{'#':>3}  {'Table':<28} {'Expected':<12} {'Predicted':<12} {'Match':<6}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results, 1):
        match = "✓" if r["predicted_db"] == r["expected_db"] else "✗"
        conf  = "" if r["confidence"] == "high" else " ~"
        print(f"{i:>3}  {r['table_name']:<28} {r['expected_db']:<12} {r['predicted_db']:<12} {match}{conf}")
    print("\n  ~ = medium-confidence case")


def print_confusion_matrix(cm):
    print_banner("CONFUSION MATRIX  [RULE-BASED]  (rows=expected, cols=predicted)")
    header_cols = "".join(f"{l:>12}" for l in LABELS)
    print(f"{'':>14}{header_cols}")
    print(f"{'':>14}" + "-" * (12 * len(LABELS)))
    for exp in LABELS:
        row = "".join(f"{cm[exp][pred]:>12}" for pred in LABELS)
        print(f"  {exp:<12}{row}")


def print_per_class(per_class):
    print_banner("PER-CLASS METRICS  [RULE-BASED]")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("  " + "-" * 56)
    for label in LABELS:
        m = per_class[label]
        print(f"  {label:<12} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} "
              f"{m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")


def print_error_analysis(results):
    wrong = [r for r in results if r["predicted_db"] != r["expected_db"]]
    if not wrong:
        print_banner("ERROR ANALYSIS — No misclassifications!")
        return
    print_banner(f"ERROR ANALYSIS — {len(wrong)} misclassification(s)")
    for r in wrong:
        print(f"\n  Table     : {r['table_name']}")
        print(f"  Expected  : {r['expected_db'].upper()}")
        print(f"  Predicted : {r['predicted_db'].upper()}")
        print(f"  Confidence: {r['confidence']}")
        print(f"  GT Reason : {r['rationale']}")
        print(f"  Rule Used : {r['routing_reason']}")


def print_comparison(baseline_metrics: dict, agent_path: str):
    """Side-by-side comparison table if agent results exist."""
    if not os.path.exists(agent_path):
        print("\n  (Run run_eval.py first to see side-by-side comparison.)")
        return

    with open(agent_path) as f:
        agent_data = json.load(f)
    agent_metrics = agent_data["metrics"]

    print_banner("SIDE-BY-SIDE COMPARISON")
    print(f"  {'Metric':<22} {'Rule-Based':>12} {'LLM Agents':>12}  {'Winner':>10}")
    print("  " + "-" * 60)

    rows = [
        ("Accuracy",      baseline_metrics["accuracy"],      agent_metrics["accuracy"],      "higher"),
        ("Macro F1",      baseline_metrics["macro_f1"],       agent_metrics["macro_f1"],       "higher"),
        ("Fallback Rate", baseline_metrics["fallback_rate"],  agent_metrics["fallback_rate"],  "lower"),
    ]

    for label, b_val, a_val, direction in rows:
        if direction == "higher":
            winner = "LLM" if a_val > b_val else ("Rule" if b_val > a_val else "Tie")
        else:
            winner = "LLM" if a_val < b_val else ("Rule" if b_val < a_val else "Tie")
        print(f"  {label:<22} {b_val:>12.3f} {a_val:>12.3f}  {winner:>10}")

    print()
    print("  Per-class F1 breakdown:")
    print(f"  {'Class':<12} {'Rule-Based F1':>14} {'LLM Agents F1':>14}")
    print("  " + "-" * 42)
    for label in LABELS:
        b_f1 = baseline_metrics["per_class"][label]["f1"]
        a_f1 = agent_metrics["per_class"][label]["f1"]
        delta = a_f1 - b_f1
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        print(f"  {label:<12} {b_f1:>14.3f} {a_f1:>14.3f}  {arrow} {abs(delta):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    with open(GROUND_TRUTH_PATH) as f:
        gt_entries = json.load(f)

    print(f"\nRunning rule-based baseline on {len(gt_entries)} tables...\n")

    results   = []
    total_t0  = time.time()

    for i, entry in enumerate(gt_entries, 1):
        db_path  = os.path.join(GROUND_TABLES_DIR, entry["db_file"])
        conn_str = f"sqlite:///{db_path}"
        tname    = entry["table_name"]

        schema = introspect_schema(conn_str)
        if tname not in schema:
            print(f"[{i:>2}] ERROR: table '{tname}' not found in {entry['db_file']}")
            continue

        try:
            sample_rows = fetch_rows(conn_str, tname, limit=2)
        except Exception:
            sample_rows = []

        routing = route_table(tname, schema[tname], sample_rows)

        result = {
            "table_name":     tname,
            "expected_db":    entry["expected_db"],
            "predicted_db":   routing["target_db"],
            "routing_reason": routing["reasoning"],
            "fallback_used":  False,
            "confidence":     entry["confidence"],
            "rationale":      entry["rationale"],
        }
        results.append(result)

        match = "✓" if result["predicted_db"] == result["expected_db"] else "✗"
        print(f"[{i:>2}/40]  {match}  {tname:<28}  "
              f"expected={entry['expected_db'].upper():<12}  "
              f"predicted={routing['target_db'].upper()}")

    total_time = time.time() - total_t0
    metrics    = compute_metrics(results)
    cm         = confusion_matrix(results)

    print_per_table(results)
    print_confusion_matrix(cm)
    print_per_class(metrics["per_class"])
    print_error_analysis(results)
    print_comparison(metrics, AGENT_OUT_PATH)

    print_banner("BASELINE SUMMARY")
    print(f"  System           : Rule-Based Heuristics (no LLM)")
    print(f"  Tables evaluated : {metrics['total']}")
    print(f"  Correct routing  : {metrics['correct']} / {metrics['total']}")
    print(f"  Accuracy         : {metrics['accuracy']:.1%}")
    print(f"  Macro F1         : {metrics['macro_f1']:.3f}")
    print(f"  Fallback rate    : 0%  (deterministic — never falls back)")
    print(f"  Total time       : {total_time:.2f}s  (no LLM calls)")

    with open(BASELINE_OUT_PATH, "w") as f:
        json.dump({"metrics": metrics, "confusion_matrix": cm, "results": results}, f, indent=2)
    print(f"\n  Results saved to: {BASELINE_OUT_PATH}\n")


if __name__ == "__main__":
    main()