"""
validation/rule_based_baseline.py

Deterministic rule-based router that predicts which SurrealDB features
each table should activate — no LLM, zero fallbacks.

The rules mirror the old polyglot heuristics translated to SurrealDB:
  use_vector  <- long text columns      (was Chroma)
  use_graph   <- 2+ FK columns          (was Neo4j)
  use_nested  <- no FKs, mixed fields   (was Mongo)
  index_cols  <- numeric/temporal/txn   (was Relational)

Ground truth is read from expected_db (always present).
expected_features is used if present, otherwise derived on the fly.

Run from project root:
    python -m validation.rule_based_baseline

Results saved to validation/baseline_results.json.
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

FEATURE_LABELS = ["use_vector", "use_graph", "use_nested", "index_cols"]

LABEL_TO_FEATURE = {
    "chroma":     "use_vector",
    "neo4j":      "use_graph",
    "mongo":      "use_nested",
    "relational": "index_cols",
}

LABEL_DISPLAY = {
    "use_vector": "use_vector  (was Chroma)",
    "use_graph":  "use_graph   (was Neo4j) ",
    "use_nested": "use_nested  (was Mongo) ",
    "index_cols": "index_cols  (was SQL)   ",
}

LONG_TEXT_CHARS  = 80
MIN_FK_FOR_GRAPH = 2


# ---------------------------------------------------------------------------
# Derive expected primary feature from an entry (works with or without
# expected_features key in the JSON)
# ---------------------------------------------------------------------------

def get_expected_feature(entry: dict) -> str:
    if "expected_features" in entry:
        return entry["expected_features"]["primary_feature"]
    return LABEL_TO_FEATURE.get(entry["expected_db"], "index_cols")


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def route_table(table_name: str, info: dict, sample_rows: list[dict]) -> dict:
    """
    Returns predicted primary SurrealDB feature + reason string.

    Priority (first match wins):
      1. use_graph  — 2+ FK columns
      2. use_vector — 1+ long-text column OR 2+ text cols, no FKs/numeric/temporal
      3. use_nested — no FKs, self-contained entity
      4. index_cols — everything else (numeric/temporal/transactional)
    """
    columns      = info.get("columns", [])
    foreign_keys = info.get("foreign_keys", [])
    primary_keys = set(info.get("primary_keys", []))

    fk_cols = {
        col
        for fk in foreign_keys
        for col in fk.get("columns", fk.get("constrained_columns", []))
    }
    fk_count = len(fk_cols)

    text_cols, long_text_cols, numeric_cols, temporal_cols = [], [], [], []

    for col in columns:
        name  = col["name"]
        ctype = str(col["type"]).upper()
        if name in primary_keys or name in fk_cols:
            continue
        if any(t in ctype for t in ("TEXT", "VARCHAR", "CHAR")):
            text_cols.append(name)
            if sample_rows:
                val = sample_rows[0].get(name, "")
                if isinstance(val, str) and len(val) > LONG_TEXT_CHARS:
                    long_text_cols.append(name)
        elif any(t in ctype for t in ("INT", "FLOAT", "REAL", "DOUBLE", "DECIMAL", "NUMERIC")):
            numeric_cols.append(name)
        elif any(t in ctype for t in ("DATE", "TIME", "TIMESTAMP")):
            temporal_cols.append(name)

    # Rule 1 — use_graph
    if fk_count >= MIN_FK_FOR_GRAPH:
        return {
            "primary_feature": "use_graph",
            "reason": (f"Rule 1: {fk_count} FK columns (>= {MIN_FK_FOR_GRAPH}). "
                       f"Junction/mapping table -> SurrealDB RELATE edges."),
        }

    # Rule 2 — use_vector
    if long_text_cols:
        return {
            "primary_feature": "use_vector",
            "reason": (f"Rule 2a: Long text in {long_text_cols} "
                       f"(sample > {LONG_TEXT_CHARS} chars) -> vector embedding."),
        }
    if len(text_cols) >= 2 and fk_count == 0 and not numeric_cols and not temporal_cols:
        return {
            "primary_feature": "use_vector",
            "reason": (f"Rule 2b: {len(text_cols)} text cols, no FKs/numeric/temporal "
                       f"-> content-oriented, vector embedding."),
        }

    # Rule 3 — use_nested
    if fk_count == 0:
        return {
            "primary_feature": "use_nested",
            "reason": (f"Rule 3: No FK columns. Self-contained entity "
                       f"({len(text_cols)} text, {len(numeric_cols)} numeric) -> nested document."),
        }

    # Rule 4 — index_cols (default)
    return {
        "primary_feature": "index_cols",
        "reason": (f"Rule 4 (default): {fk_count} FK(s), {len(numeric_cols)} numeric, "
                   f"{len(temporal_cols)} temporal -> structured/transactional, define indexes."),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    total   = len(results)
    correct = sum(1 for r in results if r["primary_correct"])

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for r in results:
        pred = r["predicted_feature"]
        exp  = r["expected_primary_feature"]
        if pred == exp:
            tp[exp] += 1
        else:
            fp[pred] += 1
            fn[exp]  += 1

    per_class = {}
    for feat in FEATURE_LABELS:
        p  = tp[feat] / (tp[feat] + fp[feat]) if (tp[feat] + fp[feat]) > 0 else 0.0
        rc = tp[feat] / (tp[feat] + fn[feat]) if (tp[feat] + fn[feat]) > 0 else 0.0
        f1 = 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0
        per_class[feat] = {
            "precision": round(p, 3), "recall": round(rc, 3), "f1": round(f1, 3),
            "tp": tp[feat], "fp": fp[feat], "fn": fn[feat],
        }

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(FEATURE_LABELS)

    return {
        "total":            total,
        "primary_accuracy": round(correct / total, 3) if total else 0,
        "macro_f1":         round(macro_f1, 3),
        "fallback_rate":    0.0,
        "per_class":        per_class,
    }


def confusion_matrix(results: list[dict]) -> dict:
    cm = {e: {p: 0 for p in FEATURE_LABELS} for e in FEATURE_LABELS}
    for r in results:
        exp  = r["expected_primary_feature"]
        pred = r["predicted_feature"]
        if exp in cm and pred in cm[exp]:
            cm[exp][pred] += 1
    return cm


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_banner(text):
    print("\n" + "=" * 66)
    print(f"  {text}")
    print("=" * 66)


def print_per_table(results):
    print_banner("PER-TABLE RESULTS  [RULE-BASED BASELINE]")
    hdr = f"  {'#':>3}  {'Table':<30} {'Expected Feature':<18} {'Predicted':<18} Match"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i, r in enumerate(results, 1):
        tick = "✓" if r["primary_correct"] else "✗"
        conf = " ~" if r["confidence"] == "medium" else ""
        print(f"  {i:>3}  {r['table_name']:<30} {r['expected_primary_feature']:<18} "
              f"{r['predicted_feature']:<18} {tick}{conf}")
    print("\n  ~ = medium-confidence case")


def print_confusion_matrix(cm):
    print_banner("CONFUSION MATRIX  [RULE-BASED]  (rows=expected, cols=predicted)")
    pad = 22
    header = "".join(f"{f:>18}" for f in FEATURE_LABELS)
    print(f"{'':>{pad}}{header}")
    print(f"{'':>{pad}}" + "-" * (18 * len(FEATURE_LABELS)))
    for exp in FEATURE_LABELS:
        row = "".join(f"{cm[exp][pred]:>18}" for pred in FEATURE_LABELS)
        print(f"  {exp:<{pad - 2}}{row}")


def print_per_class(per_class):
    print_banner("PER-CLASS METRICS  [RULE-BASED]")
    print(f"  {'Feature':<28} {'P':>8} {'R':>8} {'F1':>8}  TP  FP  FN")
    print("  " + "-" * 64)
    for feat in FEATURE_LABELS:
        m = per_class[feat]
        print(f"  {LABEL_DISPLAY[feat]}  {m['precision']:>8.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} "
              f" {m['tp']:>2}  {m['fp']:>2}  {m['fn']:>2}")


def print_errors(results):
    wrong = [r for r in results if not r["primary_correct"]]
    if not wrong:
        print_banner("ERROR ANALYSIS — No misclassifications!")
        return
    print_banner(f"ERROR ANALYSIS — {len(wrong)} misclassification(s)")
    for r in wrong:
        print(f"\n  Table     : {r['table_name']}")
        print(f"  Expected  : {r['expected_primary_feature']}  (origin: {r['origin_label']})")
        print(f"  Predicted : {r['predicted_feature']}")
        print(f"  Confidence: {r['confidence']}")
        print(f"  GT Reason : {r['rationale']}")
        print(f"  Rule Used : {r['rule_reason']}")


def print_comparison(baseline_metrics: dict):
    if not os.path.exists(AGENT_OUT_PATH):
        print("\n  (Run run_eval_against_gt.py first for a side-by-side comparison.)")
        return

    with open(AGENT_OUT_PATH) as f:
        agent_data = json.load(f)
    am = agent_data["metrics"]
    bm = baseline_metrics

    print_banner("SIDE-BY-SIDE: RULE-BASED vs LLM AGENTS")
    print(f"\n  {'Metric':<24} {'Rule-Based':>12} {'LLM Agents':>12}  {'Winner':>8}")
    print("  " + "-" * 62)

    rows = [
        ("Primary Accuracy", bm["primary_accuracy"], am["primary_accuracy"], "higher"),
        ("Macro F1",         bm["macro_f1"],         am["macro_f1"],         "higher"),
        ("Fallback Rate",    bm["fallback_rate"],     am["fallback_rate"],    "lower"),
    ]
    for label, b_val, a_val, direction in rows:
        if direction == "higher":
            winner = "LLM" if a_val > b_val else ("Rule" if b_val > a_val else "Tie")
        else:
            winner = "LLM" if a_val < b_val else ("Rule" if b_val < a_val else "Tie")
        print(f"  {label:<24} {b_val:>12.3f} {a_val:>12.3f}  {winner:>8}")

    print(f"\n  Per-class F1:")
    print(f"  {'Feature':<28} {'Rule-Based':>12} {'LLM Agents':>12}  Delta")
    print("  " + "-" * 62)
    for feat in FEATURE_LABELS:
        b_f1 = bm["per_class"][feat]["f1"]
        a_f1 = am["per_class"].get(feat, {}).get("f1", 0.0)
        delta = a_f1 - b_f1
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        print(f"  {LABEL_DISPLAY[feat]}  {b_f1:>12.3f} {a_f1:>12.3f}  {arrow} {abs(delta):.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(GROUND_TRUTH_PATH) as f:
        gt_entries = json.load(f)

    print(f"\nRunning rule-based baseline on {len(gt_entries)} tables...\n")

    results  = []
    total_t0 = time.time()

    for i, entry in enumerate(gt_entries, 1):
        db_path  = os.path.join(GROUND_TABLES_DIR, entry["db_file"])
        conn_str = f"sqlite:///{db_path}"
        tname    = entry["table_name"]

        schema = introspect_schema(conn_str)
        if tname not in schema:
            print(f"[{i:>2}] ERROR: '{tname}' not found in {entry['db_file']}")
            continue

        try:
            sample_rows = fetch_rows(conn_str, tname, limit=2)
        except Exception:
            sample_rows = []

        routing         = route_table(tname, schema[tname], sample_rows)
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

        tick = "✓" if primary_correct else "✗"
        print(f"[{i:>2}/40]  {tick}  {tname:<30}  "
              f"expected={exp_feat:<18}  predicted={routing['primary_feature']}")

    total_time = time.time() - total_t0
    metrics    = compute_metrics(results)
    cm         = confusion_matrix(results)

    print_per_table(results)
    print_confusion_matrix(cm)
    print_per_class(metrics["per_class"])
    print_errors(results)
    print_comparison(metrics)

    print_banner("BASELINE SUMMARY")
    print(f"  System           : Rule-Based Heuristics (no LLM)")
    print(f"  Tables evaluated : {metrics['total']}")
    print(f"  Primary Accuracy : {metrics['primary_accuracy']:.1%}")
    print(f"  Macro F1         : {metrics['macro_f1']:.3f}")
    print(f"  Fallback Rate    : 0%  (deterministic — never falls back)")
    print(f"  Total Time       : {total_time:.2f}s")

    with open(BASELINE_OUT_PATH, "w") as f:
        json.dump({"metrics": metrics, "confusion_matrix": cm, "results": results}, f, indent=2)
    print(f"\n  Saved -> {BASELINE_OUT_PATH}\n")


if __name__ == "__main__":
    main()