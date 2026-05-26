"""
validation/run_eval_against_gt.py

Evaluates the SurrealDB multi-agent pipeline against ground truth.

The orchestrator routes each table to a specialist agent whose output
maps to SurrealDB features:
  chroma     -> use_vector   (semantic embedding)
  neo4j      -> use_graph    (RELATE edges)
  mongo      -> use_nested   (document nesting)
  relational -> index_cols   (DEFINE INDEX)

Evaluation checks whether the agent activated the correct primary
SurrealDB feature, not the old database name.

Ground truth is read from expected_db (always present).
expected_features is used if present, otherwise derived on the fly.

Run from project root:
    python -m validation.run_eval_against_gt [--only-high]

Results saved to validation/eval_results.json.
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.introspect import introspect_schema, fetch_rows
from core.orchestrator import plan_embeddings

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")
GROUND_TABLES_DIR = os.path.join(os.path.dirname(__file__), "ground_tables")
EVAL_OUT_PATH     = os.path.join(os.path.dirname(__file__), "eval_results.json")

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


# ---------------------------------------------------------------------------
# Derive expected feature — works with or without expected_features in JSON
# ---------------------------------------------------------------------------

def get_expected_feature(entry: dict) -> str:
    if "expected_features" in entry:
        return entry["expected_features"]["primary_feature"]
    return LABEL_TO_FEATURE.get(entry["expected_db"], "index_cols")


# ---------------------------------------------------------------------------
# Feature extraction — handles both old and new agent output shapes
# ---------------------------------------------------------------------------

def extract_features(final: dict) -> dict[str, bool]:
    """
    Translate specialist agent output to SurrealDB feature flags.

    New agent_surrealdb outputs use_vector/use_graph/use_nested/index_cols directly.
    Old specialist agents output target_db string — we map that to the feature.
    """
    if "use_vector" in final or "use_graph" in final or "use_nested" in final:
        return {
            "use_vector": bool(final.get("use_vector")),
            "use_graph":  bool(final.get("use_graph")),
            "use_nested": bool(final.get("use_nested")),
            "index_cols": bool(final.get("index_cols")),
        }

    target = str(final.get("target_db", "")).lower()
    return {
        "use_vector": target == "chroma",
        "use_graph":  target == "neo4j",
        "use_nested": target == "mongo",
        "index_cols": target == "relational",
    }


def _dominant_feature(feat: dict[str, bool]) -> str:
    for f in ["use_vector", "use_graph", "use_nested", "index_cols"]:
        if feat.get(f):
            return f
    return "index_cols"


# ---------------------------------------------------------------------------
# Single-table eval
# ---------------------------------------------------------------------------

def run_single(db_file: str, table_name: str) -> dict:
    db_path  = os.path.join(GROUND_TABLES_DIR, db_file)
    conn_str = f"sqlite:///{db_path}"
    t0       = time.time()

    schema = introspect_schema(conn_str)
    if table_name not in schema:
        return {"error": f"Table '{table_name}' not found in {db_file}"}

    try:
        sample_rows = fetch_rows(conn_str, table_name, limit=2)
    except Exception:
        sample_rows = []

    strategies = plan_embeddings(
        {table_name: schema[table_name]},
        sample_rows_map={table_name: sample_rows},
    )

    elapsed = time.time() - t0
    final   = strategies.get(table_name) or {}
    feats   = extract_features(final)

    # Detail counts for advanced metrics — handle both agent shapes
    if "use_vector" in final:
        vector_cols_count = len(final.get("vector_cols", []))
        nested_keys_count = len((final.get("nested_fields") or {}).keys())
        relations_count   = len(final.get("relations", []))
        index_cols_count  = len(final.get("index_cols", []))
    else:
        target = str(final.get("target_db", "")).lower()
        used   = final.get("used_columns", [])
        vector_cols_count = len(used) if target == "chroma" else 0
        nested_keys_count = len((final.get("nested_fields") or {}).keys())
        relations_count   = len(final.get("join_related", []))
        index_cols_count  = len(used) if target == "relational" else 0

    return {
        "features":          feats,
        "fallback_used":     final.get("fallback_used", False),
        "elapsed_s":         round(elapsed, 1),
        "retries":           final.get("retries", 0),
        "reasoning":         final.get("reasoning", final.get("routing_reason", "N/A")),
        "total_columns":     len(schema[table_name]["columns"]),
        "vector_cols_count": vector_cols_count,
        "nested_keys_count": nested_keys_count,
        "relations_count":   relations_count,
        "index_cols_count":  index_cols_count,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    total     = len(results)
    fallbacks = sum(1 for r in results if r["fallback_used"])

    primary_correct = sum(1 for r in results if r["primary_correct"])
    fully_correct   = sum(1 for r in results if r.get("fully_correct", r["primary_correct"]))

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for r in results:
        exp  = r["expected_primary_feature"]
        pred = _dominant_feature(r["features"])
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

    retried   = sum(1 for r in results if r.get("retries", 0) > 0)
    recovered = sum(1 for r in results if r.get("retries", 0) > 0 and not r["fallback_used"])
    scr       = (recovered / retried) if retried > 0 else 1.0

    vec_r = [r for r in results if r["features"].get("use_vector")]
    gph_r = [r for r in results if r["features"].get("use_graph")]
    nst_r = [r for r in results if r["features"].get("use_nested")]
    idx_r = [r for r in results if r["features"].get("index_cols")]

    def avg(lst, key):
        return round(sum(r[key] for r in lst) / len(lst), 2) if lst else 0.0

    return {
        "total":            total,
        "primary_accuracy": round(primary_correct / total, 3) if total else 0,
        "full_accuracy":    round(fully_correct   / total, 3) if total else 0,
        "macro_f1":         round(macro_f1, 3),
        "fallback_rate":    round(fallbacks / total, 3) if total else 0,
        "per_class":        per_class,
        "feature_activation": {
            "use_vector_rate": round(len(vec_r) / total, 3),
            "use_graph_rate":  round(len(gph_r) / total, 3),
            "use_nested_rate": round(len(nst_r) / total, 3),
            "index_cols_rate": round(len(idx_r) / total, 3),
        },
        "advanced": {
            "self_correction_rate": round(scr, 3),
            "avg_vector_cols":      avg(vec_r, "vector_cols_count"),
            "avg_nested_keys":      avg(nst_r, "nested_keys_count"),
            "avg_graph_relations":  avg(gph_r, "relations_count"),
            "avg_index_cols":       avg(idx_r, "index_cols_count"),
        },
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(metrics: dict, total_time: float):
    w = 66
    print("\n" + "=" * w)
    print("  SURREALDB PIPELINE — EVAL RESULTS")
    print("=" * w)
    print(f"  Tables        : {metrics['total']}")
    print(f"  Primary Acc   : {metrics['primary_accuracy']:.1%}  (correct SurrealDB feature activated)")
    print(f"  Full Acc      : {metrics['full_accuracy']:.1%}  (all constraints satisfied)")
    print(f"  Macro F1      : {metrics['macro_f1']:.3f}")
    print(f"  Fallback Rate : {metrics['fallback_rate']:.1%}")

    print("\n  Feature Activation Rates:")
    fa = metrics["feature_activation"]
    for feat, key in [("use_vector","use_vector_rate"), ("use_graph","use_graph_rate"),
                      ("use_nested","use_nested_rate"), ("index_cols","index_cols_rate")]:
        print(f"    {LABEL_DISPLAY[feat]} : {fa[key]:.1%}")

    print("\n  Per-Class Metrics (primary feature):")
    print(f"    {'Feature':<28} {'P':>7} {'R':>7} {'F1':>7}  TP  FP  FN")
    print("    " + "-" * 60)
    for feat in FEATURE_LABELS:
        m = metrics["per_class"][feat]
        print(f"    {LABEL_DISPLAY[feat]}  {m['precision']:>7.3f} {m['recall']:>7.3f} "
              f"{m['f1']:>7.3f}  {m['tp']:>2}  {m['fp']:>2}  {m['fn']:>2}")

    print("\n  Advanced SurrealDB Metrics:")
    adv = metrics["advanced"]
    print(f"    Self-Correction Rate : {adv['self_correction_rate']:.1%}")
    print(f"    Avg vector_cols      : {adv['avg_vector_cols']:.2f}")
    print(f"    Avg nested_keys      : {adv['avg_nested_keys']:.2f}")
    print(f"    Avg graph relations  : {adv['avg_graph_relations']:.2f}")
    print(f"    Avg index_cols       : {adv['avg_index_cols']:.2f}")
    print(f"\n  Total Time : {total_time:.1f}s")
    print("=" * w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-high", action="store_true",
                        help="Evaluate only high-confidence ground-truth entries.")
    args = parser.parse_args()

    with open(GROUND_TRUTH_PATH) as f:
        gt_entries = json.load(f)
    if args.only_high:
        gt_entries = [e for e in gt_entries if e["confidence"] == "high"]

    results  = []
    total_t0 = time.time()

    for i, entry in enumerate(gt_entries, 1):
        print(f"[{i:>2}/{len(gt_entries)}] {entry['table_name']} ...")
        res = run_single(entry["db_file"], entry["table_name"])
        if "error" in res:
            print(f"  ERROR: {res['error']}")
            continue

        exp_feat        = get_expected_feature(entry)
        predicted_feat  = _dominant_feature(res["features"])
        primary_correct = predicted_feat == exp_feat

        res.update({
            "table_name":               entry["table_name"],
            "confidence":               entry["confidence"],
            "origin_label":             entry["expected_db"],
            "expected_primary_feature": exp_feat,
            "primary_correct":          primary_correct,
            "fully_correct":            primary_correct,  # simple form; extend if needed
        })
        results.append(res)

        tick   = "✓" if primary_correct else "✗"
        active = [f for f, v in res["features"].items() if v]
        print(f"  {tick}  [{entry['expected_db']:>10} -> {exp_feat}]  active={active}")

    metrics    = compute_metrics(results)
    total_time = time.time() - total_t0
    print_summary(metrics, total_time)

    with open(EVAL_OUT_PATH, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)
    print(f"\n  Saved -> {EVAL_OUT_PATH}\n")


if __name__ == "__main__":
    main()