import os
import sys
import json
import time
import argparse
from collections import defaultdict

# Ensure core.* imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.introspect import introspect_schema, fetch_rows
from core.orchestrator import plan_embeddings

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")
GROUND_TABLES_DIR = os.path.join(os.path.dirname(__file__), "ground_tables")
LABELS = ["chroma", "mongo", "neo4j", "relational"]

def load_ground_truth(only_high: bool = False) -> list[dict]:
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)
    if only_high:
        gt = [e for e in gt if e["confidence"] == "high"]
    return gt

def run_single(db_file: str, table_name: str) -> dict:
    db_path = os.path.join(GROUND_TABLES_DIR, db_file)
    conn_str = f"sqlite:///{db_path}"
    t0 = time.time()

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
    final = strategies.get(table_name, {})
    
    target_db = final.get("target_db", "unknown").lower()
    used_cols = final.get("used_columns", final.get("fields", []))
    all_source_cols = [c["name"] for c in schema[table_name]["columns"]]

    return {
        "predicted_db": target_db,
        "routing_reason": final.get("routing_reason", final.get("reasoning", "N/A")),
        "fallback_used": final.get("fallback_used", False),
        "elapsed_s": round(elapsed, 1),
        "retries": final.get("retries", 0),
        "used_columns_count": len(used_cols),
        "total_columns_count": len(all_source_cols),
        "nested_fields_count": len(final.get("nested_fields", {}).keys()) if target_db == "mongo" else 0,
        "text_columns_used": len([c for c in used_cols if any(x in str(c).upper() for x in ["TEXT", "VARCHAR", "CHAR"])]) if target_db == "chroma" else 0
    }

def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r["predicted_db"] == r["expected_db"])
    fallbacks = sum(1 for r in results if r["fallback_used"])

    # 1. CLASSIC ML METRICS
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for r in results:
        pred, exp = r["predicted_db"], r["expected_db"]
        if pred == exp: tp[exp] += 1
        else: fp[pred] += 1; fn[exp] += 1

    per_class = {}
    for label in LABELS:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        rc = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1 = 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0
        per_class[label] = {"precision": round(p, 3), "recall": round(rc, 3), "f1": round(f1, 3), "tp": tp[label], "fp": fp[label], "fn": fn[label]}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(LABELS)

    # 2. ADVANCED AGENT METRICS
    tables_with_retries = sum(1 for r in results if r.get("retries", 0) > 0)
    recovered = sum(1 for r in results if r.get("retries", 0) > 0 and not r["fallback_used"])
    scr = (recovered / tables_with_retries) if tables_with_retries > 0 else 1.0

    mongo_res = [r for r in results if r["predicted_db"] == "mongo"]
    avg_denorm = sum(r["nested_fields_count"] / max(1, r["used_columns_count"]) for r in mongo_res) / len(mongo_res) if mongo_res else 0.0

    chroma_res = [r for r in results if r["predicted_db"] == "chroma"]
    avg_sem = sum(r["text_columns_used"] / max(1, r["used_columns_count"]) for r in chroma_res) / len(chroma_res) if chroma_res else 0.0

    avg_noise = sum((r["total_columns_count"] - r["used_columns_count"]) / max(1, r["total_columns_count"]) for r in results) / total

    return {
        "total": total, "accuracy": round(correct / total, 3), "macro_f1": round(macro_f1, 3), "fallback_rate": round(fallbacks / total, 3),
        "per_class": per_class,
        "advanced": {
            "self_correction_rate": round(scr, 3), "mongo_denorm_depth": round(avg_denorm, 3),
            "chroma_semantic_density": round(avg_sem, 3), "global_noise_reduction": round(avg_noise, 3)
        }
    }

def print_summary(metrics: dict, total_time: float):
    print("\n" + "="*60 + "\n  EVALUATION SUMMARY\n" + "="*60)
    print(f"Tables: {metrics['total']} | Acc: {metrics['accuracy']:.1%} | Macro F1: {metrics['macro_f1']:.3f} | Fallback: {metrics['fallback_rate']:.1%}")
    
    print("\n--- CLASSIC PERFORMANCE ---")
    for label in LABELS:
        m = metrics['per_class'][label]
        print(f"[{label:10s}] Precision: {m['precision']:.3f} | Recall: {m['recall']:.3f} | F1: {m['f1']:.3f}")

    print("\n--- ADVANCED ARCHITECTURE ---")
    adv = metrics['advanced']
    print(f"Self-Correction: {adv['self_correction_rate']:.1%} (Recovered from hallucinations)")
    print(f"Semantic Density: {adv['chroma_semantic_density']:.1%} (Text quality in Vector DB)")
    print(f"Denorm Depth: {adv['mongo_denorm_depth']:.3f} (Nesting usage in Document DB)")
    print(f"Noise Reduction: {adv['global_noise_reduction']:.1%} (Cleaned redundant legacy cols)")
    print(f"\nTotal Time: {total_time:.1f}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-high", action="store_true")
    args = parser.parse_args()

    gt_entries = load_ground_truth(args.only_high)
    results = []
    total_t0 = time.time()

    for i, entry in enumerate(gt_entries, 1):
        print(f"[{i}/{len(gt_entries)}] Evaluating {entry['table_name']}...")
        res = run_single(entry["db_file"], entry["table_name"])
        if "error" in res: continue
        res.update({"expected_db": entry["expected_db"], "confidence": entry["confidence"]})
        results.append(res)

    metrics = compute_metrics(results)
    print_summary(metrics, time.time() - total_t0)

if __name__ == "__main__":
    main()