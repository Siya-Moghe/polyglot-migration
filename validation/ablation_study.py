"""
validation/ablation_study.py

Ablation study for the SurrealDB multi-agent pipeline.

Runs 5 system variants on the 40-table eval set and shows how each
design decision contributes to final feature-activation accuracy.

"Correct" = agent activated the right primary SurrealDB feature:
  chroma     -> use_vector
  neo4j      -> use_graph
  mongo      -> use_nested
  relational -> index_cols

Variants:
  V1 — Full system     (JSON mode ON, few-shot ON, enrichment ON)
  V2 — w/o JSON mode   (plain text, regex extraction)
  V3 — w/o few-shot    (routing prompt stripped of examples)
  V4 — w/o enrichment  (raw col/FK lines only)
  V5 — Weaker model    (phi3:mini, everything else full)

AIS: % of tables where predicted feature is stable under a biasing prompt.

Run from project root:
    python -m validation.ablation_study [--only-high] [--variant V3]
"""

import os
import sys
import json
import time
import argparse
import importlib
from collections import defaultdict
from contextlib import contextmanager
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")
GROUND_TABLES_DIR = os.path.join(os.path.dirname(__file__), "ground_tables")
ABLATION_OUT_PATH = os.path.join(os.path.dirname(__file__), "ablation_results.json")

# New SurrealDB feature labels — these are the keys used everywhere
FEATURE_LABELS = ["use_vector", "use_graph", "use_nested", "index_cols"]

LABEL_DISPLAY = {
    "use_vector": "use_vector  (was Chroma)",
    "use_graph":  "use_graph   (was Neo4j) ",
    "use_nested": "use_nested  (was Mongo) ",
    "index_cols": "index_cols  (was SQL)   ",
}

# Old routing label -> SurrealDB primary feature
LABEL_TO_FEATURE = {
    "chroma":     "use_vector",
    "neo4j":      "use_graph",
    "mongo":      "use_nested",
    "relational": "index_cols",
}

BIAS_PROMPT = (
    "Note: The enterprise has heavily invested in relational infrastructure. "
    "Prioritise keeping data in the relational SQL engine wherever possible."
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_expected_feature(entry: dict) -> str:
    """Derive expected SurrealDB feature from ground truth entry.
    Works whether or not expected_features key exists in the JSON."""
    if "expected_features" in entry:
        return entry["expected_features"]["primary_feature"]
    return LABEL_TO_FEATURE.get(entry["expected_db"], "index_cols")


def extract_features(final: dict) -> dict[str, bool]:
    """Translate agent output to SurrealDB feature flags.
    Handles both new agent_surrealdb shape and old specialist shape."""
    if "use_vector" in final or "use_graph" in final or "use_nested" in final:
        return {
            "use_vector": bool(final.get("use_vector")),
            "use_graph":  bool(final.get("use_graph")),
            "use_nested": bool(final.get("use_nested")),
            "index_cols": bool(final.get("index_cols")),
        }
    # Old specialist agents: derive from target_db routing label
    target = str(final.get("target_db", "")).lower()
    return {
        "use_vector": target == "chroma",
        "use_graph":  target == "neo4j",
        "use_nested": target == "mongo",
        "index_cols": target == "relational",
    }


def _dominant_feature(feat: dict[str, bool]) -> str:
    """Return the single strongest active feature."""
    for f in FEATURE_LABELS:
        if feat.get(f):
            return f
    return "index_cols"


# ---------------------------------------------------------------------------
# Core eval runner
# ---------------------------------------------------------------------------

def run_variant_eval(gt_entries: list[dict], human_context: str = "") -> list[dict]:
    """Run the currently-patched orchestrator on all entries."""
    import core.orchestrator as orch_mod
    importlib.reload(orch_mod)
    from core.introspect import introspect_schema, fetch_rows

    results = []
    for entry in gt_entries:
        db_path  = os.path.join(GROUND_TABLES_DIR, entry["db_file"])
        conn_str = f"sqlite:///{db_path}"
        tname    = entry["table_name"]

        schema = introspect_schema(conn_str)
        if tname not in schema:
            print(f"    SKIP: {tname} not found in {entry['db_file']}")
            continue

        try:
            sample_rows = fetch_rows(conn_str, tname, limit=2)
        except Exception:
            sample_rows = []

        t0 = time.time()
        strategies = orch_mod.plan_embeddings(
            {tname: schema[tname]},
            sample_rows_map={tname: sample_rows},
            human_context=human_context,
        )
        elapsed = round(time.time() - t0, 1)

        final          = strategies.get(tname) or {}
        features       = extract_features(final)
        predicted_feat = _dominant_feature(features)
        expected_feat  = get_expected_feature(entry)
        correct        = predicted_feat == expected_feat

        results.append({
            "table_name":               tname,
            "expected_primary_feature": expected_feat,   # always a feature label
            "predicted_feature":        predicted_feat,  # always a feature label
            "primary_correct":          correct,
            "fallback_used":            final.get("fallback_used", False),
            "elapsed_s":                elapsed,
        })

        tick = "✓" if correct else "✗"
        print(f"    {tick}  {tname:<30}  "
              f"exp={expected_feat:<18}  pred={predicted_feat}")

    return results


# ---------------------------------------------------------------------------
# Metrics — keyed on FEATURE_LABELS, never on old DB names
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    total     = len(results)
    if total == 0:
        return {"total": 0, "accuracy": 0, "macro_f1": 0,
                "fallback_rate": 0, "avg_time_s": 0,
                "per_class": {f: {"precision":0,"recall":0,"f1":0} for f in FEATURE_LABELS}}

    correct   = sum(1 for r in results if r["primary_correct"])
    fallbacks = sum(1 for r in results if r["fallback_used"])
    avg_time  = round(sum(r["elapsed_s"] for r in results) / total, 1)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for r in results:
        pred = r["predicted_feature"]   # e.g. "use_vector"
        exp  = r["expected_primary_feature"]  # e.g. "use_vector"
        if pred == exp:
            tp[exp] += 1
        else:
            fp[pred] += 1
            fn[exp]  += 1

    per_class = {}
    for feat in FEATURE_LABELS:
        p  = tp[feat] / (tp[feat] + fp[feat]) if (tp[feat] + fp[feat]) > 0 else 0.0
        rc = tp[feat] / (tp[feat] + fn[feat]) if (tp[feat] + fn[feat]) > 0 else 0.0
        f1 = 2 * p * rc / (p + rc)            if (p + rc) > 0             else 0.0
        per_class[feat] = {
            "precision": round(p, 3),
            "recall":    round(rc, 3),
            "f1":        round(f1, 3),
        }

    macro_f1 = round(sum(v["f1"] for v in per_class.values()) / len(FEATURE_LABELS), 3)

    return {
        "total":         total,
        "accuracy":      round(correct / total, 3),
        "macro_f1":      macro_f1,
        "fallback_rate": round(fallbacks / total, 3),
        "avg_time_s":    avg_time,
        "per_class":     per_class,
    }


def calculate_ais(neutral: list[dict], biased: list[dict]) -> float:
    if not neutral:
        return 0.0
    return sum(
        1 for n, b in zip(neutral, biased)
        if n["predicted_feature"] == b["predicted_feature"]
    ) / len(neutral)


# ---------------------------------------------------------------------------
# Context manager for stacking patches
# ---------------------------------------------------------------------------

@contextmanager
def _apply_patches(patches):
    started = []
    try:
        for p in patches:
            p.start()
            started.append(p)
        yield
    finally:
        for p in reversed(started):
            p.stop()


def _get_agent_modules():
    mods = []
    for mod_name in [
        "core.agents.agent_vector",
        "core.agents.agent_graph",
        "core.agents.agent_document",
        "core.agents.agent_index",
        "core.agent_surrealdb",
    ]:
        try:
            mods.append(importlib.import_module(mod_name))
        except ImportError:
            pass
    return mods


# ---------------------------------------------------------------------------
# Variant-specific prompt builders (V3/V4)
# ---------------------------------------------------------------------------

def _bare_table_context(table_name: str, info: dict) -> str:
    col_lines = "\n".join(f"  - {c['name']} ({c['type']})" for c in info["columns"])
    fk_lines  = "\n".join(
        f"  - {fk.get('columns', [])} -> {fk['referred_table']}"
        for fk in info.get("foreign_keys", [])
    ) or "  None"
    return f"Table: {table_name}\nColumns:\n{col_lines}\nForeign Keys:\n{fk_lines}"


def _no_examples_prompt(enriched: str, human_context: str = "") -> str:
    ctx = f"\nExpert Context: {human_context}" if human_context else ""
    return f"""You are a SurrealDB Architect. Decide which SurrealDB features to activate.

{enriched}{ctx}

Features:
- use_vector : tables with semantic/long-text columns
- use_graph  : tables with FK columns (FK -> RELATE edges)
- use_nested : self-contained entities with no FKs
- index_cols : transactional/numeric/temporal tables

Return ONLY JSON:
{{
  "use_vector": true or false,
  "use_graph":  true or false,
  "use_nested": true or false,
  "index_cols": [],
  "reasoning":  "One sentence."
}}"""


def _no_enrich_prompt(table_name: str, info: dict, human_context: str = "") -> str:
    bare = _bare_table_context(table_name, info)
    ctx  = f"\nExpert Context: {human_context}" if human_context else ""
    return f"""You are a SurrealDB Architect. Decide which SurrealDB features to activate.

{bare}{ctx}

Features:
- use_vector : semantic/long-text columns
- use_graph  : FK columns -> RELATE edges
- use_nested : no-FK entity tables
- index_cols : transactional/numeric/temporal

Examples:
- knowledge_base (full_content LONG TEXT, author_id FK)
  -> {{"use_vector": true, "use_graph": true, "use_nested": false, "index_cols": ["id"], "reasoning": "..."}}
- employee_projects (emp_id FK, proj_id FK, allocation INT)
  -> {{"use_vector": false, "use_graph": true, "use_nested": false, "index_cols": [], "reasoning": "..."}}
- departments (name TEXT, cost_center TEXT, no FKs)
  -> {{"use_vector": false, "use_graph": false, "use_nested": true, "index_cols": ["id"], "reasoning": "..."}}
- support_tickets (status TEXT, priority TEXT, created_at TIMESTAMP)
  -> {{"use_vector": false, "use_graph": false, "use_nested": false, "index_cols": ["id","status","created_at"], "reasoning": "..."}}

Return ONLY JSON matching the exact format above."""


def _make_patched_node(prompt_fn):
    """Returns a drop-in node_orchestrator that uses a custom prompt builder
    but still writes target_db so the existing LangGraph routing works."""
    def node(state):
        from core.llm_utils import ask_ollama, extract_json
        tname  = state["table_name"]
        prompt = prompt_fn(state)

        print(f"  [orchestrator] patched routing for '{tname}'...")
        raw = ask_ollama(prompt, json_mode=True)
        try:
            result = extract_json(raw)
        except ValueError:
            result = {}

        feat = result or {}
        dominant = _dominant_feature({
            "use_vector": bool(feat.get("use_vector")),
            "use_graph":  bool(feat.get("use_graph")),
            "use_nested": bool(feat.get("use_nested")),
            "index_cols": bool(feat.get("index_cols")),
        })
        feature_to_label = {v: k for k, v in LABEL_TO_FEATURE.items()}
        target = feature_to_label.get(dominant, "relational")
        print(f"  [orchestrator] patched routed '{tname}' -> {target.upper()}")
        return {**state,
                "target_db":      target,
                "routing_reason": result.get("reasoning", "N/A"),
                "strategy":       result}
    return node


# ---------------------------------------------------------------------------
# All variants
# ---------------------------------------------------------------------------

def run_all_variants(gt_entries: list[dict], selected: str = None) -> dict:
    import core.llm_utils    as llm_mod
    import core.orchestrator as orch_mod

    all_metrics = {}

    def _run_pair(label):
        print(f"  --> Neutral pass...")
        neu = run_variant_eval(gt_entries, human_context="")
        print(f"  --> Biased pass (AIS)...")
        bia = run_variant_eval(gt_entries, human_context=BIAS_PROMPT)
        m       = compute_metrics(neu)
        m["ais"] = round(calculate_ais(neu, bia), 3)
        all_metrics[label] = m
        print(f"  {label}: acc={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}  ais={m['ais']:.3f}")

    # ── V1: Full system ────────────────────────────────────────────────────
    if not selected or selected == "V1":
        print_banner("V1 — Full System")
        _run_pair("V1")

    # ── V2: No JSON mode ───────────────────────────────────────────────────
    if not selected or selected == "V2":
        print_banner("V2 — w/o JSON mode")
        orig_ask = llm_mod.ask_ollama
        def ask_no_json(prompt, json_mode=True):
            return orig_ask(prompt, json_mode=False)
        patches = [
            patch.object(llm_mod,  "ask_ollama", ask_no_json),
            patch.object(orch_mod, "ask_ollama", ask_no_json),
        ]
        for m in _get_agent_modules():
            if hasattr(m, "ask_ollama"):
                patches.append(patch.object(m, "ask_ollama", ask_no_json))
        with _apply_patches(patches):
            _run_pair("V2")

    # ── V3: No few-shot examples ───────────────────────────────────────────
    if not selected or selected == "V3":
        print_banner("V3 — w/o Few-Shot Examples")
        def prompt_no_examples(state):
            from core.llm_utils import enrich_table_context
            enriched = enrich_table_context(
                state["table_name"], state["table_info"], state.get("sample_rows", [])
            )
            return _no_examples_prompt(enriched, state.get("human_context", ""))
        with patch.object(orch_mod, "node_orchestrator", _make_patched_node(prompt_no_examples)):
            orch_mod._GRAPH = orch_mod._build_graph()
            _run_pair("V3")
            orch_mod._GRAPH = orch_mod._build_graph()

    # ── V4: No schema enrichment ───────────────────────────────────────────
    if not selected or selected == "V4":
        print_banner("V4 — w/o Schema Enrichment")
        def prompt_no_enrich(state):
            return _no_enrich_prompt(
                state["table_name"], state["table_info"], state.get("human_context", "")
            )
        with patch.object(orch_mod, "node_orchestrator", _make_patched_node(prompt_no_enrich)):
            orch_mod._GRAPH = orch_mod._build_graph()
            _run_pair("V4")
            orch_mod._GRAPH = orch_mod._build_graph()

    # ── V5: Weaker model ───────────────────────────────────────────────────
    if not selected or selected == "V5":
        print_banner("V5 — Weaker Model (phi3:mini)")
        def ask_phi3(prompt, json_mode=True):
            import requests
            payload = {"model": "phi3:mini", "prompt": prompt, "stream": False}
            if json_mode:
                payload["format"] = "json"
            try:
                resp = requests.post(llm_mod.OLLAMA_URL, json=payload, timeout=180)
                if resp.status_code != 200:
                    return "{}"
                return resp.json()["response"].strip()
            except Exception as exc:
                print(f"\n[ERROR] {exc}\n")
                return "{}"
        patches = [
            patch.object(llm_mod,  "ask_ollama", ask_phi3),
            patch.object(orch_mod, "ask_ollama", ask_phi3),
        ]
        for m in _get_agent_modules():
            if hasattr(m, "ask_ollama"):
                patches.append(patch.object(m, "ask_ollama", ask_phi3))
        with _apply_patches(patches):
            _run_pair("V5")

    return all_metrics


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_banner(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_summary_table(all_metrics: dict):
    print_banner("ABLATION STUDY — SUMMARY TABLE (WITH AIS)")

    descriptions = {
        "V1": "Full system (JSON mode + few-shot + enrichment + current model)",
        "V2": "w/o JSON mode       (plain text output, regex extraction)",
        "V3": "w/o few-shot        (no examples in routing prompt)",
        "V4": "w/o enrichment      (raw col/FK lines, no hints/samples)",
        "V5": "Original model      (phi3:mini, everything else full)",
    }

    print(f"\n  {'Variant':<8} {'Accuracy':>10} {'Macro F1':>10} {'AIS':>8} "
          f"{'Fallback%':>10} {'AvgTime':>9}  Description")
    print("  " + "-" * 110)
    for vname, m in all_metrics.items():
        marker = "  <-baseline" if vname == "V1" else ""
        ais_str = f"{m.get('ais', 0.0):.1%}"
        print(f"  {vname:<8} {m['accuracy']:>10.3f} {m['macro_f1']:>10.3f} "
              f"{ais_str:>8} {m['fallback_rate']:>10.1%} "
              f"{m['avg_time_s']:>8.1f}s  {descriptions.get(vname,'')}{marker}")

    # Per-class F1 table
    print(f"\n  Per-class F1:")
    col_w = 16
    header = f"  {'Variant':<8}" + "".join(f"{LABEL_DISPLAY[f][:14]:>{col_w}}" for f in FEATURE_LABELS)
    print(header)
    print("  " + "-" * (8 + col_w * len(FEATURE_LABELS)))
    for vname, m in all_metrics.items():
        row = f"  {vname:<8}"
        for feat in FEATURE_LABELS:
            f1 = m["per_class"].get(feat, {}).get("f1", 0.0)
            row += f"{f1:>{col_w}.3f}"
        print(row)

    # Delta vs V1
    if "V1" in all_metrics:
        v1 = all_metrics["V1"]
        print(f"\n  Delta vs V1 (full system) — positive = V1 is better:")
        print(f"  {'Variant':<8} {'Acc':>12} {'Macro F1':>12} "
              f"{'AIS':>10} {'Fallback%':>12}")
        print("  " + "-" * 58)
        for vname, m in all_metrics.items():
            if vname == "V1":
                continue
            d_acc  = v1["accuracy"]      - m["accuracy"]
            d_f1   = v1["macro_f1"]      - m["macro_f1"]
            d_ais  = v1.get("ais", 0)    - m.get("ais", 0)
            d_fall = m["fallback_rate"]  - v1["fallback_rate"]
            print(f"  {vname:<8} {d_acc:>+12.3f} {d_f1:>+12.3f} "
                  f"{d_ais:>+10.3f} {d_fall:>+12.3f}")


def print_per_class_detail(vname: str, m: dict):
    print(f"\n  {vname} — Per-class breakdown:")
    print(f"    {'Feature':<28} {'P':>10} {'R':>8} {'F1':>8}")
    print("    " + "-" * 58)
    for feat in FEATURE_LABELS:
        pc = m["per_class"].get(feat, {"precision": 0, "recall": 0, "f1": 0})
        print(f"    {LABEL_DISPLAY[feat]}  "
              f"{pc['precision']:>10.3f} {pc['recall']:>8.3f} {pc['f1']:>8.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-high", action="store_true")
    parser.add_argument("--variant", type=str, default=None,
                        choices=["V1", "V2", "V3", "V4", "V5"])
    args = parser.parse_args()

    with open(GROUND_TRUTH_PATH) as f:
        gt_entries = json.load(f)
    if args.only_high:
        gt_entries = [e for e in gt_entries if e["confidence"] == "high"]

    print(f"\nAblation study — {len(gt_entries)} tables, "
          f"variants: {args.variant or 'V1 V2 V3 V4 V5'}\n")

    all_metrics = run_all_variants(gt_entries, selected=args.variant)

    for vname, m in all_metrics.items():
        print_per_class_detail(vname, m)

    print_summary_table(all_metrics)

    with open(ABLATION_OUT_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Saved -> {ABLATION_OUT_PATH}\n")


if __name__ == "__main__":
    main()
