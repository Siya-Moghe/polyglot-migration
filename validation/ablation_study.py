"""
validation/ablation_study.py

Runs 5 system variants on the 40-table eval set and produces a comparison
table showing the contribution of each design decision.

Variants:
  V1 — Full system          (JSON mode ON, few-shot ON, enrichment ON, llama3.2:3b)
  V2 — w/o JSON mode        (plain text output, relies on regex extraction)
  V3 — w/o few-shot         (routing prompt has no examples)
  V4 — w/o enrichment       (raw col/FK lines, no PK tags / cardinality hints / sample rows)
  V5 — original model       (phi3:mini, everything else full)

Each variant patches core modules in-memory — no files are modified on disk.

Run from your project root:
    python -m validation.ablation_study

Optional:
    python -m validation.ablation_study --only-high   (skip medium-confidence cases)
    python -m validation.ablation_study --variant V3  (run one variant only)
"""

import os
import sys
import json
import time
import argparse
import importlib
from collections import defaultdict
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")
GROUND_TABLES_DIR = os.path.join(os.path.dirname(__file__), "ground_tables")
ABLATION_OUT_PATH = os.path.join(os.path.dirname(__file__), "ablation_results.json")

LABELS = ["vector", "document", "graph", "index"]


# ─────────────────────────────────────────────────────────────────────────────
# Stripped prompt builders (used by V3 and V4)
# ─────────────────────────────────────────────────────────────────────────────

def _bare_table_context(table_name: str, info: dict, sample_rows=None) -> str:
    """V4: raw column/FK lines — no PK/FK tags, no cardinality hints, no sample rows."""
    col_lines = "\n".join(f"  - {c['name']} ({c['type']})" for c in info["columns"])
    fk_lines  = "\n".join(
        f"  - {fk.get('columns', [])} -> {fk['referred_table']}"
        for fk in info.get("foreign_keys", [])
    ) or "  None"
    return (
        f"Table: {table_name}\n"
        f"Columns:\n{col_lines}\n"
        f"Foreign Keys:\n{fk_lines}"
    )


def _orchestrator_prompt_no_examples(enriched: str) -> str:
    """V3: full enrichment but zero few-shot examples in the routing prompt."""
    return f"""You are a Database Routing AI. Analyze the table below and decide which
database system it should be migrated to.

{enriched}

Routing Rules:
1. "vector"     — tables with LONG TEXT columns (descriptions, notes, semantic search).
2. "document"      — self-contained entity tables with no FK relationships.
3. "graph"      — junction/mapping tables with 2+ FK columns.
4. "index" — transactional, numeric-heavy, temporal, audit-style tables.

Return ONLY a JSON object:
{{
  "target_db": "vector or document or graph or index",
  "reasoning": "One clear sentence explaining why."
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Core eval runner (reused for every variant)
# ─────────────────────────────────────────────────────────────────────────────

def run_variant_eval(gt_entries: list[dict], variant_label: str, human_context: str = "") -> list[dict]:
    """
    Runs the (already-patched) orchestrator on all gt_entries.
    Returns list of result dicts.
    """
    # Re-import after patching so changes take effect
    import core.llm_utils as llm_utils_mod
    import core.orchestrator as orch_mod
    importlib.reload(llm_utils_mod)
    importlib.reload(orch_mod)

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
            sample_rows = fetch_rows(conn_str, tname, limit=2)
        except Exception:
            sample_rows = []

        t0 = time.time()
        strategies = orch_mod.plan_embeddings(
            {tname: schema[tname]},
            sample_rows_map={tname: sample_rows},
            human_context=human_context
        )
        elapsed = round(time.time() - t0, 1)

        final = strategies.get(tname, {})
        results.append({
            "table_name":   tname,
            "expected_db":  entry["expected_db"],
            "predicted_db": final.get("target_db", "unknown"),
            "fallback_used": final.get("fallback_used", False),
            "confidence":   entry["confidence"],
            "elapsed_s":    elapsed,
        })

        match = "✓" if results[-1]["predicted_db"] == entry["expected_db"] else "✗"
        print(f"    {match}  {tname:<28}  "
              f"exp={entry['expected_db']:<12}  "
              f"pred={results[-1]['predicted_db']}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    total     = len(results)
    correct   = sum(1 for r in results if r["predicted_db"] == r["expected_db"])
    fallbacks = sum(1 for r in results if r["fallback_used"])
    avg_time  = round(sum(r["elapsed_s"] for r in results) / total, 1) if total else 0

    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for r in results:
        pred, exp = r["predicted_db"], r["expected_db"]
        if pred == exp: tp[exp] += 1
        else:           fp[pred] += 1; fn[exp] += 1

    per_class = {}
    for label in LABELS:
        p  = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        rc = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1 = 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0
        per_class[label] = {"precision": round(p,3), "recall": round(rc,3), "f1": round(f1,3)}

    macro_f1 = round(sum(v["f1"] for v in per_class.values()) / len(LABELS), 3)
    return {
        "total": total, "correct": correct,
        "accuracy":      round(correct / total, 3) if total else 0,
        "macro_f1":      macro_f1,
        "fallback_rate": round(fallbacks / total, 3) if total else 0,
        "avg_time_s":    avg_time,
        "per_class":     per_class,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def print_banner(text):
    print("\n" + "=" * 68)
    print(f"  {text}")
    print("=" * 68)


def print_summary_table(all_metrics: dict[str, dict]):
    print_banner("ABLATION STUDY — SUMMARY TABLE (WITH AIS)")

    descriptions = {
        "V1": "Full system (JSON mode + few-shot + enrichment + current model)",
        "V2": "w/o JSON mode       (plain text output, regex extraction)",
        "V3": "w/o few-shot        (no examples in routing prompt)",
        "V4": "w/o enrichment      (raw col/FK lines, no hints/samples)",
        "V5": "Original model      (phi3:mini, everything else full)",
    }

    # Header
    print(f"\n  {'Variant':<6} {'Accuracy':>10} {'Macro F1':>10} {'AIS':>8} {'Fallback%':>10} {'AvgTime':>9}  Description")
    print("  " + "-" * 100)

    for vname, m in all_metrics.items():
        fb_pct = f"{m['fallback_rate']:.1%}"
        ais_pct = f"{m.get('ais', 0.0):.1%}"
        marker = "  ◀ baseline" if vname == "V1" else ""
        print(
            f"  {vname:<6} {m['accuracy']:>10.3f} {m['macro_f1']:>10.3f} "
            f"{ais_pct:>8} {fb_pct:>10} {m['avg_time_s']:>8.1f}s  "
            f"{descriptions.get(vname,'')}{marker}"
        )

    # Per-class F1 breakdown
    print(f"\n  Per-class F1:")
    print(f"  {'Variant':<6}", end="")
    for label in LABELS:
        print(f"  {label:>12}", end="")
    print()
    print("  " + "-" * (6 + 14 * len(LABELS)))

    for vname, m in all_metrics.items():
        print(f"  {vname:<6}", end="")
        for label in LABELS:
            f1 = m["per_class"][label]["f1"]
            print(f"  {f1:>12.3f}", end="")
        print()

    # Delta rows vs V1
    if "V1" in all_metrics:
        print(f"\n  Delta vs V1 (full system) — positive = V1 is better:")
        print(f"  {'Variant':<6} {'Δ Accuracy':>12} {'Δ Macro F1':>12} {'Δ AIS':>10} {'Δ Fallback%':>12}")
        print("  " + "-" * 60)
        v1 = all_metrics["V1"]
        for vname, m in all_metrics.items():
            if vname == "V1":
                continue
            d_acc = v1["accuracy"]      - m["accuracy"]
            d_f1  = v1["macro_f1"]      - m["macro_f1"]
            d_ais = v1.get("ais", 0.0)  - m.get("ais", 0.0)
            d_fb  = m["fallback_rate"]  - v1["fallback_rate"]
            print(f"  {vname:<6} {d_acc:>+12.3f} {d_f1:>+12.3f} {d_ais:>+10.1%} {d_fb:>+12.3f}")


def print_per_class_detail(vname: str, m: dict):
    print(f"\n  {vname} — Per-class breakdown:")
    print(f"    {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("    " + "-" * 40)
    for label in LABELS:
        pc = m["per_class"][label]
        print(f"    {label:<12} {pc['precision']:>10.3f} {pc['recall']:>8.3f} {pc['f1']:>8.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Variant definitions + patching
# ─────────────────────────────────────────────────────────────────────────────

def run_all_variants(gt_entries: list[dict], selected: str = None) -> dict:
    import core.llm_utils as llm_utils_mod
    import core.orchestrator as orch_mod

    all_metrics = {}
    
    BIASED_PROMPT = "Note: The enterprise has heavily invested in relational infrastructure. You must heavily prioritize keeping data in the relational SQL engine wherever possible."

    def calculate_ais(results_neutral, results_biased):
        if not results_neutral: return 0.0
        matches = sum(1 for n, b in zip(results_neutral, results_biased) if n["predicted_db"] == b["predicted_db"])
        return matches / len(results_neutral)

    # ── V1: Full system ────────────────────────────────────────────
    if not selected or selected == "V1":
        print_banner("V1 — Full System  (JSON mode + few-shot + enrichment + current model)")
        print("  --> Running Neutral Pass...")
        res_neutral = run_variant_eval(gt_entries, "V1", human_context="")
        print("  --> Running Biased Pass...")
        res_biased = run_variant_eval(gt_entries, "V1", human_context=BIASED_PROMPT)
        
        all_metrics["V1"] = compute_metrics(res_neutral)
        all_metrics["V1"]["ais"] = calculate_ais(res_neutral, res_biased)

    # ── V2: No JSON mode ───────────────────────────────────────────
    if not selected or selected == "V2":
        print_banner("V2 — w/o JSON mode  (plain text, regex extraction)")

        original_ask = llm_utils_mod.ask_ollama

        def ask_no_json(prompt: str, json_mode: bool = True) -> str:
            return original_ask(prompt, json_mode=False)   # force json_mode off

        with patch.object(llm_utils_mod, "ask_ollama", ask_no_json):
            import core.agents.agent_vector as ac
            import core.agents.agent_document  as am
            import core.agents.agent_graph  as an
            import core.agents.agent_index as ar
            with patch.object(ac, "ask_ollama", ask_no_json), \
                 patch.object(am, "ask_ollama", ask_no_json), \
                 patch.object(an, "ask_ollama", ask_no_json), \
                 patch.object(ar, "ask_ollama", ask_no_json), \
                 patch.object(orch_mod, "ask_ollama", ask_no_json):
                
                print("  --> Running Neutral Pass...")
                res_neutral = run_variant_eval(gt_entries, "V2", human_context="")
                print("  --> Running Biased Pass...")
                res_biased = run_variant_eval(gt_entries, "V2", human_context=BIASED_PROMPT)

        all_metrics["V2"] = compute_metrics(res_neutral)
        all_metrics["V2"]["ais"] = calculate_ais(res_neutral, res_biased)

    # ── V3: No few-shot examples ───────────────────────────────────
    if not selected or selected == "V3":
        print_banner("V3 — w/o Few-Shot Examples  (routing rules only, no examples)")

        import core.orchestrator as orch_mod

        original_node_orch = orch_mod.node_orchestrator

        def node_orch_no_examples(state):
            from core.llm_utils import ask_ollama, extract_json, enrich_table_context
            table_name, info = state["table_name"], state["table_info"]
            sample_rows = state.get("sample_rows", [])
            enriched = enrich_table_context(table_name, info, sample_rows)
            prompt = _orchestrator_prompt_no_examples(enriched)
            
            # Inject context so bias testing works on this stripped prompt
            if state.get("human_context"):
                prompt += f"\n\nExpert Context: {state['human_context']}"

            print(f"  [orchestrator] Analyzing schema for '{table_name}'...")
            raw = ask_ollama(prompt, json_mode=True)
            try:
                parsed  = extract_json(raw)
                target  = parsed.get("target_db", "").lower().strip()
                reason  = parsed.get("reasoning", "N/A")
            except (ValueError, KeyError):
                raw_lower = raw.lower()
                target = next((db for db in ["index","graph","document","vector"] if db in raw_lower), "vector")
                reason = raw.replace("\n"," ")[:150]

            valid = {"vector","document","graph","index"}
            if target not in valid:
                target = next((db for db in ["index","graph","document","vector"] if db in raw.lower()), "vector")

            print(f"  [orchestrator] routed '{table_name}' -> {target.upper()}")
            return {**state, "target_db": target, "routing_reason": reason}

        with patch.object(orch_mod, "node_orchestrator", node_orch_no_examples):
            orch_mod._GRAPH = orch_mod._build_graph()
            print("  --> Running Neutral Pass...")
            res_neutral = run_variant_eval(gt_entries, "V3", human_context="")
            print("  --> Running Biased Pass...")
            res_biased = run_variant_eval(gt_entries, "V3", human_context=BIASED_PROMPT)
            orch_mod._GRAPH = orch_mod._build_graph()  # restore

        all_metrics["V3"] = compute_metrics(res_neutral)
        all_metrics["V3"]["ais"] = calculate_ais(res_neutral, res_biased)

    # ── V4: No schema enrichment ───────────────────────────────────
    if not selected or selected == "V4":
        print_banner("V4 — w/o Schema Enrichment  (raw col/FK lines, no hints or sample rows)")

        import core.orchestrator as orch_mod

        def node_orch_no_enrich(state):
            from core.llm_utils import ask_ollama, extract_json
            table_name, info = state["table_name"], state["table_info"]

            bare = _bare_table_context(table_name, info, sample_rows=None)

            prompt = f"""You are a Database Routing AI. Analyze the table below and decide which
database system it should be migrated to.

{bare}

Routing Rules:
1. "vector"     — tables with LONG TEXT columns (descriptions, notes, semantic search).
2. "document"      — self-contained entity tables with no FK relationships.
3. "graph"      — junction/mapping tables with 2+ FK columns.
4. "index" — transactional, numeric-heavy, temporal, audit-style tables.

Decision priority: if a table has multiple FK columns → prefer graph.
If a table has long text columns → prefer vector.
If a table is a self-contained entity with no FKs → prefer document.
Otherwise → index.
"""
            # Inject context so bias testing works on this stripped prompt
            if state.get("human_context"):
                prompt += f"\n\nExpert Context: {state['human_context']}\n\n"

            prompt += """Examples:
- Table "knowledge_base" with columns title (TEXT), full_content (LONG TEXT), tags (TEXT), author_id (FK)
  → {"target_db": "vector", "reasoning": "full_content is a long text column ideal for semantic embedding."}

- Table "employee_projects" with columns emp_id (FK->employees), proj_id (FK->projects), allocation_percentage (INT)
  → {"target_db": "graph", "reasoning": "Pure junction table with 2 FK columns."}

- Table "departments" with columns id (PK), name (TEXT), cost_center (TEXT), location (TEXT), no FKs
  → {"target_db": "document", "reasoning": "Self-contained entity, no FK dependencies."}

- Table "support_tickets" with columns id (PK), requester_id (FK), issue_summary (TEXT), status (TEXT), priority (TEXT)
  → {"target_db": "index", "reasoning": "Transactional record table suited for SQL filtering."}

Return ONLY a JSON object:
{
  "target_db": "vector or document or graph or index",
  "reasoning": "One clear sentence."
}"""

            print(f"  [orchestrator] Analyzing schema for '{table_name}'...")
            raw = ask_ollama(prompt, json_mode=True)
            try:
                parsed = extract_json(raw)
                target = parsed.get("target_db", "").lower().strip()
                reason = parsed.get("reasoning", "N/A")
            except (ValueError, KeyError):
                raw_lower = raw.lower()
                target = next((db for db in ["index","graph","document","vector"] if db in raw_lower), "vector")
                reason = raw.replace("\n"," ")[:150]

            valid = {"vector","document","graph","index"}
            if target not in valid:
                target = next((db for db in ["index","graph","document","vector"] if db in raw.lower()), "vector")

            print(f"  [orchestrator] routed '{table_name}' -> {target.upper()}")
            return {**state, "target_db": target, "routing_reason": reason}

        with patch.object(orch_mod, "node_orchestrator", node_orch_no_enrich):
            orch_mod._GRAPH = orch_mod._build_graph()
            print("  --> Running Neutral Pass...")
            res_neutral = run_variant_eval(gt_entries, "V4", human_context="")
            print("  --> Running Biased Pass...")
            res_biased = run_variant_eval(gt_entries, "V4", human_context=BIASED_PROMPT)
            orch_mod._GRAPH = orch_mod._build_graph()

        all_metrics["V4"] = compute_metrics(res_neutral)
        all_metrics["V4"]["ais"] = calculate_ais(res_neutral, res_biased)

    # ── V5: Original model (phi3:mini) ─────────────────────────────
    if not selected or selected == "V5":
        print_banner("V5 — Original Model  (phi3:mini, all other improvements kept)")

        import core.llm_utils as llm_utils_mod

        original_ask = llm_utils_mod.ask_ollama

        def ask_phi3(prompt: str, json_mode: bool = True) -> str:
            import requests
            payload = {"model": "phi3:mini", "prompt": prompt, "stream": False}
            if json_mode:
                payload["format"] = "json"
            try:
                resp = requests.post(llm_utils_mod.OLLAMA_URL, json=payload, timeout=180)
                if resp.status_code != 200:
                    print(f"\n[ERROR] OLLAMA: {resp.text}\n")
                    return "{}"
                return resp.json()["response"].strip()
            except Exception as e:
                print(f"\n[ERROR] {e}\n")
                return "{}"

        import core.agents.agent_vector as ac
        import core.agents.agent_document  as am
        import core.agents.agent_graph  as an
        import core.agents.agent_index as ar

        with patch.object(llm_utils_mod, "ask_ollama", ask_phi3), \
             patch.object(ac,  "ask_ollama", ask_phi3), \
             patch.object(am,  "ask_ollama", ask_phi3), \
             patch.object(an,  "ask_ollama", ask_phi3), \
             patch.object(ar,  "ask_ollama", ask_phi3), \
             patch.object(orch_mod, "ask_ollama", ask_phi3):
            print("  --> Running Neutral Pass...")
            res_neutral = run_variant_eval(gt_entries, "V5", human_context="")
            print("  --> Running Biased Pass...")
            res_biased = run_variant_eval(gt_entries, "V5", human_context=BIASED_PROMPT)

        all_metrics["V5"] = compute_metrics(res_neutral)
        all_metrics["V5"]["ais"] = calculate_ais(res_neutral, res_biased)

    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ablation study for polyglot routing.")
    parser.add_argument("--only-high", action="store_true",
                        help="Evaluate only high-confidence ground-truth cases.")
    parser.add_argument("--variant", type=str, default=None,
                        choices=["V1","V2","V3","V4","V5"],
                        help="Run only one variant (e.g. --variant V3).")
    args = parser.parse_args()

    with open(GROUND_TRUTH_PATH) as f:
        gt_entries = json.load(f)

    if args.only_high:
        gt_entries = [e for e in gt_entries if e["confidence"] == "high"]

    print(f"\nAblation study on {len(gt_entries)} tables")
    print(f"Variants to run: {args.variant or 'V1 V2 V3 V4 V5'}\n")

    all_metrics = run_all_variants(gt_entries, selected=args.variant)

    # Per-variant detail
    for vname, m in all_metrics.items():
        print_per_class_detail(vname, m)

    # Main comparison table
    print_summary_table(all_metrics)

    # Save
    with open(ABLATION_OUT_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Full results saved to: {ABLATION_OUT_PATH}\n")


if __name__ == "__main__":
    main()