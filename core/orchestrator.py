"""
orchestrator.py    LangGraph orchestrator for SurrealDB migration.

Differences from the original polyglot orchestrator:
  - The scoring step no longer picks ONE of four databases.
    Instead it decides which SurrealDB features (use_vector, use_graph,
    use_nested, index_cols) are likely to be needed.  This pre-fills
    `feature_hints` in the state so the LLM agent has a warm start.
  - There is only one specialist agent (agent_surrealdb) and one
    validate node, so the graph is much simpler.
  - `save_migration_manifest` now records features rather than target_engine.
"""

from __future__ import annotations

import json
import time
from typing import Any

from langgraph.graph import StateGraph, END

from core.state import StrategyState
from core.llm_utils import ask_ollama, extract_json, enrich_table_context
from core.agent_surrealdb import (
    node_analyze_surrealdb,
    node_validate_surrealdb,
    route_surrealdb_validation,
)

# ---------------------------------------------------------------------------
# Schema feature extraction  (same helpers as original, kept intact)
# ---------------------------------------------------------------------------

def _is_text(col_type: str) -> bool:
    t = str(col_type).upper()
    return any(x in t for x in ["TEXT", "CHAR", "VARCHAR", "CLOB"])

def _is_numeric(col_type: str) -> bool:
    t = str(col_type).upper()
    return any(x in t for x in ["INT", "FLOAT", "REAL", "DOUBLE", "DECIMAL", "NUMERIC"])

def _is_temporal(col_type: str) -> bool:
    t = str(col_type).upper()
    return any(x in t for x in ["DATE", "TIME", "TIMESTAMP"])


def extract_table_features(table_name: str, info: dict) -> dict:
    cols   = info.get("columns", [])
    fks    = info.get("foreign_keys", [])
    pks    = set(info.get("primary_keys", []))

    col_names   = [c["name"].lower() for c in cols]
    table_lower = table_name.lower()

    text_cols    = [c["name"] for c in cols if _is_text(c["type"])]
    numeric_cols = [c["name"] for c in cols if _is_numeric(c["type"])]
    temporal_cols = [c["name"] for c in cols if _is_temporal(c["type"])]
    fk_cols      = [fk["columns"][0] for fk in fks if fk.get("columns")]

    semantic_kw      = ["content", "description", "body", "text", "summary", "review",
                        "abstract", "note", "article", "comment", "details", "bio", "title"]
    transactional_kw = ["status", "amount", "price", "total", "balance", "created_at",
                        "updated_at", "timestamp", "priority", "deadline", "quantity"]
    entity_kw        = ["profile", "product", "catalog", "config", "setting", "metadata",
                        "country", "office", "vehicle", "event", "department"]

    has_semantic      = any(any(k in c for k in semantic_kw) for c in col_names)
    has_transactional = any(any(k in c for k in transactional_kw) for c in col_names)
    entity_like       = any(k in table_lower for k in entity_kw)
    is_junction       = (len(fk_cols) >= 2 and len(cols) <= 6)

    return {
        "table_name":          table_name,
        "num_columns":         len(cols),
        "num_text_cols":       len(text_cols),
        "num_numeric_cols":    len(numeric_cols),
        "num_temporal_cols":   len(temporal_cols),
        "num_fk_cols":         len(fk_cols),
        "has_semantic_cols":   has_semantic,
        "has_transactional_cols": has_transactional,
        "entity_like_name":    entity_like,
        "is_junction":         is_junction,
        "text_cols":           text_cols,
        "fk_cols":             fk_cols,
        "col_names":           col_names,
    }


def derive_feature_hints(features: dict) -> dict:
    """
    Derive heuristic feature hints for SurrealDB from schema features.
    These are passed to the LLM as a warm start – the LLM can override them.
    """
    return {
        "use_vector": (
            features["has_semantic_cols"]
            or features["num_text_cols"] >= 2
            or any(k in features["table_name"].lower()
                   for k in ["review", "paper", "article", "blog", "note",
                              "contract", "course", "knowledge"])
        ),
        "use_graph": (
            features["num_fk_cols"] >= 1
        ),
        "use_nested": (
            features["entity_like_name"]
            or (features["num_text_cols"] >= 2 and features["num_fk_cols"] <= 1)
        ),
        "needs_indexes": (
            features["has_transactional_cols"]
            or features["num_numeric_cols"] >= 2
            or features["num_temporal_cols"] >= 1
        ),
    }


# ---------------------------------------------------------------------------
# Orchestrator node
# ---------------------------------------------------------------------------

def node_orchestrator(state: StrategyState) -> StrategyState:
    table_name  = state["table_name"]
    info        = state["table_info"]
    sample_rows = state.get("sample_rows", [])

    features      = extract_table_features(table_name, info)
    feature_hints = derive_feature_hints(features)

    print(
        f"  [orchestrator] '{table_name}' hints → "
        f"vector={feature_hints['use_vector']} "
        f"graph={feature_hints['use_graph']} "
        f"nested={feature_hints['use_nested']} "
        f"indexes={feature_hints['needs_indexes']}"
    )

    return {
        **state,
        "target_db":     "surrealdb",
        "routing_reason": "All tables go to SurrealDB; features are decided per-table.",
        "feature_hints":  feature_hints,
        "routing_features": features,
    }


# ---------------------------------------------------------------------------
# Reflect / Accept nodes  (identical logic to original)
# ---------------------------------------------------------------------------

def node_reflect(state: StrategyState) -> StrategyState:
    return {**state, "retries": state["retries"] + 1}


def node_accept(state: StrategyState) -> StrategyState:
    final_strat = state.get("strategy") or {}
    errors      = state.get("errors", [])

    if errors:
        # Fallback: safe minimal strategy so the pipeline doesn't crash
        all_cols = [c["name"] for c in state["table_info"]["columns"]]
        fk_rels  = [
            {
                "label":    f"{state['table_name'].upper()}_TO_{fk['referred_table'].upper()}",
                "from_col": fk["columns"][0] if fk.get("columns") else "",
                "to_table": fk["referred_table"],
                "to_col":   fk["referred_columns"][0] if fk.get("referred_columns") else "id",
                "edge_props": [],
            }
            for fk in state["table_info"].get("foreign_keys", [])
            if fk.get("columns")
        ]
        final_strat = {
            "use_vector":        False,
            "vector_template":   "",
            "vector_cols":       [],
            "use_graph":         len(fk_rels) > 0,
            "relations":         fk_rels,
            "use_nested":        False,
            "nested_fields":     {},
            "top_level_fields":  all_cols,
            "index_cols":        [],
            "reasoning":         "Fallback strategy — LLM validation failed.",
            "fallback_used":     True,
        }
    else:
        final_strat["target_db"]        = "surrealdb"
        final_strat["routing_reason"]   = state.get("routing_reason", "N/A")
        final_strat["fallback_used"]    = False
        final_strat["feature_hints"]    = state.get("feature_hints", {})
        final_strat["routing_features"] = state.get("routing_features", {})

    return {**state, "final": final_strat}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------

def _build_graph() -> Any:
    g = StateGraph(StrategyState)

    g.add_node("orchestrator",       node_orchestrator)
    g.add_node("analyze_surrealdb",  node_analyze_surrealdb)
    g.add_node("validate_surrealdb", node_validate_surrealdb)
    g.add_node("reflect",            node_reflect)
    g.add_node("accept",             node_accept)

    g.set_entry_point("orchestrator")
    g.add_edge("orchestrator", "analyze_surrealdb")
    g.add_edge("analyze_surrealdb", "validate_surrealdb")
    g.add_conditional_edges(
        "validate_surrealdb",
        route_surrealdb_validation,
        {"accept": "accept", "reflect": "reflect"},
    )
    g.add_edge("reflect", "analyze_surrealdb")
    g.add_edge("accept", END)

    return g.compile()


_GRAPH = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_migration(
    schema: dict[str, Any],
    sample_rows_map: dict[str, list] | None = None,
    human_context: str = "",
) -> dict[str, dict]:
    """
    Run the LangGraph orchestrator for every table in *schema*.

    Parameters
    ----------
    schema          : output of introspect_schema()
    sample_rows_map : {table_name: [row_dicts]}  –  optional, improves LLM reasoning
    human_context   : free-text expert advice passed to the LLM

    Returns
    -------
    {table_name: final_strategy_dict}
    """
    strategies      = {}
    sample_rows_map = sample_rows_map or {}

    for table_name, table_info in schema.items():
        print(f"\nPlanning: {table_name}")
        init_state: StrategyState = {
            "table_name":     table_name,
            "table_info":     table_info,
            "all_tables":     list(schema.keys()),
            "target_db":      None,
            "routing_reason": None,
            "strategy":       None,
            "errors":         [],
            "retries":        0,
            "final":          None,
            "sample_rows":    sample_rows_map.get(table_name, []),
            "human_context":  human_context,
            "bias_test_mode": False,
            # SurrealDB-specific fields (populated by orchestrator node)
            "feature_hints":      {},
            "routing_features":   {},
        }
        result = _GRAPH.invoke(init_state)
        strategies[table_name] = result["final"]

    return strategies


def save_migration_manifest(strategies: dict, source_db: str) -> None:
    """
    Write migration_manifest.json.

    The manifest now records:
      - target_engine: always "surrealdb"
      - features: list of activated SurrealDB capabilities per table
      - retained_columns / dropped_columns: same as before for audit
    """
    manifest = {
        "metadata": {
            "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_db":    source_db,
            "target_engine": "surrealdb",
            "total_tables": len(strategies),
        },
        "migrations": [],
    }

    for tname, strat in strategies.items():
        features_active = []
        if strat.get("use_vector"):  features_active.append("vector")
        if strat.get("use_graph"):   features_active.append("graph")
        if strat.get("use_nested"):  features_active.append("nested")
        if strat.get("index_cols"):  features_active.append("indexes")

        all_cols    = strat.get("top_level_fields", [])
        nested_flat = [
            c
            for children in strat.get("nested_fields", {}).values()
            for c in children
        ]
        retained = list(dict.fromkeys(all_cols + nested_flat))  # dedup, order-preserving

        manifest["migrations"].append({
            "table_name":       tname,
            "target_engine":    "surrealdb",
            "features":         features_active,
            "retained_columns": retained,
            "dropped_columns":  strat.get("skipped_columns", []),
            "relations":        strat.get("relations", []),
            "index_cols":       strat.get("index_cols", []),
            "fallback_used":    strat.get("fallback_used", False),
        })

    with open("migration_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

    print("\n[system] Migration manifest saved → migration_manifest.json")