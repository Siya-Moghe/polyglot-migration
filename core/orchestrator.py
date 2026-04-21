"""
orchestrator.py — Multi-agent LangGraph orchestrator for SurrealDB migration.

Architecture (mirrors the original polyglot design but with SurrealDB specialists):

  Stage 2 = PLANNING ONLY. No agent touches a database.
  Stage 3 = embedder_surrealdb.py reads the final strategy and writes to SurrealDB.

Per table the graph runs:
  orchestrator → [vector_agent, graph_agent, document_agent, index_agent] in sequence
               → merger

Each specialist agent has its own analyze → validate → reflect loop.
The orchestrator decides which agents are needed based on schema heuristics,
then marks the others as skipped (safe defaults, no LLM call wasted).
"""

from __future__ import annotations

import json
import time
from typing import Any

from langgraph.graph import StateGraph, END

from core.state import StrategyState
from core.agents.agent_vector   import (node_analyze_vector,   node_validate_vector,   node_reflect_vector,   route_vector)
from core.agents.agent_graph    import (node_analyze_graph,    node_validate_graph,    node_reflect_graph,    route_graph)
from core.agents.agent_document import (node_analyze_document, node_validate_document, node_reflect_document, route_document)
from core.agents.agent_index    import (node_analyze_index,    node_validate_index,    node_reflect_index,    route_index)


# ---------------------------------------------------------------------------
# Schema feature extraction — heuristic, no LLM
# ---------------------------------------------------------------------------

def _is_text(t: str) -> bool:
    return any(x in t.upper() for x in ["TEXT", "CHAR", "VARCHAR", "CLOB"])

def _is_numeric(t: str) -> bool:
    return any(x in t.upper() for x in ["INT", "FLOAT", "REAL", "DOUBLE", "DECIMAL", "NUMERIC"])

def _is_temporal(t: str) -> bool:
    return any(x in t.upper() for x in ["DATE", "TIME", "TIMESTAMP"])


def extract_table_features(table_name: str, info: dict) -> dict:
    cols       = info.get("columns", [])
    fks        = info.get("foreign_keys", [])
    col_names  = [c["name"].lower() for c in cols]

    semantic_kw      = ["content","description","body","text","summary","review",
                        "abstract","note","article","comment","details","bio","title","issue"]
    transactional_kw = ["status","amount","price","total","balance","created_at",
                        "updated_at","timestamp","priority","deadline","quantity"]
    entity_kw        = ["profile","product","catalog","config","setting","metadata",
                        "country","office","vehicle","event","department"]

    return {
        "num_text_cols":          sum(1 for c in cols if _is_text(c["type"])),
        "num_numeric_cols":       sum(1 for c in cols if _is_numeric(c["type"])),
        "num_temporal_cols":      sum(1 for c in cols if _is_temporal(c["type"])),
        "num_fk_cols":            sum(1 for fk in fks if fk.get("columns")),
        "has_semantic_cols":      any(any(k in c for k in semantic_kw) for c in col_names),
        "has_transactional_cols": any(any(k in c for k in transactional_kw) for c in col_names),
        "entity_like_name":       any(k in table_name.lower() for k in entity_kw),
        "is_junction":            (sum(1 for fk in fks if fk.get("columns")) >= 2 and len(cols) <= 6),
    }


def derive_feature_hints(features: dict, table_name: str) -> dict:
    return {
        "use_vector": (
            features["has_semantic_cols"]
            or features["num_text_cols"] >= 2
            or any(k in table_name.lower() for k in ["review","article","blog","note","knowledge"])
        ),
        "use_graph":      features["num_fk_cols"] >= 1,
        "use_nested":     features["entity_like_name"] or (features["num_text_cols"] >= 2 and features["num_fk_cols"] <= 1),
        "needs_indexes":  features["has_transactional_cols"] or features["num_numeric_cols"] >= 2 or features["num_temporal_cols"] >= 1,
    }


# ---------------------------------------------------------------------------
# Orchestrator node — sets hints, decides which agents to activate
# ---------------------------------------------------------------------------

def node_orchestrator(state: StrategyState) -> StrategyState:
    table_name = state["table_name"]
    info       = state["table_info"]

    features = extract_table_features(table_name, info)
    hints    = derive_feature_hints(features, table_name)

    print(
        f"  [orchestrator] '{table_name}' → "
        f"vector={hints['use_vector']} graph={hints['use_graph']} "
        f"nested={hints['use_nested']} indexes={hints['needs_indexes']}"
    )

    return {
        **state,
        "target_db":       "surrealdb",
        "routing_reason":  "All tables → SurrealDB; features decided per specialist agent.",
        "feature_hints":   hints,
        "routing_features": features,
        # reset per-agent fields
        "vector_strategy": None, "graph_strategy":    None,
        "document_strategy": None, "index_strategy":  None,
        "vector_errors":  [],    "graph_errors":      [],
        "document_errors":[],    "index_errors":      [],
        "vector_retries": 0,     "graph_retries":     0,
        "document_retries":0,    "index_retries":     0,
    }


# ---------------------------------------------------------------------------
# Routing functions — skip an agent if not needed
# ---------------------------------------------------------------------------

def route_to_vector(state: StrategyState) -> str:
    return "vector_agent" if state["feature_hints"].get("use_vector") else "graph_agent"

def route_to_graph(state: StrategyState) -> str:
    return "graph_agent" if state["feature_hints"].get("use_graph") else "document_agent"

def route_after_vector(state: StrategyState) -> str:
    return "graph_agent" if state["feature_hints"].get("use_graph") else "document_agent"

def route_after_graph(state: StrategyState) -> str:
    return "document_agent"

def route_to_index(state: StrategyState) -> str:
    return "index_agent" if state["feature_hints"].get("needs_indexes") else "merge"


# ---------------------------------------------------------------------------
# Merger node — combines all sub-strategies into one final strategy dict
# ---------------------------------------------------------------------------

def node_merge(state: StrategyState) -> StrategyState:
    info     = state["table_info"]
    all_cols = [c["name"] for c in info["columns"]]

    # vector
    vs = state.get("vector_strategy") or {}
    use_vector      = bool(vs.get("use_vector"))
    vector_template = vs.get("vector_template", "") if use_vector else ""
    vector_cols     = vs.get("vector_cols", [])      if use_vector else []

    # graph
    gs        = state.get("graph_strategy") or {}
    use_graph = bool(gs.get("use_graph"))
    relations = gs.get("relations", []) if use_graph else []

    # document
    ds            = state.get("document_strategy") or {}
    use_nested    = bool(ds.get("use_nested"))
    nested_fields = ds.get("nested_fields", {}) if use_nested else {}
    top_level     = ds.get("top_level_fields", all_cols) or all_cols

    # index — agent returns [{col, unique}]; normalise to flat list + unique map
    ix       = state.get("index_strategy") or {}
    raw_idx  = ix.get("index_cols", [])
    valid_col_names = {c["name"] for c in info["columns"]}
    index_cols   = []
    index_unique = {}
    for entry in raw_idx:
        if isinstance(entry, dict):
            col = entry.get("col", "")
            if col and col in valid_col_names:
                index_cols.append(col)
                index_unique[col] = bool(entry.get("unique", False))

    reasoning = " | ".join(filter(None, [
        vs.get("reasoning",""), gs.get("reasoning",""),
        ds.get("reasoning",""), ix.get("reasoning","")
    ]))

    merged = {
        "target_db":        "surrealdb",
        "use_vector":       use_vector,
        "vector_template":  vector_template,
        "vector_cols":      vector_cols,
        "use_graph":        use_graph,
        "relations":        relations,
        "use_nested":       use_nested,
        "nested_fields":    nested_fields,
        "top_level_fields": top_level,
        "index_cols":       index_cols,
        "index_unique":     index_unique,
        "reasoning":        reasoning,
        "fallback_used":    False,
        "routing_reason":   state.get("routing_reason", ""),
        "feature_hints":    state.get("feature_hints", {}),
        "agent_errors": {
            "vector":   state.get("vector_errors",   []),
            "graph":    state.get("graph_errors",    []),
            "document": state.get("document_errors", []),
            "index":    state.get("index_errors",    []),
        },
    }

    return {**state, "final": merged}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------

def _build_graph() -> Any:
    g = StateGraph(StrategyState)

    g.add_node("orchestrator",      node_orchestrator)

    # specialist agents
    g.add_node("vector_agent",      node_analyze_vector)
    g.add_node("validate_vector",   node_validate_vector)
    g.add_node("reflect_vector",    node_reflect_vector)

    g.add_node("graph_agent",       node_analyze_graph)
    g.add_node("validate_graph",    node_validate_graph)
    g.add_node("reflect_graph",     node_reflect_graph)

    g.add_node("document_agent",    node_analyze_document)
    g.add_node("validate_document", node_validate_document)
    g.add_node("reflect_document",  node_reflect_document)

    g.add_node("index_agent",       node_analyze_index)
    g.add_node("validate_index",    node_validate_index)
    g.add_node("reflect_index",     node_reflect_index)

    g.add_node("merge",             node_merge)

    # entry
    g.set_entry_point("orchestrator")

    # orchestrator → first agent (vector if needed, else graph)
    g.add_conditional_edges("orchestrator", route_to_vector,
        {"vector_agent": "vector_agent", "graph_agent": "graph_agent"})

    # vector loop → then graph
    g.add_edge("vector_agent",    "validate_vector")
    g.add_conditional_edges("validate_vector", route_vector,
        {"accept": "after_vector", "reflect": "reflect_vector"})
    g.add_edge("reflect_vector",  "vector_agent")
    g.add_node("after_vector", lambda s: s)   # passthrough to allow conditional routing
    g.add_conditional_edges("after_vector", route_after_vector,
        {"graph_agent": "graph_agent", "document_agent": "document_agent"})

    # graph loop → document
    g.add_edge("graph_agent",     "validate_graph")
    g.add_conditional_edges("validate_graph", route_graph,
        {"accept": "document_agent", "reflect": "reflect_graph"})
    g.add_edge("reflect_graph",   "graph_agent")

    # document loop → index (if needed) else merge
    g.add_edge("document_agent",    "validate_document")
    g.add_conditional_edges("validate_document", route_document,
        {"accept": "after_document", "reflect": "reflect_document"})
    g.add_edge("reflect_document",  "document_agent")
    g.add_node("after_document", lambda s: s)
    g.add_conditional_edges("after_document", route_to_index,
        {"index_agent": "index_agent", "merge": "merge"})

    # index loop → merge
    g.add_edge("index_agent",     "validate_index")
    g.add_conditional_edges("validate_index", route_index,
        {"accept": "merge", "reflect": "reflect_index"})
    g.add_edge("reflect_index",   "index_agent")

    g.add_edge("merge", END)

    return g.compile()


_GRAPH = _build_graph()


# ---------------------------------------------------------------------------
# Public API — called by pipeline.py, unchanged interface
# ---------------------------------------------------------------------------

def plan_migration(
    schema: dict[str, Any],
    sample_rows_map: dict[str, list] | None = None,
    human_context: str = "",
) -> dict[str, dict]:
    """
    Run the multi-agent planning graph for every table.
    Returns {table_name: final_strategy_dict} — no DB writes happen here.
    """
    strategies      = {}
    sample_rows_map = sample_rows_map or {}

    for table_name, table_info in schema.items():
        print(f"\nPlanning: {table_name}")
        init: StrategyState = {
            "table_name":      table_name,
            "table_info":      table_info,
            "all_tables":      list(schema.keys()),
            "target_db":       None,
            "routing_reason":  None,
            "strategy":        None,
            "errors":          [],
            "retries":         0,
            "final":           None,
            "sample_rows":     sample_rows_map.get(table_name, []),
            "human_context":   human_context,
            "bias_test_mode":  False,
            "feature_hints":   {},
            "routing_features": {},
            "vector_strategy": None, "graph_strategy":    None,
            "document_strategy": None, "index_strategy":  None,
            "vector_errors":  [],    "graph_errors":      [],
            "document_errors":[],    "index_errors":      [],
            "vector_retries": 0,     "graph_retries":     0,
            "document_retries":0,    "index_retries":     0,
        }
        result = _GRAPH.invoke(init)
        strategies[table_name] = result["final"]

    return strategies


def save_migration_manifest(strategies: dict, source_db: str) -> None:
    manifest = {
        "metadata": {
            "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_db":     source_db,
            "target_engine": "surrealdb",
            "total_tables":  len(strategies),
        },
        "migrations": [],
    }

    for tname, strat in strategies.items():
        features = []
        if strat.get("use_vector"):  features.append("vector")
        if strat.get("use_graph"):   features.append("graph")
        if strat.get("use_nested"):  features.append("nested")
        if strat.get("index_cols"):  features.append("indexes")

        top         = strat.get("top_level_fields", [])
        nested_flat = [c for ch in strat.get("nested_fields", {}).values() for c in ch]
        retained    = list(dict.fromkeys(top + nested_flat))

        manifest["migrations"].append({
            "table_name":       tname,
            "target_engine":    "surrealdb",
            "features":         features,
            "retained_columns": retained,
            "relations":        strat.get("relations", []),
            "index_cols":       strat.get("index_cols", []),
            "index_unique":     strat.get("index_unique", {}),
            "fallback_used":    strat.get("fallback_used", False),
            "agent_errors":     strat.get("agent_errors", {}),
        })

    with open("migration_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

    print("\n[system] Migration manifest saved → migration_manifest.json")
