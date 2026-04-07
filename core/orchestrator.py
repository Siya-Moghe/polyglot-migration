import json
import re
from typing import Any
from langgraph.graph import StateGraph, END

from core.state import StrategyState
from core.llm_utils import ask_ollama, extract_json, enrich_table_context
from core.agent_chroma import node_analyze_chroma, node_validate_chroma, route_chroma_validation
from core.agent_mongo import node_analyze_mongo, node_validate_mongo, route_mongo_validation
from core.agent_neo4j import node_analyze_neo4j, node_validate_neo4j, route_neo4j_validation
from core.agent_relational import (
    node_analyze_relational,
    node_validate_relational,
    route_relational_validation,
)

def is_text_type(col_type: str) -> bool:
    t = str(col_type).upper()
    return any(x in t for x in ["TEXT", "CHAR", "VARCHAR", "CLOB"])

def is_numeric_type(col_type: str) -> bool:
    t = str(col_type).upper()
    return any(x in t for x in ["INT", "FLOAT", "REAL", "DOUBLE", "DECIMAL", "NUMERIC"])

def is_temporal_type(col_type: str) -> bool:
    t = str(col_type).upper()
    return any(x in t for x in ["DATE", "TIME", "TIMESTAMP"])

def extract_table_features(table_name: str, info: dict) -> dict:
    cols = info.get("columns", [])
    fks = info.get("foreign_keys", [])
    pks = set(info.get("primary_keys", []))

    col_names = [c["name"].lower() for c in cols]
    table_lower = table_name.lower()

    text_cols = [c["name"] for c in cols if is_text_type(c["type"])]
    numeric_cols = [c["name"] for c in cols if is_numeric_type(c["type"])]
    temporal_cols = [c["name"] for c in cols if is_temporal_type(c["type"])]
    fk_cols = [fk["columns"][0] for fk in fks if fk.get("columns")]

    semantic_keywords = [
        "content", "description", "body", "text", "summary", "review",
        "abstract", "note", "article", "comment", "details", "bio"
    ]

    transactional_keywords = [
        "status", "amount", "price", "total", "balance", "created_at",
        "updated_at", "timestamp", "priority", "deadline", "quantity"
    ]

    entity_keywords = [
        "profile", "product", "catalog", "config", "setting", "metadata",
        "country", "office", "vehicle", "event"
    ]

    has_semantic_cols = any(any(k in c for k in semantic_keywords) for c in col_names)
    has_transactional_cols = any(any(k in c for k in transactional_keywords) for c in col_names)
    entity_like_name = any(k in table_lower for k in entity_keywords)

    is_junction = (
        len(fk_cols) >= 2 and
        len(cols) <= 6
    )

    return {
        "table_name": table_name,
        "num_columns": len(cols),
        "num_text_cols": len(text_cols),
        "num_numeric_cols": len(numeric_cols),
        "num_temporal_cols": len(temporal_cols),
        "num_fk_cols": len(fk_cols),
        "has_semantic_cols": has_semantic_cols,
        "has_transactional_cols": has_transactional_cols,
        "entity_like_name": entity_like_name,
        "is_junction": is_junction,
        "text_cols": text_cols,
        "fk_cols": fk_cols,
        "col_names": col_names,
    }

def score_targets(features: dict) -> dict:
    scores = {
        "chroma": 0,
        "mongo": 0,
        "neo4j": 0,
        "relational": 0,
    }

    # -------------------------
    # CHROMA
    # -------------------------
    if features["has_semantic_cols"]:
        scores["chroma"] += 4
    if features["num_text_cols"] >= 2:
        scores["chroma"] += 2
    if any(k in features["table_name"].lower() for k in [
        "review", "paper", "article", "blog", "note", "contract", "course", "knowledge"
    ]):
        scores["chroma"] += 4

    # -------------------------
    # MONGO
    # -------------------------
    if features["num_fk_cols"] == 0:
        scores["mongo"] += 3
    if features["entity_like_name"]:
        scores["mongo"] += 4
    if features["num_text_cols"] >= 1 and features["num_fk_cols"] <= 1:
        scores["mongo"] += 1

    # -------------------------
    # NEO4J
    # -------------------------
    if features["num_fk_cols"] >= 2:
        scores["neo4j"] += 5
    if features["is_junction"]:
        scores["neo4j"] += 5
    if any(k in features["table_name"].lower() for k in [
        "map", "link", "relation", "association", "membership", "projects"
    ]):
        scores["neo4j"] += 3

    # -------------------------
    # RELATIONAL
    # -------------------------
    if features["has_transactional_cols"]:
        scores["relational"] += 4
    if features["num_numeric_cols"] >= 2:
        scores["relational"] += 2
    if features["num_temporal_cols"] >= 1:
        scores["relational"] += 2

    return scores

def choose_by_scores(scores: dict):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best, second = ranked[0], ranked[1]
    confidence_gap = best[1] - second[1]
    return best[0], best[1], confidence_gap, ranked



def node_orchestrator(state: StrategyState) -> StrategyState:
    table_name, info = state["table_name"], state["table_info"]
    sample_rows = state.get("sample_rows", [])

    features = extract_table_features(table_name, info)
    scores = score_targets(features)
    best_target, best_score, confidence_gap, ranked = choose_by_scores(scores)

    # High-confidence deterministic route
    if best_score >= 6 and confidence_gap >= 2:
        print(f"  [orchestrator] heuristically routed '{table_name}' -> {best_target.upper()}")
        return {
        **state,
        "target_db": best_target,
        "routing_reason": f"Heuristic routing selected {best_target} based on schema features.",
        "strategy": {
            "target_db": best_target,
            "reasoning": f"Heuristic routing selected {best_target} based on schema features.",
            "routing_scores": scores,
            "routing_features": features,
            "routing_confidence_gap": confidence_gap,
            "routing_method": "heuristic"
        }
        }

    # Otherwise use LLM as tie-breaker
    enriched = enrich_table_context(table_name, info, sample_rows)

    prompt = f"""
You are a Database Routing AI.

A heuristic router found this table ambiguous.

Table context:
{enriched}

Extracted features:
{json.dumps(features, indent=2)}

Heuristic scores:
{json.dumps(scores, indent=2)}

Choose ONLY one target_db from:
- chroma
- mongo
- neo4j
- relational

Rules:
- Prefer chroma for long semantic text / search-oriented content.
- Prefer mongo for self-contained entity-style records.
- Prefer neo4j for relationship-heavy or junction tables.
- Prefer relational for transactional / exact-match / reporting use cases.

Return ONLY JSON:
{{
  "target_db": "chroma or mongo or neo4j or relational",
  "reasoning": "One clear sentence explaining the best fit."
}}
"""

    print(f"  [orchestrator] LLM arbitration for '{table_name}'...")
    raw_response = ask_ollama(prompt, json_mode=True)

    try:
        result = extract_json(raw_response)
        raw_target = str(result.get("target_db", "")).lower()
        valid_dbs = {"chroma", "mongo", "neo4j", "relational"}
        
        if raw_target not in valid_dbs:
            found = [db for db in valid_dbs if db in raw_target]
            if len(found) == 1:
                result["target_db"] = found[0]
            else:
                raise ValueError(f"Invalid LLM target: {raw_target}")
    except Exception:
        result = {
            "target_db": best_target,
            "reasoning": "LLM arbitration failed; fell back to heuristic routing."
        }

    result["routing_scores"] = scores
    result["routing_features"] = features
    result["routing_confidence_gap"] = confidence_gap
    result["routing_method"] = "llm_tiebreak"

    final_target = str(result.get("target_db", best_target)).strip().lower()

    print(f"   [orchestrator] routed '{table_name}' -> {final_target.upper()}")
    
    return {
        **state,
        "target_db": final_target, # This must be lowercase to match g.add_node names
        "routing_reason": result.get("reasoning", "LLM routing decision."),
        "strategy": result
    }




def node_reflect(state: StrategyState) -> StrategyState:
    return {**state, "retries": state["retries"] + 1}


def node_accept(state: StrategyState) -> StrategyState:
    final_strat = state.get("strategy") or {}
    errors = state.get("errors", [])

    if errors:
        print(f"  [fallback] Agent failed for '{state['table_name']}'. Applying safe fallback.")

        all_cols = [c["name"] for c in state["table_info"]["columns"]]
        target_db = state.get("target_db", "chroma")

        if target_db == "relational":
            final_strat = {
                "target_db": "relational",
                "preserve_as": "table",
                "used_columns": all_cols,
                "skipped_columns": [],
                "join_related": [],
                "primary_use": "transactional",
                "reasoning": "FALLBACK TRIGGERED: Relational fallback used because the AI agent failed.",
                "fallback_used": True,
            }
        elif target_db == "mongo":
            final_strat = {
                "target_db": "mongo",
                "fields": all_cols,
                "nested_fields": {},
                "join_related": [],
                "join_keys": {},
                "used_columns": all_cols,
                "skipped_columns": [],
                "reasoning": "FALLBACK TRIGGERED: Mongo fallback used because the AI agent failed.",
                "fallback_used": True,
            }
        elif target_db == "neo4j":
            final_strat = {
                "target_db": "neo4j",
                "used_columns": all_cols,
                "skipped_columns": [],
                "join_related": [fk["referred_table"] for fk in state["table_info"]["foreign_keys"]],
                "reasoning": "FALLBACK TRIGGERED: Neo4j fallback used because the AI agent failed.",
                "fallback_used": True,
            }
        else:
            fallback_template = " | ".join(f"{col}: {{{col}}}" for col in all_cols)
            final_strat = {
                "target_db": "chroma",
                "template": fallback_template,
                "used_columns": all_cols,
                "skipped_columns": [],
                "join_related": [],
                "reasoning": "FALLBACK TRIGGERED: Chroma fallback used because the AI agent failed.",
                "fallback_used": True,
            }
    else:
        final_strat["target_db"] = state["target_db"]
        final_strat["routing_reason"] = state.get("routing_reason", "N/A")
        final_strat["fallback_used"] = False

    return {**state, "final": final_strat}


def route_to_specialist(state: StrategyState) -> str:
    return f"analyze_{state['target_db']}"


def route_retry(state: StrategyState) -> str:
    return f"analyze_{state['target_db'].lower()}"


def _build_graph() -> Any:
    g = StateGraph(StrategyState)

    g.add_node("orchestrator", node_orchestrator)

    g.add_node("analyze_chroma", node_analyze_chroma)
    g.add_node("validate_chroma", node_validate_chroma)

    g.add_node("analyze_mongo", node_analyze_mongo)
    g.add_node("validate_mongo", node_validate_mongo)

    g.add_node("analyze_neo4j", node_analyze_neo4j)
    g.add_node("validate_neo4j", node_validate_neo4j)

    g.add_node("analyze_relational", node_analyze_relational)
    g.add_node("validate_relational", node_validate_relational)

    g.add_node("reflect", node_reflect)
    g.add_node("accept", node_accept)

    g.set_entry_point("orchestrator")

    g.add_conditional_edges("orchestrator", route_to_specialist, {
        "analyze_chroma": "analyze_chroma",
        "analyze_mongo": "analyze_mongo",
        "analyze_neo4j": "analyze_neo4j",
        "analyze_relational": "analyze_relational",
    })

    g.add_edge("analyze_chroma", "validate_chroma")
    g.add_conditional_edges("validate_chroma", route_chroma_validation, {
        "accept": "accept", "reflect": "reflect"
    })

    g.add_edge("analyze_mongo", "validate_mongo")
    g.add_conditional_edges("validate_mongo", route_mongo_validation, {
        "accept": "accept", "reflect": "reflect"
    })

    g.add_edge("analyze_neo4j", "validate_neo4j")
    g.add_conditional_edges("validate_neo4j", route_neo4j_validation, {
        "accept": "accept", "reflect": "reflect"
    })

    g.add_edge("analyze_relational", "validate_relational")
    g.add_conditional_edges("validate_relational", route_relational_validation, {
        "accept": "accept", "reflect": "reflect"
    })

    g.add_conditional_edges("reflect", route_retry)
    g.add_edge("accept", END)

    return g.compile()


_GRAPH = _build_graph()


def plan_embeddings(schema: dict[str, Any], sample_rows_map: dict[str, list] = None) -> dict[str, dict]:
    """
    Run the LangGraph pipeline for each table.

    Args:
        schema:          Output of introspect_schema().
        sample_rows_map: Optional dict of {table_name: [row_dicts]} for enriched prompts.
                         Pass the first 2-3 rows per table. If None, enrichment is skipped.
    """
    strategies = {}
    sample_rows_map = sample_rows_map or {}

    for table_name, table_info in schema.items():
        print(f"\nPlanning: {table_name}")

        init_state: StrategyState = {
            "table_name": table_name,
            "table_info": table_info,
            "all_tables": list(schema.keys()),
            "target_db": None,
            "routing_reason": None,
            "strategy": None,
            "errors": [],
            "retries": 0,
            "final": None,
            "sample_rows": sample_rows_map.get(table_name, []),
        }

        strategies[table_name] = _GRAPH.invoke(init_state)["final"]

    return strategies