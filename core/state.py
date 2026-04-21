from typing import TypedDict, List, Any, Optional


class StrategyState(TypedDict):
    # --- core (unchanged from original) ---
    table_name:     str
    table_info:     dict
    all_tables:     List[str]
    target_db:      Optional[str]
    routing_reason: Optional[str]
    strategy:       Optional[dict]
    errors:         List[str]
    retries:        int
    final:          Optional[dict]
    sample_rows:    List[dict]
    human_context:  str
    bias_test_mode: bool

    # --- SurrealDB-specific (added) ---
    feature_hints:    dict   # heuristic warm-start for the LLM: {use_vector, use_graph, ...}
    routing_features: dict   # raw schema feature counts for logging / audit