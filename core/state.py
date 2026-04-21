from typing import TypedDict, List, Any, Optional


class StrategyState(TypedDict):
    # --- core ---
    table_name:     str
    table_info:     dict
    all_tables:     List[str]
    target_db:      Optional[str]
    routing_reason: Optional[str]
    strategy:       Optional[dict]   # merged final strategy (set by merger node)
    errors:         List[str]
    retries:        int
    final:          Optional[dict]
    sample_rows:    List[dict]
    human_context:  str
    bias_test_mode: bool

    # --- orchestrator outputs ---
    feature_hints:    dict   # {use_vector, use_graph, use_nested, needs_indexes}
    routing_features: dict   # raw schema feature counts

    # --- per-agent sub-strategies (each agent writes its own key) ---
    vector_strategy:   Optional[dict]
    graph_strategy:    Optional[dict]
    document_strategy: Optional[dict]
    index_strategy:    Optional[dict]

    # --- per-agent error/retry tracking ---
    vector_errors:   List[str]
    graph_errors:    List[str]
    document_errors: List[str]
    index_errors:    List[str]

    vector_retries:   int
    graph_retries:    int
    document_retries: int
    index_retries:    int
