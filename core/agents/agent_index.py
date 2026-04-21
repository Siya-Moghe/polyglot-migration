"""
agent_index.py — Plans the index strategy for a table.
Pure planning only — no DB connection, no writes.
Returns: {index_cols: [{col, unique}], reasoning}
"""
from core.state import StrategyState
from core.llm_utils import ask_ollama, extract_json, enrich_table_context, MAX_RETRIES


def node_analyze_index(state: StrategyState) -> StrategyState:
    table_name  = state["table_name"]
    info        = state["table_info"]
    sample_rows = state.get("sample_rows", [])
    errors      = state.get("index_errors", [])

    enriched  = enrich_table_context(table_name, info, sample_rows)
    all_cols  = [(c["name"], c["type"]) for c in info["columns"]]
    pks       = info.get("primary_keys", [])
    fk_cols   = [fk["columns"][0] for fk in info.get("foreign_keys", []) if fk.get("columns")]
    retry_context = "\nPrevious attempt had errors — fix them:\n" + "\n".join(errors) if errors else ""

    prompt = f"""You are an Index Strategy Specialist. Your ONLY job is to decide which columns
need a DEFINE INDEX in SurrealDB for fast querying.

SurrealDB index syntax:
  DEFINE INDEX idx_name ON TABLE t FIELDS col;         -- allows duplicates
  DEFINE INDEX idx_name ON TABLE t FIELDS col UNIQUE;  -- unique constraint

Index columns that will be filtered in WHERE clauses, sorted (ORDER BY), or must be unique.
Do NOT index long text/content columns — those get a vector index handled separately.

{enriched}

Columns (name, sql_type): {all_cols}
Primary keys: {pks}
FK columns: {fk_cols}
{retry_context}

Return ONLY a JSON object:
{{
  "index_cols": [
    {{"col": "column_name", "unique": false}}
  ],
  "reasoning": "One sentence."
}}

Rules:
- index_cols is a list of objects, each with "col" (string) and "unique" (bool).
- Only include columns that exist in the table.
- unique=true only when the column must have distinct values (email, username, slug).
- Empty list [] is fine if no columns warrant indexing.

Examples:
  support_tickets (id PK, requester_id FK, issue_summary TEXT, status TEXT, priority TEXT, created_at DATETIME) →
    {{"index_cols": [{{"col":"id","unique":true}},{{"col":"status","unique":false}},{{"col":"priority","unique":false}},{{"col":"created_at","unique":false}}], "reasoning": "id unique; status/priority/created_at are common filters."}}

  employee_projects (emp_id FK, proj_id FK, allocation_pct INT) →
    {{"index_cols": [], "reasoning": "Junction table; FK columns are covered by graph edges."}}

Now decide for {table_name}:"""

    print(f"  [index]  '{table_name}' attempt {state.get('index_retries', 0) + 1}")
    try:
        result = extract_json(ask_ollama(prompt, json_mode=True))
    except ValueError:
        result = {}

    return {**state, "index_strategy": result, "index_errors": []}


def node_validate_index(state: StrategyState) -> StrategyState:
    strat      = state.get("index_strategy") or {}
    valid_cols = {c["name"] for c in state["table_info"]["columns"]}
    errors     = []

    idx_cols = strat.get("index_cols", [])
    if not isinstance(idx_cols, list):
        errors.append("index_cols must be a list.")
    else:
        for entry in idx_cols:
            if not isinstance(entry, dict):
                errors.append(f"Each index_cols entry must be a dict, got: {entry}")
                continue
            col = entry.get("col", "")
            if not col:
                errors.append("index_cols entry missing 'col' key.")
            elif col not in valid_cols:
                errors.append(f"index_cols: unknown column '{col}'.")
            if not isinstance(entry.get("unique", False), bool):
                errors.append(f"index_cols['{col}'].unique must be a bool.")

    return {**state, "index_errors": errors}


def route_index(state: StrategyState) -> str:
    if not state.get("index_errors"):       return "accept"
    if state.get("index_retries", 0) >= MAX_RETRIES: return "accept"
    return "reflect"

def node_reflect_index(state: StrategyState) -> StrategyState:
    return {**state, "index_retries": state.get("index_retries", 0) + 1}
