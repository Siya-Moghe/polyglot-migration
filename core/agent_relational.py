from core.state import StrategyState
from core.llm_utils import ask_ollama, extract_json, enrich_table_context, MAX_RETRIES


def node_analyze_relational(state: StrategyState) -> StrategyState:
    table_name, info, errors = state["table_name"], state["table_info"], state["errors"]
    sample_rows = state.get("sample_rows", [])

    enriched = enrich_table_context(table_name, info, sample_rows)
    retry_context = "\nPrevious attempt had errors — fix them:\n" + "\n".join(errors) if errors else ""

    prompt = f"""You are a Relational Database Architect. Your job is to design a SQL preservation
strategy for the table below — deciding which columns to keep, which to drop, and the primary
use case for this table in the target relational database.

{enriched}{retry_context}

Rules:
- Only preserve a table as relational if its primary value is exact filtering, reporting, transactions, auditability, or structured operational queries.
- Do NOT keep a table relational if its main value is semantic text search (better for Chroma), document-style self-contained entities (better for Mongo), or relationship traversal (better for Neo4j).
- "preserve_as": always "table".
- "used_columns": columns to keep in the target SQL table.
- "skipped_columns": columns to omit (e.g., redundant free-text columns better handled in Chroma).
- "join_related": FK-referred tables this table regularly JOINs with.
- "primary_use": one of "transactional", "reporting", "audit", "lookup", "operational".
- "reasoning": one sentence explaining why this table stays relational.

--- EXAMPLE 1 ---
Table: support_tickets | Columns: id (PK), requester_id (FK->employees), issue_summary (TEXT), status (TEXT), priority (TEXT)
Output:
{{
  "preserve_as": "table",
  "used_columns": ["id", "requester_id", "status", "priority"],
  "skipped_columns": ["issue_summary"],
  "join_related": ["employees"],
  "primary_use": "transactional",
  "reasoning": "Ticket status and priority are best queried with exact SQL filters; issue_summary is better handled in Chroma for semantic search."
}}

--- EXAMPLE 2 ---
Table: departments | Columns: id (PK), name (TEXT), cost_center (TEXT), location (TEXT)
Output:
{{
  "preserve_as": "table",
  "used_columns": ["id", "name", "cost_center", "location"],
  "skipped_columns": [],
  "join_related": [],
  "primary_use": "lookup",
  "reasoning": "Small lookup/reference table with structured fields; best kept in relational DB for FK joins from employees."
}}

--- EXAMPLE 3 ---
Table: sales_transactions | Columns: id (PK), customer_id (FK->customers), amount (FLOAT), tax (FLOAT), created_at (TIMESTAMP), status (TEXT)
Output:
{{
  "preserve_as": "table",
  "used_columns": ["id", "customer_id", "amount", "tax", "created_at", "status"],
  "skipped_columns": [],
  "join_related": ["customers"],
  "primary_use": "transactional",
  "reasoning": "Financial/transactional table with numeric and temporal fields requiring precise aggregation, filtering and JOIN operations."
}}

Now generate the relational strategy for the {table_name} table.
Return ONLY a JSON object matching the exact format above:"""

    print(f"  [relational] Generating relational strategy... (Attempt {state['retries'] + 1})")
    try:
        strategy = extract_json(ask_ollama(prompt, json_mode=True))
    except ValueError:
        strategy = {}

    return {**state, "strategy": strategy, "errors": []}


def node_validate_relational(state: StrategyState) -> StrategyState:
    strategy, info = state["strategy"] or {}, state["table_info"]
    errors = []

    valid_cols = {c["name"] for c in info["columns"]}
    fk_targets = {fk["referred_table"] for fk in info["foreign_keys"]}

    if strategy.get("preserve_as") != "table":
        errors.append("Missing or invalid 'preserve_as' (must be 'table')")
    if "primary_use" not in strategy:
        errors.append("Missing 'primary_use'")
    if "reasoning" not in strategy:
        errors.append("Missing 'reasoning'")

    for col in strategy.get("used_columns", []):
        if col not in valid_cols:
            errors.append(f"Invalid column in used_columns: '{col}'")
    for col in strategy.get("skipped_columns", []):
        if col not in valid_cols:
            errors.append(f"Invalid column in skipped_columns: '{col}'")
    for j in strategy.get("join_related", []):
        if j not in fk_targets:
            errors.append(f"Invalid join_related table: '{j}'")


    semantic_candidates = {
        c["name"] for c in info["columns"]
        if any(k in c["name"].lower() for k in [
            "content", "description", "body", "summary", "review", "abstract", "notes"
        ])
    }

    if semantic_candidates and strategy.get("primary_use") == "lookup":
        errors.append("Relational strategy suspicious: semantic-heavy table classified as lookup.")
        

    return {**state, "errors": errors}


def route_relational_validation(state: StrategyState) -> str:
    if not state["errors"]:
        return "accept"
    if state["retries"] >= MAX_RETRIES:
        return "accept"
    return "reflect"