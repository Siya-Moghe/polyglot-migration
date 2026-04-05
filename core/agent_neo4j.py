from core.state import StrategyState
from core.llm_utils import ask_ollama, extract_json, enrich_table_context, MAX_RETRIES


def node_analyze_neo4j(state: StrategyState) -> StrategyState:
    table_name, info, errors = state["table_name"], state["table_info"], state["errors"]
    sample_rows = state.get("sample_rows", [])

    enriched = enrich_table_context(table_name, info, sample_rows)
    retry_context = "\nPrevious attempt had errors — fix them:\n" + "\n".join(errors) if errors else ""

    prompt = f"""You are a Graph Database Architect. Your job is to design a Neo4j node and
relationship structure for the SQL table below. FK columns should become edges (relationships)
between nodes. Non-FK columns become node properties.

{enriched}{retry_context}

Rules:
- Neo4j is best when relationships are the primary meaning of the table.
- Junction/mapping tables should strongly emphasize edges over standalone node properties.
- FK columns should usually become relationships, not plain node attributes.
- Non-FK descriptive columns should become node or edge properties.
- "template": A Cypher-style pattern showing node labels and relationship types.
  Use (n:Label {{prop: '{{col}}'}}) notation. Show relationships for each FK.
- "used_columns": all columns that become node properties or edge properties.
- "skipped_columns": columns to omit (usually none for Neo4j; FKs become edges not properties).
- "join_related": list of FK-referred table names (these become related node labels).
- "reasoning": one sentence explaining the graph model.

--- EXAMPLE 1 ---
Table: employee_projects | Columns: emp_id (FK->employees), proj_id (FK->projects), allocation_percentage (INT)
Output:
{{
  "template": "(e:Employee {{node_id: '{{emp_id}}'}})-[:WORKS_ON {{allocation: '{{allocation_percentage}}'}}]->(p:Project {{node_id: '{{proj_id}}'}})",
  "used_columns": ["emp_id", "proj_id", "allocation_percentage"],
  "skipped_columns": [],
  "join_related": ["employees", "projects"],
  "reasoning": "Pure junction table; emp_id and proj_id become node references, allocation_percentage becomes a relationship property."
}}

--- EXAMPLE 2 ---
Table: employees | Columns: id (PK), name (TEXT), email (TEXT), role (TEXT), dept_id (FK->departments)
Output:
{{
  "template": "(e:Employee {{node_id: '{{id}}', name: '{{name}}', email: '{{email}}', role: '{{role}}'}})-[:BELONGS_TO]->(d:Departments {{node_id: '{{dept_id}}'}})",
  "used_columns": ["id", "name", "email", "role", "dept_id"],
  "skipped_columns": [],
  "join_related": ["departments"],
  "reasoning": "Employee node with personal attributes linked to Department via BELONGS_TO relationship."
}}

--- EXAMPLE 3 ---
Table: support_tickets | Columns: id (PK), requester_id (FK->employees), issue_summary (TEXT), status (TEXT), priority (TEXT)
Output:
{{
  "template": "(t:SupportTicket {{node_id: '{{id}}', summary: '{{issue_summary}}', status: '{{status}}', priority: '{{priority}}'}})-[:RAISED_BY]->(e:Employee {{node_id: '{{requester_id}}'}})",
  "used_columns": ["id", "requester_id", "issue_summary", "status", "priority"],
  "skipped_columns": [],
  "join_related": ["employees"],
  "reasoning": "Ticket node linked to the requesting employee via RAISED_BY edge."
}}

Now generate the Neo4j strategy for the {table_name} table.
Return ONLY a JSON object matching the exact format above:"""

    print(f"  [neo4j] Generating graph strategy... (Attempt {state['retries'] + 1})")
    try:
        strategy = extract_json(ask_ollama(prompt, json_mode=True))
    except ValueError:
        strategy = {}

    return {**state, "strategy": strategy, "errors": []}


def node_validate_neo4j(state: StrategyState) -> StrategyState:
    strategy, info = state["strategy"] or {}, state["table_info"]
    errors = []

    valid_cols = {c["name"] for c in info["columns"]}

    if "template" not in strategy:
        errors.append("Missing 'template' field")
    for col in strategy.get("used_columns", []):
        if col not in valid_cols:
            errors.append(f"Invalid column in used_columns: '{col}'")

    
    fk_cols = {
    col
    for fk in info.get("foreign_keys", [])
    for col in fk.get("columns", [])
    }

    used = set(strategy.get("used_columns", []))

    if len(fk_cols) >= 2 and not fk_cols.issubset(used):
        errors.append("Neo4j strategy failed to include all FK relationship columns.")

    return {**state, "errors": errors}


def route_neo4j_validation(state: StrategyState) -> str:
    if not state["errors"]:
        return "accept"
    if state["retries"] >= MAX_RETRIES:
        return "accept"
    return "reflect"