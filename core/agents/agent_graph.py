"""
agent_graph.py — Plans the graph edge strategy for a table.
Pure planning only — no DB connection, no writes.
Returns: {use_graph, relations: [{label, from_col, to_table, to_col, edge_props}], reasoning}
"""
from core.state import StrategyState
from core.llm_utils import ask_ollama, extract_json, enrich_table_context, MAX_RETRIES


def node_analyze_graph(state: StrategyState) -> StrategyState:
    table_name  = state["table_name"]
    info        = state["table_info"]
    sample_rows = state.get("sample_rows", [])
    errors      = state.get("graph_errors", [])

    enriched  = enrich_table_context(table_name, info, sample_rows)
    fk_lines  = "\n".join(
        f"  - {fk['columns']} -> {fk['referred_table']}.{fk['referred_columns']}"
        for fk in info.get("foreign_keys", [])
    ) or "  None"
    retry_context = "\nPrevious attempt had errors — fix them:\n" + "\n".join(errors) if errors else ""

    prompt = f"""You are a Graph Data Specialist. Your ONLY job is to decide how FK columns
in this table should become RELATE edges in SurrealDB.

SurrealDB graph syntax:
  RELATE employees:1->BELONGS_TO->departments:3

Junction tables (2+ FKs, few other columns) should always become edges, with non-FK
columns becoming edge properties rather than separate record fields.

{enriched}

Foreign Keys:
{fk_lines}
{retry_context}

Return ONLY a JSON object:
{{
  "use_graph": true or false,
  "relations": [
    {{
      "label": "VERB_PHRASE",
      "from_col": "local_fk_column",
      "to_table": "referred_table_name",
      "to_col": "referred_pk_column",
      "edge_props": ["col_that_describes_the_relationship"]
    }}
  ],
  "reasoning": "One sentence."
}}

Rules:
- If use_graph=false, relations must be [].
- label must be UPPER_SNAKE_CASE verb phrase (WORKS_ON, BELONGS_TO, AUTHORED_BY).
- from_col and to_col must be actual columns in the table.
- edge_props: columns that describe the relationship itself (e.g. allocation_percentage on WORKS_ON).
- A table with no FKs must have use_graph=false.

Examples:
  employee_projects (emp_id FK->employees, proj_id FK->projects, allocation_pct) →
    {{"use_graph": true, "relations": [{{"label": "WORKS_ON", "from_col": "emp_id", "to_table": "projects", "to_col": "id", "edge_props": ["allocation_pct"]}}], "reasoning": "Pure junction table."}}

  departments (id, name, location) →
    {{"use_graph": false, "relations": [], "reasoning": "No FK columns."}}

Now decide for {table_name}:"""

    print(f"  [graph]  '{table_name}' attempt {state.get('graph_retries', 0) + 1}")
    try:
        result = extract_json(ask_ollama(prompt, json_mode=True))
    except ValueError:
        result = {}

    return {**state, "graph_strategy": result, "graph_errors": []}


def node_validate_graph(state: StrategyState) -> StrategyState:
    strat      = state.get("graph_strategy") or {}
    valid_cols = {c["name"] for c in state["table_info"]["columns"]}
    errors     = []

    if strat.get("use_graph"):
        rels = strat.get("relations", [])
        if not isinstance(rels, list) or len(rels) == 0:
            errors.append("use_graph=true but relations list is empty.")
        for rel in rels:
            if not isinstance(rel, dict):
                errors.append("Each relation must be a dict.")
                continue
            if not rel.get("label"):
                errors.append("Relation missing 'label'.")
            fc = rel.get("from_col", "")
            if fc and fc not in valid_cols:
                errors.append(f"from_col '{fc}' is not a valid column.")
            for ep in rel.get("edge_props", []):
                if isinstance(ep, str) and ep not in valid_cols:
                    errors.append(f"edge_prop '{ep}' is not a valid column.")

    return {**state, "graph_errors": errors}


def route_graph(state: StrategyState) -> str:
    if not state.get("graph_errors"):       return "accept"
    if state.get("graph_retries", 0) >= MAX_RETRIES: return "accept"
    return "reflect"

def node_reflect_graph(state: StrategyState) -> StrategyState:
    return {**state, "graph_retries": state.get("graph_retries", 0) + 1}
