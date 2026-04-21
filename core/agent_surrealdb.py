"""
agent_surrealdb.py    LangGraph agent for SurrealDB migration strategy.

Replaces the four separate agents (chroma / mongo / neo4j / relational).
Instead of routing to different databases, this agent decides WHICH SurrealDB
features to activate for each table:

  use_vector   → store + search via SurrealDB vector index  (replaces Chroma)
  use_graph    → emit RELATE edges for FK relationships      (replaces Neo4j)
  use_nested   → group fields into nested objects            (replaces Mongo doc nesting)
  index_cols   → create DEFINE INDEX on high-filter columns  (replaces relational indexing)

A table can have any combination of these features active simultaneously.
"""

from core.state import StrategyState
from core.llm_utils import ask_ollama, extract_json, enrich_table_context, MAX_RETRIES


# ---------------------------------------------------------------------------
# Analyse node
# ---------------------------------------------------------------------------

def node_analyze_surrealdb(state: StrategyState) -> StrategyState:
    table_name = state["table_name"]
    info       = state["table_info"]
    errors     = state["errors"]
    sample_rows = state.get("sample_rows", [])

    enriched      = enrich_table_context(table_name, info, sample_rows)
    retry_context = (
        "\nPrevious attempt had errors — fix them:\n" + "\n".join(errors)
        if errors else ""
    )

    # Build FK summary for the prompt so the LLM can reason about edges
    fk_lines = "\n".join(
        f"  - {fk['columns']} -> {fk['referred_table']}.{fk['referred_columns']}"
        for fk in info.get("foreign_keys", [])
    ) or "  None"

    prompt = f"""You are a SurrealDB Architect. SurrealDB is a multi-model database that
supports documents, graphs, and vector search in a single engine.

Your task is to design the best SurrealDB storage strategy for the SQL table below.
You must decide which SurrealDB capabilities to activate, and how to structure the data.

{enriched}

Foreign Keys (these become RELATE edges in SurrealDB):
{fk_lines}
{retry_context}

--- SURREALDB CAPABILITIES TO CHOOSE FROM ---

1. use_vector (bool)
   Activate when the table has semantic text columns (descriptions, summaries, content,
   reviews, notes, abstracts). SurrealDB will store an embedding alongside the record
   so it can be searched with SELECT ... WHERE vector::similarity::cosine(...) > threshold.

2. use_graph (bool)
   Activate when the table has 1+ FK columns. Each FK becomes a RELATE statement:
     RELATE source_table:id -> relation_name -> target_table:fk_value
   Junction/mapping tables (2+ FKs, few other columns) should always use_graph = true.

3. use_nested (bool)
   Activate when logically-related fields should be grouped into a nested object
   for cleaner document reads (e.g. name+email -> profile: {{name, email}}).

4. index_cols (list of column names)
   Columns that will get a DEFINE INDEX for fast exact-match / range filtering.
   Always include: primary keys, timestamps, status/priority enums, numeric aggregation
   columns (amount, price, total). Do NOT index long text columns.

--- OUTPUT FORMAT ---

Return ONLY a JSON object with this exact structure:

{{
  "use_vector": true or false,
  "vector_template": "A Python .format()-style string for the text to embed. Use {{col}} placeholders. Empty string if use_vector is false.",
  "vector_cols": ["col1", "col2"],
  "use_graph": true or false,
  "relations": [
    {{
      "label": "RELATION_NAME",
      "from_col": "local_fk_column",
      "to_table": "referred_table",
      "to_col": "referred_pk_column",
      "edge_props": ["extra_col_on_edge"]
    }}
  ],
  "use_nested": true or false,
  "nested_fields": {{"parent_key": ["child_col1", "child_col2"]}},
  "top_level_fields": ["col1", "col2"],
  "index_cols": ["col1", "col2"],
  "reasoning": "One sentence explaining the strategy."
}}

Rules:
- top_level_fields: flat columns NOT included in nested_fields.
- vector_cols: columns whose text is concatenated into vector_template.
- relations[].edge_props: extra columns that become properties ON the edge (not the node).
- If use_graph is false, relations must be an empty list [].
- If use_nested is false, nested_fields must be an empty object {{}}.
- If use_vector is false, vector_template must be "" and vector_cols must be [].

--- EXAMPLES ---

Table: knowledge_base | Columns: id (PK), title (TEXT), full_content (LONG TEXT), author_id (FK->employees), tags (TEXT)
{{
  "use_vector": true,
  "vector_template": "Article: {{title}}. Content: {{full_content}}. Tags: {{tags}}",
  "vector_cols": ["title", "full_content", "tags"],
  "use_graph": true,
  "relations": [{{"label": "AUTHORED_BY", "from_col": "author_id", "to_table": "employees", "to_col": "id", "edge_props": []}}],
  "use_nested": false,
  "nested_fields": {{}},
  "top_level_fields": ["id", "title", "full_content", "tags", "author_id"],
  "index_cols": ["id"],
  "reasoning": "Rich text content warrants vector search; author FK becomes a graph edge."
}}

Table: employee_projects | Columns: emp_id (FK->employees), proj_id (FK->projects), allocation_percentage (INT)
{{
  "use_vector": false,
  "vector_template": "",
  "vector_cols": [],
  "use_graph": true,
  "relations": [{{"label": "WORKS_ON", "from_col": "emp_id", "to_table": "projects", "to_col": "id", "edge_props": ["allocation_percentage"]}}],
  "use_nested": false,
  "nested_fields": {{}},
  "top_level_fields": ["emp_id", "proj_id", "allocation_percentage"],
  "index_cols": [],
  "reasoning": "Pure junction table; both FKs become a RELATE edge with allocation as an edge property."
}}

Table: employees | Columns: id (PK), name (TEXT), email (TEXT), role (TEXT), dept_id (FK->departments)
{{
  "use_vector": true,
  "vector_template": "Employee {{name}} works as {{role}}",
  "vector_cols": ["name", "role"],
  "use_graph": true,
  "relations": [{{"label": "BELONGS_TO", "from_col": "dept_id", "to_table": "departments", "to_col": "id", "edge_props": []}}],
  "use_nested": true,
  "nested_fields": {{"contact": ["email"]}},
  "top_level_fields": ["id", "name", "role", "dept_id"],
  "index_cols": ["id", "role"],
  "reasoning": "Employee name/role suits vector search; email grouped under contact; dept_id becomes a graph edge."
}}

Now generate the SurrealDB strategy for the {table_name} table.
Return ONLY a JSON object matching the exact format above:"""

    print(f"  [surrealdb] Generating strategy... (Attempt {state['retries'] + 1})")
    try:
        strategy = extract_json(ask_ollama(prompt, json_mode=True))
    except ValueError:
        strategy = {}

    return {**state, "strategy": strategy, "errors": []}


# ---------------------------------------------------------------------------
# Validate node
# ---------------------------------------------------------------------------

def node_validate_surrealdb(state: StrategyState) -> StrategyState:
    strategy = state.get("strategy") or {}
    info     = state["table_info"]
    errors   = []

    valid_cols = {c["name"] for c in info["columns"]}
    fk_map     = {
        fk["columns"][0]: fk["referred_table"]
        for fk in info.get("foreign_keys", [])
        if fk.get("columns")
    }

    # --- vector ---
    if strategy.get("use_vector"):
        tmpl = strategy.get("vector_template", "")
        if not isinstance(tmpl, str) or not tmpl.strip():
            errors.append("use_vector=true but vector_template is empty.")
        vcols = strategy.get("vector_cols", [])
        if not isinstance(vcols, list) or len(vcols) == 0:
            errors.append("use_vector=true but vector_cols is empty.")
        for c in vcols:
            if isinstance(c, str) and c not in valid_cols:
                errors.append(f"vector_cols references unknown column '{c}'.")

    # --- graph / relations ---
    if strategy.get("use_graph"):
        relations = strategy.get("relations", [])
        if not isinstance(relations, list) or len(relations) == 0:
            errors.append("use_graph=true but relations list is empty.")
        for rel in relations:
            if not isinstance(rel, dict):
                errors.append("Each relation must be a dict.")
                continue
            if not rel.get("label"):
                errors.append("Relation is missing 'label'.")
            fc = rel.get("from_col", "")
            if fc and fc not in valid_cols:
                errors.append(f"Relation from_col '{fc}' is not a valid column.")
            for ep in rel.get("edge_props", []):
                if isinstance(ep, str) and ep not in valid_cols:
                    errors.append(f"edge_prop '{ep}' is not a valid column.")

    # --- nested ---
    if strategy.get("use_nested"):
        nested = strategy.get("nested_fields", {})
        if not isinstance(nested, dict) or len(nested) == 0:
            errors.append("use_nested=true but nested_fields is empty.")
        for parent, children in nested.items():
            if not isinstance(children, list):
                errors.append(f"nested_fields['{parent}'] must be a list.")
                continue
            for c in children:
                if isinstance(c, str) and c not in valid_cols:
                    errors.append(f"nested_fields['{parent}'] references unknown column '{c}'.")

    # --- top_level_fields ---
    top = strategy.get("top_level_fields", [])
    if not isinstance(top, list):
        errors.append("top_level_fields must be a list.")
    else:
        for c in top:
            if isinstance(c, str) and c not in valid_cols:
                errors.append(f"top_level_fields references unknown column '{c}'.")

    # --- index_cols ---
    idx = strategy.get("index_cols", [])
    if not isinstance(idx, list):
        errors.append("index_cols must be a list.")
    else:
        for c in idx:
            if isinstance(c, str) and c not in valid_cols:
                errors.append(f"index_cols references unknown column '{c}'.")

    # --- at minimum some columns must be covered ---
    all_mapped: set = set()
    for c in strategy.get("top_level_fields", []):
        if isinstance(c, str):
            all_mapped.add(c)
    for children in strategy.get("nested_fields", {}).values():
        if isinstance(children, list):
            all_mapped.update(c for c in children if isinstance(c, str))

    if len(all_mapped) == 0:
        errors.append("Strategy maps zero columns — at least top_level_fields must be non-empty.")

    return {**state, "errors": errors}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def route_surrealdb_validation(state: StrategyState) -> str:
    if not state["errors"]:
        return "accept"
    if state["retries"] >= MAX_RETRIES:
        return "accept"
    return "reflect"