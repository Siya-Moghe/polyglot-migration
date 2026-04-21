from core.state import StrategyState
from core.llm_utils import ask_ollama, extract_json, enrich_table_context, MAX_RETRIES


def node_analyze_mongo(state: StrategyState) -> StrategyState:
    table_name, info, errors = state["table_name"], state["table_info"], state["errors"]
    sample_rows = state.get("sample_rows", [])

    enriched = enrich_table_context(table_name, info, sample_rows)
    retry_context = "\nPrevious attempt had errors — fix them:\n" + "\n".join(errors) if errors else ""

    prompt = f"""You are a NoSQL Document Architect. Your job is to design a MongoDB document
structure for the SQL table below. Group related fields into nested sub-documents where it
makes logical sense (e.g., group name+email into a "profile" sub-document).

{enriched}{retry_context}

Rules:
- Think in terms of a self-contained JSON document.
- Group logically related descriptive fields into nested sub-documents where appropriate.
- Prefer denormalized, human-readable document structure over flat SQL-style preservation.
- Good Mongo candidates include profiles, product metadata, settings/config, catalogs, office/location info, and entity records.
- "fields": flat list of all top-level column names to include.
- "nested_fields": dict of {{parent_key: [child_col1, child_col2]}} for grouping related columns.
  Columns in nested_fields should NOT be repeated in the flat "fields" list.
- "join_related": list of FK-referred table names whose data should be embedded (denormalized) into this document.
- "join_keys": dict of {{table_name: {{"local_key": "col", "foreign_key": "col"}}}} for denormalization lookups.
- "reasoning": one sentence explaining the document design.

--- EXAMPLE 1 ---
Table: employees | Columns: id (PK), name (TEXT), email (TEXT), role (TEXT), dept_id (FK->departments)
Output:
{{
  "fields": ["id", "role", "dept_id"],
  "nested_fields": {{"profile": ["name", "email"]}},
  "join_related": ["departments"],
  "join_keys": {{"departments": {{"local_key": "dept_id", "foreign_key": "id"}}}},
  "reasoning": "Grouped name and email into a profile sub-document; denormalize department info for self-contained documents."
}}

--- EXAMPLE 2 ---
Table: departments | Columns: id (PK), name (TEXT), cost_center (TEXT), location (TEXT)
Output:
{{
  "fields": ["id", "name", "cost_center", "location"],
  "nested_fields": {{"address": ["location"]}},
  "join_related": [],
  "join_keys": {{}},
  "reasoning": "Self-contained entity with no FKs; location grouped as address metadata."
}}

--- EXAMPLE 3 ---
Table: projects | Columns: id (PK), project_name (TEXT), objective (TEXT), deadline (TEXT)
Output:
{{
  "fields": ["id", "deadline"],
  "nested_fields": {{"details": ["project_name", "objective"]}},
  "join_related": [],
  "join_keys": {{}},
  "reasoning": "Project name and objective are logically grouped as detail fields; deadline kept at top level for easy filtering."
}}

Now generate the MongoDB strategy for the {table_name} table.
Return ONLY a JSON object matching the exact format above:"""

    print(f"  [mongo] Generating document strategy... (Attempt {state['retries'] + 1})")
    try:
        strategy = extract_json(ask_ollama(prompt, json_mode=True))
    except ValueError:
        strategy = {}

    return {**state, "strategy": strategy, "errors": []}


def node_validate_mongo(state: StrategyState) -> StrategyState:
    strategy, info = state["strategy"] or {}, state["table_info"]
    errors = []

    valid_cols = {c["name"] for c in info["columns"]}

    if "fields" not in strategy and "used_columns" not in strategy:
        errors.append("Missing 'fields' list")

    # --- HARDENED FIX: Check if 'fields' is actually a list ---
    fields_list = strategy.get("fields", strategy.get("used_columns", []))
    if not isinstance(fields_list, list):
        errors.append("'fields' must be a list of strings.")
        fields_list = []
        
    for col in fields_list:
        # If the LLM hallucinated a dict instead of a string, catch it safely
        if not isinstance(col, str):
            errors.append(f"Invalid field format (expected string, got {type(col).__name__}). Do not use dictionaries.")
        elif col not in valid_cols:
            errors.append(f"Invalid column in fields: '{col}'")

    # --- HARDENED FIX: Check 'nested_fields' ---
    nested = strategy.get("nested_fields") or {}
    if not isinstance(nested, dict):
        errors.append("'nested_fields' must be a dictionary.")
        nested = {}

    for parent, children in nested.items():
        if not isinstance(children, list):
            errors.append(f"'nested_fields['{parent}']' must be a list of strings.")
            continue
        for col in children:
            if not isinstance(col, str):
                errors.append(f"Invalid nested field format (expected string, got {type(col).__name__}).")
            elif col not in valid_cols:
                errors.append(f"Invalid column in nested_fields['{parent}']: '{col}'")

    # Calculate all selected safely
    all_selected = set()
    for c in fields_list:
        if isinstance(c, str): all_selected.add(c)
    for children in nested.values():
        if isinstance(children, list):
            for c in children:
                if isinstance(c, str): all_selected.add(c)

    if len(all_selected) < max(1, len(valid_cols) // 2):
        errors.append("Mongo strategy selected too few columns for a useful document.")
        
    return {**state, "errors": errors}


def route_mongo_validation(state: StrategyState) -> str:
    if not state["errors"]:
        return "accept"
    if state["retries"] >= MAX_RETRIES:
        return "accept"
    return "reflect"