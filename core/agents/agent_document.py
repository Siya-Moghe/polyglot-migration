"""
agent_document.py — Plans the document structure for a table.
Pure planning only — no DB connection, no writes.
Returns: {use_nested, nested_fields, top_level_fields, reasoning}
"""
from core.state import StrategyState
from core.llm_utils import ask_ollama, extract_json, enrich_table_context, MAX_RETRIES


def node_analyze_document(state: StrategyState) -> StrategyState:
    table_name  = state["table_name"]
    info        = state["table_info"]
    sample_rows = state.get("sample_rows", [])
    errors      = state.get("document_errors", [])

    enriched  = enrich_table_context(table_name, info, sample_rows)
    all_cols  = [c["name"] for c in info["columns"]]
    retry_context = "\nPrevious attempt had errors — fix them:\n" + "\n".join(errors) if errors else ""

    prompt = f"""You are a Document Model Specialist. Your ONLY job is to decide the document
structure for this table in SurrealDB — which columns stay flat at the top level, and which
get grouped into nested sub-objects.

SurrealDB stores records as schemaless documents. You can nest:
  Instead of: {{ name: "Alice", email: "a@x.com", phone: "555" }}
  Store as:   {{ name: "Alice", contact: {{ email: "a@x.com", phone: "555" }} }}

Group columns when they form a clear logical sub-object (contact info, address, metadata).
Don't nest unless it genuinely improves readability.

{enriched}

All columns: {all_cols}
{retry_context}

Return ONLY a JSON object:
{{
  "use_nested": true or false,
  "nested_fields": {{"parent_key": ["child_col1", "child_col2"]}},
  "top_level_fields": ["col1", "col2"],
  "reasoning": "One sentence."
}}

Rules:
- Every column must appear in EITHER top_level_fields OR nested_fields — not both, not neither.
- nested_fields keys are new parent names you're creating (e.g. "contact", "address", "meta").
- If use_nested=false, nested_fields must be {{}}.
- All column names must exactly match the list above.

Examples:
  employees (id, name, email, phone, role, dept_id) →
    {{"use_nested": true, "nested_fields": {{"contact": ["email","phone"]}}, "top_level_fields": ["id","name","role","dept_id"], "reasoning": "email and phone form a natural contact sub-object."}}

  employee_projects (emp_id, proj_id, allocation_pct) →
    {{"use_nested": false, "nested_fields": {{}}, "top_level_fields": ["emp_id","proj_id","allocation_pct"], "reasoning": "Junction table, nothing to nest."}}

Now decide for {table_name}:"""

    print(f"  [doc]    '{table_name}' attempt {state.get('document_retries', 0) + 1}")
    try:
        result = extract_json(ask_ollama(prompt, json_mode=True))
    except ValueError:
        result = {}

    return {**state, "document_strategy": result, "document_errors": []}


def node_validate_document(state: StrategyState) -> StrategyState:
    strat      = state.get("document_strategy") or {}
    valid_cols = {c["name"] for c in state["table_info"]["columns"]}
    errors     = []

    top    = strat.get("top_level_fields", [])
    nested = strat.get("nested_fields", {})

    if not isinstance(top, list):
        errors.append("top_level_fields must be a list.")
    else:
        for c in top:
            if isinstance(c, str) and c not in valid_cols:
                errors.append(f"top_level_fields: unknown column '{c}'.")

    if not isinstance(nested, dict):
        errors.append("nested_fields must be a dict.")
    else:
        if strat.get("use_nested") and len(nested) == 0:
            errors.append("use_nested=true but nested_fields is empty.")
        for parent, children in nested.items():
            if not isinstance(children, list):
                errors.append(f"nested_fields['{parent}'] must be a list.")
                continue
            for c in children:
                if isinstance(c, str) and c not in valid_cols:
                    errors.append(f"nested_fields['{parent}']: unknown column '{c}'.")

    nested_flat = {c for ch in nested.values() if isinstance(ch, list) for c in ch}
    covered     = (set(top) if isinstance(top, list) else set()) | nested_flat
    if len(covered) == 0:
        errors.append("No columns mapped — top_level_fields and nested_fields are both empty.")

    return {**state, "document_errors": errors}


def route_document(state: StrategyState) -> str:
    if not state.get("document_errors"):       return "accept"
    if state.get("document_retries", 0) >= MAX_RETRIES: return "accept"
    return "reflect"

def node_reflect_document(state: StrategyState) -> StrategyState:
    return {**state, "document_retries": state.get("document_retries", 0) + 1}
