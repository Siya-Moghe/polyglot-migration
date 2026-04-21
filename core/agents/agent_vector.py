"""
agent_vector.py — Plans the vector search strategy for a table.
Pure planning only — no DB connection, no writes.
Returns: {use_vector, vector_template, vector_cols, reasoning}
"""
from core.state import StrategyState
from core.llm_utils import ask_ollama, extract_json, enrich_table_context, MAX_RETRIES


def node_analyze_vector(state: StrategyState) -> StrategyState:
    table_name  = state["table_name"]
    info        = state["table_info"]
    sample_rows = state.get("sample_rows", [])
    errors      = state.get("vector_errors", [])

    enriched      = enrich_table_context(table_name, info, sample_rows)
    retry_context = "\nPrevious attempt had errors — fix them:\n" + "\n".join(errors) if errors else ""

    prompt = f"""You are a Vector Search Specialist. Your ONLY job is to decide whether this
table needs vector/semantic search in SurrealDB, and if so, design the embedding template.

SurrealDB stores an embedding float array on each record and supports:
  SELECT * FROM t WHERE vector::similarity::cosine(embedding, $q) > 0.7

Activate vector search for columns with natural language text users would search semantically:
descriptions, summaries, article content, reviews, notes, issue text, bios.
Do NOT activate for IDs, dates, status enums, or numeric values.

{enriched}
{retry_context}

Return ONLY a JSON object:
{{
  "use_vector": true or false,
  "vector_template": "Python .format() string using {{col}} placeholders. Empty string if use_vector=false.",
  "vector_cols": ["col1", "col2"],
  "reasoning": "One sentence."
}}

Rules:
- vector_cols must only reference columns that exist in the table.
- vector_template must contain at least one {{col}} placeholder when use_vector=true.
- If use_vector=false then vector_template="" and vector_cols=[].

Examples:
  knowledge_base (id, title, full_content, tags) →
    {{"use_vector": true, "vector_template": "Article: {{title}}. {{full_content}}. Tags: {{tags}}", "vector_cols": ["title","full_content","tags"], "reasoning": "Rich text content warrants semantic search."}}

  employee_projects (emp_id, proj_id, allocation_pct) →
    {{"use_vector": false, "vector_template": "", "vector_cols": [], "reasoning": "No semantic text columns."}}

Now decide for {table_name}:"""

    print(f"  [vector] '{table_name}' attempt {state.get('vector_retries', 0) + 1}")
    try:
        result = extract_json(ask_ollama(prompt, json_mode=True))
    except ValueError:
        result = {}

    return {**state, "vector_strategy": result, "vector_errors": []}


def node_validate_vector(state: StrategyState) -> StrategyState:
    strat      = state.get("vector_strategy") or {}
    valid_cols = {c["name"] for c in state["table_info"]["columns"]}
    errors     = []

    if strat.get("use_vector"):
        tmpl = strat.get("vector_template", "")
        if not isinstance(tmpl, str) or not tmpl.strip():
            errors.append("use_vector=true but vector_template is empty.")
        elif "{" not in tmpl:
            errors.append("vector_template has no {col} placeholders.")
        vcols = strat.get("vector_cols", [])
        if not isinstance(vcols, list) or len(vcols) == 0:
            errors.append("use_vector=true but vector_cols is empty.")
        for c in vcols:
            if isinstance(c, str) and c not in valid_cols:
                errors.append(f"vector_cols: unknown column '{c}'.")

    return {**state, "vector_errors": errors}


def route_vector(state: StrategyState) -> str:
    if not state.get("vector_errors"):       return "accept"
    if state.get("vector_retries", 0) >= MAX_RETRIES: return "accept"
    return "reflect"

def node_reflect_vector(state: StrategyState) -> StrategyState:
    return {**state, "vector_retries": state.get("vector_retries", 0) + 1}
