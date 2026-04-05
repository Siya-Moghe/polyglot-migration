from core.state import StrategyState
from core.llm_utils import ask_ollama, extract_json, enrich_table_context, MAX_RETRIES


def node_analyze_chroma(state: StrategyState) -> StrategyState:
    table_name, info, errors = state["table_name"], state["table_info"], state["errors"]
    sample_rows = state.get("sample_rows", [])

    enriched = enrich_table_context(table_name, info, sample_rows)
    retry_context = "\nPrevious attempt had errors — fix them:\n" + "\n".join(errors) if errors else ""

    prompt = f"""You are a Vector Database Architect. Your job is to design a ChromaDB embedding
strategy for the SQL table below. The goal is to combine meaningful text columns into a
single semantic sentence that can be embedded and searched.

{enriched}{retry_context}

Rules:
- Prioritize the most semantically meaningful text fields first:
  examples include title, description, content, abstract, summary, notes, review text, body.
- The template must read like a natural searchable document, not a raw database dump.
- Avoid using pure IDs, numeric values, timestamps, or FK integers unless they add useful retrieval context.
- If a long text field exists, it should almost always be included.
- "template": A Python .format()-style string using {{column_name}} placeholders.
- "used_columns": only columns referenced in the template.
- "skipped_columns": columns deliberately excluded.
- "join_related": list of FK-referred tables whose text data would enrich the document.
- "reasoning": one sentence explaining the embedding design.
--- EXAMPLE 1 ---
Table: knowledge_base | Columns: id (PK), title (TEXT), full_content (LONG TEXT), author_id (FK->employees), tags (TEXT)
Output:
{{
  "template": "Article: {{title}}. Content: {{full_content}}. Tags: {{tags}}",
  "used_columns": ["title", "full_content", "tags"],
  "skipped_columns": ["id", "author_id"],
  "join_related": ["employees"],
  "reasoning": "Combined title, full content, and tags into a rich searchable document; skipped id and FK integer."
}}

--- EXAMPLE 2 ---
Table: products | Columns: product_id (PK), name (TEXT), description (LONG TEXT), price (FLOAT), category_id (FK->categories)
Output:
{{
  "template": "Product: {{name}}. Description: {{description}}",
  "used_columns": ["name", "description"],
  "skipped_columns": ["product_id", "price", "category_id"],
  "join_related": ["categories"],
  "reasoning": "Name and description carry semantic meaning; price is numeric and unsuitable for text embedding."
}}

--- EXAMPLE 3 ---
Table: support_tickets | Columns: id (PK), requester_id (FK->employees), issue_summary (TEXT), status (TEXT), priority (TEXT)
Output:
{{
  "template": "Issue: {{issue_summary}}. Status: {{status}}. Priority: {{priority}}",
  "used_columns": ["issue_summary", "status", "priority"],
  "skipped_columns": ["id", "requester_id"],
  "join_related": ["employees"],
  "reasoning": "Issue summary drives semantic search; status and priority add filterable context."
}}

Now generate the ChromaDB strategy for the {table_name} table.
Return ONLY a JSON object matching the exact format above:"""

    print(f"  [chroma] Generating vector strategy... (Attempt {state['retries'] + 1})")
    try:
        strategy = extract_json(ask_ollama(prompt, json_mode=True))
    except ValueError:
        strategy = {}

    return {**state, "strategy": strategy, "errors": []}


def node_validate_chroma(state: StrategyState) -> StrategyState:
    strategy, info = state["strategy"] or {}, state["table_info"]
    errors = []

    valid_cols = {c["name"] for c in info["columns"]}
    fk_targets = {fk["referred_table"] for fk in info["foreign_keys"]}

    # -------------------------
    # Basic structure validation
    # -------------------------
    if "template" not in strategy:
        errors.append("Missing 'template' field")

    for col in strategy.get("used_columns", []):
        if col not in valid_cols:
            errors.append(f"Invalid column in used_columns: '{col}'")

    for j in strategy.get("join_related", []):
        if j not in fk_targets:
            errors.append(f"Invalid join_related table: '{j}' (not a FK target)")

    # -------------------------
    # Semantic quality validation
    # -------------------------
    semantic_candidates = {
        c["name"] for c in info["columns"]
        if any(k in c["name"].lower() for k in [
            "title", "description", "content", "abstract", "summary",
            "review", "body", "text", "notes", "comment"
        ])
    }

    used = set(strategy.get("used_columns", []))

    if semantic_candidates and not (used & semantic_candidates):
        errors.append(
            f"Chroma strategy ignored semantic text columns: {sorted(semantic_candidates)}"
        )

    return {**state, "errors": errors}



def route_chroma_validation(state: StrategyState) -> str:
    if not state["errors"]:
        return "accept"
    if state["retries"] >= MAX_RETRIES:
        return "accept"
    return "reflect"