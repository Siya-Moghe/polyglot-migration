import json
import re
import requests
from typing import Any

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"   # upgraded from phi3:mini
MAX_RETRIES = 3


# -------------------------
# Core LLM call (with JSON mode)
# -------------------------
def ask_ollama(prompt: str, json_mode: bool = True) -> str:
    """
    Call Ollama with optional JSON mode.
    JSON mode forces the model to output valid JSON — eliminates most
    extract_json failures and slashes the retry rate dramatically.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if json_mode:
        payload["format"] = "json"   # <-- Ollama native JSON mode

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        if resp.status_code != 200:
            print(f"\n[ERROR] OLLAMA SERVER ERROR: {resp.text}\n")
            return "{}"
        return resp.json()["response"].strip()

    except requests.exceptions.Timeout:
        print("\n[TIMEOUT] Ollama took longer than 180s. Forcing retry.\n")
        return "{}"
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] OLLAMA CONNECTION ERROR: Is Ollama running?\n")
        return "{}"


def extract_json(text: str) -> dict:
    """
    Parse JSON from LLM output. With JSON mode enabled this is almost always
    a direct json.loads — the regex fallbacks are kept as a safety net.
    """
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in LLM output:\n{text}")


# -------------------------
# Schema-aware prompt enrichment
# -------------------------
def enrich_table_context(table_name: str, info: dict, sample_rows: list[dict] = None) -> str:
    """
    Build a rich, model-friendly description of a table by adding:
      - PK/FK annotations on each column
      - Cardinality hints (long TEXT → semantic; pure INT → structural)
      - Sample row values (first 2 rows) so the model sees real data
      - Data shape summary

    This replaces the raw col_lines / fk_lines approach used in agents.
    """
    pk_set = set(info.get("primary_keys", []))
    fk_map = {
        col: fk["referred_table"]
        for fk in info.get("foreign_keys", [])
        for col in fk.get("columns", fk.get("constrained_columns", []))
    }

    lines = []
    text_col_count = 0
    int_col_count = 0
    fk_col_count = len(fk_map)

    for col in info["columns"]:
        name = col["name"]
        ctype = str(col["type"]).upper()
        tags = []

        if name in pk_set:
            tags.append("PRIMARY KEY")
        if name in fk_map:
            tags.append(f"FK -> {fk_map[name]}")

        # Cardinality hints from type
        if "TEXT" in ctype or "VARCHAR" in ctype or "CHAR" in ctype:
            text_col_count += 1
            if sample_rows:
                sample_val = sample_rows[0].get(name, "")
                if isinstance(sample_val, str) and len(sample_val) > 80:
                    tags.append("LONG TEXT — good for semantic embedding")
                else:
                    tags.append("SHORT TEXT")
        elif "INT" in ctype:
            int_col_count += 1
            tags.append("NUMERIC/ID")
        elif "FLOAT" in ctype or "REAL" in ctype or "DOUBLE" in ctype or "DECIMAL" in ctype:
            tags.append("NUMERIC — transactional/financial")
        elif "DATE" in ctype or "TIME" in ctype or "TIMESTAMP" in ctype:
            tags.append("TEMPORAL — audit/transactional")

        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        lines.append(f"  - {name} ({col['type']}){tag_str}")

    col_section = "\n".join(lines)

    # FK summary
    fk_lines = "\n".join(
        f"  - {fk.get('columns', fk.get('constrained_columns', []))} -> {fk['referred_table']}"
        for fk in info.get("foreign_keys", [])
    ) or "  None"

    # Data shape summary (routing hint)
    shape_hints = []
    if text_col_count >= 2:
        shape_hints.append(f"{text_col_count} text columns → may suit semantic search (Chroma)")
    if fk_col_count >= 2 and int_col_count >= 2:
        shape_hints.append(f"{fk_col_count} FK columns → may suit graph traversal (Neo4j)")
    if fk_col_count == 0 and text_col_count >= 1:
        shape_hints.append("No FKs, text fields → may suit document store (Mongo)")
    if int_col_count >= 2 and text_col_count == 0:
        shape_hints.append("Mostly numeric/temporal → may suit relational SQL")

    shape_str = "\n".join(f"  * {h}" for h in shape_hints) if shape_hints else "  * No strong structural hints."

    # Sample rows
    sample_str = ""
    if sample_rows:
        previews = []
        for row in sample_rows[:2]:
            short = {k: (str(v)[:60] + "..." if isinstance(v, str) and len(str(v)) > 60 else v)
                     for k, v in row.items()}
            previews.append(f"  {short}")
        sample_str = "\nSample rows (first 2):\n" + "\n".join(previews)

    return (
        f"Table: {table_name}\n"
        f"Columns:\n{col_section}\n"
        f"Foreign Keys:\n{fk_lines}\n"
        f"Data shape hints:\n{shape_str}"
        f"{sample_str}"
    )