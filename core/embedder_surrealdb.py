"""
embedder_surrealdb.py  –  Stage 3 export: write everything to SurrealDB.

Uses the blocking (synchronous) SurrealDB Python SDK.
All method signatures verified against the installed SDK source.

Key SDK methods used:
  db.signin({"username": ..., "password": ...})
  db.use(namespace, database)
  db.query(sql_string)                          — DDL statements
  db.insert(table_name, list_of_dicts)          — batch insert records
  db.insert_relation(label, {"in":..,"out":..}) — graph edges
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

from sentence_transformers import SentenceTransformer
from surrealdb import BlockingWsSurrealConnection

from core.introspect import fetch_rows_in_batches


# ---------------------------------------------------------------------------
# Shared embedding model  (loaded once)
# ---------------------------------------------------------------------------
_embed_model: SentenceTransformer | None = None

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


# ---------------------------------------------------------------------------
# PII masking
# ---------------------------------------------------------------------------
_PII_KEYWORDS = ["email", "ssn", "password", "phone", "credit_card", "social", "address"]

def mask_pii(row: Dict[str, Any]) -> Dict[str, Any]:
    masked = row.copy()
    for key, value in masked.items():
        if any(k in key.lower() for k in _PII_KEYWORDS) and value:
            masked[key] = f"REDACTED-{hashlib.sha256(str(value).encode()).hexdigest()[:8]}"
    return masked


# ---------------------------------------------------------------------------
# Text formatting helper
# ---------------------------------------------------------------------------

class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"

def safe_format(template: str, row: Dict[str, Any]) -> str:
    return template.format_map(_SafeDict(row))


# ---------------------------------------------------------------------------
# Strategy normalisation
# ---------------------------------------------------------------------------

def normalize_strategy(
    table_name: str,
    strategy: Dict[str, Any],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    info     = schema.get(table_name, {})
    all_cols = [c["name"] for c in info.get("columns", [])]

    def _valid_cols(lst: Any) -> list:
        if not isinstance(lst, list): return []
        return [c for c in lst if isinstance(c, str) and c in all_cols]

    top_fields = _valid_cols(strategy.get("top_level_fields", all_cols))
    if not top_fields: top_fields = all_cols

    nested: dict = {}
    if isinstance(strategy.get("nested_fields"), dict):
        for parent, children in strategy.get("nested_fields").items():
            safe_children = _valid_cols(children)
            if safe_children: nested[parent] = safe_children

    relations: list = []
    for rel in strategy.get("relations", []):
        if isinstance(rel, dict) and rel.get("from_col") in all_cols:
            relations.append({
                "label":      rel.get("label", f"RELATES_TO_{rel.get('to_table','?').upper()}"),
                "from_col":   rel["from_col"],
                "to_table":   rel.get("to_table", ""),
                "to_col":     rel.get("to_col", "id"),
                "edge_props": _valid_cols(rel.get("edge_props", [])),
            })

    use_vector = bool(strategy.get("use_vector"))
    
    return {
        "use_vector":       use_vector,
        "vector_template":  strategy.get("vector_template", "") if use_vector else "",
        "vector_cols":      _valid_cols(strategy.get("vector_cols", [])),
        "use_graph":        bool(strategy.get("use_graph")) and len(relations) > 0,
        "relations":        relations,
        "use_nested":       bool(strategy.get("use_nested")) and len(nested) > 0,
        "nested_fields":    nested,
        "top_level_fields": top_fields,
        "index_cols":       _valid_cols(strategy.get("index_cols", [])),
        "reasoning":        strategy.get("reasoning", ""),
    }

# ---------------------------------------------------------------------------
# Build a single record dict from a source row
# ---------------------------------------------------------------------------

def _build_record(
    row: Dict[str, Any],
    strategy: Dict[str, Any],
    embedding: list[float] | None,
) -> Dict[str, Any]:
    doc: Dict[str, Any] = {}

    for col in strategy["top_level_fields"]:
        if col in row:
            doc[col] = row[col]

    for parent, children in strategy["nested_fields"].items():
        doc[parent] = {c: row[c] for c in children if c in row}

    if embedding is not None:
        doc["embedding"] = embedding

    return doc


# ---------------------------------------------------------------------------
# DDL  (run once per table before inserting rows)
# ---------------------------------------------------------------------------

def _define_table_schema(
    db: BlockingWsSurrealConnection,
    table_name: str,
    strategy: Dict[str, Any],
) -> None:
    db.query(f"DEFINE TABLE IF NOT EXISTS {table_name} SCHEMALESS;")

    for col in strategy["index_cols"]:
        db.query(
            f"DEFINE INDEX IF NOT EXISTS idx_{table_name}_{col} "
            f"ON TABLE {table_name} FIELDS {col};"
        )

    if strategy["use_vector"]:
        db.query(
            f"DEFINE INDEX IF NOT EXISTS idx_{table_name}_embedding "
            f"ON TABLE {table_name} "
            f"FIELDS embedding "
            f"HNSW DIMENSION 384 DIST COSINE;"
        )

    print(f"  [surreal] Schema defined for '{table_name}'")


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export_to_surrealdb(
    table_name: str,
    strategy: Dict[str, Any],
    schema: Dict[str, Any],
    db_url: str,
    *,
    surreal_url: str  = "ws://localhost:8000/rpc",
    surreal_ns: str   = "migration",
    surreal_db: str   = "target",
    surreal_user: str = "root",
    surreal_pass: str = "root",
    batch_size: int   = 500,
) -> int:
    print(f"\n[SurrealDB] Exporting '{table_name}' ...")
    print(
        f"  features → "
        f"vector={strategy['use_vector']} "
        f"graph={strategy['use_graph']} "
        f"nested={strategy['use_nested']} "
        f"indexes={strategy['index_cols']}"
    )

    embed_model = _get_embed_model() if strategy["use_vector"] else None

    # BlockingWsSurrealConnection appends /rpc to the URL itself
    db = BlockingWsSurrealConnection(surreal_url)

    # signin expects {"username": ..., "password": ...}  (verified from SDK source)
    db.signin({"username": surreal_user, "password": surreal_pass})
    db.use(surreal_ns, surreal_db)

    _define_table_schema(db, table_name, strategy)

    total_inserted = 0

    for batch in fetch_rows_in_batches(db_url, table_name, batch_size=batch_size):

        # Compute embeddings for the whole batch at once
        if strategy["use_vector"] and embed_model:
            texts      = [safe_format(strategy["vector_template"], r) for r in batch]
            embeddings = embed_model.encode(texts).tolist()
        else:
            embeddings = [None] * len(batch)

        records           = []
        relations_to_emit = []

        for i, raw_row in enumerate(batch):
            row    = mask_pii(raw_row)
            pk_val = row.get("id", total_inserted + i)
            doc    = _build_record(row, strategy, embeddings[i])
            # SurrealDB record id format:  table:pk
            doc["id"] = f"{table_name}:{pk_val}"
            records.append(doc)

            if strategy["use_graph"]:
                for rel in strategy["relations"]:
                    fc = rel["from_col"]
                    fv = row.get(fc)
                    if fv is None:
                        continue
                    edge: Dict[str, Any] = {
                        "in":  f"{table_name}:{pk_val}",
                        "out": f"{rel['to_table']}:{fv}",
                    }
                    for ep in rel["edge_props"]:
                        if ep in row:
                            edge[ep] = row[ep]
                    relations_to_emit.append((rel["label"], edge))

        # Batch insert records  — insert(table, list_of_dicts)
        if records:
            db.insert(table_name, records)
            total_inserted += len(records)
            print(f"  ... inserted batch of {len(records)} records")

        # Insert graph edges  — insert_relation(label, edge_dict)
        for label, edge in relations_to_emit:
            query = f"RELATE {edge['in']}->{label}->{edge['out']};"
            db.query(query)

        if relations_to_emit:
            print(f"  ... created {len(relations_to_emit)} graph edges")

    db.close()
    print(f"  [+] '{table_name}' complete — {total_inserted} records")
    return total_inserted


# ---------------------------------------------------------------------------
# Public entry point — called by pipeline.py and app.py
# ---------------------------------------------------------------------------

def embed_and_store(
    table_name: str,
    strategy: Dict[str, Any],
    schema: Dict[str, Any],
    db_url: str,
    *,
    surreal_url: str  = "ws://localhost:8000/rpc",
    surreal_ns: str   = "migration",
    surreal_db: str   = "target",
    surreal_user: str = "root",
    surreal_pass: str = "root",
    batch_size: int   = 500,
) -> int:
    clean = normalize_strategy(table_name, strategy, schema)
    return export_to_surrealdb(
        table_name,
        clean,
        schema,
        db_url,
        surreal_url=surreal_url,
        surreal_ns=surreal_ns,
        surreal_db=surreal_db,
        surreal_user=surreal_user,
        surreal_pass=surreal_pass,
        batch_size=batch_size,
    )