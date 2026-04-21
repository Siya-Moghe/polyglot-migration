import hashlib 
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from pymongo import MongoClient
from neo4j import GraphDatabase
from sqlalchemy import create_engine, MetaData, Table, Column, text
from sqlalchemy.types import Integer, Float, String, Text
from urllib.parse import quote_plus

from core.introspect import fetch_rows, fetch_lookup, fetch_rows_in_batches


MYSQL_PASSWORD = quote_plus("Siya123@root")
MYSQL_TARGET_URL = f"mysql+pymysql://root:{MYSQL_PASSWORD}@localhost:3306/polyglot_target"


# -------------------------
# Shared model / clients
# -------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------
# Helper: format text safely
# -------------------------
def safe_format(template: str, row: Dict[str, Any]) -> str:
    class SafeDict(dict):
        def __missing__(self, key):
            return f"{{{key}}}"
    return template.format_map(SafeDict(row))


def normalize_join_related(join_related_raw):
    """
    Normalize join_related into a clean list of table names.
    Supports:
      ["departments"]
      [{"tableName": "departments"}]
      [{"table": "departments"}]
    """
    clean = []

    if not isinstance(join_related_raw, list):
        return clean

    for item in join_related_raw:
        if isinstance(item, str):
            clean.append(item)
        elif isinstance(item, dict):
            if "tableName" in item and isinstance(item["tableName"], str):
                clean.append(item["tableName"])
            elif "table" in item and isinstance(item["table"], str):
                clean.append(item["table"])

    return clean

def mask_pii(row: Dict[str, Any]) -> Dict[str, Any]:
    pii_keywords = ["email", "ssn", "password", "phone", "credit_card", "social", "address"]
    masked_row = row.copy()
    
    for key, value in masked_row.items():
        if any(pii in key.lower() for pii in pii_keywords) and value:
            masked_row[key] = f"REDACTED-{hashlib.sha256(str(value).encode()).hexdigest()[:8]}"
            
    return masked_row

def normalize_strategy(
    table_name: str,
    strategy: Dict[str, Any],
    schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Normalize LLM strategy into a guaranteed-safe export format.
    Prevent malformed agent outputs from breaking ETL.
    """
    info = schema.get(table_name, {})
    all_cols = [c["name"] for c in info.get("columns", [])]
    fk_tables = [fk["referred_table"] for fk in info.get("foreign_keys", [])]

    target = (strategy.get("target_db") or "").lower()

    # -------------------------
    # CHROMA
    # -------------------------
    if target == "chroma":
        template = strategy.get("template")
        if not template:
            template = " | ".join(f"{col}: {{{col}}}" for col in all_cols)

        return {
            "target_db": "chroma",
            "template": template,
            "used_columns": [c for c in strategy.get("used_columns", all_cols) if c in all_cols],
            "skipped_columns": [c for c in strategy.get("skipped_columns", []) if c in all_cols],
            "join_related": [j for j in normalize_join_related(strategy.get("join_related", [])) if j in fk_tables],
            "reasoning": strategy.get("reasoning", "Normalized safe Chroma strategy.")
        }

    # -------------------------
    # MONGO
    # -------------------------
    elif target == "mongo":
        fields = strategy.get("fields") or strategy.get("used_columns") or all_cols
        fields = [f for f in fields if isinstance(f, str) and f in all_cols]

        nested_fields = strategy.get("nested_fields", {})
        if not isinstance(nested_fields, dict):
            nested_fields = {}

        safe_nested = {}
        for parent, children in nested_fields.items():
            if isinstance(parent, str) and isinstance(children, list):
                safe_nested[parent] = [c for c in children if isinstance(c, str) and c in all_cols]

        clean_join_related = [j for j in normalize_join_related(strategy.get("join_related", [])) if j in fk_tables]

        safe_join_keys = {}
        raw_join_keys = strategy.get("join_keys", {})
        if isinstance(raw_join_keys, dict):
            for tbl, mapping in raw_join_keys.items():
                if (
                    isinstance(tbl, str)
                    and isinstance(mapping, dict)
                    and "local_key" in mapping
                    and "foreign_key" in mapping
                    and mapping["local_key"] in all_cols
                ):
                    safe_join_keys[tbl] = {
                        "local_key": mapping["local_key"],
                        "foreign_key": mapping["foreign_key"]
                    }

        return {
            "target_db": "mongo",
            "fields": fields if fields else all_cols,
            "nested_fields": safe_nested,
            "join_related": clean_join_related,
            "join_keys": safe_join_keys,
            "reasoning": strategy.get("reasoning", "Normalized safe Mongo strategy.")
        }

    # -------------------------
    # NEO4J
    # -------------------------
    elif target == "neo4j":
        return {
            "target_db": "neo4j",
            "used_columns": [c for c in strategy.get("used_columns", all_cols) if c in all_cols],
            "skipped_columns": [c for c in strategy.get("skipped_columns", []) if c in all_cols],
            "join_related": [j for j in normalize_join_related(strategy.get("join_related", [])) if j in fk_tables],
            "reasoning": strategy.get("reasoning", "Normalized safe Neo4j strategy.")
        }

    # -------------------------
    # RELATIONAL
    # -------------------------
    elif target == "relational":
        return {
            "target_db": "relational",
            "preserve_as": "table",
            "used_columns": [c for c in strategy.get("used_columns", all_cols) if c in all_cols],
            "skipped_columns": [c for c in strategy.get("skipped_columns", []) if c in all_cols],
            "join_related": [],
            "primary_use": strategy.get("primary_use", "transactional"),
            "reasoning": strategy.get("reasoning", "Normalized safe relational strategy.")
        }

    # -------------------------
    # Unknown target fallback
    # -------------------------
    fallback_template = " | ".join(f"{col}: {{{col}}}" for col in all_cols)
    return {
        "target_db": "chroma",
        "template": fallback_template,
        "used_columns": all_cols,
        "skipped_columns": [],
        "join_related": [],
        "reasoning": "Emergency fallback: unknown target, preserved as searchable Chroma text."
    }


# -------------------------
# CHROMA EXPORT
# -------------------------
def export_to_chroma(
    table_name: str,
    strategy: Dict[str, Any],
    schema: Dict[str, Any],
    db_url: str
) -> int:
    print(f"\n[Chroma] Exporting table: {table_name} (Using Batch Streaming & PII Masking)")

    client = chromadb.PersistentClient(path="./chroma_store")
    collection = client.get_or_create_collection(name=table_name)
    template = strategy.get("template", "{id}")
    
    total_inserted = 0

    # Stream in safe chunks
    for batch in fetch_rows_in_batches(db_url, table_name, batch_size=5000):
        documents = []
        ids = []
        metadatas = []

        for i, row in enumerate(batch):
            safe_row = mask_pii(row) # Mask PII!
            
            doc = safe_format(template, safe_row)
            documents.append(doc)
            # Use total_inserted + i to ensure unique IDs across batches
            ids.append(str(safe_row.get("id", total_inserted + i)))
            metadatas.append({k: str(v) for k, v in safe_row.items()})

        if documents:
            embeddings = embed_model.encode(documents).tolist()
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            total_inserted += len(documents)
            print(f"  ... embedded & inserted batch of {len(documents)} vectors.")

    print(f"  [+] Completed! Total {total_inserted} records inserted into Chroma collection '{table_name}'")
    return total_inserted


# -------------------------
# MONGO EXPORT
# -------------------------
def build_mongo_document(
    row: Dict[str, Any],
    strategy: Dict[str, Any],
    db_url: str
) -> Dict[str, Any]:
    doc = {}

    fields = strategy.get("fields", list(row.keys()))
    for f in fields:
        if f in row:
            doc[f] = row[f]

    nested = strategy.get("nested_fields", {})
    for parent_key, child_fields in nested.items():
        doc[parent_key] = {}
        for cf in child_fields:
            if cf in row:
                doc[parent_key][cf] = row[cf]

    join_related = strategy.get("join_related", [])
    join_keys = strategy.get("join_keys", {})

    for related_table in join_related:
        if related_table in join_keys:
            local_key = join_keys[related_table]["local_key"]
            foreign_key = join_keys[related_table]["foreign_key"]

            if local_key in row:
                related = fetch_lookup(
                    db_url=db_url,
                    table_name=related_table,
                    key_column=foreign_key,
                    key_value=row[local_key]
                )
                if related:
                    doc[related_table] = related

    return doc


def export_to_mongo(
    table_name: str,
    strategy: Dict[str, Any],
    schema: Dict[str, Any],
    db_url: str,
    mongo_uri: str = "mongodb://localhost:27017",
    mongo_db_name: str = "polyglot_migration"
) -> int:
    print(f"\n[Mongo] Exporting table: {table_name} (Using Batch Streaming & PII Masking)")

    client = MongoClient(mongo_uri)
    db = client[mongo_db_name]
    collection = db[table_name]
    
    # Wipe old data
    collection.delete_many({})
    
    total_inserted = 0

    # Stream the data in safe chunks!
    for batch in fetch_rows_in_batches(db_url, table_name, batch_size=5000):
        docs = []
        for row in batch:
            # Mask PII before building the document!
            safe_row = mask_pii(row)
            doc = build_mongo_document(safe_row, strategy, db_url)
            docs.append(doc)
        
        if docs:
            collection.insert_many(docs)
            total_inserted += len(docs)
            print(f"  ... inserted batch of {len(docs)} documents.")

    print(f"  [+] Completed! Total {total_inserted} documents inserted into MongoDB collection '{table_name}'")
    return total_inserted


# -------------------------
# NEO4J EXPORT
# -------------------------
def export_to_neo4j(
    table_name: str,
    strategy: Dict[str, Any],
    schema: Dict[str, Any],
    db_url: str,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "test1234"
) -> int:
    print(f"\n[Neo4j] Exporting table: {table_name} (Using Batch Streaming & PII Masking)")

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    label = table_name.capitalize()
    fk_info = schema.get(table_name, {}).get("foreign_keys", [])

    def _safe_props(row: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in row.items() if v is not None}

    total_inserted = 0

    with driver.session() as session:
        # Stream in safe chunks
        for batch in fetch_rows_in_batches(db_url, table_name, batch_size=5000):
            batch_count = 0
            
            # Create main nodes
            for i, row in enumerate(batch):
                safe_row = mask_pii(row) # Mask PII!
                node_id = safe_row.get("id", total_inserted + i)
                props = _safe_props(safe_row)

                session.run(
                    f"MERGE (n:{label} {{node_id: $node_id}}) SET n += $props",
                    node_id=node_id,
                    props=props
                )
                batch_count += 1

            # Create FK relationships for this batch
            for fk in fk_info:
                constrained = fk.get("constrained_columns") or fk.get("columns") or []
                if not constrained:
                    continue

                local_col = constrained[0]
                ref_table = fk["referred_table"]
                ref_label = ref_table.capitalize()
                rel_type = f"{table_name.upper()}_TO_{ref_table.upper()}"

                for i, row in enumerate(batch):
                    safe_row = mask_pii(row)
                    source_id = safe_row.get("id", total_inserted + i)
                    
                    if local_col not in safe_row or safe_row[local_col] is None:
                        continue

                    session.run(
                        f"""
                        MATCH (a:{label} {{node_id: $source_id}})
                        MERGE (b:{ref_label} {{node_id: $target_id}})
                        MERGE (a)-[:{rel_type}]->(b)
                        """,
                        source_id=source_id,
                        target_id=safe_row[local_col]
                    )
            
            total_inserted += batch_count
            print(f"  ... inserted batch of {batch_count} nodes & edges.")

    driver.close()
    print(f"  [+] Completed! Total {total_inserted} nodes inserted into Neo4j label '{label}'")
    return total_inserted


# -------------------------
# RELATIONAL EXPORT (MYSQL)
# -------------------------
def _map_sqlalchemy_type(col_type: str):
    t = str(col_type).upper()

    if "INT" in t:
        return Integer
    elif "FLOAT" in t or "REAL" in t or "DOUBLE" in t or "NUMERIC" in t:
        return Float
    elif "TEXT" in t or "CHAR" in t or "VARCHAR" in t:
        return Text
    else:
        return String(255)



from sqlalchemy import create_engine, text

def ensure_mysql_database(mysql_target_url: str):
    """
    Ensure the target MySQL database exists.
    """
    # Strip database name from URL
    base_url = mysql_target_url.rsplit("/", 1)[0]

    # Extract DB name
    db_name = mysql_target_url.rsplit("/", 1)[-1]

    engine = create_engine(base_url)

    with engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
        print(f"[MySQL] Ensured database '{db_name}' exists.")


def export_to_relational(
    table_name: str,
    strategy: Dict[str, Any],
    schema: Dict[str, Any],
    db_url: str,
    mysql_target_url: str = MYSQL_TARGET_URL
) -> int:
    print(f"\n[Relational] Exporting table: {table_name} (Using Batch Streaming & PII Masking)")
    ensure_mysql_database(mysql_target_url)

    if table_name not in schema:
        raise ValueError(f"Schema info missing for table '{table_name}'")

    table_info = schema[table_name]
    source_columns = table_info["columns"]
    primary_keys = set(table_info.get("primary_keys", []))

    target_engine = create_engine(mysql_target_url)
    metadata = MetaData()

    columns = []
    for col in source_columns:
        col_name = col["name"]
        col_type = _map_sqlalchemy_type(col["type"])
        is_pk = col_name in primary_keys
        columns.append(Column(col_name, col_type, primary_key=is_pk))

    target_table = Table(table_name, metadata, *columns)
    metadata.create_all(target_engine)

    total_inserted = 0

    with target_engine.begin() as conn:
        conn.execute(text(f"DELETE FROM `{table_name}`")) # Wipe old data
        
        # Stream in safe chunks
        for batch in fetch_rows_in_batches(db_url, table_name, batch_size=5000):
            masked_batch = [mask_pii(row) for row in batch] # Mask PII!
            
            if masked_batch:
                conn.execute(target_table.insert(), masked_batch)
                total_inserted += len(masked_batch)
                print(f"  ... inserted batch of {len(masked_batch)} rows into SQL.")

    print(f"  [+] Completed! Total {total_inserted} rows inserted into Relational table '{table_name}'")
    return total_inserted


# -------------------------
# ROUTER
# -------------------------
def embed_and_store(
    table_name: str,
    strategy: Dict[str, Any],
    schema: Dict[str, Any],
    db_url: str
) -> int:
    strategy = normalize_strategy(table_name, strategy, schema)
    target = strategy.get("target_db", "").lower()

    if target == "chroma":
        return export_to_chroma(table_name, strategy, schema, db_url)
    elif target == "mongo":
        return export_to_mongo(table_name, strategy, schema, db_url)
    elif target == "neo4j":
        return export_to_neo4j(table_name, strategy, schema, db_url)
    elif target == "relational":
        return export_to_relational(table_name, strategy, schema, db_url)
    else:
        print(f"  Unknown target_db for {table_name}: {target}")
        return 0