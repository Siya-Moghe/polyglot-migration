import sqlite3
import urllib.parse
import chromadb
from pymongo import MongoClient
from neo4j import GraphDatabase
from sqlalchemy import create_engine, text

AUDIT_CONFIG = {
    "sqlite_path": "enterprise_system.db",
    "valid_tables": {"departments", "support_tickets", "employees", "knowledge_base", "projects", "employee_projects"},
    "noise_reduction": {
        "table": "knowledge_base",
        "dropped_columns": [],
        "threshold": 80.0,
    },
    "query_equivalence": {
        "table": "projects",
        "filter_col": "project_name",
        "filter_val": "Cloud Migration 2026",
        "tolerance": 0,
    },
    "referential_integrity": {
        "source_table": "employee_projects",
        "fk_column": "emp_id",
        "rel_type": "EMPLOYEE_PROJECTS_TO_EMPLOYEES",
    },
    "retrieval_ground_truth": {
        "How do I connect to the corporate VPN?": ["1"],
        "What are the engineering guidelines?": ["2"],
    },
    "connections": {
        "mongo_uri":   "mongodb://localhost:27017",
        "mongo_db":    "polyglot_migration",
        "neo4j_uri":   "bolt://localhost:7687",
        "neo4j_user":  "neo4j",
        "neo4j_pwd":   "test1234",          
        "chroma_path": "./chroma_store",
        "chroma_coll": "knowledge_base",
        "mysql_pwd":   "your pwd"
    },
}

def _safe_table(name: str) -> str:
    if name not in AUDIT_CONFIG["valid_tables"]:
        raise ValueError(f"Table '{name}' is not in the whitelist.")
    return name

def _print_result(label: str, passed: bool | float) -> None:
    if isinstance(passed, bool):
        tag = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"  {label:<30}: {tag}")
    else:
        colour = "\033[92m" if passed >= 0.7 else "\033[93m" if passed >= 0.4 else "\033[91m"
        print(f"  {label:<30}: {colour}{passed:.2%}\033[0m")

def check_noise_reduction(cursor) -> bool:
    cfg = AUDIT_CONFIG["noise_reduction"]
    table   = _safe_table(cfg["table"])
    entries = cfg["dropped_columns"]
    threshold = cfg["threshold"]

    if not entries:
        print("  No columns dropped — skipping.")
        return True

    total_rows = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    violations = []

    for entry in entries:
        col  = entry["name"]
        pre  = entry.get("pre_migration_sparsity")

        if pre is not None and pre < threshold:
            violations.append(f"'{col}' had pre-migration sparsity {pre:.1f}%")
            continue

        try:
            non_nulls = cursor.execute(f"SELECT COUNT({col}) FROM {table}").fetchone()[0]
            live_sparsity = ((total_rows - non_nulls) / total_rows * 100 if total_rows > 0 else 0)
            print(f"  '{col}' live sparsity: {live_sparsity:.1f}%")
            if live_sparsity < threshold:
                violations.append(f"'{col}' is only {live_sparsity:.1f}% sparse")
        except sqlite3.OperationalError:
            print(f"  '{col}' not found in source.")

    if violations:
        for v in violations:
            print(f"  [!] {v}")
        return False

    return True

def check_query_equivalence(cursor) -> bool:
    cfg       = AUDIT_CONFIG["query_equivalence"]
    table     = _safe_table(cfg["table"])
    col       = cfg["filter_col"]
    val       = cfg["filter_val"]
    tolerance = cfg.get("tolerance", 0)

    sql_count = cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", (val,)).fetchone()[0]

    conn = MongoClient(AUDIT_CONFIG["connections"]["mongo_uri"])
    try:
        db_name = AUDIT_CONFIG["connections"]["mongo_db"]
        mongo_count = conn[db_name][table].count_documents({col: val})
    finally:
        conn.close()

    delta = abs(sql_count - mongo_count)
    print(f"  SQLite rows : {sql_count}")
    print(f"  Mongo docs  : {mongo_count}")
    return delta <= tolerance

def check_referential_integrity(cursor) -> bool:
    cfg      = AUDIT_CONFIG["referential_integrity"]
    src      = _safe_table(cfg["source_table"])
    fk_col   = cfg["fk_column"]
    rel_type = cfg["rel_type"]

    sql_edges = cursor.execute(f"SELECT COUNT(*) FROM {src} WHERE {fk_col} IS NOT NULL").fetchone()[0]

    c = AUDIT_CONFIG["connections"]
    driver = GraphDatabase.driver(c["neo4j_uri"], auth=(c["neo4j_user"], c["neo4j_pwd"]))
    try:
        with driver.session() as session:
            neo_edges = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r)").single()[0]
    finally:
        driver.close()

    print(f"  SQLite FK rows  : {sql_edges}")
    print(f"  Neo4j edges     : {neo_edges}")

    return sql_edges == neo_edges

def check_retrieval_quality() -> float:
    ground_truth = AUDIT_CONFIG["retrieval_ground_truth"]
    c = AUDIT_CONFIG["connections"]

    chroma_client = chromadb.PersistentClient(path=c["chroma_path"])
    collection    = chroma_client.get_collection(name=c["chroma_coll"])

    hits  = 0
    total = 0

    for query, relevant_ids in ground_truth.items():
        results    = collection.query(query_texts=[query], n_results=10)
        retrieved  = set(results["ids"][0]) if results["ids"] else set()
        found      = len(retrieved & set(relevant_ids))
        hits      += found
        total     += len(relevant_ids)
        print(f"  Query: '{query[:50]}...'  ->  {found}/{len(relevant_ids)} relevant found")

    return hits / total if total > 0 else 0.0

def check_mysql_infrastructure() -> bool:
    c = AUDIT_CONFIG["connections"]
    encoded_pwd = urllib.parse.quote_plus(c["mysql_pwd"])
    engine_url = f"mysql+pymysql://root:{encoded_pwd}@localhost:3306/polyglot_target"
    
    try:
        engine = create_engine(engine_url)
        with engine.connect() as conn:
            result = conn.execute(text("SHOW TABLES;")).fetchall()
            print(f"  MySQL connection active. Tables found: {len(result)}")
            return True
    except Exception as e:
        print(f"  MySQL Connection Error: {e}")
        return False

def run_advanced_audit() -> None:
    print("=" * 50)
    print("  POST-MIGRATION ACADEMIC AUDIT & VALIDATION")
    print("=" * 50 + "\n")

    scores: dict[str, bool | float] = {}

    sqlite_conn   = sqlite3.connect(AUDIT_CONFIG["sqlite_path"])
    sqlite_cursor = sqlite_conn.cursor()

    try:
        print("[1] NOISE REDUCTION")
        try:
            scores["noise_reduction"] = check_noise_reduction(sqlite_cursor)
        except Exception as e:
            print(f"  ERROR: {e}")
            scores["noise_reduction"] = False

        print("\n[2] QUERY EQUIVALENCE")
        try:
            scores["query_equivalence"] = check_query_equivalence(sqlite_cursor)
        except Exception as e:
            print(f"  ERROR: {e}")
            scores["query_equivalence"] = False

        print("\n[3] REFERENTIAL INTEGRITY")
        try:
            scores["referential_integrity"] = check_referential_integrity(sqlite_cursor)
        except Exception as e:
            print(f"  ERROR: {e}")
            scores["referential_integrity"] = False

        print("\n[4] RETRIEVAL QUALITY  (Recall@10)")
        try:
            scores["retrieval_recall@10"] = check_retrieval_quality()
        except Exception as e:
            print(f"  ERROR: {e}")
            scores["retrieval_recall@10"] = 0.0

        print("\n[5] MYSQL INFRASTRUCTURE")
        try:
            scores["mysql_infrastructure"] = check_mysql_infrastructure()
        except Exception as e:
            print(f"  ERROR: {e}")
            scores["mysql_infrastructure"] = False

    finally:
        sqlite_conn.close()

    print("\n" + "=" * 50)
    print("  AUDIT SUMMARY SCORECARD")
    print("=" * 50)
    for label, result in scores.items():
        _print_result(label, result)

    numeric = [
        (1.0 if v is True else 0.0 if v is False else float(v))
        for v in scores.values()
    ]
    overall = sum(numeric) / len(numeric) if numeric else 0.0

    print("-" * 50)
    colour = "\033[92m" if overall >= 0.75 else "\033[93m" if overall >= 0.5 else "\033[91m"
    print(f"  {'OVERALL PIPELINE SCORE':<30}: {colour}{overall:.1%}\033[0m")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_advanced_audit()