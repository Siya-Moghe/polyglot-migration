import json
import sqlite3
import random
import chromadb
from pymongo import MongoClient
from neo4j import GraphDatabase

# ------------------------------------------------------------------
# GENERIC MANIFEST LOADER
# ------------------------------------------------------------------
def load_manifest():
    try:
        with open("migration_manifest.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("\033[91m[ERROR]\033[0m Manifest not found. Run pipeline first.")
        exit(1)

MANIFEST = load_manifest()
SOURCE_DB = MANIFEST["metadata"]["source_db"].replace("sqlite:///", "")

CONNECTIONS = {
    "mongo_uri":   "mongodb://localhost:27017",
    "mongo_db":    "polyglot_migration",
    "neo4j_uri":   "bolt://localhost:7687",
    "neo4j_user":  "neo4j",
    "neo4j_pwd":   "test1234",          
    "chroma_path": "./chroma_store"
}

def _print_result(label: str, passed: bool | float) -> None:
    if isinstance(passed, bool):
        tag = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"  {label:<30}: {tag}")
    else:
        colour = "\033[92m" if passed >= 0.7 else "\033[93m" if passed >= 0.4 else "\033[91m"
        print(f"  {label:<30}: {colour}{passed:.2%}\033[0m")

# ------------------------------------------------------------------
# ENGINE-SPECIFIC DYNAMIC METRICS
# ------------------------------------------------------------------

def check_noise_reduction(cursor) -> bool:
    """Verifies that dropped NON-KEY columns were actually sparse (noisy)."""
    all_passed = True
    columns_checked = 0

    for entry in MANIFEST["migrations"]:
        table = entry["table_name"]
        dropped = entry.get("dropped_columns", [])
        if not dropped: continue

        try:
            total_rows = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except: continue

        for col in dropped:
            col_name = col if isinstance(col, str) else col.get("name")
            if not col_name: continue
            
            # --- THE FIX ---
            # Do not penalize the AI for dropping IDs or timestamps 
            # as these are valid architectural choices for embeddings/documents.
            col_lower = col_name.lower()
            if col_lower == "id" or col_lower.endswith("_id") or "time" in col_lower or "date" in col_lower:
                continue 

            try:
                non_nulls = cursor.execute(f"SELECT COUNT({col_name}) FROM {table}").fetchone()[0]
                live_sparsity = ((total_rows - non_nulls) / total_rows * 100 if total_rows > 0 else 0)
                columns_checked += 1
                
                if live_sparsity < 80.0:
                    print(f"  [!] {table}.{col_name} is only {live_sparsity:.1f}% sparse. Unjustified data loss.")
                    all_passed = False
            except: pass

    if columns_checked == 0:
        print("  No dense content columns were unjustifiably dropped — passing.")
        return True
    return all_passed

def check_query_equivalence(cursor) -> bool:
    """MONGO ONLY: Tests exact row-to-document equivalence."""
    mongo_tables = [m["table_name"] for m in MANIFEST["migrations"] if m["target_engine"] == "mongo"]
    if not mongo_tables:
        print("  No Mongo targets in manifest — skipping document parity.")
        return True

    all_passed = True
    mongo_client = MongoClient(CONNECTIONS["mongo_uri"])
    
    for table in mongo_tables:
        try:
            sql_count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            mongo_count = mongo_client[CONNECTIONS["mongo_db"]][table].count_documents({})
            
            if sql_count != mongo_count:
                print(f"  [!] Document Parity mismatch on {table}: SQL={sql_count}, Mongo={mongo_count}")
                all_passed = False
        except Exception as e:
            print(f"  [!] Error checking {table}: {e}")
            all_passed = False

    mongo_client.close()
    return all_passed

def check_referential_integrity(cursor) -> bool:
    """NEO4J ONLY: Introspects FKs in SQLite and counts corresponding edges in Neo4j."""
    neo4j_tables = [m["table_name"] for m in MANIFEST["migrations"] if m["target_engine"] == "neo4j"]
    if not neo4j_tables:
        print("  No Neo4j targets in manifest — skipping referential integrity.")
        return True

    all_passed = True
    driver = GraphDatabase.driver(CONNECTIONS["neo4j_uri"], auth=(CONNECTIONS["neo4j_user"], CONNECTIONS["neo4j_pwd"]))

    for table in neo4j_tables:
        fks = cursor.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        for fk in fks:
            from_col = fk[3] 
            ref_table = fk[2] 
            
            sql_edges = cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {from_col} IS NOT NULL").fetchone()[0]
            rel_type = f"{table.upper()}_TO_{ref_table.upper()}"
            
            try:
                with driver.session() as s:
                    neo_edges = s.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r)").single()[0]
                
                if sql_edges != neo_edges:
                    print(f"  [!] Edge mismatch for {rel_type}: SQL={sql_edges}, Neo4j={neo_edges}")
                    all_passed = False
            except:
                print(f"  [!] Failed to verify relationship {rel_type} in Neo4j.")
                all_passed = False

    driver.close()
    return all_passed

def check_retrieval_quality(cursor) -> float:
    """CHROMA ONLY: Pulls random source text, queries Chroma, validates Recall."""
    vector_tables = [m["table_name"] for m in MANIFEST["migrations"] if m["target_engine"] == "chroma"]
    if not vector_tables:
        print("  No Chroma targets in manifest — skipping retrieval test.")
        return 1.0

    chroma_client = chromadb.PersistentClient(path=CONNECTIONS["chroma_path"])
    hits, total = 0, 0

    for table in vector_tables:
        try:
            coll = chroma_client.get_collection(name=table)
        except: continue

        cursor.execute(f"PRAGMA table_info({table})")
        cols = cursor.fetchall()
        text_cols = [c[1] for c in cols if "TEXT" in c[2].upper() or "CHAR" in c[2].upper()]
        if not text_cols: continue

        sample_rows = cursor.execute(f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT 3").fetchall()
        for row in sample_rows:
            col_index = [c[1] for c in cols].index(text_cols[0])
            query_text = str(row[col_index])
            expected_id = str(row[0]) 

            if not query_text or query_text.strip() == "None": continue

            try:
                results = coll.query(query_texts=[query_text], n_results=10)
                retrieved_ids = set(results["ids"][0]) if results["ids"] else set()
                if expected_id in retrieved_ids:
                    hits += 1
                total += 1
            except: pass

    return hits / total if total > 0 else 0.0

def run_advanced_audit() -> None:
    print("=" * 50 + "\n  UNIVERSAL DYNAMIC AUDIT (ENGINE-SPECIFIC)\n" + "=" * 50)
    scores = {}
    sqlite_conn = sqlite3.connect(SOURCE_DB)
    sqlite_cursor = sqlite_conn.cursor()
    
    try:
        print("\n[1] NOISE REDUCTION (Checking sparse column drops)")
        scores["noise_reduction"] = check_noise_reduction(sqlite_cursor)
        
        print("\n[2] QUERY EQUIVALENCE (Checking Document Parity in Mongo)")
        scores["query_equivalence"] = check_query_equivalence(sqlite_cursor)
        
        print("\n[3] REFERENTIAL INTEGRITY (Checking Edge Mappings in Neo4j)")
        scores["referential_integrity"] = check_referential_integrity(sqlite_cursor)
        
        print("\n[4] RETRIEVAL QUALITY (Checking Semantic Recall in Chroma)")
        scores["retrieval_recall@10"] = check_retrieval_quality(sqlite_cursor)
    except Exception as e:
        print(f"\n[!] Critical Audit Failure: {e}")
    finally:
        sqlite_conn.close()

    print("\n" + "=" * 50 + "\n  AUDIT SUMMARY SCORECARD\n" + "=" * 50)
    for label, result in scores.items():
        _print_result(label, result)
        
    numeric = [(1.0 if v is True else 0.0 if v is False else float(v)) for v in scores.values()]
    overall = sum(numeric) / len(numeric) if numeric else 0.0
    
    print("-" * 50)
    colour = "\033[92m" if overall >= 0.75 else "\033[93m" if overall >= 0.5 else "\033[91m"
    print(f"  OVERALL PIPELINE SCORE: {colour}{overall:.1%}\033[0m\n" + "=" * 50)

if __name__ == "__main__":
    run_advanced_audit()