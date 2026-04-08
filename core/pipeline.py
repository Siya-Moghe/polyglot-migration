import argparse
import time
from core.introspect import introspect_schema, fetch_rows
from core.orchestrator import plan_embeddings
from core.embedder import embed_and_store


def run_pipeline(connection_string: str, tables: list[str] | None = None) -> None:
    total_start = time.time()

    print("\nSTAGE 1 - Schema Introspection")
    print("-" * 40)
    schema = introspect_schema(connection_string)

    if tables:
        schema = {t: v for t, v in schema.items() if t in tables}

    for tname, tinfo in schema.items():
        fk_list = [f["referred_table"] for f in tinfo["foreign_keys"]]
        print(f"  {tname:20s} cols={len(tinfo['columns'])} fk->{fk_list}")

    print("\n  Fetching sample rows for prompt enrichment...")
    sample_rows_map: dict[str, list] = {}
    for tname in schema:
        try:
            sample_rows_map[tname] = fetch_rows(connection_string, tname, limit=2)
        except Exception as e:
            print(f"  [warn] Could not fetch sample rows for '{tname}': {e}")
            sample_rows_map[tname] = []

    print("\nSTAGE 2 - LangGraph Multi-Agent Orchestrator")
    print("-" * 40)
    strategies = plan_embeddings(schema, sample_rows_map=sample_rows_map)
    from core.orchestrator import save_migration_manifest 
    save_migration_manifest(strategies, connection_string)
    
    print("\nAGENT REASONING & STRATEGY SUMMARY")
    print("-" * 40)
    fallback_count = 0
    retry_total = 0

    for tname, strat in strategies.items():
        db = strat.get("target_db", "UNKNOWN").upper()
        fallback = strat.get("fallback_used", False)
        if fallback:
            fallback_count += 1

        print(f"\nTABLE: {tname.upper()} -> Routed to [{db}]")
        print(f"  Routing Logic: {strat.get('routing_reason', strat.get('reasoning', 'N/A'))}")
        print(f"  Columns Kept : {strat.get('used_columns', strat.get('fields', []))}")
        print(f"  Cols Dropped : {strat.get('skipped_columns', [])}")
        print(f"  Tables Joined: {strat.get('join_related', [])}")
        print(f"  Column Logic : {strat.get('reasoning', 'N/A')}")
        print(f"  Fallback Used: {'YES' if fallback else 'NO '}")

        if db == "CHROMA":
            template_preview = str(strat.get("template", ""))[:120].replace("\n", "")
            print(f"  Structure    : {template_preview}...")
        elif db == "MONGO":
            print(f"  Structure    : fields={strat.get('fields', [])}, nested={strat.get('nested_fields', {})}")
        elif db == "NEO4J":
            print(f"  Structure    : {str(strat.get('template', 'graph node + FK relationship mapping'))[:120]}")
        elif db == "RELATIONAL":
            print(f"  Structure    : preserved SQL table ({strat.get('primary_use', 'transactional')})")

    print(f"\n  Fallback rate: {fallback_count}/{len(strategies)} tables")

    print("\nSTAGE 3 - Export to Target Databases")
    print("-" * 40)

    counts = {}
    for tname, strat in strategies.items():
        try:
            counts[tname] = embed_and_store(tname, strat, schema, connection_string)
        except Exception as e:
            print(f"  ERROR exporting {tname}: {e}")
            counts[tname] = 0

    print("\nPIPELINE COMPLETE")
    print("-" * 40)
    for tname, n in counts.items():
        print(f"  {tname:20s} {n} records exported")

    print("\nFINAL STORAGE DESTINATIONS")
    print("-" * 40)
    print("  MongoDB  : polyglot_migration")
    print("  Neo4j    : bolt://localhost:7687")
    print("  Chroma   : ./chroma_store")
    print("  MySQL    : polyglot_target")

    print(f"\nTotal time: {time.time() - total_start:.1f}s\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="sqlite:///enterprise_system.db")
    args = parser.parse_args()
    run_pipeline(args.db)