"""
pipeline.py  –  CLI entry point for the SurrealDB migration pipeline.

Usage
-----
    python -m core.pipeline --db sqlite:///enterprise_system.db
    python -m core.pipeline --db sqlite:///myapp.db --tables users orders
    python -m core.pipeline --db sqlite:///myapp.db \\
        --surreal-url ws://localhost:8000/rpc \\
        --surreal-ns  myproject \\
        --surreal-db  production \\
        --surreal-user root --surreal-pass secret

All connection details are flags – nothing is hardcoded.
"""

from __future__ import annotations

import argparse
import time

from core.introspect import introspect_schema, fetch_rows
from core.orchestrator import plan_migration, save_migration_manifest
from core.embedder_surrealdb import embed_and_store


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    db_url: str,
    *,
    tables: list[str] | None = None,
    surreal_url: str   = "ws://localhost:8000/rpc",
    surreal_ns: str    = "migration",
    surreal_db: str    = "target",
    surreal_user: str  = "root",
    surreal_pass: str  = "root",
    batch_size: int    = 500,
    human_context: str = "",
) -> None:
    total_start = time.time()

    # ------------------------------------------------------------------
    # STAGE 1 – Schema introspection
    # ------------------------------------------------------------------
    print("\nSTAGE 1 — Schema Introspection")
    print("-" * 40)

    schema = introspect_schema(db_url)

    if tables:
        schema = {t: v for t, v in schema.items() if t in tables}

    for tname, tinfo in schema.items():
        fk_list = [f["referred_table"] for f in tinfo["foreign_keys"]]
        print(f"  {tname:25s} cols={len(tinfo['columns'])} fk->{fk_list}")

    print("\n  Fetching sample rows for LLM prompt enrichment ...")
    sample_rows_map: dict[str, list] = {}
    for tname in schema:
        try:
            sample_rows_map[tname] = fetch_rows(db_url, tname, limit=2)
        except Exception as exc:
            print(f"  [warn] Could not fetch sample rows for '{tname}': {exc}")
            sample_rows_map[tname] = []

    # ------------------------------------------------------------------
    # STAGE 2 – LangGraph orchestrator
    # ------------------------------------------------------------------
    print("\nSTAGE 2 — LangGraph SurrealDB Orchestrator")
    print("-" * 40)

    strategies = plan_migration(schema, sample_rows_map=sample_rows_map, human_context=human_context)
    save_migration_manifest(strategies, db_url)

    # Pretty-print strategy summary
    print("\nSTRATEGY SUMMARY")
    print("-" * 40)
    fallback_count = 0

    for tname, strat in strategies.items():
        fallback = strat.get("fallback_used", False)
        if fallback:
            fallback_count += 1

        features = []
        if strat.get("use_vector"):  features.append("VECTOR")
        if strat.get("use_graph"):   features.append("GRAPH")
        if strat.get("use_nested"):  features.append("NESTED")
        if strat.get("index_cols"):  features.append(f"INDEXES({','.join(strat['index_cols'])})")

        print(f"\nTABLE: {tname.upper()}")
        print(f"  Features   : {' | '.join(features) or 'plain document'}")
        print(f"  Reasoning  : {strat.get('reasoning', 'N/A')}")
        print(f"  Top fields : {strat.get('top_level_fields', [])}")
        print(f"  Nested     : {strat.get('nested_fields', {})}")
        print(f"  Relations  : {[r['label'] for r in strat.get('relations', [])]}")
        print(f"  Fallback   : {'YES' if fallback else 'NO'}")

    print(f"\n  Fallback rate: {fallback_count}/{len(strategies)} tables")

    # ------------------------------------------------------------------
    # STAGE 3 – Export to SurrealDB
    # ------------------------------------------------------------------
    print(f"\nSTAGE 3 — Export to SurrealDB ({surreal_url})")
    print("-" * 40)

    counts: dict[str, int] = {}
    for tname, strat in strategies.items():
        try:
            counts[tname] = embed_and_store(
                tname,
                strat,
                schema,
                db_url,
                surreal_url=surreal_url,
                surreal_ns=surreal_ns,
                surreal_db=surreal_db,
                surreal_user=surreal_user,
                surreal_pass=surreal_pass,
                batch_size=batch_size,
            )
        except Exception as exc:
            print(f"  ERROR exporting '{tname}': {exc}")
            counts[tname] = 0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\nPIPELINE COMPLETE")
    print("-" * 40)
    for tname, n in counts.items():
        feats = []
        s = strategies[tname]
        if s.get("use_vector"):  feats.append("vec")
        if s.get("use_graph"):   feats.append("graph")
        if s.get("use_nested"):  feats.append("nested")
        feat_str = f"[{','.join(feats)}]" if feats else ""
        print(f"  {tname:25s} {n:>6} records  {feat_str}")

    print(f"\n  Target : SurrealDB @ {surreal_url}  ns={surreal_ns}  db={surreal_db}")
    print(f"  Time   : {time.time() - total_start:.1f}s\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate a SQLite (or any SQLAlchemy-supported) database to SurrealDB."
    )

    # Source
    parser.add_argument(
        "--db",
        required=True,
        help="SQLAlchemy connection string for the SOURCE database.  "
             "Example: sqlite:///enterprise_system.db  "
             "or  postgresql://user:pass@host/dbname",
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        default=None,
        help="Optional list of specific tables to migrate.  Defaults to ALL tables.",
    )

    # SurrealDB target
    parser.add_argument("--surreal-url",  default="ws://localhost:8000/rpc",
                        help="SurrealDB WebSocket URL  (default: ws://localhost:8000/rpc)")
    parser.add_argument("--surreal-ns",   default="migration",
                        help="SurrealDB namespace  (default: migration)")
    parser.add_argument("--surreal-db",   default="target",
                        help="SurrealDB database name  (default: target)")
    parser.add_argument("--surreal-user", default="root",
                        help="SurrealDB username  (default: root)")
    parser.add_argument("--surreal-pass", default="root",
                        help="SurrealDB password  (default: root)")

    # Tuning
    parser.add_argument("--batch-size",    type=int, default=500,
                        help="Rows per INSERT batch  (default: 500)")
    parser.add_argument("--human-context", default="",
                        help="Optional free-text expert advice for the LLM orchestrator.")

    args = parser.parse_args()

    run_pipeline(
        db_url        = args.db,
        tables        = args.tables,
        surreal_url   = args.surreal_url,
        surreal_ns    = args.surreal_ns,
        surreal_db    = args.surreal_db,
        surreal_user  = args.surreal_user,
        surreal_pass  = args.surreal_pass,
        batch_size    = args.batch_size,
        human_context = args.human_context,
    )