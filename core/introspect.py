"""
introspect.py -  Stage 1: Schema Introspection
Connects to any SQLAlchemy-supported DB, reflects schema, fetches rows.
"""

from __future__ import annotations
from typing import Any
from sqlalchemy import create_engine, inspect, text


def introspect_schema(connection_string: str) -> dict[str, Any]:
    """
    Reflect every table in the database.

    Returns
    -------
    {
      "table_name": {
        "columns":      [{"name", "type", "nullable"}],
        "primary_keys": [...],
        "foreign_keys": [{"columns", "referred_table", "referred_columns"}]
      }
    }
    """
    engine    = create_engine(connection_string)
    inspector = inspect(engine)
    schema: dict[str, Any] = {}

    for table_name in inspector.get_table_names():
        columns = [
            {"name": c["name"], "type": str(c["type"]), "nullable": c.get("nullable", True)}
            for c in inspector.get_columns(table_name)
        ]
        pk  = inspector.get_pk_constraint(table_name)
        fks = inspector.get_foreign_keys(table_name)

        schema[table_name] = {
            "columns":      columns,
            "primary_keys": pk.get("constrained_columns", []),
            "foreign_keys": [
                {
                    "columns":          fk["constrained_columns"],
                    "referred_table":   fk["referred_table"],
                    "referred_columns": fk["referred_columns"],
                }
                for fk in fks
            ],
        }

    engine.dispose()
    return schema

def fetch_rows(db_url: str, table_name: str, limit: int = 100):
    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT {limit}"))
        return [dict(row._mapping) for row in result]


def fetch_lookup(db_url: str, table_name: str, key_column: str, key_value):
    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT * FROM {table_name} WHERE {key_column} = :val LIMIT 1"),
            {"val": key_value}
        ).fetchone()
        return dict(result._mapping) if result else None

if __name__ == "__main__":
    import json
    conn_str = input("Connection string (e.g. sqlite:///sample_store.db): ").strip()
    print(json.dumps(introspect_schema(conn_str), indent=2))
