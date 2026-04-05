from typing import Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP

from create_complex_db import create_db
from core.introspect import introspect_schema
from core.orchestrator import plan_embeddings
from core.embedder import (
    export_to_chroma,
    export_to_mongo,
    export_to_neo4j,
    export_to_relational,
)
from core.pipeline import run_pipeline

mcp = FastMCP("sql-migration-agent")


@mcp.tool()
def create_sample_db() -> str:
    """
    Create the sample enterprise SQLite database.
    """
    create_db()
    return "Sample database created: enterprise_system.db"


@mcp.tool()
def inspect_schema(connection_string: str = "sqlite:///enterprise_system.db") -> Dict[str, Any]:
    """
    Introspect a SQL database schema and return tables, columns, PKs, and FKs.
    """
    return introspect_schema(connection_string)


@mcp.tool()
def plan_migration(
    connection_string: str = "sqlite:///enterprise_system.db",
    tables: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze schema and generate migration strategies for each table.
    Routes tables to ChromaDB, MongoDB, Neo4j, or Relational SQL using your LangGraph agents.
    """
    schema = introspect_schema(connection_string)

    if tables:
        schema = {t: v for t, v in schema.items() if t in tables}

    strategies = plan_embeddings(schema)

    return {
        "connection_string": connection_string,
        "tables": list(schema.keys()),
        "strategies": strategies,
    }


@mcp.tool()
def export_table(
    table_name: str,
    target_db: str,
    connection_string: str = "sqlite:///enterprise_system.db"
) -> Dict[str, Any]:
    """
    Export a single table to its target database using the generated strategy.
    target_db must be one of: chroma, mongo, neo4j, relational
    """
    schema = introspect_schema(connection_string)

    if table_name not in schema:
        return {"error": f"Table '{table_name}' not found"}

    target_db = target_db.lower()
    valid_targets = {"chroma", "mongo", "neo4j", "relational"}

    if target_db not in valid_targets:
        return {"error": "target_db must be one of: chroma, mongo, neo4j, relational"}

    strategies = plan_embeddings({table_name: schema[table_name]})
    strategy = strategies[table_name]

    planned_target = strategy.get("target_db", "").lower()
    override_warning = None

    if planned_target != target_db:
        override_warning = (
            f"Planner recommended '{planned_target}' for table '{table_name}', "
            f"but export was forced to '{target_db}'."
        )

    # keep original recommendation visible
    strategy["planned_target_db"] = planned_target
    strategy["target_db"] = target_db

    # Safe export dispatch
    if target_db == "chroma":
        count = export_to_chroma(table_name, strategy, schema, connection_string)
    elif target_db == "mongo":
        count = export_to_mongo(table_name, strategy, schema, connection_string)
    elif target_db == "neo4j":
        count = export_to_neo4j(table_name, strategy, schema, connection_string)
    elif target_db == "relational":
        count = export_to_relational(table_name, strategy, schema, connection_string)
    else:
        return {"error": f"Unexpected target_db: {target_db}"}

    result = {
        "table_name": table_name,
        "target_db": target_db,
        "records_exported": count,
        "strategy": strategy,
    }

    if override_warning:
        result["warning"] = override_warning

    return result


@mcp.tool()
def run_full_pipeline(
    connection_string: str = "sqlite:///enterprise_system.db",
    tables: Optional[List[str]] = None
) -> str:
    """
    Run the complete SQL -> AI-routed target DB migration pipeline.
    """
    run_pipeline(connection_string, tables)
    return "Pipeline execution completed successfully."


if __name__ == "__main__":
    print("Starting MCP server...")
    mcp.run()