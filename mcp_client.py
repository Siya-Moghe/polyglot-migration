import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_FILE = "mcp_server.py"   # change if your filename is different


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=[SERVER_FILE],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("\nConnected to MCP server!")

            # -------------------------
            # 1. List available tools
            # -------------------------
            tools = await session.list_tools()
            print("\nAvailable Tools:")
            for tool in tools.tools:
                print(f" - {tool.name}")

            # -------------------------
            # 2. Create sample DB
            # -------------------------
            print("\nCreating sample DB...")
            result = await session.call_tool("create_sample_db", {})
            print(result)

            # -------------------------
            # 3. Inspect schema
            # -------------------------
            print("\nInspecting schema...")
            result = await session.call_tool("inspect_schema", {
                "connection_string": "sqlite:///enterprise_system.db"
            })
            print(result)

            # -------------------------
            # 4. Plan migration
            # -------------------------
            print("\nPlanning migration...")
            result = await session.call_tool("plan_migration", {
                "connection_string": "sqlite:///enterprise_system.db"
            })
            print(result)

            # -------------------------
            # 5. Export one table
            # -------------------------
            print("\nExporting employees -> mongo...")
            result = await session.call_tool("export_table", {
                "table_name": "employees",
                "target_db": "mongo",
                "connection_string": "sqlite:///enterprise_system.db"
            })
            print(result)

            # -------------------------
            # 6. Run full pipeline
            # -------------------------
            print("\nRunning full pipeline...")
            result = await session.call_tool("run_full_pipeline", {
                "connection_string": "sqlite:///enterprise_system.db"
            })
            print(result)


if __name__ == "__main__":
    asyncio.run(main())