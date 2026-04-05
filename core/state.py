from typing import TypedDict

class StrategyState(TypedDict):
    table_name: str
    table_info: dict
    all_tables: list[str]
    target_db: str | None
    routing_reason: str | None
    strategy: dict | None
    errors: list[str]
    retries: int
    final: dict | None