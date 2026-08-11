from typing import Any
from datetime import datetime
from pydantic import BaseModel

from .supabase_client import supabase_service_client


class UserMemorySummary(BaseModel):
    user_id: str
    preferences: dict[str, Any]
    favorite_count: int


def get_user_memory_summary(user_id: str) -> UserMemorySummary:
    """Read a lightweight summary of durable user memory.

    This method returns only personalization hints, not the full favorite records.
    """
    response = (
        supabase_service_client
        .table("user_memory")
        .select("key,value")
        .eq("user_id", user_id)
        .execute()
    )
    if response.error:
        raise RuntimeError(f"Failed to read user memory: {response.error.message}")

    preferences: dict[str, Any] = {}
    for row in response.data:
        if row["key"] != "favorites":
            preferences[row["key"]] = row["value"]

    favorite_count = 0
    count_response = (
        supabase_service_client
        .table("favorites")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .execute()
    )
    if count_response.error:
        raise RuntimeError(f"Failed to count favorites: {count_response.error.message}")
    favorite_count = len(count_response.data)

    return UserMemorySummary(
        user_id=user_id,
        preferences=preferences,
        favorite_count=favorite_count,
    )


def write_user_memory(user_id: str, key: str, value: Any) -> None:
    """Persist a durable preference into user memory.

    The graph may call this when it detects a durable preference statement.
    """
    response = (
        supabase_service_client
        .table("user_memory")
        .upsert(
            {
                "user_id": user_id,
                "key": key,
                "value": value,
                "updated_at": datetime.utcnow().isoformat(),
            },
            on_conflict=["user_id", "key"],
        )
        .execute()
    )
    if response.error:
        raise RuntimeError(f"Failed to write user memory: {response.error.message}")
