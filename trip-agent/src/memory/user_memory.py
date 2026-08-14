"""Cross-thread user memory, keyed by user_id.

This is the outermost of the three scopes:

  State   - thread-scoped, persisted by the LangGraph checkpointer on thread_id.
  Context - derived at call time from state; never stored on its own.
  Memory  - this module. Cross-thread, keyed on user_id, in agent_user_memory.

supabase-py v2 raises APIError on failure and its responses have no `.error`
attribute — the previous code read `response.error` on every call, which
raised AttributeError before any real work happened.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .supabase_client import get_service_client, supabase_available

logger = logging.getLogger(__name__)

TABLE = "agent_user_memory"

# Keys the agent is allowed to persist. An open-ended key space fills up with
# one-off junk that then pollutes every future prompt.
ALLOWED_KEYS = {
    "travel_style",          # "budget backpacking", "luxury", "family"
    "preferred_stay_type",   # "hostel", "cabin", "boutique hotel"
    "interests",             # ["museums", "hiking", "nightlife"]
    "dietary",               # "vegetarian"
    "home_city",
    "budget_level",
    "pace",                  # "packed itinerary" vs "slow travel"
    "avoid",                 # things they don't want suggested
    "hil_preference",        # "always ask me to pick" / "just decide"
    "accessibility",
}


class UserMemorySummary(BaseModel):
    user_id: str
    preferences: dict[str, Any] = Field(default_factory=dict)
    available: bool = True

    def as_prompt(self) -> str:
        """Render for injection into the agent prompt. Empty string when blank."""
        if not self.preferences:
            return ""
        lines = [f"- {key.replace('_', ' ')}: {value}" for key, value in sorted(self.preferences.items())]
        return "What you already know about this traveller (from earlier conversations):\n" + "\n".join(lines)


def get_user_memory_summary(user_id: str) -> UserMemorySummary:
    """Read durable preferences for a user. Never raises — memory is optional."""
    if not user_id or not supabase_available():
        return UserMemorySummary(user_id=user_id or "", available=False)

    try:
        response = (
            get_service_client()
            .table(TABLE)
            .select("key,value")
            .eq("user_id", user_id)
            .execute()
        )
    except Exception as exc:
        logger.warning("could not read user memory for %s: %s", user_id, exc)
        return UserMemorySummary(user_id=user_id, available=False)

    preferences = {
        row["key"]: row["value"]
        for row in (response.data or [])
        if row.get("key") in ALLOWED_KEYS
    }
    return UserMemorySummary(user_id=user_id, preferences=preferences)


def write_user_memory(user_id: str, key: str, value: Any) -> bool:
    """Persist one durable preference. Returns whether it was written."""
    if not user_id or key not in ALLOWED_KEYS or not supabase_available():
        if key not in ALLOWED_KEYS:
            logger.debug("refusing to persist unrecognised memory key %r", key)
        return False

    try:
        (
            get_service_client()
            .table(TABLE)
            .upsert(
                {
                    "user_id": user_id,
                    "key": key,
                    "value": value,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                on_conflict="user_id,key",
            )
            .execute()
        )
        return True
    except Exception as exc:
        logger.warning("could not write user memory %s=%r for %s: %s", key, value, user_id, exc)
        return False


def delete_user_memory(user_id: str, key: str) -> bool:
    if not user_id or not supabase_available():
        return False
    try:
        get_service_client().table(TABLE).delete().eq("user_id", user_id).eq("key", key).execute()
        return True
    except Exception as exc:
        logger.warning("could not delete user memory %s for %s: %s", key, user_id, exc)
        return False
