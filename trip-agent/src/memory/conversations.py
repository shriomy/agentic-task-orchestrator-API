"""Conversation and chat-history persistence in Supabase (`agent_*` tables).

This is a durable, queryable mirror of the chat for the UI's sidebar and thread
reload. It is deliberately separate from the checkpointer: the checkpointer owns
the graph's execution state (and is opaque), while these tables own the
human-readable transcript.

Every function is user-scoped and non-fatal — a logging outage must never fail
a user's chat turn.
"""

from __future__ import annotations

import logging
from typing import Any

from .supabase_client import get_service_client, supabase_available

logger = logging.getLogger(__name__)

CONVERSATIONS = "agent_conversations"
MESSAGES = "agent_messages"
SELECTION_LOG = "agent_selection_log"
MESSAGE_USAGE = "agent_message_usage"


def _title_from(message: str) -> str:
    text = " ".join((message or "").split())
    return (text[:57] + "...") if len(text) > 60 else (text or "New conversation")


def ensure_conversation(thread_id: str, user_id: str, first_message: str = "") -> bool:
    """Create the conversation row if absent. Returns whether the row exists."""
    if not (supabase_available() and thread_id and user_id):
        return False
    client = get_service_client()
    try:
        existing = (
            client.table(CONVERSATIONS)
            .select("id")
            .eq("id", thread_id)
            .eq("user_id", user_id)  # authorization: never adopt another user's thread
            .limit(1)
            .execute()
        )
        if existing.data:
            return True
        client.table(CONVERSATIONS).insert(
            {
                "id": thread_id,
                "user_id": user_id,
                "title": _title_from(first_message),
                "preview": _title_from(first_message),
            }
        ).execute()
        return True
    except Exception as exc:
        logger.warning("could not ensure conversation %s: %s", thread_id, exc)
        return False


def record_message(
    thread_id: str,
    user_id: str,
    role: str,
    content: str,
    *,
    interrupt_data: dict[str, Any] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    turn_index: int = 0,
) -> str | None:
    """Insert one message row. Returns its id (for record_message_usage to
    link against), or None if it wasn't written."""
    if not (supabase_available() and thread_id and user_id):
        return None
    try:
        client = get_service_client()
        response = (
            client.table(MESSAGES)
            .insert(
                {
                    "conversation_id": thread_id,
                    "role": role,
                    "content": content or "",
                    "interrupt_data": interrupt_data,
                    "tool_calls": tool_calls,
                    "turn_index": turn_index,
                }
            )
            .execute()
        )
        if role == "assistant" and content:
            client.table(CONVERSATIONS).update({"preview": _title_from(content)}).eq(
                "id", thread_id
            ).eq("user_id", user_id).execute()
        rows = response.data or []
        return rows[0]["id"] if rows else None
    except Exception as exc:
        logger.warning("could not record %s message on %s: %s", role, thread_id, exc)
        return None


def record_message_usage(
    thread_id: str,
    message_id: str | None,
    turn_index: int,
    breakdown: dict[str, Any],
) -> None:
    """Persist one turn's token/cost breakdown, linked to its assistant
    message. Non-fatal like every other write here — a logging outage must
    never fail a user's chat turn."""
    if not (supabase_available() and thread_id):
        return
    try:
        get_service_client().table(MESSAGE_USAGE).insert(
            {
                "conversation_id": thread_id,
                "message_id": message_id,
                "turn_index": turn_index,
                "model": breakdown.get("model"),
                "context_tokens": breakdown.get("context_tokens", 0),
                "memory_tokens": breakdown.get("memory_tokens", 0),
                "system_prompt_tokens": breakdown.get("system_prompt_tokens", 0),
                "tools_tokens": breakdown.get("tools_tokens", 0),
                "other_tokens": breakdown.get("other_tokens", 0),
                "total_tokens": breakdown.get("total_tokens", 0),
                "cost_usd": breakdown.get("cost_usd"),
            }
        ).execute()
    except Exception as exc:
        logger.warning("could not record usage for %s: %s", thread_id, exc)


def list_conversations(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    if not (supabase_available() and user_id):
        return []
    try:
        response = (
            get_service_client()
            .table(CONVERSATIONS)
            .select("id,title,preview,turn_count,created_at,updated_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        logger.warning("could not list conversations for %s: %s", user_id, exc)
        return []


def get_history(thread_id: str, user_id: str, limit: int = 200) -> list[dict[str, Any]]:
    """Read a thread's transcript, after confirming the caller owns the thread."""
    if not (supabase_available() and thread_id and user_id):
        return []
    try:
        client = get_service_client()
        owned = (
            client.table(CONVERSATIONS)
            .select("id")
            .eq("id", thread_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if not owned.data:
            return []
        response = (
            client.table(MESSAGES)
            .select("id,role,content,interrupt_data,tool_calls,turn_index,created_at")
            .eq("conversation_id", thread_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        logger.warning("could not read history for %s: %s", thread_id, exc)
        return []


def delete_conversation(thread_id: str, user_id: str) -> bool:
    if not (supabase_available() and thread_id and user_id):
        return False
    try:
        get_service_client().table(CONVERSATIONS).delete().eq("id", thread_id).eq(
            "user_id", user_id
        ).execute()
        return True
    except Exception as exc:
        logger.warning("could not delete conversation %s: %s", thread_id, exc)
        return False


def log_selection(
    thread_id: str,
    selection_id: str,
    kind: str,
    options: list[dict[str, Any]],
    *,
    destination: str | None = None,
    picked_ids: list[str] | None = None,
    status: str = "pending",
    turn_index: int = 0,
) -> None:
    """Audit a HIL interrupt and, on resume, its outcome."""
    if not (supabase_available() and thread_id):
        return
    try:
        get_service_client().table(SELECTION_LOG).upsert(
            {
                "conversation_id": thread_id,
                "selection_id": selection_id,
                "kind": kind,
                "destination": destination,
                "options": options,
                "picked_ids": picked_ids or [],
                "status": status,
                "turn_index": turn_index,
            },
            on_conflict="conversation_id,selection_id",
        ).execute()
    except Exception as exc:
        logger.debug("could not log selection %s: %s", selection_id, exc)
