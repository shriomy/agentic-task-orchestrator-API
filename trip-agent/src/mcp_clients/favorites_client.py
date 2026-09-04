"""MCP client for favorites-mcp-server.

Every call opens a fresh streamable-HTTP session and forwards the caller's
verified Supabase token from `config.configurable.auth_token` (see
auth/supabase_auth.py, main.py) as the bearer token — the server verifies it
independently rather than trusting a `user_id` argument, same invariant the
in-process favorites tools used to enforce locally.

The two destructive tools (`remove_trip_favorite_section`, `delete_trip_favorite`)
use a probe -> interrupt() -> answer pattern, validated against a real running
server before this was written (see the plan's Phase 2 spike):

1. Probe: call the tool with a callback that declines and just records the
   server's exact confirmation message. The server takes its "cancelled,
   nothing changed" branch and the call returns cleanly — nothing mutates.
2. Pause: call LangGraph's interrupt() from plain sync code (not from inside
   the async callback — an earlier design tried that and the spike proved the
   mcp SDK swallows exceptions raised inside elicitation_callback rather than
   letting them propagate) with the captured message, in the same payload
   shape `_confirm()` already used in graph/tools.py, so the existing
   frontend confirmation UI needs no changes.
3. On replay (after /chat/resume), interrupt() returns the cached answer
   immediately instead of pausing again; step 1 runs again too (harmless —
   declines again) purely to reach the same call site.
4. Answer: call the tool again with a callback that returns the real yes/no,
   letting the server proceed with (or skip) the actual mutation.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client

from ..config import settings
from ..tools.http import ToolError


def _auth_token(config: RunnableConfig | None) -> str:
    token = ((config or {}).get("configurable") or {}).get("auth_token")
    if not token:
        raise ToolError("Not signed in.")
    return str(token)


async def _decline_elicitation(context: Any, params: Any) -> types.ElicitResult:
    """Non-destructive tools never elicit; this exists only so a session has
    a valid callback registered in case one ever unexpectedly does."""
    return types.ElicitResult(action="decline")


async def _call_tool_async(
    tool_name: str,
    args: dict[str, Any],
    token: str,
    elicitation_callback: Any,
) -> dict[str, Any]:
    try:
        async with streamablehttp_client(
            settings.favorites_mcp_url,
            headers={"Authorization": f"Bearer {token}"},
        ) as (read, write, _):
            async with ClientSession(read, write, elicitation_callback=elicitation_callback) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, args)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Could not reach the favorites service: {exc}") from exc

    text = result.content[0].text if result.content else "{}"
    if result.isError:
        raise ToolError(f"The favorites service rejected the request: {text}")
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ToolError("The favorites service returned a malformed response.") from exc


def _call_tool(
    tool_name: str,
    args: dict[str, Any],
    config: RunnableConfig | None,
    elicitation_callback: Any = _decline_elicitation,
) -> dict[str, Any]:
    token = _auth_token(config)
    return asyncio.run(_call_tool_async(tool_name, args, token, elicitation_callback))


def list_trip_favorites(
    config: RunnableConfig | None,
    destination: str | None = None,
    section: str | None = None,
) -> dict:
    return _call_tool("list_trip_favorites", {"destination": destination, "section": section}, config)


def save_trip_favorite(
    config: RunnableConfig | None,
    destination: str,
    places: list[dict] | None = None,
    events: list[dict] | None = None,
    accommodations: list[dict] | None = None,
    notes: str | None = None,
    thread_id: str | None = None,
) -> dict:
    return _call_tool(
        "save_trip_favorite",
        {
            "destination": destination,
            "places": places,
            "events": events,
            "accommodations": accommodations,
            "notes": notes,
            "thread_id": thread_id,
        },
        config,
    )


def update_trip_favorite(
    config: RunnableConfig | None,
    destination: str,
    places: list[dict] | None = None,
    events: list[dict] | None = None,
    accommodations: list[dict] | None = None,
    notes: str | None = None,
) -> dict:
    return _call_tool(
        "update_trip_favorite",
        {
            "destination": destination,
            "places": places,
            "events": events,
            "accommodations": accommodations,
            "notes": notes,
        },
        config,
    )


def _picked_ids_from(reply: Any) -> list[str]:
    """Same shape-tolerance as graph/tools.py's _picked_ids_from."""
    if reply is None:
        return []
    if isinstance(reply, str):
        return [reply]
    if isinstance(reply, list):
        return [str(item) for item in reply]
    if isinstance(reply, dict):
        for key in ("selected_options", "picked_ids", "picked", "selection", "ids"):
            value = reply.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                return [value]
            if isinstance(value, list):
                return [str(item) for item in value]
    return []


def _resolve_with_confirmation(
    tool_name: str,
    args: dict[str, Any],
    config: RunnableConfig | None,
    *,
    destination: str | None,
) -> dict:
    token = _auth_token(config)
    captured: dict[str, Any] = {}

    async def probing_callback(context: Any, params: Any) -> types.ElicitResult:
        captured["message"] = params.message
        return types.ElicitResult(action="decline")

    probe_result = asyncio.run(_call_tool_async(tool_name, args, token, probing_callback))
    if "message" not in captured:
        # The server didn't need to ask at all (e.g. nothing found) — done.
        return probe_result

    reply = interrupt(
        {
            "type": "selection",
            "selection_id": f"confirm_mcp_{uuid.uuid4().hex[:10]}",
            "kind": "confirmation",
            "destination": destination,
            "reason": captured["message"],
            "options": [
                {"id": "yes", "label": "Yes"},
                {"id": "no", "label": "No, keep it"},
            ],
        }
    )
    confirmed = "yes" in {str(item).strip().lower() for item in _picked_ids_from(reply)}

    async def answering_callback(context: Any, params: Any) -> types.ElicitResult:
        if confirmed:
            return types.ElicitResult(action="accept", content={"value": True})
        return types.ElicitResult(action="decline")

    return asyncio.run(_call_tool_async(tool_name, args, token, answering_callback))


def remove_trip_favorite_section(
    config: RunnableConfig | None,
    destination: str,
    section: str,
    item_ids: list[str] | None = None,
) -> dict:
    return _resolve_with_confirmation(
        "remove_trip_favorite_section",
        {"destination": destination, "section": section, "item_ids": item_ids},
        config,
        destination=destination,
    )


def delete_trip_favorite(config: RunnableConfig | None, destination: str) -> dict:
    return _resolve_with_confirmation(
        "delete_trip_favorite",
        {"destination": destination},
        config,
        destination=destination,
    )
