"""Favorites MCP server.

Three plain tools (list/save/update) plus two destructive tools that elicit a
yes/no confirmation from the connected client *before* touching MongoDB —
`ctx.elicit()` always runs before any write, so a declined or dropped
confirmation never leaves a partial mutation behind.

No tool takes `user_id` as an argument. See auth.py / authorization.py: the
acting identity comes only from the verified bearer token on the connection.
"""

from __future__ import annotations

import logging

from fastmcp import Context, FastMCP
from fastmcp.server.dependencies import get_access_token

from .auth import SupabaseTokenVerifier
from .config import settings
from .favorites import (
    delete_favorite,
    get_favorites,
    remove_favorite_section,
    save_favorite,
    update_favorite,
)
from .db import resolve_section

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("favorites", auth=SupabaseTokenVerifier())


def _current_user_id() -> str:
    """Resolve the acting user from the verified token, never from an argument."""
    token = get_access_token()
    if token is None or not token.client_id:
        raise PermissionError("This call requires a signed-in user; no verified token was present.")
    return token.client_id


@mcp.tool()
def list_trip_favorites(destination: str | None = None, section: str | None = None) -> dict:
    """List the caller's saved trips.

    Args:
        destination: Optional — limit to one destination, e.g. "Kyoto".
        section: Optional — return only "places", "events" or "accommodations".
    """
    return get_favorites(_current_user_id(), destination=destination, section=section)


@mcp.tool()
def save_trip_favorite(
    destination: str,
    places: list[dict] | None = None,
    events: list[dict] | None = None,
    accommodations: list[dict] | None = None,
    notes: str | None = None,
    thread_id: str | None = None,
) -> dict:
    """Create or extend the saved trip for a destination.

    Saving twice for the same destination merges into the one document rather
    than creating a duplicate. Callers should resolve any implicit picks
    (e.g. "save what we picked earlier") to explicit lists before calling —
    this tool has no conversation state to draw on.

    Args:
        destination: The destination name, e.g. "Kyoto".
        places: Place objects the user chose.
        events: Event objects the user chose.
        accommodations: Accommodation objects the user chose.
        notes: Any free-text note the user wants kept with the trip.
        thread_id: The conversation this was saved from, for provenance.
    """
    return save_favorite(
        _current_user_id(),
        destination,
        places=places,
        events=events,
        accommodations=accommodations,
        notes=notes,
        thread_id=thread_id,
    )


@mcp.tool()
def update_trip_favorite(
    destination: str,
    places: list[dict] | None = None,
    events: list[dict] | None = None,
    accommodations: list[dict] | None = None,
    notes: str | None = None,
) -> dict:
    """Add to a saved trip. Sections not mentioned are left untouched, and
    within a mentioned section items are ADDED to what's already saved.

    Args:
        destination: Which saved trip to change, e.g. "Paris".
        places: Place objects to add.
        events: Event objects to add.
        accommodations: Accommodation objects to add.
        notes: Replacement note text.
    """
    return update_favorite(
        _current_user_id(),
        destination,
        places=places,
        events=events,
        accommodations=accommodations,
        notes=notes,
        replace_sections=False,
    )


@mcp.tool()
async def remove_trip_favorite_section(
    ctx: Context,
    destination: str,
    section: str,
    item_ids: list[str] | None = None,
) -> dict:
    """Remove one part of a saved trip, keeping the rest of it intact.

    This is the tool for "delete my Paris trip's hotels": it clears the
    accommodations and leaves the saved places and events exactly as they
    were. Do not use delete_trip_favorite for that. Elicits a yes/no
    confirmation from the user before making the change — just call it,
    there is no need to confirm yourself first.

    Args:
        destination: Which saved trip, e.g. "Paris".
        section: "places", "events" or "accommodations".
        item_ids: Optional — remove only these specific items instead of the
            whole section.
    """
    resolved = resolve_section(section) or section
    what = f"{len(item_ids)} item(s) from" if item_ids else "all of"
    result = await ctx.elicit(
        f"Remove {what} the {resolved} in your {destination} trip? "
        "The rest of that trip stays as it is.",
        response_type=bool,
    )
    if not (result.action == "accept" and result.data):
        return {
            "action": "cancelled",
            "destination": destination,
            "section": resolved,
            # Blunt on purpose, matching the wording the model already relies
            # on: a cancelled action must be impossible to misread as done.
            "outcome_for_user": f"NOT removed. The user said no, so the {resolved} in {destination} is unchanged.",
        }
    return remove_favorite_section(_current_user_id(), destination, section, item_ids=item_ids)


@mcp.tool()
async def delete_trip_favorite(ctx: Context, destination: str) -> dict:
    """Delete an ENTIRE saved trip and everything under it.

    Only for when the user wants the whole destination gone. If they named
    one part of it (the hotels, the events), use
    remove_trip_favorite_section instead. Elicits a yes/no confirmation from
    the user before deleting — just call it, there is no need to confirm
    yourself first.

    Args:
        destination: Which saved trip to delete, e.g. "Paris".
    """
    result = await ctx.elicit(
        f"Delete your entire {destination} trip — all its places, events and stays? "
        "This can't be undone.",
        response_type=bool,
    )
    if not (result.action == "accept" and result.data):
        return {
            "action": "cancelled",
            "destination": destination,
            "outcome_for_user": f"NOT deleted. The user said no, so the {destination} trip is still saved exactly as it was.",
        }
    return delete_favorite(_current_user_id(), destination)


if __name__ == "__main__":
    logger.info("favorites-mcp-server starting on port %s", settings.mcp_port)
    mcp.run(transport="streamable-http", host="0.0.0.0", port=settings.mcp_port)
