"""Tool definitions exposed to the agent.

Three things to note about identity, pausing, and confirmation:

* The favorites tools take NO user_id argument. The acting user is read from the
  runtime config, which the API layer fills in from the verified Supabase JWT.
  A model that hallucinates or is talked into naming another user's id has no
  parameter to put it in.

* `request_user_selection` calls `interrupt()`. It also returns a ToolMessage
  alongside the state update: a tool_call with no matching ToolMessage makes the
  next model request invalid, which is why the earlier version broke on resume.

* Destructive favorites actions (delete, section removal) confirm with the user
  BEFORE acting, and that confirmation is enforced inside the tool itself rather
  than left to the system prompt. The same reliability gap that made HIL pausing
  need code-level enforcement (see require_selection in nodes.py) applies here:
  a model instructed to "confirm before deleting" will not always do it. Calling
  `interrupt()` is not limited to a dedicated tool — any tool function can pause
  mid-execution, so these tools raise their own yes/no confirmation and only
  proceed once it comes back affirmative.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt

from ..db.mongo import resolve_section
from ..guardrails.authorization import AuthorizationError, require_user_id
from ..tools.activities import search_events
from ..tools.destinations import search_destinations
from ..tools.favorites import (
    delete_favorite,
    get_favorites,
    remove_favorite_section,
    save_favorite,
    update_favorite,
)
from ..tools.hotels import get_accommodation_details, search_accommodations
from ..tools.poi import get_place_details, search_places
from .state import Selection, SelectionOption

logger = logging.getLogger(__name__)

# The tool name the router watches for, so the pause is handled by the node that
# knows how to interrupt rather than by the generic tool executor.
SELECTION_TOOL = "request_user_selection"


def acting_user_id(config: RunnableConfig | None, *, operation: str) -> str:
    """Read the verified user id out of the runtime config.

    This is the single source of identity for every database operation. It is
    set once per request by the API layer from the decoded access token.
    """
    configurable = (config or {}).get("configurable") or {}
    return require_user_id(configurable.get("auth_user_id"), operation=operation)


# --------------------------------------------------------------------------- #
# Discovery tools
# --------------------------------------------------------------------------- #


@tool("web_search")
def web_search_tool(query: str, max_results: int = 6) -> dict:
    """Search the web for travel information.

    Use for finding which destinations to consider (country- or region-level,
    seasonal or "best of" questions), and for practical traveller questions such
    as what to pack, tap water safety, visas, or how long to spend somewhere.

    Args:
        query: A natural-language search query.
        max_results: How many results to return (1-10).
    """
    return search_destinations(query, max_results=max_results)


@tool("search_places")
def search_places_tool(
    city: str,
    category: str | None = None,
    radius_m: int = 12000,
    limit: int = 15,
) -> dict:
    """Find attractions and places to visit in a named city or town.

    Use when the destination is already known. Returns places with an id you can
    pass to request_user_selection or save as a favorite.

    Args:
        city: The city or town to search around, e.g. "Kyoto".
        category: Optional comma-separated OpenTripMap kinds, e.g.
            "museums,historic" or "beaches,natural".
        radius_m: Search radius in metres around the city centre.
        limit: Maximum number of places to return.
    """
    return search_places(city=city, category=category, radius_m=radius_m, limit=limit)


@tool("get_place_details")
def get_place_details_tool(place_id: str) -> dict:
    """Get the full description, image and address for one place.

    Args:
        place_id: The place_id returned by search_places.
    """
    return get_place_details(place_id)


@tool("search_events")
def search_events_tool(
    city: str,
    start_date: str | None = None,
    end_date: str | None = None,
    keyword: str | None = None,
    classification: str | None = None,
) -> dict:
    """Find events and dated activities in a city — concerts, sports, shows.

    Args:
        city: The city to search in, e.g. "Paris".
        start_date: Window start as YYYY-MM-DD. Defaults to today.
        end_date: Window end as YYYY-MM-DD. Defaults to 30 days out.
        keyword: Optional free-text filter, e.g. "jazz".
        classification: Optional category, e.g. "music" or "sports".
    """
    return search_events(
        city=city,
        keyword=keyword,
        start_date=start_date,
        end_date=end_date,
        classification=classification,
    )


@tool("search_accommodations")
def search_accommodations_tool(
    location: str,
    check_in: str | None = None,
    check_out: str | None = None,
    adults: int = 2,
    rooms: int = 1,
    max_price: float | None = None,
    stay_type: str | None = None,
    currency: str = "USD",
) -> dict:
    """Find places to stay — hotels, hostels, cabins, apartments — at a destination.

    Args:
        location: City, area or landmark to stay near, e.g. "Chiang Mai".
        check_in: Arrival date as YYYY-MM-DD. Defaults to two weeks out.
        check_out: Departure date as YYYY-MM-DD. Defaults to a 2-night stay.
        adults: Number of adult guests.
        rooms: Number of rooms.
        max_price: Optional maximum nightly price.
        stay_type: Optional filter — "hotel", "hostel", "cabin", "apartment".
        currency: Currency code for prices, e.g. "USD" or "EUR".
    """
    return search_accommodations(
        location=location,
        check_in=check_in,
        check_out=check_out,
        adults=adults,
        rooms=rooms,
        max_price=max_price,
        stay_type=stay_type,
        currency=currency,
    )


@tool("get_accommodation_details")
def get_accommodation_details_tool(
    hotel_id: str,
    check_in: str | None = None,
    check_out: str | None = None,
) -> dict:
    """Get full details and facilities for one property.

    Args:
        hotel_id: The hotel_id returned by search_accommodations.
        check_in: Arrival date as YYYY-MM-DD.
        check_out: Departure date as YYYY-MM-DD.
    """
    return get_accommodation_details(hotel_id, check_in=check_in, check_out=check_out)


# --------------------------------------------------------------------------- #
# Favorites (MongoDB) — user identity comes from config, never from arguments
# --------------------------------------------------------------------------- #


@tool("list_trip_favorites")
def list_trip_favorites_tool(
    destination: str | None = None,
    section: str | None = None,
    config: RunnableConfig = None,
) -> dict:
    """List the trips this user has saved.

    Args:
        destination: Optional — limit to one destination, e.g. "Kyoto".
        section: Optional — return only "places", "events" or "accommodations".
    """
    user_id = acting_user_id(config, operation="get_favorites")
    return get_favorites(user_id, destination=destination, section=section)


@tool("save_trip_favorite")
def save_trip_favorite_tool(
    destination: str,
    places: list[dict] | None = None,
    events: list[dict] | None = None,
    accommodations: list[dict] | None = None,
    notes: str | None = None,
    state: Annotated[Any, InjectedState] = None,
    config: RunnableConfig = None,
) -> dict:
    """Save a destination and its picked places, events and stays as a favorite.

    Saving again for the same destination merges into the existing saved trip
    rather than creating a duplicate. If the user says "save these" without
    listing them, leave the item lists empty — everything they picked earlier in
    this conversation is filled in automatically.

    Args:
        destination: The destination name, e.g. "Kyoto".
        places: Place objects from search_places that the user chose.
        events: Event objects from search_events that the user chose.
        accommodations: Accommodation objects from search_accommodations chosen.
        notes: Any free-text note the user wants kept with the trip.
    """
    user_id = acting_user_id(config, operation="save_favorite")

    # Backfill from the thread's answered selections so "save these details
    # regarding Kyoto" captures what was actually picked, turns ago, verbatim.
    from .context import saved_picks_payloads

    picked = saved_picks_payloads(state) if state is not None else {}
    key = (destination or "").strip().lower()

    def relevant(items: list[dict]) -> list[dict]:
        if not items:
            return []
        scoped = [
            item
            for item in items
            if not item.get("destination") or key in str(item.get("destination", "")).lower()
        ]
        return scoped or items

    places = places if places else relevant(picked.get("places", []))
    events = events if events else relevant(picked.get("events", []))
    accommodations = accommodations if accommodations else relevant(picked.get("accommodations", []))

    thread_id = getattr(state, "thread_id", None) if state is not None else None
    return save_favorite(
        user_id,
        destination,
        places=places,
        events=events,
        accommodations=accommodations,
        notes=notes,
        thread_id=thread_id,
    )


@tool("update_trip_favorite")
def update_trip_favorite_tool(
    destination: str,
    places: list[dict] | None = None,
    events: list[dict] | None = None,
    accommodations: list[dict] | None = None,
    notes: str | None = None,
    replace_sections: bool = False,
    config: RunnableConfig = None,
) -> dict:
    """Change a saved trip. Sections you do not mention are left untouched.

    Args:
        destination: Which saved trip to change, e.g. "Paris".
        places: Place objects to add (or to replace with).
        events: Event objects to add (or to replace with).
        accommodations: Accommodation objects to add (or to replace with).
        notes: Replacement note text.
        replace_sections: True overwrites the sections you supplied instead of
            merging into them.
    """
    user_id = acting_user_id(config, operation="update_favorite")
    return update_favorite(
        user_id,
        destination,
        places=places,
        events=events,
        accommodations=accommodations,
        notes=notes,
        replace_sections=replace_sections,
    )


def _confirm(reason: str, *, kind: str, destination: str | None) -> bool:
    """Pause for a yes/no and return whether the user confirmed.

    Called from inside a destructive tool, before it touches the database.
    `interrupt()` works from any tool function, not only a dedicated one — it
    just pauses this call until the graph is resumed.
    """
    reply = interrupt(
        {
            "type": "selection",
            "selection_id": f"confirm_{uuid.uuid4().hex[:10]}",
            "kind": "confirmation",
            "destination": destination,
            "reason": reason,
            "options": [
                {"id": "yes", "label": f"Yes, {kind}"},
                {"id": "no", "label": "No, keep it"},
            ],
        }
    )
    picked = {str(item).strip().lower() for item in _picked_ids_from(reply, [])}
    return "yes" in picked


@tool("remove_trip_favorite_section")
def remove_trip_favorite_section_tool(
    destination: str,
    section: str,
    item_ids: list[str] | None = None,
    config: RunnableConfig = None,
) -> dict:
    """Remove one part of a saved trip, keeping the rest of it intact.

    This is the tool for "delete my Paris trip's hotels": it clears the
    accommodations and leaves the saved places and events exactly as they were.
    Do not use delete_trip_favorite for that. Pauses to confirm before making
    the change — just call it, there is no need to confirm yourself first.

    Args:
        destination: Which saved trip, e.g. "Paris".
        section: "places", "events" or "accommodations".
        item_ids: Optional — remove only these specific items instead of the
            whole section.
    """
    user_id = acting_user_id(config, operation="remove_favorite_section")
    resolved = resolve_section(section) or section
    what = f"{len(item_ids)} item(s) from" if item_ids else "all of"
    confirmed = _confirm(
        f"Remove {what} the {resolved} in your {destination} trip? "
        "The rest of that trip stays as it is.",
        kind=f"remove the {resolved}",
        destination=destination,
    )
    if not confirmed:
        return {"action": "cancelled", "destination": destination, "section": resolved}
    return remove_favorite_section(user_id, destination, section, item_ids=item_ids)


@tool("delete_trip_favorite")
def delete_trip_favorite_tool(destination: str, config: RunnableConfig = None) -> dict:
    """Delete an ENTIRE saved trip and everything under it.

    Only for when the user wants the whole destination gone. If they named one
    part of it (the hotels, the events), use remove_trip_favorite_section.
    Pauses to confirm before deleting — just call it, there is no need to
    confirm yourself first.

    Args:
        destination: Which saved trip to delete, e.g. "Paris".
    """
    user_id = acting_user_id(config, operation="delete_favorite")
    confirmed = _confirm(
        f"Delete your entire {destination} trip — all its places, events and stays? "
        "This can't be undone.",
        kind="delete it",
        destination=destination,
    )
    if not confirmed:
        return {"action": "cancelled", "destination": destination}
    return delete_favorite(user_id, destination)


# --------------------------------------------------------------------------- #
# Human-in-the-loop selection
# --------------------------------------------------------------------------- #


@tool(SELECTION_TOOL)
def request_user_selection_tool(
    prompt: str,
    options: list[dict],
    kind: Literal["destination", "place", "event", "accommodation", "confirmation"] = "destination",
    destination: str | None = None,
    state: Annotated[Any, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Pause and ask the user to choose from a list before continuing.

    Use when the user asked to pick ("let me pick", "so I can choose"), and to
    confirm a destructive change to their saved trips. Call this on its own, not
    alongside other tools, since the conversation stops here until they answer.

    Args:
        prompt: What you are asking them to choose, in one sentence.
        options: The choices. Each needs {"id", "label"}, optionally
            "description", and "payload" holding the full object from the search
            result so their pick can be saved later.
        kind: What is being chosen — destination, place, event, accommodation,
            or confirmation for a yes/no on a destructive action.
        destination: The destination these options belong to, if applicable.
    """
    normalised: list[SelectionOption] = []
    for index, raw in enumerate(options or []):
        if isinstance(raw, str):
            raw = {"label": raw}
        label = str(raw.get("label") or raw.get("name") or f"Option {index + 1}")

        payload = raw.get("payload")
        if not isinstance(payload, dict):
            # Keep the whole object when the model didn't nest it under payload.
            payload = {k: v for k, v in raw.items() if k not in {"id", "label", "description", "payload"}}
        # Never leave a payload empty. Models frequently omit it, and an empty
        # one means "save these as favorites" later has nothing real to write.
        payload = dict(payload)
        payload.setdefault("name", label)
        if raw.get("description"):
            payload.setdefault("description", raw["description"])
        if destination:
            payload.setdefault("destination", destination)

        normalised.append(
            SelectionOption(
                id=str(raw.get("id") or f"opt_{index + 1}"),
                label=label,
                description=raw.get("description"),
                payload=payload,
            )
        )

    turn_index = int(getattr(state, "turn_index", 0) or 0)
    selection = Selection(
        kind=kind,
        prompt=prompt,
        destination=destination,
        options=normalised,
        turn_index=turn_index,
        tool_call_id=tool_call_id,
    )

    # Execution stops here. The payload is checkpointed, so the user can answer
    # now, in ten minutes, or after reloading the page.
    reply = interrupt(
        {
            "type": "selection",
            "selection_id": selection.selection_id,
            "kind": kind,
            "destination": destination,
            "reason": prompt,
            "options": [option.model_dump() for option in normalised],
        }
    )

    picked_ids = _picked_ids_from(reply, normalised)
    selection.resolve(picked_ids)

    if selection.status == "answered":
        chosen = ", ".join(option.label for option in selection.picked)
        summary = f"The user picked: {chosen}. Continue with only these."
    else:
        summary = (
            "The user picked nothing from that list. Do not continue with all of them — "
            "acknowledge it and ask what they would like to do."
        )

    return Command(
        update={
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id, name=SELECTION_TOOL)],
            "selections": [selection],
        }
    )


def _picked_ids_from(reply: Any, options: list[SelectionOption]) -> list[str]:
    """Accept whatever shape the resume payload arrives in.

    The UI sends {"selected_options": [...]}, but a direct API caller may send a
    bare list or a single string, and all three should work.
    """
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


# --------------------------------------------------------------------------- #

DISCOVERY_TOOLS = [
    web_search_tool,
    search_places_tool,
    get_place_details_tool,
    search_events_tool,
    search_accommodations_tool,
    get_accommodation_details_tool,
]

FAVORITE_TOOLS = [
    list_trip_favorites_tool,
    save_trip_favorite_tool,
    update_trip_favorite_tool,
    remove_trip_favorite_section_tool,
    delete_trip_favorite_tool,
]

ALL_TOOLS = [*DISCOVERY_TOOLS, *FAVORITE_TOOLS, request_user_selection_tool]

TOOLS_BY_NAME = {getattr(tool_fn, "name", ""): tool_fn for tool_fn in ALL_TOOLS}

# Tools that touch the user's stored data — logged and authorization-checked.
SENSITIVE_TOOL_NAMES = {getattr(tool_fn, "name", "") for tool_fn in FAVORITE_TOOLS}
