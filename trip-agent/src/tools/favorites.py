"""MongoDB favorites — a saved trip per destination.

Every function here takes `user_id` as its first argument and routes the query
through the authorization guardrail. The document hierarchy is
destination -> {places, events} -> accommodations, described in db/mongo.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from ..db.mongo import (
    SECTION_ID_FIELD,
    SECTIONS,
    destination_key,
    favorites_collection,
    resolve_section,
)
from ..guardrails.authorization import (
    assert_owned,
    filter_owned,
    require_user_id,
    scoped_filter,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _public(document: dict[str, Any]) -> dict[str, Any]:
    """Shape a document for the model / API: id renamed, user_id withheld."""
    shaped = dict(document)
    shaped["favorite_id"] = str(shaped.pop("_id", ""))
    shaped.pop("user_id", None)
    for section in SECTIONS:
        shaped.setdefault(section, [])
    return shaped


def _merge_items(
    existing: Iterable[dict[str, Any]],
    incoming: Iterable[dict[str, Any]],
    id_field: str,
) -> list[dict[str, Any]]:
    """Union two item lists, de-duplicating on the section's id field.

    Items without an id fall back to their name, so a hand-typed pick from the
    user does not create a duplicate of an API-sourced one.
    """
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    def key_of(item: dict[str, Any]) -> str:
        return str(item.get(id_field) or item.get("name") or uuid.uuid4())

    for item in list(existing) + list(incoming):
        if not isinstance(item, dict):
            continue
        key = key_of(item)
        if key in seen:
            # Later wins on conflict: an update should refresh a stale price.
            merged = [item if key_of(m) == key else m for m in merged]
            continue
        seen.add(key)
        merged.append(item)
    return merged


def save_favorite(
    user_id: str,
    destination: str | dict[str, Any],
    places: list[dict] | None = None,
    events: list[dict] | None = None,
    accommodations: list[dict] | None = None,
    notes: str | None = None,
    thread_id: str | None = None,
) -> dict:
    """Create or extend the saved trip for a destination.

    Saving twice for the same destination merges into the one document rather
    than creating a second Kyoto trip.
    """
    owner = require_user_id(user_id, operation="save_favorite")
    collection = favorites_collection()

    dest = destination if isinstance(destination, dict) else {"name": destination}
    name = str(dest.get("name") or "").strip()
    if not name:
        raise ValueError("A destination name is required to save a favorite.")
    key = destination_key(name)

    current = collection.find_one(scoped_filter(owner, {"destination_key": key}, operation="save_favorite"))
    if current is not None:
        assert_owned(current, owner, operation="save_favorite")

    incoming = {
        "places": places or [],
        "events": events or [],
        "accommodations": accommodations or [],
    }
    merged = {
        section: _merge_items(
            (current or {}).get(section, []),
            incoming[section],
            SECTION_ID_FIELD[section],
        )
        for section in SECTIONS
    }

    merged_destination = {**((current or {}).get("destination") or {}), **dest}
    updates: dict[str, Any] = {
        "destination": merged_destination,
        "destination_key": key,
        **merged,
        "updated_at": _now(),
    }
    if notes:
        existing_notes = (current or {}).get("notes")
        updates["notes"] = f"{existing_notes}\n{notes}".strip() if existing_notes else notes
    if thread_id:
        updates["thread_id"] = thread_id

    if current is None:
        document = {
            "_id": str(uuid.uuid4()),
            "user_id": owner,
            "notes": notes,
            "created_at": _now(),
            **updates,
        }
        collection.insert_one(document)
        return {"action": "created", "favorite": _public(document)}

    collection.update_one(
        scoped_filter(owner, {"_id": current["_id"]}, operation="save_favorite"),
        {"$set": updates},
    )
    return {"action": "updated", "favorite": _public({**current, **updates})}


def get_favorites(
    user_id: str,
    destination: str | None = None,
    section: str | None = None,
) -> dict:
    """List the caller's saved trips, optionally one destination or one section."""
    owner = require_user_id(user_id, operation="get_favorites")
    collection = favorites_collection()

    extra: dict[str, Any] = {}
    if destination:
        extra["destination_key"] = destination_key(destination)

    raw = list(
        collection.find(scoped_filter(owner, extra, operation="get_favorites")).sort("updated_at", -1)
    )
    owned = filter_owned(raw, owner, operation="get_favorites")
    favorites = [_public(document) for document in owned]

    resolved_section = resolve_section(section) if section else None
    if resolved_section:
        favorites = [
            {
                "favorite_id": favorite["favorite_id"],
                "destination": favorite.get("destination"),
                resolved_section: favorite.get(resolved_section, []),
            }
            for favorite in favorites
        ]

    return {"count": len(favorites), "favorites": favorites}


def update_favorite(
    user_id: str,
    destination: str,
    places: list[dict] | None = None,
    events: list[dict] | None = None,
    accommodations: list[dict] | None = None,
    notes: str | None = None,
    replace_sections: bool = False,
) -> dict:
    """Change a saved trip.

    With `replace_sections=False` (default) the supplied items are merged in;
    with True the named sections are overwritten. Sections not mentioned are
    always left untouched.
    """
    owner = require_user_id(user_id, operation="update_favorite")
    collection = favorites_collection()
    key = destination_key(destination)

    current = collection.find_one(scoped_filter(owner, {"destination_key": key}, operation="update_favorite"))
    if current is None:
        return {"action": "not_found", "destination": destination}
    assert_owned(current, owner, operation="update_favorite")

    supplied = {"places": places, "events": events, "accommodations": accommodations}
    updates: dict[str, Any] = {"updated_at": _now()}
    for section, items in supplied.items():
        if items is None:
            continue
        updates[section] = (
            list(items)
            if replace_sections
            else _merge_items(current.get(section, []), items, SECTION_ID_FIELD[section])
        )
    if notes is not None:
        updates["notes"] = notes

    collection.update_one(
        scoped_filter(owner, {"_id": current["_id"]}, operation="update_favorite"),
        {"$set": updates},
    )
    return {"action": "updated", "favorite": _public({**current, **updates})}


def remove_favorite_section(
    user_id: str,
    destination: str,
    section: str,
    item_ids: list[str] | None = None,
) -> dict:
    """Remove one section of a trip, or specific items inside it.

    This is what "delete Paris trip's hotels" resolves to: the accommodations
    array is cleared while the saved places and events stay exactly as they were.
    """
    owner = require_user_id(user_id, operation="remove_favorite_section")
    resolved = resolve_section(section)
    if resolved is None:
        raise ValueError(
            f"'{section}' is not part of a saved trip. Expected one of: places, events, accommodations."
        )

    collection = favorites_collection()
    key = destination_key(destination)
    current = collection.find_one(
        scoped_filter(owner, {"destination_key": key}, operation="remove_favorite_section")
    )
    if current is None:
        return {"action": "not_found", "destination": destination}
    assert_owned(current, owner, operation="remove_favorite_section")

    before = list(current.get(resolved, []))
    id_field = SECTION_ID_FIELD[resolved]
    if item_ids:
        wanted = {str(item) for item in item_ids}
        after = [
            item
            for item in before
            if str(item.get(id_field)) not in wanted and str(item.get("name")) not in wanted
        ]
    else:
        after = []

    collection.update_one(
        scoped_filter(owner, {"_id": current["_id"]}, operation="remove_favorite_section"),
        {"$set": {resolved: after, "updated_at": _now()}},
    )
    return {
        "action": "section_removed",
        "destination": (current.get("destination") or {}).get("name", destination),
        "section": resolved,
        "removed_count": len(before) - len(after),
        # Proof the rest of the trip survived, so the agent can say so.
        "remaining": {other: len(current.get(other, [])) for other in SECTIONS if other != resolved}
        | {resolved: len(after)},
    }


def delete_favorite(user_id: str, destination: str) -> dict:
    """Delete an entire saved trip for a destination."""
    owner = require_user_id(user_id, operation="delete_favorite")
    collection = favorites_collection()
    key = destination_key(destination)

    current = collection.find_one(scoped_filter(owner, {"destination_key": key}, operation="delete_favorite"))
    if current is None:
        return {"action": "not_found", "destination": destination}
    assert_owned(current, owner, operation="delete_favorite")

    result = collection.delete_one(
        scoped_filter(owner, {"_id": current["_id"]}, operation="delete_favorite")
    )
    return {
        "action": "deleted" if result.deleted_count else "not_found",
        "destination": (current.get("destination") or {}).get("name", destination),
        "deleted_count": result.deleted_count,
    }
