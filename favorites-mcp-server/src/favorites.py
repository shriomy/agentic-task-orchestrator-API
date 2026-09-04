"""MongoDB favorites — ported unchanged from trip-agent/src/tools/favorites.py.

Every function takes `user_id` as its first argument and routes the query
through the authorization guardrail. server.py is the only caller, and it
always supplies `user_id` from the verified token, never from a tool argument.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from .authorization import assert_owned, filter_owned, require_user_id, scoped_filter
from .db import SECTION_ID_FIELD, SECTIONS, destination_key, favorites_collection, resolve_section


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _public(document: dict[str, Any]) -> dict[str, Any]:
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
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    def key_of(item: dict[str, Any]) -> str:
        return str(item.get(id_field) or item.get("name") or uuid.uuid4())

    for item in list(existing) + list(incoming):
        if not isinstance(item, dict):
            continue
        key = key_of(item)
        if key in seen:
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
    owner = require_user_id(user_id, operation="get_favorites")
    collection = favorites_collection()

    extra: dict[str, Any] = {}
    if destination:
        extra["destination_key"] = destination_key(destination)

    raw = list(collection.find(scoped_filter(owner, extra, operation="get_favorites")).sort("updated_at", -1))
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
        "remaining": {other: len(current.get(other, [])) for other in SECTIONS if other != resolved}
        | {resolved: len(after)},
    }


def delete_favorite(user_id: str, destination: str) -> dict:
    owner = require_user_id(user_id, operation="delete_favorite")
    collection = favorites_collection()
    key = destination_key(destination)

    current = collection.find_one(scoped_filter(owner, {"destination_key": key}, operation="delete_favorite"))
    if current is None:
        return {"action": "not_found", "destination": destination}
    assert_owned(current, owner, operation="delete_favorite")

    result = collection.delete_one(scoped_filter(owner, {"_id": current["_id"]}, operation="delete_favorite"))
    return {
        "action": "deleted" if result.deleted_count else "not_found",
        "destination": (current.get("destination") or {}).get("name", destination),
        "deleted_count": result.deleted_count,
    }
