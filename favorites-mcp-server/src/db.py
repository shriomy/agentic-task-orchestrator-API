"""MongoDB client and favorites schema.

Ported from trip-agent/src/db/mongo.py unchanged — same pooled-client pattern,
same document shape, same collection. This server and trip-agent point at the
identical MongoDB cluster/database/collection; this is not a second database,
just a second way to reach the one favorites collection.
"""

from __future__ import annotations

import threading
from typing import Any

from .config import settings

_client: Any = None
_lock = threading.Lock()


def get_client() -> Any:
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        from pymongo import MongoClient

        if not settings.mongodb_uri:
            raise RuntimeError("MONGODB_URI is required for favorites persistence.")

        kwargs: dict[str, Any] = {
            "serverSelectionTimeoutMS": settings.request_timeout_seconds * 1000,
            "tz_aware": True,
            "appname": "favorites-mcp-server",
        }
        if settings.mongodb_uri.startswith("mongodb+srv://"):
            try:
                import certifi

                kwargs["tlsCAFile"] = certifi.where()
            except ImportError:
                pass

        _client = MongoClient(settings.mongodb_uri, **kwargs)
        return _client


def favorites_collection() -> Any:
    collection = get_client()[settings.mongodb_db_name][settings.mongodb_favorites_collection]
    _ensure_indexes(collection)
    return collection


_indexes_ready = False


def _ensure_indexes(collection: Any) -> None:
    global _indexes_ready
    if _indexes_ready:
        return
    try:
        collection.create_index(
            [("user_id", 1), ("destination_key", 1)],
            unique=True,
            name="uniq_user_destination",
        )
        collection.create_index([("user_id", 1), ("updated_at", -1)], name="user_recent")
        _indexes_ready = True
    except Exception:
        _indexes_ready = True


def ping() -> bool:
    try:
        get_client().admin.command("ping")
        return True
    except Exception:
        return False


SECTIONS = ("places", "events", "accommodations")

SECTION_ID_FIELD = {
    "places": "place_id",
    "events": "event_id",
    "accommodations": "hotel_id",
}

SECTION_ALIASES = {
    "places": {"place", "places", "poi", "attraction", "attractions", "sights", "sight", "visits", "visit", "things to do"},
    "events": {"event", "events", "activity", "activities", "concert", "concerts", "shows", "show"},
    "accommodations": {
        "accommodation", "accommodations", "hotel", "hotels", "stay", "stays",
        "hostel", "hostels", "cabin", "cabins", "lodging", "night stays", "booking", "bookings",
    },
}


def resolve_section(word: str) -> str | None:
    needle = (word or "").strip().lower()
    if needle in SECTIONS:
        return needle
    for section, aliases in SECTION_ALIASES.items():
        if needle in aliases:
            return section
    return None


def destination_key(name: str) -> str:
    return " ".join((name or "").strip().lower().split())
