"""MongoDB client and favorites schema.

One pooled client is shared process-wide. The previous implementation built a
new `MongoClient` on every call, which leaks a connection pool per tool
invocation.
"""

from __future__ import annotations

import threading
from typing import Any

from ..config import settings

_client: Any = None
_lock = threading.Lock()


def get_client() -> Any:
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        try:
            from pymongo import MongoClient
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError("pymongo is required for favorites persistence.") from exc
        if not settings.mongodb_uri:
            raise RuntimeError("MONGODB_URI is required for favorites persistence.")
        _client = MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=settings.request_timeout_seconds * 1000,
            tz_aware=True,
        )
        return _client


def favorites_collection() -> Any:
    collection = get_client()[settings.mongodb_db_name][settings.mongodb_favorites_collection]
    _ensure_indexes(collection)
    return collection


_indexes_ready = False


def _ensure_indexes(collection: Any) -> None:
    """One trip document per (user, destination); every read is user-scoped."""
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
        # A replica that refuses index creation must not break reads/writes.
        _indexes_ready = True


def ping() -> bool:
    try:
        get_client().admin.command("ping")
        return True
    except Exception:
        return False


# --- Document shape ---------------------------------------------------------
#
# The favorites object is centred on the destination, with places-to-visit and
# events/activities as siblings one level down, and accommodations below those:
#
# {
#   "_id":             "<uuid4>",
#   "user_id":         "<supabase auth uid>",     # authorization key, never from the LLM
#   "destination_key": "kyoto",                   # normalised, unique per user
#   "destination": {
#       "name": "Kyoto", "country": "JP",
#       "lat": 35.01, "lon": 135.76, "summary": "..."
#   },
#   "places":         [ { place_id, name, kinds, lat, lon, url, source, notes } ],
#   "events":         [ { event_id, name, date, venue, url, price_min, source } ],
#   "accommodations": [ { hotel_id, name, price, currency, review_score,
#                         check_in, check_out, url, source,
#                         near_place_id } ],     # links a stay to a chosen place
#   "notes":       "free text the user added",
#   "thread_id":   "conversation this was saved from",
#   "created_at":  "<iso8601>",
#   "updated_at":  "<iso8601>"
# }

SECTIONS = ("places", "events", "accommodations")

# The identity field inside each section, used for de-duplicating on merge.
SECTION_ID_FIELD = {
    "places": "place_id",
    "events": "event_id",
    "accommodations": "hotel_id",
}

# Words a user might use for each section, so "delete Paris trip's hotels"
# resolves to the accommodations array.
SECTION_ALIASES = {
    "places": {"place", "places", "poi", "attraction", "attractions", "sights", "sight", "visits", "visit", "things to do"},
    "events": {"event", "events", "activity", "activities", "concert", "concerts", "shows", "show"},
    "accommodations": {
        "accommodation", "accommodations", "hotel", "hotels", "stay", "stays",
        "hostel", "hostels", "cabin", "cabins", "lodging", "night stays", "booking", "bookings",
    },
}


def resolve_section(word: str) -> str | None:
    """Map a user's word for a section onto the canonical array name."""
    needle = (word or "").strip().lower()
    if needle in SECTIONS:
        return needle
    for section, aliases in SECTION_ALIASES.items():
        if needle in aliases:
            return section
    return None


def destination_key(name: str) -> str:
    """Normalise a destination name into its document key."""
    return " ".join((name or "").strip().lower().split())
