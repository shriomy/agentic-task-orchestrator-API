import uuid
from typing import Any

from ..config import settings


def _get_collection():
    try:
        from pymongo import MongoClient
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("pymongo is required for MongoDB favorites persistence.") from exc

    if not settings.mongodb_uri:
        raise RuntimeError("MONGODB_URI is required for favorites persistence.")

    client = MongoClient(settings.mongodb_uri)
    database = client[settings.mongodb_db_name]
    return database[settings.mongodb_favorites_collection]


def save_favorite(user_id: str, city: str, events: list[dict], hotel: dict | None = None, notes: str | None = None) -> dict:
    collection = _get_collection()
    payload = {
        "_id": str(uuid.uuid4()),
        "user_id": user_id,
        "city": city,
        "events": events,
        "hotel": hotel or {},
        "notes": notes,
    }
    collection.insert_one(payload)
    return {"favorite": payload}


def get_favorites(user_id: str) -> dict:
    collection = _get_collection()
    favorites = list(collection.find({"user_id": user_id}))
    for item in favorites:
        item["id"] = str(item.pop("_id"))
    return {"favorites": favorites}


def update_favorite(user_id: str, favorite_id: str, updates: dict[str, Any]) -> dict:
    collection = _get_collection()
    result = collection.update_one({"_id": favorite_id, "user_id": user_id}, {"$set": updates})
    return {"matched": result.matched_count, "modified": result.modified_count, "favorite_id": favorite_id}


def delete_favorite(user_id: str, favorite_id: str) -> dict:
    collection = _get_collection()
    result = collection.delete_one({"_id": favorite_id, "user_id": user_id})
    return {"deleted": result.deleted_count, "favorite_id": favorite_id}
