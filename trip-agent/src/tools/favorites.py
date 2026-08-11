from ..memory.supabase_client import supabase_service_client


def save_favorite(user_id: str, city: str, events: list[dict], hotel: dict, notes: str | None = None) -> dict:
    payload = {
        "user_id": user_id,
        "city": city,
        "events": events,
        "hotel": hotel,
        "notes": notes,
    }
    response = supabase_service_client.table("favorites").insert(payload).execute()
    if response.error:
        raise RuntimeError(f"Failed to save favorite: {response.error.message}")
    return response.data


def get_favorites(user_id: str) -> dict:
    response = (
        supabase_service_client
        .table("favorites")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )
    if response.error:
        raise RuntimeError(f"Failed to get favorites: {response.error.message}")
    return {"favorites": response.data}
