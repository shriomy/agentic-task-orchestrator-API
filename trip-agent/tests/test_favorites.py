"""MongoDB favorites: hierarchy, merging, partial removal, and cross-user isolation."""

from __future__ import annotations

import pytest

from src.db.mongo import resolve_section
from src.guardrails.authorization import AuthorizationError
from src.tools.favorites import (
    delete_favorite,
    get_favorites,
    remove_favorite_section,
    save_favorite,
    update_favorite,
)

KYOTO_PLACES = [
    {"place_id": "W1", "name": "Fushimi Inari", "kinds": "religion"},
    {"place_id": "W2", "name": "Kinkaku-ji", "kinds": "historic"},
]
KYOTO_EVENTS = [{"event_id": "E1", "name": "Gion Matsuri", "date": "2026-07-17"}]
KYOTO_STAYS = [{"hotel_id": "H1", "name": "Ryokan Sakura", "price": 180, "currency": "USD"}]


def test_saved_object_follows_the_destination_hierarchy(fake_collection):
    """destination -> {places, events} -> accommodations"""
    result = save_favorite(
        "user-1",
        {"name": "Kyoto", "country": "JP", "lat": 35.01, "lon": 135.76},
        places=KYOTO_PLACES,
        events=KYOTO_EVENTS,
        accommodations=KYOTO_STAYS,
        notes="Cherry blossom trip",
    )
    favorite = result["favorite"]

    assert result["action"] == "created"
    assert favorite["destination"]["name"] == "Kyoto"
    assert favorite["destination"]["country"] == "JP"
    assert [place["name"] for place in favorite["places"]] == ["Fushimi Inari", "Kinkaku-ji"]
    assert [event["name"] for event in favorite["events"]] == ["Gion Matsuri"]
    assert [stay["name"] for stay in favorite["accommodations"]] == ["Ryokan Sakura"]
    # user_id is the authorization key and is never handed back out.
    assert "user_id" not in favorite


def test_saving_the_same_destination_twice_merges_instead_of_duplicating(fake_collection):
    save_favorite("user-1", "Kyoto", places=KYOTO_PLACES[:1])
    result = save_favorite("user-1", "Kyoto", places=KYOTO_PLACES[1:], events=KYOTO_EVENTS)

    assert result["action"] == "updated"
    assert len(fake_collection.docs) == 1
    assert len(result["favorite"]["places"]) == 2
    assert len(result["favorite"]["events"]) == 1


def test_merging_refreshes_a_stale_item_rather_than_duplicating_it(fake_collection):
    save_favorite("user-1", "Kyoto", accommodations=[{"hotel_id": "H1", "name": "Ryokan Sakura", "price": 180}])
    result = save_favorite("user-1", "Kyoto", accommodations=[{"hotel_id": "H1", "name": "Ryokan Sakura", "price": 210}])

    stays = result["favorite"]["accommodations"]
    assert len(stays) == 1
    assert stays[0]["price"] == 210


def test_destination_names_are_matched_case_and_space_insensitively(fake_collection):
    save_favorite("user-1", "Kyoto", places=KYOTO_PLACES[:1])
    save_favorite("user-1", "  kyoto ", places=KYOTO_PLACES[1:])
    assert len(fake_collection.docs) == 1


# --------------------------------------------------------------------------- #
# "delete Paris trips' hotels"
# --------------------------------------------------------------------------- #


def test_removing_hotels_keeps_places_and_events(fake_collection):
    save_favorite(
        "user-1",
        "Paris",
        places=[{"place_id": "P1", "name": "Louvre"}],
        events=[{"event_id": "PE1", "name": "Opera Garnier"}],
        accommodations=[
            {"hotel_id": "PH1", "name": "Hotel Lumiere"},
            {"hotel_id": "PH2", "name": "Le Marais Hostel"},
        ],
    )

    result = remove_favorite_section("user-1", "Paris", "hotels")

    assert result["action"] == "section_removed"
    assert result["section"] == "accommodations"
    assert result["removed_count"] == 2
    assert result["remaining"] == {"places": 1, "events": 1, "accommodations": 0}

    remaining = get_favorites("user-1", destination="Paris")["favorites"][0]
    assert [p["name"] for p in remaining["places"]] == ["Louvre"]
    assert [e["name"] for e in remaining["events"]] == ["Opera Garnier"]
    assert remaining["accommodations"] == []
    # The trip itself survives — only one section was named.
    assert len(fake_collection.docs) == 1


def test_removing_specific_items_leaves_the_rest_of_the_section(fake_collection):
    save_favorite(
        "user-1",
        "Paris",
        accommodations=[{"hotel_id": "PH1", "name": "Hotel Lumiere"}, {"hotel_id": "PH2", "name": "Le Marais Hostel"}],
    )
    result = remove_favorite_section("user-1", "Paris", "accommodations", item_ids=["PH1"])

    assert result["removed_count"] == 1
    remaining = get_favorites("user-1", destination="Paris")["favorites"][0]
    assert [s["name"] for s in remaining["accommodations"]] == ["Le Marais Hostel"]


@pytest.mark.parametrize(
    "word,expected",
    [
        ("hotels", "accommodations"),
        ("hostels", "accommodations"),
        ("cabins", "accommodations"),
        ("stays", "accommodations"),
        ("bookings", "accommodations"),
        ("events", "events"),
        ("activities", "events"),
        ("concerts", "events"),
        ("places", "places"),
        ("attractions", "places"),
        ("sights", "places"),
        ("flights", None),
    ],
)
def test_user_words_map_onto_the_right_section(word, expected):
    assert resolve_section(word) == expected


def test_an_unknown_section_is_refused_rather_than_guessed(fake_collection):
    save_favorite("user-1", "Paris", places=[{"place_id": "P1", "name": "Louvre"}])
    with pytest.raises(ValueError, match="not part of a saved trip"):
        remove_favorite_section("user-1", "Paris", "flights")


def test_deleting_a_whole_trip_removes_it(fake_collection):
    save_favorite("user-1", "Paris", places=[{"place_id": "P1", "name": "Louvre"}])
    result = delete_favorite("user-1", "Paris")
    assert result["action"] == "deleted"
    assert fake_collection.docs == []


def test_deleting_a_trip_that_does_not_exist_is_reported_not_raised(fake_collection):
    assert delete_favorite("user-1", "Atlantis")["action"] == "not_found"


# --------------------------------------------------------------------------- #
# Guardrail 3 in practice
# --------------------------------------------------------------------------- #


def test_user_1_cannot_read_user_2s_trips(fake_collection):
    save_favorite("user-2", "Rome", places=[{"place_id": "R1", "name": "Colosseum"}])
    assert get_favorites("user-1") == {"count": 0, "favorites": []}


def test_user_1_cannot_delete_user_2s_trip(fake_collection):
    save_favorite("user-2", "Rome", places=[{"place_id": "R1", "name": "Colosseum"}])
    assert delete_favorite("user-1", "Rome")["action"] == "not_found"
    assert len(fake_collection.docs) == 1  # user-2's trip is untouched


def test_user_1_cannot_strip_a_section_from_user_2s_trip(fake_collection):
    save_favorite("user-2", "Rome", accommodations=[{"hotel_id": "RH1", "name": "Hotel Roma"}])
    assert remove_favorite_section("user-1", "Rome", "hotels")["action"] == "not_found"
    assert len(fake_collection.docs[0]["accommodations"]) == 1


def test_two_users_keep_separate_trips_for_the_same_destination(fake_collection):
    save_favorite("user-1", "Kyoto", places=[{"place_id": "W1", "name": "Fushimi Inari"}])
    save_favorite("user-2", "Kyoto", places=[{"place_id": "W2", "name": "Kinkaku-ji"}])

    assert len(fake_collection.docs) == 2
    assert [p["name"] for p in get_favorites("user-1")["favorites"][0]["places"]] == ["Fushimi Inari"]
    assert [p["name"] for p in get_favorites("user-2")["favorites"][0]["places"]] == ["Kinkaku-ji"]


def test_an_unauthenticated_call_is_refused_before_any_query(fake_collection):
    for operation in (
        lambda: get_favorites(""),
        lambda: save_favorite("", "Kyoto"),
        lambda: delete_favorite(None, "Kyoto"),
        lambda: remove_favorite_section("", "Kyoto", "hotels"),
        lambda: update_favorite("", "Kyoto"),
    ):
        with pytest.raises(AuthorizationError):
            operation()


# --------------------------------------------------------------------------- #
# Updates
# --------------------------------------------------------------------------- #


def test_update_merges_by_default_and_leaves_other_sections_alone(fake_collection):
    save_favorite("user-1", "Kyoto", places=KYOTO_PLACES, events=KYOTO_EVENTS)
    result = update_favorite("user-1", "Kyoto", places=[{"place_id": "W3", "name": "Arashiyama"}])

    assert len(result["favorite"]["places"]) == 3
    assert len(result["favorite"]["events"]) == 1


def test_update_can_replace_a_section_outright(fake_collection):
    save_favorite("user-1", "Kyoto", places=KYOTO_PLACES, events=KYOTO_EVENTS)
    result = update_favorite(
        "user-1",
        "Kyoto",
        places=[{"place_id": "W3", "name": "Arashiyama"}],
        replace_sections=True,
    )

    assert [p["name"] for p in result["favorite"]["places"]] == ["Arashiyama"]
    assert len(result["favorite"]["events"]) == 1  # not mentioned, so not touched


def test_reading_one_section_returns_only_that_section(fake_collection):
    save_favorite("user-1", "Kyoto", places=KYOTO_PLACES, events=KYOTO_EVENTS, accommodations=KYOTO_STAYS)
    result = get_favorites("user-1", destination="Kyoto", section="hotels")

    favorite = result["favorites"][0]
    assert "accommodations" in favorite
    assert "places" not in favorite
