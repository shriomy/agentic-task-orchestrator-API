"""Guardrail tests.

The two model-backed guardrails (scope, request-split) are exercised through
their pure-code halves and their fail-open behaviour; the redaction and
authorization layers are fully deterministic and tested directly.
"""

from __future__ import annotations

import pytest

from src.guardrails.authorization import (
    AuthorizationError,
    assert_owned,
    filter_owned,
    require_user_id,
    scoped_filter,
)
from src.guardrails.output import apply_output_guardrail, has_leaks, redact


# --------------------------------------------------------------------------- #
# Guardrail 2 — output redaction and the withheld notice
# --------------------------------------------------------------------------- #


def test_withheld_query_produces_success_plus_refusal():
    """The example case: the delete succeeds, the query is declined."""
    verdict = apply_output_guardrail(
        "Your saved Paris trip has been deleted.",
        ["internal_mechanics"],
    )
    assert "deleted" in verdict.text
    assert "can't share" in verdict.text
    assert "query" in verdict.text


def test_multiple_withheld_kinds_are_listed_once():
    verdict = apply_output_guardrail("Done.", ["internal_mechanics", "credentials", "credentials"])
    assert verdict.text.count("can't share") == 1
    assert "API keys" in verdict.text


def test_nothing_withheld_leaves_the_reply_alone():
    verdict = apply_output_guardrail("Here are five places in Kyoto.", [])
    assert verdict.text == "Here are five places in Kyoto."
    assert verdict.redacted is False
    assert verdict.withheld_kinds == []


@pytest.mark.parametrize(
    "leaky",
    [
        "I ran db.agent_favorites.deleteOne({'user_id': 'abc'})",
        "Used mongodb+srv://admin:hunter2@cluster0.mongodb.net/travel",
        "The key is sk-abcdefghijklmnopqrstuvwx",
        "apikey=9f8e7d6c5b4a3210",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcdefghij.klmnopqrst",
        "DELETE FROM agent_favorites WHERE user_id = 'abc';",
    ],
)
def test_internals_are_redacted(leaky: str):
    cleaned, modified = redact(leaky)
    assert modified is True
    assert has_leaks(cleaned) is False, cleaned
    for secret in ("hunter2", "sk-abcdefghijklmnopqrstuvwx", "9f8e7d6c5b4a3210"):
        assert secret not in cleaned


def test_unrequested_leak_is_disclosed_not_hidden():
    """If internals leak without being asked for, the user is still told."""
    verdict = apply_output_guardrail("Deleted. I ran db.agent_favorites.deleteOne({'user_id': 'x'})", [])
    assert verdict.redacted is True
    assert "internal_mechanics" in verdict.withheld_kinds
    assert "deleteOne" not in verdict.text


def test_legitimate_links_and_prices_survive_redaction():
    """Regression: an earlier version stripped every URL, killing booking links."""
    reply = (
        "**Grand Hyatt Tokyo** — $320/night, 4.8 stars.\n"
        "Book: https://www.booking.com/hotel/jp/grand-hyatt-tokyo.html\n"
        "Tickets: https://www.ticketmaster.com/event/G5vYZbZAdyCka"
    )
    cleaned, modified = redact(reply)
    assert modified is False
    assert "https://www.booking.com/hotel/jp/grand-hyatt-tokyo.html" in cleaned
    assert "$320/night" in cleaned


def test_tool_names_never_reach_the_user():
    cleaned, modified = redact("I called search_accommodations and then save_trip_favorite.")
    assert modified is True
    assert "search_accommodations" not in cleaned
    assert "save_trip_favorite" not in cleaned


# --------------------------------------------------------------------------- #
# Guardrail 3 — authorization
# --------------------------------------------------------------------------- #


def test_blank_identity_is_rejected():
    for blank in (None, "", "   "):
        with pytest.raises(AuthorizationError):
            require_user_id(blank, operation="get_favorites")


def test_scoped_filter_always_pins_the_owner():
    query = scoped_filter("user-1", {"destination_key": "paris"}, operation="get_favorites")
    assert query == {"destination_key": "paris", "user_id": "user-1"}


def test_scoped_filter_refuses_a_conflicting_user_id():
    """User 1 cannot smuggle user 2's id in through the extra filter."""
    with pytest.raises(AuthorizationError):
        scoped_filter("user-1", {"user_id": "user-2"}, operation="get_favorites")


def test_assert_owned_blocks_another_users_document():
    with pytest.raises(AuthorizationError):
        assert_owned({"_id": "d1", "user_id": "user-2"}, "user-1", operation="delete_favorite")


def test_assert_owned_passes_own_document():
    doc = {"_id": "d1", "user_id": "user-1"}
    assert assert_owned(doc, "user-1", operation="get_favorites") is doc


def test_filter_owned_drops_foreign_rows_on_a_list_read():
    rows = [
        {"_id": "a", "user_id": "user-1"},
        {"_id": "b", "user_id": "user-2"},
        {"_id": "c", "user_id": "user-1"},
    ]
    kept = filter_owned(rows, "user-1", operation="get_favorites")
    assert [row["_id"] for row in kept] == ["a", "c"]
