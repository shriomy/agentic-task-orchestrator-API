"""Tool RAG — semantic tool retrieval.

The fail-open tests never touch a real Qdrant instance. The recall tests do,
and are skipped when one isn't reachable locally — they're the check that
actually matters for retrieval quality, so run them explicitly with a local
Qdrant up (`docker run -p 6333:6333 qdrant/qdrant`) before trusting the top_k
tuning in config.py.
"""

from __future__ import annotations

import pytest

from src.graph import tool_retrieval
from src.graph.tools import FAVORITE_TOOLS, SELECTION_TOOL


def test_retrieval_disabled_short_circuits_before_touching_qdrant(monkeypatch):
    monkeypatch.setattr(tool_retrieval.settings, "tool_rag_enabled", False)
    assert tool_retrieval.retrieve_relevant_tools("places to visit in Kyoto") is None


def test_empty_query_returns_none_without_a_qdrant_call(monkeypatch):
    calls = []
    monkeypatch.setattr(tool_retrieval, "_client", lambda: calls.append("called"))
    assert tool_retrieval.retrieve_relevant_tools("   ") is None
    assert calls == []


def test_a_qdrant_failure_fails_open_to_none(monkeypatch):
    def broken_client():
        raise ConnectionError("qdrant unreachable")

    monkeypatch.setattr(tool_retrieval, "_client", broken_client)
    assert tool_retrieval.retrieve_relevant_tools("places to visit in Kyoto") is None


def test_index_seeding_failure_fails_open(monkeypatch):
    def broken_client():
        raise ConnectionError("qdrant unreachable")

    monkeypatch.setattr(tool_retrieval, "_client", broken_client)
    assert tool_retrieval.ensure_tool_index() is False


def test_tool_index_ready_is_false_when_disabled(monkeypatch):
    monkeypatch.setattr(tool_retrieval.settings, "tool_rag_enabled", False)
    assert tool_retrieval.tool_index_ready() is False


def test_favorites_tools_are_indexed_as_one_atomic_unit():
    """A hit on the favorites unit must resolve to all five tool names, never a
    subset — several system-prompt rules only make sense if the model can see
    all of them together (e.g. use remove_trip_favorite_section rather than
    delete_trip_favorite for a partial removal)."""
    units = tool_retrieval._index_units()
    favorites_unit = next(u for u in units if u["key"] == tool_retrieval._FAVORITES_UNIT_ID)
    assert set(favorites_unit["names"]) == {getattr(t, "name", "") for t in FAVORITE_TOOLS}


def test_selection_tool_is_never_indexed():
    """request_user_selection is always force-bound by PreprocessNode; it must
    not be a retrievable unit that could fail to come back."""
    units = tool_retrieval._index_units()
    all_names = {name for unit in units for name in unit["names"]}
    assert SELECTION_TOOL not in all_names


@pytest.fixture
def live_qdrant():
    """Skip the recall tests when no local Qdrant is reachable."""
    try:
        client = tool_retrieval._client()
        client.collection_exists(tool_retrieval.settings.qdrant_collection)
    except Exception:
        pytest.skip("no local Qdrant reachable on qdrant_url; skipping recall check")
    if not tool_retrieval.ensure_tool_index():
        pytest.skip("could not seed the tool RAG index; skipping recall check")
    yield


@pytest.mark.usefixtures("live_qdrant")
def test_a_places_query_retrieves_search_places():
    names = tool_retrieval.retrieve_relevant_tools("what are the best places to visit in Kyoto")
    assert names is not None
    assert "search_places" in names


@pytest.mark.usefixtures("live_qdrant")
def test_a_delete_favorite_query_retrieves_the_favorites_group():
    names = tool_retrieval.retrieve_relevant_tools("delete my saved Paris trip")
    assert names is not None
    assert "delete_trip_favorite" in names
    assert "list_trip_favorites" in names  # the atomic favorites unit, not a subset
