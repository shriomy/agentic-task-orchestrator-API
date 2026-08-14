"""API surface: authentication, the cross-user boundary, and SSE framing."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.auth import AuthenticatedUser, AuthError
from src.main import app, current_user

USER_1 = AuthenticatedUser(user_id="user-1", email="one@example.com")
USER_2 = AuthenticatedUser(user_id="user-2", email="two@example.com")


@pytest.fixture
def client(monkeypatch):
    """A client authenticated as user-1, with persistence stubbed out."""
    app.dependency_overrides[current_user] = lambda: USER_1
    for name in ("ensure_conversation", "record_message", "log_selection"):
        monkeypatch.setattr(f"src.main.{name}", lambda *a, **k: None)
    yield TestClient(app)
    app.dependency_overrides.clear()


def sse_events(response: Any) -> list[dict[str, Any]]:
    return [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #


def test_every_protected_endpoint_requires_a_token():
    app.dependency_overrides.clear()
    unauthenticated = TestClient(app)

    for method, path, body in [
        ("post", "/chat/send", {"message": "hi"}),
        ("post", "/chat/resume", {"thread_id": "t1"}),
        ("get", "/conversations", None),
        ("get", "/favorites", None),
        ("delete", "/favorites/Paris", None),
    ]:
        response = getattr(unauthenticated, method)(path, json=body) if body else getattr(unauthenticated, method)(path)
        assert response.status_code == 401, f"{method.upper()} {path} was not protected"


def test_health_is_open():
    assert TestClient(app).get("/health").status_code == 200


def test_an_invalid_token_is_a_401(monkeypatch):
    app.dependency_overrides.clear()

    def reject(token):
        raise AuthError("Session expired; please sign in again.")

    monkeypatch.setattr("src.main.verify_access_token", reject)
    response = TestClient(app).get("/conversations", headers={"Authorization": "Bearer nonsense"})

    assert response.status_code == 401
    assert "sign in" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# Guardrail 3 at the API boundary
# --------------------------------------------------------------------------- #


def test_acting_as_another_user_is_rejected(client):
    """A body user_id that disagrees with the token is a 403, not a silent swap."""
    response = client.post("/chat/send", json={"message": "show my trips", "user_id": USER_2.user_id})
    assert response.status_code == 403
    assert "your own account" in response.json()["detail"]


def test_favorites_are_read_with_the_token_identity_not_a_query_param(client, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr("src.main.get_favorites", lambda user_id, **kw: seen.append(user_id) or {"count": 0, "favorites": []})

    # Even with a user_id in the query string, the token's identity is used.
    assert client.get("/favorites?user_id=user-2").status_code == 200
    assert seen == ["user-1"]


def test_resuming_another_users_thread_is_forbidden(client, monkeypatch):
    class Snapshot:
        values = {"user_id": USER_2.user_id}
        next = ("tools",)

    monkeypatch.setattr("src.main.get_graph", lambda: type("G", (), {"get_state": staticmethod(lambda config: Snapshot())})())

    response = client.post("/chat/resume", json={"thread_id": "someone-elses-thread", "selected_options": ["o1"]})
    assert response.status_code == 403


def test_resuming_a_thread_that_is_not_waiting_is_a_409(client, monkeypatch):
    class Snapshot:
        values = {"user_id": USER_1.user_id}
        next = ()  # nothing pending

    monkeypatch.setattr("src.main.get_graph", lambda: type("G", (), {"get_state": staticmethod(lambda config: Snapshot())})())

    response = client.post("/chat/resume", json={"thread_id": "t1", "selected_options": ["o1"]})
    assert response.status_code == 409
    assert "waiting" in response.json()["detail"]


def test_resuming_an_unknown_thread_is_a_404(client, monkeypatch):
    class Snapshot:
        values = {}
        next = ()

    monkeypatch.setattr("src.main.get_graph", lambda: type("G", (), {"get_state": staticmethod(lambda config: Snapshot())})())
    assert client.post("/chat/resume", json={"thread_id": "nope"}).status_code == 404


# --------------------------------------------------------------------------- #
# SSE framing
# --------------------------------------------------------------------------- #


class FakeGraph:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.configs: list[Any] = []

    def stream(self, graph_input, config=None, stream_mode=None):
        self.configs.append(config)
        yield from self.chunks

    def get_state(self, config):
        return type("S", (), {"values": {"user_id": USER_1.user_id}, "next": ("tools",)})()


def test_a_plain_turn_streams_thread_then_text_then_done(client, monkeypatch):
    from langchain_core.messages import AIMessage

    graph = FakeGraph(
        [
            {"agent": {"messages": [AIMessage(content="", tool_calls=[{"name": "search_places", "args": {}, "id": "c1", "type": "tool_call"}])]}},
            {"finalize": {"bot_response": "Here are places in Kyoto."}},
        ]
    )
    monkeypatch.setattr("src.main.get_graph", lambda: graph)

    events = sse_events(client.post("/chat/send", json={"message": "places in Kyoto"}))

    assert events[0]["type"] == "thread"
    assert events[0]["thread_id"]
    assert {"type": "tool", "content": "Looking up places to visit…"} in events
    assert {"type": "text", "content": "Here are places in Kyoto."} in events
    assert events[-1] == {"type": "done"}


def test_progress_labels_never_mention_a_tool_name(client, monkeypatch):
    from langchain_core.messages import AIMessage

    graph = FakeGraph(
        [
            {
                "agent": {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {"name": "web_search", "args": {}, "id": "c1", "type": "tool_call"},
                                {"name": "search_accommodations", "args": {}, "id": "c2", "type": "tool_call"},
                                {"name": "delete_trip_favorite", "args": {}, "id": "c3", "type": "tool_call"},
                            ],
                        )
                    ]
                }
            },
            {"finalize": {"bot_response": "Done."}},
        ]
    )
    monkeypatch.setattr("src.main.get_graph", lambda: graph)

    events = sse_events(client.post("/chat/send", json={"message": "anything"}))
    progress = " ".join(event.get("content", "") for event in events if event["type"] == "tool")

    for tool_name in ("web_search", "search_accommodations", "delete_trip_favorite"):
        assert tool_name not in progress
    assert "Searching the web" in progress


def test_an_interrupt_is_streamed_in_the_uis_shape(client, monkeypatch):
    from langgraph.types import Interrupt

    interrupt = Interrupt(
        value={
            "type": "selection",
            "selection_id": "sel_abc",
            "kind": "destination",
            "destination": None,
            "reason": "Which of these should I dig into?",
            "options": [
                {"id": "o1", "label": "Tromso", "description": "Northern lights", "payload": {"name": "Tromso"}},
                {"id": "o2", "label": "Bergen", "description": "Fjords", "payload": {"name": "Bergen"}},
            ],
        }
    )
    monkeypatch.setattr("src.main.get_graph", lambda: FakeGraph([{"__interrupt__": (interrupt,)}]))

    events = sse_events(client.post("/chat/send", json={"message": "let me pick"}))
    payload = next(event for event in events if event["type"] == "interrupt")["data"]

    assert payload["reason"] == "Which of these should I dig into?"
    assert payload["selection_id"] == "sel_abc"
    assert [option["id"] for option in payload["options"]] == ["o1", "o2"]
    assert [option["label"] for option in payload["options"]] == ["Tromso", "Bergen"]
    # The upstream payload is internal and is not sent to the browser.
    assert "payload" not in payload["options"][0]
    assert events[-1] == {"type": "done"}


def test_the_verified_identity_is_put_into_the_runtime_config(client, monkeypatch):
    graph = FakeGraph([{"finalize": {"bot_response": "ok"}}])
    monkeypatch.setattr("src.main.get_graph", lambda: graph)

    client.post("/chat/send", json={"message": "hi"})

    configurable = graph.configs[0]["configurable"]
    assert configurable["auth_user_id"] == "user-1"
    assert configurable["thread_id"]


def test_a_graph_failure_becomes_an_error_frame_not_a_stack_trace(client, monkeypatch):
    class Exploding(FakeGraph):
        def stream(self, graph_input, config=None, stream_mode=None):
            raise RuntimeError("psycopg: connection refused to db.supabase.co:5432")

    monkeypatch.setattr("src.main.get_graph", lambda: Exploding([]))

    events = sse_events(client.post("/chat/send", json={"message": "hi"}))
    error = next(event for event in events if event["type"] == "error")

    assert "psycopg" not in error["message"]
    assert "supabase.co" not in error["message"]
    assert events[-1] == {"type": "done"}


def test_an_empty_message_is_rejected(client):
    assert client.post("/chat/send", json={"message": "   "}).status_code == 422


# --------------------------------------------------------------------------- #
# Favorites REST
# --------------------------------------------------------------------------- #


def test_deleting_a_section_only_touches_that_section(client, monkeypatch):
    seen: list[tuple] = []
    monkeypatch.setattr(
        "src.main.remove_favorite_section",
        lambda user_id, destination, section: seen.append((user_id, destination, section)) or {"action": "section_removed"},
    )
    monkeypatch.setattr("src.main.delete_favorite", lambda *a, **k: pytest.fail("should not delete the whole trip"))

    response = client.delete("/favorites/Paris?section=hotels")

    assert response.status_code == 200
    assert seen == [("user-1", "Paris", "hotels")]


def test_deleting_without_a_section_removes_the_whole_trip(client, monkeypatch):
    seen: list[tuple] = []
    monkeypatch.setattr(
        "src.main.delete_favorite",
        lambda user_id, destination: seen.append((user_id, destination)) or {"action": "deleted"},
    )

    assert client.delete("/favorites/Paris").status_code == 200
    assert seen == [("user-1", "Paris")]


def test_an_unknown_section_is_a_422(client, monkeypatch):
    def reject(user_id, destination, section):
        raise ValueError("'flights' is not part of a saved trip.")

    monkeypatch.setattr("src.main.remove_favorite_section", reject)
    assert client.delete("/favorites/Paris?section=flights").status_code == 422
