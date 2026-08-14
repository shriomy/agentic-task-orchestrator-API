"""Graph plumbing: tool chaining, HIL interrupts, and state across turns.

The model is scripted, so these tests check the machinery — that a pause
actually pauses, that resuming carries the picks forward, that two pauses in one
turn both work, and that an abandoned pick can be answered many messages later.
They do not test the model's judgment about which tool to reach for; that lives
in the system prompt and is checked separately in test_behaviour.py.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.graph.graph import build_graph, route_after_agent, route_after_preprocess
from src.graph.state import GraphState, Selection, SelectionOption, merge_selections
from src.guardrails.output import RequestSplit
from src.guardrails.scope import ScopeVerdict

from helpers import ScriptedModel

USER = "user-1"
CONFIG_BASE = {"auth_user_id": USER}


def tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


@pytest.fixture
def harness(monkeypatch, no_memory):
    """Build a graph whose model is scripted and whose travel APIs are stubbed."""
    calls: list[tuple[str, dict]] = []

    def stub(name: str, payload):
        def inner(**kwargs):
            calls.append((name, kwargs))
            return payload(**kwargs) if callable(payload) else payload

        return inner

    monkeypatch.setattr(
        "src.tools.destinations.search_destinations",
        lambda query, **kw: calls.append(("web_search", {"query": query}))
        or {"results": [{"title": "Tromso"}, {"title": "Bergen"}]},
    )
    monkeypatch.setattr(
        "src.graph.tools.search_destinations",
        lambda query, **kw: calls.append(("web_search", {"query": query}))
        or {"results": [{"title": "Tromso"}, {"title": "Bergen"}]},
    )
    monkeypatch.setattr(
        "src.graph.tools.search_places",
        stub("search_places", lambda **kw: {"destination": kw.get("city"), "places": [{"place_id": "W1", "name": "A place"}]}),
    )
    monkeypatch.setattr(
        "src.graph.tools.search_events",
        stub("search_events", lambda **kw: {"destination": kw.get("city"), "events": [{"event_id": "E1", "name": "A gig"}]}),
    )
    monkeypatch.setattr(
        "src.graph.tools.search_accommodations",
        stub("search_accommodations", lambda **kw: {"destination": kw.get("location"), "accommodations": [{"hotel_id": "H1", "name": "A hotel"}]}),
    )

    # Guardrails: let everything through so the plumbing is what is under test.
    monkeypatch.setattr(
        "src.graph.nodes.classify_scope",
        lambda message, context="": ScopeVerdict(label="in_scope", reason="test"),
    )
    monkeypatch.setattr(
        "src.graph.nodes.split_request",
        lambda message: RequestSplit(allowed_request=message, withheld=[]),
    )

    def make(script: list) -> tuple:
        graph = build_graph(checkpointer=InMemorySaver())
        model = ScriptedModel(script)
        for node in graph.nodes.values():
            runnable = getattr(node, "runnable", None) or getattr(node, "bound", None)
            target = getattr(runnable, "func", runnable)
            if type(target).__name__ == "AgentNode":
                target._model = model
        return graph, model, calls

    return make


def _patch_agent(graph, model):
    """Force every AgentNode in the compiled graph to use the scripted model."""
    from src.graph.nodes import AgentNode

    patched = 0
    for node in graph.nodes.values():
        for attribute in ("runnable", "bound", "func"):
            candidate = getattr(node, attribute, None)
            for inner in (candidate, getattr(candidate, "func", None)):
                if isinstance(inner, AgentNode):
                    inner._model = model
                    patched += 1
    return patched


# --------------------------------------------------------------------------- #
# Routers
# --------------------------------------------------------------------------- #


def test_scope_router_sends_each_verdict_to_its_own_branch():
    assert route_after_preprocess(GraphState(scope="out_of_scope")) == "out_of_scope"
    assert route_after_preprocess(GraphState(scope="smalltalk")) == "smalltalk"
    assert route_after_preprocess(GraphState(scope="in_scope")) == "summarize"


def test_agent_router_loops_only_while_tools_are_requested():
    with_tools = GraphState(messages=[AIMessage(content="", tool_calls=[tool_call("web_search", {"query": "x"}, "c1")])])
    assert route_after_agent(with_tools) == "tools"

    plain = GraphState(messages=[AIMessage(content="Here you go.")])
    assert route_after_agent(plain) == "finalize"


def test_agent_router_sends_back_an_agent_that_skipped_a_requested_pause():
    state = GraphState(
        messages=[AIMessage(content="Here are all five, plus everything in them.")],
        wants_selection=True,
        tool_round_count=1,
        turn_index=1,
    )
    assert route_after_agent(state) == "require_selection"


def test_the_pause_nudge_is_bounded_so_a_turn_cannot_hang():
    from src.graph.nodes import RequireSelectionNode

    state = GraphState(
        messages=[AIMessage(content="Still not pausing.")],
        wants_selection=True,
        tool_round_count=1,
        turn_index=1,
        selection_nudges=RequireSelectionNode.MAX_NUDGES,
    )
    assert route_after_agent(state) == "finalize"


def test_no_nudge_before_any_lookup_has_run():
    """There is nothing to offer as options until a tool has returned something."""
    state = GraphState(
        messages=[AIMessage(content="Which country did you have in mind?")],
        wants_selection=True,
        tool_round_count=0,
        turn_index=1,
    )
    assert route_after_agent(state) == "finalize"


def test_no_nudge_once_a_pause_has_already_happened_this_turn():
    state = GraphState(
        messages=[AIMessage(content="Here's what to see in Bergen.")],
        wants_selection=True,
        tool_round_count=2,
        turn_index=3,
        selections=[Selection(turn_index=3, status="answered", picked_ids=["o1"])],
    )
    assert route_after_agent(state) == "finalize"


def test_a_pause_answered_on_an_earlier_turn_does_not_count_for_this_one():
    state = GraphState(
        messages=[AIMessage(content="Here you go.")],
        wants_selection=True,
        tool_round_count=1,
        turn_index=5,
        selections=[Selection(turn_index=2, status="answered", picked_ids=["o1"])],
    )
    assert route_after_agent(state) == "require_selection"


def test_agent_router_stops_at_the_tool_budget():
    from src.config import settings

    state = GraphState(
        messages=[AIMessage(content="", tool_calls=[tool_call("web_search", {"query": "x"}, "c1")])],
        tool_round_count=settings.max_tool_rounds,
    )
    assert route_after_agent(state) == "finalize"


# --------------------------------------------------------------------------- #
# Selections reducer — the mechanism behind re-picking
# --------------------------------------------------------------------------- #


def test_selections_upsert_by_id_rather_than_appending():
    pending = Selection(
        selection_id="sel_a",
        options=[SelectionOption(id="o1", label="Tromso"), SelectionOption(id="o2", label="Bergen")],
    )
    answered = pending.model_copy(deep=True)
    answered.resolve(["o2"])

    merged = merge_selections([pending], [answered])

    assert len(merged) == 1
    assert merged[0].status == "answered"
    assert merged[0].picked_ids == ["o2"]


def test_a_selection_resolves_from_labels_as_well_as_ids():
    selection = Selection(options=[SelectionOption(id="o1", label="Tromso"), SelectionOption(id="o2", label="Bergen")])
    selection.resolve(["bergen"])
    assert selection.picked_ids == ["o2"]


def test_picking_nothing_marks_the_selection_abandoned_not_answered():
    selection = Selection(options=[SelectionOption(id="o1", label="Tromso")])
    selection.resolve([])
    assert selection.status == "abandoned"
    assert selection.skipped_ids == ["o1"]


# --------------------------------------------------------------------------- #
# End-to-end turns
# --------------------------------------------------------------------------- #


def test_a_plain_answer_finishes_the_turn(harness):
    graph, model, _ = harness([AIMessage(content="Norway is lovely in winter.")])
    _patch_agent(graph, model)

    result = graph.invoke(
        {"thread_id": "t1", "user_id": USER, "message": "Tell me about Norway"},
        config={"configurable": {"thread_id": "t1", **CONFIG_BASE}},
    )
    assert result["bot_response"] == "Norway is lovely in winter."


def test_two_tools_chain_in_one_turn_without_pausing(harness):
    """'Shanghai's best places along with events next week' — no HIL."""
    graph, model, calls = harness(
        [
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("search_places", {"city": "Shanghai"}, "c1"),
                    tool_call("search_events", {"city": "Shanghai"}, "c2"),
                ],
            ),
            AIMessage(content="Here are Shanghai's highlights and what's on."),
        ]
    )
    _patch_agent(graph, model)

    result = graph.invoke(
        {"thread_id": "t2", "user_id": USER, "message": "Shanghai's best places plus events next week"},
        config={"configurable": {"thread_id": "t2", **CONFIG_BASE}},
    )

    assert "__interrupt__" not in result
    assert [name for name, _ in calls] == ["search_places", "search_events"]
    assert result["bot_response"] == "Here are Shanghai's highlights and what's on."


def test_hil_pauses_then_continues_with_only_the_picked_destination(harness):
    """'top 5 in Norway, let me pick some' -> search, pause, then places for the pick."""
    graph, model, calls = harness(
        [
            AIMessage(content="", tool_calls=[tool_call("web_search", {"query": "top destinations Norway"}, "c1")]),
            AIMessage(
                content="",
                tool_calls=[
                    tool_call(
                        "request_user_selection",
                        {
                            "prompt": "Which of these should I dig into?",
                            "kind": "destination",
                            "options": [
                                {"id": "o1", "label": "Tromso", "payload": {"name": "Tromso"}},
                                {"id": "o2", "label": "Bergen", "payload": {"name": "Bergen"}},
                            ],
                        },
                        "c2",
                    )
                ],
            ),
            AIMessage(content="", tool_calls=[tool_call("search_places", {"city": "Bergen"}, "c3")]),
            AIMessage(content="Here's what to see in Bergen."),
        ]
    )
    _patch_agent(graph, model)
    config = {"configurable": {"thread_id": "t3", **CONFIG_BASE}}

    paused = graph.invoke(
        {"thread_id": "t3", "user_id": USER, "message": "top 5 in Norway, let me pick some"},
        config=config,
    )

    assert "__interrupt__" in paused
    payload = paused["__interrupt__"][0].value
    assert [o["label"] for o in payload["options"]] == ["Tromso", "Bergen"]
    # The pause happened before any place lookup ran.
    assert [name for name, _ in calls] == ["web_search"]

    resumed = graph.invoke(Command(resume={"selected_options": ["o2"]}), config=config)

    place_lookups = [args for name, args in calls if name == "search_places"]
    assert len(place_lookups) == 1
    assert place_lookups[0]["city"] == "Bergen"  # only the picked one
    assert resumed["bot_response"] == "Here's what to see in Bergen."

    selection = resumed["selections"][0]
    assert selection.status == "answered"
    assert [option.label for option in selection.picked] == ["Bergen"]


def test_two_pauses_in_a_single_turn(harness):
    """Pick places, then pick which of those to price hotels for."""
    graph, model, calls = harness(
        [
            AIMessage(content="", tool_calls=[tool_call("web_search", {"query": "places in Thailand"}, "c1")]),
            AIMessage(
                content="",
                tool_calls=[
                    tool_call(
                        "request_user_selection",
                        {
                            "prompt": "Which places interest you?",
                            "kind": "place",
                            "options": [
                                {"id": "p1", "label": "Chiang Mai"},
                                {"id": "p2", "label": "Krabi"},
                                {"id": "p3", "label": "Phuket"},
                            ],
                        },
                        "c2",
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    tool_call(
                        "request_user_selection",
                        {
                            "prompt": "Hostel or hotel?",
                            "kind": "accommodation",
                            "options": [{"id": "s1", "label": "Hostels"}, {"id": "s2", "label": "Hotels"}],
                        },
                        "c3",
                    )
                ],
            ),
            AIMessage(content="", tool_calls=[tool_call("search_accommodations", {"location": "Krabi", "stay_type": "hostel"}, "c4")]),
            AIMessage(content="Here are hostels in Krabi."),
        ]
    )
    _patch_agent(graph, model)
    config = {"configurable": {"thread_id": "t4", **CONFIG_BASE}}

    first = graph.invoke(
        {"thread_id": "t4", "user_id": USER, "message": "places in Thailand so I can choose a few for accommodation"},
        config=config,
    )
    assert "__interrupt__" in first
    assert first["__interrupt__"][0].value["kind"] == "place"

    second = graph.invoke(Command(resume={"selected_options": ["p2"]}), config=config)
    assert "__interrupt__" in second, "the turn should pause a second time"
    assert second["__interrupt__"][0].value["kind"] == "accommodation"

    final = graph.invoke(Command(resume={"selected_options": ["s1"]}), config=config)

    assert final["bot_response"] == "Here are hostels in Krabi."
    assert len(final["selections"]) == 2
    assert all(selection.status == "answered" for selection in final["selections"])
    stay_lookups = [args for name, args in calls if name == "search_accommodations"]
    assert stay_lookups[0]["location"] == "Krabi"


def test_an_agent_that_skips_a_requested_pause_is_forced_to_pause(harness):
    """End-to-end: the model answers straight away, the graph makes it pause anyway.

    Prompting alone did not achieve this reliably in live runs, so the behaviour
    is enforced by the router.
    """
    graph, model, _ = harness(
        [
            AIMessage(content="", tool_calls=[tool_call("web_search", {"query": "top destinations Norway"}, "c1")]),
            # Skips the pause and answers everything.
            AIMessage(content="Here are all five, and everything to see in each."),
            # After the nudge, it complies.
            AIMessage(
                content="",
                tool_calls=[
                    tool_call(
                        "request_user_selection",
                        {
                            "prompt": "Which should I dig into?",
                            "kind": "destination",
                            "options": [{"id": "o1", "label": "Tromso"}, {"id": "o2", "label": "Bergen"}],
                        },
                        "c2",
                    )
                ],
            ),
            AIMessage(content="Here's what to see in Bergen."),
        ]
    )
    _patch_agent(graph, model)
    config = {"configurable": {"thread_id": "t14", **CONFIG_BASE}}

    result = graph.invoke(
        {"thread_id": "t14", "user_id": USER, "message": "top 5 in Norway and let me pick some of them"},
        config=config,
    )

    assert "__interrupt__" in result, "the graph should have forced a pause"
    assert result["selection_nudges"] == 1
    # The skipped-over draft must not have leaked out as the reply.
    assert not result.get("bot_response")

    final = graph.invoke(Command(resume={"selected_options": ["o2"]}), config=config)
    assert final["bot_response"] == "Here's what to see in Bergen."


def test_the_forced_pause_gives_up_rather_than_looping_forever(harness):
    """A model that never complies must still produce an answer."""
    graph, model, _ = harness(
        [
            AIMessage(content="", tool_calls=[tool_call("web_search", {"query": "x"}, "c1")]),
            AIMessage(content="Refusing to pause, attempt 1."),
            AIMessage(content="Refusing to pause, attempt 2."),
            AIMessage(content="Refusing to pause, attempt 3."),
        ]
    )
    _patch_agent(graph, model)

    result = graph.invoke(
        {"thread_id": "t15", "user_id": USER, "message": "top 5 in Norway, let me pick"},
        config={"configurable": {"thread_id": "t15", **CONFIG_BASE}},
    )

    assert "__interrupt__" not in result
    assert result["selection_nudges"] == 2  # capped
    assert result["bot_response"], "the turn must still answer rather than hang"


@pytest.mark.parametrize(
    "message,expected",
    [
        ("top 5 in Norway and let me pick some of them", True),
        ("places in Thailand so I can choose a few for accommodation", True),
        ("give me options for hotels in Rome", True),
        ("ask me which ones I want", True),
        ("Shanghai's best places to visit along with events next week", False),
        ("what are the places to visit in Moscow", False),
        ("events in Paris this weekend", False),
    ],
)
def test_selection_intent_is_detected_from_the_users_wording(harness, message, expected):
    graph, model, _ = harness([AIMessage(content="ok")])
    _patch_agent(graph, model)

    result = graph.invoke(
        {"thread_id": f"t-cue-{abs(hash(message))}", "user_id": USER, "message": message},
        config={"configurable": {"thread_id": f"t-cue-{abs(hash(message))}", **CONFIG_BASE}},
    )
    assert result["wants_selection"] is expected


def test_picking_nothing_does_not_silently_continue_with_everything(harness):
    graph, model, calls = harness(
        [
            AIMessage(
                content="",
                tool_calls=[
                    tool_call(
                        "request_user_selection",
                        {
                            "prompt": "Which ones?",
                            "kind": "destination",
                            "options": [{"id": "o1", "label": "Tromso"}, {"id": "o2", "label": "Bergen"}],
                        },
                        "c1",
                    )
                ],
            ),
            AIMessage(content="No problem — want me to cover all of them instead?"),
        ]
    )
    _patch_agent(graph, model)
    config = {"configurable": {"thread_id": "t5", **CONFIG_BASE}}

    graph.invoke({"thread_id": "t5", "user_id": USER, "message": "let me pick"}, config=config)
    result = graph.invoke(Command(resume={"selected_options": []}), config=config)

    selection = result["selections"][0]
    assert selection.status == "abandoned"
    # The tool told the model not to proceed with all of them.
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "picked nothing" in tool_messages[-1].content
    assert not [name for name, _ in calls if name == "search_places"]


def test_an_abandoned_selection_can_be_answered_many_messages_later(harness):
    """The user drops a pick, chats on, then comes back to it."""
    graph, model, calls = harness(
        [
            AIMessage(
                content="",
                tool_calls=[
                    tool_call(
                        "request_user_selection",
                        {
                            "prompt": "Which Norway destinations?",
                            "kind": "destination",
                            "options": [{"id": "o1", "label": "Tromso"}, {"id": "o2", "label": "Bergen"}],
                        },
                        "c1",
                    )
                ],
            ),
            AIMessage(content="Fine — tell me when you'd like to choose."),
            AIMessage(content="Paris has plenty on this month."),
            AIMessage(content="", tool_calls=[tool_call("search_places", {"city": "Tromso"}, "c9")]),
            AIMessage(content="Back to Tromso, then — here's what to see."),
        ]
    )
    _patch_agent(graph, model)
    config = {"configurable": {"thread_id": "t6", **CONFIG_BASE}}

    graph.invoke({"thread_id": "t6", "user_id": USER, "message": "Norway, let me pick"}, config=config)
    abandoned = graph.invoke(Command(resume={"selected_options": []}), config=config)
    selection_id = abandoned["selections"][0].selection_id
    assert abandoned["selections"][0].status == "abandoned"

    # An unrelated turn in between.
    graph.invoke({"thread_id": "t6", "user_id": USER, "message": "what about Paris"}, config=config)

    # Now the user circles back. The old selection is still in state, and the
    # context digest tells the agent it can be revisited.
    from src.graph.context import selection_context

    state = graph.get_state(config).values
    digest = selection_context(GraphState.model_validate(state))
    assert selection_id in digest
    assert "ABANDONED" in digest
    assert "revisited" in digest or "re-pick" in digest

    revisit = graph.invoke(
        {"thread_id": "t6", "user_id": USER, "message": "actually go with Tromso from earlier"},
        config=config,
    )
    assert revisit["bot_response"] == "Back to Tromso, then — here's what to see."
    # State was never duplicated by the intervening turns.
    assert len(revisit["selections"]) == 1


def test_state_carries_across_turns_and_the_turn_index_advances(harness):
    graph, model, _ = harness(
        [AIMessage(content="First answer."), AIMessage(content="Second answer.")]
    )
    _patch_agent(graph, model)
    config = {"configurable": {"thread_id": "t7", **CONFIG_BASE}}

    first = graph.invoke({"thread_id": "t7", "user_id": USER, "message": "one"}, config=config)
    second = graph.invoke({"thread_id": "t7", "user_id": USER, "message": "two"}, config=config)

    assert first["turn_index"] == 1
    assert second["turn_index"] == 2
    human = [m for m in second["messages"] if isinstance(m, HumanMessage)]
    assert [str(m.content) for m in human] == ["one", "two"]


def test_a_tool_failure_becomes_a_message_the_agent_can_recover_from(harness, monkeypatch):
    from src.tools.http import ToolError

    graph, model, _ = harness(
        [
            AIMessage(content="", tool_calls=[tool_call("search_events", {"city": "Paris"}, "c1")]),
            AIMessage(content="I couldn't reach the events listings just now."),
        ]
    )
    _patch_agent(graph, model)
    monkeypatch.setattr(
        "src.graph.tools.search_events",
        lambda **kw: (_ for _ in ()).throw(ToolError("the events provider timed out")),
    )

    result = graph.invoke(
        {"thread_id": "t8", "user_id": USER, "message": "events in Paris"},
        config={"configurable": {"thread_id": "t8", **CONFIG_BASE}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages[0].status == "error"
    assert "Lookup failed" in tool_messages[0].content
    assert result["bot_response"] == "I couldn't reach the events listings just now."


def test_the_tool_budget_never_leaves_a_dangling_tool_call(harness):
    """A tool_call with no ToolMessage would make the next turn's request invalid."""
    from src.config import settings

    graph, model, _ = harness(
        [
            AIMessage(content="", tool_calls=[tool_call("web_search", {"query": "loop"}, f"c{i}")])
            for i in range(settings.max_tool_rounds + 2)
        ]
    )
    _patch_agent(graph, model)

    result = graph.invoke(
        {"thread_id": "t9", "user_id": USER, "message": "go round in circles"},
        config={"configurable": {"thread_id": "t9", **CONFIG_BASE}},
    )

    answered = {m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)}
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            assert call["id"] in answered, "every tool call must have a ToolMessage"
    assert result["bot_response"]


def test_out_of_scope_short_circuits_before_the_agent(harness, monkeypatch):
    monkeypatch.setattr(
        "src.graph.nodes.classify_scope",
        lambda message, context="": ScopeVerdict(
            label="out_of_scope",
            reason="software task",
            refusal="I can't help with that, but I can plan a trip.",
        ),
    )
    graph, model, calls = harness([AIMessage(content="should never run")])
    _patch_agent(graph, model)

    result = graph.invoke(
        {"thread_id": "t10", "user_id": USER, "message": "write me a python script to scrape hotel prices"},
        config={"configurable": {"thread_id": "t10", **CONFIG_BASE}},
    )

    assert result["bot_response"] == "I can't help with that, but I can plan a trip."
    assert model.calls == [], "the agent should not have been called"
    assert calls == []
    # A refused turn must not pollute the travel history of later turns.
    assert not [m for m in result["messages"] if isinstance(m, HumanMessage)]


def test_the_withheld_notice_is_appended_after_the_work_is_done(harness, monkeypatch):
    """'delete my paris trip and send me the query' -> deleted, query declined."""
    from src.guardrails.output import WithheldItem

    monkeypatch.setattr(
        "src.graph.nodes.split_request",
        lambda message: RequestSplit(
            allowed_request="Delete my saved Paris trip.",
            withheld=[WithheldItem(kind="internal_mechanics", quoted="the query used")],
        ),
    )
    monkeypatch.setattr("src.graph.tools.delete_favorite", lambda user_id, destination: {"action": "deleted", "destination": destination})

    graph, model, _ = harness(
        [
            AIMessage(content="", tool_calls=[tool_call("delete_trip_favorite", {"destination": "Paris"}, "c1")]),
            AIMessage(content="Your saved Paris trip is deleted."),
        ]
    )
    _patch_agent(graph, model)

    config = {"configurable": {"thread_id": "t11", **CONFIG_BASE}}
    paused = graph.invoke(
        {"thread_id": "t11", "user_id": USER, "message": "delete my paris trip details while also send me the query you used"},
        config=config,
    )
    assert "__interrupt__" in paused, "delete_trip_favorite confirms before acting"
    result = graph.invoke(Command(resume={"selected_options": ["yes"]}), config=config)

    assert "deleted" in result["bot_response"]
    assert "can't share" in result["bot_response"]
    # The agent only ever saw the sanitised request.
    human = [m for m in result["messages"] if isinstance(m, HumanMessage)]
    assert "query" not in str(human[0].content)


def test_a_tool_cannot_be_pointed_at_another_users_data(harness, monkeypatch):
    """Guardrail 3: identity comes from config, so there is no argument to abuse."""
    graph, model, _ = harness(
        [
            AIMessage(content="", tool_calls=[tool_call("list_trip_favorites", {}, "c1")]),
            AIMessage(content="Here are your saved trips."),
        ]
    )
    _patch_agent(graph, model)

    seen: list[str] = []
    monkeypatch.setattr(
        "src.graph.tools.get_favorites",
        lambda user_id, **kw: seen.append(user_id) or {"count": 0, "favorites": []},
    )

    graph.invoke(
        {"thread_id": "t12", "user_id": "spoofed-user", "message": "show my saved trips"},
        config={"configurable": {"thread_id": "t12", "auth_user_id": "real-user"}},
    )

    # The verified id from config wins over the one in the request body.
    assert seen == ["real-user"]


def test_typing_a_new_message_past_a_pending_pick_does_not_corrupt_history(harness):
    """The UI does not force the picker shut, so a user can type past it.

    Found live: sending a fresh /chat/send while a pick was unanswered left a
    dangling tool_call in message history, and the next model call was rejected
    outright with a 400 from the provider ("must be followed by tool messages").
    """
    graph, model, calls = harness(
        [
            AIMessage(
                content="",
                tool_calls=[
                    tool_call(
                        "request_user_selection",
                        {
                            "prompt": "Which Norway destinations?",
                            "kind": "destination",
                            "options": [{"id": "o1", "label": "Tromso"}, {"id": "o2", "label": "Bergen"}],
                        },
                        "c1",
                    )
                ],
            ),
            AIMessage(content="Here's what to see in Rome."),
        ]
    )
    _patch_agent(graph, model)
    config = {"configurable": {"thread_id": "t16", **CONFIG_BASE}}

    graph.invoke({"thread_id": "t16", "user_id": USER, "message": "Norway, let me pick"}, config=config)

    # Instead of resuming, the user types something unrelated.
    overridden = graph.invoke(
        {"thread_id": "t16", "user_id": USER, "message": "actually never mind, tell me about Rome"},
        config=config,
    )

    assert overridden["bot_response"] == "Here's what to see in Rome."
    tool_messages = [m for m in overridden["messages"] if isinstance(m, ToolMessage)]
    assert any(m.tool_call_id == "c1" for m in tool_messages), "the dangling call must be closed"
    selection = overridden["selections"][0]
    assert selection.status == "abandoned"

    # And the thread keeps working normally afterward — no corrupted history left over.
    followup = graph.invoke({"thread_id": "t16", "user_id": USER, "message": "anything else nearby?"}, config=config)
    assert followup["turn_index"] == 3


def test_typing_past_a_pick_is_closed_even_on_an_out_of_scope_follow_up(harness, monkeypatch):
    """The closer must run before the scope branch returns, not after."""
    graph, model, _ = harness(
        [
            AIMessage(
                content="",
                tool_calls=[tool_call("request_user_selection", {"prompt": "pick", "options": [{"id": "o1", "label": "A"}]}, "c1")],
            ),
        ]
    )
    _patch_agent(graph, model)
    config = {"configurable": {"thread_id": "t17", **CONFIG_BASE}}

    graph.invoke({"thread_id": "t17", "user_id": USER, "message": "let me pick"}, config=config)

    monkeypatch.setattr(
        "src.graph.nodes.classify_scope",
        lambda message, context="": ScopeVerdict(label="out_of_scope", reason="x", refusal="Out of scope."),
    )
    result = graph.invoke(
        {"thread_id": "t17", "user_id": USER, "message": "write me a scraper"},
        config=config,
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any(m.tool_call_id == "c1" for m in tool_messages)


def test_a_declined_deletion_is_reported_as_kept_not_as_done(harness, monkeypatch):
    """Regression: the tool correctly blocked the delete, but the model once
    reported it as deleted anyway — the tool's cancelled payload and the system
    prompt rule together must make that impossible to say."""
    deleted: list[str] = []
    monkeypatch.setattr(
        "src.graph.tools.delete_favorite",
        lambda user_id, destination: deleted.append(destination) or {"action": "deleted"},
    )

    graph, model, _ = harness(
        [
            AIMessage(content="", tool_calls=[tool_call("delete_trip_favorite", {"destination": "Paris"}, "c1")]),
            AIMessage(content="Your Paris trip has not been deleted and is still saved."),
        ]
    )
    _patch_agent(graph, model)
    config = {"configurable": {"thread_id": "t18", **CONFIG_BASE}}

    graph.invoke({"thread_id": "t18", "user_id": USER, "message": "delete my paris trip"}, config=config)
    result = graph.invoke(Command(resume={"selected_options": ["no"]}), config=config)

    assert deleted == [], "declining must not touch the database"
    assert "not" in result["bot_response"].lower()
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "NOT deleted" in tool_messages[-1].content


def test_deleting_a_favorite_pauses_for_confirmation_before_touching_the_db(harness, monkeypatch):
    deleted: list[str] = []
    monkeypatch.setattr(
        "src.graph.tools.delete_favorite",
        lambda user_id, destination: deleted.append(destination) or {"action": "deleted"},
    )

    graph, model, _ = harness(
        [
            AIMessage(content="", tool_calls=[tool_call("delete_trip_favorite", {"destination": "Paris"}, "c1")]),
            AIMessage(content="Deleted your Paris trip."),
        ]
    )
    _patch_agent(graph, model)
    config = {"configurable": {"thread_id": "t19", **CONFIG_BASE}}

    paused = graph.invoke({"thread_id": "t19", "user_id": USER, "message": "delete my paris trip"}, config=config)

    assert "__interrupt__" in paused
    payload = paused["__interrupt__"][0].value
    assert payload["kind"] == "confirmation"
    assert deleted == [], "must not delete before the user confirms"

    result = graph.invoke(Command(resume={"selected_options": ["yes"]}), config=config)
    assert deleted == ["Paris"]
    assert result["bot_response"] == "Deleted your Paris trip."


def test_a_tool_call_with_no_verified_identity_is_refused(harness):
    graph, model, _ = harness(
        [
            AIMessage(content="", tool_calls=[tool_call("list_trip_favorites", {}, "c1")]),
            AIMessage(content="I need you to be signed in for that."),
        ]
    )
    _patch_agent(graph, model)

    result = graph.invoke(
        {"thread_id": "t13", "user_id": "", "message": "show my saved trips"},
        config={"configurable": {"thread_id": "t13"}},  # no auth_user_id
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages[0].status == "error"
    assert "your own saved trips" in tool_messages[0].content
