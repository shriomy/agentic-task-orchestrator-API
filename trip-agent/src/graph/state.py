"""Graph state — the thread-scoped scope.

Persisted automatically by the checkpointer against thread_id. Two reducers do
the real work:

`add_messages`      appends and de-duplicates chat history by message id.
`merge_selections`  upserts selections by selection_id instead of appending, so
                    a selection the user abandoned twenty messages ago can be
                    answered later and updates in place rather than creating a
                    second copy.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

SelectionKind = Literal["destination", "place", "event", "accommodation", "confirmation"]
SelectionStatus = Literal["pending", "answered", "abandoned"]


class SelectionOption(BaseModel):
    """One thing the user can pick.

    `payload` carries the full upstream object (place, event or hotel) so that a
    later "save these as favorites" writes real data to MongoDB rather than a
    label the model had to reconstruct from memory.
    """

    id: str
    label: str
    description: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class Selection(BaseModel):
    selection_id: str = Field(default_factory=lambda: f"sel_{uuid.uuid4().hex[:12]}")
    kind: SelectionKind = "destination"
    prompt: str = ""
    destination: str | None = None
    options: list[SelectionOption] = Field(default_factory=list)
    picked_ids: list[str] = Field(default_factory=list)
    skipped_ids: list[str] = Field(default_factory=list)
    status: SelectionStatus = "pending"
    turn_index: int = 0
    # The tool call this selection was raised from, for traceability.
    tool_call_id: str | None = None

    @property
    def picked(self) -> list[SelectionOption]:
        wanted = set(self.picked_ids)
        return [option for option in self.options if option.id in wanted]

    def resolve(self, picked_ids: list[str]) -> None:
        """Record the user's answer, tolerating labels instead of ids."""
        valid = {option.id for option in self.options}
        by_label = {option.label.strip().lower(): option.id for option in self.options}

        resolved: list[str] = []
        for raw in picked_ids or []:
            candidate = str(raw).strip()
            if candidate in valid:
                resolved.append(candidate)
            elif candidate.lower() in by_label:
                resolved.append(by_label[candidate.lower()])

        self.picked_ids = list(dict.fromkeys(resolved))
        self.skipped_ids = [option.id for option in self.options if option.id not in set(self.picked_ids)]
        self.status = "answered" if self.picked_ids else "abandoned"


def merge_selections(
    left: list[Selection] | None,
    right: list[Selection] | None,
) -> list[Selection]:
    """Upsert selections by selection_id, preserving first-seen order.

    Answering an old selection replaces it in place; the ordering is kept stable
    so the injected context reads chronologically.
    """
    merged: dict[str, Selection] = {}
    for selection in list(left or []) + list(right or []):
        if isinstance(selection, dict):
            selection = Selection.model_validate(selection)
        merged[selection.selection_id] = selection
    return list(merged.values())


class GraphState(BaseModel):
    """Thread-scoped state. `thread_id` and `user_id` are set by the API layer
    from the verified session, never by the model."""

    model_config = {"arbitrary_types_allowed": True}

    thread_id: str = ""
    # Mirrors the authenticated user for convenience; the authoritative copy
    # lives in the runtime config so tools cannot be tricked into changing it.
    user_id: str = ""

    # ---- conversation ----------------------------------------------------
    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)
    message: str | None = None            # the raw incoming user message
    allowed_request: str | None = None    # after the output guardrail's request split
    turn_index: int = 0
    tool_round_count: int = 0

    # ---- selections (HIL) ------------------------------------------------
    selections: Annotated[list[Selection], merge_selections] = Field(default_factory=list)
    # Set when the user's phrasing implies they want to choose ("let me pick").
    wants_selection: bool = False
    # How many times this turn the agent has been sent back to raise the pause it
    # skipped. Bounded, so a model that keeps refusing cannot loop forever.
    selection_nudges: int = 0
    # A one-shot instruction injected into the next agent call. Held in state
    # rather than pushed into `messages` so it never becomes fake chat history.
    pending_directive: str | None = None

    # ---- tool retrieval ----------------------------------------------------
    # Tool names bound to the model this turn, set once by PreprocessNode via
    # semantic retrieval and reused for every agent<->tools round within the
    # turn (see graph/tool_retrieval.py). Empty only before preprocessing runs.
    active_tools: list[str] = Field(default_factory=list)

    # ---- usage tracking ------------------------------------------------
    # Plain replace field, not an accumulator: each node returns only ITS OWN
    # new entries for that call. main.py's streaming loop (stream_mode=
    # "updates") sums them across every chunk of one graph.stream() call, so
    # accumulation is scoped per HTTP request without needing to reset this
    # across the /chat/resume boundary. See graph/usage.py.
    usage_log: list[dict[str, Any]] = Field(default_factory=list)

    def has_selection_this_turn(self) -> bool:
        return any(selection.turn_index == self.turn_index for selection in self.selections or [])

    # ---- guardrails ------------------------------------------------------
    scope: str | None = None              # in_scope | smalltalk | out_of_scope
    scope_reason: str | None = None
    withheld_kinds: list[str] = Field(default_factory=list)

    # ---- memory / context ------------------------------------------------
    user_memory_summary: dict[str, Any] | None = None
    running_summary: Any | None = None    # langmem RunningSummary, set by the summarizer
    preferences_to_persist: dict[str, Any] = Field(default_factory=dict)

    # ---- output ----------------------------------------------------------
    bot_response: str | None = None
    last_agent_output: str | None = None
