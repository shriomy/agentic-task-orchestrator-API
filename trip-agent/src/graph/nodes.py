"""Graph nodes.

Turn shape:

    preprocess -> tool_search -> summarize -> agent -> [tools -> agent]* -> finalize -> persist

`preprocess` runs guardrail 1 (scope) and the request-splitting half of
guardrail 2, loads cross-thread memory, and detects whether the user wants to
pick. `tool_search` decides, via one real tool call, whether this turn needs
any domain tool at all — and if so, which ones — before `agent` ever runs; a
confidently toolless turn short-circuits straight to a direct answer. `finalize`
runs the redaction half of guardrail 2. Guardrail 3 is enforced inside the
tools themselves, since that is the only place a query is issued.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

from ..config import settings
from ..guardrails.authorization import AuthorizationError
from ..guardrails.output import apply_output_guardrail, split_request
from ..guardrails.scope import classify_scope
from ..llm import build_model, fast_model
from ..memory.user_memory import (
    ALLOWED_KEYS,
    get_user_memory_summary,
    write_user_memory,
)
from ..tools.http import ToolError
from .context import build_prompt
from .state import GraphState, Selection, SelectionOption
from .tool_retrieval import retrieve_relevant_tools, search_tools
from .tools import ALL_TOOLS, SELECTION_TOOL, TOOLS_BY_NAME
from .usage import agent_usage_entry, count_tokens, other_usage_entry

logger = logging.getLogger(__name__)

# Tools that take the graph state via InjectedState. Everything else is called
# with the model's arguments alone.
_STATE_INJECTED_TOOLS = {SELECTION_TOOL, "save_trip_favorite"}

# Phrasings that mean "let me choose before you go further". This is a hint that
# biases the agent, not a gate — the agent decides whether to actually pause.
_SELECTION_CUES = (
    "let me pick",
    "let me choose",
    "let me select",
    "so i can pick",
    "so i can choose",
    "so i can decide",
    "i'll pick",
    "i will pick",
    "i'll choose",
    "i'll tell you which",
    "ask me which",
    "let me decide",
    "give me options",
    "i want to pick",
    "i want to choose",
    "before you",
)


def _close_abandoned_tool_calls(state: GraphState) -> tuple[list[Any], Selection | None]:
    """Close out a tool call an interrupt left dangling from the PRIOR turn.

    Happens when the user types a new message instead of answering a pending
    pick — a real path in the UI, since the input box is not forced shut while
    a picker is showing. Without this, the next model call includes an
    AIMessage with tool_calls that have no matching ToolMessage, which every
    provider rejects outright (verified live: a 400 "must be followed by tool
    messages" from OpenAI). Runs at the start of every turn, before the new
    HumanMessage is added, regardless of how the turn is ultimately routed.
    """
    messages = state.messages or []
    if not messages:
        return [], None
    last = messages[-1]
    tool_calls = list(getattr(last, "tool_calls", None) or [])
    if not tool_calls:
        return [], None

    answered = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id
    }
    pending = [call for call in tool_calls if call.get("id") not in answered]
    if not pending:
        return [], None

    closers: list[Any] = []
    abandoned_selection: Selection | None = None

    for call in pending:
        call_id = call.get("id") or ""
        if call.get("name") == SELECTION_TOOL:
            args = call.get("args") or {}
            options: list[SelectionOption] = []
            for index, raw in enumerate(args.get("options") or []):
                if isinstance(raw, str):
                    raw = {"label": raw}
                options.append(
                    SelectionOption(
                        id=str(raw.get("id") or f"opt_{index + 1}"),
                        label=str(raw.get("label") or raw.get("name") or f"Option {index + 1}"),
                        description=raw.get("description"),
                        payload=raw.get("payload") if isinstance(raw.get("payload"), dict) else {},
                    )
                )
            abandoned_selection = Selection(
                kind=args.get("kind", "destination"),
                prompt=args.get("prompt", ""),
                destination=args.get("destination"),
                options=options,
                skipped_ids=[option.id for option in options],
                status="abandoned",
                turn_index=int(state.turn_index or 0),
                tool_call_id=call_id,
            )
            closers.append(
                ToolMessage(
                    content="The user moved on without picking from this list.",
                    tool_call_id=call_id,
                    name=SELECTION_TOOL,
                    status="error",
                )
            )
        else:
            closers.append(
                ToolMessage(
                    content="Not completed: the user sent a new message before this finished.",
                    tool_call_id=call_id,
                    name=call.get("name", ""),
                    status="error",
                )
            )

    return closers, abandoned_selection


def _recent_context(state: GraphState, limit: int = 4) -> str:
    """A few recent turns, given to the scope classifier so follow-ups make sense."""
    lines: list[str] = []
    for message in (state.messages or [])[-limit * 2 :]:
        if isinstance(message, HumanMessage):
            lines.append(f"user: {str(message.content)[:300]}")
        elif isinstance(message, AIMessage) and message.content:
            lines.append(f"assistant: {str(message.content)[:300]}")
    return "\n".join(lines)


class PreprocessNode:
    """Guardrail 1 + the request-split half of guardrail 2 + memory load."""

    def __call__(self, state: GraphState, config: RunnableConfig = None) -> dict[str, Any]:
        message = (state.message or "").strip()
        updates: dict[str, Any] = {
            "turn_index": int(state.turn_index or 0) + 1,
            "tool_round_count": 0,
            "withheld_kinds": [],
            "bot_response": None,
            "last_agent_output": None,
        }

        updates["selection_nudges"] = 0
        updates["pending_directive"] = None

        # Close out anything the PRIOR turn left dangling (e.g. the user typed
        # past a pending pick instead of answering it). Must happen before any
        # scope branch returns, so a corrupted history never reaches the model
        # even on an off-topic or smalltalk follow-up.
        closers, abandoned_selection = _close_abandoned_tool_calls(state)
        if closers:
            updates["messages"] = list(closers)
        if abandoned_selection:
            updates["selections"] = [abandoned_selection]

        usage_entries: list[dict[str, Any]] = []

        verdict = classify_scope(message, _recent_context(state))
        updates["scope"] = verdict.label
        updates["scope_reason"] = verdict.reason
        if verdict.usage:
            usage_entries.append(
                other_usage_entry(
                    "classify_scope", settings.llm_fast_model, verdict.usage["input_tokens"], verdict.usage["output_tokens"]
                )
            )

        if verdict.label == "out_of_scope":
            # Refuse without adding the user's message to history — an
            # off-topic turn should not pollute the travel context of later
            # turns — but the dangling-tool-call closers above still apply.
            updates["bot_response"] = verdict.refusal
            updates["usage_log"] = usage_entries
            return updates

        if verdict.label == "smalltalk":
            if message:
                updates["messages"] = [*closers, HumanMessage(content=message)]
            updates["usage_log"] = usage_entries
            return updates

        split = split_request(message)
        updates["withheld_kinds"] = [item.kind for item in split.withheld]
        updates["allowed_request"] = split.allowed_request or message
        if split.usage:
            usage_entries.append(
                other_usage_entry(
                    "split_request", settings.llm_fast_model, split.usage["input_tokens"], split.usage["output_tokens"]
                )
            )

        if split.withheld:
            logger.info(
                "output guardrail withheld %s from turn %s",
                [item.kind for item in split.withheld],
                updates["turn_index"],
            )

        lowered = message.lower()
        updates["wants_selection"] = any(cue in lowered for cue in _SELECTION_CUES)

        # The agent sees the sanitised request, not the raw one, so a bolted-on
        # "and send me the query" never reaches the model at all.
        if message:
            updates["messages"] = [*closers, HumanMessage(content=updates["allowed_request"])]

        user_id = ((config or {}).get("configurable") or {}).get("auth_user_id") or state.user_id
        summary = get_user_memory_summary(user_id)
        updates["user_memory_summary"] = summary.model_dump()
        updates["usage_log"] = usage_entries

        return updates


_TOOL_SEARCH_SYSTEM_PROMPT = (
    "You decide whether answering the user's travel request needs a real lookup "
    "(destinations, places, events, accommodations, or the user's saved trips) or "
    "an action (saving, updating, removing a saved trip). If it does, call "
    "search_tools with a clear, self-contained description of the capability "
    "needed — expand short or ambiguous phrasing using the conversation so far. "
    "If the request is answerable from general knowledge or conversation alone, "
    "with no lookup or action needed, just answer it directly instead of calling "
    "search_tools."
)


class ToolSearchNode:
    """Gate: one real tool call decides whether this turn needs any domain
    tool before AgentNode runs, and if so, with a model-rewritten query.

    Replaces the old always-top-k retrieval that ran unconditionally in
    PreprocessNode: that approach had no relevance floor, so even a fully
    toolless in-scope request ("what's the best time to visit Portugal?")
    still got 3 tool schemas forced onto AgentNode every turn. Putting a real
    tool call here means the model itself decides whether to search at all —
    same mechanism as Anthropic's tool_search_tool — and a confidently
    toolless turn can skip AgentNode's full system prompt and bind_tools call
    entirely instead of merely binding fewer tools.

    Never short-circuits a turn where `wants_selection` is true: that flag
    drives RequireSelectionNode's HIL pause downstream of `agent`, and this
    node's narrow single-tool judgment must never be trusted over that gate.
    """

    def __call__(self, state: GraphState, config: RunnableConfig = None) -> dict[str, Any]:
        if not settings.tool_rag_enabled:
            return {"active_tools": [tool.name for tool in ALL_TOOLS]}

        query_text = (state.allowed_request or state.message or "").strip()
        if not query_text:
            return {"active_tools": [tool.name for tool in ALL_TOOLS]}

        try:
            response = fast_model().bind_tools([search_tools]).invoke(
                [
                    SystemMessage(content=_TOOL_SEARCH_SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"{_recent_context(state)}\n\ncurrent request: {query_text}".strip()
                    ),
                ],
                config=config,
            )
        except Exception as exc:
            logger.warning("tool_search model unavailable, binding all tools: %s", exc)
            return {"active_tools": [tool.name for tool in ALL_TOOLS]}

        usage_metadata = getattr(response, "usage_metadata", None) or {}
        usage_log = (
            [
                other_usage_entry(
                    "tool_search",
                    settings.llm_fast_model,
                    usage_metadata.get("input_tokens", 0),
                    usage_metadata.get("output_tokens", 0),
                )
            ]
            if usage_metadata
            else []
        )

        tool_calls = list(getattr(response, "tool_calls", None) or [])
        if tool_calls:
            search_query = str(tool_calls[0].get("args", {}).get("query") or query_text)
            retrieved = retrieve_relevant_tools(search_query)
            active = list(dict.fromkeys([*(retrieved or [tool.name for tool in ALL_TOOLS]), SELECTION_TOOL]))
            return {"active_tools": active, "usage_log": usage_log}

        # No tool call: the model judged this answerable without a lookup.
        if state.wants_selection:
            # Never trust this over HIL — fail open to the full pipeline so
            # RequireSelectionNode still gets its normal chance to pause.
            return {"active_tools": [tool.name for tool in ALL_TOOLS], "usage_log": usage_log}

        text = str(response.content or "")
        return {
            "bot_response": text,
            "messages": [AIMessage(content=text)],
            "active_tools": [SELECTION_TOOL],
            "usage_log": usage_log,
        }


class SmalltalkNode:
    def __call__(self, state: GraphState) -> dict[str, Any]:
        try:
            model = fast_model()
            reply = model.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a friendly trip organiser assistant. Reply to this social "
                            "message in one or two warm sentences, and mention that you can help "
                            "plan a trip — finding destinations, places to visit, events, and "
                            "places to stay — or pull up the trips they've already saved. "
                            "Do not list tool names."
                        )
                    ),
                    HumanMessage(content=state.message or "hello"),
                ]
            )
            text = str(getattr(reply, "content", "") or "").strip()
            usage_metadata = getattr(reply, "usage_metadata", None) or {}
        except Exception as exc:
            logger.warning("smalltalk model unavailable: %s", exc)
            text = ""
            usage_metadata = {}

        if not text:
            text = (
                "Happy to help! Tell me where you're thinking of going and I can suggest "
                "destinations, places to visit, what's on while you're there, and where to "
                "stay — or pull up a trip you've already saved."
            )
        usage_log = (
            [
                other_usage_entry(
                    "smalltalk",
                    settings.llm_fast_model,
                    usage_metadata.get("input_tokens", 0),
                    usage_metadata.get("output_tokens", 0),
                )
            ]
            if usage_metadata
            else []
        )
        return {"bot_response": text, "messages": [AIMessage(content=text)], "usage_log": usage_log}


class OutOfScopeNode:
    def __call__(self, state: GraphState) -> dict[str, Any]:
        text = state.bot_response or (
            "That one is outside what I can help with. I'm here for trip planning — "
            "destinations, places to visit, events, places to stay, and the trips you've saved."
        )
        return {"bot_response": text}


class SummarizationNode:
    """Compress older history once it outgrows the budget, via langmem.

    The summary is kept in `running_summary` and rendered back into the prompt by
    context.build_prompt, rather than being pushed into the message list as a
    SystemMessage — that keeps exactly one system block per model call.
    """

    def __call__(self, state: GraphState) -> dict[str, Any]:
        messages = [
            message
            for message in (state.messages or [])
            if isinstance(message, (HumanMessage, AIMessage, ToolMessage))
        ]
        if len(messages) < 6:
            return {}

        try:
            from langmem.short_term.summarization import RunningSummary, summarize_messages
        except ImportError:
            logger.debug("langmem not installed; skipping summarization")
            return {}

        running_summary = state.running_summary
        if isinstance(running_summary, dict):
            # A checkpoint round-trip through a serializer that doesn't know
            # RunningSummary turns it back into a plain dict — summarize_messages
            # needs the real dataclass (it reads .summarized_message_ids). The
            # checkpointer's allowlist is the real fix; this is a second layer
            # so an already-corrupted or future-library-version checkpoint
            # degrades to "start summarizing fresh" instead of crashing outright.
            try:
                running_summary = RunningSummary(**running_summary)
            except TypeError:
                logger.warning("running_summary checkpoint was unreadable; starting a fresh summary")
                running_summary = None

        try:
            result = summarize_messages(
                messages,
                running_summary=running_summary,
                model=fast_model(),
                max_tokens=settings.summary_token_threshold,
                max_summary_tokens=512,
            )
        except Exception as exc:
            logger.warning("summarization failed, keeping full history: %s", exc)
            return {}

        if result.running_summary is None or result.running_summary is running_summary:
            # Compare against the (possibly just-coerced) local, not state.running_summary
            # directly — after coercion those are never the same object even when
            # summarize_messages left the summary untouched, which would otherwise
            # make this look "changed" every turn and needlessly rebuild history.
            return {}  # under budget; nothing was compressed

        # langmem prepends its own SystemMessage carrying the summary; drop it,
        # since the summary now travels in state.running_summary instead.
        kept = [message for message in result.messages if not isinstance(message, SystemMessage)]

        # langmem doesn't expose usage_metadata for its internal model call, so
        # this is an estimate over the text it summarized and produced.
        summary_text = getattr(result.running_summary, "summary", "") or ""
        original_text = "\n".join(str(getattr(m, "content", "") or "") for m in messages)
        summarize_usage = other_usage_entry(
            "summarize",
            settings.llm_fast_model,
            count_tokens(original_text, settings.llm_fast_model),
            count_tokens(summary_text, settings.llm_fast_model),
        )
        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept],
            "running_summary": result.running_summary,
            "usage_log": [summarize_usage],
        }


class AgentNode:
    """Calls the model with tools bound.

    Which tools are bound is decided per turn by ToolSearchNode's semantic
    retrieval (state.active_tools), not fixed to the full tool set — so the
    model here stays raw/unbound and gets `.bind_tools(...)` applied fresh on
    every call with whatever subset this turn resolved to.
    """

    def __init__(self) -> None:
        self._model: Any = None

    @property
    def model(self) -> Any:
        # Built lazily so the module imports without LLM credentials present.
        if self._model is None:
            self._model = build_model()
        return self._model

    def __call__(self, state: GraphState, config: RunnableConfig = None) -> dict[str, Any]:
        prompt = build_prompt(state)
        tool_names = state.active_tools or [tool.name for tool in ALL_TOOLS]
        tools = [TOOLS_BY_NAME[name] for name in tool_names if name in TOOLS_BY_NAME]
        try:
            # A list of messages, not {"messages": [...]} — the old call passed a
            # dict, which a chat model cannot accept.
            response = self.model.bind_tools(tools).invoke(prompt, config=config)
        except Exception as exc:
            logger.exception("agent model call failed")
            text = "I hit a problem working on that. Could you try again in a moment?"
            return {"messages": [AIMessage(content=text)], "last_agent_output": text}

        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(getattr(response, "content", response)))

        usage_entry = agent_usage_entry(state, tools, settings.llm_model, response)
        return {
            "messages": [response],
            "last_agent_output": str(response.content or ""),
            "usage_log": [usage_entry],
        }


class ToolExecutorNode:
    """Runs the model's tool calls.

    Written by hand rather than using the prebuilt ToolNode for three reasons:

    * Tool calls that already have a ToolMessage are skipped, so resuming after
      an interrupt does not re-run work that already completed.
    * Non-pausing tools run first and the pausing tool runs last, so an
      interrupt cannot discard a sibling tool's result.
    * Errors become a ToolMessage the model can read and recover from, while
      authorization failures are surfaced as a refusal rather than a retry.
    """

    def __call__(self, state: GraphState, config: RunnableConfig = None) -> dict[str, Any] | Command:
        messages = state.messages or []
        last = messages[-1] if messages else None
        tool_calls = list(getattr(last, "tool_calls", None) or [])
        if not tool_calls:
            return {}

        answered = {
            message.tool_call_id
            for message in messages
            if isinstance(message, ToolMessage) and message.tool_call_id
        }
        pending = [call for call in tool_calls if call.get("id") not in answered]
        if not pending:
            return {}

        # Selection tools last: they interrupt, and an interrupt throws away the
        # whole node's uncommitted work.
        pending.sort(key=lambda call: call.get("name") == SELECTION_TOOL)

        new_messages: list[Any] = []
        extra_updates: dict[str, Any] = {}

        for call in pending:
            name = call.get("name") or ""
            call_id = call.get("id") or ""
            tool_fn = TOOLS_BY_NAME.get(name)

            if tool_fn is None:
                new_messages.append(
                    ToolMessage(
                        content=f"There is no tool called {name}. Use one of: {', '.join(TOOLS_BY_NAME)}.",
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                    )
                )
                continue

            args = dict(call.get("args") or {})
            if name in _STATE_INJECTED_TOOLS:
                args["state"] = state

            try:
                # Invoked in ToolCall form so LangChain fills InjectedToolCallId
                # and wraps the return value in a ToolMessage for us.
                result = tool_fn.invoke(
                    {"name": name, "args": args, "id": call_id, "type": "tool_call"},
                    config=config,
                )
            except GraphBubbleUp:
                # A pause (or any other control-flow signal) must reach the
                # runtime untouched. Catching it here would turn the interrupt
                # into an error message and the conversation would never pause.
                raise
            except AuthorizationError as exc:
                logger.warning("authorization guardrail blocked %s: %s", name, exc)
                new_messages.append(
                    ToolMessage(
                        content=(
                            "That was blocked: you can only read or change your own saved trips. "
                            "Tell the user plainly and do not retry."
                        ),
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                    )
                )
                continue
            except ToolError as exc:
                new_messages.append(
                    ToolMessage(content=f"Lookup failed: {exc}", tool_call_id=call_id, name=name, status="error")
                )
                continue
            except Exception as exc:
                logger.exception("tool %s failed", name)
                new_messages.append(
                    ToolMessage(
                        content=f"That lookup could not be completed: {type(exc).__name__}.",
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                    )
                )
                continue

            # `request_user_selection` returns a Command carrying its own
            # ToolMessage plus the resolved selection.
            if isinstance(result, Command):
                update = result.update or {}
                for key, value in update.items():
                    if key == "messages":
                        new_messages.extend(value)
                    else:
                        extra_updates[key] = value
                continue

            if isinstance(result, ToolMessage):
                new_messages.append(result)
                continue

            new_messages.append(
                ToolMessage(
                    content=result if isinstance(result, str) else json.dumps(result, default=str),
                    tool_call_id=call_id,
                    name=name,
                )
            )

        return {
            "messages": new_messages,
            "tool_round_count": int(state.tool_round_count or 0) + 1,
            **extra_updates,
        }


class RequireSelectionNode:
    """Send the agent back when it skipped a pause the user asked for.

    The prompt alone is not reliable enough here: asked for "top 5 in Norway and
    let me pick some", a smaller model will happily search and then answer the
    whole thing in one go. Since "pause when the user asks to choose" is a
    behavioural requirement rather than a preference, it is enforced in code.

    Bounded by `selection_nudges` — after two attempts the turn is allowed to
    finish normally. Failing to pause is worse than not pausing, but hanging is
    worse than both.
    """

    MAX_NUDGES = 2

    def __call__(self, state: GraphState) -> dict[str, Any]:
        attempt = int(state.selection_nudges or 0) + 1
        logger.info(
            "agent skipped a requested pause on turn %s; nudge %s/%s",
            state.turn_index,
            attempt,
            self.MAX_NUDGES,
        )
        return {
            "selection_nudges": attempt,
            "pending_directive": (
                "STOP — you answered without pausing, but the user explicitly asked to pick "
                "from the results first. Do not write a final answer yet. Call "
                "request_user_selection now, passing the items you just found as options: "
                "each needs an id, a short label, and the full result object as `payload`. "
                "Once they have picked, continue with only their picks."
            ),
            # Clear the draft so a half-finished answer cannot leak out if the
            # nudge budget runs out on the next pass.
            "last_agent_output": "",
        }


class FinalizeNode:
    """Guardrail 2, output half: redact internals and disclose what was withheld."""

    def __call__(self, state: GraphState) -> dict[str, Any]:
        messages = state.messages or []
        updates: dict[str, Any] = {}

        # If the loop stopped while tool calls were still outstanding, answer them
        # here. An AIMessage whose tool_calls have no matching ToolMessage makes
        # the next turn's model request invalid, so this cannot be left dangling.
        closing = self._close_dangling_tool_calls(messages)
        if closing:
            updates["messages"] = closing

        draft = state.last_agent_output or ""
        if not draft.strip():
            draft = (
                "I ran out of steps before I could pull that together into an answer. "
                "Want me to narrow it down and try again?"
            )

        verdict = apply_output_guardrail(draft, state.withheld_kinds)
        if verdict.redacted:
            logger.info("output guardrail redacted content on turn %s", state.turn_index)

        updates["bot_response"] = verdict.text
        updates["last_agent_output"] = verdict.text
        # The directive was for one agent call only.
        updates["pending_directive"] = None
        return updates

    @staticmethod
    def _close_dangling_tool_calls(messages: list[Any]) -> list[Any]:
        last = messages[-1] if messages else None
        tool_calls = list(getattr(last, "tool_calls", None) or [])
        if not tool_calls:
            return []
        answered = {
            message.tool_call_id
            for message in messages
            if isinstance(message, ToolMessage) and message.tool_call_id
        }
        return [
            ToolMessage(
                content="Not run: this turn reached its limit on lookups.",
                tool_call_id=call["id"],
                name=call.get("name", ""),
                status="error",
            )
            for call in tool_calls
            if call.get("id") and call["id"] not in answered
        ]


class PersistPreferencesNode:
    """Extract durable preferences worth remembering across threads."""

    def __call__(self, state: GraphState, config: RunnableConfig = None) -> dict[str, Any]:
        user_id = ((config or {}).get("configurable") or {}).get("auth_user_id") or state.user_id
        if not user_id:
            return {}

        # Explicit writes queued by another node take priority.
        for key, value in (state.preferences_to_persist or {}).items():
            write_user_memory(user_id, key, value)

        message = (state.message or "").strip()
        if len(message) < 12:
            return {"preferences_to_persist": {}}

        extracted, usage = self._extract(message)
        for key, value in extracted.items():
            write_user_memory(user_id, key, value)
        if extracted:
            logger.info("persisted user memory keys %s", sorted(extracted))

        updates: dict[str, Any] = {"preferences_to_persist": {}}
        if usage:
            updates["usage_log"] = [
                other_usage_entry("preferences", settings.llm_fast_model, usage["input_tokens"], usage["output_tokens"])
            ]
        return updates

    def _extract(self, message: str) -> tuple[dict[str, Any], dict[str, int] | None]:
        """Pull durable preferences out of the message, if there are any.

        Only lasting facts, never one-trip details — "I'm going to Rome in May"
        is not a preference, "I always travel with my kids" is.
        """
        from pydantic import BaseModel, Field

        class Preferences(BaseModel):
            travel_style: str | None = None
            preferred_stay_type: str | None = None
            interests: list[str] | None = None
            dietary: str | None = None
            home_city: str | None = None
            budget_level: str | None = None
            pace: str | None = None
            avoid: list[str] | None = None
            hil_preference: str | None = Field(
                default=None,
                description="Set to 'always ask' if they want to choose from options themselves, 'decide for me' if they want the assistant to just proceed.",
            )
            accessibility: str | None = None

        try:
            model = fast_model().with_structured_output(Preferences, include_raw=True)
            raw_result = model.invoke(
                [
                    SystemMessage(
                        content=(
                            "Extract only LASTING traveller preferences from the message — things "
                            "that would still be true on their next trip. Leave a field null unless "
                            "the message clearly states it. A specific destination, date or budget "
                            "for one trip is NOT a preference. Return all nulls if there is nothing."
                        )
                    ),
                    HumanMessage(content=message),
                ]
            )
            result = raw_result["parsed"]
            if result is None:
                raise ValueError(f"could not parse Preferences: {raw_result.get('parsing_error')}")
            usage_metadata = getattr(raw_result.get("raw"), "usage_metadata", None) or {}
            usage = (
                {
                    "input_tokens": usage_metadata.get("input_tokens", 0),
                    "output_tokens": usage_metadata.get("output_tokens", 0),
                }
                if usage_metadata
                else None
            )
        except Exception as exc:
            logger.debug("preference extraction unavailable: %s", exc)
            return {}, None

        extracted = {
            key: value
            for key, value in result.model_dump().items()
            if value not in (None, "", []) and key in ALLOWED_KEYS
        }
        return extracted, usage
