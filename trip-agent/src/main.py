"""FastAPI surface for the trip organiser agent.

Two notes on deliberate choices:

* Authentication. Every endpoint requires a Supabase bearer token. The verified
  `sub` claim becomes `configurable.auth_user_id`, which is the only identity the
  tools and the authorization guardrail will accept. A `user_id` in the request
  body is checked against it and rejected on mismatch — it is never trusted.

* Streaming. Progress is streamed (which lookups are running, when a pick is
  needed), but the assistant's prose is NOT streamed token-by-token. The output
  guardrail has to see a complete draft before any of it reaches the user;
  streaming raw model tokens would send text past the guardrail by definition.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Iterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .auth import AuthError, AuthenticatedUser, verify_access_token
from .config import settings
from .db.mongo import status as mongo_status
from .graph.graph import get_graph
from .graph.tool_retrieval import ensure_tool_index, tool_index_ready
from .graph.tools import SELECTION_TOOL
from .guardrails.authorization import AuthorizationError
from .memory.checkpointer import close_checkpointer
from .memory.conversations import (
    delete_conversation,
    ensure_conversation,
    get_history,
    list_conversations,
    log_selection,
    record_message,
)
from .memory.supabase_client import supabase_available
from .tools.favorites import delete_favorite, get_favorites, remove_favorite_section

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_graph()  # compile up front so the first request isn't slow
    tool_rag_ready = ensure_tool_index()
    logger.info(
        "trip agent ready | supabase=%s mongo=%s tracing=%s tool_rag=%s",
        supabase_available(),
        bool(settings.mongodb_uri),
        bool(settings.langsmith_api_key and settings.langsmith_tracing),
        tool_rag_ready,
    )
    yield
    close_checkpointer()


app = FastAPI(title="Trip Organiser Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Auth dependency
# --------------------------------------------------------------------------- #


def current_user(authorization: str | None = Header(default=None)) -> AuthenticatedUser:
    try:
        return verify_access_token(authorization)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


def assert_body_user_matches(user: AuthenticatedUser, claimed: str | None) -> None:
    """Reject a request that claims to act as somebody else.

    Guardrail 3 at the API boundary: the body's user_id is decoration, and a
    mismatch is a caller trying to read or write another account.
    """
    if claimed and str(claimed) != user.user_id:
        logger.warning("rejected request claiming user_id=%s as user_id=%s", claimed, user.user_id)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only act on your own account.",
        )


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class SendMessageRequest(BaseModel):
    message: str
    thread_id: str | None = None
    user_id: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    selection_id: str | None = None
    selected_options: list[str] = Field(default_factory=list)
    user_id: str | None = None


# --------------------------------------------------------------------------- #
# Streaming helpers
# --------------------------------------------------------------------------- #

# Plain-language progress labels. The guardrails forbid showing tool names, so
# the UI is told what is happening in travel terms instead.
TOOL_PROGRESS = {
    "web_search": "Searching the web…",
    "search_places": "Looking up places to visit…",
    "get_place_details": "Reading up on a place…",
    "search_events": "Checking what's on…",
    "get_accommodation_details": "Getting the details on a stay…",
    "search_accommodations": "Finding places to stay…",
    "list_trip_favorites": "Opening your saved trips…",
    "save_trip_favorite": "Saving to your trips…",
    "update_trip_favorite": "Updating your saved trip…",
    "remove_trip_favorite_section": "Updating your saved trip…",
    "delete_trip_favorite": "Removing that saved trip…",
    SELECTION_TOOL: "Waiting for you to choose…",
}


def sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def _interrupt_event(interrupt: Any) -> dict[str, Any]:
    """Shape a LangGraph Interrupt into the UI's InterruptData contract."""
    value = getattr(interrupt, "value", interrupt) or {}
    if not isinstance(value, dict):
        value = {"reason": str(value)}
    return {
        "type": "interrupt",
        "data": {
            "reason": value.get("reason") or "Which of these would you like?",
            "selection_id": value.get("selection_id") or "",
            "kind": value.get("kind") or "destination",
            "destination": value.get("destination"),
            "options": [
                {
                    "id": option.get("id"),
                    "label": option.get("label"),
                    "description": option.get("description"),
                }
                for option in (value.get("options") or [])
            ],
        },
    }


def run_graph_stream(
    graph_input: Any,
    *,
    thread_id: str,
    user: AuthenticatedUser,
    turn_label: str,
) -> Iterator[str]:
    """Drive one graph run and yield SSE frames.

    Runs to completion or to the next interrupt, whichever comes first.
    """
    graph = get_graph()
    config = {
        "configurable": {
            "thread_id": thread_id,
            # The one trusted identity. Tools read it from here.
            "auth_user_id": user.user_id,
            "user_email": user.email,
        },
        "recursion_limit": settings.max_tool_rounds * 3 + 10,
        "metadata": {"user_id": user.user_id, "thread_id": thread_id},
        "run_name": turn_label,
    }

    final_text = ""
    interrupt_payload: dict[str, Any] | None = None
    announced: set[str] = set()

    try:
        for chunk in graph.stream(graph_input, config=config, stream_mode="updates"):
            if not isinstance(chunk, dict):
                continue

            for node_name, update in chunk.items():
                if node_name == "__interrupt__":
                    interrupts = update if isinstance(update, (list, tuple)) else [update]
                    if interrupts:
                        interrupt_payload = _interrupt_event(interrupts[0])
                    continue

                if not isinstance(update, dict):
                    continue

                # Announce lookups as the agent requests them.
                if node_name == "agent":
                    for message in update.get("messages") or []:
                        for call in getattr(message, "tool_calls", None) or []:
                            label = TOOL_PROGRESS.get(call.get("name", ""))
                            if label and label not in announced:
                                announced.add(label)
                                yield sse({"type": "tool", "content": label})

                if node_name in ("finalize", "smalltalk", "out_of_scope"):
                    final_text = str(update.get("bot_response") or final_text)

    except AuthorizationError as exc:
        yield sse({"type": "error", "message": str(exc)})
        yield sse({"type": "done"})
        return
    except Exception:
        logger.exception("graph run failed for thread %s", thread_id)
        yield sse({"type": "error", "message": "Something went wrong on my side. Please try again."})
        yield sse({"type": "done"})
        return

    if interrupt_payload:
        # A pause is not the end of the turn; the thread stays open for /chat/resume.
        data = interrupt_payload["data"]
        log_selection(
            thread_id,
            data.get("selection_id") or "",
            data.get("kind") or "destination",
            data.get("options") or [],
            destination=data.get("destination"),
            status="pending",
        )
        record_message(
            thread_id,
            user.user_id,
            "assistant",
            data.get("reason") or "",
            interrupt_data=data,
        )
        yield sse(interrupt_payload)
        yield sse({"type": "done"})
        return

    if final_text:
        record_message(thread_id, user.user_id, "assistant", final_text)
        yield sse({"type": "text", "content": final_text})

    yield sse({"type": "done"})


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Stops nginx buffering the stream into one blob.
    "X-Accel-Buffering": "no",
}


# --------------------------------------------------------------------------- #
# Chat endpoints
# --------------------------------------------------------------------------- #


@app.post("/chat/send")
def chat_send(
    request: SendMessageRequest,
    user: AuthenticatedUser = Depends(current_user),
) -> StreamingResponse:
    """Start a turn. Streams progress, then either an answer or a pause."""
    assert_body_user_matches(user, request.user_id)

    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message cannot be empty.")

    thread_id = request.thread_id or str(uuid.uuid4())
    ensure_conversation(thread_id, user.user_id, request.message)
    record_message(thread_id, user.user_id, "user", request.message)

    graph_input = {
        "thread_id": thread_id,
        "user_id": user.user_id,
        "message": request.message,
    }

    def frames() -> Iterator[str]:
        # Tell the client the thread id first, so a new conversation can be
        # tracked before the answer arrives.
        yield sse({"type": "thread", "thread_id": thread_id})
        yield from run_graph_stream(
            graph_input, thread_id=thread_id, user=user, turn_label="chat_send"
        )

    return StreamingResponse(frames(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/chat/resume")
def chat_resume(
    request: ResumeRequest,
    user: AuthenticatedUser = Depends(current_user),
) -> StreamingResponse:
    """Answer a pending pick and let the turn continue.

    The same turn can pause again — the client should keep handling interrupt
    events until it sees a text event.
    """
    assert_body_user_matches(user, request.user_id)

    graph = get_graph()
    config = {"configurable": {"thread_id": request.thread_id, "auth_user_id": user.user_id}}

    # Authorization: only resume a thread this user owns, and only one that is
    # genuinely waiting. Without this check a guessed thread_id could be driven.
    try:
        snapshot = graph.get_state(config)
    except Exception as exc:
        logger.warning("could not load thread %s: %s", request.thread_id, exc)
        raise HTTPException(status_code=404, detail="That conversation could not be found.") from exc

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="That conversation could not be found.")

    owner = snapshot.values.get("user_id")
    if owner and str(owner) != user.user_id:
        logger.warning("blocked resume of thread %s owned by %s", request.thread_id, owner)
        raise HTTPException(status_code=403, detail="You can only act on your own conversations.")

    if not snapshot.next:
        raise HTTPException(
            status_code=409,
            detail="That conversation isn't waiting on a choice right now.",
        )

    from langgraph.types import Command

    graph_input = Command(
        resume={
            "selection_id": request.selection_id,
            "selected_options": request.selected_options,
        }
    )

    if request.selection_id:
        log_selection(
            request.thread_id,
            request.selection_id,
            "selection",
            [],
            picked_ids=request.selected_options,
            status="answered" if request.selected_options else "abandoned",
        )

    def frames() -> Iterator[str]:
        yield from run_graph_stream(
            graph_input, thread_id=request.thread_id, user=user, turn_label="chat_resume"
        )

    return StreamingResponse(frames(), media_type="text/event-stream", headers=SSE_HEADERS)


# --------------------------------------------------------------------------- #
# Conversations
# --------------------------------------------------------------------------- #


@app.get("/conversations")
def conversations(user: AuthenticatedUser = Depends(current_user)) -> dict[str, Any]:
    return {"conversations": list_conversations(user.user_id)}


@app.get("/conversations/{thread_id}/messages")
def conversation_messages(
    thread_id: str,
    user: AuthenticatedUser = Depends(current_user),
) -> dict[str, Any]:
    return {"thread_id": thread_id, "messages": get_history(thread_id, user.user_id)}


@app.delete("/conversations/{thread_id}", status_code=204)
def conversation_delete(
    thread_id: str,
    user: AuthenticatedUser = Depends(current_user),
) -> Response:
    delete_conversation(thread_id, user.user_id)
    return Response(status_code=204)


# --------------------------------------------------------------------------- #
# Favorites (read/delete also exposed directly, for the UI's saved-trips view)
# --------------------------------------------------------------------------- #


@app.get("/favorites")
def favorites(
    destination: str | None = None,
    section: str | None = None,
    user: AuthenticatedUser = Depends(current_user),
) -> dict[str, Any]:
    try:
        return get_favorites(user.user_id, destination=destination, section=section)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.delete("/favorites/{destination}")
def favorites_delete(
    destination: str,
    section: str | None = None,
    user: AuthenticatedUser = Depends(current_user),
) -> dict[str, Any]:
    """Delete a saved trip, or with `?section=` just one part of it."""
    try:
        if section:
            return remove_favorite_section(user.user_id, destination, section)
        return delete_favorite(user.user_id, destination)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Ops
# --------------------------------------------------------------------------- #


def tool_rag_status() -> dict[str, Any]:
    return {"enabled": settings.tool_rag_enabled, "index_ready": tool_index_ready()}


@app.get("/health")
def health() -> dict[str, Any]:
    """What's actually wired up. Reports why a dependency is down, not just that
    it is, since each failure mode needs a different fix."""
    return {
        "status": "ok",
        "supabase": supabase_available(),
        "mongodb": mongo_status(),
        "tracing": bool(settings.langsmith_api_key and settings.langsmith_tracing),
        "checkpointer": "postgres" if settings.supabase_db_url else "in-memory (not durable)",
        "tools": {
            "web_search": bool(settings.tavily_api_key),
            "places": bool(settings.opentripmap_api_key),
            "events": bool(settings.ticketmaster_api_key),
            "accommodations": bool(settings.booking_api_key),
        },
        "tool_rag": tool_rag_status(),
    }


@app.get("/graph.png")
def graph_png() -> Response:
    """The compiled graph as a diagram, handy for docs and debugging."""
    try:
        image = get_graph().get_graph().draw_mermaid_png()
        return Response(content=image, media_type="image/png")
    except Exception as exc:
        logger.warning("could not render graph png: %s", exc)
        return Response(
            content=get_graph().get_graph().draw_mermaid(),
            media_type="text/plain",
        )


@app.exception_handler(AuthorizationError)
def authorization_error_handler(request: Request, exc: AuthorizationError) -> Response:
    """Never leak the underlying query or ids when authorization fails."""
    logger.warning("authorization error on %s: %s", request.url.path, exc)
    return Response(
        content=json.dumps({"detail": "You can only access your own data."}),
        status_code=403,
        media_type="application/json",
    )
