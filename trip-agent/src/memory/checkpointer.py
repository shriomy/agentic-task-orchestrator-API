"""Thread-scoped conversation state, persisted by LangGraph's checkpointer.

This is the State scope: messages, selections, turn_index and tool_round_count,
saved automatically against thread_id on every super-step. It is also what makes
`interrupt()` resumable — the pending pick is part of the checkpoint, so a user
can answer an interrupt minutes later, or reopen the thread and continue.

Built lazily with an in-memory fallback: no SUPABASE_DB_URL means the graph
still runs (single process, non-durable) instead of failing at import.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)

_checkpointer: Any = None
_pool: Any = None
_lock = threading.Lock()


def build_serializer() -> Any:
    """Serializer with strict msgpack plus an allowlist for every custom type
    that ends up in GraphState.

    LangGraph's default is permissive-with-a-warning and is due to start
    rejecting unregistered types. An explicit allowlist silences that AND keeps
    deserialization restricted — but every type actually reachable from state
    must be named, or it silently comes back as a plain dict on the next load
    instead of the real object. That happened here: `RunningSummary` (langmem's
    running-summary dataclass, held in `state.running_summary`) was missing, so
    a checkpoint round-trip turned it into a dict, and passing that dict back
    into `summarize_messages()` blew up with `'dict' object has no attribute
    'summarized_message_ids'` on the second summarization of a thread.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from ..graph.state import Selection, SelectionOption

    allowed: list[Any] = [Selection, SelectionOption]
    try:
        from langmem.short_term.summarization import RunningSummary

        allowed.append(RunningSummary)
    except ImportError:
        pass

    try:
        return JsonPlusSerializer(allowed_msgpack_modules=tuple(allowed)).with_msgpack_allowlist(allowed)
    except TypeError:
        # Older langgraph without the allowlist parameter.
        return JsonPlusSerializer()


def _build_postgres_checkpointer() -> Any | None:
    if not settings.supabase_db_url:
        return None
    global _pool
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg import Connection
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        def configure(conn: Connection) -> None:
            # The checkpointer manages its own transactions and expects dict rows.
            conn.autocommit = True
            conn.row_factory = dict_row

        _pool = ConnectionPool(
            settings.supabase_db_url,
            connection_class=Connection,
            configure=configure,
            min_size=settings.postgres_pool_min,
            max_size=settings.postgres_pool_max,
            timeout=settings.postgres_pool_timeout,
            open=True,
            kwargs={"prepare_threshold": None},  # required behind Supabase's pooler
        )
        saver = PostgresSaver(_pool, serde=build_serializer())
        saver.setup()  # creates the library-owned checkpoint tables
        logger.info("conversation checkpointer: postgres")
        return saver
    except Exception as exc:
        logger.warning(
            "postgres checkpointer unavailable (%s); falling back to in-memory. "
            "Threads will not survive a restart.",
            exc,
        )
        return None


def get_checkpointer() -> Any:
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    with _lock:
        if _checkpointer is None:
            _checkpointer = _build_postgres_checkpointer()
            if _checkpointer is None:
                from langgraph.checkpoint.memory import InMemorySaver

                _checkpointer = InMemorySaver(serde=build_serializer())
                logger.info("conversation checkpointer: in-memory")
    return _checkpointer


def close_checkpointer() -> None:
    """Release the connection pool on application shutdown."""
    global _pool, _checkpointer
    if _pool is not None:
        try:
            _pool.close()
        except Exception:
            pass
        _pool = None
    _checkpointer = None
