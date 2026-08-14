"""Guardrail 3 — authorization.

Purely code-based, no model in the loop. Two rules, applied to every MongoDB
operation:

1. The acting `user_id` is taken from the verified Supabase JWT and threaded
   through the graph's runtime config. It is never a tool argument, so the model
   cannot be talked into supplying someone else's id.
2. Every filter is user-scoped on the way in, and every document is re-checked
   on the way out. A row that somehow comes back with a different `user_id` is
   treated as a breach and dropped, not returned.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Mapping[str, Any])


class AuthorizationError(PermissionError):
    """Raised when an operation would touch data the caller does not own."""


def require_user_id(user_id: str | None, *, operation: str) -> str:
    """Reject an unauthenticated or blank identity before any query runs."""
    if not user_id or not str(user_id).strip():
        raise AuthorizationError(
            f"'{operation}' requires a signed-in user; no verified identity was present."
        )
    return str(user_id).strip()


def scoped_filter(user_id: str, extra: Mapping[str, Any] | None = None, *, operation: str) -> dict[str, Any]:
    """Build a query filter that is always pinned to the owner.

    `user_id` is written last so a caller-supplied `extra` can never override it.
    """
    owner = require_user_id(user_id, operation=operation)
    query: dict[str, Any] = dict(extra or {})
    if "user_id" in query and str(query["user_id"]) != owner:
        raise AuthorizationError(
            f"'{operation}' attempted to query another user's records."
        )
    query["user_id"] = owner
    return query


def assert_owned(document: Mapping[str, Any] | None, user_id: str, *, operation: str) -> Mapping[str, Any] | None:
    """Verify a single document belongs to the caller."""
    if document is None:
        return None
    owner = require_user_id(user_id, operation=operation)
    if str(document.get("user_id")) != owner:
        logger.error(
            "authorization guardrail blocked %s: document owner mismatch (doc=%s)",
            operation,
            document.get("_id"),
        )
        raise AuthorizationError(
            f"'{operation}' was blocked: that record belongs to a different user."
        )
    return document


def filter_owned(documents: Iterable[T], user_id: str, *, operation: str) -> list[T]:
    """Drop any document that is not the caller's instead of raising.

    Used on list reads, where one stray row should not fail the whole request —
    but it is logged as an error because it should be impossible.
    """
    owner = require_user_id(user_id, operation=operation)
    kept: list[T] = []
    for document in documents:
        if str(document.get("user_id")) == owner:
            kept.append(document)
        else:
            logger.error(
                "authorization guardrail dropped a foreign document during %s (doc=%s)",
                operation,
                document.get("_id"),
            )
    return kept
