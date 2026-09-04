"""Authorization guardrail — ported unchanged from
trip-agent/src/guardrails/authorization.py.

Two rules, applied to every MongoDB operation:

1. The acting `user_id` comes only from `auth.SupabaseTokenVerifier` (i.e. a
   verified token's `sub` claim) — server.py never accepts it as a tool
   argument, so nothing a caller (or a model driving this server through the
   MCP client) sends can put a different user's id here.
2. Every filter is user-scoped on the way in, and every document is
   re-checked on the way out.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Mapping[str, Any])


class AuthorizationError(PermissionError):
    """Raised when an operation would touch data the caller does not own."""


def require_user_id(user_id: str | None, *, operation: str) -> str:
    if not user_id or not str(user_id).strip():
        raise AuthorizationError(
            f"'{operation}' requires a signed-in user; no verified identity was present."
        )
    return str(user_id).strip()


def scoped_filter(user_id: str, extra: Mapping[str, Any] | None = None, *, operation: str) -> dict[str, Any]:
    owner = require_user_id(user_id, operation=operation)
    query: dict[str, Any] = dict(extra or {})
    if "user_id" in query and str(query["user_id"]) != owner:
        raise AuthorizationError(f"'{operation}' attempted to query another user's records.")
    query["user_id"] = owner
    return query


def assert_owned(document: Mapping[str, Any] | None, user_id: str, *, operation: str) -> Mapping[str, Any] | None:
    if document is None:
        return None
    owner = require_user_id(user_id, operation=operation)
    if str(document.get("user_id")) != owner:
        logger.error(
            "authorization guardrail blocked %s: document owner mismatch (doc=%s)",
            operation,
            document.get("_id"),
        )
        raise AuthorizationError(f"'{operation}' was blocked: that record belongs to a different user.")
    return document


def filter_owned(documents: Iterable[T], user_id: str, *, operation: str) -> list[T]:
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
