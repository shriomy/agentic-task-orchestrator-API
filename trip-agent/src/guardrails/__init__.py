"""Guardrails: scope (input), redaction/intent-split (output), authorization (data)."""

from .authorization import (
    AuthorizationError,
    assert_owned,
    filter_owned,
    require_user_id,
    scoped_filter,
)
from .output import (
    OutputVerdict,
    RequestSplit,
    apply_output_guardrail,
    has_leaks,
    redact,
    split_request,
)
from .scope import ScopeVerdict, classify_scope

__all__ = [
    "AuthorizationError",
    "assert_owned",
    "filter_owned",
    "require_user_id",
    "scoped_filter",
    "OutputVerdict",
    "RequestSplit",
    "apply_output_guardrail",
    "has_leaks",
    "redact",
    "split_request",
    "ScopeVerdict",
    "classify_scope",
]
