"""Shared HTTP helper for the outbound travel APIs.

Centralises the rules from tool_query.md §7 that every wrapper must follow:
always set a timeout, always raise on non-2xx, and never send empty params.
"""

from typing import Any

import requests

from ..config import settings


class ToolError(RuntimeError):
    """A tool failed in a way the agent is allowed to see and react to.

    The message is written for the model, so it must never contain a key,
    a full URL with credentials, or a raw stack trace.
    """


def compact(params: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is None or an empty string."""
    return {key: value for key, value in params.items() if value is not None and value != ""}


def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    provider: str = "upstream provider",
) -> Any:
    try:
        response = requests.get(
            url,
            params=compact(params or {}),
            headers=headers,
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        if status == 401 or status == 403:
            raise ToolError(f"{provider} rejected the request (not authorised).") from exc
        if status == 429:
            raise ToolError(f"{provider} rate limit reached; try again shortly.") from exc
        raise ToolError(f"{provider} returned an error (status {status}).") from exc
    except requests.Timeout as exc:
        raise ToolError(f"{provider} timed out after {settings.request_timeout_seconds}s.") from exc
    except requests.RequestException as exc:
        raise ToolError(f"Could not reach {provider}.") from exc
    except ValueError as exc:  # non-JSON body
        raise ToolError(f"{provider} returned a malformed response.") from exc


def post_json(
    url: str,
    *,
    json_body: dict[str, Any],
    headers: dict[str, str] | None = None,
    provider: str = "upstream provider",
) -> Any:
    try:
        response = requests.post(
            url,
            json=compact(json_body),
            headers=headers,
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        if status in (401, 403):
            raise ToolError(f"{provider} rejected the request (not authorised).") from exc
        if status == 429:
            raise ToolError(f"{provider} rate limit reached; try again shortly.") from exc
        raise ToolError(f"{provider} returned an error (status {status}).") from exc
    except requests.Timeout as exc:
        raise ToolError(f"{provider} timed out after {settings.request_timeout_seconds}s.") from exc
    except requests.RequestException as exc:
        raise ToolError(f"Could not reach {provider}.") from exc
    except ValueError as exc:
        raise ToolError(f"{provider} returned a malformed response.") from exc
