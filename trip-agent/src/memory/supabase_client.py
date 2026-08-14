"""Lazily-constructed Supabase clients.

Built on first use rather than at import time, so the module can be imported
(and the graph inspected/tested) without live Supabase credentials.
"""

from __future__ import annotations

import threading
from typing import Any

from ..config import settings

_anon_client: Any = None
_service_client: Any = None
_lock = threading.Lock()


def supabase_available() -> bool:
    return bool(settings.supabase_url and settings.supabase_service_role_key)


def get_anon_client() -> Any:
    """Client bound to the anon key; every query is subject to RLS."""
    global _anon_client
    if _anon_client is not None:
        return _anon_client
    if not (settings.supabase_url and settings.supabase_anon_key):
        raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY are required.")
    with _lock:
        if _anon_client is None:
            from supabase import create_client

            _anon_client = create_client(settings.supabase_url, settings.supabase_anon_key)
    return _anon_client


def get_service_client() -> Any:
    """Client bound to the service role key.

    This bypasses RLS, so every call site must scope its own queries by
    user_id. The authorization guardrail exists precisely because this key
    removes the database's own safety net.
    """
    global _service_client
    if _service_client is not None:
        return _service_client
    if not supabase_available():
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required.")
    with _lock:
        if _service_client is None:
            from supabase import create_client

            _service_client = create_client(settings.supabase_url, settings.supabase_service_role_key)
    return _service_client
