"""Verify Supabase access tokens for the MCP transport.

Ported from trip-agent/src/auth/supabase_auth.py's three verification paths
(asymmetric JWKS, symmetric HS256, live Supabase lookup) into FastMCP's
TokenVerifier protocol. This is the only place `user_id` is ever established —
no tool on this server accepts `user_id` as an argument, mirroring the same
invariant trip-agent's guardrails/authorization.py enforces: identity comes
only from a cryptographically verified token, never from anything a caller
(or a model on the other end) supplies directly.
"""

from __future__ import annotations

import logging
import threading

import jwt
import requests
from jwt import PyJWKClient
from fastmcp.server.auth import AccessToken, TokenVerifier

from .config import settings

logger = logging.getLogger(__name__)

_jwks_client: PyJWKClient | None = None
_jwks_lock = threading.Lock()


def _get_jwks_client() -> PyJWKClient | None:
    global _jwks_client
    if _jwks_client is not None:
        return _jwks_client
    if not settings.supabase_url:
        return None
    with _jwks_lock:
        if _jwks_client is None:
            url = f"{settings.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
            _jwks_client = PyJWKClient(url, cache_keys=True, lifespan=600)
    return _jwks_client


def _access_token_from_claims(token: str, claims: dict) -> AccessToken | None:
    user_id = claims.get("sub")
    if not user_id:
        return None
    return AccessToken(
        token=token,
        client_id=str(user_id),
        scopes=[],
        expires_at=claims.get("exp"),
        claims=claims,
    )


def _verify_asymmetric(token: str) -> AccessToken | None:
    client = _get_jwks_client()
    if client is None:
        return None
    try:
        signing_key = client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience=settings.supabase_jwt_audience,
            options={"verify_aud": bool(settings.supabase_jwt_audience)},
        )
        return _access_token_from_claims(token, claims)
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        # Wrong algorithm family or no matching key — let the next path try.
        return None
    except requests.RequestException:
        return None
    except Exception as exc:  # pragma: no cover - unexpected JWKS shape
        logger.debug("asymmetric verification unavailable: %s", exc)
        return None


def _verify_symmetric(token: str) -> AccessToken | None:
    if not settings.supabase_jwt_secret:
        return None
    try:
        claims = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience=settings.supabase_jwt_audience,
            options={"verify_aud": bool(settings.supabase_jwt_audience)},
        )
        return _access_token_from_claims(token, claims)
    except jwt.InvalidTokenError:
        return None


def _verify_remote(token: str) -> AccessToken | None:
    """Ask Supabase directly who this token belongs to."""
    if not (settings.supabase_url and settings.supabase_anon_key):
        return None
    try:
        response = requests.get(
            f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": settings.supabase_anon_key},
            timeout=settings.request_timeout_seconds,
        )
    except requests.RequestException as exc:
        logger.warning("could not reach Supabase to verify a token: %s", exc)
        return None

    if not response.ok:
        return None

    payload = response.json()
    user_id = payload.get("id")
    if not user_id:
        return None
    return AccessToken(token=token, client_id=str(user_id), scopes=[])


class SupabaseTokenVerifier(TokenVerifier):
    """Rejects anything that isn't a live, correctly-signed Supabase token.

    No unverified-token fallback exists here (unlike trip-agent's
    REQUIRE_AUTH escape hatch) — this server only handles destructive
    favorites operations, so it stays strict unconditionally.
    """

    async def verify_token(self, token: str) -> AccessToken | None:
        for verify in (_verify_asymmetric, _verify_symmetric, _verify_remote):
            result = verify(token)
            if result is not None:
                return result
        return None
