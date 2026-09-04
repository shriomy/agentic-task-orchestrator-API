"""Verify Supabase access tokens and derive the acting user id.

The `user_id` this returns is the only identity the rest of the system trusts.
It is what the authorization guardrail pins every MongoDB query to, so it must
come from a cryptographically verified token — never from the request body and
never from anything the model produced.

Three verification paths, tried in order:
  1. Asymmetric (RS256/ES256) via the project's JWKS endpoint — current default
     for new Supabase projects using signing keys.
  2. Symmetric (HS256) with SUPABASE_JWT_SECRET — legacy projects.
  3. A live call to Supabase's /auth/v1/user endpoint — always correct, one
     network round trip, used as the fallback when no local key material exists.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import jwt
import requests
from jwt import PyJWKClient

from ..config import settings

logger = logging.getLogger(__name__)


class AuthError(Exception):
    """The caller could not be authenticated."""


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    email: str | None = None
    role: str = "authenticated"
    # The raw, already-verified bearer token — forwarded as-is to services
    # (e.g. favorites-mcp-server) that need to independently verify identity
    # themselves rather than trusting a derived user_id passed as an argument.
    token: str = ""


_jwks_client: PyJWKClient | None = None
_jwks_lock = threading.Lock()

# /auth/v1/user results are cached briefly so a burst of requests on one thread
# does not become one Supabase round trip per message.
_user_cache: dict[str, tuple[float, AuthenticatedUser]] = {}
_USER_CACHE_TTL = 60.0


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


def _user_from_claims(claims: dict[str, Any], token: str = "") -> AuthenticatedUser:
    user_id = claims.get("sub")
    if not user_id:
        raise AuthError("Token has no subject claim.")
    return AuthenticatedUser(
        user_id=str(user_id),
        email=claims.get("email"),
        role=str(claims.get("role") or "authenticated"),
        token=token,
    )


def _verify_asymmetric(token: str) -> AuthenticatedUser | None:
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
        return _user_from_claims(claims, token)
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Session expired; please sign in again.") from exc
    except jwt.InvalidTokenError:
        # Wrong algorithm family or no matching key — let the next path try.
        return None
    except requests.RequestException:
        return None
    except Exception as exc:  # pragma: no cover - unexpected JWKS shape
        logger.debug("asymmetric verification unavailable: %s", exc)
        return None


def _verify_symmetric(token: str) -> AuthenticatedUser | None:
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
        return _user_from_claims(claims, token)
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Session expired; please sign in again.") from exc
    except jwt.InvalidTokenError:
        return None


def _verify_remote(token: str) -> AuthenticatedUser | None:
    """Ask Supabase directly who this token belongs to."""
    if not (settings.supabase_url and settings.supabase_anon_key):
        return None

    cached = _user_cache.get(token)
    if cached and cached[0] > time.monotonic():
        return cached[1]

    try:
        response = requests.get(
            f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": settings.supabase_anon_key},
            timeout=settings.request_timeout_seconds,
        )
    except requests.RequestException as exc:
        logger.warning("could not reach Supabase to verify a token: %s", exc)
        return None

    if response.status_code == 401:
        raise AuthError("Invalid or expired session.")
    if not response.ok:
        return None

    payload = response.json()
    user_id = payload.get("id")
    if not user_id:
        return None
    user = AuthenticatedUser(
        user_id=str(user_id),
        email=payload.get("email"),
        role=str(payload.get("role") or "authenticated"),
        token=token,
    )
    _user_cache[token] = (time.monotonic() + _USER_CACHE_TTL, user)
    return user


def verify_access_token(token: str | None) -> AuthenticatedUser:
    """Verify a Supabase access token and return the acting user.

    Raises AuthError on anything unverifiable. When `require_auth` is off the
    token's unverified subject is accepted so the graph can be exercised
    locally without a live Supabase project — never enable that in production.
    """
    cleaned = (token or "").strip()
    if cleaned.lower().startswith("bearer "):
        cleaned = cleaned[7:].strip()

    if not cleaned:
        raise AuthError("A Supabase access token is required.")

    for verify in (_verify_asymmetric, _verify_symmetric, _verify_remote):
        user = verify(cleaned)
        if user is not None:
            return user

    if not settings.require_auth:
        logger.warning("REQUIRE_AUTH is false: accepting an unverified token subject.")
        try:
            claims = jwt.decode(cleaned, options={"verify_signature": False})
            return _user_from_claims(claims, cleaned)
        except Exception as exc:
            raise AuthError("Token could not be read.") from exc

    raise AuthError("Could not verify your session. Please sign in again.")
