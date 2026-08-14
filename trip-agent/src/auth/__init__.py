"""Supabase authentication for the trip agent API."""

from .supabase_auth import AuthenticatedUser, AuthError, verify_access_token

__all__ = ["AuthenticatedUser", "AuthError", "verify_access_token"]
