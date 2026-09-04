"""Application settings.

Mirrors trip-agent/src/config.py's shape for the values this service actually
needs — it manages the same MongoDB collection and verifies tokens against the
same Supabase project, but has no LLM, no LangGraph, no tool-RAG concerns.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- MongoDB ---- (same cluster/collection trip-agent writes to)
    mongodb_uri: str | None = None
    mongodb_db_name: str = "travel_discovery"
    mongodb_favorites_collection: str = "agent_favorites"

    # ---- Supabase (token verification only) ----
    supabase_url: str | None = None
    supabase_anon_key: str | None = None
    supabase_jwt_secret: str | None = None
    supabase_jwt_audience: str = "authenticated"

    # ---- App ----
    request_timeout_seconds: int = 20
    mcp_port: int = 8001


settings = Settings()
