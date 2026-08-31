"""Application settings.

Every external credential and endpoint is read from the environment here and
nowhere else, so no module ever hardcodes a key or a base URL.
"""

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "trip-agent/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM -------------------------------------------------------------
    llm_provider: str = "openai"
    llm_api_key: str | None = None
    llm_api_base: str | None = None
    llm_model: str = "gpt-4.1-mini"
    # Cheap/fast model used by the classifier guardrails and the summarizer.
    llm_fast_model: str = "gpt-4o-mini"

    openrouter_api_key: str | None = None
    # The .env in this repo spells it OPENROUTER_BASE_URL; accept both spellings.
    openrouter_api_base: str | None = Field(
        default=None, validation_alias="OPENROUTER_BASE_URL"
    )
    openrouter_default_model: str | None = None

    # ---- Supabase --------------------------------------------------------
    supabase_url: str | None = None
    supabase_anon_key: str | None = None
    supabase_service_role_key: str | None = None
    # Postgres connection string behind Supabase, used by the LangGraph checkpointer.
    supabase_db_url: str | None = None
    # Legacy HS256 projects sign access tokens with this shared secret.
    supabase_jwt_secret: str | None = None
    supabase_jwt_audience: str = "authenticated"

    # ---- Tavily ----------------------------------------------------------
    tavily_api_key: str | None = None
    tavily_root_url: str = "https://api.tavily.com"

    # ---- OpenTripMap -----------------------------------------------------
    opentripmap_api_key: str | None = None
    opentripmap_root_url: str = "https://api.opentripmap.com/0.1"
    opentripmap_lang: str = "en"

    # ---- Ticketmaster ----------------------------------------------------
    ticketmaster_api_key: str | None = None
    ticketmaster_root_url: str = "https://app.ticketmaster.com/discovery/v2/"

    # ---- Booking.com via RapidAPI ---------------------------------------
    # The .env spells the key X_RapidAPI_Key; HOTEL_SEARCH_API_KEY is the
    # older name kept as a fallback so both work.
    rapidapi_key: str | None = Field(default=None, validation_alias="X_RAPIDAPI_KEY")
    hotel_search_api_key: str | None = None
    rapidapi_booking_host: str = "booking-com15.p.rapidapi.com"

    @property
    def booking_api_key(self) -> str | None:
        return self.rapidapi_key or self.hotel_search_api_key

    @property
    def booking_root_url(self) -> str:
        return f"https://{self.rapidapi_booking_host}/api/v1"

    # ---- MongoDB ---------------------------------------------------------
    mongodb_uri: str | None = None
    mongodb_db_name: str = "travel_discovery"
    mongodb_favorites_collection: str = "agent_favorites"

    # ---- Tool RAG (Qdrant) -------------------------------------------------
    # Semantic tool retrieval: only tools relevant to the turn's request are
    # bound to the model, instead of always sending every tool schema.
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "trip_agent_tools"
    tool_rag_enabled: bool = True
    tool_rag_top_k: int = 3

    # ---- Tracing ---------------------------------------------------------
    langsmith_api_key: str | None = None
    langsmith_project_name: str = "travel-discovery-agent"
    langsmith_tracing: bool = True
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # ---- Behaviour tuning ------------------------------------------------
    default_max_iterations: int = 6
    request_timeout_seconds: int = 20
    # Hard ceiling on agent<->tool round trips inside a single turn.
    max_tool_rounds: int = 12
    # Message-history token budget before the summarization node compresses.
    summary_token_threshold: int = 6000

    postgres_pool_min: int = 1
    postgres_pool_max: int = 10
    postgres_pool_timeout: float = 30.0

    # Comma-separated origins allowed to call the API from a browser.
    cors_allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # When true, an unverifiable bearer token is rejected instead of trusted.
    require_auth: bool = True

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


settings = Settings()


def configure_tracing() -> None:
    """Export the env vars LangSmith's auto-instrumentation reads.

    LangChain/LangGraph trace automatically once these are set, so no node has
    to wrap itself in an explicit tracing context.
    """
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        # Make sure a stale shell export doesn't turn tracing on without a key.
        os.environ.setdefault("LANGSMITH_TRACING", "false")
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project_name
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint


configure_tracing()
