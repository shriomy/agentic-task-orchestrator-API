from pydantic import BaseSettings


class Settings(BaseSettings):
    llm_provider: str = "openai"
    llm_api_key: str
    llm_api_base: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4.1-mini"

    supabase_url: str
    supabase_service_role_key: str
    supabase_anon_key: str
    supabase_db_url: str

    tavily_api_key: str
    opentripmap_api_key: str | None = None
    ticketmaster_api_key: str | None = None
    hotel_search_provider: str = "amadeus"
    hotel_search_api_key: str | None = None

    mongodb_uri: str | None = None
    mongodb_db_name: str = "travel_discovery"
    mongodb_favorites_collection: str = "favorites"

    langsmith_api_key: str | None = None
    langsmith_project_name: str = "travel-discovery-agent"

    openrouter_api_key: str | None = None
    openrouter_api_base: str | None = None

    default_max_iterations: int = 6
    request_timeout_seconds: int = 20
    postgres_pool_min: int = 4
    postgres_pool_max: int | None = None
    postgres_pool_timeout: float = 30.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
