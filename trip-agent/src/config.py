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

    amadeus_client_id: str
    amadeus_client_secret: str
    amadeus_environment: str = "production"

    tavily_api_key: str

    langsmith_api_key: str | None = None
    langsmith_project_name: str = "travel-discovery-agent"

    default_max_iterations: int = 6
    request_timeout_seconds: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
