from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    docext_base_url: str
    docext_read_key: str
    docext_admin_key: str = ""
    docext_verify_tls: bool = False
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    postgres_dsn: str
    webhook_secret: str = ""
    webhook_public_url: str = ""
    poll_interval_seconds: int = 1800
    webhook_debounce_seconds: int = 300
    semantic_daily_token_budget: int = 5_000_000
    semantic_max_attempts: int = 5
    semantic_backoff_base_seconds: float = 30.0
    semantic_backoff_cap_seconds: float = 3600.0
    semantic_reaper_lease_seconds: float = 1800.0

@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
