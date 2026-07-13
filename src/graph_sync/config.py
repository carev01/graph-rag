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

@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
