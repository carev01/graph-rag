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
    # Gates the BOOTSTRAP lane only (`run_worker_once`: bootstrap jobs are claimed
    # while today's tokens are under this; the incremental lane always runs).
    # At ~200k tokens/article the old 5M default was ~25 articles/day -- roughly
    # 14 YEARS for the 126k-article corpus, which silently capped every throughput
    # gain the concurrency work bought. 840M/day is ~4,200 articles/day, a ~30-day
    # bootstrap, and at the measured ~$0.12/article about $500/day.
    # It is a throttle, not a cost cap: raise or lower it deliberately.
    semantic_daily_token_budget: int = 840_000_000
    semantic_max_attempts: int = 5
    semantic_backoff_base_seconds: float = 30.0
    semantic_backoff_cap_seconds: float = 3600.0
    semantic_reaper_lease_seconds: float = 1800.0
    # Serialise warm-up (cold) articles across worker PROCESSES, so scaling out
    # to N workers keeps the cross-source hub protection a single worker gets
    # from the in-process barrier. ON by default: with one worker it costs one
    # Postgres round-trip per cold article against ~5 minutes of LLM work, and
    # defaulting it off would make scaling out silently lose the guarantee.
    semantic_global_warmup_lock: bool = True
    semantic_warmup_lock_timeout_seconds: float = 1800.0
    # Scopes the worker's `claim_semantic_jobs` to a subset of sources -- the
    # k3s bootstrap-first rehearsal (docs/deploy/k3s.md): comma-separated
    # source ids, e.g. "s1,s2". Empty (the default) is unscoped, i.e. today's
    # behaviour: claim across every source. Set via `SEMANTIC_CLAIM_SOURCE_IDS`.
    semantic_claim_source_ids: str = ""

    @property
    def semantic_claim_source_id_list(self) -> list[str]:
        """Parsed, stripped, non-empty ids from `semantic_claim_source_ids`."""
        return [s.strip() for s in self.semantic_claim_source_ids.split(",") if s.strip()]

@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
