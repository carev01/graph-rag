from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtractSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    docext_base_url: str
    docext_read_key: str
    docext_verify_tls: bool = False
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    llm_base_url: str = "http://srv-llm.home.lan:8080/v1"
    llm_model: str = "gpt-oss-20b"
    embed_base_url: str = "http://srv-llm.home.lan:8082/v1"
    embed_model: str = "jinaai/jina-embeddings-v5-text-nano-retrieval"
    embed_dim: int = 768
    chonkie_base_url: str = "http://srv-llm.home.lan:8084"
    chonkie_model: str = "mirth/chonky_modernbert_base_1"
    group_id: str = "backup-docs"
    max_chunk_tokens: int = 1800
    min_chunk_tokens: int = 128
    max_coroutines: int = 3
    llm_client_mode: Literal["structured", "generic_json_schema", "generic_json_object"] = "structured"
    judge_base_url: str = ""
    judge_model: str = ""


@lru_cache
def get_extract_settings() -> ExtractSettings:
    return ExtractSettings()  # type: ignore[call-arg]
