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
    # API key for the LLM endpoint. Local llama-server ignores it ("not-needed");
    # a cloud endpoint (e.g. OpenRouter) needs a real key, supplied via LLM_API_KEY
    # in .env (never committed). Only the LLM/extraction path uses this — the
    # embedder stays local (TEI/Jina, "not-needed").
    llm_api_key: str = "not-needed"
    embed_base_url: str = "http://srv-llm.home.lan:8082/v1"
    embed_model: str = "jinaai/jina-embeddings-v5-text-nano-retrieval"
    embed_dim: int = 768
    chonkie_base_url: str = "http://srv-llm.home.lan:8084"
    chonkie_model: str = "mirth/chonky_modernbert_base_1"
    group_id: str = "backup-docs"
    max_chunk_tokens: int = 1800
    min_chunk_tokens: int = 128
    max_coroutines: int = 3
    # Default set to generic_json_schema per Task 6 evidence: gpt-oss-20b via
    # llama-server fails the OpenAIClient "structured" (Responses API) path
    # (markdown-fenced/malformed JSON, 0 entities), but OpenAIGenericClient
    # with json_schema extracts cleanly. Task 7's probe compares json_schema
    # vs json_object and confirms this.
    llm_client_mode: Literal["structured", "generic_json_schema", "generic_json_object"] = "generic_json_schema"
    judge_base_url: str = ""
    judge_model: str = ""
    judge_api_key: str = ""  # if empty, the judge reuses llm_api_key (fallback path)


@lru_cache
def get_extract_settings() -> ExtractSettings:
    return ExtractSettings()  # type: ignore[call-arg]
