from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtractSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    docext_base_url: str
    docext_read_key: str
    docext_admin_key: str = ""  # passed to docext.client.make_docext_client(admin_key=...)
    docext_verify_tls: bool = False
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    llm_base_url: str = "http://srv-llm.home.lan:8080/v1"
    # gpt-5-mini is the chosen extraction tier (Azure Responses API,
    # reasoning=minimal). gpt-oss-120b was evaluated 2026-07 (same Azure
    # endpoint, cheaper) but on an identical 8-article sample it ran ~2x
    # slower and lost AvailableIn/region extraction (0 vs 6 AvailableIn facts,
    # 1 vs 4 regions) -> NO-GO. The pipeline still supports it via .env
    # (LLM_MODEL=gpt-oss-120b, generic_json_schema, reasoning_effort=low) if
    # revisited. Azure endpoint/api-version/key come from .env (never committed).
    llm_model: str = "gpt-5-mini"
    # API key for the LLM endpoint. Local llama-server ignores it ("not-needed");
    # a cloud endpoint (e.g. OpenRouter) needs a real key, supplied via LLM_API_KEY
    # in .env (never committed). Only the LLM/extraction path uses this — the
    # embedder stays local (TEI/Jina, "not-needed").
    llm_api_key: str = "not-needed"
    # Azure OpenAI: when llm_base_url points at *.azure.com/*.cognitiveservices,
    # an AsyncAzureOpenAI client is used (Responses API + structured mode). Set
    # the api version and (for reasoning models like gpt-5-mini) the effort.
    llm_api_version: str = ""
    llm_reasoning_effort: str = "minimal"  # minimal|low|medium|high (gpt-5 family); gpt-oss uses low|medium|high
    embed_base_url: str = "http://srv-llm.home.lan:8082/v1"
    embed_model: str = "jinaai/jina-embeddings-v5-text-nano-retrieval"
    embed_dim: int = 768
    embed_max_batch: int = 32  # TEI/Jina max_client_batch_size; cap embed batches to this
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
    # --- hybrid extraction routing (design: hybrid-extraction-router) ---
    # ON by default; degrades to strong-only when cheap_llm_api_key is empty.
    extraction_routing: bool = True
    # Cheap tier = ling-2.6-flash via OpenRouter. api_key from .env (never committed).
    cheap_llm_base_url: str = "https://openrouter.ai/api/v1"
    cheap_llm_model: str = "inclusionai/ling-2.6-flash"
    cheap_llm_api_key: str = ""
    cheap_llm_client_mode: Literal[
        "structured", "generic_json_schema", "generic_json_object"] = "generic_json_schema"
    cheap_max_chunk_tokens: int = 900   # smaller chunks for the verbose cheap model
    # Route an article to the STRONG tier when it is a dense table:
    dense_table_line_ratio: float = 0.25
    dense_pipe_count: int = 200


@lru_cache
def get_extract_settings() -> ExtractSettings:
    return ExtractSettings()  # type: ignore[call-arg]
