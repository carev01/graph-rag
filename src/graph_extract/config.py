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
    # Optional overrides letting the compatibility harness (src/compat/) target a
    # DIFFERENT Neo4j than the one the rest of the stack uses, without editing the
    # neo4j_* values. Empty means "fall back to the neo4j_* value", field by field.
    compat_neo4j_uri: str = ""
    compat_neo4j_user: str = ""
    compat_neo4j_password: str = ""
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
    # Repetition penalties for the extraction tier. graphiti never sends these (its
    # client passes only model/messages/temperature/max_tokens/response_format) and we
    # pin temperature=0.0, so a weaker model that starts an ascending-integer run in an
    # unbounded list[int] field (prompts/extract_edges.py: episode_indices has no
    # maxItems) cannot escape it -- under a strict JSON schema the only legal next
    # tokens there are digits, ',' and ']'. It then burns the whole max_tokens budget
    # and truncates into invalid JSON. A frequency penalty makes the repeated digit and
    # comma tokens progressively less attractive so ']' eventually wins.
    # Default 0.0 = inject nothing, preserving the proven gpt-5-mini behaviour; raise
    # only for models that need it.
    # Cap on graphiti's three unbounded array<integer> schema fields
    # (episode_indices, duplicate_facts, contradicted_facts). These are index lists
    # whose legitimate length is tiny (one entry per episode/candidate), so a generous
    # cap is inert for a well-behaved model but makes the ascending-integer runaway
    # unrepresentable. 0 disables the bound.
    llm_max_index_array: int = 25
    llm_frequency_penalty: float = 0.0
    llm_presence_penalty: float = 0.0
    judge_base_url: str = ""
    judge_model: str = ""
    judge_api_key: str = ""  # if empty, the judge reuses llm_api_key (fallback path)
    # --- theme-builder / community layer (design: theme-builder-community-layer) ---
    # Report tier defaults to the synthesis/judge tier (GLM-5.2) when left empty.
    # The faithfulness judge for the router golden-set eval. Falls back to judge_*,
    # but eval_router GUARDS against the fallback resolving to the synthesis tier --
    # a judge grading its own output inflates faithfulness, and the failure is
    # invisible in the score. Point this at a DIFFERENT model family from synthesis
    # so the two do not share failure modes.
    eval_judge_base_url: str = ""
    eval_judge_model: str = ""
    eval_judge_api_key: str = ""
    report_llm_base_url: str = ""
    report_llm_model: str = ""
    report_llm_api_key: str = ""
    # The report verifier. Falls back to eval_judge_*. MUST NOT resolve to the
    # report model -- theme_builder.report._verify_client_and_model raises if it
    # does, because a model grading its own findings inflates the result in a way
    # the result cannot reveal.
    verify_llm_base_url: str = ""
    verify_llm_model: str = ""
    verify_llm_api_key: str = ""
    # Reasoning effort for the verifier tier. Measured 2026-09 on an 11-finding
    # report (~deepseek/deepseek-v4-flash-latest, a REASONING model) via
    # theme_builder.report.verify_report: max_tokens=4000, retried at 12000, both
    # exhausted on reasoning tokens alone (finish_reason='length', content=None) --
    # raising the cap does not help, reasoning just expands to fill it. "low" fixed
    # it (finish_reason='stop', 3028 completion tokens, a real verdict flagging 2 of
    # 11 findings unsupported). Measured "enabled: false" too: it "works"
    # (finish_reason='stop', valid JSON) but in 13 completion tokens it approved
    # everything -- a rubber stamp, not a check. NEVER set this to disable
    # reasoning entirely; "low" is the floor that keeps the verifier a judge rather
    # than a reflex.
    verify_reasoning_effort: str = "low"
    leiden_min_community_size: int = 3   # drop dust communities smaller than this
    leiden_max_levels: int = 3           # cap on intermediate Leiden levels
    # Output cap for a community report. 8000 was measured against GLM-5.2 (3000
    # skipped ~24% of communities, 8000 skipped ~0). GLM-5.3-flash reasons more and
    # truncates at 8000 -- and a truncated report is DROPPED silently, because
    # generate_report returns None and writeback only writes communities that have
    # one. Reasoning counts against this budget, so cap the reasoning separately
    # rather than only raising the ceiling.
    report_max_tokens: int = 16000
    # Reasoning effort for the report tier: "low" keeps reasoning (it helps report
    # quality) while stopping it from consuming the whole output budget. Empty
    # string sends no reasoning parameter at all.
    report_reasoning_effort: str = "low"
    report_token_budget: int = 12000     # per-community context budget (~chars/4)
    report_top_entities: int = 30        # member entities included in a report's context
    # --- incremental community refresh (design: incremental-community-refresh) ---
    theme_refresh_jaccard_tau: float = 0.5   # min member-set Jaccard to treat a fresh community as the same as a persisted one
    # --- global (map-reduce) search (design: global-search) ---
    map_llm_base_url: str = ""   # map tier; defaults to the judge/synthesis tier when empty
    map_llm_model: str = ""
    map_llm_api_key: str = ""
    # Pre-rerank shortlist size. Inert once a reranker is configured: rerank_top_n
    # bounds what reaches the map step instead (see rerank_top_n below).
    global_shortlist_k: int = 10
    global_default_level: int = 1
    # --- DRIFT search (design: drift-search) ---
    drift_primer_level: int = 1      # community level the primer shortlists at
    # Pre-rerank primer shortlist size. Same caveat as global_shortlist_k: once a
    # reranker is configured, rerank_top_n bounds the primer set, not this.
    drift_primer_k: int = 5          # reports shortlisted for the primer
    drift_max_followups: int = 4     # follow-ups kept per round (relevance-budgeted)
    drift_followup_k: int = 8        # local-search k per follow-up
    drift_iterations: int = 1        # follow-up rounds; clamped to [1,2] at call time
    # --- /answer router (design: answer-router) ---
    router_default_mode: str = "drift"   # mode when no heuristic fires and the cheap classifier is absent/uncertain
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
    # Cross-encoder reranker (Voyage AI, Cohere-compatible /v1/rerank). Scores
    # community relevance for the global path, replacing the LLM's improvised
    # 0-10 rating. rerank_top_n / rerank_score_floor are set from measured score
    # distributions -- see the 2026-09-10 spec, not guessed.
    rerank_base_url: str = ""
    rerank_model: str = ""
    rerank_api_key: str = ""
    rerank_candidates: int = 50    # cosine pre-cut before the API call
    # Max communities reaching extraction. Once configured, this -- not
    # global_shortlist_k / drift_primer_k -- is what bounds the shortlist.
    rerank_top_n: int = 4          # max communities reaching extraction
    # 0.40, not the provisional 0.45: measured across all 10 golden global/DRIFT
    # questions (docs/superpowers/rerank-threshold-measurement.md). Scores are
    # compressed -- the two communities the inversion complaint was about sit
    # 0.023 apart -- so the floor only removes the clear tail and rerank_top_n
    # does the selecting. 0.45 would have cut the very community that complaint
    # wanted ranked higher; 0.55+ refuses 3 of 10 answerable questions.
    rerank_score_floor: float = 0.40


@lru_cache
def get_extract_settings() -> ExtractSettings:
    return ExtractSettings()  # type: ignore[call-arg]
