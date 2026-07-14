from __future__ import annotations
from datetime import datetime
from openai import AsyncAzureOpenAI, AsyncOpenAI
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.graphiti import AddEpisodeResults
from graph_extract.config import ExtractSettings
from graph_extract.usage import instrument
from graph_extract.ontology import (
    ENTITY_TYPES, EDGE_TYPES, EDGE_TYPE_MAP, EXCLUDED_ENTITY_TYPES,
    EXTRACTION_INSTRUCTIONS,
)

def _llm_config(s: ExtractSettings) -> LLMConfig:
    return LLMConfig(api_key=s.llm_api_key, model=s.llm_model,
                     small_model=s.llm_model, base_url=s.llm_base_url, temperature=0.0)

def _inject_openrouter_provider(client: AsyncOpenAI) -> AsyncOpenAI:
    """For OpenRouter endpoints, attach a provider preference so structured-
    output extraction requests can route to a matching provider.

    Graphiti's strict-json_schema extraction requests otherwise 404 with
    'No endpoints available matching your guardrail restrictions and data
    policy' when the account's privacy policy excludes the providers that
    serve this model with structured outputs. `data_collection: allow` +
    `allow_fallbacks` widens the eligible set for these requests.
    """
    orig = client.chat.completions.create

    async def create(*args, **kwargs):  # type: ignore[no-untyped-def]
        extra = dict(kwargs.get("extra_body") or {})
        # require_parameters: only route to providers that actually support the
        # request params (response_format/json_schema) -- prevents routing to a
        # provider that ignores structured output and returns malformed/truncated
        # JSON. data_collection=allow widens past the account data-policy 404.
        extra.setdefault("provider", {"data_collection": "allow",
                                      "require_parameters": True,
                                      "allow_fallbacks": True})
        kwargs["extra_body"] = extra
        return await orig(*args, **kwargs)

    client.chat.completions.create = create  # type: ignore[assignment]
    return client


def _is_azure(base_url: str) -> bool:
    return "azure.com" in base_url or "cognitiveservices" in base_url

def _llm_client(s: ExtractSettings):
    cfg = _llm_config(s)
    # Explicit timeout + retries: a cloud endpoint can drop a connection
    # (observed: an OpenRouter socket stuck in CLOSE_WAIT hung the whole run).
    # A bounded per-request timeout makes a dead request abort and retry.
    if _is_azure(s.llm_base_url):
        # Azure OpenAI: Responses API + structured mode (native reasoning). The
        # base_url is the resource endpoint (host); the SDK builds /openai/...
        raw = instrument(AsyncAzureOpenAI(
            azure_endpoint=s.llm_base_url, api_key=s.llm_api_key,
            api_version=s.llm_api_version, timeout=90.0, max_retries=4))
        return OpenAIClient(config=cfg, client=raw,
                            reasoning=s.llm_reasoning_effort, verbosity="low")
    raw = instrument(AsyncOpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url,
                                 timeout=90.0, max_retries=4))
    if "openrouter" in s.llm_base_url:
        raw = _inject_openrouter_provider(raw)
    if s.llm_client_mode == "structured":
        return OpenAIClient(config=cfg, client=raw,
                            reasoning=s.llm_reasoning_effort, verbosity="low")
    mode = "json_schema" if s.llm_client_mode == "generic_json_schema" else "json_object"
    return OpenAIGenericClient(config=cfg, client=raw, structured_output_mode=mode)

def _batch_capped_embeddings(client: AsyncOpenAI, max_batch: int) -> AsyncOpenAI:
    """Cap embedding requests to max_batch inputs per call.

    TEI/Jina enforces max_client_batch_size (32); graphiti can send larger
    batches (observed: 34), which TEI rejects with HTTP 422 and drops the whole
    article. Split oversized inputs into <=max_batch sub-batches, preserving order.
    """
    orig = client.embeddings.create

    async def create(*args, **kwargs):  # type: ignore[no-untyped-def]
        inp = kwargs.get("input")
        if isinstance(inp, list) and len(inp) > max_batch:
            merged = None
            data = []
            for i in range(0, len(inp), max_batch):
                sub = dict(kwargs)
                sub["input"] = inp[i:i + max_batch]
                r = await orig(*args, **sub)
                merged = r
                data.extend(r.data)
            for idx, d in enumerate(data):
                d.index = idx
            merged.data = data  # type: ignore[union-attr]
            return merged
        return await orig(*args, **kwargs)

    client.embeddings.create = create  # type: ignore[assignment]
    return client


def build_graphiti(s: ExtractSettings) -> Graphiti:
    embed_client = _batch_capped_embeddings(
        AsyncOpenAI(api_key="not-needed", base_url=s.embed_base_url,
                    timeout=90.0, max_retries=4), s.embed_max_batch)
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key="not-needed", embedding_model=s.embed_model,
            embedding_dim=s.embed_dim, base_url=s.embed_base_url),
        client=embed_client)
    # CRITICAL: pass an explicit LOCAL reranker so Graphiti does not build its
    # default OpenAIRerankerClient() pointed at api.openai.com.
    reranker = OpenAIRerankerClient(config=_llm_config(s))
    return Graphiti(s.neo4j_uri, s.neo4j_user, s.neo4j_password,
                    llm_client=_llm_client(s), embedder=embedder,
                    cross_encoder=reranker, max_coroutines=s.max_coroutines)

async def init_indices(graphiti: Graphiti) -> None:
    await graphiti.build_indices_and_constraints()

async def add_text_episode(graphiti: Graphiti, s: ExtractSettings, *, name: str,
                           body: str, source_description: str,
                           reference_time: datetime) -> AddEpisodeResults:
    return await graphiti.add_episode(
        name=name, episode_body=body, source_description=source_description,
        reference_time=reference_time, source=EpisodeType.text, group_id=s.group_id,
        entity_types=ENTITY_TYPES, excluded_entity_types=EXCLUDED_ENTITY_TYPES,
        edge_types=EDGE_TYPES, edge_type_map=EDGE_TYPE_MAP,
        custom_extraction_instructions=EXTRACTION_INSTRUCTIONS)
