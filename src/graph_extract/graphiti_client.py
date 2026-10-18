from __future__ import annotations
import copy
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from openai import AsyncAzureOpenAI, AsyncOpenAI
from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.graphiti import AddEpisodeResults
from graph_extract.anthropic_chat import AnthropicChat, is_anthropic
from graph_extract.cache_layout import StaticTailLearner, wrap_client
from graph_extract.config import ExtractSettings
from graph_extract.contradiction_gate import install_contradiction_gate
from graph_extract.deterministic_valid_at import install_deterministic_valid_at
from graph_extract.lean_edge_search import install_lean_edge_search
from graph_extract.usage import instrument
from graph_extract.vector_search import (
    DEFAULT_STARTUP_WAIT_SECONDS,
    ensure_vector_indexes,
    install_vector_search,
)
from graph_extract.ontology import (
    ENTITY_TYPES, EDGE_TYPES, EDGE_TYPE_MAP, EXCLUDED_ENTITY_TYPES,
    EXTRACTION_INSTRUCTIONS,
)

@dataclass
class ExtractionTier:
    """One extraction tier: which graphiti client, which instructions, which chunk
    size. Two are used by the hybrid router (cheap=ling, strong=gpt-5-mini)."""
    name: str            # "cheap" | "strong" -> IngestArticleResult.tier
    graphiti: Graphiti
    instructions: str
    max_chunk_tokens: int

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

    async def create(*args, **kwargs):
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

    client.chat.completions.create = create  # type: ignore[method-assign]
    return client


def _inject_penalties(client: AsyncOpenAI, *, frequency_penalty: float,
                      presence_penalty: float) -> AsyncOpenAI:
    """Attach repetition penalties to every chat completion.

    Why this exists: graphiti's `episode_indices` field is an unbounded `list[int]`
    (prompts/extract_edges.py) and we pin temperature=0.0. Greedy decoding plus an
    unbounded array is a trap -- once a weaker model emits `[0, 1, 2` the most likely
    continuation is another integer, and under a strict JSON schema the grammar masks
    out every token that could break the pattern (prose, another key), leaving only
    digits, ',' and ']'. The model counts until it exhausts max_tokens and the reply
    truncates into invalid JSON. Observed on mistral-nemo and solar-pro4; gpt-5-mini
    escapes it only by assigning low probability to continuing the run.

    A frequency penalty scales with how often a token has already been emitted, so the
    repeated digit/comma tokens decay in attractiveness until ']' wins. This is a
    decoding-level fix, so it protects every unbounded field, not just this one.
    """
    orig = client.chat.completions.create

    async def create(*args, **kwargs):
        kwargs.setdefault("frequency_penalty", frequency_penalty)
        kwargs.setdefault("presence_penalty", presence_penalty)
        return await orig(*args, **kwargs)

    client.chat.completions.create = create  # type: ignore[method-assign]
    return client


def _bound_index_arrays(client: AsyncOpenAI, max_items: int) -> AsyncOpenAI:
    """Add `maxItems` to every unbounded array-of-integer in the request schema.

    graphiti emits three such fields, all index lists, all unbounded:
    `ExtractedEdges.Edge.episode_indices`, `EdgeDuplicate.duplicate_facts` and
    `EdgeDuplicate.contradicted_facts`. Nothing stops a model from enumerating
    forever in them -- an ascending integer run is the strongest sequence prior
    there is, and under a JSON grammar the only legal continuations at that
    position are digits, ',' and ']'. Observed twice: an `episode_indices` run
    past 2,900 that exhausted max_tokens and truncated into invalid JSON, and
    dedup indices referring to candidates that never existed.

    Bounding the arrays makes the runaway *unrepresentable* rather than merely
    discouraged, and unlike a repetition penalty it cannot suppress legitimate
    output elsewhere -- arrays of OBJECTS (the `edges` list itself) are left
    alone, so fact yield is untouched.

    Caveat: graphiti deliberately omits `strict: true` from the response_format
    (see openai_generic_client._build_response_format), so enforcement is
    provider-dependent -- treat this as strong pressure, not a hard guarantee.
    """
    orig = client.chat.completions.create

    def _bound(node: object) -> None:
        if isinstance(node, dict):
            items = node.get("items")
            if (node.get("type") == "array" and isinstance(items, dict)
                    and items.get("type") == "integer" and "maxItems" not in node):
                node["maxItems"] = max_items
            for value in node.values():
                _bound(value)
        elif isinstance(node, list):
            for value in node:
                _bound(value)

    async def create(*args, **kwargs):
        fmt = kwargs.get("response_format")
        if isinstance(fmt, dict):
            fmt = copy.deepcopy(fmt)          # never mutate graphiti's own dict
            _bound(fmt)
            kwargs["response_format"] = fmt
        return await orig(*args, **kwargs)

    client.chat.completions.create = create  # type: ignore[method-assign]
    return client


def strict_schema(schema: dict) -> dict:
    """`schema` in OpenAI's strict structured-output form: every object closed and
    every property required. Optional fields keep their `null` alternative, so the
    model writes null instead of omitting them. `properties` / `$defs` maps hold
    field and definition names, never schema keywords, and are not treated as nodes."""
    out = copy.deepcopy(schema)

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                if isinstance(node.get("properties"), dict):
                    node["required"] = list(node["properties"])
            for key, value in node.items():
                if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
                    for sub in value.values():
                        walk(sub)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(out)
    return out


def _strict_schemas(client: AsyncOpenAI) -> AsyncOpenAI:
    """Rewrite each json_schema response_format with `strict_schema` (never
    mutating graphiti's own dict)."""
    orig = client.chat.completions.create

    async def create(*args, **kwargs):
        fmt = kwargs.get("response_format")
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            fmt = copy.deepcopy(fmt)
            fmt["json_schema"]["schema"] = strict_schema(fmt["json_schema"]["schema"])
            kwargs["response_format"] = fmt
        return await orig(*args, **kwargs)

    client.chat.completions.create = create  # type: ignore[method-assign]
    return client


def _inject_openrouter_reasoning(client: AsyncOpenAI, effort: str) -> AsyncOpenAI:
    orig = client.chat.completions.create

    async def create(*args, **kwargs):
        extra = dict(kwargs.get("extra_body") or {})
        extra.setdefault("reasoning", {"effort": effort})
        kwargs["extra_body"] = extra
        return await orig(*args, **kwargs)

    client.chat.completions.create = create  # type: ignore[method-assign]
    return client


def _drop_temperature(client: AsyncOpenAI) -> AsyncOpenAI:
    orig = client.chat.completions.create

    async def create(*args, **kwargs):
        kwargs.pop("temperature", None)
        return await orig(*args, **kwargs)

    client.chat.completions.create = create  # type: ignore[method-assign]
    return client


def _is_azure(base_url: str) -> bool:
    return "azure.com" in base_url or "cognitiveservices" in base_url

def _llm_client(s: ExtractSettings, *, tier: str = "strong", transport: Any = None):
    """graphiti's LLM client for one tier.

    `transport`, when given, replaces ONLY the raw `AsyncOpenAI(...)` of the
    chat-completions branch below; every wrapper applied to the raw client is
    applied to it, in the same order (the distillation harness swaps in a fake
    AsyncOpenAI that serves teacher answers). It must expose a writable
    `chat.completions.create`. The Azure (Responses API) and Anthropic (own
    adapter, strips maxItems) branches build a different transport surface, so a
    `transport` there raises ValueError rather than silently skipping a wrapper.
    """
    cfg = _llm_config(s)
    if transport is not None and (_is_azure(s.llm_base_url) or is_anthropic(s.llm_base_url)):
        raise ValueError(
            "an injected transport is only supported on the OpenAI-compatible "
            f"chat-completions branch, not for base URL {s.llm_base_url!r} "
            "(Azure/Anthropic build their own transport)")
    # Explicit timeout + retries: a cloud endpoint can drop a connection
    # (observed: an OpenRouter socket stuck in CLOSE_WAIT hung the whole run).
    # A bounded per-request timeout makes a dead request abort and retry.
    if _is_azure(s.llm_base_url):
        # Azure OpenAI: Responses API + structured mode (native reasoning). The
        # base_url is the resource endpoint (host); the SDK builds /openai/...
        raw = instrument(AsyncAzureOpenAI(
            azure_endpoint=s.llm_base_url, api_key=s.llm_api_key,
            api_version=s.llm_api_version, timeout=s.llm_timeout_seconds, max_retries=4),
            tier=tier, capture_path=s.llm_capture_path)
        return OpenAIClient(config=cfg, client=raw,
                            reasoning=s.llm_reasoning_effort, verbosity="low")
    if is_anthropic(s.llm_base_url):
        # Claude (BACKLOG 59): graphiti's generic client over the chat-completions
        # adapter, which turns its json_schema response_format into Claude's
        # structured outputs and caches the system block -- where cache_layout puts
        # the static prompt tails. Only json_schema mode exists on Claude: the
        # Responses-API "structured" client and json_object have no equivalent.
        # The maxItems that _bound_index_arrays adds is stripped by the adapter
        # (Claude rejects it) and checked on the reply instead (ARRAY_OVERRUNS).
        claude = instrument(AnthropicChat(
            api_key=s.llm_api_key, reasoning_effort=s.llm_reasoning_effort,
            timeout=s.llm_timeout_seconds, max_retries=4),
            tier=tier, capture_path=s.llm_capture_path)
        if s.llm_max_index_array:
            claude = _bound_index_arrays(claude, s.llm_max_index_array)
        if s.llm_cache_layout:
            claude = wrap_client(claude, StaticTailLearner())
        return OpenAIGenericClient(config=cfg, client=claude,
                                   structured_output_mode="json_schema")
    if transport is None:
        transport = AsyncOpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url,
                                timeout=s.llm_timeout_seconds, max_retries=4)
    raw = instrument(transport, tier=tier, capture_path=s.llm_capture_path)
    if "openrouter" in s.llm_base_url:
        raw = _inject_openrouter_provider(raw)
        if s.llm_openrouter_reasoning_effort:
            raw = _inject_openrouter_reasoning(raw, s.llm_openrouter_reasoning_effort)
    # Only wrap when configured: 0.0/0.0 leaves the proven gpt-5-mini path untouched
    # (and avoids sending params some reasoning endpoints reject).
    if s.llm_frequency_penalty or s.llm_presence_penalty:
        raw = _inject_penalties(raw, frequency_penalty=s.llm_frequency_penalty,
                                presence_penalty=s.llm_presence_penalty)
    if s.llm_max_index_array:
        raw = _bound_index_arrays(raw, s.llm_max_index_array)
    if s.llm_strict_schema:
        raw = _strict_schemas(raw)
    if not s.llm_send_temperature:
        raw = _drop_temperature(raw)
    if s.llm_cache_layout:
        # Static prompt tails into the system message, learned per prompt type
        # (graph_extract.cache_layout); one learner per client, i.e. per tier.
        raw = wrap_client(raw, StaticTailLearner())
    if s.llm_client_mode == "structured":
        return OpenAIClient(config=cfg, client=raw,
                            reasoning=s.llm_reasoning_effort, verbosity="low")
    mode: Literal["json_schema", "json_object"] = (
        "json_schema" if s.llm_client_mode == "generic_json_schema" else "json_object")
    return OpenAIGenericClient(config=cfg, client=raw, structured_output_mode=mode)

def _batch_capped_embeddings(client: AsyncOpenAI, max_batch: int) -> AsyncOpenAI:
    """Cap embedding requests to max_batch inputs per call.

    TEI/Jina enforces max_client_batch_size (32); graphiti can send larger
    batches (observed: 34), which TEI rejects with HTTP 422 and drops the whole
    article. Split oversized inputs into <=max_batch sub-batches, preserving order.
    """
    orig = client.embeddings.create

    async def create(*args, **kwargs):
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

    client.embeddings.create = create  # type: ignore[method-assign]
    return client


def build_graphiti(s: ExtractSettings, *, tier: str = "strong", transport: Any = None,
                   graph_driver: GraphDriver | None = None) -> Graphiti:
    """`transport` -> `_llm_client`; `graph_driver`, when given, is used instead of
    a driver built from `s.neo4j_uri`. Both default to the production behaviour."""
    # Candidate searches shipped fact_embedding (768 floats x 20 candidates,
    # twice per extracted fact) only for graphiti to pop it on arrival -- 2.8x on
    # the query that dominates ingestion. Idempotent; safe to call per build.
    install_lean_edge_search()
    # Skips the O(corpus) invalidation-candidate search (BACKLOG 6/8/30).
    # Idempotent; safe to call per build.
    install_contradiction_gate(
        detect_contradictions=s.ingest_detect_contradictions)
    # valid_at from the episode reference time, skipping a per-fact LLM call that
    # dated 23% of facts for 21.7% of run time. Idempotent.
    install_deterministic_valid_at(enabled=s.valid_at_from_content_changed)
    # Unbounded edge/node similarity searches -> tuned vector indexes (Phase B).
    # Idempotent; enabled=False restores graphiti's own functions exactly.
    install_vector_search(enabled=s.vector_search_enabled,
                          fetch_k=s.vector_search_fetch_k)
    embed_client = _batch_capped_embeddings(
        AsyncOpenAI(api_key="not-needed", base_url=s.embed_base_url,
                    timeout=s.embed_timeout_seconds, max_retries=4), s.embed_max_batch)
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key="not-needed", embedding_model=s.embed_model,
            embedding_dim=s.embed_dim, base_url=s.embed_base_url),
        client=embed_client)
    # CRITICAL: pass an explicit LOCAL reranker so Graphiti does not build its
    # default OpenAIRerankerClient() pointed at api.openai.com.
    reranker = OpenAIRerankerClient(config=_llm_config(s))
    llm = _llm_client(s, tier=tier, transport=transport)
    if graph_driver is not None:
        g = Graphiti(graph_driver=graph_driver, llm_client=llm, embedder=embedder,
                     cross_encoder=reranker, max_coroutines=s.max_coroutines)
    else:
        g = Graphiti(s.neo4j_uri, s.neo4j_user, s.neo4j_password,
                     llm_client=llm, embedder=embedder,
                     cross_encoder=reranker, max_coroutines=s.max_coroutines)
    # Graphiti.close() only closes the Neo4j driver; also close the AsyncOpenAI
    # clients we created (LLM + embedder) so they don't leak, esp. in a
    # long-running CLI that builds one Graphiti per invocation.
    owned = [embed_client]
    # The LLM and reranker clients both store their raw AsyncOpenAI as `.client`
    # (the reranker builds its own from config when no client= is passed).
    for extra in (getattr(llm, "client", None), getattr(reranker, "client", None)):
        if extra is not None:
            owned.append(extra)
    orig_close = g.close

    async def _close() -> None:
        await orig_close()
        for c in owned:
            try:
                await c.close()
            except Exception:  # best-effort cleanup
                pass

    g.close = _close  # type: ignore[method-assign]
    return g

def _cheap_view_settings(s: ExtractSettings) -> ExtractSettings:
    """The strong-tier settings with every LLM field swapped for its cheap_* twin."""
    return s.model_copy(update=dict(
        llm_base_url=s.cheap_llm_base_url, llm_model=s.cheap_llm_model,
        llm_api_key=s.cheap_llm_api_key, llm_client_mode=s.cheap_llm_client_mode,
        llm_cache_layout=s.cheap_llm_cache_layout,
        llm_timeout_seconds=s.cheap_llm_timeout_seconds))


def build_cheap_graphiti(s: ExtractSettings, *, transport: Any = None,
                         graph_driver: GraphDriver | None = None) -> Graphiti:
    """Build a Graphiti whose LLM points at the cheap model (ling). Reuses
    build_graphiti with a cheap-tier settings view; embedder/reranker/Neo4j and
    the group_id are unchanged, so both tiers share one embedding space.

    `transport` replaces only the raw AsyncOpenAI (every production wrapper is
    still applied to it) and `graph_driver` replaces the driver built from the
    URI -- the distillation harness's two substitutions. Both default to None,
    which is today's behaviour exactly."""
    cheap = _cheap_view_settings(s)
    # The reranker built by build_graphiti is harmless here: it is not invoked
    # during add_episode (the cheap client is used for ingestion only).
    injected: dict[str, Any] = {}
    if transport is not None:
        injected["transport"] = transport
    if graph_driver is not None:
        injected["graph_driver"] = graph_driver
    return build_graphiti(cheap, tier="cheap", **injected)

def build_embedder(s: ExtractSettings) -> OpenAIEmbedder:
    """The single shared embedder (TEI/Jina) used everywhere — extracted so the
    theme-builder can embed community title+summary in the same space (design
    decision #4)."""
    client = _batch_capped_embeddings(
        AsyncOpenAI(api_key="not-needed", base_url=s.embed_base_url,
                    timeout=s.embed_timeout_seconds, max_retries=4), s.embed_max_batch)
    return OpenAIEmbedder(config=OpenAIEmbedderConfig(
        api_key="not-needed", embedding_model=s.embed_model,
        embedding_dim=s.embed_dim, base_url=s.embed_base_url), client=client)

async def init_indices(graphiti: Graphiti, *, embed_dim: int | None = None,
                       vector_index_wait_seconds: float = DEFAULT_STARTUP_WAIT_SECONDS,
                       ) -> None:
    """graphiti's own indices, plus -- when `embed_dim` is given -- the tuned
    vector indexes, created if absent, waited on for up to
    `vector_index_wait_seconds` and VERIFIED (a mismatch or FAILED index raises;
    one still POPULATING is logged; nothing is ever rebuilt automatically)."""
    await graphiti.build_indices_and_constraints()
    if embed_dim is not None:
        await ensure_vector_indexes(graphiti.driver, embed_dim,
                                    wait_seconds=vector_index_wait_seconds)

# The DOCUMENT CONTEXT block `document_context` returns, below, is always a
# single LINE (no embedded "\n") after its leading marker: ONE sentence
# ("This chunk comes from ... documentation for ..."), whatever is filled in
# for vendor/product/source. `DOCUMENT_CONTEXT_PATTERN` matches exactly that
# line and is exported (compiled, not just the marker text) so
# `cache_layout.StaticTailLearner` can find and cut OUT the per-article
# sentence -- wherever graphiti placed it in the rendered user message --
# before learning or matching the surrounding STATIC text (graphiti follows
# `custom_extraction_instructions` with its own static <EXAMPLE>/EXTRACTION
# RULES blocks; it does not sit at the end of the user message). Both halves
# (the marker and the fixed sentence opener) are shared between the producer
# and the pattern below, not duplicated as separate literals, so they cannot
# drift apart. See cache_layout's module docstring for how the pattern is used.
DOCUMENT_CONTEXT_MARKER = "\n\nDOCUMENT CONTEXT:"
_DOCUMENT_CONTEXT_SENTENCE_OPENER = " This chunk comes from "
DOCUMENT_CONTEXT_PATTERN: re.Pattern[str] = re.compile(
    re.escape(DOCUMENT_CONTEXT_MARKER + _DOCUMENT_CONTEXT_SENTENCE_OPENER) + r"[^\n]*")


def document_context(vendor: str, product: str, source: str) -> str:
    """The per-chunk DOCUMENT CONTEXT block the extraction instructions refer
    to: the article's real vendor/product/document set, from the structural
    graph chain. Appended to `instructions` so the model is told the subject
    instead of being invited to invent one (extraction-grounding, 2026-10-17).
    Empty when there is no product (no structural chain for the article).
    Vendor and source are each optional (an empty string omits that part
    cleanly -- no "from  documentation" / "(document set: )"). Always a single
    line -- see `DOCUMENT_CONTEXT_PATTERN`, above."""
    if not product:
        return ""
    vendor_part = f"{vendor} " if vendor else ""
    source_clause = f" (document set: {source})" if source else ""
    return (
        f"{DOCUMENT_CONTEXT_MARKER}{_DOCUMENT_CONTEXT_SENTENCE_OPENER}{vendor_part}"
        f"documentation for {product}{source_clause}. The documented product is "
        f'"{product}".')


async def add_text_episode(graphiti: Graphiti, s: ExtractSettings, *, name: str,
                           body: str, source_description: str,
                           reference_time: datetime,
                           instructions: str = EXTRACTION_INSTRUCTIONS,
                           context: str = "",
                           previous_episode_uuids: list[str] | None = None,
                           ) -> AddEpisodeResults:
    """`previous_episode_uuids` is passed straight to graphiti: None (the default)
    lets graphiti look up the group's last episodes itself, as before."""
    return await graphiti.add_episode(
        name=name, episode_body=body, source_description=source_description,
        reference_time=reference_time, source=EpisodeType.text, group_id=s.group_id,
        entity_types=ENTITY_TYPES, excluded_entity_types=EXCLUDED_ENTITY_TYPES,
        edge_types=EDGE_TYPES, edge_type_map=EDGE_TYPE_MAP,
        custom_extraction_instructions=instructions + context,
        previous_episode_uuids=previous_episode_uuids)
