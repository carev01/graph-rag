from __future__ import annotations
from datetime import datetime
from openai import AsyncOpenAI
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
    return LLMConfig(api_key="not-needed", model=s.llm_model,
                     small_model=s.llm_model, base_url=s.llm_base_url, temperature=0.0)

def _llm_client(s: ExtractSettings):
    raw = instrument(AsyncOpenAI(api_key="not-needed", base_url=s.llm_base_url))
    cfg = _llm_config(s)
    if s.llm_client_mode == "structured":
        return OpenAIClient(config=cfg, client=raw, reasoning="auto", verbosity="low")
    mode = "json_schema" if s.llm_client_mode == "generic_json_schema" else "json_object"
    return OpenAIGenericClient(config=cfg, client=raw, structured_output_mode=mode)

def build_graphiti(s: ExtractSettings) -> Graphiti:
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
        api_key="not-needed", embedding_model=s.embed_model,
        embedding_dim=s.embed_dim, base_url=s.embed_base_url))
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
