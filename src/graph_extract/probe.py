"""Step-0 client-mode probe.

Runs a handful of real article chunks through Graphiti in one or more LLM
client modes and reports, per mode: extraction errors, token usage, wall-clock,
and the extracted entities/facts (so a human can compare quality and confirm
gpt-oss's reasoning survived). Writes nothing itself — the caller inspects the
returned dict / writes the markdown report.

Note: the `structured` (OpenAIClient / Responses API) mode is NOT probed by
default: Task 6 established it is unusable with gpt-oss-20b (markdown-fenced /
malformed JSON, 0 entities). Default probe compares the two generic modes.
"""
from __future__ import annotations
import time
from dataclasses import replace
from datetime import datetime, timezone

import httpx

from graph_extract import chonkie_client, content_fetch, episode_builder
from graph_extract.config import ExtractSettings
from graph_extract.graphiti_client import add_text_episode, build_graphiti, init_indices
from graph_extract.usage import get_tally, reset_tally
from graph_sync.delta_client import make_client as make_docext_client

DEFAULT_MODES = ("generic_json_schema", "generic_json_object")


async def _chunks_for(s: ExtractSettings, docext: httpx.AsyncClient,
                      article_id: str, n: int):
    art = await content_fetch.fetch_article(docext, article_id)
    async with httpx.AsyncClient(base_url=s.chonkie_base_url, timeout=120) as ch:
        chunks = await chonkie_client.neural_chunk(ch, art.content_markdown, s.chonkie_model)
    eps = episode_builder.build_episodes(
        article_id=art.id, title=art.title, chapter_path="",
        content_hash="probe0000", chunks=chunks,
        max_chunk_tokens=s.max_chunk_tokens, min_chunk_tokens=s.min_chunk_tokens)
    return art, eps[:n]


async def run_probe(s: ExtractSettings, article_ids: list[str], n_chunks: int,
                    modes: tuple[str, ...] = DEFAULT_MODES) -> dict:
    docext = make_docext_client(s)  # type: ignore[arg-type]  # structurally compatible
    report: dict = {}
    try:
        for mode in modes:
            sm = replace(s, llm_client_mode=mode)  # type: ignore[arg-type]
            g = build_graphiti(sm)
            await init_indices(g)
            reset_tally()
            t0 = time.time()
            errors = 0
            dumps: list[dict] = []
            for aid in article_ids:
                art, eps = await _chunks_for(sm, docext, aid, n_chunks)
                for e in eps:
                    try:
                        res = await add_text_episode(
                            g, sm, name=f"probe-{mode}-{e.name}", body=e.body,
                            source_description=art.source_url,
                            reference_time=datetime.now(timezone.utc))
                        dumps.append({"chunk": e.body[:200],
                                      "entities": [n.name for n in res.nodes],
                                      "facts": [ed.fact for ed in res.edges]})
                    except Exception as ex:  # parse/schema failure
                        errors += 1
                        dumps.append({"chunk": e.body[:200], "error": str(ex)})
            report[mode] = {"wall_s": round(time.time() - t0, 1), "errors": errors,
                            "usage": vars(get_tally()), "dumps": dumps}
            await g.close()
    finally:
        await docext.aclose()
    return report
