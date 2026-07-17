"""Slice-2a pilot runner: ingest a curated article-id list end-to-end.

Reads scripts/pilot-ids.txt (tab-separated: article_id, vendor,
tokens, title), ingests each article through the full extraction pipeline
IN-PROCESS (so the token-usage tally is captured), prints per-article
progress, and emits the cost report at the end.

Resumable: already-extracted chunks are hash-gated and skipped, so a
re-run continues where an interrupted run left off.

Run (long — hours on a local GPU):
    uv run --extra dev python scripts/run_pilot.py
"""
from __future__ import annotations
import asyncio
import os
import time
from pathlib import Path

from neo4j import AsyncGraphDatabase

from graph_extract.config import get_extract_settings
from graph_extract.eval import cost_report
from graph_extract.graphiti_client import build_graphiti, init_indices, ExtractionTier
from graph_extract.ingest_driver import IngestDriver
from graph_extract.ontology import EXTRACTION_INSTRUCTIONS
from graph_extract.provenance import Provenance
from graph_sync.delta_client import make_client

# Default to the full pilot list; PILOT_IDS_FILE overrides it (used by the
# slice-2b quality-iteration loop to ingest the small AWS/Azure overlap sample).
IDS_FILE = Path(os.environ.get("PILOT_IDS_FILE", "scripts/pilot-ids.txt"))


def _load_ids() -> list[tuple[str, str, str]]:
    rows = []
    for line in IDS_FILE.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        rows.append((parts[0], parts[1], parts[3] if len(parts) > 3 else ""))
    return rows


async def main() -> None:
    s = get_extract_settings()
    rows = _load_ids()
    print(f"[pilot] {len(rows)} articles to ingest (resumable/hash-gated)", flush=True)

    docext = make_client(s)
    driver = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    graphiti = build_graphiti(s)
    await init_indices(graphiti)
    prov = Provenance(driver)
    strong_tier = ExtractionTier("strong", graphiti, EXTRACTION_INSTRUCTIONS, s.max_chunk_tokens)
    ingest = IngestDriver(s, strong_tier, None, docext, prov, driver)

    total_added = total_skipped = 0
    t0 = time.time()
    try:
        for n, (aid, vendor, title) in enumerate(rows, 1):
            at = time.time()
            try:
                r = await ingest.ingest_article(aid)
                total_added += r.episodes_added
                total_skipped += r.episodes_skipped
                print(f"[pilot] {n}/{len(rows)} {vendor:5s} +{r.episodes_added} "
                      f"~{r.episodes_skipped} ({round(time.time()-at)}s) {title}", flush=True)
            except Exception as ex:  # keep going; the run is resumable
                print(f"[pilot] {n}/{len(rows)} {vendor:5s} ERROR {aid}: {ex}", flush=True)
        print(f"[pilot] done: episodes_added={total_added} skipped={total_skipped} "
              f"wall={round(time.time()-t0)}s", flush=True)
        print("[pilot] --- cost (extraction-LLM tokens this run) ---", flush=True)
        import json
        print(json.dumps(await cost_report(total_added), indent=2, default=str), flush=True)
    finally:
        await graphiti.close()
        await docext.aclose()
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
