import json

import httpx
import pytest
from pathlib import Path
from graph_sync.catalog import Catalog
from graph_sync.models import ContentRecord
from graph_sync.sync_core import BootstrapResult, SyncCore
from graph_sync.config import Settings

pytestmark = pytest.mark.asyncio(loop_scope="module")
FIX = Path(__file__).parent.parent / "fixtures"

def _settings():
    return Settings(docext_base_url="https://x", docext_read_key="k",
                    neo4j_uri="x", neo4j_user="x", neo4j_password="x",
                    postgres_dsn="x", webhook_debounce_seconds=0)

def _client():
    delta = FIX.joinpath("aws_delta.ndjson").read_bytes()
    toc = FIX.joinpath("aws_toc.json").read_bytes()
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/articles/delta":
            return httpx.Response(200, content=delta)
        if req.url.path.startswith("/api/articles/toc/"):
            return httpx.Response(200, content=toc)
        raise AssertionError(req.url.path)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://x")

def _catalog():
    def load(n):
        return json.loads(FIX.joinpath(n).read_text())
    return Catalog.from_lists(load("vendors.json")["vendors"],
                              load("products.json")["products"],
                              load("sources.json")["sources"])

async def test_bootstrap_ingests_and_gates(neo4j_repo, state_store):
    src = "21632f3b-5a4c-4c93-9f00-6701d0e9f677"
    # Reset this source so the test is order-independent (the module-scoped
    # neo4j_repo fixture is shared with the resume test, which also uses it).
    await neo4j_repo.delete_source_articles(src)
    core = SyncCore(_client(), _catalog(), neo4j_repo, state_store, _settings())
    r1 = await core.bootstrap(source_id=src)
    assert r1.applied == 146 and r1.skipped == 0
    assert await neo4j_repo.article_count_by_source(src) == 146
    # Second bootstrap: content_hash gate skips everything
    r2 = await core.bootstrap(source_id=src)
    assert r2.applied == 0 and r2.skipped == 146
    # TOC pass ran -> a known article is linked to a chapter
    assert await neo4j_repo.article_chapter_id(
        "2c92266f-f84c-49c2-9e89-3b4376ec9043") is not None

async def test_bootstrap_requeues_articles_stranded_by_a_state_reset(neo4j_repo, state_store):
    """BACKLOG 50, end to end: after the job table loses its rows, a plain re-run
    of the bootstrap re-queues exactly the articles that were never extracted --
    not the ones with episodes, not the ones with a done job -- and a further
    re-run is a no-op."""
    src = "21632f3b-5a4c-4c93-9f00-6701d0e9f677"
    await neo4j_repo.delete_source_articles(src)
    core = SyncCore(_client(), _catalog(), neo4j_repo, state_store, _settings())
    assert (await core.bootstrap(source_id=src)).applied == 146

    async with neo4j_repo._driver.session() as s:
        ids = [r["id"] async for r in await s.run(
            "MATCH (a:Article {source_id:$s}) RETURN a.id AS id ORDER BY a.id", s=src)]
    assert len(ids) == 146
    extracted, completed = ids[:3], ids[3:5]
    pool = await state_store._get_pool()
    # The reset: every job row for this source is gone ...
    await pool.execute("DELETE FROM semantic_jobs WHERE article_id = ANY($1::text[])", ids)
    # ... except two that completed with no episodes (navigation-style) ...
    for a in completed:
        await pool.execute("INSERT INTO semantic_jobs (article_id, op, status, lane) "
                           "VALUES ($1, 'upsert', 'done', 'bootstrap')", a)
    # ... and three articles the graph shows as extracted.
    async with neo4j_repo._driver.session() as s:
        await s.run("UNWIND $ids AS i MATCH (a:Article {id:i}) "
                    "CREATE (a)-[:HAS_EPISODE]->(:Episodic {uuid: 'ep-' + i})", ids=extracted)

    try:
        r2 = await core.bootstrap(source_id=src)
        assert r2.applied == 0 and r2.skipped == 5 and r2.requeued == 141
        rows = await pool.fetch(
            "SELECT article_id, lane FROM semantic_jobs WHERE status='pending' "
            "AND article_id = ANY($1::text[])", ids)
        assert {r["article_id"] for r in rows} == set(ids[5:])
        assert {r["lane"] for r in rows} == {"bootstrap"}

        r3 = await core.bootstrap(source_id=src)
        assert r3.requeued == 0 and r3.skipped == 146
    finally:
        async with neo4j_repo._driver.session() as s:
            await s.run("MATCH (e:Episodic) WHERE e.uuid STARTS WITH 'ep-' DETACH DELETE e")

def _resuming_client():
    """First delta call truncates (no terminal); the bootstrap_after retry completes."""
    lines = FIX.joinpath("aws_delta.ndjson").read_bytes().split(b"\n")
    boot = lines[0]
    data = [ln for ln in lines[1:] if ln.strip() and b'"control"' not in ln]
    terminal = lines[-1] if b'"control":"cursor"' in lines[-1] else lines[-2]
    first = b"\n".join([boot] + data[:80])            # truncated: no terminal
    second = b"\n".join([boot] + data[80:] + [terminal])
    calls = {"n": 0}
    toc = FIX.joinpath("aws_toc.json").read_bytes()
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.startswith("/api/articles/toc/"):
            return httpx.Response(200, content=toc)
        if req.url.path == "/api/articles/delta":
            if "bootstrap_after" in req.url.params:
                return httpx.Response(200, content=second)
            calls["n"] += 1
            return httpx.Response(200, content=first)
        raise AssertionError(req.url.path)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://x"), calls

async def test_bootstrap_resumes_after_truncation(neo4j_repo, state_store):
    src = "21632f3b-5a4c-4c93-9f00-6701d0e9f677"
    # neo4j_repo/state_store are module-scoped and shared with
    # test_bootstrap_ingests_and_gates above, which already wrote this same
    # source's 146 articles with these exact content_hash values. Reset so this
    # test measures genuine applies across the resume boundary instead of being
    # hash-gated by the earlier test's leftover state.
    await neo4j_repo.delete_source_articles(src)
    client, calls = _resuming_client()
    core = SyncCore(client, _catalog(), neo4j_repo, state_store, _settings())
    res = await core.bootstrap(source_id=src)
    # No missed or duplicated articles across the resume boundary
    assert res.applied == 146
    assert await neo4j_repo.article_count_by_source(src) == 146
    # The original bootstrap watermark (not a resume-recomputed one) became the cursor
    assert await state_store.get_cursor() is not None


def _empty_catalog_client():
    """A client whose catalog endpoints are empty, so refresh() can't resolve
    the source -- exercises the unknown-source path."""
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/vendors":
            return httpx.Response(200, content=b'{"vendors":[]}')
        if req.url.path == "/api/products":
            return httpx.Response(200, content=b'{"products":[]}')
        if req.url.path == "/api/sources":
            return httpx.Response(200, content=b'{"sources":[]}')
        raise AssertionError(req.url.path)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://x")


async def test_unknown_source_record_is_persisted_not_dropped(neo4j_repo, state_store):
    # Spec §7: a record whose source can't be resolved (even after refresh) must
    # NOT be dropped/dead-lettered -- it is persisted as a minimal, catalog_incomplete
    # Article so it survives and is hash-gated on retry.
    rec = ContentRecord.model_validate({
        "change_type": "added", "id": "orphan-1", "topic_key": "tk",
        "source_id": "no-such-source", "vendor": "Ghost", "product": "GhostProd",
        "title": "Orphan", "source_url": "https://x/orphan", "content_hash": "oh1",
        "estimated_tokens": 42, "sort_order": 0, "run_id": "r1", "seq": None,
    })
    client = _empty_catalog_client()
    core = SyncCore(client, Catalog(client), neo4j_repo, state_store, _settings())
    dl_before = await state_store.dead_letter_count()
    res = BootstrapResult()
    touched = await core._apply_record(rec, res)
    assert touched == "no-such-source" and res.applied == 1
    # Persisted, not dropped: node exists with the right hash and the flag set.
    assert await neo4j_repo.get_content_hash("orphan-1") == "oh1"
    assert await neo4j_repo.is_catalog_incomplete("orphan-1") is True
    # Not dead-lettered.
    assert await state_store.dead_letter_count() == dl_before
