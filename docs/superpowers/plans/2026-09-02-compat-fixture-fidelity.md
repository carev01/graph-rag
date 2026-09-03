# Compat-Harness Fixture Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Neo4j compatibility harness a realistic synthetic fixture so its checks verify results — above all the provenance join — instead of passing over empty result sets.

**Architecture:** Build the fixture through the REAL ingestion write path (`Neo4jRepo.apply_structural` + `Provenance.link` + `write_communities_incremental`) rather than hand-written Cypher. Because `_APPLY_STRUCTURAL` is `MERGE ... SET v += $vendor`, injecting `group_id` keeps every created node inside teardown's existing scoped sweep, so one move fixes the fixture and covers the previously untested ingestion writes.

**Tech Stack:** Python 3.12, `uv`, `neo4j` async driver, `graphiti-core==0.29.2`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-compat-fixture-fidelity-design.md`

## Global Constraints

- **SAFETY, non-negotiable:** every structural id the harness writes is a literal constant prefixed `compat-check-`. `MERGE (v:Vendor {id: $vendor.id})` against a REAL vendor id would stamp `group_id="compat-check"` onto a production node, which teardown would then delete. No id may be derived from configuration or from anything read off the target.
- Every node the harness creates carries `group_id = "compat-check"` (the existing `COMPAT_GROUP_ID` constant), so the existing teardown removes it.
- Groups 1–7 stay **LLM-free and embedder-free**; only the `e2e` group may reach a model endpoint.
- The e2e skip contract is unchanged: only `httpx.ConnectError`, `httpx.ConnectTimeout`, `openai.APIConnectionError` yield `skip`. Everything else fails.
- `source_url` values use the reserved non-resolving TLD `.invalid` so no fabricated URL in a report is clickable.
- Registry order is execution order and is load-bearing.
- Target is Neo4j 2026.07.1 Community, `CYPHER_25` default, GDS 2026.07.0, **APOC not installed** — no check may call `apoc.*`.
- CI gate, clean at every commit: `uv run ruff check src tests` (lints tests too: E702 no semicolons, E402 imports at top), `uv run mypy src`, `uv run --extra dev pytest -m "not live"`.

## Verified signatures (do not guess these)

```python
# src/graph_sync/models.py
StructuralWrite(vendor: dict, product: dict, source: dict, article: dict)
# src/graph_sync/neo4j_repo.py — MERGE (v:Vendor {id: $vendor.id}) SET v += $vendor, etc.
Neo4jRepo(uri, user, password); await repo.init_schema(); await repo.apply_structural(w); await repo.close()

# src/graph_extract/provenance.py
await Provenance(driver).link(article_id, episode_uuid, *, chunk_index: int,
                              heading_path: str, token_count: int, content_hash: str)
await Provenance(driver).resolve_citations(fact_uuids: list[str]) -> dict[str, dict]
#   -> {uuid: {valid_at, invalid_at, sources: [{url, title, article_id, section, vendor, product}]}}
#   An absent uuid stays ABSENT from the dict; callers use .get(uuid, {}).

# src/theme_builder/writeback.py
await write_communities_incremental(driver, group_id, entries: list[dict], *, corpus_cursor: str | None)
#   entry keys: community_id, level, title, summary, full_report, rating,
#               rating_explanation, tags, cited_fact_uuids, embedding,
#               member_uuids, generated_at, and optional parent_id

# src/answer_api/global_search.py — NOTE: there is NO min_rating parameter
await shortlist_communities(driver, embedder, q, *, level: int, k: int,
                            group_id: str, rating_boost: float = 0.1) -> list[CommunityHit]
#   embedder must expose: await embedder.create_batch([q]) -> list[list[float]]
#   _rank_hits SKIPS any row whose cited_fact_uuids is empty, and the Cypher
#   drops rows with a falsy embedding.

# src/graph_extract/staleness_sweep.py
await sweep_stale_facts(driver, group_id) -> {"expired": int, "sample": list[str]}

# graphiti_core.graphiti.AddEpisodeResults has .episode (an EpisodicNode, so .uuid)
```

---

### Task 1: Structural fixture through the real write path

**Files:**
- Modify: `src/compat/checks.py`
- Test: `tests/unit/test_compat_checks.py`

**Interfaces:**
- Consumes: existing `COMPAT_GROUP_ID`, `CypherCheck`, `CallableCheck`, `CheckContext`, `SkipCheck`, `Check` from `src/compat/model.py`; `compat_target` from `src/compat/runner.py`; the existing `EP_UUID`, `ENT_A/B/C`, `FACT_AB`, `FACT_BC` constants.
- Produces: new module constants `VENDOR_ID`, `PRODUCT_ID`, `SOURCE_ID`, `ARTICLE_ID`, `ARTICLE_URL`, `HEADING_PATH`, `EP_ORPHAN`, `FACT_ORPHAN`, `E2E_ARTICLE_ID`, `E2E_ARTICLE_URL`; `_structural_fixture` and `_link_episode` check functions. Task 2 asserts on the values these write; Task 3 uses `E2E_ARTICLE_ID`/`E2E_ARTICLE_URL`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_compat_checks.py`:

```python
def test_every_structural_id_is_compat_prefixed():
    """SAFETY: MERGE against a real vendor/article id would stamp group_id onto a
    production node, which teardown would then DELETE. Ids must be literal and
    unmistakably ours."""
    for value in [checks.VENDOR_ID, checks.PRODUCT_ID, checks.SOURCE_ID,
                  checks.ARTICLE_ID, checks.E2E_ARTICLE_ID]:
        assert value.startswith("compat-check-"), value


def test_orphan_ids_are_compat_prefixed():
    for value in [checks.EP_ORPHAN, checks.FACT_ORPHAN]:
        assert value.startswith("compat-")


def test_article_urls_use_the_reserved_invalid_tld():
    """A fabricated URL must never look like real documentation in a report."""
    for url in [checks.ARTICLE_URL, checks.E2E_ARTICLE_URL]:
        assert ".invalid/" in url


def test_bootstrap_group_writes_the_structural_chain():
    names = " ".join(c.name.lower() for c in checks.bootstrap_checks())
    assert "structural fixture" in names


def test_graphiti_write_group_links_the_article_to_the_episode():
    names = [c.name.lower() for c in checks.graphiti_write_checks()]
    joined = " ".join(names)
    assert "provenance link" in joined
    # The link must come AFTER the episode write: registry order is execution order,
    # and Provenance.link MATCHes an existing (:Episodic).
    write_idx = next(i for i, n in enumerate(names) if "synthetic graph" in n)
    link_idx = next(i for i, n in enumerate(names) if "provenance link" in n)
    assert write_idx < link_idx


def test_structural_write_precedes_the_episode_write_across_groups():
    groups = [c.group for c in checks.all_checks([0.5] * 768)]
    assert groups.index("bootstrap") < groups.index("graphiti-write")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_compat_checks.py -q`
Expected: FAIL — `AttributeError: module 'compat.checks' has no attribute 'VENDOR_ID'`

- [ ] **Step 3: Add the constants**

In `src/compat/checks.py`, next to the existing `EP_UUID`/`ENT_A` constants, add:

```python
# --- structural fixture ids -------------------------------------------------
# SAFETY: these are LITERAL and `compat-check-`-prefixed on purpose. apply_structural
# runs `MERGE (v:Vendor {id: $vendor.id}) SET v += $vendor`, so a collision with a real
# vendor/article id would stamp group_id="compat-check" onto a PRODUCTION node -- which
# teardown would then delete. Never derive these from settings or from the target.
VENDOR_ID = "compat-check-vendor"
PRODUCT_ID = "compat-check-product"
SOURCE_ID = "compat-check-source"
ARTICLE_ID = "compat-check-article"
E2E_ARTICLE_ID = "compat-check-article-e2e"
# .invalid is a reserved non-resolving TLD (RFC 2606): a fabricated URL appearing in a
# report must never be clickable or mistakable for real vendor documentation.
ARTICLE_URL = "https://example.invalid/compat-check/vault-lock"
E2E_ARTICLE_URL = "https://example.invalid/compat-check/e2e"
HEADING_PATH = "Retention"
VENDOR_NAME = "AWS"          # _vendor_episode_uuids(driver, "AWS") must find this
PRODUCT_NAME = "AWS Backup"

# An episode deliberately NEVER linked to an :Article, plus a fact supported only by
# it. The staleness sweep must expire exactly this fact and leave the supported ones
# alone -- covering both directions of a query that silently invalidates data.
EP_ORPHAN = "compat-ep-orphan"
FACT_ORPHAN = "compat-fact-orphan"
```

- [ ] **Step 4: Add the structural-write check to the bootstrap group**

In `src/compat/checks.py`, add this function next to `_structural_schema`:

```python
async def _structural_fixture(ctx: CheckContext) -> str:
    """Write the Vendor->Product->Source->Article chain through the REAL ingestion
    path (graph_sync.neo4j_repo.apply_structural), not hand-written Cypher. This
    covers a write query that was previously untested, and because _APPLY_STRUCTURAL
    is `MERGE ... SET v += $vendor`, passing group_id puts every created node inside
    teardown's scoped sweep."""
    from compat.runner import compat_target
    from graph_sync.models import StructuralWrite
    from graph_sync.neo4j_repo import Neo4jRepo

    uri, user, password = compat_target(ctx.settings)
    repo = Neo4jRepo(uri, user, password)
    try:
        await repo.apply_structural(StructuralWrite(
            vendor={"id": VENDOR_ID, "name": VENDOR_NAME, "group_id": COMPAT_GROUP_ID},
            product={"id": PRODUCT_ID, "name": PRODUCT_NAME,
                     "group_id": COMPAT_GROUP_ID},
            source={"id": SOURCE_ID, "name": "compat source",
                    "group_id": COMPAT_GROUP_ID},
            article={"id": ARTICLE_ID, "title": "Vault Lock",
                     "source_url": ARTICLE_URL, "source_id": SOURCE_ID,
                     "group_id": COMPAT_GROUP_ID}))
    finally:
        await repo.close()
    return f"structural chain written: {VENDOR_ID} -> {ARTICLE_ID}"
```

Then add it to `bootstrap_checks()`'s returned list, **after** the existing
`graph_sync structural schema` check (the constraints must exist before the writes):

```python
        CallableCheck("structural fixture chain", "bootstrap", _structural_fixture),
```

- [ ] **Step 5: Add the orphan episode and orphan fact to the synthetic write**

In `_write_synthetic_graph` in `src/compat/checks.py`, after the existing `episode`
save, add the orphan episode:

```python
    orphan_episode = EpisodicNode(
        uuid=EP_ORPHAN, name="compat orphan episode", group_id=COMPAT_GROUP_ID,
        created_at=now, source=EpisodeType.text,
        source_description="compatibility harness (deliberately unlinked)",
        content="An episode intentionally never linked to an :Article.",
        valid_at=now)
    await orphan_episode.save(gdriver)
```

and add a third edge to the `edges` list:

```python
        EntityEdge(uuid=FACT_ORPHAN, group_id=COMPAT_GROUP_ID, source_node_uuid=ENT_A,
                   target_node_uuid=ENT_C, created_at=now, name="UNSUPPORTED",
                   fact="A fact whose only supporting episode has no article.",
                   fact_embedding=ctx.embedding, episodes=[EP_ORPHAN],
                   valid_at=now, invalid_at=None),
```

Update the function's return string to:

```python
    return "2 episodes (1 orphan), 3 entities, 3 bi-temporal facts written"
```

- [ ] **Step 6: Add the Provenance.link check**

Add this function to `src/compat/checks.py`:

```python
async def _link_episode(ctx: CheckContext) -> str:
    """Attach the Article to the episode via the REAL provenance writer. In production
    this edge is graph-sync's job, never graphiti's -- which is why the fixture had no
    HAS_EPISODE chain and resolve_citations returned zero sources."""
    from graph_extract.provenance import Provenance

    await Provenance(ctx.driver).link(
        ARTICLE_ID, EP_UUID, chunk_index=0, heading_path=HEADING_PATH,
        token_count=42, content_hash="compat-check-hash")
    async with ctx.driver.session() as s:
        result = await s.run(
            "MATCH (:Article {id:$a})-[r:HAS_EPISODE]->(e:Episodic {uuid:$u}) "
            "RETURN r.heading_path AS section", a=ARTICLE_ID, u=EP_UUID)
        rows = [dict(rec) async for rec in result]
    if not rows:
        raise RuntimeError("Provenance.link did not create the HAS_EPISODE edge")
    return f"article linked to episode, section={rows[0]['section']!r}"
```

Add it to `graphiti_write_checks()` as the **last** entry, after the bi-temporal check.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_compat_checks.py -q`
Expected: PASS

- [ ] **Step 8: Lint and type-check**

Run: `uv run ruff check src tests && uv run mypy src`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add src/compat/checks.py tests/unit/test_compat_checks.py
git commit -m "feat(compat): structural fixture via the real ingestion write path"
```

---

### Task 2: Community layer and strengthened assertions

**Files:**
- Modify: `src/compat/checks.py`
- Test: `tests/unit/test_compat_checks.py`

**Interfaces:**
- Consumes: everything Task 1 produced (`VENDOR_ID`, `PRODUCT_ID`, `SOURCE_ID`, `ARTICLE_ID`, `ARTICLE_URL`, `HEADING_PATH`, `VENDOR_NAME`, `PRODUCT_NAME`, `EP_ORPHAN`, `FACT_ORPHAN`).
- Produces: `COMMUNITY_ID`, `COMMUNITY_LEVEL`; `_FakeEmbedder`; `_write_community`, `_shortlist_communities` check functions; strengthened versions of `_resolve_citations`, `_vendor_scope`, `_load_persisted`, `_staleness_sweep`, `_timeline_sweep_flags`.

- [ ] **Step 1: Write the failing tests**

Add `import asyncio` to the **top** of `tests/unit/test_compat_checks.py` alongside its
existing imports (CI lints tests, and E402 forbids mid-file imports), then append:

```python
def test_our_cypher_group_writes_then_reads_the_community():
    names = [c.name.lower() for c in checks.our_cypher_checks()]
    write_idx = next(i for i, n in enumerate(names) if "write community" in n)
    read_idx = next(i for i, n in enumerate(names) if "load_persisted" in n)
    shortlist_idx = next(i for i, n in enumerate(names) if "shortlist" in n)
    assert write_idx < read_idx
    assert write_idx < shortlist_idx


def test_sweep_runs_before_the_timeline_flag_check():
    """_timeline_sweep_flags asserts the orphan carries expired_by_sweep, so the
    sweep must have already run."""
    names = [c.name.lower() for c in checks.our_cypher_checks()]
    sweep_idx = next(i for i, n in enumerate(names) if "staleness sweep" in n)
    flags_idx = next(i for i, n in enumerate(names) if "timeline sweep flags" in n)
    assert sweep_idx < flags_idx


def test_fake_embedder_returns_the_vector_shortlist_expects():
    # `import asyncio` goes at the TOP of the test file, not here: CI runs
    # `ruff check src tests` and E402 applies to test files too.
    emb = checks._FakeEmbedder([0.25] * 768)
    got = asyncio.run(emb.create_batch(["anything"]))
    assert got == [[0.25] * 768]


def test_community_entry_satisfies_every_shortlist_filter():
    """shortlist_communities drops rows with a falsy embedding, and _rank_hits SKIPS
    rows whose cited_fact_uuids is empty. A community failing either would make the
    shortlist check pass vacuously."""
    entry = checks._community_entry([0.5] * 768)
    assert entry["level"] == checks.COMMUNITY_LEVEL
    assert entry["cited_fact_uuids"]
    assert entry["embedding"]
    assert entry["member_uuids"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_compat_checks.py -q`
Expected: FAIL — `AttributeError: module 'compat.checks' has no attribute '_FakeEmbedder'`

- [ ] **Step 3: Add the community constants, the entry builder, and the fake embedder**

Add to `src/compat/checks.py`, near the other constants:

```python
COMMUNITY_ID = "compat-check-community"
COMMUNITY_LEVEL = 0          # written AND queried at this level; they must match


class _FakeEmbedder:
    """The one method shortlist_communities calls. Keeps group 7 free of network I/O
    while still exercising the real shortlist query and ranking."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    async def create_batch(self, inputs: list[str]) -> list[list[float]]:
        return [self._vector for _ in inputs]


def _community_entry(embedding: list[float]) -> dict:
    """A complete writeback entry. cited_fact_uuids MUST be non-empty: _rank_hits
    skips rows without citable facts, so an empty list would silently produce an
    empty shortlist and a vacuous pass."""
    return {
        "community_id": COMMUNITY_ID, "level": COMMUNITY_LEVEL,
        "title": "Compat immutability theme",
        "summary": "Vault Lock enforces immutable retention.",
        "full_report": "[]", "rating": 7.5,
        "rating_explanation": "fixture", "tags": ["compat"],
        "cited_fact_uuids": [FACT_AB, FACT_BC],
        "embedding": embedding,
        "member_uuids": [ENT_A, ENT_B, ENT_C],
        "generated_at": "2026-09-02T00:00:00Z",
    }
```

- [ ] **Step 4: Add the community write and shortlist checks**

Add to `src/compat/checks.py`:

```python
async def _write_community(ctx: CheckContext) -> str:
    """Write one :Community through the REAL writeback. write_communities_incremental
    needs no embedder (entries carry their own vector), so this stays LLM-free. It
    dissolves communities absent from `entries`, but is group_id-scoped and so can
    only ever affect the compat-check namespace."""
    from theme_builder.writeback import write_communities_incremental

    outcome = await write_communities_incremental(
        ctx.driver, COMPAT_GROUP_ID, [_community_entry(ctx.embedding)],
        corpus_cursor="2026-09-02T00:00:00Z")
    if outcome.get("reports_written") != 1:
        raise RuntimeError(f"expected 1 community written, got {outcome}")
    return f"community layer written: {outcome}"


async def _shortlist_communities(ctx: CheckContext) -> str:
    from answer_api.global_search import shortlist_communities

    hits = await shortlist_communities(
        ctx.driver, _FakeEmbedder(ctx.embedding), "immutable retention",
        level=COMMUNITY_LEVEL, k=5, group_id=COMPAT_GROUP_ID)
    if not hits:
        raise RuntimeError("shortlist_communities returned no hits despite a written "
                           "community at the queried level")
    return f"shortlist returned {len(hits)} hit(s), top={hits[0].community_id}"
```

- [ ] **Step 5: Strengthen the reading assertions**

Replace the bodies of these five functions in `src/compat/checks.py`:

```python
async def _resolve_citations(ctx: CheckContext) -> str:
    """The provenance join -- design decision #2, the system's core citation
    guarantee. Asserts the resolved VALUES, not just that the query ran: with no
    structural chain this previously returned entries with empty `sources`."""
    from graph_extract.provenance import Provenance

    resolved = await Provenance(ctx.driver).resolve_citations([FACT_AB, FACT_BC])
    entry = resolved.get(FACT_AB)
    if entry is None:
        raise RuntimeError(f"{FACT_AB} absent from resolve_citations: {resolved}")
    sources = entry.get("sources") or []
    if not sources:
        raise RuntimeError("resolve_citations returned no sources -- the provenance "
                           "join (fact -> episode -> article -> url) did not resolve")
    src = sources[0]
    expected = {"url": ARTICLE_URL, "article_id": ARTICLE_ID, "vendor": VENDOR_NAME,
                "product": PRODUCT_NAME, "section": HEADING_PATH}
    wrong = {k: (src.get(k), v) for k, v in expected.items() if src.get(k) != v}
    if wrong:
        raise RuntimeError(f"resolved source has wrong values (got, expected): {wrong}")
    return f"provenance join resolved: {src['vendor']}/{src['product']} {src['url']}"


async def _vendor_scope(ctx: CheckContext) -> str:
    from answer_api.search import _vendor_episode_uuids

    uuids = await _vendor_episode_uuids(ctx.driver, VENDOR_NAME)
    if EP_UUID not in uuids:
        raise RuntimeError(f"vendor scope for {VENDOR_NAME!r} did not find {EP_UUID}; "
                           f"got {sorted(uuids)}")
    return f"vendor scope resolved {len(uuids)} episode uuid(s) for {VENDOR_NAME}"


async def _staleness_sweep(ctx: CheckContext) -> str:
    """Both directions of a query that silently invalidates data: it must expire the
    orphan fact (whose only episode has no article) and leave the supported facts
    alone."""
    from graph_extract.staleness_sweep import sweep_stale_facts

    outcome = await sweep_stale_facts(ctx.driver, COMPAT_GROUP_ID)
    expired = set(outcome.get("sample") or [])
    if expired != {FACT_ORPHAN}:
        raise RuntimeError(f"sweep expired {sorted(expired)}, expected exactly "
                           f"{{{FACT_ORPHAN}}} (count={outcome.get('expired')})")
    return f"sweep expired exactly the unsupported fact: {outcome}"


async def _timeline_sweep_flags(ctx: CheckContext) -> str:
    """Runs AFTER the sweep, so the orphan must be flagged and the supported fact
    must not."""
    from answer_api.timeline import _sweep_flags

    flags = await _sweep_flags(ctx.driver, [FACT_AB, FACT_ORPHAN], COMPAT_GROUP_ID)
    if not flags.get(FACT_ORPHAN):
        raise RuntimeError(f"{FACT_ORPHAN} not flagged expired_by_sweep: {flags}")
    if flags.get(FACT_AB):
        raise RuntimeError(f"{FACT_AB} wrongly flagged expired_by_sweep: {flags}")
    return f"sweep flags correct: {flags}"


async def _load_persisted(ctx: CheckContext) -> str:
    from theme_builder.incremental import load_persisted, prev_corpus_cursor

    persisted = await load_persisted(ctx.driver, COMPAT_GROUP_ID)
    if not persisted:
        raise RuntimeError("load_persisted returned no communities despite the "
                           "community layer having been written")
    cursor = await prev_corpus_cursor(ctx.driver, COMPAT_GROUP_ID)
    if cursor is None:
        raise RuntimeError("prev_corpus_cursor is None despite a written community")
    members = persisted[0].members
    if ENT_A not in members:
        raise RuntimeError(f"community membership did not resolve: {members}")
    return f"load_persisted={len(persisted)} community(ies), prev_cursor={cursor}"
```

- [ ] **Step 6: Reorder `our_cypher_checks()` and register the new checks**

Replace the returned list in `our_cypher_checks()` with this exact order. The
community write must precede its readers, and the sweep must precede the timeline
flag check:

```python
def our_cypher_checks() -> list[Check]:
    return [
        CallableCheck("write community layer", "our-cypher", _write_community),
        CallableCheck("provenance resolve_citations", "our-cypher", _resolve_citations),
        CallableCheck("vendor episode scope", "our-cypher", _vendor_scope),
        CallableCheck("freshness stamps", "our-cypher", _freshness),
        CallableCheck("theme-builder corpus cursor (CALL {} UNION ALL)", "our-cypher",
                      _corpus_cursor_subquery),
        CallableCheck("incremental touched_entities", "our-cypher", _touched_entities),
        CallableCheck("incremental load_persisted", "our-cypher", _load_persisted),
        CallableCheck("global-search community shortlist", "our-cypher",
                      _shortlist_communities),
        CallableCheck("staleness sweep (scoped CALL (eps))", "our-cypher",
                      _staleness_sweep),
        CallableCheck("timeline sweep flags", "our-cypher", _timeline_sweep_flags),
        CallableCheck("GDS projection + seeded leiden detect", "our-cypher",
                      _leiden_detect),
    ]
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_compat_checks.py -q`
Expected: PASS

- [ ] **Step 8: Lint and type-check**

Run: `uv run ruff check src tests && uv run mypy src`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add src/compat/checks.py tests/unit/test_compat_checks.py
git commit -m "feat(compat): community-layer fixture and result-asserting our-Cypher checks"
```

---

### Task 3: e2e mini-ingestion, integration coverage, and the live report

**Files:**
- Modify: `src/compat/checks.py` (the `e2e` group)
- Modify: `tests/integration/test_compat_harness.py`
- Modify: `docs/superpowers/neo4j-compat-report.md` (regenerated by the live run)

**Interfaces:**
- Consumes: `E2E_ARTICLE_ID`, `E2E_ARTICLE_URL`, `VENDOR_ID`, `PRODUCT_ID`, `SOURCE_ID`, `VENDOR_NAME`, `PRODUCT_NAME`, `HEADING_PATH`, `COMPAT_GROUP_ID`, `_ENDPOINT_DOWN` from Tasks 1–2 and the existing module.
- Produces: no new interfaces.

- [ ] **Step 1: Rewrite the e2e check as a faithful mini-ingestion**

Replace `_e2e_ingest_and_retrieve` in `src/compat/checks.py` with:

```python
async def _e2e_ingest_and_retrieve(ctx: CheckContext) -> str:
    """One inlined article through the REAL pipeline: structural write -> extraction
    -> provenance link -> retrieval -> resolved citations.

    The link step matters. add_text_episode creates episodes but never attaches them
    to an :Article -- in production that is graph-sync's job. Without it, retrieval
    returns results whose citations resolve to nothing, which is exactly how this
    check used to pass while proving nothing about provenance.

    Inlined rather than fetched so the check does not depend on DocExtractor. Only
    extraction and retrieval -- never synthesis -- so it does not need the strong
    evaluation tier."""
    from datetime import datetime, timezone

    from answer_api.search import search_local
    from compat.runner import compat_target
    from graph_extract.graphiti_client import add_text_episode
    from graph_extract.provenance import Provenance
    from graph_sync.models import StructuralWrite
    from graph_sync.neo4j_repo import Neo4jRepo

    uri, user, password = compat_target(ctx.settings)
    repo = Neo4jRepo(uri, user, password)
    try:
        await repo.apply_structural(StructuralWrite(
            vendor={"id": VENDOR_ID, "name": VENDOR_NAME, "group_id": COMPAT_GROUP_ID},
            product={"id": PRODUCT_ID, "name": PRODUCT_NAME,
                     "group_id": COMPAT_GROUP_ID},
            source={"id": SOURCE_ID, "name": "compat source",
                    "group_id": COMPAT_GROUP_ID},
            article={"id": E2E_ARTICLE_ID, "title": "Vault Lock (e2e)",
                     "source_url": E2E_ARTICLE_URL, "source_id": SOURCE_ID,
                     "group_id": COMPAT_GROUP_ID}))
    finally:
        await repo.close()

    harness_settings = ctx.settings.model_copy(update={"group_id": COMPAT_GROUP_ID})
    try:
        added = await add_text_episode(
            ctx.graphiti, harness_settings, name="compat-e2e-article",
            body=_E2E_ARTICLE, source_description="compatibility harness",
            reference_time=datetime.now(timezone.utc))
    except _ENDPOINT_DOWN as exc:
        raise SkipCheck(
            f"model endpoint unreachable ({type(exc).__name__}: {exc})") from exc

    await Provenance(ctx.driver).link(
        E2E_ARTICLE_ID, added.episode.uuid, chunk_index=0,
        heading_path=HEADING_PATH, token_count=len(_E2E_ARTICLE.split()),
        content_hash="compat-check-e2e-hash")

    found = await search_local(
        ctx.graphiti, ctx.driver, q="What does Vault Lock enforce?", k=5,
        group_id=COMPAT_GROUP_ID)
    if found["count"] == 0:
        raise RuntimeError("search_local returned no results after ingest")
    resolved = [s for r in found["results"] for s in r["sources"]]
    if not resolved:
        raise RuntimeError(
            "search_local returned results but NO resolved sources -- the provenance "
            "chain (fact -> episode -> article -> url) did not resolve end to end")
    urls = {s["url"] for s in resolved}
    if E2E_ARTICLE_URL not in urls:
        raise RuntimeError(f"expected {E2E_ARTICLE_URL} among resolved sources, "
                           f"got {sorted(urls)}")
    return (f"ingested + linked; search_local returned {found['count']} result(s) "
            f"with {len(resolved)} resolved source(s)")
```

- [ ] **Step 2: Extend the integration test's teardown assertion**

In `tests/integration/test_compat_harness.py`, replace the body of
`test_teardown_leaves_no_harness_data` with a check across every fixture label:

```python
async def test_teardown_leaves_no_harness_data(extract_driver):
    """Teardown must remove the structural and community nodes too, not just the
    graphiti ones -- they are created through the real write path and would
    otherwise be left behind on a production instance."""
    async with extract_driver.session() as s:
        result = await s.run(
            "MATCH (n {group_id:'compat-check'}) "
            "RETURN labels(n) AS labels, count(n) AS n")
        rows = [dict(rec) async for rec in result]
    assert rows == [], f"residual harness nodes: {rows}"
```

- [ ] **Step 3: Run the unit and integration suites**

Run:
```
uv run --extra dev pytest tests/unit/test_compat_checks.py -q
uv run --extra dev pytest tests/integration/test_compat_harness.py -q
```
Expected: PASS. The integration run spins up a Neo4j testcontainer and takes a few
minutes. The `e2e` group may `pass` or `skip` depending on whether the configured
model endpoints are reachable from the machine; both are acceptable there. If the
harness reports a genuine `fail` in the `bootstrap`, `our-cypher`, or `graphiti-*`
groups, that is a real defect in Tasks 1–2 — fix it rather than relaxing the
assertion.

- [ ] **Step 4: Full gate**

Run: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -q`
Expected: all pass (`@live` excluded by default `addopts`).

- [ ] **Step 5: Re-run the harness live and regenerate the report**

The `COMPAT_NEO4J_*` target is already configured in the gitignored `.env`. Run:

```
uv run --extra dev python -m compat.cli
```

Report the verdict and the per-group tally verbatim. Then verify teardown directly
against the target — no `compat-check` nodes across any label, and no `compat_`
indexes:

```
uv run --extra dev python -c "
import asyncio
from neo4j import AsyncGraphDatabase
from compat.runner import compat_target
from graph_extract.config import get_extract_settings
async def m():
    uri, user, pw = compat_target(get_extract_settings())
    d = AsyncGraphDatabase.driver(uri, auth=(user, pw))
    async with d.session() as s:
        r = await s.run(\"MATCH (n {group_id:'compat-check'}) RETURN labels(n) AS l, count(n) AS n\")
        print('residual nodes:', [dict(x) async for x in r])
        r = await s.run(\"SHOW INDEXES YIELD name WHERE name STARTS WITH 'compat_' RETURN collect(name) AS n\")
        print('residual indexes:', [dict(x) async for x in r])
    await d.close()
asyncio.run(m())
"
```

If anything remains, clean it up and say so in your report.

- [ ] **Step 6: Update the report's scope section**

The live run regenerates `docs/superpowers/neo4j-compat-report.md`, but the
"Scope of this verdict" section is prose the CLI does not generate. Rewrite it so it
reflects the new coverage: the provenance join, the structural write path
(`apply_structural`, `Provenance.link`), the community layer (`write_communities_incremental`,
`load_persisted`, `shortlist_communities`), and both directions of the staleness
sweep are now **verified**, not merely parsed. The remaining disclosed gaps are
`drift`'s follow-up query, `apply_toc`, and the tombstone/`delete_source_articles`
paths. Do not overstate: if any group skipped in the live run, say so.

- [ ] **Step 7: Commit**

```bash
git add src/compat/checks.py tests/integration/test_compat_harness.py \
        docs/superpowers/neo4j-compat-report.md
git commit -m "test(compat): e2e mini-ingestion asserting resolved citations; re-run live"
```

---

## Verification checklist

Confirm every spec acceptance criterion after Task 3:

1. The fixture has the full structural chain, written via `apply_structural` and `Provenance.link`, every id `compat-check-`prefixed, every node carrying `group_id="compat-check"`.
2. `resolve_citations` asserts the expected url/vendor/product/section — the provenance join is verified, not merely parsed.
3. The sweep asserts it expires exactly `FACT_ORPHAN`, covering both directions.
4. e2e does structural write → extract → link → retrieve and asserts ≥1 resolved source.
5. The community layer is written via the real writeback and read back by `load_persisted` and `shortlist_communities`.
6. Teardown removes every fixture node, verified in the integration test and against the live target.
7. The regenerated report's scope section reflects the new coverage without overstating it.
8. Full non-live suite, ruff, and mypy clean.
