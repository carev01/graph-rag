# Temporal Coherence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make design decision #3 coherent — one definition of episode liveness, applied only by the staleness sweep, with citations that survive supersession.

**Architecture:** Stop deleting the `HAS_EPISODE` edge on re-key. Its **existence** means "citable"; a `superseded` flag **on the relationship** means "dead". One predicate module defines liveness; the sweep is its only evaluator and materialises the result into `invalid_at`, which search and timeline already consume unchanged.

**Tech Stack:** Python 3.12, `uv`, `neo4j` async driver, `graphiti-core==0.30.1`, pytest + testcontainers.

**Spec:** `docs/superpowers/specs/2026-09-04-temporal-coherence-design.md`

## Global Constraints

- **This is a deliberate contract change, not only a bug fix.** The invariant "exactly one `HAS_EPISODE` per (article, chunk_index)" becomes "exactly one **non-superseded** `HAS_EPISODE` per (article, chunk_index)". Three existing tests assert the old contract and must be updated to assert the new one — **strengthened, never weakened or deleted**.
- Liveness is decided **per edge**, never from the `Episodic` node's `superseded` property. One episode can be referenced by more than one article.
- A **missing** `superseded` property means **alive** (`coalesce(x, false)`), which is what makes the 161 existing unflagged edges correct with no backfill.
- `resolve_citations` must traverse `HAS_EPISODE` **regardless** of `superseded` — that is design decision #2 holding for historical answers. Do not add a liveness filter to it.
- The sweep is the **only** evaluator of the predicate. Do not add liveness filtering to `search_local` or `timeline_local`; they consume `invalid_at`.
- `Provenance.link` with a non-existent episode uuid must remain a **no-op** (an existing test pins this; the leading `MATCH (e:Episodic {uuid:$u})` yields zero rows, so nothing downstream executes).
- Target is Neo4j 2026.07.1 Community, `CYPHER_25` default, GDS 2026.07.0, **no APOC**.
- CI gate, clean at every commit: `uv run ruff check src tests` (lints tests too: E702 no semicolons, E402 imports at top), `uv run mypy src`, `uv run --extra dev pytest -m "not live"`.

## Verified current state (do not re-derive)

```python
# src/graph_extract/provenance.py:20-31 -- the DELETE being removed
"MATCH (a:Article {id:$a}) MATCH (e:Episodic {uuid:$u}) "
"OPTIONAL MATCH (a)-[old:HAS_EPISODE {chunk_index:$i}]->(oldE:Episodic) "
"WHERE oldE.uuid <> $u "
"SET oldE.superseded = true DELETE old "
"WITH a, e "
"MERGE (a)-[r:HAS_EPISODE {chunk_index:$i}]->(e) "
"SET r.heading_path=$hp, r.token_count=$tc, r.content_hash=$h, e.superseded = false"

# src/graph_extract/ingest_driver.py:111-114 -- flags the NODE, keeps the edge
"MATCH (:Article {id:$a})-[r:HAS_EPISODE]->(e:Episodic) "
"WHERE r.chunk_index >= $n SET e.superseded=true"
```

The live graph holds 161 `HAS_EPISODE` edges, none superseded. `sweep_stale_facts` returns `{"scanned", "expired", "expired_sample"}`.

---

### Task 1: The liveness predicate

**Files:**
- Create: `src/graph_extract/episode_liveness.py`
- Test: `tests/unit/test_episode_liveness.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ALIVE_LINK: str` (a Cypher boolean over bound `a` and `he`), `ALIVE_EPISODE: str` (a Cypher boolean over bound `e` and `live_links`). Tasks 4 and 5 compose these into queries.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_episode_liveness.py`:

```python
"""The one definition of episode liveness. These are string-shape tests; Task 5
proves the composed Cypher actually behaves correctly against a real Neo4j."""
from __future__ import annotations

from graph_extract.episode_liveness import ALIVE_EPISODE, ALIVE_LINK


def test_link_predicate_covers_article_removal_and_edge_supersession():
    assert "a.removed" in ALIVE_LINK
    assert "he.superseded" in ALIVE_LINK


def test_episode_predicate_covers_episode_removal_and_live_links():
    assert "e.removed" in ALIVE_EPISODE
    assert "live_links" in ALIVE_EPISODE


def test_missing_properties_default_to_alive():
    """A missing property must mean ALIVE -- this is what makes the 161 existing
    unflagged edges correct without a backfill migration."""
    for pred in (ALIVE_LINK, ALIVE_EPISODE):
        for prop in ("a.removed", "he.superseded", "e.removed"):
            if prop in pred:
                assert f"coalesce({prop}, false)" in pred, prop


def test_liveness_never_consults_the_episode_node_superseded_flag():
    """Liveness is per-EDGE: one episode can be referenced by more than one article,
    so a node-level flag would kill it for every article at once."""
    assert "e.superseded" not in ALIVE_LINK
    assert "e.superseded" not in ALIVE_EPISODE
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_episode_liveness.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'graph_extract.episode_liveness'`

- [ ] **Step 3: Write the module**

Create `src/graph_extract/episode_liveness.py`:

```python
"""The single definition of episode liveness (design decision #3).

`HAS_EPISODE` carries two independent meanings and they must not be conflated:

* the edge's **existence** means the episode is CITABLE -- `resolve_citations`
  traverses it regardless of any flag, so a fact that was ever true stays
  attributable to the document version that asserted it;
* the edge's **`superseded` flag** means the episode is DEAD -- it no longer
  reflects current content.

Before this module the system used edge *existence* for both, so `Provenance.link`
deleted the edge on re-key (destroying the citation) while
`_supersede_trailing_episodes` kept it (so a shrunk article's facts never expired).

An episode is ALIVE iff its edge is not superseded, the episode is not removed, and
the article is not removed. A fact is LIVE iff at least one supporting episode is
alive. **The staleness sweep is the only evaluator** -- it materialises the result
into `invalid_at`, which search and timeline consume. Evaluating it a second time in
retrieval would reintroduce exactly the divergence this module exists to remove.

Every clause uses `coalesce(..., false)` so a MISSING property means alive: that is
what lets pre-existing edges keep working with no migration.
"""
from __future__ import annotations

#: Is this Article->Episodic link live? Requires `a` (:Article) and `he`
#: (the HAS_EPISODE relationship) in scope. Deliberately reads the EDGE's
#: `superseded`, never the Episodic node's -- one episode may be referenced by
#: several articles, and it dies only for the one that superseded it.
ALIVE_LINK = (
    "coalesce(a.removed, false) = false "
    "AND coalesce(he.superseded, false) = false"
)

#: Is this episode alive? Requires `e` (:Episodic, possibly NULL from an
#: OPTIONAL MATCH) and `live_links` (the count of links passing ALIVE_LINK).
ALIVE_EPISODE = (
    "e IS NOT NULL "
    "AND coalesce(e.removed, false) = false "
    "AND live_links > 0"
)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra dev pytest tests/unit/test_episode_liveness.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src tests && uv run mypy src`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/graph_extract/episode_liveness.py tests/unit/test_episode_liveness.py
git commit -m "feat(temporal): single definition of episode liveness"
```

---

### Task 2: `Provenance.link` marks instead of deleting

**Files:**
- Modify: `src/graph_extract/provenance.py:18-31`
- Test: `tests/integration/test_provenance_rekey.py` (update three existing tests)

**Interfaces:**
- Consumes: nothing from Task 1 (this task changes writes, not reads).
- Produces: the new edge contract — after a re-key, **both** edges exist at the same `chunk_index`; the old one has `superseded = true`, the new one has no flag. Tasks 4–6 depend on this.

**Context:** `test_provenance_rekey.py` currently asserts `count(r) == 1` after a re-key. That assertion encodes the behaviour being removed. Update it to assert the new contract — two edges, exactly one live — rather than deleting it or relaxing it to `>= 1`.

- [ ] **Step 1: Update the tests to the new contract**

In `tests/integration/test_provenance_rekey.py`, replace `test_link_rekey_supersedes_and_dedupes` with:

```python
async def test_link_rekey_keeps_old_edge_and_flags_it(extract_driver):
    """The old edge SURVIVES so the superseded fact stays citable (design decision
    #2 for history); the flag is what makes it dead for liveness."""
    from graph_extract.provenance import Provenance
    p = Provenance(extract_driver)
    async with extract_driver.session() as s:
        await s.run("CREATE (:Article {id:'a1'})")
        await s.run("CREATE (:Episodic {uuid:'e_old'})")
        await s.run("CREATE (:Episodic {uuid:'e_new'})")
    await p.link("a1", "e_old", chunk_index=0, heading_path="", token_count=1, content_hash="h1")
    await p.link("a1", "e_new", chunk_index=0, heading_path="", token_count=1, content_hash="h2")
    async with extract_driver.session() as s:
        rows = [rec async for rec in await s.run(
            "MATCH (:Article {id:'a1'})-[r:HAS_EPISODE {chunk_index:0}]->(e:Episodic) "
            "RETURN e.uuid AS uuid, coalesce(r.superseded, false) AS sup ORDER BY uuid")]
    # BOTH edges present -- the old one is retained, not deleted
    assert [r["uuid"] for r in rows] == ["e_new", "e_old"]
    assert {r["uuid"]: r["sup"] for r in rows} == {"e_old": True, "e_new": False}
    # exactly one LIVE edge at this chunk_index -- the new invariant
    assert sum(1 for r in rows if not r["sup"]) == 1


async def test_link_rekey_is_idempotent(extract_driver):
    """Re-linking the same episode must not add another edge."""
    from graph_extract.provenance import Provenance
    p = Provenance(extract_driver)
    async with extract_driver.session() as s:
        await s.run("CREATE (:Article {id:'a1b'})")
        await s.run("CREATE (:Episodic {uuid:'e_b'})")
    for _ in range(3):
        await p.link("a1b", "e_b", chunk_index=0, heading_path="", token_count=1, content_hash="h")
    async with extract_driver.session() as s:
        n = (await (await s.run(
            "MATCH (:Article {id:'a1b'})-[r:HAS_EPISODE {chunk_index:0}]->() "
            "RETURN count(r) AS c")).single())["c"]
    assert n == 1
```

Then replace `test_link_re_activate_clears_superseded` with:

```python
async def test_link_re_activation_flips_which_edge_is_live(extract_driver):
    """Shrink-then-restore: relinking the original episode must make ITS edge live
    again and supersede the other. Both edges remain, so both stay citable."""
    from graph_extract.provenance import Provenance
    p = Provenance(extract_driver)
    async with extract_driver.session() as s:
        await s.run("CREATE (:Article {id:'a3'})")
        await s.run("CREATE (:Episodic {uuid:'e_old3'})")
        await s.run("CREATE (:Episodic {uuid:'e_new3'})")
    await p.link("a3", "e_old3", chunk_index=0, heading_path="", token_count=1, content_hash="h1")
    await p.link("a3", "e_new3", chunk_index=0, heading_path="", token_count=1, content_hash="h2")
    await p.link("a3", "e_old3", chunk_index=0, heading_path="", token_count=1, content_hash="h3")
    async with extract_driver.session() as s:
        rows = [rec async for rec in await s.run(
            "MATCH (:Article {id:'a3'})-[r:HAS_EPISODE {chunk_index:0}]->(e:Episodic) "
            "RETURN e.uuid AS uuid, coalesce(r.superseded, false) AS sup")]
    live = [r["uuid"] for r in rows if not r["sup"]]
    assert live == ["e_old3"]
    assert len(rows) == 2
```

Leave `test_link_bad_uuid_is_noop` and `test_link_supersede_does_not_delete_episode_node` unchanged — both still hold and both are load-bearing.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/integration/test_provenance_rekey.py -q`
Expected: FAIL — the current implementation deletes the old edge, so only one row comes back.

- [ ] **Step 3: Change the query**

In `src/graph_extract/provenance.py`, replace the body of `link`'s `s.run(...)` Cypher with:

```python
            await s.run(
                # The old edge is FLAGGED, never deleted: its existence is what makes
                # the superseded fact citable (design decision #2 for history), while
                # the flag is what makes it dead for liveness (see episode_liveness).
                # The leading MATCH on $u means a non-existent episode uuid yields zero
                # rows and the whole statement is a no-op -- a pinned behaviour.
                "MATCH (a:Article {id:$a}) "
                "MATCH (e:Episodic {uuid:$u}) "
                "OPTIONAL MATCH (a)-[old:HAS_EPISODE {chunk_index:$i}]->(oldE:Episodic) "
                "WHERE oldE.uuid <> $u "
                "SET old.superseded = true, oldE.superseded = true "
                "WITH a, e "
                "MERGE (a)-[r:HAS_EPISODE {chunk_index:$i}]->(e) "
                "SET r.heading_path=$hp, r.token_count=$tc, r.content_hash=$h, "
                "    r.superseded = false, e.superseded = false",
                a=article_id, u=episode_uuid, i=chunk_index, hp=heading_path,
                tc=token_count, h=content_hash)
```

Note `r.superseded = false` on the MERGE: re-activating a previously superseded episode must clear its own edge flag, which is what `test_link_re_activation_flips_which_edge_is_live` checks. The `oldE.superseded`/`e.superseded` node flags are retained for diagnostics only — nothing reads them for liveness.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/integration/test_provenance_rekey.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Confirm citations survive supersession**

Run: `uv run --extra dev pytest tests/integration/test_resolve_citations.py tests/integration/test_provenance.py -q`
Expected: PASS. These exercise `resolve_citations`; it must keep resolving and must not have gained a liveness filter.

- [ ] **Step 6: Commit**

```bash
git add src/graph_extract/provenance.py tests/integration/test_provenance_rekey.py
git commit -m "fix(temporal): flag the superseded HAS_EPISODE edge instead of deleting it"
```

---

### Task 3: `_supersede_trailing_episodes` uses the same mechanism

**Files:**
- Modify: `src/graph_extract/ingest_driver.py:111-114`
- Test: `tests/integration/test_temporal_update.py` (update `test_shrink_supersedes_trailing`)

**Interfaces:**
- Consumes: the edge contract from Task 2.
- Produces: dropped trailing chunks carry `superseded = true` on their **edge**, so Task 4's sweep can expire them.

- [ ] **Step 1: Update the test**

In `tests/integration/test_temporal_update.py`, replace `test_shrink_supersedes_trailing`'s assertions so it checks the **edge** flag (the node flag alone is what let Defect B through):

```python
async def test_shrink_supersedes_trailing(extract_driver):
    driver = _ingest_driver(extract_driver)
    async with extract_driver.session() as s:
        await s.run("""
        CREATE (a:Article {id:'a2'})
        CREATE (a)-[:HAS_EPISODE {chunk_index:0}]->(:Episodic {uuid:'e_s0'})
        CREATE (a)-[:HAS_EPISODE {chunk_index:1}]->(:Episodic {uuid:'e_s1'})
        CREATE (a)-[:HAS_EPISODE {chunk_index:2}]->(:Episodic {uuid:'e_s2'})
        """)
    await driver._supersede_trailing_episodes("a2", 1)
    async with extract_driver.session() as s:
        rows = [rec async for rec in await s.run(
            "MATCH (:Article {id:'a2'})-[r:HAS_EPISODE]->(e:Episodic) "
            "RETURN e.uuid AS uuid, coalesce(r.superseded, false) AS sup ORDER BY uuid")]
    # the EDGE carries the flag -- that is what the sweep reads
    assert {r["uuid"]: r["sup"] for r in rows} == {
        "e_s0": False, "e_s1": True, "e_s2": True}
    # every edge is retained, so the dropped chunks stay citable
    assert len(rows) == 3
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra dev pytest tests/integration/test_temporal_update.py::test_shrink_supersedes_trailing -q`
Expected: FAIL — the current code sets `e.superseded` on the node, not `r.superseded` on the edge, so all three come back `False`.

- [ ] **Step 3: Change the query**

In `src/graph_extract/ingest_driver.py`, replace `_supersede_trailing_episodes`'s Cypher:

```python
    async def _supersede_trailing_episodes(self, article_id: str, new_count: int) -> None:
        """A shrunk article drops trailing chunks. Flag the EDGE (not just the node)
        so the staleness sweep sees them as dead -- flagging only the node is what
        previously let facts from deleted content stay current forever."""
        async with self._driver.session() as s:
            await s.run(
                "MATCH (:Article {id:$a})-[r:HAS_EPISODE]->(e:Episodic) "
                "WHERE r.chunk_index >= $n "
                "SET r.superseded=true, e.superseded=true", a=article_id, n=new_count)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run --extra dev pytest tests/integration/test_temporal_update.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/graph_extract/ingest_driver.py tests/integration/test_temporal_update.py
git commit -m "fix(temporal): shrink flags the HAS_EPISODE edge, matching re-key"
```

---

### Task 4: The sweep consumes the shared predicate

**Files:**
- Modify: `src/graph_extract/staleness_sweep.py` (the module docstring and `_SWEEP`)
- Test: `tests/integration/test_staleness_sweep.py` (add one case)

**Interfaces:**
- Consumes: `ALIVE_LINK` and `ALIVE_EPISODE` from Task 1; the edge contract from Tasks 2–3.
- Produces: no signature change. `sweep_stale_facts(driver, group_id) -> {"scanned", "expired", "expired_sample"}` is unchanged.

**Context:** the module docstring currently argues at length that liveness must **never** read `superseded`, because `link` deletes the edge. Task 2 removed that premise. Leaving the docstring would leave a trap for the next reader — replace it.

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_staleness_sweep.py`:

```python
async def test_sweep_expires_facts_from_a_superseded_edge(extract_driver):
    """Defect B: a shrunk article's dropped chunk keeps its edge, so the old sweep
    (which keyed liveness on linkage alone) saw it as alive and never expired it."""
    from graph_extract.staleness_sweep import sweep_stale_facts
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("""
        CREATE (a:Article {id:'sup1'})
        CREATE (live:Episodic {uuid:'ep_live', group_id:'g'})
        CREATE (dead:Episodic {uuid:'ep_dead', group_id:'g'})
        CREATE (a)-[:HAS_EPISODE {chunk_index:0}]->(live)
        CREATE (a)-[:HAS_EPISODE {chunk_index:1, superseded:true}]->(dead)
        CREATE (x:Entity {uuid:'x', group_id:'g'})
        CREATE (y:Entity {uuid:'y', group_id:'g'})
        CREATE (x)-[:RELATES_TO {uuid:'f_live', group_id:'g', episodes:['ep_live']}]->(y)
        CREATE (x)-[:RELATES_TO {uuid:'f_dead', group_id:'g', episodes:['ep_dead']}]->(y)
        """)
    out = await sweep_stale_facts(extract_driver, "g")
    assert out["expired"] == 1
    assert out["expired_sample"] == ["f_dead"]
    async with extract_driver.session() as s:
        rows = [rec async for rec in await s.run(
            "MATCH ()-[f:RELATES_TO {group_id:'g'}]->() "
            "RETURN f.uuid AS uuid, f.invalid_at IS NOT NULL AS expired ORDER BY uuid")]
    assert {r["uuid"]: r["expired"] for r in rows} == {"f_dead": True, "f_live": False}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra dev pytest tests/integration/test_staleness_sweep.py::test_sweep_expires_facts_from_a_superseded_edge -q`
Expected: FAIL — `expired == 0`, because the current `_SWEEP` counts any `HAS_EPISODE` from a non-removed article as a live link.

- [ ] **Step 3: Rewrite the sweep query**

In `src/graph_extract/staleness_sweep.py`, add the import and rebuild `_SWEEP`:

```python
from graph_extract.episode_liveness import ALIVE_EPISODE, ALIVE_LINK

_SWEEP = f"""
MATCH ()-[f:RELATES_TO {{group_id:$g}}]->()
WHERE f.invalid_at IS NULL AND f.episodes IS NOT NULL AND size(f.episodes) > 0
WITH f, f.episodes AS eps
CALL (eps) {{
  UNWIND eps AS epu
  OPTIONAL MATCH (e:Episodic {{uuid: epu}})
  OPTIONAL MATCH (a:Article)-[he:HAS_EPISODE]->(e) WHERE {ALIVE_LINK}
  WITH e, count(he) AS live_links
  RETURN sum(CASE WHEN {ALIVE_EPISODE} THEN 1 ELSE 0 END) AS alive
}}
WITH f WHERE alive = 0
SET f.invalid_at = datetime(), f.expired_by_sweep = true
RETURN count(f) AS expired, collect(f.uuid)[..20] AS sample
"""
```

Note `count(he)` replaces `count(a)`: the count must be of links passing `ALIVE_LINK`, and `he` is the bound relationship.

- [ ] **Step 4: Replace the stale docstring**

In the same file, replace the paragraph beginning `Correctness rule (see task brief "#6")` and ending `so this sweep never reads it.` with:

```
Correctness rule (see graph_extract.episode_liveness, the single definition): a
supporting episode is ALIVE iff its `HAS_EPISODE` edge is not `superseded`, the
episode is not `removed`, and the article is not `removed`. Liveness is keyed on
the EDGE's flag, never the Episodic node's -- one episode can be referenced by
more than one article and dies only for the one that superseded it.

This module is the ONLY evaluator of that rule; it materialises the outcome into
`invalid_at`, which search and timeline consume. (Before 2026-09-04 the rule was
keyed on edge *existence* because `Provenance.link` deleted the edge on re-key --
which destroyed the citation for superseded facts, and meant a shrunk article's
dropped chunks, whose edges were kept, never expired at all.)
```

- [ ] **Step 5: Run the sweep tests**

Run: `uv run --extra dev pytest tests/integration/test_staleness_sweep.py -q`
Expected: PASS — the new case plus every pre-existing one. The older cases cover removal-based death, which `ALIVE_LINK`/`ALIVE_EPISODE` still express.

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check src tests && uv run mypy src`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/graph_extract/staleness_sweep.py tests/integration/test_staleness_sweep.py
git commit -m "fix(temporal): sweep reads the shared liveness predicate"
```

---

### Task 5: The agreement matrix

**Files:**
- Create: `tests/integration/test_temporal_coherence.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: no source interfaces.

**Context:** the property that was missing is not any single behaviour but *agreement* — the sweep, search and timeline reaching the same verdict. This test builds one graph covering every edge case in spec §6 and asserts that agreement in one place.

- [ ] **Step 1: Write the test**

Create `tests/integration/test_temporal_coherence.py`:

```python
"""Every temporal edge case from the spec's table, in one graph, asserted once.

The bug this slice fixed was not any single wrong behaviour -- each path was
internally consistent -- but that the paths disagreed. So the thing worth pinning
is agreement.
"""
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")

_GRAPH = """
MATCH (n) DETACH DELETE n
CREATE (art:Article {id:'art1', source_url:'https://example.invalid/art1'})
CREATE (gone:Article {id:'art2', removed:true, source_url:'https://example.invalid/art2'})
// 1. plain live episode
CREATE (e1:Episodic {uuid:'e1', group_id:'g'})
CREATE (art)-[:HAS_EPISODE {chunk_index:0}]->(e1)
// 2. re-keyed: old edge retained + flagged, new edge live
CREATE (e2old:Episodic {uuid:'e2old', group_id:'g'})
CREATE (e2new:Episodic {uuid:'e2new', group_id:'g'})
CREATE (art)-[:HAS_EPISODE {chunk_index:1, superseded:true}]->(e2old)
CREATE (art)-[:HAS_EPISODE {chunk_index:1}]->(e2new)
// 3. dropped trailing chunk (shrink)
CREATE (e3:Episodic {uuid:'e3', group_id:'g'})
CREATE (art)-[:HAS_EPISODE {chunk_index:2, superseded:true}]->(e3)
// 4. episode with NO superseded property at all -> must be alive (no backfill)
CREATE (e4:Episodic {uuid:'e4', group_id:'g'})
CREATE (art)-[:HAS_EPISODE {chunk_index:3}]->(e4)
// 5. episode whose only article is tombstoned
CREATE (e5:Episodic {uuid:'e5', group_id:'g'})
CREATE (gone)-[:HAS_EPISODE {chunk_index:0}]->(e5)
// 6. episode itself removed
CREATE (e6:Episodic {uuid:'e6', group_id:'g', removed:true})
CREATE (art)-[:HAS_EPISODE {chunk_index:4}]->(e6)
// 7. episode referenced by TWO articles, superseded by only one
CREATE (e7:Episodic {uuid:'e7', group_id:'g'})
CREATE (art)-[:HAS_EPISODE {chunk_index:5, superseded:true}]->(e7)
CREATE (other:Article {id:'art3', source_url:'https://example.invalid/art3'})
CREATE (other)-[:HAS_EPISODE {chunk_index:0}]->(e7)
CREATE (x:Entity {uuid:'x', group_id:'g'})
CREATE (y:Entity {uuid:'y', group_id:'g'})
CREATE (x)-[:RELATES_TO {uuid:'f1', group_id:'g', episodes:['e1']}]->(y)
CREATE (x)-[:RELATES_TO {uuid:'f2old', group_id:'g', episodes:['e2old']}]->(y)
CREATE (x)-[:RELATES_TO {uuid:'f2new', group_id:'g', episodes:['e2new']}]->(y)
CREATE (x)-[:RELATES_TO {uuid:'f3', group_id:'g', episodes:['e3']}]->(y)
CREATE (x)-[:RELATES_TO {uuid:'f4', group_id:'g', episodes:['e4']}]->(y)
CREATE (x)-[:RELATES_TO {uuid:'f5', group_id:'g', episodes:['e5']}]->(y)
CREATE (x)-[:RELATES_TO {uuid:'f6', group_id:'g', episodes:['e6']}]->(y)
CREATE (x)-[:RELATES_TO {uuid:'f7', group_id:'g', episodes:['e7']}]->(y)
"""


async def test_sweep_expires_exactly_the_dead_facts(extract_driver):
    from graph_extract.staleness_sweep import sweep_stale_facts
    async with extract_driver.session() as s:
        await s.run(_GRAPH)
    out = await sweep_stale_facts(extract_driver, "g")
    async with extract_driver.session() as s:
        rows = [rec async for rec in await s.run(
            "MATCH ()-[f:RELATES_TO {group_id:'g'}]->() "
            "RETURN f.uuid AS uuid, f.invalid_at IS NOT NULL AS dead ORDER BY uuid")]
    got = {r["uuid"]: r["dead"] for r in rows}
    assert got == {
        "f1": False,      # plain live
        "f2old": True,    # superseded by re-key
        "f2new": False,   # the replacement
        "f3": True,       # dropped trailing chunk
        "f4": False,      # no superseded property -> alive, no backfill needed
        "f5": True,       # article tombstoned
        "f6": True,       # episode removed
        "f7": False,      # still live via the OTHER article
    }
    assert out["expired"] == 4


async def test_superseded_facts_still_resolve_citations(extract_driver):
    """Design decision #2 must hold for history: a fact that was ever true stays
    attributable to the document version that asserted it."""
    from graph_extract.provenance import Provenance
    async with extract_driver.session() as s:
        await s.run(_GRAPH)
    resolved = await Provenance(extract_driver).resolve_citations(["f2old", "f3"])
    for uuid in ("f2old", "f3"):
        sources = resolved.get(uuid, {}).get("sources") or []
        assert sources, f"{uuid} lost its citation"
        assert sources[0]["url"] == "https://example.invalid/art1"


async def test_liveness_is_per_edge_not_per_episode(extract_driver):
    """e7 is superseded by art1 but still live via art3 -- a node-level flag would
    have killed it for both."""
    from graph_extract.staleness_sweep import sweep_stale_facts
    async with extract_driver.session() as s:
        await s.run(_GRAPH)
    await sweep_stale_facts(extract_driver, "g")
    async with extract_driver.session() as s:
        dead = (await (await s.run(
            "MATCH ()-[f:RELATES_TO {uuid:'f7'}]->() "
            "RETURN f.invalid_at IS NOT NULL AS dead")).single())["dead"]
    assert dead is False
```

- [ ] **Step 2: Run it**

Run: `uv run --extra dev pytest tests/integration/test_temporal_coherence.py -q`
Expected: PASS (3 passed). A failure here is a real defect in Tasks 1–4 — fix the source, not the expectations.

- [ ] **Step 3: Full gate**

Run: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_temporal_coherence.py
git commit -m "test(temporal): agreement matrix across every liveness edge case"
```

---

### Task 6: One real update, end to end

**Files:**
- Create: `tests/integration/test_temporal_update_live.py`
- Modify: `docs/superpowers/temporal-coherence-report.md` (create)

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: no source interfaces.

**Context and the decision already made for you:** the `semantic_jobs` queue has **never run** — 593 rows, all `pending`, empty `token_ledger`. The queue is the preferred vehicle for this test, but **the five assertions are the deliverable**. If the worker proves broken, record precisely what failed as a finding for a follow-up slice, then run the same assertions through `graph_extract.cli ingest` and ship. Do not attempt to fix the worker in this slice. Note the worker's daily budget default admits ~25 bootstrap articles/day, so inject a targeted job rather than draining the 593-row backlog.

- [ ] **Step 1: Write the `@live` test**

The ingest path (`ingest_driver.ingest_article`) does exactly this per article: chunk
the markdown, then for each chunk `add_text_episode(...)` followed by
`Provenance.link(article_id, r.episode.uuid, chunk_index=..., ...)`, and finally
`_supersede_trailing_episodes(article_id, len(episodes))`. This test drives that same
sequence directly with two hand-written versions, so it exercises the real write path
without depending on DocExtractor.

Create `tests/integration/test_temporal_update_live.py`:

```python
"""The scenario nobody has ever run: a real article, updated, through the pipeline.

Chunk 0 is edited and chunk 1 is dropped, then the temporal rule is asserted end to
end. Uses its own group_id so the pilot corpus is untouched.
"""
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")

_ARTICLE_ID = "temporal-live-article"
_ARTICLE_URL = "https://example.invalid/temporal-live"
_GROUP = "temporal-live"

_V1_CHUNK0 = "Vault Lock enforces a minimum retention period of 7 days."
_V1_CHUNK1 = "Vault Lock is available in all commercial AWS Regions."
_V2_CHUNK0 = "Vault Lock enforces a minimum retention period of 30 days."


@pytest.mark.live
async def test_article_update_preserves_citations_and_expires_dropped_content():
    from neo4j import AsyncGraphDatabase

    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import add_text_episode, build_graphiti
    from graph_extract.ingest_driver import IngestDriver
    from graph_extract.provenance import Provenance
    from graph_extract.staleness_sweep import sweep_stale_facts

    settings = get_extract_settings().model_copy(update={"group_id": _GROUP})
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    graphiti = build_graphiti(settings)
    prov = Provenance(driver)
    # Only the two Cypher-only helpers are used, so the other deps are unused here.
    ing = IngestDriver(settings, None, None, None, prov, driver)
    now = datetime.now(timezone.utc)

    async def _ingest(chunk_index: int, body: str) -> str:
        added = await add_text_episode(
            graphiti, settings, name=f"{_ARTICLE_ID}-c{chunk_index}", body=body,
            source_description=_ARTICLE_URL, reference_time=now)
        await prov.link(_ARTICLE_ID, added.episode.uuid, chunk_index=chunk_index,
                        heading_path="Retention", token_count=len(body.split()),
                        content_hash=f"h-{chunk_index}-{len(body)}")
        return added.episode.uuid

    async def _facts_for(episode_uuid: str) -> list[str]:
        async with driver.session() as s:
            r = await s.run(
                "MATCH ()-[f:RELATES_TO {group_id:$g}]->() "
                "WHERE $u IN f.episodes RETURN f.uuid AS uuid", g=_GROUP, u=episode_uuid)
            return [rec["uuid"] async for rec in r]

    try:
        async with driver.session() as s:
            await s.run("MATCH (n {group_id:$g}) DETACH DELETE n", g=_GROUP)
            await s.run(
                "MERGE (a:Article {id:$a}) SET a.source_url=$u, a.removed=false",
                a=_ARTICLE_ID, u=_ARTICLE_URL)

        # --- v1: two chunks ---
        ep0_old = await _ingest(0, _V1_CHUNK0)
        ep1 = await _ingest(1, _V1_CHUNK1)
        await ing._supersede_trailing_episodes(_ARTICLE_ID, 2)
        old_facts = await _facts_for(ep0_old)
        dropped_facts = await _facts_for(ep1)
        assert old_facts, "v1 chunk 0 produced no facts; cannot test supersession"
        assert dropped_facts, "v1 chunk 1 produced no facts; cannot test expiry"

        # --- v2: chunk 0 edited, chunk 1 dropped ---
        ep0_new = await _ingest(0, _V2_CHUNK0)
        await ing._supersede_trailing_episodes(_ARTICLE_ID, 1)

        # 1 (defect A): the superseded episode KEEPS its edge, flagged
        async with driver.session() as s:
            rows = [rec async for rec in await s.run(
                "MATCH (:Article {id:$a})-[r:HAS_EPISODE]->(e:Episodic) "
                "RETURN e.uuid AS uuid, coalesce(r.superseded, false) AS sup",
                a=_ARTICLE_ID)]
        flags = {r["uuid"]: r["sup"] for r in rows}
        assert flags.get(ep0_old) is True, "defect A: superseded edge was deleted"
        assert flags.get(ep0_new) is False, "the replacement edge must be live"
        assert flags.get(ep1) is True, "defect B: dropped chunk edge not flagged"

        # 2 (defect A): its facts still resolve to the article URL
        resolved = await prov.resolve_citations(old_facts)
        sources = resolved.get(old_facts[0], {}).get("sources") or []
        assert sources, "defect A: superseded fact lost its citation"
        assert sources[0]["url"] == _ARTICLE_URL

        # 3 (defect B): the sweep expires the dropped chunk's facts
        await sweep_stale_facts(driver, _GROUP)
        async with driver.session() as s:
            r = await s.run(
                "MATCH ()-[f:RELATES_TO {group_id:$g}]->() WHERE f.uuid IN $u "
                "RETURN f.uuid AS uuid, f.invalid_at IS NOT NULL AS dead, "
                "coalesce(f.expired_by_sweep, false) AS swept",
                g=_GROUP, u=dropped_facts + old_facts)
            state = {rec["uuid"]: (rec["dead"], rec["swept"]) async for rec in r}
        for uuid in dropped_facts:
            assert state[uuid] == (True, True), f"defect B: {uuid} survived the sweep"

        # 4: the superseded chunk-0 facts are also dead (their only edge is flagged)
        for uuid in old_facts:
            assert state[uuid][0] is True, f"{uuid} should be dead after supersession"

        # 5: the NEW chunk-0 facts are alive
        new_facts = await _facts_for(ep0_new)
        assert new_facts, "v2 chunk 0 produced no facts"
        async with driver.session() as s:
            r = await s.run(
                "MATCH ()-[f:RELATES_TO {group_id:$g}]->() WHERE f.uuid IN $u "
                "RETURN count(f) AS dead", g=_GROUP,
                u=new_facts)
            # none of the new facts may be invalidated
            r2 = await s.run(
                "MATCH ()-[f:RELATES_TO {group_id:$g}]->() "
                "WHERE f.uuid IN $u AND f.invalid_at IS NOT NULL RETURN count(f) AS n",
                g=_GROUP, u=new_facts)
            assert [dict(x) async for x in r2][0]["n"] == 0
    finally:
        await graphiti.close()
        await driver.close()
```

- [ ] **Step 2: Run the live test**

Run: `uv run --extra dev pytest -m live tests/integration/test_temporal_update_live.py -q`
Expected: PASS. Report each of the five assertion groups verbatim. If a `_facts_for`
guard trips ("produced no facts"), extraction returned nothing for that chunk — retry
once with slightly longer chunk text rather than weakening the assertion, since the
test is meaningless without facts to supersede.

- [ ] **Step 3: Then attempt the queue path**

The assertions above already prove
the temporal rule, so the queue is now attempted from a position of safety rather
than as a prerequisite. Insert one `semantic_jobs` row for the test article and run `uv run --extra dev python -m graph_sync.cli worker` briefly to drain it. Record in your report whether the worker: claimed the job, completed it, wrote to `token_ledger`, and left the job in a terminal state. If any step fails, capture the exact error, stop pursuing it, and proceed via `cli ingest`.

- [ ] **Step 4: Clean up the test namespace**

Run this and confirm zero rows, so the pilot corpus is unaffected:

```
uv run --extra dev python -c "
import asyncio
from neo4j import AsyncGraphDatabase
from graph_extract.config import get_extract_settings
async def m():
    s = get_extract_settings()
    d = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    async with d.session() as sess:
        await sess.run(\"MATCH (a:Article)-[r:HAS_EPISODE]->(e:Episodic {group_id:'temporal-live'}) DELETE r\")
        await sess.run(\"MATCH ()-[f:RELATES_TO {group_id:'temporal-live'}]->() DELETE f\")
        await sess.run(\"MATCH (n {group_id:'temporal-live'}) DETACH DELETE n\")
        r = await sess.run(\"MATCH (n {group_id:'temporal-live'}) RETURN count(n) AS n\")
        print('residue:', [dict(x) async for x in r])
        r = await sess.run('MATCH (n:Episodic) RETURN count(n) AS n')
        print('pilot episodes:', [dict(x) async for x in r])
    await d.close()
asyncio.run(m())
"
```

Expected: `residue: 0`, `pilot episodes: 247`.

- [ ] **Step 5: Write the demonstration report**

Create `docs/superpowers/temporal-coherence-report.md` covering: the two defects and how each is now prevented; the agreement matrix results; the five live assertions; whether the queue path worked (and if not, exactly how it failed, as input to a follow-up slice); and the two deferrals (43 phantom invalidations, label collision).

- [ ] **Step 6: Full gate and commit**

Run: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -q`

```bash
git add tests/integration/test_temporal_update_live.py docs/superpowers/temporal-coherence-report.md
git commit -m "test(temporal): live article-update proof; demonstration report"
```

---

## Verification checklist

1. `Provenance.link` no longer deletes `HAS_EPISODE`; re-keyed edges are flagged (Task 2).
2. `_supersede_trailing_episodes` flags the same edge property (Task 3).
3. Exactly one definition of liveness exists, in `episode_liveness.py`, and the sweep is its only evaluator; search and timeline are unchanged (Tasks 1, 4).
4. `resolve_citations` resolves a superseded fact to its source URL (Task 5).
5. The sweep expires facts from a shrunk article's dropped chunks (Tasks 4, 5).
6. The live update runs and all five assertions hold (Task 6).
7. The sweep docstring no longer claims liveness must ignore `superseded` (Task 4).
8. Full non-live suite, ruff and mypy clean.
