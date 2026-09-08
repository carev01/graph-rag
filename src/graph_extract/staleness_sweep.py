"""Residual-staleness sweep: expire facts whose supporting episodes are all dead.

Graphiti invalidates a fact (`invalid_at`) when it sees a *contradicting*
fact during extraction -- but a fact whose only supporting episode(s) were
silently removed (a `removed` tombstone from DocExtractor, or an article
that shrank and dropped a chunk-episode) leaves no contradiction for
Graphiti to observe. This module is the deterministic, Cypher-only backstop
for that case: a weekly sweep that expires facts with zero live supporting
episodes.

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

A fact is expired iff ALL its supporting episodes are dead (no alive
episode survives). Only facts with `invalid_at IS NULL` and a non-empty
`episodes` list are candidates -- already-invalidated facts (by Graphiti or
a prior sweep run) are left untouched.

Find-then-SET is folded into a SINGLE Cypher statement (one transaction) so
the `invalid_at IS NULL` candidate filter is evaluated atomically with the
SET: if it were two separate auto-commit queries, Graphiti's own ingestion
process could invalidate a fact in the window between them, and the SET
(if unguarded) would clobber Graphiti's `invalid_at` with the sweep's
timestamp -- a real race, since the sweep and ingestion are independent
processes. Doing it as one query keeps the `invalid_at IS NULL` predicate
and the SET in a single statement, shrinking that window to negligible
(single-statement execution is read-committed, not fully serializable, but
a sweep-eligible fact has all-dead episodes, so a concurrent contradiction
of the same fact is vanishingly unlikely for a weekly batch job).
Uses the scoped `CALL (eps) { ... }` call-scope form (Neo4j 5.23+). The target is
Neo4j 2026.07 and nothing older is deployed, so the legacy importing-WITH form is
no longer needed.
"""

from __future__ import annotations

from neo4j import AsyncDriver

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

_SCAN_COUNT = """
MATCH ()-[f:RELATES_TO {group_id:$g}]->()
WHERE f.invalid_at IS NULL
RETURN count(f) AS c
"""


async def sweep_stale_facts(driver: AsyncDriver, group_id: str) -> dict:
    """Expire RELATES_TO facts with no live supporting episode.

    Returns `{"scanned": int, "expired": int, "expired_sample": list[str]}`.
    `scanned` counts every not-yet-invalidated fact in the group (the sweep's
    candidate pool, a separate read-only query -- race-free by nature);
    `expired` and `expired_sample` describe what this run actually flipped,
    computed and written by a single atomic query so a fact concurrently
    invalidated by Graphiti between the scan and the sweep is never
    overwritten.
    """
    async with driver.session() as s:
        scan_record = await (await s.run(_SCAN_COUNT, g=group_id)).single()
        assert scan_record is not None  # count() always returns exactly one row
        scanned = scan_record["c"]

        sweep_record = await (await s.run(_SWEEP, g=group_id)).single()
        assert sweep_record is not None  # count()/collect() always return exactly one row
        expired = sweep_record["expired"]
        expired_sample = sweep_record["sample"]

    return {
        "scanned": scanned,
        "expired": expired,
        "expired_sample": expired_sample,
    }
