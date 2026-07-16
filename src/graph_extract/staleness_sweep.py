"""Residual-staleness sweep: expire facts whose supporting episodes are all dead.

Graphiti invalidates a fact (`invalid_at`) when it sees a *contradicting*
fact during extraction -- but a fact whose only supporting episode(s) were
silently removed (a `removed` tombstone from DocExtractor, or an article
that shrank and dropped a chunk-episode) leaves no contradiction for
Graphiti to observe. This module is the deterministic, Cypher-only backstop
for that case: a weekly sweep that expires facts with zero live supporting
episodes.

Correctness rule (see task brief "#6"): a supporting episode is DEAD iff
`removed=true` OR it has NO `HAS_EPISODE` edge from a non-removed
`:Article`. Liveness is keyed on HAS_EPISODE linkage, NEVER on the
`superseded` flag -- a superseded episode had its HAS_EPISODE edge deleted
by `Provenance.link`'s re-key (so it's dead by linkage anyway), but a
shrink-then-restore episode can be wrongly left `superseded=true` while
STILL holding its HAS_EPISODE edge, and that episode is ALIVE. Consulting
`superseded` directly would wrongly expire facts still backed by real
content -- so this sweep never reads it.

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
processes. Doing it as one query/transaction closes that window entirely.
Note: the newer `CALL (eps) { ... }` call-scope syntax is rejected by the
Neo4j 5.22 testcontainer used in tests/integration/conftest.py ("expected
an identifier or '{'") -- the classic `CALL { WITH eps ... }` importing
form below is what's portable and was verified against the real container,
including with the outer SET in the same statement.
"""

from __future__ import annotations

from neo4j import AsyncDriver

_SWEEP = """
MATCH ()-[f:RELATES_TO {group_id:$g}]->()
WHERE f.invalid_at IS NULL AND f.episodes IS NOT NULL AND size(f.episodes) > 0
WITH f, f.episodes AS eps
CALL {
  WITH eps
  UNWIND eps AS epu
  OPTIONAL MATCH (e:Episodic {uuid: epu})
  OPTIONAL MATCH (a:Article)-[:HAS_EPISODE]->(e) WHERE coalesce(a.removed, false) = false
  WITH e, count(a) AS live_links
  RETURN sum(
    CASE WHEN e IS NOT NULL AND coalesce(e.removed, false) = false AND live_links > 0
    THEN 1 ELSE 0 END
  ) AS alive
}
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
