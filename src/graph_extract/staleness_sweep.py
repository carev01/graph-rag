"""Residual-staleness sweep: expire facts whose supporting episodes are all dead.

Graphiti invalidates a fact (`invalid_at`) when it sees a *contradicting*
fact during extraction -- but a fact whose only supporting episode(s) were
silently removed (a `removed` tombstone from DocExtractor, or an article
that shrank and dropped a chunk-episode) leaves no contradiction for
Graphiti to observe. This module is the deterministic, Cypher-only backstop
for that case: a weekly sweep that expires facts with zero live supporting
episodes.

Since 2026-09-12 this is, in practice, the PRIMARY invalidation mechanism, not a
backstop: ingest-time contradiction detection is suspended by default
(`ingest_detect_contradictions=False`, graph_extract.contradiction_gate), which
removes cross-pair invalidation entirely. The only remaining ingest-time path is
a same-pair contradiction through the duplicate candidates (BACKLOG 33), so most
`invalid_at` values written from here on come from this sweep. See the spec,
docs/superpowers/specs/2026-09-12-suspend-contradiction-detection-design.md.

Since 2026-09-23 that same-pair path is suppressed by default as well
(`ingest_same_pair_contradictions=False`: graph_extract.dedup_guard clears the
dedup reply's `contradicted_facts`), so ingest writes no contradiction-driven
invalidation at all and this sweep is the only one. Ingest can still set
`invalid_at` from an end date stated in the text.

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
  // Pattern comprehension, NOT a second OPTIONAL MATCH: that would cross-product
  // with the rows above and inflate `live_links`. Harmless today (the predicate
  // is `> 0`) and a trap the moment it becomes a count comparison.
  // An episode dies either because its article was tombstoned (removed_at, from
  // upstream's soft delete) or because a re-extraction superseded its link
  // (superseded_at, stamped with the REPLACING content's content_changed_at).
  // Both are properties of the content, not of when a job happened to run.
  WITH e, live_links,
       [(dead:Article)-[:HAS_EPISODE]->(e)
         WHERE dead.removed_at IS NOT NULL | dead.removed_at]
     + [(:Article)-[sl:HAS_EPISODE]->(e)
         WHERE sl.superseded_at IS NOT NULL | sl.superseded_at] AS deaths
  RETURN sum(CASE WHEN {ALIVE_EPISODE} THEN 1 ELSE 0 END) AS alive,
         max(reduce(m = null, d IN deaths |
             CASE WHEN m IS NULL OR d > m THEN d ELSE m END)) AS died_at
}}
WITH f, died_at WHERE alive = 0
// Date the expiry by WHEN THE SOURCE DIED, not when the sweep happened to run.
// The sweep's own clock is a schedule artefact of exactly the kind that made
// crawl-ordered `valid_at` meaningless. `removed_at` is upstream's soft-delete
// timestamp (100% populated on tombstones). The regex guard keeps one malformed
// value from failing the whole sweep -- `datetime()` throws on bad input and
// Cypher has no try. A superseded-but-not-removed episode carries no death
// timestamp anywhere, so those still fall back to now.
SET f.invalid_at = CASE
      WHEN died_at =~ '\\d{{4}}-\\d{{2}}-\\d{{2}}T.*' THEN datetime(died_at)
      ELSE datetime() END,
    f.expired_by_sweep = true
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
