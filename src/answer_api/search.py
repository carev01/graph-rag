from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_NODE_DISTANCE, EDGE_HYBRID_SEARCH_RRF)

from answer_api.rerank import rerank, rerank_configured
from answer_api.scope import Scope, scope_episode_uuids
from answer_api.temporal import is_current
from graph_extract.config import ExtractSettings
from graph_extract.provenance import Provenance

logger = logging.getLogger(__name__)

# The scope's own facts, ranked by similarity to the query (the fallback when the
# corpus-wide top-N holds too few of them, see search_local).
#
# Never scans every RELATES_TO edge: from the scope's episodes (the structural chain,
# as scope._EPISODES) to the DISTINCT entities they mention, then those entities'
# edges, keeping an edge only if its `episodes` list names a scope episode. Any such
# edge is reached: the episode that produced it mentions its endpoints. Entities are
# de-duplicated BEFORE expanding -- expanding once per mention multiplies every hub's
# degree by its mention count (the six Veeam Agent guides timed out at 30 s that way;
# 1.4 s de-duplicated, 2026-10-13). `x IN uu` is not a linear scan per row in
# practice (Neo4j caches the list: the same check over all 240k edges takes ~1 s).
#
# A scope above $max_episodes is not scanned: `LIMIT $episode_cap` (max + 1) bounds
# the chain walk, and n = episode_cap tells the caller it was skipped. Neo4j cosine
# is (1+cos)/2; ranking is unaffected.
#
# The episode count is checked with this separate, vec-free query BEFORE the query
# is embedded (task 17, item 2): an over-cap scope then costs no embedding call.
_EPISODE_COUNT = (
    "MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->(s:Source)"
    "-[:HAS_ARTICLE]->(:Article)-[:HAS_EPISODE]->(e:Episodic) "
    "WHERE v.id IS NOT NULL AND p.id IS NOT NULL "
    "AND (toLower(v.name) IN $vendors OR toLower(p.name) IN $products "
    "OR toLower(s.name) IN $sources) "
    "WITH DISTINCT e LIMIT $episode_cap "
    "RETURN count(e) AS n")
_SCAN = (
    "MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->(s:Source)"
    "-[:HAS_ARTICLE]->(:Article)-[:HAS_EPISODE]->(e:Episodic) "
    "WHERE v.id IS NOT NULL AND p.id IS NOT NULL "
    "AND (toLower(v.name) IN $vendors OR toLower(p.name) IN $products "
    "OR toLower(s.name) IN $sources) "
    "WITH DISTINCT e LIMIT $episode_cap "
    "WITH collect(e) AS eps "
    "WITH eps, size(eps) AS n "
    "OPTIONAL CALL (eps, n) { "
    "  WITH eps WHERE 0 < n <= $max_episodes "
    "  CALL (eps) { "
    "    UNWIND eps AS e "
    "    MATCH (e)-[:MENTIONS]->(m:Entity) "
    "    RETURN collect(DISTINCT m) AS ms "
    "  } "
    "  WITH [x IN eps | x.uuid] AS uu, ms "
    "  UNWIND ms AS m "
    "  MATCH (m)-[f:RELATES_TO]-(:Entity) "
    "  WHERE f.group_id = $group_id AND NOT f.uuid IN $exclude "
    "  AND ($include_invalid OR f.invalid_at IS NULL OR f.invalid_at > datetime()) "
    "  AND any(x IN f.episodes WHERE x IN uu) "
    "  AND f.fact_embedding IS NOT NULL "
    "  WITH DISTINCT f "
    "  ORDER BY vector.similarity.cosine(f.fact_embedding, $vec) DESC LIMIT $k "
    "  RETURN f.uuid AS uuid, f.fact AS fact, f.episodes AS episodes, "
    "         f.valid_at AS valid_at, f.invalid_at AS invalid_at "
    "} "
    "RETURN n, uuid, fact, episodes, valid_at, invalid_at")


@dataclass(frozen=True)
class ScannedEdge:
    """The fields search_local reads from a graphiti EntityEdge."""
    uuid: str
    fact: str
    episodes: list[str]
    valid_at: datetime | None = None
    invalid_at: datetime | None = None


def _native(value: Any) -> Any:
    """neo4j.time.DateTime -> datetime, as graphiti's own edges carry."""
    return value.to_native() if hasattr(value, "to_native") else value


async def _retrieve_edges(graphiti, q, *, fetch_limit, group_id,
                          center_node_uuid=None) -> list:
    # SearchConfig.limit is the top-level fetch cap consulted by Graphiti._search
    # (graphiti-core==0.29.2). A center node switches edge reranking from RRF to
    # graph-distance (node_distance) so retrieval is biased toward that node.
    recipe = EDGE_HYBRID_SEARCH_NODE_DISTANCE if center_node_uuid else EDGE_HYBRID_SEARCH_RRF
    config = recipe.model_copy(deep=True)
    config.limit = fetch_limit
    results = await graphiti._search(q, config, group_ids=[group_id],
                                     center_node_uuid=center_node_uuid)
    return list(results.edges)


async def _scope_episode_count(driver, scope: Scope, max_episodes: int) -> int:
    """Episodes in `scope`, counted up to `max_episodes + 1` (enough to tell
    whether the scope is over the scan cap without counting all of it)."""
    records, _, _ = await driver.execute_query(
        _EPISODE_COUNT, vendors=[v.lower() for v in scope.vendors],
        products=[p.lower() for p in scope.products],
        sources=[x.lower() for x in scope.sources], episode_cap=max_episodes + 1)
    return records[0]["n"] if records else 0


async def _scoped_scan(graphiti, driver, *, q, scope: Scope, k: int, group_id: str,
                       exclude: set[str], include_invalid: bool = False,
                       max_episodes: int = 6000,
                       settings: ExtractSettings | None = None,
                       rerank: bool = True,
                       episode_count: int | None = None) -> list[ScannedEdge]:
    """The top `k` RELATES_TO edges of `scope` (not in `exclude`) for `q`.

    `q` is embedded exactly as graphiti's search embeds it (graphiti-core 0.30.1,
    search/search.py). With `settings` and a configured reranker, the top
    `local_scoped_scan_candidates` by cosine are reranked by the cross-encoder and
    the best `k` returned: cosine alone ranked Cohesity's first DataLock fact ~103rd
    of 2,158 for "Which backup vendors support immutable backups?" (2026-10-13).
    Without a reranker, or when it fails, the cosine top `k` -- never empty because
    of the reranker. A scope with more than `max_episodes` episodes returns []
    -- logged, not scanned: large scopes are served by the post-filter. That episode
    count is checked first, before `q` is embedded, so an over-cap scope costs no
    embedding call (task 17).
    `rerank=False` returns the cosine top `k` without the cross-encoder: a pooled
    caller reranks the merged candidates itself, once. `episode_count`, when the
    caller has already counted the scope, skips the count query."""
    episode_cap = max_episodes + 1
    vendors = [v.lower() for v in scope.vendors]
    products = [p.lower() for p in scope.products]
    sources = [x.lower() for x in scope.sources]
    if episode_count is None:
        episode_count = await _scope_episode_count(driver, scope, max_episodes)
    if episode_count > max_episodes:
        logger.info("scoped scan skipped: scope %s has more than %d episodes",
                    scope.as_dict(), max_episodes)
        return []
    use_rerank = rerank and settings is not None and rerank_configured(settings)
    fetch = max(k, settings.local_scoped_scan_candidates) \
        if settings is not None and use_rerank else k
    vec = await graphiti.embedder.create(input_data=[q.replace("\n", " ")])
    records, _, _ = await driver.execute_query(
        _SCAN, vec=vec, k=fetch, exclude=sorted(exclude), group_id=group_id,
        include_invalid=include_invalid, max_episodes=max_episodes,
        episode_cap=episode_cap, vendors=vendors, products=products, sources=sources)
    if records and records[0]["n"] > max_episodes:
        logger.info("scoped scan skipped: scope %s has more than %d episodes",
                    scope.as_dict(), max_episodes)
        return []
    cands = [ScannedEdge(uuid=r["uuid"], fact=r["fact"], episodes=list(r["episodes"] or []),
                         valid_at=_native(r["valid_at"]), invalid_at=_native(r["invalid_at"]))
             for r in records if r["uuid"] is not None]
    if settings is None or not rerank:
        return cands[:k]
    if not use_rerank:
        logger.warning("scoped scan: no reranker configured; returning cosine order")
        return cands[:k]
    return await _rerank_top(q, cands, k=k, settings=settings, what="scoped scan",
                             order_name="cosine order")


async def _rerank_top(q: str, cands: list, *, k: int, settings: ExtractSettings,
                      what: str, order_name: str) -> list:
    """The best `k` of `cands` for `q` by the cross-encoder. A short (or empty)
    rerank reply is padded from `cands` in their given order; a failed one (None or
    an exception) logs a warning and returns `cands[:k]` -- never empty because of
    the reranker."""
    if not cands:
        return []
    try:
        ranked = await rerank(q, [e.fact for e in cands], top_k=k, settings=settings)
    except Exception as exc:
        logger.warning("%s rerank failed (%s); returning %s", what, exc, order_name)
        return cands[:k]
    if ranked is None:
        logger.warning("%s rerank unavailable; returning %s", what, order_name)
        return cands[:k]
    order = list(dict.fromkeys(i for i, _ in ranked if 0 <= i < len(cands)))
    if len(order) < k:
        # A short (or empty) rerank reply still returns k: pad from the
        # candidates not already chosen, in their given order.
        chosen = set(order)
        order += [i for i in range(len(cands)) if i not in chosen]
    return [cands[i] for i in order][:k]


async def search_local(graphiti, driver, *, q, k=10, scope: Scope | None = None,
                       include_invalid=False, group_id, center_node_uuid=None,
                       fetch_cap: int = 450, scan_max_episodes: int = 6000,
                       settings: ExtractSettings | None = None, rerank_pool: int = 0):
    """`settings` enables the cross-encoder rerank of the scoped scan's candidates;
    None keeps the scan in cosine order.

    `rerank_pool` > 0 (facet, single-question and list searches), with `settings` and a configured
    reranker: gather up to max(k, rerank_pool) edges the same way (adaptive limits
    and scan fallback aimed at the pool size) and return the cross-encoder's best k
    of them -- the hybrid order alone ranked large scopes poorly (gate run 2,
    2026-10-13). A scoped pool within `scan_max_episodes` also always gets the
    scope's own top `local_scoped_scan_candidates` facts by cosine before that
    rerank (task 17). A failed rerank returns the original top k. 0 (the default)
    leaves the search exactly as it was."""
    scoped = scope is not None and not scope.is_empty()
    pooled = rerank_pool > 0 and settings is not None and rerank_configured(settings)
    target = max(k, rerank_pool) if pooled else k
    # Over-fetch so post-filters (validity, scope) still leave ~target results. A
    # narrow scope can filter out most of one fetch (the six Veeam Agent guides kept
    # 7 of 45, 2026-10-12), so a scoped search re-fetches wider until target survive.
    first = max(target * 3, target)
    # A pooled search in a scope small enough to scan merges the scope's own top
    # facts below whatever the post-filter finds, so widening the corpus-wide fetch
    # buys it nothing but time: 120 -> 400 -> 450 edges cost ~20 s per facet on AWS
    # and Azure Backup, the whole run 9 slowdown (fetch 450 alone is ~9 s). It
    # fetches once. Over-cap scopes (no scan) and a failed count keep the widening.
    episode_count: int | None = None
    if scoped and pooled:
        assert scope is not None
        try:
            episode_count = await _scope_episode_count(driver, scope, scan_max_episodes)
        except Exception as exc:
            logger.warning("scope episode count failed for scope %s: %s; widening the "
                           "fetch as for an unscannable scope", scope.as_dict(), exc)
    scannable = episode_count is not None and 0 < episode_count <= scan_max_episodes
    limits = [first] if not scoped or scannable else list(dict.fromkeys(
        min(n, fetch_cap) for n in (first, target * 10, target * 30)))
    edges: list = []
    for limit in limits:
        edges = await _retrieve_edges(graphiti, q, fetch_limit=limit,
                                      group_id=group_id,
                                      center_node_uuid=center_node_uuid)
        if not include_invalid:
            edges = [e for e in edges if is_current(getattr(e, "invalid_at", None))]
        if scoped:
            assert scope is not None
            allowed = await scope_episode_uuids(
                driver, scope, candidates=[u for e in edges for u in (e.episodes or [])])
            edges = [e for e in edges if allowed.intersection(e.episodes or [])]
        if len(edges) >= target:
            break
    edges = edges[:target]
    over_cap = episode_count is not None and episode_count > scan_max_episodes
    if scoped and (pooled or len(edges) < target) and not over_cap:
        # The corpus-wide top-N can hold none of a small scope's facts next to a
        # huge one (Cohesity's 29 DataLock/FortKnox facts vs Veeam's 20k articles,
        # 2026-10-12): rank the scope's own facts and append them after the
        # post-filter's. A pooled search ALWAYS adds the scope's own cosine
        # candidates (a scope over the episode cap returns none), even when the
        # post-filter filled the pool: AWS's "Restore testing" facts never entered
        # the corpus-wide search at any k, yet the scope scan ranks them first (run
        # 8, task 17); the pool's single rerank below orders the merged candidates.
        # A failure keeps what the post-filter found -- never raised, which would
        # empty the facet.
        assert scope is not None
        seen = {e.uuid for e in edges}
        if pooled:
            assert settings is not None
            scan_k = settings.local_scoped_scan_candidates
            cap = len(edges) + scan_k
        else:
            scan_k = target - len(edges)
            cap = target
        try:
            extra = await _scoped_scan(
                graphiti, driver, q=q, scope=scope, k=scan_k,
                group_id=group_id,
                exclude=seen, include_invalid=include_invalid,
                max_episodes=scan_max_episodes, settings=settings,
                rerank=not pooled, episode_count=episode_count)
        except Exception as exc:
            logger.warning("scoped scan failed for scope %s: %s; keeping %d "
                           "post-filter results", scope.as_dict(), exc, len(edges))
            extra = []
        for e in extra:
            if len(edges) >= cap:
                break
            if e.uuid not in seen:
                seen.add(e.uuid)
                edges.append(e)
    if pooled:
        assert settings is not None
        edges = await _rerank_top(q, edges, k=k, settings=settings,
                                  what="facet search", order_name="original order")
    citations = await Provenance(driver).resolve_citations([e.uuid for e in edges])
    return {
        "query": q, "count": len(edges),
        "results": [{"fact": e.fact, "fact_uuid": e.uuid,
                     "valid_at": getattr(e, "valid_at", None),
                     "invalid_at": getattr(e, "invalid_at", None),
                     "sources": citations.get(e.uuid, {}).get("sources", [])} for e in edges],
    }
