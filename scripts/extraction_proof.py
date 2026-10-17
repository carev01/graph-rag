"""Extraction-grounding proof (spec 2026-10-17 section 3): re-extract a fixed sample of
articles through the PRODUCTION ingest path into a throwaway Neo4j testcontainer and
compare what comes out with what production holds. PAID (~$0.30 for the full sample)
-- run only with the user's authorisation, smoke first.

    uv run --extra dev python scripts/extraction_proof.py \
        --articles <id,...> [--title "<exact article title>" ...] \
        [--sample-affected N] [--token-cap 4000000] --out report.md

Two graphs, never mixed:

- `prod_read_driver` -- production, from `.env`. READ ONLY: every query runs in a read
  transaction (`routing_=READ`). Used to pick the sample, read each article's structural
  chain (to copy) and read production's facts for the before/after comparison.
- `test_write_driver` -- the throwaway container (`NEO4J_TEST_IMAGE`, the image the
  integration tests use). Every graph write goes here. `assert_test_target` refuses any
  write target that is not the container's URI (or that equals production's), and after
  the production ingest stack is built on the container, `verify_write_targets` checks
  that each of its drivers (IngestDriver/Provenance, strong graphiti, cheap graphiti)
  sees a sentinel written only to the container -- before any ingest write.

The ingest stack is `graph_extract.cli._build_ingest_driver` with `.env`'s settings
and only the Neo4j connection swapped for the container's: the same LLM tiers, the same
dedup/grounding guard on each tier, graphiti's index setup plus the tuned vector
indexes, and `IngestDriver.ingest_article` (content from DocExtractor via
`content_fetch.fetch_article`, chunking, document context, grounding context). The
provenance repository reads the container too, so the already-ingested gates see an
empty graph and every chunk is extracted. Articles run one at a time through
`ingest_article`, not `ingest_source`, so the fan-out, the warm-up barrier and sibling
spreading are not exercised; none of them touches extraction or the grounding gate.

Reported per article: facts written, grounding_calls / grounding_dropped and the
dropped names (first 20), foreign-unnamed-subject facts (test vs production),
RestoresTo facts (subject -> target), tokens and an estimated $ cost
(`haiku_extract_ab.RATES` at the article's routed tier). Totals cover only the
successfully ingested articles; failed, navigation-skipped, token-capped and unrun
articles are listed in their own section with the reason. A foreign-unnamed-subject
fact is one whose subject is a structural product name (production's Product nodes)
that differs from the article's product and is not contained in a supporting live
episode's text (spec section 1); superseded and removed episodes are ignored.

Spend guards: `--token-cap` (default 1.5M LLM tokens) is checked before every
episode's extraction, so the overshoot is at most one episode; and the run stops at
once, exit 1 "gate never fired", when an article extracted episodes while no
extraction reply went through the grounding gate.

Exit status: 1 with "gate never fired" on that early stop, when no extraction reply was
filtered, or when the gate dropped nothing on every successfully ingested article
production got wrong (or none was wrong); 2 when the gate fired but an article failed
or the token cap stopped the sample. The report is written in every case.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path
from typing import Any

from neo4j import AsyncGraphDatabase, RoutingControl
from testcontainers.neo4j import Neo4jContainer

from graph_extract.cli import _build_ingest_driver
from graph_extract.config import get_extract_settings
from graph_extract.usage import CURRENT_USAGE_TALLY, UsageTally, get_tally

sys.path.insert(0, str(Path(__file__).parent))
from haiku_extract_ab import RATES, article_cost  # noqa: E402

logger = logging.getLogger("extraction_proof")

NEO4J_TEST_IMAGE = "neo4j:2026.07.1-community"     # tests/integration/conftest.py
# The extraction instructions' former example products (spec section 1).
EXAMPLE_SUBJECTS = ("AWS Backup", "Azure Backup", "Veeam Backup & Replication")
DROPPED_NAMES_SHOWN = 20
FOREIGN_FACTS_SHOWN = 10
READ = RoutingControl.READ

# Structural products only: graphiti entities carry the Product label too, and have
# no id / no HAS_PRODUCT edge from a structural Vendor.
PRODUCT_NAMES_QUERY = (
    "MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product) "
    "WHERE v.id IS NOT NULL AND p.id IS NOT NULL "
    "RETURN collect(DISTINCT toLower(p.name)) AS names")

# Spec section 1: a fact whose subject is a structural product name, supported by a
# live (not superseded, not removed) episode of an article of a DIFFERENT product whose text does not name it.
# $example splits the sample: subject among $example_subjects (true) or not (false).
AFFECTED_ARTICLES_QUERY = (
    "MATCH (s:Entity)-[f:RELATES_TO]->(:Entity) "
    "WHERE toLower(s.name) IN $product_names "
    "AND (toLower(s.name) IN $example_subjects) = $example "
    "UNWIND f.episodes AS epu "
    "MATCH (e:Episodic {uuid: epu})<-[he:HAS_EPISODE]-(a:Article)"
    "<-[:HAS_ARTICLE]-(:Source)<-[:HAS_SOURCE]-(p:Product) "
    "WHERE p.id IS NOT NULL AND coalesce(he.superseded, false) = false "
    "AND coalesce(e.removed, false) = false "
    "AND NOT a.id IN $exclude "
    "AND toLower(p.name) <> toLower(s.name) "
    "AND NOT toLower(coalesce(e.content, '')) CONTAINS toLower(s.name) "
    "RETURN DISTINCT a.id AS id ORDER BY id LIMIT $limit")

# One article's facts (through its live episodes' MENTIONS, so production is not
# scanned edge by edge), each flagged foreign_unnamed per the definition above. Run
# against production AND the container with the same $product_names (production's).
ARTICLE_FACTS_QUERY = (
    "MATCH (a:Article {id: $a})-[he:HAS_EPISODE]->(e:Episodic) "
    "WHERE coalesce(he.superseded, false) = false AND coalesce(e.removed, false) = false "
    "MATCH (e)-[:MENTIONS]->(s:Entity)-[f:RELATES_TO]->(t:Entity) "
    "WHERE e.uuid IN f.episodes "
    "OPTIONAL MATCH (p:Product)-[:HAS_SOURCE]->(:Source)-[:HAS_ARTICLE]->(a) "
    "WHERE p.id IS NOT NULL "
    "WITH f, s, t, collect(DISTINCT toLower(p.name)) AS article_products, "
    "     collect(DISTINCT toLower(coalesce(e.content, ''))) AS texts "
    "RETURN f.uuid AS uuid, s.name AS subject, t.name AS target, f.name AS relation, "
    "       f.fact AS fact, "
    "       (toLower(s.name) IN $product_names "
    "        AND NOT toLower(s.name) IN article_products "
    "        AND any(x IN texts WHERE NOT x CONTAINS toLower(s.name))) AS foreign_unnamed "
    "ORDER BY uuid")

CHAIN_QUERY = (
    "MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->(s:Source)"
    "-[:HAS_ARTICLE]->(a:Article {id: $a}) "
    "WHERE v.id IS NOT NULL AND p.id IS NOT NULL "
    "OPTIONAL MATCH (a)-[:IN_CHAPTER]->(c:Chapter) "
    "RETURN properties(v) AS vendor, properties(p) AS product, properties(s) AS source, "
    "       properties(a) AS article, "
    "       CASE WHEN c IS NULL THEN NULL ELSE properties(c) END AS chapter "
    "ORDER BY s.id, c.id")

TITLE_QUERY = (
    "MATCH (a:Article {title: $t}) WHERE coalesce(a.removed, false) = false "
    "OPTIONAL MATCH (p:Product)-[:HAS_SOURCE]->(s:Source)-[:HAS_ARTICLE]->(a) "
    "WHERE p.id IS NOT NULL "
    "RETURN a.id AS id, p.name AS product, s.name AS source ORDER BY id")

COPY_CHAIN_WRITE = (
    "MERGE (v:Vendor {id: $vendor.id}) SET v += $vendor "
    "MERGE (p:Product {id: $product.id}) SET p += $product "
    "MERGE (s:Source {id: $source.id}) SET s += $source "
    "MERGE (a:Article {id: $article.id}) SET a += $article "
    "MERGE (v)-[:HAS_PRODUCT]->(p) "
    "MERGE (p)-[:HAS_SOURCE]->(s) "
    "MERGE (s)-[:HAS_ARTICLE]->(a)")
COPY_CHAPTER_WRITE = (
    "MATCH (a:Article {id: $aid}) "
    "MERGE (c:Chapter {id: $chapter.id}) SET c += $chapter "
    "MERGE (a)-[:IN_CHAPTER]->(c)")
SENTINEL_WRITE = "CREATE (:ExtractionProofSentinel {token: $t})"
SENTINEL_READ = "MATCH (n:ExtractionProofSentinel {token: $t}) RETURN count(n) AS c"


# --- production (read only) ----------------------------------------------------------

def parse_ids(raw: str) -> list[str]:
    out: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if part and part not in out:
            out.append(part)
    return out


async def read_product_names(prod_read_driver) -> list[str]:
    r = await prod_read_driver.execute_query(PRODUCT_NAMES_QUERY, routing_=READ)
    return sorted(r.records[0]["names"]) if r.records else []


async def select_affected(prod_read_driver, product_names: list[str], n: int, *,
                          exclude: list[str]) -> list[tuple[str, str]]:
    """N affected articles by ascending id: ceil(N/2) whose foreign-unnamed subject is
    an example product, then the rest whose subject is another product. The second
    bucket excludes the first's picks, so no article is sampled twice."""
    if n <= 0:
        return []
    picked: list[tuple[str, str]] = []
    excluded = list(exclude)
    for example, limit, group in ((True, n - n // 2, "affected-example"),
                                  (False, n // 2, "affected-other")):
        if limit <= 0:
            continue
        r = await prod_read_driver.execute_query(
            AFFECTED_ARTICLES_QUERY, routing_=READ, parameters_={
                "product_names": list(product_names),
                "example_subjects": [s.lower() for s in EXAMPLE_SUBJECTS],
                "example": example, "exclude": list(excluded), "limit": limit})
        ids = [rec["id"] for rec in r.records]
        if len(ids) < limit:
            logger.warning("only %d of %d %s articles found", len(ids), limit, group)
        picked.extend((i, group) for i in ids)
        excluded.extend(ids)
    return picked


async def resolve_titles(prod_read_driver, titles: list[str]) -> list[str]:
    """Exact-title lookup; refuses an unknown or ambiguous title (listing candidates)."""
    ids: list[str] = []
    for t in titles:
        r = await prod_read_driver.execute_query(TITLE_QUERY, routing_=READ,
                                                 parameters_={"t": t})
        found = sorted({rec["id"] for rec in r.records})
        if len(found) != 1:
            listing = "; ".join(f"{rec['id']} ({rec['product']} / {rec['source']})"
                                for rec in r.records) or "none"
            raise SystemExit(f"title {t!r} matches {len(found)} articles: {listing}. "
                             "Pass the id with --articles instead.")
        ids.append(found[0])
    return ids


async def read_chains(prod_read_driver, article_id: str) -> list[dict]:
    r = await prod_read_driver.execute_query(CHAIN_QUERY, routing_=READ,
                                             parameters_={"a": article_id})
    return [dict(rec) for rec in r.records]


async def read_article_facts(driver, article_id: str, product_names: list[str], *,
                             routing: RoutingControl = READ) -> list[dict]:
    r = await driver.execute_query(ARTICLE_FACTS_QUERY, routing_=routing, parameters_={
        "a": article_id, "product_names": list(product_names)})
    return [dict(rec) for rec in r.records]


@dataclass
class FactSummary:
    total: int
    foreign: list[tuple[str, str]]     # (subject, fact)
    restores: list[tuple[str, str]]    # (subject, target)


def _is_restores_to(relation: str | None) -> bool:
    return (relation or "").replace("_", "").lower() == "restoresto"


def summarise_facts(rows: list[dict]) -> FactSummary:
    return FactSummary(
        total=len(rows),
        foreign=[(r["subject"], r["fact"]) for r in rows if r["foreign_unnamed"]],
        restores=[(r["subject"], r["target"]) for r in rows if _is_restores_to(r["relation"])])


# --- the throwaway graph (every write) -------------------------------------------------

class WriteTargetRefused(RuntimeError):
    """A graph write would not land on the throwaway container."""


def _norm_uri(uri: str) -> str:
    return uri.strip().rstrip("/").lower()


def assert_test_target(write_uri: str, *, container_uri: str, production_uri: str) -> None:
    if _norm_uri(write_uri) != _norm_uri(container_uri):
        raise WriteTargetRefused(
            f"refusing to write: {write_uri!r} is not the test container ({container_uri!r})")
    if _norm_uri(write_uri) == _norm_uri(production_uri):
        raise WriteTargetRefused(f"refusing to write: {write_uri!r} is production")


class ThrowawayGraph:
    """The only object that writes; it cannot be built for a non-container URI."""

    def __init__(self, test_write_driver, uri: str, *, container_uri: str,
                 production_uri: str) -> None:
        assert_test_target(uri, container_uri=container_uri, production_uri=production_uri)
        self.driver = test_write_driver
        self.uri = uri

    async def write_sentinel(self, token: str) -> None:
        await self.driver.execute_query(SENTINEL_WRITE, parameters_={"t": token})

    async def copy_chain(self, chain: dict) -> None:
        await self.driver.execute_query(COPY_CHAIN_WRITE, parameters_={
            k: chain[k] for k in ("vendor", "product", "source", "article")})
        if chain.get("chapter"):
            await self.driver.execute_query(COPY_CHAPTER_WRITE, parameters_={
                "aid": chain["article"]["id"], "chapter": chain["chapter"]})

    async def facts(self, article_id: str, product_names: list[str]) -> list[dict]:
        return await read_article_facts(self.driver, article_id, product_names,
                                        routing=RoutingControl.WRITE)


async def verify_write_targets(drivers: dict[str, Any], token: str) -> None:
    """Each named driver must see the sentinel written only to the container."""
    for name, drv in drivers.items():
        r = await drv.execute_query(SENTINEL_READ, parameters_={"t": token})
        if not r.records or r.records[0]["c"] != 1:
            raise WriteTargetRefused(
                f"refusing to ingest: the {name} driver does not see the test container's "
                "sentinel, so it is not pointed at the throwaway graph")


def ingest_write_drivers(ingest, graphiti, ingest_neo4j) -> dict[str, Any]:
    """Every Neo4j driver the production ingest stack writes through, by name: the
    IngestDriver's and Provenance's (one driver), and each tier's graphiti."""
    if ingest._driver is not ingest_neo4j or ingest._prov._driver is not ingest_neo4j:
        raise WriteTargetRefused("IngestDriver/Provenance use an unexpected driver")
    drivers = {"ingest/provenance": ingest_neo4j, "strong graphiti": graphiti.driver.client}
    if ingest._strong.graphiti is not graphiti:
        raise WriteTargetRefused("the strong tier uses an unexpected graphiti")
    if ingest._cheap is not None:
        drivers["cheap graphiti"] = ingest._cheap.graphiti.driver.client
    return drivers


# --- outcome, verdict, report ------------------------------------------------------------

# Why an article is out of the totals (`ArticleOutcome.excluded`). Errors are their own
# field (`error`) and exclude the article too.
NAVIGATION_SKIP = "navigation page, skipped"
TOKEN_CAP_SKIP = "token cap reached"
GATE_ABORT_SKIP = "not run: ingest stopped because the gate never fired"


@dataclass
class ArticleOutcome:
    article_id: str
    title: str
    vendor: str
    product: str
    group: str                       # requested | affected-example | affected-other
    prod_facts: int = 0
    prod_foreign: int = 0
    prod_foreign_subjects: dict[str, int] = field(default_factory=dict)
    test_facts: int = 0
    test_foreign: int = 0
    test_foreign_facts: list[tuple[str, str]] = field(default_factory=list)
    restores: list[tuple[str, str]] = field(default_factory=list)
    grounding_calls: int = 0
    grounding_dropped: int = 0
    dropped_names: list[str] = field(default_factory=list)
    episodes_added: int = 0
    tier: str = ""
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0                # estimated $, haiku_extract_ab.RATES at `tier`
    note: str = ""
    excluded: str = ""               # NAVIGATION_SKIP / TOKEN_CAP_SKIP... / GATE_ABORT_SKIP
    error: str | None = None


def exclusion(o: ArticleOutcome) -> str:
    """Why `o` is out of the totals; "" when it was ingested successfully."""
    return f"error: {o.error}" if o.error else o.excluded


def included(outcomes: list[ArticleOutcome]) -> list[ArticleOutcome]:
    return [o for o in outcomes if not exclusion(o)]


def estimate_cost(tier: str, prompt: int, cached: int, completion: int) -> float:
    """haiku_extract_ab's per-tier rates (cached input at the cached rate). An article
    that failed before routing is priced at the strong tier, the dearer one."""
    return article_cost(tier if tier in RATES else "strong", prompt, cached, completion)


def engagement_abort(o: ArticleOutcome) -> str | None:
    """Episodes were extracted but no extraction reply reached the gate: the variable
    under test is not engaged, so spending on the rest of the sample proves nothing."""
    if o.episodes_added > 0 and o.grounding_calls == 0:
        return (f"gate never fired: article {o.article_id} added {o.episodes_added} "
                "episode(s) with grounding_calls == 0; stopped before ingesting the rest")
    return None


def gate_verdict(outcomes: list[ArticleOutcome], abort: str | None = None) -> str | None:
    """None when the gate demonstrably fired; otherwise the refusal."""
    if abort is not None:
        return abort
    if sum(o.grounding_calls for o in outcomes) == 0:
        return ("gate never fired: grounding_calls == 0 on every article (no extraction "
                "reply went through the grounding gate)")
    wrong = [o for o in included(outcomes) if o.prod_foreign > 0]
    if not wrong:
        return ("gate never fired: no successfully ingested article in the sample has a "
                "foreign-unnamed-subject fact in production, so nothing shows the gate "
                "dropping one")
    # Engagement is the gate SEEING the extraction replies (grounding_calls > 0 on
    # the articles production got wrong). It need not drop anything: the 2026-10-17
    # smoke showed the rewritten instructions alone stopped the inventions, leaving
    # the gate nothing to remove -- a zero there is a result, not a dead mechanism.
    if all(o.grounding_calls == 0 for o in wrong):
        return ("gate never fired: grounding_calls == 0 on every article production got "
                "wrong (" + ", ".join(o.article_id for o in wrong) + ")")
    return None


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _pct(before: int, after: int) -> str:
    return f"{(after - before) / before * 100:+.1f}%" if before else "n/a"


def render_report(outcomes: list[ArticleOutcome], *, meta: dict,
                  abort: str | None = None) -> str:
    verdict = gate_verdict(outcomes, abort)
    ok = included(outcomes)
    out = [o for o in outcomes if exclusion(o)]
    wrong = [o for o in ok if o.prod_foreign > 0]
    right = [o for o in ok if o.prod_foreign == 0]
    fired = sum(1 for o in wrong if o.grounding_dropped > 0)
    prod_foreign = sum(o.prod_foreign for o in ok)
    test_foreign = sum(o.test_foreign for o in ok)
    reduction = (f"reduction {(1 - test_foreign / prod_foreign) * 100:.1f}%"
                 if prod_foreign else "reduction n/a")
    right_prod = sum(o.prod_facts for o in right)
    right_test = sum(o.test_facts for o in right)
    groups = Counter(o.group for o in outcomes)
    lines = [
        "# Extraction grounding proof (throwaway Neo4j)", "",
        f"- Run: {meta.get('started', '?')}; commit `{meta.get('commit', '?')}`; "
        f"test image `{meta.get('image', '?')}`",
        f"- Articles: {len(outcomes)} ("
        + ", ".join(f"{g} {n}" for g, n in sorted(groups.items())) + ")",
        "- Gate verdict: " + (verdict or f"fired (dropped entities on {fired} of "
                                          f"{len(wrong)} articles production got wrong)"),
        f"- Failed articles: {sum(1 for o in outcomes if o.error)}; excluded from totals: "
        f"{len(out)}", "",
        f"## Totals (over {len(ok)} of {len(outcomes)} articles: the successfully ingested "
        "ones)", "",
        "| | production | test | |", "|---|---|---|---|",
        f"| facts | {sum(o.prod_facts for o in ok)} | {sum(o.test_facts for o in ok)} | |",
        f"| foreign-unnamed-subject facts | {prod_foreign} | {test_foreign} | {reduction} |",
        f"| facts on articles production got right | {right_prod} | {right_test} | "
        f"change {_pct(right_prod, right_test)} |", "",
        f"included articles={len(ok)} "
        f"grounding_calls={sum(o.grounding_calls for o in ok)} "
        f"grounding_dropped={sum(o.grounding_dropped for o in ok)} "
        f"episodes_added={sum(o.episodes_added for o in ok)}", "",
        f"Spend (every article run, included or not): "
        f"prompt_tokens={sum(o.prompt_tokens for o in outcomes)} "
        f"cached_tokens={sum(o.cached_tokens for o in outcomes)} "
        f"completion_tokens={sum(o.completion_tokens for o in outcomes)} "
        f"estimated cost ${sum(o.cost for o in outcomes):.3f} (haiku_extract_ab.RATES at "
        "each article's routed tier; an estimate)", "",
        "## Excluded from totals", "",
    ]
    lines += [f"- `{o.article_id}` {o.title}: {exclusion(o)}" for o in out] or ["- none"]
    lines += [
        "", "## Per article", "",
        "| article | title | group | product | facts prod -> test | foreign prod -> test "
        "| grounding calls / dropped | RestoresTo | tier | est. $ | status |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        status = exclusion(o) or o.note or "ok"
        lines.append(
            f"| `{o.article_id}` | {_cell(o.title)} | {o.group} | {_cell(o.product)} | "
            f"{o.prod_facts} -> {o.test_facts} | {o.prod_foreign} -> {o.test_foreign} | "
            f"{o.grounding_calls} / {o.grounding_dropped} | {len(o.restores)} | "
            f"{o.tier or '-'} | {o.cost:.4f} | {_cell(status)} |")
    lines += ["", "## Details", ""]
    for o in outcomes:
        lines += [f"### {o.title} (`{o.article_id}`)", "",
                  f"- {o.vendor} / {o.product}; {o.group}; episodes_added={o.episodes_added}; "
                  f"tokens {o.prompt_tokens} in ({o.cached_tokens} cached) / "
                  f"{o.completion_tokens} out; est. ${o.cost:.4f}"]
        if exclusion(o):
            lines.append(f"- excluded from totals: {exclusion(o)}")
        shown = o.dropped_names[:DROPPED_NAMES_SHOWN]
        lines.append(f"- dropped (first {DROPPED_NAMES_SHOWN} of {o.grounding_dropped}): "
                     + (", ".join(shown) if shown else "none"))
        subjects = ", ".join(f"{s} x{n}" for s, n in sorted(
            o.prod_foreign_subjects.items(), key=lambda kv: (-kv[1], kv[0])))
        lines.append(f"- production foreign-unnamed subjects: {subjects or 'none'}")
        lines.append(f"- test foreign-unnamed facts ({o.test_foreign}):")
        lines += [f"  - {s}: {fact}" for s, fact in o.test_foreign_facts[:FOREIGN_FACTS_SHOWN]]
        lines.append(f"- RestoresTo facts written ({len(o.restores)}):")
        lines += [f"  - {s} -> {t}" for s, t in o.restores]
        lines.append("")
    return "\n".join(lines) + "\n"


def finish(outcomes: list[ArticleOutcome], out: Path, *, meta: dict,
           abort: str | None = None) -> int:
    """Write the report, then the exit status: 1 gate never fired, 2 failures or a
    token-cap stop (the sample is incomplete), else 0."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(outcomes, meta=meta, abort=abort))
    print(f"report written to {out}")
    verdict = gate_verdict(outcomes, abort)
    if verdict is not None:
        print(verdict, file=sys.stderr)
        return 1
    failed = [o.article_id for o in outcomes if o.error]
    capped = [o.article_id for o in outcomes if o.excluded.startswith(TOKEN_CAP_SKIP)]
    if failed:
        print(f"{len(failed)} article(s) failed: {', '.join(failed)}", file=sys.stderr)
    if capped:
        print(f"{len(capped)} article(s) not completed (token cap): {', '.join(capped)}",
              file=sys.stderr)
    return 2 if failed or capped else 0


# --- the ingest loop ----------------------------------------------------------------------

class TokenCapReached(RuntimeError):
    """The run's LLM token budget is spent; raised before the next episode's extraction."""


def install_episode_token_cap(ingest, cap: int, spent: Callable[[], int]) -> None:
    """Stop before any episode's extraction once `cap` tokens are spent.

    IngestDriver asks its provenance gate (`already_ingested`) once per episode, right
    before that episode's `add_text_episode`. Wrapping that one method on THIS driver's
    Provenance instance is the cheapest per-episode hook: nothing else in the ingest
    path changes. The overshoot is at most the one episode in flight (~20-35k tokens)."""
    prov = ingest._prov
    orig = prov.already_ingested

    async def already_ingested(*args, **kwargs):
        used = spent()
        if used >= cap:
            raise TokenCapReached(f"{used} of {cap} tokens spent")
        return await orig(*args, **kwargs)

    prov.already_ingested = already_ingested


def _tokens() -> int:
    t = get_tally()
    return t.prompt_tokens + t.completion_tokens


async def ingest_sample(ingest, graph: ThrowawayGraph, outcomes: list[ArticleOutcome],
                        product_names: list[str], *, token_cap: int,
                        spent: Callable[[], int]) -> str | None:
    """Ingest every not-yet-failed article in order and read back its facts. Returns
    the engagement-abort message when the gate was not engaged (the rest of the sample
    is then left unrun), else None."""
    abort: str | None = None
    for n, o in enumerate(outcomes, 1):
        if o.error:
            continue
        if abort is not None:
            o.excluded = GATE_ABORT_SKIP
            continue
        if spent() >= token_cap:
            o.excluded = f"{TOKEN_CAP_SKIP} before this article ({spent()} of {token_cap})"
            continue
        tally = UsageTally()
        tok = CURRENT_USAGE_TALLY.set(tally)
        try:
            res = await ingest.ingest_article(o.article_id)
            o.episodes_added, o.tier = res.episodes_added, res.tier
            o.grounding_calls = res.grounding.calls
            o.grounding_dropped = res.grounding.dropped
            o.dropped_names = list(res.grounding.dropped_names)
            if res.skipped_navigation:
                o.excluded = NAVIGATION_SKIP
            elif res.grounding.parse_failures:
                o.note = f"grounding parse failures={res.grounding.parse_failures}"
            facts = summarise_facts(await graph.facts(o.article_id, product_names))
            o.test_facts, o.test_foreign = facts.total, len(facts.foreign)
            o.test_foreign_facts, o.restores = facts.foreign, facts.restores
        except TokenCapReached as e:
            o.excluded = f"{TOKEN_CAP_SKIP} mid-article ({e}); partially ingested"
        except Exception as e:          # recorded and reported, never skipped
            logger.exception("article %s failed", o.article_id)
            o.error = f"{type(e).__name__}: {e}"
        finally:
            CURRENT_USAGE_TALLY.reset(tok)
        o.prompt_tokens, o.cached_tokens = tally.prompt_tokens, tally.cached_tokens
        o.completion_tokens = tally.completion_tokens
        o.cost = estimate_cost(o.tier, o.prompt_tokens, o.cached_tokens, o.completion_tokens)
        abort = engagement_abort(o)
        print(f"[{n}/{len(outcomes)}] {o.article_id} facts {o.prod_facts}->{o.test_facts} "
              f"foreign {o.prod_foreign}->{o.test_foreign} grounding calls="
              f"{o.grounding_calls} dropped={o.grounding_dropped} restores={len(o.restores)} "
              f"est ${o.cost:.4f} tokens={spent()}"
              + (f" EXCLUDED {exclusion(o)}" if exclusion(o) else ""), flush=True)
        if abort is not None:
            print(abort, file=sys.stderr, flush=True)
    return abort


# --- the run ------------------------------------------------------------------------------

def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


async def _plan(prod_read_driver, args) -> tuple[list[str], list[tuple[str, str]]]:
    product_names = await read_product_names(prod_read_driver)
    if not product_names:
        raise SystemExit("production has no structural Product nodes; wrong database?")
    requested = parse_ids(args.articles or "")
    for i in await resolve_titles(prod_read_driver, args.title or []):
        if i not in requested:
            requested.append(i)
    sample = [(i, "requested") for i in requested]
    sample += await select_affected(prod_read_driver, product_names, args.sample_affected,
                                    exclude=requested)
    if not sample:
        raise SystemExit("empty sample: pass --articles, --title or --sample-affected")
    return product_names, sample


async def run(args) -> int:
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    settings = get_extract_settings()
    production_uri = settings.neo4j_uri
    # --- production, read only -------------------------------------------------------
    prod_read_driver = AsyncGraphDatabase.driver(
        production_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    outcomes: list[ArticleOutcome] = []
    chains: dict[str, list[dict]] = {}
    try:
        product_names, sample = await _plan(prod_read_driver, args)
        print(f"sample: {len(sample)} articles", flush=True)
        for article_id, group in sample:
            found = await read_chains(prod_read_driver, article_id)
            first = found[0] if found else {}
            o = ArticleOutcome(
                article_id=article_id, group=group,
                title=(first.get("article") or {}).get("title", "?"),
                vendor=(first.get("vendor") or {}).get("name", "?"),
                product=(first.get("product") or {}).get("name", "?"))
            if not found:
                o.error = "no structural chain in production (Vendor->Product->Source)"
            facts = summarise_facts(
                await read_article_facts(prod_read_driver, article_id, product_names))
            o.prod_facts, o.prod_foreign = facts.total, len(facts.foreign)
            o.prod_foreign_subjects = dict(Counter(s for s, _ in facts.foreign))
            chains[article_id] = found
            outcomes.append(o)
    finally:
        await prod_read_driver.close()

    # --- the throwaway graph: every write ---------------------------------------------
    abort: str | None = None
    with Neo4jContainer(NEO4J_TEST_IMAGE) as neo:
        container_uri = neo.get_connection_url()
        test_settings = settings.model_copy(update={
            "neo4j_uri": container_uri, "neo4j_user": "neo4j",
            "neo4j_password": neo.password})
        assert_test_target(test_settings.neo4j_uri, container_uri=container_uri,
                           production_uri=production_uri)
        test_write_driver = AsyncGraphDatabase.driver(
            container_uri, auth=("neo4j", neo.password))
        graph = ThrowawayGraph(test_write_driver, container_uri,
                               container_uri=container_uri, production_uri=production_uri)
        token = uuid.uuid4().hex
        try:
            await graph.write_sentinel(token)
            ingest, graphiti, docext, ingest_neo4j = await _build_ingest_driver(test_settings)
            try:
                await verify_write_targets(
                    ingest_write_drivers(ingest, graphiti, ingest_neo4j), token)
                print(f"write targets verified on {container_uri}; routing="
                      f"{'hybrid' if ingest._cheap is not None else 'strong-only'}", flush=True)
                for o in outcomes:
                    for chain in chains[o.article_id]:
                        await graph.copy_chain(chain)
                start = _tokens()

                def spent() -> int:
                    return _tokens() - start

                install_episode_token_cap(ingest, args.token_cap, spent)
                abort = await ingest_sample(ingest, graph, outcomes, product_names,
                                            token_cap=args.token_cap, spent=spent)
            finally:
                await ingest_neo4j.close()
                await docext.aclose()
                await graphiti.close()
                if ingest._cheap is not None:
                    await ingest._cheap.graphiti.close()
        finally:
            await test_write_driver.close()
    return finish(outcomes, Path(args.out), abort=abort,
                  meta={"started": started, "commit": _commit(), "image": NEO4J_TEST_IMAGE})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--articles", default="", help="comma-separated article ids")
    p.add_argument("--title", action="append", help="an exact article title (repeatable)")
    p.add_argument("--sample-affected", type=int, default=0,
                   help="add N affected articles (half example-product subjects)")
    p.add_argument("--token-cap", type=int, default=1_500_000,
                   help="LLM token budget for the ingest, checked before every episode's "
                        "extraction; the overshoot is at most one episode (~20-35k tokens)")
    p.add_argument("--out", required=True, help="markdown report path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s "
                        "%(message)s")
    for noisy in ("neo4j", "httpx", "httpcore", "openai", "testcontainers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
