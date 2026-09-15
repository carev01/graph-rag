"""`ingest_source` fans out over ARTICLES. Episodes stay sequential inside
`ingest_article`, because consecutive chunks of one document share entities most
heavily and graphiti resolves entities against the graph as it currently stands.
"""
from __future__ import annotations

import asyncio

import pytest

from graph_extract.config import ExtractSettings
from graph_extract.dedup_guard import DedupIndexGuard, DedupIndexStats
from graph_extract.ingest_driver import IngestArticleResult, IngestDriver


def _settings(**kw) -> ExtractSettings:
    base = dict(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    base.update(kw)
    return ExtractSettings(**base)


def _driver(settings, results):
    drv = IngestDriver.__new__(IngestDriver)
    drv._s = settings
    drv.warmup_gate = None
    drv._article_ids = list(results)
    drv._results = results

    async def _list(source_id, *a, **k):
        return list(results)

    async def _ingest(aid):
        return await results[aid]()

    drv.list_article_ids = _list
    drv.ingest_article = _ingest
    return drv


def _driver_with(ingest_article, *, concurrency: int, gate, n: int = 4):
    """A driver over a source of articles a1..a{n} whose `ingest_article` is the
    given coroutine function (it receives the article id) and whose warm-up gate
    is `gate` (None = no barrier)."""
    drv = IngestDriver.__new__(IngestDriver)
    drv._s = _settings(ingest_article_concurrency=concurrency)
    drv.warmup_gate = gate
    ids = [f"a{i}" for i in range(1, n + 1)]

    async def _list(source_id, *a, **k):
        return list(ids)

    drv.list_article_ids = _list
    drv.ingest_article = ingest_article
    return drv


def test_the_default_is_sequential():
    assert _settings().ingest_article_concurrency == 1


@pytest.mark.parametrize("value", [0, -1])
def test_a_concurrency_below_one_is_rejected(value):
    # Rejected, not clamped: a clamped 0 would silently run sequentially and hide
    # the misconfiguration from whoever set it.
    with pytest.raises(ValueError):
        _settings(ingest_article_concurrency=value)


async def test_articles_overlap_when_concurrency_is_raised():
    """`started` is recorded at the moment the slow article FINISHES, not after
    the batch: a strictly sequential loop also dispatches in list order, so an
    after-the-fact `started == [a, b, c]` would pass without any overlap."""
    started: list[str] = []
    finished: list[str] = []
    started_when_a_finished: list[str] = []

    def make(aid, delay, skipped=0):
        async def _work():
            started.append(aid)
            await asyncio.sleep(delay)
            finished.append(aid)
            if aid == "a":
                started_when_a_finished.extend(started)
            return IngestArticleResult(article_id=aid, episodes_added=1,
                                       episodes_skipped=skipped)
        return _work

    results = {"a": make("a", 0.05), "b": make("b", 0.0, skipped=2),
               "c": make("c", 0.0, skipped=3)}
    drv = _driver(_settings(ingest_article_concurrency=3), results)
    out = await drv.ingest_source("s")
    assert started_when_a_finished == ["a", "b", "c"], \
        "all dispatched before the slow one finished"
    assert finished == ["b", "c", "a"], "the fast siblings did not wait for the slow one"
    assert out.articles == 3 and out.episodes_added == 3
    assert out.episodes_skipped == 5


async def _in_flight_peak(concurrency: int, n: int = 4) -> tuple[int, list[str]]:
    """Run `n` equally slow articles at `concurrency`; return the peak number in
    flight at once and the completion order."""
    in_flight = 0
    peak = 0
    finished: list[str] = []

    def make(aid):
        async def _work():
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            finished.append(aid)
            return IngestArticleResult(article_id=aid, episodes_added=1)
        return _work

    ids = [f"a{i}" for i in range(n)]
    drv = _driver(_settings(ingest_article_concurrency=concurrency),
                  {aid: make(aid) for aid in ids})
    out = await drv.ingest_source("s")
    assert out.articles == n
    return peak, finished


async def test_the_setting_bounds_the_number_in_flight():
    """The configured value is what reaches the fan-out -- neither a hard-coded
    constant nor "everything at once"."""
    peak, _ = await _in_flight_peak(concurrency=2, n=4)
    assert peak == 2


async def test_concurrency_one_is_strictly_sequential():
    """The default preserves today's call order exactly: one article at a time,
    completing in list order."""
    peak, finished = await _in_flight_peak(concurrency=1, n=3)
    assert peak == 1
    assert finished == ["a0", "a1", "a2"]


@pytest.mark.parametrize("concurrency", [1, 3])
async def test_a_failing_article_does_not_prevent_its_siblings(concurrency):
    """Behaviour CHANGE, and it applies at any concurrency: today a raising
    article propagates out of the loop and every article after it is silently
    never attempted."""
    done: list[str] = []

    def ok(aid):
        async def _work():
            done.append(aid)
            return IngestArticleResult(article_id=aid, episodes_added=1)
        return _work

    async def boom():
        raise ValueError("boom")

    results = {"a": ok("a"), "b": boom, "c": ok("c")}
    drv = _driver(_settings(ingest_article_concurrency=concurrency), results)
    with pytest.raises(ValueError, match="boom"):
        await drv.ingest_source("s")
    assert done == ["a", "c"], "siblings must still have run"


async def test_the_first_failure_in_article_order_is_the_one_raised():
    """Article `a` fails AFTER an await, so it completes second: completion
    order is b, a while article order is a, b. Only the article order may
    decide which exception propagates."""
    def boom(msg, delay=0.0):
        async def _work():
            if delay:
                await asyncio.sleep(delay)
            raise ValueError(msg)
        return _work

    results = {"a": boom("first", delay=0.02), "b": boom("second")}
    drv = _driver(_settings(ingest_article_concurrency=2), results)
    with pytest.raises(ValueError, match="first"):
        await drv.ingest_source("s")


async def test_every_failure_is_logged_and_only_the_first_is_raised(caplog):
    """Five articles, three fail: none of the three may vanish. Each ERROR must
    name its own article id, and the exception raised must still be the first
    failure in article order."""
    def ok(aid):
        async def _work():
            return IngestArticleResult(article_id=aid, episodes_added=1)
        return _work

    def boom(aid):
        async def _work():
            raise ValueError(f"boom-{aid}")
        return _work

    results = {"a": ok("a"), "b": boom("b"), "c": ok("c"),
               "d": boom("d"), "e": boom("e")}
    drv = _driver(_settings(ingest_article_concurrency=5), results)
    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError, match="boom-b"):
            await drv.ingest_source("s")

    # ERROR, not WARNING: an article that failed hard and is about to be
    # re-raised is an error, the same level the worker gives an escape.
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 3, "every failing article must get its own ERROR"
    messages = [r.getMessage() for r in errors]
    for aid in ("b", "d", "e"):
        assert any(aid in m for m in messages), f"article {aid} must be named in an ERROR"
    for r in errors:
        assert r.exc_info is not None, "the traceback must be preserved, not just str(e)"


async def test_concurrent_articles_keep_their_own_dedup_attribution():
    """The design's §9.5: per-article dedup attribution stays correct under
    concurrency. This goes through the REAL `ingest_article` -- the scope it
    opens on `CURRENT_DEDUP_STATS` -- with only `_ingest_article` stubbed, and
    records through the guard's own `stats()` resolution. Two overlapping
    articles interleave across an await and record distinct counts; each must
    see only its own. A shared (non-context) holder would hand `a`'s second
    record to `b` and `b`'s to the unscoped sink."""
    guard = DedupIndexGuard(fallback=None, unscoped=DedupIndexStats(), raw=None)
    seen: dict[str, DedupIndexStats] = {}
    plan = {"a": (1, 0.01, 1), "b": (1, 0.02, 2)}   # (before, sleep, after)

    async def _fake_ingest(article_id, res):
        before, delay, after = plan[article_id]
        seen[article_id] = res.dedup
        guard.stats().invalid_calls += before
        await asyncio.sleep(delay)
        guard.stats().invalid_calls += after
        return res

    drv = IngestDriver.__new__(IngestDriver)
    drv._s = _settings(ingest_article_concurrency=2)
    drv.warmup_gate = None
    drv._ingest_article = _fake_ingest

    async def _list(source_id, *a, **k):
        return list(plan)

    drv.list_article_ids = _list
    out = await drv.ingest_source("s")
    assert seen["a"].invalid_calls == 2, f"a saw a sibling's records: {seen['a'].summary()}"
    assert seen["b"].invalid_calls == 3, f"b saw a sibling's records: {seen['b'].summary()}"
    assert guard.unscoped.invalid_calls == 0, "nothing may fall through to the unscoped sink"
    assert out.dedup.invalid_calls == 5


# --- The warm-up barrier -----------------------------------------------------
#
# Warmth is a property of the SOURCE, so every article of one `ingest_source`
# call asks the gate the same question. "Warm" means the source has at least
# `ingest_warmup_articles` articles with at least one live episode -- NOT that
# that many articles have finished: the count is over `HAS_EPISODE` edges as the
# graph currently stands.


async def test_the_first_articles_of_a_cold_source_do_not_overlap():
    """The opening articles run one at a time; once the source is warm the rest
    overlap. Asserted on FINISH-before-START, not on dispatch order -- a
    sequential implementation satisfies dispatch order identically.

    Two details of the fake gate are load-bearing, both modelled on the real one:
    an article counts as live from its FIRST episode (`live` is bumped at start,
    not at end -- `count(DISTINCT a)` over HAS_EPISODE is not a count of finished
    articles), and the predicate yields once (the real one awaits a Neo4j
    round-trip, during which a launched task progresses). With the count bumped
    at the end instead, a barrier that launches the cold article and forgets to
    wait for it is invisible: the next article's predicate under-counts, calls
    itself cold, and its own pre-launch drain repairs the missing wait -- the
    correct and the broken log coincide. Modelled faithfully, that same mutant
    reads the source WARM ONE ARTICLE EARLY: a2, launched and not waited for,
    links its first episode (n=2) while still extracting; a3's predicate then
    calls the source warm and a3 launches beside the still-running a2 -- the
    barrier visibly ends too soon, and the SECOND assertion (`end-a2 <
    start-a3`) catches it. a1 and a2 still do not overlap under the mutant
    (a2's own pre-launch drain finishes a1), so the first assertion is not
    what kills it."""
    log: list[str] = []
    live = {"n": 0}

    class _Gate:
        async def is_cold_source(self, source_id: str) -> bool:
            await asyncio.sleep(0)         # the count query's round-trip
            return live["n"] < 2

    async def fake_ingest_article(article_id: str):
        log.append(f"start-{article_id}")
        live["n"] += 1                     # first episode linked: now "live"
        await asyncio.sleep(0.03)          # ...the rest of the article extracts
        log.append(f"end-{article_id}")
        return IngestArticleResult(article_id=article_id)

    driver = _driver_with(fake_ingest_article, concurrency=4, gate=_Gate())
    await driver.ingest_source("s1")

    # a1 and a2 are cold: each finished before the next started
    assert log.index("end-a1") < log.index("start-a2")
    assert log.index("end-a2") < log.index("start-a3")
    # a3 and a4 are warm: they overlap
    assert log.index("start-a4") < log.index("end-a3")


async def test_ingest_source_asks_the_gate_about_the_source_it_is_ingesting():
    """The gate counts live articles of the source it is GIVEN. Asked about any
    other id it counts 0, answers cold forever, the whole source serialises and
    `_warm` never populates -- the barrier silently costing its own benefit,
    with no failure to notice. The overlap test's gate ignores its argument, so
    this fake records the id and pins it."""
    asked: list[str] = []

    class _Gate:
        async def is_cold_source(self, source_id: str) -> bool:
            asked.append(source_id)
            return False

    async def fake_ingest_article(article_id: str):
        return IngestArticleResult(article_id=article_id)

    driver = _driver_with(fake_ingest_article, concurrency=4, gate=_Gate(), n=3)
    await driver.ingest_source("s1")
    assert asked == ["s1", "s1", "s1"], (
        f"every article must ask about ITS source, s1; got {asked}")


async def test_warmup_is_inert_at_concurrency_one():
    """THE test that justifies shipping ingest_warmup_articles defaulted to 8.
    It must assert on ORDER, never on counts. If it is ever weakened, the
    default loses its justification (spec 3.6)."""
    orders = {}
    for threshold in (0, 8):
        log: list[str] = []

        class _Gate:
            async def is_cold_source(self, source_id: str) -> bool:
                return threshold > 0

        async def fake_ingest_article(article_id: str):
            log.append(f"start-{article_id}")
            await asyncio.sleep(0.01)
            log.append(f"end-{article_id}")
            return IngestArticleResult(article_id=article_id)

        driver = _driver_with(fake_ingest_article, concurrency=1, gate=_Gate())
        await driver.ingest_source("s1")
        orders[threshold] = list(log)
    assert orders[0] == orders[8], (
        f"warm-up must be inert at concurrency 1; got {orders}")


async def test_a_predicate_failure_propagates_out_of_ingest_source():
    """The CLI path and the worker path DELIBERATELY differ when the warm-up
    lookup raises. The worker (`semantic_worker._group_is_cold`) catches, logs
    at WARNING and answers cold: it is a long-running process with per-job
    backoff behind it, and a transient Neo4j blip must not kill it. This path
    has no such catch: `ingest_source` is what the `ingest` command runs, a
    human is watching, and a lookup failure must SURFACE. Swallowing it here
    would degrade a paid run into silent full serialisation -- every article
    "cold", one at a time, at concurrency 1 for the whole source, with nothing
    in the output saying so. A visible error is the cheaper failure.

    The observable is the exception reaching the caller: the failed article's
    slot holds it (run_concurrently), `ingest_source` re-raises it after the
    batch, and the article itself was never attempted. A copy of the worker's
    try/except-return-True here completes normally with all four ingested, so
    a test that only checked the articles ran would pass under both."""
    done: list[str] = []
    asked = {"n": 0}

    class _Gate:
        async def is_cold_source(self, source_id: str) -> bool:
            asked["n"] += 1
            if asked["n"] == 2:                # a blip on a2's lookup only
                raise ConnectionError("neo4j unreachable")
            return False

    async def fake_ingest_article(article_id: str):
        done.append(article_id)
        return IngestArticleResult(article_id=article_id)

    driver = _driver_with(fake_ingest_article, concurrency=4, gate=_Gate())
    with pytest.raises(ConnectionError, match="neo4j unreachable"):
        await driver.ingest_source("s1")
    assert sorted(done) == ["a1", "a3", "a4"], (
        f"siblings still run; the article whose lookup failed must NOT be "
        f"silently ingested as cold; got {done}")
