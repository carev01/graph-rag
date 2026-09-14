"""`ingest_source` fans out over ARTICLES. Episodes stay sequential inside
`ingest_article`, because consecutive chunks of one document share entities most
heavily and graphiti resolves entities against the graph as it currently stands.
"""
from __future__ import annotations

import asyncio

import pytest

from graph_extract.config import ExtractSettings
from graph_extract.ingest_driver import IngestArticleResult, IngestDriver


def _settings(**kw) -> ExtractSettings:
    base = dict(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    base.update(kw)
    return ExtractSettings(**base)


def _driver(settings, results):
    drv = IngestDriver.__new__(IngestDriver)
    drv._s = settings
    drv._article_ids = list(results)
    drv._results = results

    async def _list(source_id, *a, **k):
        return list(results)

    async def _ingest(aid):
        return await results[aid]()

    drv.list_article_ids = _list
    drv.ingest_article = _ingest
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
    def boom(msg):
        async def _work():
            raise ValueError(msg)
        return _work

    results = {"a": boom("first"), "b": boom("second")}
    drv = _driver(_settings(ingest_article_concurrency=2), results)
    with pytest.raises(ValueError, match="first"):
        await drv.ingest_source("s")
