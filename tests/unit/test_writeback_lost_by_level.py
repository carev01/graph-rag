"""lost_by_level counts communities that were detected and end up neither written
nor staged -- genuinely dropped from the graph by write_communities' DETACH DELETE
rebuild. by_level only ever counted survivors, which is how level 1 silently lost
a 219-entity community for two slices: reports_skipped moved, but nothing said
WHICH level absorbed the loss, and global search reads level 1 only."""
import pytest

from theme_builder.detect import Community
from theme_builder.report import CommunityReport
from theme_builder.writeback import write_communities

REP = CommunityReport(title="T", summary="S", full_report="[]", rating=1.0,
                      rating_explanation="", tags=[], cited_fact_uuids=[])


class _FakeTx:
    async def run(self, query, **kwargs):
        return None


class _FakeSession:
    async def execute_write(self, fn):
        return await fn(_FakeTx())

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeDriver:
    def session(self):
        return _FakeSession()


class _FakeEmbedder:
    async def create_batch(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _community(cid: str, level: int, parent_id: str | None = None) -> Community:
    return Community(cid, level, [f"{cid}-e1", f"{cid}-e2"], parent_id)


@pytest.mark.asyncio
async def test_no_losses_reports_empty_dict():
    comms = [_community("a", 0), _community("b", 1)]
    reports = {"a": REP, "b": REP}
    res = await write_communities(_FakeDriver(), _FakeEmbedder(), "g", comms,
                                  reports, corpus_cursor=None)
    assert res["lost_by_level"] == {}


@pytest.mark.asyncio
async def test_one_level_1_community_with_no_report_is_lost():
    comms = [_community("a", 0), _community("b", 1)]
    reports = {"a": REP}  # "b" produced no report and was not staged
    res = await write_communities(_FakeDriver(), _FakeEmbedder(), "g", comms,
                                  reports, corpus_cursor=None)
    assert res["lost_by_level"] == {1: 1}


@pytest.mark.asyncio
async def test_staged_community_is_not_counted_as_lost():
    comms = [_community("a", 0), _community("b", 1)]
    reports = {"a": REP}
    pending = {"b": REP}  # "b" is staged, not dropped -- recoverable via --verify-pending
    res = await write_communities(_FakeDriver(), _FakeEmbedder(), "g", comms,
                                  reports, corpus_cursor=None, pending=pending)
    assert res["lost_by_level"] == {}


@pytest.mark.asyncio
async def test_losses_at_different_levels_are_counted_separately():
    comms = [
        _community("a", 0), _community("b", 0),
        _community("c", 1),
        _community("d", 2), _community("e", 2), _community("f", 2),
    ]
    # "a" is lost at level 0; "d" and "e" are lost at level 2; everything else written
    reports = {"b": REP, "c": REP, "f": REP}
    res = await write_communities(_FakeDriver(), _FakeEmbedder(), "g", comms,
                                  reports, corpus_cursor=None)
    assert res["lost_by_level"] == {0: 1, 2: 2}
