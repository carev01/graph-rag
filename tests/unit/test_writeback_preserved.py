"""BACKLOG 5c, `--full` half. `write_communities` rebuilds the whole :Community
layer with a DETACH DELETE, writing back only what it is handed -- so a community
whose report generation failed was deleted along with the verified report it
already had. `preserved` carries such a community through untouched.

The same rule as the incremental path: a failed NEW attempt never removes what is
already persisted."""
import pytest

from theme_builder.detect import Community
from theme_builder.incremental import PersistedCommunity
from theme_builder.report import CommunityReport
from theme_builder.writeback import write_communities

REP = CommunityReport(title="T", summary="S", full_report="[]", rating=1.0,
                      rating_explanation="", tags=[], cited_fact_uuids=[])


class _CapturingTx:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def run(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return None

    def params_for(self, cid: str) -> dict:
        """The SET params of the write that created `cid` (the MERGE ... SET c +=
        call, not the later IN_COMMUNITY / PARENT_OF ones)."""
        for q, kw in self.calls:
            if kw.get("cid") == cid and "SET c +=" in q:
                return kw
        raise AssertionError(f"{cid} was never written")

    def wrote(self, cid: str) -> bool:
        return any(kw.get("cid") == cid and "SET c +=" in q for q, kw in self.calls)


class _FakeSession:
    def __init__(self, tx):
        self._tx = tx

    async def execute_write(self, fn):
        return await fn(self._tx)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeDriver:
    def __init__(self):
        self.tx = _CapturingTx()

    def session(self):
        return _FakeSession(self.tx)


class _FakeEmbedder:
    async def create_batch(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _community(cid: str, level: int, parent_id: str | None = None) -> Community:
    return Community(cid, level, [f"{cid}-e1", f"{cid}-e2"], parent_id)


def _persisted(cid: str, level: int, *, verified=True, embedding=(0.9,)) -> PersistedCommunity:
    return PersistedCommunity(
        community_id=cid, level=level, members={f"{cid}-e1"}, title="OLD T",
        summary="OLD S", full_report='[{"finding":"old"}]', rating=7.0,
        rating_explanation="old re", tags=["aws"], cited_fact_uuids=["f1"],
        embedding=list(embedding) if embedding else embedding,
        generated_at="2026-03-01", verified=verified,
        pending_summary="PS", pending_full_report='[{"finding":"pending"}]')


@pytest.mark.asyncio
async def test_a_preserved_community_is_not_counted_as_lost():
    comms = [_community("a", 0), _community("b", 1)]
    res = await write_communities(_FakeDriver(), _FakeEmbedder(), "g", comms,
                                  {"a": REP}, corpus_cursor=None,
                                  preserved={"b": _persisted("b", 1)})
    assert res["lost_by_level"] == {}
    assert res["reports_preserved"] == 1


@pytest.mark.asyncio
async def test_a_preserved_verified_report_keeps_its_text_embedding_and_age():
    """It must stay RETRIEVABLE -- shortlist_communities drops rows with no
    embedding, and losing it from retrieval is the harm being avoided. Its
    generated_at is the ORIGINAL report's: this run did not write it."""
    driver = _FakeDriver()
    comms = [_community("a", 0), _community("b", 1)]
    await write_communities(driver, _FakeEmbedder(), "g", comms, {"a": REP},
                            corpus_cursor="CUR", preserved={"b": _persisted("b", 1)})
    p = driver.tx.params_for("b")
    assert p["summary"] == "OLD S"
    assert p["full_report"] == '[{"finding":"old"}]'
    assert p["emb"] == [0.9], "the persisted embedding, not a fresh one"
    assert p["ga"] == "2026-03-01", "the original report's age, not this run's"
    assert p["cur"] == "CUR"
    q = [q for q, kw in driver.tx.calls if kw.get("cid") == "b" and "SET c +=" in q][0]
    assert "stale:true" in q, "carried over, so the next incremental run must retry it"
    assert "verified:true" in q, "it WAS verified; a failed new attempt does not unverify it"
    # The regenerated community is not stale -- prove the flag discriminates.
    qa = [q for q, kw in driver.tx.calls if kw.get("cid") == "a" and "SET c +=" in q][0]
    assert "stale:true" not in qa


@pytest.mark.asyncio
async def test_a_preserved_staged_report_stays_staged():
    """A staged row's `summary` reads back as '' (load_persisted coalesces it), so
    writing it in the normal shape would publish an EMPTY report -- and an
    unverified one, with an embedding, reachable from every answering path."""
    driver = _FakeDriver()
    comms = [_community("a", 0), _community("b", 1)]
    await write_communities(
        driver, _FakeEmbedder(), "g", comms, {"a": REP}, corpus_cursor=None,
        preserved={"b": _persisted("b", 1, verified=False, embedding=None)})
    p = driver.tx.params_for("b")
    assert p["summary"] == "PS", "staged writes pending_summary into pending_summary"
    assert "emb" not in p, "a staged report must carry NO embedding"
    assert p["full_report"] == '[{"finding":"pending"}]'
    q = [q for q, kw in driver.tx.calls if kw.get("cid") == "b" and "SET c +=" in q][0]
    assert "pending_summary:$summary" in q and "verified:false" in q


@pytest.mark.asyncio
async def test_a_preserved_community_keeps_its_members_and_hierarchy():
    """A node written without its IN_COMMUNITY edges is invisible to
    load_persisted's member collection, so the next run would see it as an empty
    community and never match it."""
    driver = _FakeDriver()
    comms = [_community("a", 0), _community("b", 1, parent_id="a")]
    await write_communities(driver, _FakeEmbedder(), "g", comms, {"a": REP},
                            corpus_cursor=None, preserved={"b": _persisted("b", 1)})
    assert any("IN_COMMUNITY" in q and kw.get("cid") == "b"
               for q, kw in driver.tx.calls), "preserved community lost its members"
    assert any("PARENT_OF" in q and kw.get("cid") == "b" and kw.get("pid") == "a"
               for q, kw in driver.tx.calls), "preserved community lost its parent"


@pytest.mark.asyncio
async def test_a_community_cannot_be_both_regenerated_and_preserved():
    """Both writes target the same node; the second `SET c +=` would merge into the
    first, and the resulting row would be neither cleanly new nor cleanly old."""
    comms = [_community("a", 0)]
    with pytest.raises(AssertionError):
        await write_communities(_FakeDriver(), _FakeEmbedder(), "g", comms,
                                {"a": REP}, corpus_cursor=None,
                                preserved={"a": _persisted("a", 0)})
