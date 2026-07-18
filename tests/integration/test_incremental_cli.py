from datetime import datetime

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")

G = "backup-docs"


class _FakeEmbedder:
    class _Client:
        async def close(self):
            pass
    def __init__(self):
        self.client = _FakeEmbedder._Client()
    async def create_batch(self, texts):
        return [[0.5] for _ in texts]


class _FakeClient:
    async def close(self):
        pass


def _settings():
    from graph_extract.config import ExtractSettings
    return ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                           neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


async def _seed_persisted_and_entities(driver, *, e_a_created):
    async with driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        # persisted A {e1,e2}, B {e3,e4}, corpus_cursor T0
        for cid, members in (("sA", ["e1", "e2"]), ("sB", ["e3", "e4"])):
            await s.run("CREATE (c:Community {group_id:$g, community_id:$cid, level:1, title:'old', "
                        "summary:'old', full_report:'[]', rating:1.0, rating_explanation:'', tags:[], "
                        "cited_fact_uuids:[], embedding:[0.0], generated_at: datetime('2026-03-01'), "
                        "corpus_cursor:'2026-03-01T00:00:00Z', member_count:2})", g=G, cid=cid)
            for mu in members:
                await s.run("MATCH (c:Community {community_id:$cid, group_id:$g}) "
                            "MERGE (e:Entity {uuid:$mu, group_id:$g}) MERGE (e)-[:IN_COMMUNITY]->(c)",
                            cid=cid, g=G, mu=mu)
        # entity created_at: e1 varies (touch A or not), others old; plus an Episodic for _corpus_cursor
        await s.run("MATCH (e:Entity {group_id:$g}) SET e.created_at = datetime('2026-01-01')", g=G)
        await s.run("MATCH (e:Entity {uuid:'e1', group_id:$g}) SET e.created_at = datetime($t)", g=G, t=e_a_created)
        await s.run("CREATE (:Episodic {group_id:$g, uuid:'ep1', created_at: datetime('2026-04-01')})", g=G)


class _FrozenDatetime(datetime):
    """Pins cli._run_theme_build_incremental's `now` to 2026-04-15 so the
    generated_at assertions below are independent of the real wall clock."""
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 4, 15, tzinfo=tz)


def _patch(monkeypatch):
    import theme_builder.cli as cli
    from theme_builder.detect import Community
    from theme_builder.report import CommunityReport
    calls = {"n": 0}

    async def _fake_detect(driver, group_id, *, min_community_size, max_levels):
        return [Community("hA", 1, ["e1", "e2"], None), Community("hB", 1, ["e3", "e4"], None)]

    async def _fake_generate(client, model, ctx):
        calls["n"] += 1
        return CommunityReport(title="NEW", summary="NEW", full_report="[]", rating=9.0,
                               rating_explanation="", tags=[], cited_fact_uuids=[])

    monkeypatch.setattr(cli, "detect_communities", _fake_detect)
    monkeypatch.setattr(cli, "generate_report", _fake_generate)
    monkeypatch.setattr(cli, "build_embedder", lambda s: _FakeEmbedder())
    monkeypatch.setattr(cli, "_report_client_and_model", lambda s: (_FakeClient(), "m"))
    # context assembly is irrelevant (generate_report is faked); stub it so the
    # real _fetch_members/assemble_context path can't crash on unnamed seed entities.
    monkeypatch.setattr(cli, "assemble_context", lambda *a, **k: None)
    monkeypatch.setattr(cli, "datetime", _FrozenDatetime)
    return calls


async def test_incremental_regenerates_only_touched(extract_driver, monkeypatch):
    import theme_builder.cli as cli
    calls = _patch(monkeypatch)
    await _seed_persisted_and_entities(extract_driver, e_a_created="2026-04-15")   # e1 touched (> T0)
    res = await cli._run_theme_build_incremental(_settings(), driver=extract_driver)
    assert calls["n"] == 1                    # only community A regenerated
    assert res["reports_regenerated"] == 1 and res["reports_reused"] == 1
    async with extract_driver.session() as s:
        r = await s.run("MATCH (c:Community {group_id:$g, community_id:'sA'}) RETURN c.title AS t, toString(c.generated_at) AS ga", g=G)
        a = await r.single()
        assert a["t"] == "NEW" and a["ga"].startswith("2026-04")     # A regenerated, ga advanced
        r = await s.run("MATCH (c:Community {group_id:$g, community_id:'sB'}) RETURN c.title AS t, toString(c.generated_at) AS ga", g=G)
        b = await r.single()
        assert b["t"] == "old" and b["ga"].startswith("2026-03")     # B reused, ga unchanged (stable id kept)


async def test_incremental_no_change_regenerates_nothing(extract_driver, monkeypatch):
    import theme_builder.cli as cli
    calls = _patch(monkeypatch)
    await _seed_persisted_and_entities(extract_driver, e_a_created="2026-01-01")   # nothing after T0
    res = await cli._run_theme_build_incremental(_settings(), driver=extract_driver)
    assert calls["n"] == 0                    # THE headline: zero LLM report calls
    assert res["reports_reused"] == 2 and res["reports_regenerated"] == 0
