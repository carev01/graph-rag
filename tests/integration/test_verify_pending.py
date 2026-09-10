import json

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_verify_pending_recovers_staged_reports(extract_driver, monkeypatch):
    """theme-build --verify-pending: one verify call per staged community, no
    regeneration. Covers promote (clean + a dropped finding), still-pending
    (verifier still down), reject (unsupported summary), and leaves an
    already-verified community untouched."""
    import theme_builder.cli as tc
    from theme_builder.report import VerifyResult
    from graph_extract.config import get_extract_settings

    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("MATCH (n {group_id:$g}) DETACH DELETE n", g=g)   # clean slate
        # facts backing the staged findings
        for uuid, fact in [
            ("f1", "AWS Backup supports Amazon S3"),
            ("f2", "Veeam supports incremental backup"),
            ("f3", "Veeam guarantees zero data loss"),   # not stated by any fact -> dropped
            ("f4", "Rubrik supports immutable snapshots"),
            ("f5", "Cohesity supports ransomware detection"),
        ]:
            await s.run(
                "CREATE (a:Entity {uuid:$a, group_id:$g, name:$a}) "
                "CREATE (b:Entity {uuid:$b, group_id:$g, name:$b}) "
                "CREATE (a)-[:RELATES_TO {group_id:$g, uuid:$u, fact:$fact, name:'Rel'}]->(b)",
                a=f"{uuid}-a", b=f"{uuid}-b", g=g, u=uuid, fact=fact)

        # community 1: staged, everything verifies clean -> promoted
        await s.run(
            "CREATE (:Community {community_id:'promote1', group_id:$g, level:0, "
            "title:'T1', pending_summary:'Good summary', "
            "pending_full_report:$pfr, cited_fact_uuids:['f1'], verified:false})",
            g=g, pfr=json.dumps([{"finding": "F1 finding", "fact_ids": ["f1"]}]))

        # community 2: staged, one finding unsupported -> dropped, rest promoted
        await s.run(
            "CREATE (:Community {community_id:'dropfinding', group_id:$g, level:0, "
            "title:'T2', pending_summary:'Two findings summary', "
            "pending_full_report:$pfr, cited_fact_uuids:['f2','f3'], verified:false})",
            g=g, pfr=json.dumps([
                {"finding": "Good finding", "fact_ids": ["f2"]},
                {"finding": "Bad finding", "fact_ids": ["f3"]}]))

        # community 3: verifier endpoint still down -> stays staged
        await s.run(
            "CREATE (:Community {community_id:'stillpending', group_id:$g, level:0, "
            "title:'T3', pending_summary:'Pending summary', "
            "pending_full_report:$pfr, cited_fact_uuids:['f4'], verified:false})",
            g=g, pfr=json.dumps([{"finding": "Pending finding", "fact_ids": ["f4"]}]))

        # community 4: summary unsupported -> rejected outright
        await s.run(
            "CREATE (:Community {community_id:'rejectsummary', group_id:$g, level:0, "
            "title:'T4', pending_summary:'Reject summary', "
            "pending_full_report:$pfr, cited_fact_uuids:['f5'], verified:false})",
            g=g, pfr=json.dumps([{"finding": "Reject finding", "fact_ids": ["f5"]}]))

        # community 5: already verified -> must not be touched or re-verified
        await s.run(
            "CREATE (:Community {community_id:'alreadyverified', group_id:$g, level:0, "
            "title:'T5', summary:'Verified summary', full_report:'[]', "
            "cited_fact_uuids:[], embedding:[0.9], verified:true})",
            g=g)

    calls: list[str] = []

    async def _fake_verify(client, model, findings, summary, fact_texts):
        calls.append(summary)
        if summary == "Good summary":
            return VerifyResult(unsupported=set(), summary_supported=True)
        if summary == "Two findings summary":
            return VerifyResult(unsupported={2}, summary_supported=True)
        if summary == "Pending summary":
            return None
        if summary == "Reject summary":
            return VerifyResult(unsupported=set(), summary_supported=False)
        raise AssertionError(f"unexpected verify call for summary {summary!r}")
    monkeypatch.setattr(tc, "verify_report", _fake_verify)

    class _Closeable:
        async def close(self): pass
    monkeypatch.setattr(tc, "_verify_client_and_model", lambda s: (_Closeable(), "vm"))

    class _FakeEmb:
        class _C:
            async def close(self): pass
        client = _C()
        async def create_batch(self, texts): return [[0.42] for _ in texts]
    monkeypatch.setattr(tc, "build_embedder", lambda s: _FakeEmb())

    settings = get_extract_settings.__wrapped__().model_copy(update=dict(group_id=g))
    res = await tc._run_verify_pending(settings, driver=extract_driver)

    assert res == {"reports_promoted": 2, "reports_rejected": 1,
                   "reports_still_pending": 1, "findings_dropped": 1}
    # the already-verified community was never sent to the verifier
    assert "Verified summary" not in calls

    async with extract_driver.session() as s:
        rows = {r["cid"]: dict(r) async for r in await s.run(
            "MATCH (c:Community {group_id:$g}) RETURN c.community_id AS cid, "
            "c.summary AS summary, c.full_report AS full_report, "
            "c.pending_summary AS pending_summary, "
            "c.pending_full_report AS pending_full_report, "
            "c.cited_fact_uuids AS cited, c.embedding AS embedding, "
            "c.verified AS verified", g=g)}

    p1 = rows["promote1"]
    assert p1["summary"] == "Good summary"
    assert p1["full_report"] == json.dumps([{"finding": "F1 finding", "fact_ids": ["f1"]}])
    assert p1["verified"] is True
    assert p1["embedding"] == [0.42]
    assert p1["pending_summary"] is None
    assert p1["pending_full_report"] is None

    p2 = rows["dropfinding"]
    assert p2["summary"] == "Two findings summary"
    assert p2["full_report"] == json.dumps([{"finding": "Good finding", "fact_ids": ["f2"]}])
    assert p2["cited"] == ["f2"]
    assert p2["verified"] is True
    assert p2["embedding"] == [0.42]
    assert p2["pending_summary"] is None
    assert p2["pending_full_report"] is None

    p3 = rows["stillpending"]
    assert p3["pending_summary"] == "Pending summary"
    assert p3["pending_full_report"] is not None
    assert p3["embedding"] is None
    assert p3["verified"] is False

    p4 = rows["rejectsummary"]
    assert p4["pending_summary"] is None
    assert p4["pending_full_report"] is None
    assert p4["summary"] is None
    assert p4["embedding"] is None
    assert p4["verified"] is False

    p5 = rows["alreadyverified"]
    assert p5["summary"] == "Verified summary"
    assert p5["embedding"] == [0.9]
    assert p5["verified"] is True

    from answer_api.global_search import shortlist_communities

    class _QueryEmb:
        async def create_batch(self, texts): return [[0.42] for _ in texts]

    hits = await shortlist_communities(extract_driver, _QueryEmb(), "anything",
                                       level=0, k=10, group_id=g)
    ids = {h.community_id for h in hits}
    assert "promote1" in ids
    assert "dropfinding" in ids
    assert "stillpending" not in ids
    assert "rejectsummary" not in ids
