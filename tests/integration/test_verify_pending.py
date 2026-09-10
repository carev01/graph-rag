import json

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_verify_pending_recovers_staged_reports(extract_driver, monkeypatch):
    """theme-build --verify-pending: one verify call per staged community, no
    regeneration. Covers promote (clean + a dropped finding), still-pending
    (verifier still down), the summary verdict NOT costing a report, genuine
    rejection (every finding dropped), and leaving an already-verified community
    untouched."""
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
            ("f6", "Commvault supports air-gapped copies"),
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

        # community 4: summary judged unsupported but every finding survives ->
        # MUST be promoted. The summary verdict destroyed 15 of 41 communities and
        # is never a reason a report is lost (spec 4.4 / 4.5.5).
        await s.run(
            "CREATE (:Community {community_id:'summaryfailed', group_id:$g, level:0, "
            "title:'T4', pending_summary:'Unsupported summary', "
            "pending_full_report:$pfr, cited_fact_uuids:['f5'], verified:false})",
            g=g, pfr=json.dumps([{"finding": "Kept finding", "fact_ids": ["f5"]}]))

        # community 5: already verified -> must not be touched or re-verified
        await s.run(
            "CREATE (:Community {community_id:'alreadyverified', group_id:$g, level:0, "
            "title:'T5', summary:'Verified summary', full_report:'[]', "
            "cited_fact_uuids:[], embedding:[0.9], verified:true})",
            g=g)

        # community 6: EVERY finding unsupported -> the only genuine rejection
        await s.run(
            "CREATE (:Community {community_id:'allfindingsdropped', group_id:$g, level:0, "
            "title:'T6', pending_summary:'Doomed summary', "
            "pending_full_report:$pfr, cited_fact_uuids:['f6'], verified:false})",
            g=g, pfr=json.dumps([
                {"finding": "Doomed one", "fact_ids": ["f6"]},
                {"finding": "Doomed two", "fact_ids": ["f6"]}]))

    calls: list[str] = []

    async def _fake_verify(client, model, findings, summary, fact_texts):
        calls.append(summary)
        if summary == "Good summary":
            return VerifyResult(unsupported=set(), summary_supported=True)
        if summary == "Two findings summary":
            return VerifyResult(unsupported={2}, summary_supported=True)
        if summary == "Pending summary":
            return None
        if summary == "Unsupported summary":
            return VerifyResult(unsupported=set(), summary_supported=False)
        if summary == "Doomed summary":
            return VerifyResult(unsupported={1, 2}, summary_supported=True)
        raise AssertionError(f"unexpected verify call for summary {summary!r}")
    monkeypatch.setattr(tc, "verify_report", _fake_verify)

    regen_seen: list[list[str]] = []

    async def _fake_regen(client, model, findings, title):
        regen_seen.append([f["finding"] for f in findings])
        if title == "T4":
            return None                      # summariser hiccup -> keep the staged one
        return f"REGEN({title}): " + "; ".join(f["finding"] for f in findings)
    monkeypatch.setattr(tc, "_regenerate_summary", _fake_regen)

    class _Closeable:
        async def close(self): pass
    monkeypatch.setattr(tc, "_verify_client_and_model", lambda s: (_Closeable(), "vm"))
    monkeypatch.setattr(tc, "_report_client_and_model", lambda s: (_Closeable(), "rm"))

    class _FakeEmb:
        class _C:
            async def close(self): pass
        client = _C()
        async def create_batch(self, texts): return [[0.42] for _ in texts]
    monkeypatch.setattr(tc, "build_embedder", lambda s: _FakeEmb())

    settings = get_extract_settings.__wrapped__().model_copy(update=dict(group_id=g))
    res = await tc._run_verify_pending(settings, driver=extract_driver)

    assert res == {"reports_promoted": 3, "reports_rejected": 1,
                   "reports_still_pending": 1, "findings_dropped": 3}
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
    assert p1["summary"] == "REGEN(T1): F1 finding"
    assert p1["full_report"] == json.dumps([{"finding": "F1 finding", "fact_ids": ["f1"]}])
    assert p1["verified"] is True
    assert p1["embedding"] == [0.42]
    assert p1["pending_summary"] is None
    assert p1["pending_full_report"] is None

    p2 = rows["dropfinding"]
    # the promoted summary is REGENERATED from the kept findings -- promoting the
    # original would embed and publish a summary written against a dropped finding.
    assert p2["summary"] == "REGEN(T2): Good finding"
    assert "Bad finding" not in p2["summary"]
    assert p2["full_report"] == json.dumps([{"finding": "Good finding", "fact_ids": ["f2"]}])
    assert p2["cited"] == ["f2"]
    assert p2["verified"] is True
    assert p2["embedding"] == [0.42]
    assert p2["pending_summary"] is None
    assert p2["pending_full_report"] is None
    # the summariser only ever saw the surviving finding
    assert ["Good finding"] in regen_seen
    assert all("Bad finding" not in fs for fs in regen_seen)

    p3 = rows["stillpending"]
    assert p3["pending_summary"] == "Pending summary"
    assert p3["pending_full_report"] is not None
    assert p3["embedding"] is None
    assert p3["verified"] is False

    p4 = rows["summaryfailed"]
    assert p4["verified"] is True, "an unsupported SUMMARY must never lose a report"
    assert p4["full_report"] == json.dumps([{"finding": "Kept finding", "fact_ids": ["f5"]}])
    # the summariser returned nothing usable -> the staged summary is kept, not lost
    assert p4["summary"] == "Unsupported summary"
    assert p4["embedding"] == [0.42]
    assert p4["pending_summary"] is None
    assert p4["pending_full_report"] is None

    p5 = rows["alreadyverified"]
    assert p5["summary"] == "Verified summary"
    assert p5["embedding"] == [0.9]
    assert p5["verified"] is True

    p6 = rows["allfindingsdropped"]
    assert p6["pending_summary"] is None
    assert p6["pending_full_report"] is None
    assert p6["summary"] is None
    assert p6["embedding"] is None
    assert p6["verified"] is False

    from answer_api.global_search import shortlist_communities

    class _QueryEmb:
        async def create_batch(self, texts): return [[0.42] for _ in texts]

    hits = await shortlist_communities(extract_driver, _QueryEmb(), "anything",
                                       level=0, k=10, group_id=g)
    ids = {h.community_id for h in hits}
    assert "promote1" in ids
    assert "dropfinding" in ids
    assert "summaryfailed" in ids
    assert "stillpending" not in ids
    assert "allfindingsdropped" not in ids


async def test_verify_pending_tolerates_a_finding_with_no_fact_ids_key(
        extract_driver, monkeypatch):
    """A staged full_report is model-shaped JSON read back from the graph; a finding
    that lost its `fact_ids` key must not KeyError the whole recovery run."""
    import theme_builder.cli as tc
    from theme_builder.report import VerifyResult
    from graph_extract.config import get_extract_settings

    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("MATCH (n {group_id:$g}) DETACH DELETE n", g=g)
        await s.run(
            "CREATE (:Community {community_id:'nokey', group_id:$g, level:0, "
            "title:'T', pending_summary:'S', pending_full_report:$pfr, "
            "cited_fact_uuids:[], verified:false})",
            g=g, pfr=json.dumps([{"finding": "keyless"}]))

    async def _fake_verify(client, model, findings, summary, fact_texts):
        return VerifyResult(unsupported=set(), summary_supported=True)
    monkeypatch.setattr(tc, "verify_report", _fake_verify)

    async def _fake_regen(client, model, findings, title):
        return "regenerated"
    monkeypatch.setattr(tc, "_regenerate_summary", _fake_regen)

    class _Closeable:
        async def close(self): pass
    monkeypatch.setattr(tc, "_verify_client_and_model", lambda s: (_Closeable(), "vm"))
    monkeypatch.setattr(tc, "_report_client_and_model", lambda s: (_Closeable(), "rm"))

    class _FakeEmb:
        class _C:
            async def close(self): pass
        client = _C()
        async def create_batch(self, texts): return [[0.1] for _ in texts]
    monkeypatch.setattr(tc, "build_embedder", lambda s: _FakeEmb())

    settings = get_extract_settings.__wrapped__().model_copy(update=dict(group_id=g))
    res = await tc._run_verify_pending(settings, driver=extract_driver)
    assert res["reports_promoted"] == 1
