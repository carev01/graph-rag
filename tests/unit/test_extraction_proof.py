"""Hermetic tests for scripts/extraction_proof.py: the deterministic affected-article
selection, the write-target guard, the "gate never fired" refusal and the report."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from neo4j import RoutingControl

from graph_extract import usage
from graph_extract.grounding import GroundingStats

_ROOT = Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location(
    "extraction_proof", _ROOT / "scripts" / "extraction_proof.py")
proof = importlib.util.module_from_spec(_spec)
sys.modules["extraction_proof"] = proof     # dataclasses resolve their module
_spec.loader.exec_module(proof)


class FakeDriver:
    """Records every execute_query call; answers from a queue of record lists."""

    def __init__(self, responses=None):
        self.calls = []
        self._responses = list(responses or [])

    async def execute_query(self, query, parameters_=None, routing_=None, **kwargs):
        self.calls.append({"query": query, "params": dict(parameters_ or {}),
                           "routing": routing_})
        records = self._responses.pop(0) if self._responses else []
        return SimpleNamespace(records=records)


def _outcome(article_id="a1", **kw):
    base = dict(article_id=article_id, title=f"T {article_id}", vendor="Veeam",
                product="Securiti.ai", group="requested")
    base.update(kw)
    return proof.ArticleOutcome(**base)


# --- selection --------------------------------------------------------------

async def test_affected_selection_is_parameterised_and_split_in_halves():
    drv = FakeDriver([[{"id": "e1"}, {"id": "e2"}], [{"id": "o1"}, {"id": "o2"}]])
    picked = await proof.select_affected(drv, ["aws backup", "securiti.ai"], 4,
                                         exclude=["req1"])
    assert picked == [("e1", "affected-example"), ("e2", "affected-example"),
                      ("o1", "affected-other"), ("o2", "affected-other")]
    first, second = drv.calls
    for call in drv.calls:
        assert call["query"] == proof.AFFECTED_ARTICLES_QUERY
        assert call["routing"] == RoutingControl.READ
        assert call["params"]["product_names"] == ["aws backup", "securiti.ai"]
        assert call["params"]["example_subjects"] == [
            "aws backup", "azure backup", "veeam backup & replication"]
    assert first["params"]["example"] is True and first["params"]["limit"] == 2
    assert first["params"]["exclude"] == ["req1"]
    assert second["params"]["example"] is False and second["params"]["limit"] == 2
    # the second bucket never repeats an article the first one picked
    assert second["params"]["exclude"] == ["req1", "e1", "e2"]


def test_affected_query_orders_by_article_id_and_names_no_product_literal():
    q = proof.AFFECTED_ARTICLES_QUERY
    assert "ORDER BY id" in q and "LIMIT $limit" in q
    for literal in ("AWS Backup", "Azure Backup", "Veeam"):
        assert literal not in q
    assert "$product_names" in q and "$example_subjects" in q and "$exclude" in q


async def test_affected_selection_is_deterministic():
    def responses():
        return [[{"id": "e1"}], [{"id": "o1"}, {"id": "o2"}]]
    d1, d2 = FakeDriver(responses()), FakeDriver(responses())
    p1 = await proof.select_affected(d1, ["x"], 3, exclude=[])
    p2 = await proof.select_affected(d2, ["x"], 3, exclude=[])
    assert p1 == p2 and d1.calls == d2.calls
    # odd N: the example half gets the extra one
    assert [c["params"]["limit"] for c in d1.calls] == [2, 1]


async def test_affected_selection_of_zero_reads_nothing():
    drv = FakeDriver()
    assert await proof.select_affected(drv, ["x"], 0, exclude=[]) == []
    assert drv.calls == []


async def test_production_fact_reads_are_read_routed():
    drv = FakeDriver([[]])
    await proof.read_article_facts(drv, "a1", ["aws backup"])
    (call,) = drv.calls
    assert call["routing"] == RoutingControl.READ
    assert call["params"] == {"a": "a1", "product_names": ["aws backup"]}


def test_parse_ids_dedupes_and_keeps_order():
    assert proof.parse_ids(" b, a,,b ,c") == ["b", "a", "c"]
    assert proof.parse_ids("") == []


# --- facts summary ------------------------------------------------------------

def test_summarise_facts_counts_foreign_and_restores():
    rows = [
        {"uuid": "f1", "subject": "AWS Backup", "target": "Azure Blob Storage",
         "relation": "Supports", "fact": "AWS Backup supports Azure Blob Storage",
         "foreign_unnamed": True},
        {"uuid": "f2", "subject": "Veeam Agent", "target": "HPE Morpheus",
         "relation": "RestoresTo", "fact": "restore to Morpheus", "foreign_unnamed": False},
        {"uuid": "f3", "subject": "VBR", "target": "Proxmox VE",
         "relation": "RESTORES_TO", "fact": "restore to Proxmox", "foreign_unnamed": False},
    ]
    s = proof.summarise_facts(rows)
    assert s.total == 3
    assert s.foreign == [("AWS Backup", "AWS Backup supports Azure Blob Storage")]
    assert s.restores == [("Veeam Agent", "HPE Morpheus"), ("VBR", "Proxmox VE")]


# --- the write guard --------------------------------------------------------------

def test_guard_refuses_a_write_uri_that_is_not_the_container():
    with pytest.raises(proof.WriteTargetRefused):
        proof.assert_test_target("bolt://prod.lan:7687", container_uri="bolt://localhost:3201",
                                 production_uri="bolt://prod.lan:7687")


def test_guard_refuses_when_the_container_is_production():
    with pytest.raises(proof.WriteTargetRefused):
        proof.assert_test_target("bolt://prod.lan:7687", container_uri="bolt://prod.lan:7687",
                                 production_uri="bolt://PROD.lan:7687/")


def test_guard_accepts_the_container():
    proof.assert_test_target("bolt://localhost:3201", container_uri="bolt://localhost:3201",
                             production_uri="bolt://prod.lan:7687")


async def test_throwaway_graph_refuses_before_any_write():
    drv = FakeDriver()
    with pytest.raises(proof.WriteTargetRefused):
        proof.ThrowawayGraph(drv, "bolt://prod.lan:7687", container_uri="bolt://localhost:1",
                             production_uri="bolt://prod.lan:7687")
    assert drv.calls == []


async def test_copy_chain_writes_plain_merges_with_chapter():
    drv = FakeDriver()
    g = proof.ThrowawayGraph(drv, "bolt://localhost:1", container_uri="bolt://localhost:1",
                             production_uri="bolt://prod.lan:7687")
    chain = {"vendor": {"id": "v"}, "product": {"id": "p"}, "source": {"id": "s"},
             "article": {"id": "a", "content_hash": "h"}, "chapter": {"id": "c"}}
    await g.copy_chain(chain)
    assert len(drv.calls) == 2
    assert drv.calls[0]["params"]["article"] == {"id": "a", "content_hash": "h"}
    assert "MERGE (a:Article" in drv.calls[0]["query"]
    assert drv.calls[1]["params"] == {"aid": "a", "chapter": {"id": "c"}}
    drv2 = FakeDriver()
    g2 = proof.ThrowawayGraph(drv2, "bolt://localhost:1", container_uri="bolt://localhost:1",
                              production_uri="bolt://prod.lan:7687")
    await g2.copy_chain({**chain, "chapter": None})
    assert len(drv2.calls) == 1


async def test_verify_write_targets_refuses_a_driver_without_the_sentinel():
    good, bad = FakeDriver([[{"c": 1}]]), FakeDriver([[{"c": 0}]])
    with pytest.raises(proof.WriteTargetRefused, match="graphiti"):
        await proof.verify_write_targets({"ingest": good, "graphiti": bad}, "tok")
    assert good.calls[0]["params"] == {"t": "tok"}


async def test_verify_write_targets_accepts_all_sentinels():
    await proof.verify_write_targets({"ingest": FakeDriver([[{"c": 1}]])}, "tok")


def test_the_test_image_is_the_integration_tests_image():
    conftest = (_ROOT / "tests" / "integration" / "conftest.py").read_text()
    assert f'Neo4jContainer("{proof.NEO4J_TEST_IMAGE}")' in conftest


# --- gate verdict -------------------------------------------------------------------

def test_gate_never_fired_when_no_extraction_call_was_filtered():
    rows = [_outcome(prod_foreign=5, grounding_calls=0, grounding_dropped=0)]
    assert proof.gate_verdict(rows).startswith("gate never fired")


def test_gate_never_fired_when_nothing_dropped_on_articles_production_got_wrong():
    rows = [_outcome("a1", prod_foreign=5, grounding_calls=3, grounding_dropped=0),
            _outcome("a2", prod_foreign=0, grounding_calls=3, grounding_dropped=4)]
    verdict = proof.gate_verdict(rows)
    assert verdict.startswith("gate never fired") and "a1" in verdict


def test_gate_never_fired_when_no_article_was_wrong_in_production():
    rows = [_outcome(prod_foreign=0, grounding_calls=3, grounding_dropped=2)]
    assert proof.gate_verdict(rows).startswith("gate never fired")


def test_gate_fired_on_one_wrong_article_is_enough():
    rows = [_outcome("a1", prod_foreign=5, grounding_calls=3, grounding_dropped=0),
            _outcome("a2", prod_foreign=2, grounding_calls=2, grounding_dropped=1)]
    assert proof.gate_verdict(rows) is None


def test_finish_refuses_with_gate_never_fired_and_still_writes_the_report(tmp_path, capsys):
    out = tmp_path / "r.md"
    rows = [_outcome(prod_foreign=5, grounding_calls=0)]
    assert proof.finish(rows, out, meta={}) == 1
    assert "gate never fired" in capsys.readouterr().err
    assert "gate never fired" in out.read_text()


def test_finish_fails_on_article_errors_even_when_the_gate_fired(tmp_path, capsys):
    rows = [_outcome("a1", prod_foreign=5, grounding_calls=2, grounding_dropped=3),
            _outcome("a2", error="RuntimeError: boom")]
    assert proof.finish(rows, tmp_path / "r.md", meta={}) == 2
    assert "1 article(s) failed" in capsys.readouterr().err


def test_finish_succeeds(tmp_path):
    rows = [_outcome("a1", prod_foreign=5, grounding_calls=2, grounding_dropped=3)]
    assert proof.finish(rows, tmp_path / "r.md", meta={}) == 0


# --- report -------------------------------------------------------------------------

def test_report_formatting():
    rows = [
        _outcome("a1", title="Access | verification", prod_facts=37, test_facts=4,
                 prod_foreign=37, test_foreign=0, grounding_calls=1, grounding_dropped=21,
                 dropped_names=[f"n{i}" for i in range(30)],
                 prod_foreign_subjects={"AWS Backup": 37}, episodes_added=1, tier="cheap",
                 prompt_tokens=1000, completion_tokens=200),
        _outcome("a2", group="affected-other", product="HPE Morpheus", prod_facts=10,
                 test_facts=8, prod_foreign=0, test_foreign=1, grounding_calls=2,
                 grounding_dropped=0, test_foreign_facts=[("Azure Backup", "it backs up")],
                 restores=[("Veeam Agent", "HPE Morpheus VM Essentials")]),
    ]
    md = proof.render_report(rows, meta={"commit": "abc123", "image": "neo4j:x"})
    assert md.startswith("# Extraction grounding proof")
    assert "abc123" in md and "neo4j:x" in md
    assert "Access \\| verification" in md            # pipes escaped in table cells
    assert "| foreign-unnamed-subject facts | 37 | 1 |" in md
    assert "reduction 97.3%" in md
    # facts on articles production got right: a2 only, 10 -> 8
    assert "| facts on articles production got right | 10 | 8 |" in md
    assert "change -20.0%" in md
    assert "grounding_calls=3 grounding_dropped=21" in md
    assert "n19" in md and "n20" not in md            # dropped names capped at 20
    assert "AWS Backup x37" in md
    assert "- Azure Backup: it backs up" in md
    assert "- Veeam Agent -> HPE Morpheus VM Essentials" in md
    assert "Gate verdict: fired" in md


def test_report_shows_errors_and_the_refusal():
    rows = [_outcome("a1", prod_foreign=3, error="RuntimeError: boom")]
    md = proof.render_report(rows, meta={})
    assert "RuntimeError: boom" in md
    assert "Gate verdict: gate never fired" in md


# --- fix round 1: totals over ingested articles, early abort, cost, token cap -------

def test_totals_cover_successfully_ingested_articles_only():
    rows = [
        _outcome("ok1", prod_facts=10, test_facts=9, prod_foreign=4, test_foreign=0,
                 grounding_calls=2, grounding_dropped=3),
        _outcome("err", prod_facts=50, prod_foreign=40, error="RuntimeError: boom"),
        _outcome("nav", prod_facts=7, prod_foreign=7, excluded=proof.NAVIGATION_SKIP),
        _outcome("cap", prod_facts=5, prod_foreign=5, test_facts=1,
                 excluded=f"{proof.TOKEN_CAP_SKIP} before this article (9 of 1)"),
    ]
    md = proof.render_report(rows, meta={})
    assert "## Totals (over 1 of 4 articles: the successfully ingested ones)" in md
    assert "| facts | 10 | 9 | |" in md
    assert "| foreign-unnamed-subject facts | 4 | 0 | reduction 100.0% |" in md
    assert "included articles=1 " in md
    excluded = md.split("## Excluded from totals")[1].split("## Per article")[0]
    assert "`err` T err: error: RuntimeError: boom" in excluded
    assert f"`nav` T nav: {proof.NAVIGATION_SKIP}" in excluded
    assert "`cap` T cap: token cap reached" in excluded
    assert "`ok1`" not in excluded


def test_excluded_wrong_articles_do_not_satisfy_the_gate():
    rows = [_outcome("nav", prod_foreign=5, grounding_calls=1, grounding_dropped=2,
                     excluded=proof.NAVIGATION_SKIP)]
    assert proof.gate_verdict(rows).startswith("gate never fired")


def test_report_lists_no_exclusions_as_none():
    md = proof.render_report([_outcome(prod_foreign=1, grounding_calls=1,
                                       grounding_dropped=1)], meta={})
    assert md.split("## Excluded from totals")[1].split("## Per article")[0].strip() == "- none"


class FakeIngest:
    """ingest_article answers from a script: (result, prompt, cached, completion) or an
    exception; usage is recorded the way instrument() does, so tallies move."""

    def __init__(self, script):
        self.script = script
        self.called = []
        self._prov = SimpleNamespace()

    async def ingest_article(self, article_id):
        self.called.append(article_id)
        entry = self.script[article_id]
        if isinstance(entry, BaseException):
            raise entry
        res, prompt, cached, completion = entry
        usage.record("llm", prompt=prompt, completion=completion, cached=cached)
        return res


class FakeGraph:
    def __init__(self, facts=None, fail=()):
        self._facts = facts or {}
        self._fail = set(fail)

    async def facts(self, article_id, product_names):
        if article_id in self._fail:
            raise RuntimeError("read failed")
        return self._facts.get(article_id, [])


def _res(episodes=1, calls=1, dropped=0, tier="cheap", navigation=False):
    return SimpleNamespace(episodes_added=episodes, tier=tier, skipped_navigation=navigation,
                           grounding=GroundingStats(calls=calls, dropped=dropped,
                                                    dropped_names=["x"] * dropped))


def _never_spent():
    return 0


async def test_ingest_sample_stops_when_episodes_were_extracted_without_the_gate(
        tmp_path, capsys):
    ingest = FakeIngest({"a1": (_res(episodes=2, calls=0), 100, 0, 10),
                         "a2": (_res(), 100, 0, 10)})
    rows = [_outcome("a1", prod_foreign=3), _outcome("a2", prod_foreign=3)]
    abort = await proof.ingest_sample(ingest, FakeGraph(), rows, ["x"], token_cap=10**9,
                                      spent=_never_spent)
    assert abort.startswith("gate never fired") and "a1" in abort
    assert ingest.called == ["a1"]                       # nothing spent on a2
    assert rows[1].excluded == proof.GATE_ABORT_SKIP
    out = tmp_path / "r.md"
    assert proof.finish(rows, out, meta={}, abort=abort) == 1
    assert "gate never fired" in capsys.readouterr().err
    assert abort in out.read_text()


async def test_ingest_sample_does_not_abort_on_an_article_with_no_episodes():
    ingest = FakeIngest({"nav": (_res(episodes=0, calls=0, navigation=True), 0, 0, 0),
                         "a2": (_res(calls=1, dropped=1), 100, 0, 10)})
    rows = [_outcome("nav"), _outcome("a2", prod_foreign=2)]
    abort = await proof.ingest_sample(ingest, FakeGraph(), rows, ["x"], token_cap=10**9,
                                      spent=_never_spent)
    assert abort is None and ingest.called == ["nav", "a2"]
    assert rows[0].excluded == proof.NAVIGATION_SKIP
    assert proof.included(rows) == [rows[1]]


async def test_ingest_sample_records_a_failed_fact_read_and_carries_on(tmp_path):
    ingest = FakeIngest({"a1": (_res(calls=1, dropped=2), 100, 0, 10),
                         "a2": (_res(calls=1, dropped=1), 100, 0, 10)})
    rows = [_outcome("a1", prod_foreign=2), _outcome("a2", prod_foreign=2)]
    facts = {"a2": [{"uuid": "f", "subject": "S", "target": "T", "relation": "RestoresTo",
                     "fact": "f", "foreign_unnamed": False}]}
    await proof.ingest_sample(ingest, FakeGraph(facts, fail={"a1"}), rows, ["x"],
                              token_cap=10**9, spent=_never_spent)
    assert rows[0].error == "RuntimeError: read failed"
    assert rows[1].error is None and rows[1].restores == [("S", "T")]
    out = tmp_path / "r.md"
    assert proof.finish(rows, out, meta={}) == 2
    assert "read failed" in out.read_text()


async def test_ingest_sample_estimates_cost_with_cached_tokens():
    ingest = FakeIngest({"a1": (_res(tier="cheap"), 1_000_000, 500_000, 100_000),
                         "a2": (_res(tier="strong"), 1_000_000, 0, 0)})
    rows = [_outcome("a1"), _outcome("a2")]
    await proof.ingest_sample(ingest, FakeGraph(), rows, ["x"], token_cap=10**9,
                              spent=_never_spent)
    a1, a2 = rows
    assert (a1.prompt_tokens, a1.cached_tokens, a1.completion_tokens) == (
        1_000_000, 500_000, 100_000)
    assert a1.cost == pytest.approx(proof.article_cost("cheap", 1_000_000, 500_000, 100_000))
    assert a1.cost < proof.article_cost("cheap", 1_000_000, 0, 100_000)   # cached is cheaper
    assert a2.cost == pytest.approx(proof.RATES["strong"][0])
    md = proof.render_report(rows, meta={})
    assert f"estimated cost ${a1.cost + a2.cost:.3f}" in md
    assert "cached_tokens=500000" in md


def test_cost_of_an_unrouted_article_is_priced_at_the_strong_tier():
    assert proof.estimate_cost("", 1_000_000, 0, 0) == pytest.approx(proof.RATES["strong"][0])


async def test_episode_token_cap_raises_before_the_next_episode():
    calls = []

    async def already_ingested(*args, **kwargs):
        calls.append(args)
        return False

    ingest = SimpleNamespace(_prov=SimpleNamespace(already_ingested=already_ingested))
    spent = [0]
    proof.install_episode_token_cap(ingest, 100, lambda: spent[0])
    assert await ingest._prov.already_ingested("a", 0, "h") is False
    spent[0] = 100
    with pytest.raises(proof.TokenCapReached, match="100 of 100"):
        await ingest._prov.already_ingested("a", 1, "h")
    assert calls == [("a", 0, "h")]                      # the capped episode never ran


async def test_ingest_sample_token_cap_mid_article_and_before_the_next(tmp_path, capsys):
    ingest = FakeIngest({"a1": (_res(calls=1, dropped=1), 100, 0, 10),
                         "a2": proof.TokenCapReached("200 of 150 tokens spent"),
                         "a3": (_res(), 100, 0, 10)})
    rows = [_outcome("a1", prod_foreign=1), _outcome("a2"), _outcome("a3")]
    spent = [0]
    orig = ingest.ingest_article

    async def ingest_and_spend(article_id):
        try:
            return await orig(article_id)
        finally:
            spent[0] += 100

    ingest.ingest_article = ingest_and_spend
    await proof.ingest_sample(ingest, FakeGraph(), rows, ["x"], token_cap=150,
                              spent=lambda: spent[0])
    assert ingest.called == ["a1", "a2"]
    assert rows[1].excluded.startswith(proof.TOKEN_CAP_SKIP + " mid-article")
    assert rows[2].excluded.startswith(proof.TOKEN_CAP_SKIP + " before this article")
    assert proof.finish(rows, tmp_path / "r.md", meta={}) == 2
    assert "token cap" in capsys.readouterr().err


def test_token_cap_default_and_help():
    p = proof.build_parser()
    args = p.parse_args(["--out", "x.md"])
    assert args.token_cap == 1_500_000
    assert "at most one episode" in " ".join(p.format_help().split())


def test_removed_and_superseded_episodes_are_filtered_in_both_queries():
    for q in (proof.AFFECTED_ARTICLES_QUERY, proof.ARTICLE_FACTS_QUERY):
        assert "coalesce(e.removed, false) = false" in q
        assert "coalesce(he.superseded, false) = false" in q
