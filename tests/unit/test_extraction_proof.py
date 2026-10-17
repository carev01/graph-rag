"""Hermetic tests for scripts/extraction_proof.py: the deterministic affected-article
selection, the write-target guard, the "gate never fired" refusal and the report."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from neo4j import RoutingControl

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
