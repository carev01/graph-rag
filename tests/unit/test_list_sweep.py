"""Keyword sweep (spec 2026-10-15 §12.1)."""
import logging

import pytest

import answer_api.list_mode as lm
import answer_api.list_sweep as ls
from answer_api.scope import _EPISODES_AMONG, Scope
from graph_extract.config import ExtractSettings


def _settings(**kw):
    return ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                           neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p", **kw)


@pytest.mark.parametrize("q,scope,terms", [
    ("Which workloads does Veeam support immutability for?", Scope(vendors=("Veeam",)),
     ["immutability"]),
    ("Where can I restore a Windows agent backup?", Scope(),
     ["restore", "windows", "agent", "backup"]),
    ("Which hypervisors does Veeam Backup & Replication support?",
     Scope(products=("Veeam Backup & Replication",)), []),
    ("Which resource types can AWS Backup protect?", Scope(products=("AWS Backup",)),
     ["resource", "protect"]),
    ("What recovery options does Azure Backup offer for Azure VMs?",
     Scope(vendors=("Microsoft",), products=("Azure Backup",)), ["recovery", "vms"]),
])
def test_question_terms_keep_only_distinctive_words(q, scope, terms):
    assert ls.question_terms(q, scope) == terms


@pytest.mark.parametrize("word,stemmed", [
    ("immutability", "immutab"), ("restore", "restor"), ("recovery", "recover"),
    ("databases", "databas"), ("restores", "restor"), ("agent", "agent"), ("vms", "vms"),
    ("copies", "copy"), ("policies", "policy"), ("ties", "ties")])
def test_stem(word, stemmed):
    assert ls.stem(word) == stemmed


@pytest.mark.parametrize("term,forms", [
    ("restore", ["restore", "restores", "restored", "restoring"]),
    ("agent", ["agent", "agents"]),
    ("windows", ["windows", "window"]),
    ("backup", ["backup", "backups"]),
    ("box", ["box", "boxes"]),
])
def test_word_forms(term, forms):
    assert ls.word_forms(term) == forms


def test_lucene_query_adds_stem_prefixes_and_escapes_hyphens():
    assert (ls.lucene_query(["restore", "windows", "agent", "vms", "hyper-v"])
            == "restore restores restored restoring restor* windows window window* "
               "agent agents agent* vms vmses hyper\\-v")



def test_lucene_query_finds_both_y_and_ies_forms():
    assert (ls.lucene_query(["copies", "copy"])
            == "copies copie copied copiing copy* copie* copy copys copy* copie*")


def test_is_limitation_lives_in_list_sweep_and_list_mode_reexports_it():
    assert lm.is_limitation is ls.is_limitation
    assert ls.is_limitation("X is not supported for Y.") and not ls.is_limitation("X works.")


class _Driver:
    # `cypher`, not `query`: the real driver's first positional parameter is `query_`
    # (trailing underscore) precisely so a Cypher `$query` parameter -- SWEEP_QUERY's
    # lucene string -- never collides with it; a fake named plain `query` would.
    def __init__(self, rows, in_scope=()):
        self.rows, self.in_scope, self.calls = rows, set(in_scope), []

    async def execute_query(self, cypher, **p):
        self.calls.append((cypher, p))
        if cypher is ls.SWEEP_QUERY:
            return (self.rows[: p["fetch"]], None, None)
        if cypher is _EPISODES_AMONG:
            return ([{"u": [e for e in p["eps"] if e in self.in_scope]}], None, None)
        raise AssertionError(cypher)


class _Prov:
    def __init__(self, driver):
        pass

    async def resolve_citations(self, uuids):
        return {u: {"sources": [{"url": f"https://d/{u}", "vendor": "Veeam"}],
                    "valid_at": None, "invalid_at": None} for u in uuids}


def _row(u, score, eps):
    return {"uuid": u, "fact": f"fact {u}", "episodes": eps, "score": score}


async def test_scoped_sweep_overfetches_filters_by_scope_and_caps(monkeypatch):
    monkeypatch.setattr(ls, "Provenance", _Prov)
    rows = [_row("a", 1.0, ["e1"]), _row("b", 3.0, ["e2"]), _row("c", 2.0, ["e3"]),
            _row("d", 0.5, ["e4"])]
    drv = _Driver(rows, in_scope={"e2", "e3", "e4"})
    got = await ls.keyword_sweep(drv, "Which workloads does Veeam support immutability for?",
                                 scope=Scope(vendors=("Veeam",)), group_id="g",
                                 settings=_settings(list_sweep_cap=2,
                                                    list_sweep_scoped_overfetch=4))
    query, p = drv.calls[0]
    assert p["index"] == "edge_name_and_fact" and p["fetch"] == 8 and p["group_id"] == "g"
    assert p["query"] == "immutability immutabilitys immutabed immutabing immutab*"
    assert [f["fact_uuid"] for f in got] == ["b", "c"]          # BM25 order, in scope, cap 2
    assert got[0]["sources"][0]["url"] == "https://d/b"


async def test_unscoped_sweep_fetches_the_cap_and_skips_the_scope_query(monkeypatch):
    monkeypatch.setattr(ls, "Provenance", _Prov)
    drv = _Driver([_row("a", 1.0, ["e1"]), _row("b", 2.0, [])])
    got = await ls.keyword_sweep(drv, "Where can I restore a Windows agent backup?",
                                 scope=None, group_id="g", settings=_settings())
    assert [c[0] for c in drv.calls] == [ls.SWEEP_QUERY]
    assert drv.calls[0][1]["fetch"] == 2000
    assert [f["fact_uuid"] for f in got] == ["b", "a"]


async def test_a_question_with_no_distinctive_words_runs_no_query():
    drv = _Driver([])
    got = await ls.keyword_sweep(drv, "Which hypervisors does Veeam Backup & Replication support?",
                                 scope=Scope(products=("Veeam Backup & Replication",)),
                                 group_id="g", settings=_settings())
    assert got == [] and drv.calls == []


async def test_the_sweep_logs_its_terms_and_counts(monkeypatch, caplog):
    monkeypatch.setattr(ls, "Provenance", _Prov)
    rows = [_row("a", 1.0, ["e1"]), _row("b", 3.0, ["e2"]), _row("c", 2.0, ["e3"])]
    drv = _Driver(rows, in_scope={"e2", "e3"})
    with caplog.at_level(logging.INFO, logger="answer_api.list_sweep"):
        await ls.keyword_sweep(drv, "Which workloads does Veeam support immutability for?",
                               scope=Scope(vendors=("Veeam",)), group_id="g",
                               settings=_settings(list_sweep_cap=1,
                                                  list_sweep_scoped_overfetch=4))
    assert ("list keyword sweep: terms=['immutability'] fetched=3 in_scope=2 kept=1"
            in caplog.text)
