"""`load_graph(..., article_ids=, with_embeddings=)` at production scale.

Production is ~240k RELATES_TO edges with 768-d embeddings on every one --
loading the whole graph (let alone its vectors) is infeasible for the
distillation capture's ~100-article sample. `load_graph` must therefore be
able to load ONLY those articles' episodes, the entities those episodes
MENTION, and the facts on those episodes, and to skip the embedding fields
entirely when asked -- while the unscoped (`article_ids=None`) default stays
byte-for-byte the old pilot-workflow query.

No Neo4j: a fake driver/session records every query + its parameters and
answers from an in-memory fixture, so the "returns only their episodes" claim
is actually exercised (not just asserted) by filtering on the parameters the
production queries pass, exactly as a real session would.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import build_extraction_dataset as bed
from scripts.build_extraction_dataset import (
    CaptureAttributor,
    Episode,
    Graph,
    PromptRenderer,
    build_from_capture,
    load_graph,
)


# --------------------------------------------------------------------------- #
# fake Neo4j driver/session
# --------------------------------------------------------------------------- #
class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __aiter__(self):
        async def _gen():
            for r in self._rows:
                yield r
        return _gen()


class _FakeSession:
    def __init__(self, fixture: "_Fixture", calls: list[tuple[str, dict]]) -> None:
        self._fixture = fixture
        self._calls = calls

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def run(self, query: str, **params: Any) -> _FakeResult:
        self._calls.append((query, dict(params)))
        return _FakeResult(self._fixture.answer(query, params))


class _FakeDriver:
    def __init__(self, fixture: "_Fixture", calls: list[tuple[str, dict]]) -> None:
        self._fixture = fixture
        self._calls = calls
        self.closed = False

    def session(self, **_kw: Any) -> _FakeSession:
        return _FakeSession(self._fixture, self._calls)

    async def close(self) -> None:
        self.closed = True


class _Fixture:
    """An in-memory stand-in for the live graph, filtering like Neo4j would.

    Dispatch is by the `//`-tagged comment each real query carries (Cypher
    ignores `//` comments, so this changes nothing on the wire) -- never by
    guessing from unrelated substrings, so a query that silently stopped using
    `$article_ids`/`$episode_uuids` would show up here as "wrong rows", not
    pass by accident.
    """

    def __init__(self, episodes: list[dict], entities: list[dict],
                facts: list[dict], mentions: list[dict],
                mentioned_by: dict[str, set[str]]) -> None:
        self.episodes = episodes
        self.entities = entities
        self.facts = facts
        self.mentions = mentions
        # entity uuid -> set of episode uuids that MENTION it (for the scoped
        # entities query, which joins through MENTIONS).
        self.mentioned_by = mentioned_by

    def answer(self, query: str, params: dict) -> list[dict]:
        if "// entities_scoped" in query or "// entities" in query:
            rows = list(self.entities)
            if "// entities_scoped" in query:
                eps = set(params["episode_uuids"])
                rows = [e for e in rows if self.mentioned_by.get(e["uuid"], set()) & eps]
            return self._strip_embedding(query, rows)
        if "// facts_scoped" in query or "// facts" in query:
            rows = list(self.facts)
            if "// facts_scoped" in query:
                eps = set(params["episode_uuids"])
                rows = [f for f in rows if set(f["episodes"]) & eps]
            return self._strip_embedding(query, rows)
        if "// episodes_scoped" in query:
            wanted = set(params["article_ids"])
            return [e for e in self.episodes if e["article_id"] in wanted]
        if "// episodes" in query:
            return list(self.episodes)
        if "// mentions_scoped" in query:
            eps = set(params["episode_uuids"])
            return [m for m in self.mentions if m["ep"] in eps]
        if "// mentions" in query:
            return list(self.mentions)
        raise AssertionError(f"unrecognised query: {query!r}")

    @staticmethod
    def _strip_embedding(query: str, rows: list[dict]) -> list[dict]:
        """Mirrors `load_graph` replacing the embedding column with `null` in
        the query text: when that substitution happened, a real session would
        come back with `emb=None` on every row, not the stored vector."""
        if "name_embedding" in query or "fact_embedding" in query:
            return rows
        return [{**r, "emb": None} for r in rows]


def _row(uuid: str, article_id: str) -> dict:
    return {"uuid": uuid, "name": f"ep-{uuid}", "content": f"content of {uuid}",
           "valid_at": "2024-01-01T00:00:00+00:00", "created_at": "2024-01-01T00:00:00+00:00",
           "sd": "src", "article_id": article_id, "chunk_index": 0}


def _entity(uuid: str, emb: list[float] | None = None) -> dict:
    return {"uuid": uuid, "name": f"ent-{uuid}", "labels": ["Entity"], "summary": "",
           "created_at": "2024-01-01T00:00:00+00:00", "emb": emb or [0.1, 0.2]}


def _fact(uuid: str, episodes: list[str], emb: list[float] | None = None) -> dict:
    return {"uuid": uuid, "name": "RELATES_TO", "fact": f"fact {uuid}",
           "src": "e1", "tgt": "e2", "episodes": episodes,
           "created_at": "2024-01-01T00:00:00+00:00", "valid_at": None, "invalid_at": None,
           "emb": emb or [0.3, 0.4]}


def _three_article_fixture() -> _Fixture:
    """3 articles' worth of episodes/entities/facts, only 2 of which are asked for."""
    episodes = [_row("ep-a1", "art-1"), _row("ep-a2", "art-2"), _row("ep-a3", "art-3")]
    entities = [_entity("ent-1"), _entity("ent-2"), _entity("ent-3")]
    facts = [_fact("f1", ["ep-a1"]), _fact("f2", ["ep-a2"]), _fact("f3", ["ep-a3"])]
    mentions = [{"ep": "ep-a1", "ent": "ent-1"}, {"ep": "ep-a2", "ent": "ent-2"},
               {"ep": "ep-a3", "ent": "ent-3"}]
    mentioned_by = {"ent-1": {"ep-a1"}, "ent-2": {"ep-a2"}, "ent-3": {"ep-a3"}}
    return _Fixture(episodes, entities, facts, mentions, mentioned_by)


async def _run_load(monkeypatch: pytest.MonkeyPatch, fixture: _Fixture, *,
                    article_ids: list[str] | None,
                    with_embeddings: bool = True) -> tuple[Graph, list[tuple[str, dict]]]:
    calls: list[tuple[str, dict]] = []
    driver = _FakeDriver(fixture, calls)

    class _FakeGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple) -> _FakeDriver:
            return driver

    monkeypatch.setattr(bed, "AsyncGraphDatabase", _FakeGraphDatabase)
    graph = await load_graph("bolt://x", "u", "p", "g", with_embeddings=with_embeddings,
                             article_ids=article_ids)
    assert driver.closed, "load_graph must close the driver even on the scoped path"
    return graph, calls


# --------------------------------------------------------------------------- #
# scoped load
# --------------------------------------------------------------------------- #
async def test_scoped_load_is_parameterised_by_article_ids_and_returns_only_those_episodes(
        monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _three_article_fixture()
    graph, calls = await _run_load(monkeypatch, fixture, article_ids=["art-1", "art-2"])

    assert set(graph.episodes) == {"ep-a1", "ep-a2"}
    assert graph.episodes["ep-a1"].article_id == "art-1"
    assert graph.episodes["ep-a2"].article_id == "art-2"

    # entities/facts/mentions follow the scoped episode set, not the whole graph.
    assert set(graph.entities) == {"ent-1", "ent-2"}
    assert set(graph.facts) == {"f1", "f2"}
    assert sorted(graph.mentions["ep-a1"]) == ["ent-1"]
    assert sorted(graph.mentions["ep-a2"]) == ["ent-2"]
    assert "ep-a3" not in graph.mentions

    episodes_call = next(c for c in calls if "// episodes_scoped" in c[0])
    assert episodes_call[1]["article_ids"] == ["art-1", "art-2"]
    # every later query is scoped by the episode uuids the first query found,
    # not by the article ids directly.
    for query, params in calls[1:]:
        assert sorted(params["episode_uuids"]) == ["ep-a1", "ep-a2"]


async def test_scoped_load_with_no_matching_articles_returns_an_empty_graph_and_stops_early(
        monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _three_article_fixture()
    graph, calls = await _run_load(monkeypatch, fixture, article_ids=["no-such-article"])

    assert graph.episodes == {}
    assert graph.entities == {}
    assert graph.facts == {}
    assert graph.mentions == {}
    # only the episodes query ran: no point issuing three more with an empty
    # episode_uuids scope.
    assert len(calls) == 1


async def test_unscoped_load_is_unchanged_and_ignores_article_ids(
        monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _three_article_fixture()
    graph, calls = await _run_load(monkeypatch, fixture, article_ids=None)

    assert set(graph.episodes) == {"ep-a1", "ep-a2", "ep-a3"}
    assert set(graph.entities) == {"ent-1", "ent-2", "ent-3"}
    assert set(graph.facts) == {"f1", "f2", "f3"}
    assert len(calls) == 4
    for query, _params in calls:
        assert "_scoped" not in query


# --------------------------------------------------------------------------- #
# with_embeddings=False
# --------------------------------------------------------------------------- #
async def test_with_embeddings_false_leaves_embeddings_empty_scoped_and_unscoped(
        monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _three_article_fixture()

    scoped_graph, _ = await _run_load(monkeypatch, fixture, article_ids=["art-1"],
                                      with_embeddings=False)
    assert scoped_graph.entities["ent-1"].embedding == []
    assert scoped_graph.facts["f1"].embedding == []

    unscoped_graph, _ = await _run_load(monkeypatch, fixture, article_ids=None,
                                        with_embeddings=False)
    assert all(e.embedding == [] for e in unscoped_graph.entities.values())
    assert all(f.embedding == [] for f in unscoped_graph.facts.values())


async def test_with_embeddings_true_keeps_the_stored_vectors(
        monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _three_article_fixture()
    graph, _ = await _run_load(monkeypatch, fixture, article_ids=["art-1"],
                               with_embeddings=True)
    assert graph.entities["ent-1"].embedding == [0.1, 0.2]
    assert graph.facts["f1"].embedding == [0.3, 0.4]


# --------------------------------------------------------------------------- #
# --articles-from parsing (sample.json schema, and the bare-list fallback)
# --------------------------------------------------------------------------- #
def test_load_article_ids_accepts_the_sample_json_object_schema(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    path.write_text(json.dumps({"articles": [
        {"article_id": "art-1", "vendor": "Veeam", "product": "VBR",
         "source": "src", "split": "train", "stratum": "veeam-vbr"},
        {"article_id": "art-2", "vendor": "AWS", "product": "S3",
         "source": "src2", "split": "val", "stratum": "aws-s3"},
    ]}), encoding="utf-8")
    assert bed._load_article_ids(path) == ["art-1", "art-2"]


def test_load_article_ids_accepts_a_bare_list_of_ids(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    path.write_text(json.dumps(["art-1", "art-2"]), encoding="utf-8")
    assert bed._load_article_ids(path) == ["art-1", "art-2"]


def test_load_article_ids_rejects_an_object_with_no_articles_key(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"not_articles": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="articles"):
        bed._load_article_ids(path)


# --------------------------------------------------------------------------- #
# build_from_capture over a scoped graph (no embeddings, no SimIndex)
# --------------------------------------------------------------------------- #
def _episode(uuid: str, content: str, article_id: str) -> Episode:
    return Episode(uuid=uuid, name=f"ep-{uuid}", content=content,
                   valid_at="2024-01-01T00:00:00+00:00",
                   created_at="2024-01-01T00:00:00+00:00",
                   source_description="src", article_id=article_id, chunk_index=0)


def test_build_from_capture_attributes_a_record_to_the_right_episode_on_a_scoped_graph(
        tmp_path: Path) -> None:
    # A scoped Graph as `load_graph(article_ids=[...])` would return it: no
    # Entity/Fact rows at all (this synthetic capture has none to attribute
    # through), which is exactly what the capture path needs -- it never
    # touches `ent_sim`/`fact_sim` (those belong to `Builder`, the
    # graph-derived path, and are never constructed here).
    graph = Graph()
    graph.episodes["ep1"] = _episode("ep1", "hello world content one", "art-1")
    graph.episodes["ep2"] = _episode("ep2", "a different second episode", "art-2")

    renderer = PromptRenderer(group_id="backup-docs")
    user = ("<ENTITY TYPES>\n[]\n</ENTITY TYPES>\n"
           "<TEXT>\nhello world content one\n</TEXT>")
    rec = {
        "prompt_name": "extract_nodes.extract_text",
        "tier": "cheap",
        "finish_reason": "stop",
        "call_id": "c1",
        "model": "m",
        "messages": [{"role": "system", "content": "sys"},
                     {"role": "user", "content": user}],
        "completion": json.dumps({"extracted_entities": []}),
    }
    path = tmp_path / "capture.jsonl"
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    records, report = build_from_capture(
        path, graph, renderer, wanted={"extract_nodes.extract_text"}, tiers={"cheap"},
        allow_overlong_summaries=False, raw_targets=False, repair=True)

    assert len(records) == 1
    out = records[0]
    assert out["meta"]["article_id"] == "art-1"
    assert out["meta"]["episode_uuid"] == "ep1"
    assert out["meta"]["chunk_index"] == 0
    assert report["stats"]["extract_nodes.extract_text"]["kept"] == 1


def test_capture_attributor_never_constructs_a_simindex(tmp_path: Path) -> None:
    """`CaptureAttributor`/`build_from_capture` need no embeddings at all -- a
    `Graph` with zero entities/facts (as a scoped, with_embeddings=False load
    for a capture-only run would produce) is enough to attribute."""
    graph = Graph()
    graph.episodes["ep1"] = _episode("ep1", "content for attribution", "art-1")
    renderer = PromptRenderer(group_id="backup-docs")
    attributor = CaptureAttributor(graph, renderer)
    attribution, reason = attributor.attribute(
        "extract_nodes.extract_text",
        "<TEXT>\ncontent for attribution\n</TEXT>")
    assert reason == ""
    assert attribution is not None
    assert attribution.article_id == "art-1"
    assert attribution.episode_uuid == "ep1"
