"""The reduce prompt's per-community header must not claim a relevance score
that was never computed. `MapResult.relevance` is `None` when the shortlist
was never reranked (the default deployment, since `rerank_base_url` defaults
to ""), and rendering "(relevance 0.0)" there would be false information
handed to the reducer -- the parenthetical must be omitted, not filled with a
placeholder. See finding 1."""
from __future__ import annotations

from answer_api import global_search as global_search_mod
from graph_extract.config import ExtractSettings

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


class _FakeSynthClient:
    """Records the reduce prompt so a test can inspect the rendered blocks."""

    def __init__(self, content):
        self._content = content
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    async def create(self, **kw):
        self.calls.append(kw)
        choice = type("C", (), {"message": type("M", (), {"content": self._content})()})()
        return type("R", (), {"choices": [choice]})()


class _EmptyRows:
    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        return
        yield  # pragma: no cover - makes this an async generator


class _FactRows:
    """One row per requested uuid: the reduce step now reads fact text (BACKLOG
    0b) and drops a fact it cannot read, so the fake must serve some."""

    def __init__(self, uuids):
        self._uuids = uuids

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for u in self._uuids:
            yield {"uuid": u, "fact": f"fact {u}"}


class _FakeProvenanceSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def run(self, cypher, **kw):
        if "f.fact AS fact" in cypher:
            return _FactRows(kw["u"])
        return _EmptyRows()


class _FakeProvenanceDriver:
    def session(self):
        return _FakeProvenanceSession()


def _hit(cid, title, relevance):
    return global_search_mod.CommunityHit(cid, title, "sum", 1, 7.0, ["f1"], "[]", 0.9,
                                          relevance)


async def _fake_shortlist(driver, embedder, q, *, level, k, group_id, rating_boost=0.1,
                          settings=None, stats=None):
    return [_hit("c1", "Unreranked", None), _hit("c2", "Reranked", 0.83)]


async def _fake_map_report(client, model, q, hit):
    return global_search_mod.MapResult(community_id=hit.community_id, title=hit.title,
                                       relevance=hit.relevance, key_points=["kp"],
                                       fact_ids=["f1"])


async def test_reduce_prompt_omits_relevance_when_unreranked_and_shows_it_when_reranked(
        monkeypatch):
    monkeypatch.setattr(global_search_mod, "shortlist_communities", _fake_shortlist)
    monkeypatch.setattr(global_search_mod, "map_report", _fake_map_report)
    synth_client = _FakeSynthClient("a synthesized answer [1].")
    await global_search_mod.global_search(
        _FakeProvenanceDriver(), None, object(), "map-model", synth_client, "synth-model",
        q="q", level=1, k=3, group_id="g", settings=_S)
    prompt = synth_client.calls[0]["messages"][0]["content"]
    unreranked_block = prompt.split('COMMUNITY "Reranked"')[0]
    assert 'COMMUNITY "Unreranked":' in unreranked_block
    assert "relevance" not in unreranked_block.lower()
    assert 'COMMUNITY "Reranked" (relevance 0.83):' in prompt
