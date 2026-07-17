import pytest
from types import SimpleNamespace

pytestmark = pytest.mark.asyncio(loop_scope="module")


class _RecordingTier:
    def __init__(self, name):
        self.name, self.calls, self.instructions, self.max_chunk_tokens = name, [], f"instr-{name}", 1234
        self.graphiti = self

    async def add_episode(self, **kw):        # graphiti stand-in
        self.calls.append(kw["episode_body"])
        return SimpleNamespace(episode=SimpleNamespace(uuid=f"ep-{len(self.calls)}"),
                               nodes=[], edges=[])


async def _driver(extract_driver, monkeypatch, strong, cheap, dense):
    import graph_extract.ingest_driver as idmod
    from graph_extract.ingest_driver import IngestDriver
    from graph_extract.config import get_extract_settings

    # stub fetch + chunking + provenance so the test is deterministic and offline
    monkeypatch.setattr(idmod.content_fetch, "fetch_article", _fake_fetch)
    monkeypatch.setattr(idmod, "is_dense_matrix", lambda md, **k: dense)
    async def _chunk(ch, text, model): return [idmod.chonkie_client.Chunk(text="body", start_index=0, end_index=4, token_count=10)]
    monkeypatch.setattr(idmod.chonkie_client, "neural_chunk", _chunk)
    s = get_extract_settings.__wrapped__()
    prov = idmod.Provenance(extract_driver)
    return IngestDriver(s, strong, cheap, None, prov, extract_driver)


async def _fake_fetch(docext, aid):
    return SimpleNamespace(id=aid, title="T", content_markdown="md", source_url="u",
                           last_updated_at=None, extracted_at=None)


async def test_prose_routes_to_cheap(extract_driver, monkeypatch):
    strong, cheap = _RecordingTier("strong"), _RecordingTier("cheap")
    d = await _driver(extract_driver, monkeypatch, strong, cheap, dense=False)
    res = await d.ingest_article("a-prose")
    assert res.tier == "cheap"
    assert len(cheap.calls) == 1 and len(strong.calls) == 0


async def test_dense_routes_to_strong(extract_driver, monkeypatch):
    strong, cheap = _RecordingTier("strong"), _RecordingTier("cheap")
    d = await _driver(extract_driver, monkeypatch, strong, cheap, dense=True)
    res = await d.ingest_article("a-dense")
    assert res.tier == "strong"
    assert len(strong.calls) == 1 and len(cheap.calls) == 0


async def test_no_cheap_tier_falls_back_to_strong(extract_driver, monkeypatch):
    strong = _RecordingTier("strong")
    d = await _driver(extract_driver, monkeypatch, strong, None, dense=False)
    res = await d.ingest_article("a-prose")
    assert res.tier == "strong" and len(strong.calls) == 1
