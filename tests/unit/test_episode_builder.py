from graph_extract.chonkie_client import Chunk
from graph_extract.episode_builder import build_episodes, needs_presplit, Episode

def _mk(toks):  # chunks with given token counts, text length ~4 chars/token
    out, pos = [], 0
    for i, t in enumerate(toks):
        text = f"chunk{i} " + ("w " * t)
        out.append(Chunk(text=text, start_index=pos, end_index=pos + len(text), token_count=t))
        pos += len(text)
    return out

def test_prefix_and_indexing():
    eps = build_episodes(article_id="a1", title="Vault Lock",
                         chapter_path="Backup vaults", content_hash="abcd1234ef",
                         chunks=_mk([200, 300]), max_chunk_tokens=1800, min_chunk_tokens=50)
    assert [e.chunk_index for e in eps] == [0, 1]
    assert eps[0].body.startswith("[Vault Lock › Backup vaults]\n")
    assert eps[0].name == "a1:0:abcd1234"
    assert eps[0].heading_path == "Backup vaults"

def test_tiny_chunk_merges_into_previous():
    eps = build_episodes(article_id="a1", title="T", chapter_path="C",
                         content_hash="h", chunks=_mk([300, 20]),
                         max_chunk_tokens=1800, min_chunk_tokens=50)
    assert len(eps) == 1  # the 20-token chunk merged up

def test_oversize_chunk_splits():
    eps = build_episodes(article_id="a1", title="T", chapter_path="C",
                         content_hash="h", chunks=_mk([4000]),
                         max_chunk_tokens=1800, min_chunk_tokens=50)
    assert len(eps) >= 2
    assert all(e.token_count <= 1800 for e in eps)

def test_needs_presplit():
    assert needs_presplit(9000) is True
    assert needs_presplit(3000) is False
