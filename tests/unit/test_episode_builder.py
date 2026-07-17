from graph_extract.chonkie_client import Chunk
from graph_extract.episode_builder import build_episodes, needs_presplit, _split_oversize


def test_split_cuts_on_line_boundaries():
    # 4 paragraphs, each ~600 "tokens"; max 1000 -> must cut BETWEEN lines,
    # never mid-line, so every piece but the last ends at a newline.
    paras = [f"para{i} " + ("w " * 600) for i in range(4)]
    text = "\n".join(paras)
    chunk = Chunk(text=text, start_index=0, end_index=len(text), token_count=2400)
    pieces = _split_oversize(chunk, 1000)
    assert "".join(p.text for p in pieces) == text          # content-preserving
    assert len(pieces) >= 2
    for p in pieces[:-1]:
        assert p.text.endswith("\n")                        # cut landed on a line break
    assert all(p.token_count <= 1000 for p in pieces)


def test_split_table_rows_stay_intact():
    # a markdown availability table must split BETWEEN rows, not mid-cell.
    header = "| Feature | Region |\n|---|---|\n"
    rows = "".join(f"| feature{i} | us-east-{i} |\n" for i in range(60))
    text = header + rows
    chunk = Chunk(text=text, start_index=0, end_index=len(text), token_count=2400)
    pieces = _split_oversize(chunk, 800)
    assert "".join(p.text for p in pieces) == text
    assert len(pieces) >= 2
    for p in pieces:
        for line in p.text.splitlines():
            assert line.startswith("|")                     # only whole table rows


def test_split_giant_single_line_falls_back_to_char():
    # one line, no breaks, far over max -> equal-char fallback still splits it.
    text = "x" * 8000
    chunk = Chunk(text=text, start_index=0, end_index=8000, token_count=4000)
    pieces = _split_oversize(chunk, 1800)
    assert len(pieces) >= 2
    assert "".join(p.text for p in pieces) == text
    assert all(p.token_count <= 1800 for p in pieces)

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

def test_no_orphan_after_split():
    # A chunk just over max_chunk_tokens splits into two roughly-equal pieces;
    # neither piece should be a tiny orphan left unmerged after splitting.
    eps = build_episodes(article_id="a1", title="T", chapter_path="C",
                         content_hash="h", chunks=_mk([1830]),
                         max_chunk_tokens=1800, min_chunk_tokens=50)
    assert all(e.token_count >= 50 for e in eps)
    assert not any(e.token_count < 50 for e in eps)

def test_split_preserves_full_text():
    chunk = _mk([4000])[0]
    pieces = _split_oversize(chunk, 1800)
    assert len(pieces) >= 2
    assert "".join(p.text for p in pieces) == chunk.text

def test_split_token_count_proportional_to_span():
    chunk = _mk([4000])[0]
    pieces = _split_oversize(chunk, 1800)
    assert len(pieces) == 3
    expected = 4000 / 3
    for p in pieces:
        assert abs(p.token_count - expected) <= expected * 0.15
    assert abs(sum(p.token_count for p in pieces) - 4000) <= 3

def test_merge_respects_max_ceiling():
    # A near-max chunk followed by a tiny chunk: merging would exceed max,
    # so the tiny chunk must stay standalone rather than push the total over.
    eps = build_episodes(article_id="a1", title="T", chapter_path="C",
                         content_hash="h", chunks=_mk([1790, 20]),
                         max_chunk_tokens=1800, min_chunk_tokens=50)
    assert len(eps) == 2
    assert all(e.token_count <= 1800 for e in eps)
    assert eps[0].token_count == 1790

def test_split_high_density_never_exceeds_max():
    # Dense chunks (more tokens per char than the 2-char/token _mk() helper
    # produces) can push proportional-rounding token_count slightly OVER
    # max_tokens for a piece. Every resulting episode must still be <= max.
    cases = [
        (2301, 3600, 1800),
        (5000, 9000, 1800),
        (777, 3333, 500),
        (1, 4000, 1800),  # degenerate: declared tokens far exceed text length
    ]
    for text_len, token_count, max_tokens in cases:
        chunk = Chunk(text="x" * text_len, start_index=0, end_index=text_len,
                     token_count=token_count)
        eps = build_episodes(article_id="a1", title="T", chapter_path="C",
                             content_hash="h", chunks=[chunk],
                             max_chunk_tokens=max_tokens, min_chunk_tokens=50)
        assert all(e.token_count <= max_tokens for e in eps), (
            text_len, token_count, max_tokens, [e.token_count for e in eps])
