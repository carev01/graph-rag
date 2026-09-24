from __future__ import annotations
import math
from dataclasses import dataclass
from graph_extract.chonkie_client import Chunk

@dataclass
class Episode:
    name: str
    body: str
    chunk_index: int
    heading_path: str
    token_count: int
    content_hash: str

def needs_presplit(token_count_total: int, ceiling: int = 7000) -> bool:
    return token_count_total > ceiling

def _join(p: Chunk, c: Chunk) -> Chunk:
    return Chunk(text=p.text + "\n" + c.text, start_index=p.start_index,
                 end_index=c.end_index, token_count=p.token_count + c.token_count)

def _merge_tiny(chunks: list[Chunk], min_tokens: int, max_tokens: int) -> list[Chunk]:
    out: list[Chunk] = []
    for c in chunks:
        if (out and c.token_count < min_tokens
                and out[-1].token_count + c.token_count <= max_tokens):
            out[-1] = _join(out[-1], c)
        else:
            out.append(c)
    return out

def _pack(chunks: list[Chunk], limit: int) -> list[Chunk]:
    """Greedily merge CONSECUTIVE chunks while the total stays <= limit (D6).
    ~70% of an episode's prompt volume is fixed per-episode overhead, so fewer,
    larger episodes cost less -- but extraction yield per call is roughly constant,
    so they also yield fewer facts: measured -36% cost, -28% facts at 1,200 tokens
    (docs/superpowers/chunk-packing-ab-2026-09-24.md). Off by default for that reason."""
    out: list[Chunk] = []
    for c in chunks:
        if out and out[-1].token_count + c.token_count <= limit:
            out[-1] = _join(out[-1], c)
        else:
            out.append(c)
    return out

def _equal_char_segments(text: str, max_tokens: int, tok_per_char: float) -> list[str]:
    """Equal-character split of a single oversized run of text — the fallback
    used only when there is no line boundary to cut on (e.g. one giant line).
    Content-preserving: the segments concatenate back to `text`."""
    n = len(text)
    if n == 0:
        return [text]
    parts = math.ceil((tok_per_char * n) / max_tokens)
    # Degenerate guard: declared tokens wildly exceed char length, which would
    # otherwise produce more parts than there are characters (empty pieces).
    parts = min(parts, max(1, n))
    bounds = [round(i * n / parts) for i in range(parts + 1)]
    bounds[-1] = n
    return [text[bounds[i]:bounds[i + 1]] for i in range(parts)]


def _split_oversize(c: Chunk, max_tokens: int) -> list[Chunk]:
    """Split a chunk exceeding max_tokens into pieces that each fit.

    Cuts on LINE boundaries (paragraph / markdown table-row breaks) so document
    structure survives — a dense availability table splits BETWEEN rows, not
    mid-cell, which keeps the model from seeing half a row. Only a single line
    that alone exceeds max_tokens falls back to an equal-character split. The
    concatenation of the piece texts always equals the original text
    (content-preserving), and every piece's estimated token_count stays <= max.
    """
    if c.token_count <= max_tokens or len(c.text) == 0:
        return [c]
    text_len = len(c.text)
    tok_per_char = c.token_count / text_len

    # 1) atomic segments = lines (newlines kept); char-split any single line
    #    that is itself too big. Concatenation of segments == original text.
    segments: list[str] = []
    for line in c.text.splitlines(keepends=True):
        if tok_per_char * len(line) <= max_tokens:
            segments.append(line)
        else:
            segments.extend(_equal_char_segments(line, max_tokens, tok_per_char))

    # 2) greedily pack consecutive segments into pieces up to max_tokens, so a
    #    cut only ever lands at a segment (line) boundary.
    pieces_text: list[str] = []
    cur = ""
    cur_tok = 0.0
    for seg in segments:
        seg_tok = tok_per_char * len(seg)
        if cur and cur_tok + seg_tok > max_tokens:
            pieces_text.append(cur)
            cur, cur_tok = seg, seg_tok
        else:
            cur += seg
            cur_tok += seg_tok
    if cur:
        pieces_text.append(cur)

    # 3) build Chunks with span-proportional token_count, clamped <= max.
    pieces: list[Chunk] = []
    pos = 0
    for seg in pieces_text:
        token_count = min(round(tok_per_char * len(seg)), max_tokens)
        pieces.append(Chunk(
            text=seg,
            start_index=c.start_index + pos,
            end_index=c.start_index + pos + len(seg),
            token_count=token_count,
        ))
        pos += len(seg)
    return pieces

def build_episodes(*, article_id: str, title: str, chapter_path: str, content_hash: str,
                   chunks: list[Chunk], max_chunk_tokens: int, min_chunk_tokens: int, pack_target_tokens: int = 0) -> list[Episode]:
    # Split first (equal, content-preserving pieces), THEN merge tiny remainders.
    # This ordering avoids orphaning a sub-min piece created by splitting.
    split: list[Chunk] = []
    for c in chunks:
        split.extend(_split_oversize(c, max_chunk_tokens))
    sized = _merge_tiny(split, min_chunk_tokens, max_chunk_tokens)
    if pack_target_tokens > 0:
        sized = _pack(sized, min(pack_target_tokens, max_chunk_tokens))
    prefix = f"[{title} › {chapter_path}]" if chapter_path else f"[{title}]"
    episodes: list[Episode] = []
    for idx, c in enumerate(sized):
        episodes.append(Episode(
            name=f"{article_id}:{idx}:{content_hash[:8]}",
            body=f"{prefix}\n{c.text}", chunk_index=idx, heading_path=chapter_path,
            token_count=c.token_count, content_hash=content_hash))
    return episodes
