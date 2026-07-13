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

def _merge_tiny(chunks: list[Chunk], min_tokens: int, max_tokens: int) -> list[Chunk]:
    out: list[Chunk] = []
    for c in chunks:
        if (out and c.token_count < min_tokens
                and out[-1].token_count + c.token_count <= max_tokens):
            p = out[-1]
            out[-1] = Chunk(text=p.text + "\n" + c.text, start_index=p.start_index,
                            end_index=c.end_index, token_count=p.token_count + c.token_count)
        else:
            out.append(c)
    return out

def _split_oversize(c: Chunk, max_tokens: int) -> list[Chunk]:
    if c.token_count <= max_tokens:
        return [c]
    if len(c.text) == 0:
        return [c]
    parts = math.ceil(c.token_count / max_tokens)
    text_len = len(c.text)
    # Degenerate guard: declared token_count wildly exceeds text length,
    # which would otherwise produce more parts than there are characters
    # (and thus empty-text pieces).
    parts = min(parts, max(1, text_len))
    # Equal-ish char boundaries covering the FULL text (last boundary == text_len).
    boundaries = [round(i * text_len / parts) for i in range(parts + 1)]
    boundaries[-1] = text_len
    pieces: list[Chunk] = []
    for i in range(parts):
        start, end = boundaries[i], boundaries[i + 1]
        seg = c.text[start:end]
        # Proportional-by-char-span estimate, clamped so rounding can never
        # push a piece's declared token_count over max_tokens (the ≤max
        # invariant is a hard constraint; a slight under-estimate is fine).
        token_count = min(round(c.token_count * len(seg) / text_len), max_tokens)
        pieces.append(Chunk(
            text=seg,
            start_index=c.start_index + start,
            end_index=c.start_index + end,
            token_count=token_count,
        ))
    return pieces

def build_episodes(*, article_id: str, title: str, chapter_path: str, content_hash: str,
                   chunks: list[Chunk], max_chunk_tokens: int, min_chunk_tokens: int) -> list[Episode]:
    # Split first (equal, content-preserving pieces), THEN merge tiny remainders.
    # This ordering avoids orphaning a sub-min piece created by splitting.
    split: list[Chunk] = []
    for c in chunks:
        split.extend(_split_oversize(c, max_chunk_tokens))
    sized = _merge_tiny(split, min_chunk_tokens, max_chunk_tokens)
    prefix = f"[{title} › {chapter_path}]" if chapter_path else f"[{title}]"
    episodes: list[Episode] = []
    for idx, c in enumerate(sized):
        episodes.append(Episode(
            name=f"{article_id}:{idx}:{content_hash[:8]}",
            body=f"{prefix}\n{c.text}", chunk_index=idx, heading_path=chapter_path,
            token_count=c.token_count, content_hash=content_hash))
    return episodes
