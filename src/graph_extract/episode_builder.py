from __future__ import annotations
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

def _merge_tiny(chunks: list[Chunk], min_tokens: int) -> list[Chunk]:
    out: list[Chunk] = []
    for c in chunks:
        if out and c.token_count < min_tokens:
            p = out[-1]
            out[-1] = Chunk(text=p.text + "\n" + c.text, start_index=p.start_index,
                            end_index=c.end_index, token_count=p.token_count + c.token_count)
        else:
            out.append(c)
    return out

def _split_oversize(c: Chunk, max_tokens: int) -> list[Chunk]:
    if c.token_count <= max_tokens:
        return [c]
    parts = (c.token_count + max_tokens - 1) // max_tokens
    span = max(1, len(c.text) // parts)
    pieces: list[Chunk] = []
    for i in range(parts):
        seg = c.text[i * span:(i + 1) * span] if i < parts - 1 else c.text[i * span:]
        if seg:
            pieces.append(Chunk(text=seg, start_index=0, end_index=0,
                                token_count=min(max_tokens, c.token_count - i * max_tokens)))
    return pieces

def build_episodes(*, article_id: str, title: str, chapter_path: str, content_hash: str,
                   chunks: list[Chunk], max_chunk_tokens: int, min_chunk_tokens: int) -> list[Episode]:
    merged = _merge_tiny(chunks, min_chunk_tokens)
    sized: list[Chunk] = []
    for c in merged:
        sized.extend(_split_oversize(c, max_chunk_tokens))
    prefix = f"[{title} › {chapter_path}]" if chapter_path else f"[{title}]"
    episodes: list[Episode] = []
    for idx, c in enumerate(sized):
        episodes.append(Episode(
            name=f"{article_id}:{idx}:{content_hash[:8]}",
            body=f"{prefix}\n{c.text}", chunk_index=idx, heading_path=chapter_path,
            token_count=c.token_count, content_hash=content_hash))
    return episodes
