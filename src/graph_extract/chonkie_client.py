from __future__ import annotations
from dataclasses import dataclass
import httpx

@dataclass
class Chunk:
    text: str
    start_index: int
    end_index: int
    token_count: int

async def neural_chunk(client: httpx.AsyncClient, text: str, model: str) -> list[Chunk]:
    resp = await client.post("/v1/chunk/neural", json={"text": text, "model": model})
    resp.raise_for_status()
    return [
        Chunk(text=c["text"], start_index=c.get("start_index", 0),
              end_index=c.get("end_index", 0), token_count=c.get("token_count", 0))
        for c in resp.json().get("chunks", [])
    ]
