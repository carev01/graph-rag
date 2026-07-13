import httpx, pytest
from pathlib import Path
from graph_sync.delta_client import DeltaStream, build_delta_params
from graph_sync.models import ContentRecord

FIX = Path(__file__).parent.parent / "fixtures" / "aws_delta.ndjson"

def test_build_params_omits_none():
    assert build_delta_params(since="cur") == {"since": "cur"}
    assert build_delta_params(source_id="s") == {"source_id": "s"}
    assert build_delta_params(bootstrap_after="x") == {"bootstrap_after": "x"}
    assert build_delta_params() == {}

def _transport(body: bytes) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)
    return httpx.MockTransport(handler)

async def test_stream_applies_cursor_discipline_on_clean_terminal():
    body = FIX.read_bytes()
    client = httpx.AsyncClient(transport=_transport(body), base_url="https://x")
    stream = DeltaStream(client, params={"source_id": "s"})
    recs = [r async for r in stream.records()]
    assert all(isinstance(r, ContentRecord) for r in recs)
    assert len(recs) == 146
    assert stream.terminated_clean is True
    assert stream.next_since is not None
    assert stream.bootstrap_start_since is not None

async def test_stream_truncated_has_no_next_since():
    lines = FIX.read_bytes().split(b"\n")
    truncated = b"\n".join(lines[:50])  # drop terminal control line
    client = httpx.AsyncClient(transport=_transport(truncated), base_url="https://x")
    stream = DeltaStream(client, params={})
    _ = [r async for r in stream.records()]
    assert stream.terminated_clean is False
    assert stream.next_since is None
