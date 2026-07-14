import httpx
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

async def test_stream_records_malformed_lines_and_keeps_going():
    # A bad line is captured (not raised) so remaining records + the terminal
    # control line are still seen; the caller uses `.malformed` to block advance.
    body = (b'{"seq":1,"change_type":"added","id":"a1","topic_key":"t","source_id":"s",'
            b'"vendor":"V","product":"P","title":"T","source_url":"u","content_hash":"h",'
            b'"estimated_tokens":1,"sort_order":0}\n'
            b'{ this is not valid json\n'
            b'{"control":"cursor","next_since":"cur","count":1}')
    client = httpx.AsyncClient(transport=_transport(body), base_url="https://x")
    stream = DeltaStream(client, params={})
    recs = [r async for r in stream.records()]
    assert len(recs) == 1                     # the one good record still yielded
    assert stream.malformed == ["{ this is not valid json"]
    assert stream.terminated_clean is True    # terminal line still seen
    await client.aclose()


async def test_stream_truncated_has_no_next_since():
    lines = FIX.read_bytes().split(b"\n")
    truncated = b"\n".join(lines[:50])  # drop terminal control line
    client = httpx.AsyncClient(transport=_transport(truncated), base_url="https://x")
    stream = DeltaStream(client, params={})
    _ = [r async for r in stream.records()]
    assert stream.terminated_clean is False
    assert stream.next_since is None
