import httpx, json, pytest
from graph_extract.chonkie_client import neural_chunk, Chunk

pytestmark = pytest.mark.asyncio

async def test_neural_chunk_parses():
    body = json.dumps({"chunks": [
        {"text": "# A\n\nx", "start_index": 0, "end_index": 6, "token_count": 4},
        {"text": "## B\n\ny", "start_index": 6, "end_index": 13, "token_count": 5},
    ]}).encode()
    def handler(req):
        assert req.url.path == "/v1/chunk/neural"
        return httpx.Response(200, content=body)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://c")
    chunks = await neural_chunk(client, "# A\n\nx## B\n\ny", "m")
    assert [c.token_count for c in chunks] == [4, 5]
    assert isinstance(chunks[0], Chunk) and chunks[0].start_index == 0
