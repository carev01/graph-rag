import pytest
from answer_api.router import classify, _cheap_classify_client
from graph_extract.config import ExtractSettings

_MIN = dict(docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x",
            neo4j_user="u", neo4j_password="p")


class _FakeCheap:
    def __init__(self, label):
        self._label = label
        self.chat = self
        self.completions = self
    async def create(self, **kw):
        return type("R", (), {"choices": [type("m", (), {"message": type("mm", (), {"content": self._label})()})()]})


@pytest.mark.asyncio
async def test_override_wins():
    mode, via = await classify("anything", cheap_client=None, cheap_model="",
                               mode_override="global")
    assert (mode, via) == ("global", "override")


@pytest.mark.asyncio
async def test_temporal_heuristic():
    mode, via = await classify("How has Veeam immutability changed over time?",
                               cheap_client=None, cheap_model="")
    assert (mode, via) == ("timeline", "heuristic")


@pytest.mark.asyncio
async def test_cross_vendor_heuristic():
    mode, via = await classify("Compare all vendors on ransomware protection",
                               cheap_client=None, cheap_model="")
    assert (mode, via) == ("global", "heuristic")


@pytest.mark.asyncio
async def test_llm_path_for_uncaught_query():
    mode, via = await classify("Does Veeam support S3 Object Lock?",
                               cheap_client=_FakeCheap("local"), cheap_model="ling")
    assert (mode, via) == ("local", "llm")


@pytest.mark.asyncio
async def test_no_cheap_client_defaults_to_drift():
    mode, via = await classify("What should I consider for retention?",
                               cheap_client=None, cheap_model="")
    assert (mode, via) == ("drift", "default")


@pytest.mark.asyncio
async def test_unknown_llm_label_defaults():
    mode, via = await classify("some question", cheap_client=_FakeCheap("banana"),
                               cheap_model="ling")
    assert (mode, via) == ("drift", "default")


def test_cheap_client_none_without_key():
    s = ExtractSettings(_env_file=None, cheap_llm_api_key="", **_MIN)
    client, model = _cheap_classify_client(s)
    assert client is None and model == ""
