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


class _FakeCheapEmpty:
    """A HTTP 200 carrying no answer: content=None (a reasoning model spent the
    whole max_tokens=8 on reasoning) or an empty `choices` list."""
    def __init__(self, *, choices):
        self._choices = choices
        self.chat = self
        self.completions = self
    async def create(self, **kw):
        if not self._choices:
            return type("R", (), {"choices": []})()
        choice = type("m", (), {"message": type("mm", (), {"content": None})(),
                                "finish_reason": "length"})()
        return type("R", (), {"choices": [choice]})()


@pytest.mark.asyncio
async def test_empty_reply_defaults_and_is_logged(caplog):
    """BACKLOG 5: `content or ""` coerced an empty reply into a silent default.
    The default is still the right outcome, but it must be LOGGED as an empty
    reply -- a classifier that starts returning nothing was otherwise invisible
    (only `routing.via == "default"`, which nothing aggregates)."""
    mode, via = await classify("some question", cheap_client=_FakeCheapEmpty(choices=True),
                               cheap_model="ling")
    assert (mode, via) == ("drift", "default")
    assert any("no usable content" in r.getMessage() and r.levelname == "WARNING"
               for r in caplog.records)


@pytest.mark.asyncio
async def test_no_choices_defaults_without_a_traceback(caplog):
    """An empty `choices` list used to raise IndexError on `resp.choices[0]` and
    be caught as a generic 'classifier failed' with a traceback. It is the same
    empty-reply case and must take the same clean path."""
    mode, via = await classify("some question", cheap_client=_FakeCheapEmpty(choices=False),
                               cheap_model="ling")
    assert (mode, via) == ("drift", "default")
    assert any("no usable content" in r.getMessage() for r in caplog.records)
    assert not any(r.exc_info for r in caplog.records)


def test_cheap_client_none_without_key():
    s = ExtractSettings(_env_file=None, cheap_llm_api_key="", **_MIN)
    client, model = _cheap_classify_client(s)
    assert client is None and model == ""
