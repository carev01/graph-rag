"""A degraded answer must tell the READER, not just the envelope."""
import re

from answer_api.global_search import _REFUSAL, _with_disclaimer


def test_adds_a_disclaimer_when_degraded():
    out = _with_disclaimer("Both vendors encrypt at rest [1].", "rerank-unavailable")
    assert "Both vendors encrypt at rest [1]." in out
    assert out != "Both vendors encrypt at rest [1]."
    assert "relevance ranking was unavailable" in out.lower()


def test_no_disclaimer_when_not_degraded():
    text = "Both vendors encrypt at rest [1]."
    assert _with_disclaimer(text, None) == text


def test_disclaimer_carries_no_citation_markers():
    """It must not read as cited content -- design decision #2."""
    out = _with_disclaimer("Answer [1].", "rerank-unavailable")
    added = out.replace("Answer [1].", "")
    assert re.search(r"\[\d+\]", added) is None
    assert "http" not in added.lower()


def test_a_refusal_is_never_disclaimed():
    """There is no answer to qualify."""
    assert _with_disclaimer(_REFUSAL, "rerank-unavailable") == _REFUSAL


def test_an_unknown_reason_adds_nothing():
    text = "Answer [1]."
    assert _with_disclaimer(text, "some-future-reason") == text


def test_empty_answer_is_not_disclaimed():
    assert _with_disclaimer("", "rerank-unavailable") == ""
