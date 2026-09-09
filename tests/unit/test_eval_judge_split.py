"""The faithfulness judge must be independent of synthesis.

Before this, `eval_router` resolved its judge via `_synthesis_client_and_model` --
so the model that wrote an answer also scored that answer's faithfulness. That
inflates the score in a way the score itself cannot reveal, which makes a
self-graded eval worse than no eval: it reads as evidence.

The guard therefore FAILS LOUDLY rather than falling back, mirroring the stance
`graph_extract.eval._judge_client_and_model` already takes against judging with
the extraction model.
"""
from __future__ import annotations

import pytest

from answer_api.eval_router import _eval_judge_client_and_model
from graph_extract.config import ExtractSettings


def _settings(**kw) -> ExtractSettings:
    base = dict(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                judge_base_url="https://synth.example/v1", judge_model="synth-model")
    base.update(kw)
    return ExtractSettings(**base)


def test_refuses_when_the_judge_would_be_the_synthesis_model():
    """The default fallback lands on the synthesis tier -- that must raise, not
    quietly proceed."""
    with pytest.raises(ValueError, match="grade its own answers"):
        _eval_judge_client_and_model(_settings())


def test_accepts_a_distinct_judge():
    client, model = _eval_judge_client_and_model(_settings(
        eval_judge_base_url="https://judge.example/v1",
        eval_judge_model="judge-model", eval_judge_api_key="k"))
    assert model == "judge-model"
    assert str(client.base_url).startswith("https://judge.example")


def test_same_endpoint_different_model_is_allowed():
    """Two models on one provider do not share a failure mode the way one model
    grading itself does, so this is permitted."""
    _, model = _eval_judge_client_and_model(_settings(
        eval_judge_base_url="https://synth.example/v1",
        eval_judge_model="a-different-model"))
    assert model == "a-different-model"


def test_same_model_on_a_different_endpoint_still_refuses():
    """Routing the identical model through another gateway does not make it an
    independent judge."""
    with pytest.raises(ValueError, match="grade its own answers"):
        _eval_judge_client_and_model(_settings(
            eval_judge_base_url="https://synth.example/v1",
            eval_judge_model="synth-model"))


def test_refuses_when_nothing_is_configured_at_all():
    with pytest.raises(ValueError, match="No eval judge configured"):
        _eval_judge_client_and_model(_settings(judge_base_url="", judge_model=""))
