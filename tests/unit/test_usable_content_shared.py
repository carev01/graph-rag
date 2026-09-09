"""The empty-reply guard is shared by answer_api and theme_builder, so it lives in
graph_extract.usage. Four separate defects in this codebase came from an LLM reply
carrying no content being coerced into a legitimate value."""
from types import SimpleNamespace

from graph_extract.usage import usable_content


def _resp(content, *, choices=True):
    if not choices:
        return SimpleNamespace(choices=[])
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_returns_text_for_a_normal_reply():
    assert usable_content(_resp("hello")) == "hello"


def test_none_for_empty_choices_list():
    assert usable_content(_resp(None, choices=False)) is None


def test_none_for_none_content():
    assert usable_content(_resp(None)) is None


def test_none_for_whitespace_only_content():
    assert usable_content(_resp("   \n ")) is None


def test_answer_api_still_exposes_it_under_the_old_name():
    from answer_api.synthesize import _usable_content
    assert _usable_content(_resp("hi")) == "hi"
    assert _usable_content(_resp(None, choices=False)) is None
