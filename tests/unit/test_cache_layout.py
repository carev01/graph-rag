"""graph_extract.cache_layout: move each prompt type's static user-message tail into
the system message so provider prefix caches can serve it.

graphiti puts per-call content (previous/current messages, entities) BEFORE the
static rules, so consecutive calls share almost no prefix (~7% cache hits on the
cheap tier). Measured on llama.cpp (scripts/reorder_prompts.py): the static block
caches only when it sits in the SYSTEM message (6.4x), not at the front of the user
message. The static tail is learned online per prompt type, so a graphiti upgrade or
an ontology change re-learns instead of drifting from committed prompt text.
"""
from __future__ import annotations

import pytest

from graph_extract.cache_layout import StaticTailLearner, clean_block, wrap_client
from graph_extract.graphiti_client import DOCUMENT_CONTEXT_MARKER

STATIC = ("\n\n# TASK\nExtract every entity.\nUse only the text above.\n"
          "Return JSON matching the schema.\n")


def _user(i: int, *, tail: str = STATIC) -> str:
    return (f"<PREVIOUS MESSAGES>\nprevious {i}\n</PREVIOUS MESSAGES>\n"
            f"<CURRENT MESSAGE>\ncontent number {i} {'x' * i}\n</CURRENT MESSAGE>{tail}")


def _context(article: str) -> str:
    # graphiti renders custom_extraction_instructions (STATIC + this) at the
    # END of the user message -- document_context's real leading text, keyed
    # by article so distinct articles are distinct contexts.
    return f'{DOCUMENT_CONTEXT_MARKER} This chunk is from article {article}.'


def _user_with_context(i: int, article: str, *, tail: str = STATIC) -> str:
    return _user(i, tail=tail) + _context(article)


def _msgs(i: int, system: str = "You extract entities.") -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": _user(i)}]


def _msgs_with_context(i: int, article: str, system: str = "You extract entities.") -> list[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": _user_with_context(i, article)}]


def _learner(**kw) -> StaticTailLearner:
    return StaticTailLearner(**{"min_samples": 3, "min_chars": 20, **kw})


def test_nothing_moves_while_learning_then_the_static_tail_moves_to_system():
    ln = _learner()
    for i in range(3):
        out = ln.transform("ExtractedEntities", _msgs(i))
        assert out == _msgs(i), "no change before the block is learned"
    out = ln.transform("ExtractedEntities", _msgs(7))
    assert out[0]["content"] == "You extract entities.\n\n" + STATIC.strip() + "\n"
    assert out[1]["content"].endswith("</CURRENT MESSAGE>\n")
    assert "# TASK" not in out[1]["content"]
    # No content lost or duplicated: every static line is now in the system message.
    assert all(line in out[0]["content"] for line in STATIC.strip().splitlines())


def test_the_system_prefix_is_byte_identical_across_calls_once_learned():
    """That identity is the whole point: it is what a prefix cache can reuse."""
    ln = _learner()
    for i in range(3):
        ln.transform("ExtractedEntities", _msgs(i))
    systems = {ln.transform("ExtractedEntities", _msgs(i))[0]["content"] for i in range(10, 15)}
    assert len(systems) == 1


def test_prompt_types_learn_independently():
    ln = _learner()
    other = STATIC.replace("entity", "fact")
    for i in range(3):
        ln.transform("A", _msgs(i))
        ln.transform("B", [{"role": "system", "content": "S"},
                           {"role": "user", "content": _user(i, tail=other)}])
    a = ln.transform("A", _msgs(9))
    b = ln.transform("B", [{"role": "system", "content": "S"},
                           {"role": "user", "content": _user(9, tail=other)}])
    assert "Extract every entity." in a[0]["content"]
    assert "Extract every fact." in b[0]["content"]


def test_a_user_message_not_ending_with_the_learned_block_is_left_alone():
    ln = _learner()
    for i in range(3):
        ln.transform("A", _msgs(i))
    odd = [{"role": "system", "content": "You extract entities."},
           {"role": "user", "content": "something else entirely"}]
    assert ln.transform("A", odd) == odd
    assert ln.stats()["skipped"] == 1


def test_short_common_tails_are_not_moved():
    """A tail shorter than min_chars is coincidence, not a static block."""
    ln = _learner(min_chars=500)
    for i in range(5):
        assert ln.transform("A", _msgs(i)) == _msgs(i)
    assert ln.stats()["moved"] == 0


@pytest.mark.parametrize("msgs", [
    [{"role": "user", "content": "no system message"}],
    [{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text"}]}],
    [],
])
def test_unusual_message_shapes_pass_through(msgs):
    ln = _learner(min_samples=1)
    for _ in range(3):
        assert ln.transform("A", msgs) == msgs


def test_clean_block_drops_a_partial_line_and_a_leading_closing_tag():
    """Carried over from scripts/reorder_prompts.py: the common suffix can start
    mid-token (a timestamp ending in a constant '+00:00') or with the closer of the
    preceding variable block -- neither belongs to the static rules."""
    raw = "43656+00:00\n</REFERENCE_TIME>\n\n# RULES\nDo this.\n"
    assert clean_block(raw) == "# RULES\nDo this.\n"


async def test_wrap_client_transforms_the_messages_sent_and_keys_by_schema_name():
    sent: list[list[dict]] = []

    class _Completions:
        async def create(self, *args, **kwargs):
            sent.append(kwargs["messages"])
            return "ok"

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    client = wrap_client(_Client(), _learner())
    fmt = {"type": "json_schema", "json_schema": {"name": "ExtractedEntities"}}
    for i in range(4):
        await client.chat.completions.create(model="m", messages=_msgs(i), response_format=fmt)
    assert sent[0] == _msgs(0)
    assert "# TASK" in sent[3][0]["content"] and "# TASK" not in sent[3][1]["content"]


# -- extraction-grounding (2026-10-17): per-article DOCUMENT CONTEXT must not
# poison the learned static block -- regression tests ------------------------

def test_same_article_warmup_learns_without_baking_in_that_articles_context():
    """Reviewer probe: warming up on ONE article's calls must not bake that
    article's DOCUMENT CONTEXT into the learned block -- otherwise every
    later, different-article call fails to match and nothing ever moves."""
    ln = _learner()
    for i in range(3):
        ln.transform("ExtractedEntities", _msgs_with_context(i, "A1"))
    out = ln.transform("ExtractedEntities", _msgs_with_context(7, "A2"))
    assert "# TASK" in out[0]["content"]          # static block moved...
    assert "article A1" not in out[0]["content"]  # ...with no article baked in
    assert "article A2" not in out[0]["content"]
    assert "# TASK" not in out[1]["content"]       # static rules gone from user
    assert "article A2" in out[1]["content"]       # THIS call's context stays
    assert ln.stats()["moved"] == 1


def test_mixed_article_warmup_still_learns_and_moves():
    """A different article's context on every warm-up call must not collapse
    the learned common suffix to ~nothing -- only the context varied, not the
    static rules, so the fix must strip the context before comparing."""
    ln = _learner()
    for i, article in enumerate(("A1", "A2", "A3")):
        ln.transform("ExtractedEntities", _msgs_with_context(i, article))
    assert ln.stats()["learned"] == 1
    out = ln.transform("ExtractedEntities", _msgs_with_context(9, "A4"))
    assert "# TASK" in out[0]["content"]
    assert "article A4" in out[1]["content"]
    assert ln.stats()["moved"] == 1


def test_context_text_always_stays_in_the_user_message():
    """Whether still warming up or already matching and moving, the per-call
    context must never be dropped -- only excluded from what's COMPARED."""
    ln = _learner()
    for i, article in enumerate(("A1", "A2", "A3")):
        out = ln.transform("ExtractedEntities", _msgs_with_context(i, article))
        assert f"article {article}" in out[1]["content"]   # still learning: unchanged
    for i, article in enumerate(("A4", "A5", "A6", "A7")):
        out = ln.transform("ExtractedEntities", _msgs_with_context(10 + i, article))
        assert f"article {article}" in out[1]["content"]   # moved: re-appended


def test_no_marker_path_is_unchanged():
    """Baseline behaviour (no DOCUMENT CONTEXT tail) must be untouched by the
    split -- mirrors test_nothing_moves_while_learning_then_the_static_tail_moves_to_system."""
    ln = _learner()
    for i in range(3):
        assert ln.transform("ExtractedEntities", _msgs(i)) == _msgs(i)
    out = ln.transform("ExtractedEntities", _msgs(7))
    assert out[0]["content"] == "You extract entities.\n\n" + STATIC.strip() + "\n"
    assert out[1]["content"].endswith("</CURRENT MESSAGE>\n")
    assert "# TASK" not in out[1]["content"]
    assert ln.stats()["moved"] == 1
