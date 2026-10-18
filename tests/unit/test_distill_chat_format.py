"""`chat_format.build_training_text`: the training text always matches the thinking-off
SERVING prompt (`<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`) plus the answer plus
the end-of-turn marker -- never the plain full-conversation render, which a template is
free to render differently for a historical (already-completed) assistant turn than for
the live generation prompt of that same turn (fix round 1, 2026-10-17).

`jinja2` is not installed in this environment, so templates are exercised through a
stub tokenizer that mimics qwen3.5's documented `chat_template.jinja` semantics (thinking
off -> an empty `<think></think>` block in the GENERATION prompt) rather than a real
Jinja render -- the brief's documented fallback when jinja2 is unavailable.
"""
from __future__ import annotations

import pytest

from scripts.distill import chat_format


class StubQwenTokenizer:
    """Mimics qwen3.5's chat-template semantics closely enough to exercise both of
    `build_training_text`'s paths.

    `historical_think_block` controls whether a PLAIN (non-generation-prompt) render of
    an assistant turn also carries the empty `<think></think>` block that the
    GENERATION PROMPT always carries when `enable_thinking=False`. The real qwen3.5
    template does NOT (confirmed against `carev01/qwen35-4b-graphrag`'s
    `chat_template.jinja`, 2026-10-17) -- `historical_think_block=False` reproduces
    that and exercises the "generation_prompt" path; `True` exercises the "plain" path,
    for templates where the two happen to coincide.
    """

    THINK = "<think>\n\n</think>\n\n"

    def __init__(self, historical_think_block: bool = False) -> None:
        self.historical_think_block = historical_think_block
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            enable_thinking):
        self.calls.append({"messages": messages, "tokenize": tokenize,
                           "add_generation_prompt": add_generation_prompt,
                           "enable_thinking": enable_thinking})
        assert tokenize is False, "this module never tokenizes through the template"
        assert enable_thinking is False, "thinking must be off end to end"
        rendered = ""
        for m in messages:
            think = self.THINK if (m["role"] == "assistant"
                                   and self.historical_think_block) else ""
            rendered += f"<|im_start|>{m['role']}\n{think}{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            rendered += f"<|im_start|>assistant\n{self.THINK}"
        return rendered


EXAMPLE = [
    {"role": "system", "content": "You are an extraction model."},
    {"role": "user", "content": "<TEXT>Veeam Backup & Replication</TEXT>"},
    {"role": "assistant", "content": '{"entities": []}'},
]


def test_serving_prompt_excludes_the_assistant_turn_and_includes_think_block():
    tok = StubQwenTokenizer()
    prompt = chat_format.serving_prompt(tok, EXAMPLE[:-1])

    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert '"entities"' not in prompt


def test_training_text_is_serving_prompt_plus_answer_plus_end_of_turn():
    tok = StubQwenTokenizer()
    built = chat_format.build_training_text(tok, EXAMPLE)

    serving = chat_format.serving_prompt(tok, EXAMPLE[:-1])
    answer = EXAMPLE[-1]["content"]
    assert built.text == serving + answer + chat_format.END_OF_TURN
    assert built.mask_start == len(serving)
    # The masked prefix is exactly the serving prompt; nothing of the answer leaks
    # into it and nothing of the serving prompt leaks into the loss region.
    assert built.text[:built.mask_start] == serving
    assert built.text[built.mask_start:] == answer + chat_format.END_OF_TURN


def test_path_is_generation_prompt_when_the_template_diverges_on_history():
    # The realistic qwen3.5 case: the plain render omits the think block on a
    # historical (completed) assistant turn, so the plain and constructed texts
    # differ -- the module must still produce the CONSTRUCTED text, not the plain one.
    tok = StubQwenTokenizer(historical_think_block=False)
    built = chat_format.build_training_text(tok, EXAMPLE)

    plain = tok.apply_chat_template(EXAMPLE, tokenize=False, add_generation_prompt=False,
                                    enable_thinking=False)
    assert plain != built.text, "the stub should reproduce the real divergence"
    assert built.path == "generation_prompt"
    assert "<think>" in built.text  # the serving-prompt-derived think block, kept


def test_path_is_plain_when_the_template_agrees_on_history():
    # A hypothetical template where the historical render already matches the
    # generation prompt: the constructed text coincides with the plain render.
    tok = StubQwenTokenizer(historical_think_block=True)
    built = chat_format.build_training_text(tok, EXAMPLE)

    plain = tok.apply_chat_template(EXAMPLE, tokenize=False, add_generation_prompt=False,
                                    enable_thinking=False)
    assert plain == built.text
    assert built.path == "plain"


def test_never_calls_apply_chat_template_with_tokenize_true():
    tok = StubQwenTokenizer()
    chat_format.build_training_text(tok, EXAMPLE)
    assert all(c["tokenize"] is False for c in tok.calls)


def test_rejects_messages_not_ending_in_assistant():
    tok = StubQwenTokenizer()
    with pytest.raises(ValueError):
        chat_format.build_training_text(tok, EXAMPLE[:-1])


def test_rejects_empty_messages():
    tok = StubQwenTokenizer()
    with pytest.raises(ValueError):
        chat_format.build_training_text(tok, [])


@pytest.mark.skipif(True, reason="jinja2 not installed in this environment; "
                                 "the module docstring documents the expected real "
                                 "template semantics this verifies against a stub "
                                 "instead (see StubQwenTokenizer)")
def test_against_a_real_jinja_render_if_available():  # pragma: no cover
    import jinja2  # noqa: F401
