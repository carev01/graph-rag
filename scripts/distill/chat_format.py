"""Builds SFT training text aligned, byte-for-byte, with the thinking-off serving
prompt (fix round 1, 2026-10-17; spec section 7).

**Why this exists.** qwen3.5's chat template, called with `enable_thinking=False`,
renders the GENERATION PROMPT for an assistant turn -- the exact text vLLM sends the
model to complete at serving time -- as (checked against the model's own
`chat_template.jinja` in `carev01/qwen35-4b-graphrag`, 2026-10-17):

    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n

The empty `<think></think>` block is already part of the PROMPT at serving time, not
something the model emits. If training instead treats that block as part of the
model's own completion -- which a naive full-sequence render can do, see below -- the
model is trained to also produce the empty think block as output, which is not the
task it is asked to do when served.

**The complication this module exists to avoid.** The SAME template can render a
HISTORICAL assistant turn, inside a full multi-turn `apply_chat_template` call,
DIFFERENTLY from the GENERATION PROMPT for that same turn -- for example without the
empty think block, once the turn is no longer the one being generated. A naive
approach -- render the whole conversation once, then string-search for where the
assistant turn "starts" -- can therefore land the mask boundary in the wrong place
depending on which rendering branch the template takes for a completed (not
in-progress) turn. `build_training_text` sidesteps this: it NEVER computes the mask
boundary from the plain full-conversation render. It always renders the PREFIX (every
message except the final assistant one) with `add_generation_prompt=True` -- which is,
by definition, the serving-time prompt for that example -- and appends the answer plus
an end-of-turn marker directly to it. The plain full render is used only to classify,
for bookkeeping (`path`), whether the two happen to coincide on a given tokenizer's
template; training always uses the constructed text, never the plain one.
"""
from __future__ import annotations

from typing import NamedTuple, Protocol

#: Qwen's ChatML end-of-turn marker. Passable as `end_of_turn=` for a different
#: template.
END_OF_TURN = "<|im_end|>\n"


class ChatTemplateTokenizer(Protocol):
    """The one method this module needs from a tokenizer -- matches both a real HF
    `PreTrainedTokenizer` and the stub used in `tests/unit/test_distill_chat_format.py`."""

    def apply_chat_template(self, messages: list[dict[str, str]], *, tokenize: bool,
                            add_generation_prompt: bool, enable_thinking: bool) -> str:
        ...


class BuiltExample(NamedTuple):
    #: The exact text to train on: `serving_prompt(messages[:-1]) + answer + end_of_turn`.
    text: str
    #: `len(text)` up to and including the serving prompt -- mask every token before
    #: this index; the loss covers only `text[mask_start:]` (the answer + end-of-turn
    #: marker), matching what the model actually emits when served.
    mask_start: int
    #: "plain" if the whole-conversation render (no generation prompt) happened to be
    #: byte-identical to the constructed text on this tokenizer's template; otherwise
    #: "generation_prompt", meaning the template renders the historical turn
    #: differently from the generation prompt and the constructed text was needed.
    #: Bookkeeping only -- `text`/`mask_start` are the same in both cases.
    path: str


def serving_prompt(tokenizer: ChatTemplateTokenizer, messages: list[dict[str, str]]) -> str:
    """The exact prompt text the server sends the model to complete at inference,
    thinking off. `messages` must NOT include the assistant's completion -- this is
    what `add_generation_prompt=True` is for."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def build_training_text(tokenizer: ChatTemplateTokenizer, messages: list[dict[str, str]],
                        *, end_of_turn: str = END_OF_TURN) -> BuiltExample:
    """`messages` is `[..., {"role": "assistant", "content": <answer>}]`. Returns the
    training text, the completion-loss mask boundary, and which construction path
    this tokenizer's template took (see the module docstring)."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("the last message must be the assistant's completion")
    answer = messages[-1]["content"]
    prefix = serving_prompt(tokenizer, messages[:-1])
    constructed = prefix + answer + end_of_turn
    plain = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    path = "plain" if plain == constructed else "generation_prompt"
    return BuiltExample(text=constructed, mask_start=len(prefix), path=path)
