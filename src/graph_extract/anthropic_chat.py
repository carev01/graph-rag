"""Claude (Anthropic Messages API) behind the chat-completions surface the synthesis
tiers already call.

The report writer, report verifier, map and eval-judge tiers all call
`client.chat.completions.create(model=, messages=, max_tokens=, temperature=)` and
read `usable_content(resp)` / `resp.choices[0].finish_reason` / `resp.usage`. This
adapter serves that exact surface from the official Anthropic SDK, so a tier moves to
Claude by configuration alone -- point its `*_BASE_URL` at https://api.anthropic.com
and its `*_MODEL` at a Claude model -- and every caller, `instrument`'s usage tally and
the LLM capture keep working untouched.

Mapping notes:
- `system` chat messages become the top-level `system` parameter; Claude takes no
  system role inside `messages`. That block carries a prompt-cache breakpoint
  (BACKLOG 58): `cache_layout` moves graphiti's static prompt tails into the system
  message, so it is the one part of a request that repeats byte for byte. The
  user message is never marked -- it is per-call content, and a breakpoint after it
  pays the 1.25x write on every request for reads that never come. A prefix under
  the model's minimum (measured: Claude Haiku 5.5 caches a 782-token system block,
  2026-10-14) is simply not cached, at no cost.
- `temperature` is dropped: current Claude models (and SDK 1.x) no longer take it.
- The tier's reasoning effort becomes `output_config.effort` when it is a level Claude
  accepts; anything else (an empty string, OpenRouter's "minimal") sends none, which
  runs the model's default.
- An empty reply comes back as `content=None`, never "", so `usable_content` sees an
  unusable reply exactly as it would from any other provider.
- Requests are streamed (large report budgets); a reply cut off at max_tokens is
  logged, since thinking shares that budget with the answer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

ANTHROPIC_HOST = "api.anthropic.com"
_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
# Claude stop_reason -> the chat-completions finish_reason callers log and branch on.
_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
           "refusal": "content_filter", "tool_use": "tool_calls"}


def is_anthropic(base_url: str) -> bool:
    return ANTHROPIC_HOST in (base_url or "")


@dataclass
class _Message:
    content: str | None
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message
    finish_reason: str
    index: int = 0


@dataclass
class _PromptDetails:
    cached_tokens: int = 0


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_details: _PromptDetails = field(default_factory=_PromptDetails)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ChatResponse:
    """The subset of a chat completion the synthesis tiers read."""
    model: str
    choices: list[_Choice]
    usage: _Usage
    id: str = ""


def to_chat_response(msg: Any) -> ChatResponse:
    """Convert an Anthropic `Message` into the chat-completions shape."""
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    u = msg.usage
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
    prompt = (u.input_tokens or 0) + cache_read + cache_write
    return ChatResponse(
        model=msg.model, id=msg.id,
        choices=[_Choice(message=_Message(content=text if text.strip() else None),
                         finish_reason=_FINISH.get(msg.stop_reason, msg.stop_reason or ""))],
        usage=_Usage(prompt_tokens=prompt, completion_tokens=u.output_tokens or 0,
                     prompt_tokens_details=_PromptDetails(cached_tokens=cache_read)))


def split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Pull chat `system` messages out into Claude's top-level system prompt."""
    system = [m["content"] for m in messages if m.get("role") == "system"]
    rest = [{"role": m["role"], "content": m["content"]}
            for m in messages if m.get("role") != "system"]
    return ("\n\n".join(system) if system else None), rest


class AnthropicChat:
    """Duck-types `AsyncOpenAI` for `chat.completions.create` only."""

    def __init__(self, *, api_key: str, reasoning_effort: str = "",
                 timeout: float = 180.0, max_retries: int = 3,
                 client: anthropic.AsyncAnthropic | None = None,
                 cache_system: bool = True):
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key, timeout=timeout, max_retries=max_retries)
        effort = (reasoning_effort or "").strip().lower()
        if effort and effort not in _EFFORTS:
            logger.warning("reasoning effort %r is not a Claude effort level; sending "
                           "none (the model's default)", reasoning_effort)
            effort = ""
        self._effort = effort
        self._cache_system = cache_system
        # A mutable namespace, so `instrument` can wrap `create` like on AsyncOpenAI.
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def close(self) -> None:
        await self._client.close()

    async def _create(self, *, model: str, messages: list[dict], max_tokens: int,
                      temperature: float | None = None, **ignored: Any) -> ChatResponse:
        if ignored:
            logger.debug("anthropic adapter ignoring chat parameters: %s", sorted(ignored))
        system, msgs = split_system(messages)
        kwargs: dict[str, Any] = {}
        if system is not None:
            kwargs["system"] = ([{"type": "text", "text": system,
                                  "cache_control": {"type": "ephemeral"}}]
                                if self._cache_system else system)
        if self._effort:
            kwargs["output_config"] = {"effort": self._effort}
        # Always streamed: report calls carry large max_tokens (thinking + a long JSON
        # report), and the SDK recommends streaming there to stay clear of HTTP
        # timeouts. get_final_message() returns the same Message create() would.
        async with self._client.messages.stream(
                model=model, max_tokens=max_tokens, messages=msgs,  # type: ignore[arg-type]
                **kwargs) as stream:
            msg = await stream.get_final_message()
        if msg.stop_reason == "max_tokens":
            # Thinking counts against max_tokens; a report cut off here is unparseable
            # JSON, and the caller only sees "no usable reply". Say why.
            logger.warning("claude model %s hit max_tokens=%d (output %d tokens); the "
                           "reply is truncated", model, max_tokens, msg.usage.output_tokens)
        if msg.stop_reason == "refusal":
            logger.warning("claude model %s refused (category=%s)", model,
                           getattr(getattr(msg, "stop_details", None), "category", None))
        return to_chat_response(msg)
