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
  accepts; "none" sends `thinking: {"type": "disabled"}` (Claude Haiku 5.5 thinks by
  default: +83% output tokens on an entity extraction with the same entities,
  2026-10-14); anything else (an empty string, OpenRouter's "minimal") sends none,
  which runs the model's default.
- A `response_format` json_schema (graphiti's structured extraction) becomes
  `output_config.format`, made acceptable to Claude: every object gets
  `additionalProperties: false` (a 400 without it) and the constraints Claude rejects
  -- `maxItems` included (a 400, measured) -- are removed. The index-array cap that
  `maxItems` carried for other providers is checked on the reply instead and counted
  in `ARRAY_OVERRUNS`. `json_object` has no Claude equivalent and sends no format
  (graphiti then puts the schema in the prompt).
- An empty reply comes back as `content=None`, never "", so `usable_content` sees an
  unusable reply exactly as it would from any other provider.
- Requests are streamed (large report budgets); a reply cut off at max_tokens is
  logged, since thinking shares that budget with the answer.
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

ANTHROPIC_HOST = "api.anthropic.com"
_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_NO_THINKING = frozenset({"none", "disabled", "off"})
# JSON-schema keywords Claude's structured outputs reject (docs: numerical, string and
# complex array constraints). Removed from schema nodes, never from `properties` maps,
# whose keys are field names.
_UNSUPPORTED = frozenset({"maxItems", "minItems", "uniqueItems", "minimum", "maximum",
                          "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
                          "minLength", "maxLength"})


@dataclass
class ArrayOverruns:
    """Replies whose integer arrays exceeded the request schema's `maxItems`."""
    replies: int = 0
    longest: int = 0


ARRAY_OVERRUNS = ArrayOverruns()


@dataclass
class ClaudeUsage:
    """Process-wide Claude token tally with cache writes kept apart: Claude bills a
    write at 1.25x input and a read at 0.1x, which the chat-completions usage shape
    (prompt / cached) cannot express. Dollar accounting reads this."""
    calls: int = 0
    input_tokens: int = 0          # uncached, full price
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage: Any) -> None:
        self.calls += 1
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0

    def dollars(self, input_per_m: float, output_per_m: float) -> float:
        return (self.input_tokens * input_per_m + self.cache_read_tokens * input_per_m
                * 0.1 + self.cache_write_tokens * input_per_m * 1.25
                + self.output_tokens * output_per_m) / 1e6


CLAUDE_USAGE = ClaudeUsage()
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


def claude_schema(schema: dict) -> tuple[dict, int | None]:
    """`schema` made acceptable to Claude, plus the smallest `maxItems` it carried."""
    out = copy.deepcopy(schema)
    caps: list[int] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("maxItems"), int):
                caps.append(node["maxItems"])
            for key in _UNSUPPORTED & node.keys():
                del node[key]
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
            for key, value in node.items():
                if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
                    for sub_schema in value.values():
                        walk(sub_schema)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(out)
    return out, (min(caps) if caps else None)


def _longest_int_array(value: Any) -> int:
    if isinstance(value, list):
        own = len(value) if value and all(isinstance(v, int) for v in value) else 0
        return max([own] + [_longest_int_array(v) for v in value])
    if isinstance(value, dict):
        return max([0] + [_longest_int_array(v) for v in value.values()])
    return 0


def _note_overrun(text: str | None, cap: int | None, model: str) -> None:
    if not text or cap is None:
        return
    try:
        longest = _longest_int_array(json.loads(text))
    except ValueError:
        return
    if longest > cap:
        ARRAY_OVERRUNS.replies += 1
        ARRAY_OVERRUNS.longest = max(ARRAY_OVERRUNS.longest, longest)
        logger.warning("claude model %s returned an integer array of %d items, over the "
                       "schema's maxItems=%d (not enforceable on Claude)", model, longest,
                       cap)


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
        self._thinking_disabled = effort in _NO_THINKING
        if self._thinking_disabled:
            effort = ""
        elif effort and effort not in _EFFORTS:
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
                      temperature: float | None = None,
                      response_format: dict | None = None,
                      **ignored: Any) -> ChatResponse:
        if ignored:
            logger.debug("anthropic adapter ignoring chat parameters: %s", sorted(ignored))
        system, msgs = split_system(messages)
        kwargs: dict[str, Any] = {}
        if system is not None:
            kwargs["system"] = ([{"type": "text", "text": system,
                                  "cache_control": {"type": "ephemeral"}}]
                                if self._cache_system else system)
        output_config: dict[str, Any] = {}
        if self._effort:
            output_config["effort"] = self._effort
        if self._thinking_disabled:
            kwargs["thinking"] = {"type": "disabled"}
        cap: int | None = None
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            schema, cap = claude_schema(response_format["json_schema"]["schema"])
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if output_config:
            kwargs["output_config"] = output_config
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
        CLAUDE_USAGE.add(msg.usage)
        resp = to_chat_response(msg)
        _note_overrun(resp.choices[0].message.content, cap, model)
        return resp
