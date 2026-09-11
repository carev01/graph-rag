from __future__ import annotations
from dataclasses import dataclass, field

from openai import AsyncOpenAI

@dataclass
class UsageTally:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0
    by_call: dict = field(default_factory=dict)

    def add(self, kind: str, *, prompt: int, completion: int, cached: int = 0) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cached_tokens += cached
        self.calls += 1
        b = self.by_call.setdefault(kind, {"prompt": 0, "completion": 0, "calls": 0})
        b["prompt"] += prompt
        b["completion"] += completion
        b["calls"] += 1
        b["cached"] = b.setdefault("cached", 0) + cached

_TALLY = UsageTally()
def get_tally() -> UsageTally: return _TALLY
def reset_tally() -> None:
    global _TALLY
    _TALLY = UsageTally()

def _tally_usage(resp) -> None:
    u = getattr(resp, "usage", None)
    if u is None:
        return
    # chat.completions -> prompt_tokens/completion_tokens; responses API ->
    # input_tokens/output_tokens. Support both so cost is captured in either mode.
    prompt = getattr(u, "prompt_tokens", None)
    if prompt is None:
        prompt = getattr(u, "input_tokens", 0)
    completion = getattr(u, "completion_tokens", None)
    if completion is None:
        completion = getattr(u, "output_tokens", 0)
    details = getattr(u, "prompt_tokens_details", None) or getattr(u, "input_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details is not None else 0
    _TALLY.add("llm", prompt=prompt or 0, completion=completion or 0, cached=cached or 0)


def instrument(async_openai):
    """Wrap an openai AsyncOpenAI/AsyncAzureOpenAI so LLM calls tally usage.

    Patches chat.completions.create AND responses.create/parse (the Responses
    API path used by graphiti's structured OpenAIClient), so token usage is
    captured regardless of client mode. Returns the same object.
    """
    cc_orig = async_openai.chat.completions.create
    async def cc_create(*args, **kwargs):
        resp = await cc_orig(*args, **kwargs)
        _tally_usage(resp)
        return resp
    async_openai.chat.completions.create = cc_create

    responses = getattr(async_openai, "responses", None)
    for name in ("create", "parse"):
        fn = getattr(responses, name, None)
        if fn is None:
            continue
        def _wrap(orig):
            async def wrapped(*args, **kwargs):
                resp = await orig(*args, **kwargs)
                _tally_usage(resp)
                return resp
            return wrapped
        setattr(responses, name, _wrap(fn))
    return async_openai


def prefer_fast_provider(client: AsyncOpenAI, reasoning_effort: str = "") -> AsyncOpenAI:
    """Route by throughput, and bound reasoning so it cannot eat the output budget.

    OpenRouter serves one model through several providers at very different
    speeds (measured 29 tok/s vs 9.4 tok/s on an identical prompt, one unpinned
    call at 380s); `provider.sort=throughput` prefers the fast one.

    Reasoning tokens count as completion tokens, so on a reasoning model an
    unbounded effort level competes with the answer for the same max_tokens --
    and losing that race returns NO content (finish_reason='length',
    content=None). `reasoning_effort` bounds it. Use "low", never `enabled:
    false`: measured on the report verifier, `low` produced a real verdict in
    3028 tokens while `enabled: false` produced a 13-token rubber stamp that
    approved everything. An empty string sends no reasoning parameter at all --
    which is REQUIRED for a model that does not reason by default (measured
    2026-09-11 on upstage/solar-pro4: `effort: low` switched thinking ON and
    every max_tokens=8 classifier call came back empty).
    """
    orig = client.chat.completions.create

    async def create(*args, **kwargs):
        extra = dict(kwargs.get("extra_body") or {})
        provider = dict(extra.get("provider") or {})
        provider.setdefault("sort", "throughput")
        provider.setdefault("allow_fallbacks", True)
        extra["provider"] = provider
        if reasoning_effort:
            extra.setdefault("reasoning", {"effort": reasoning_effort})
        kwargs["extra_body"] = extra
        return await orig(*args, **kwargs)

    client.chat.completions.create = create  # type: ignore[method-assign]
    return client


def bounded_llm_client(base_url: str, api_key: str, *, reasoning_effort: str,
                       timeout: float = 180.0, max_retries: int = 3) -> AsyncOpenAI:
    """The one way to build a chat-completions client for a synthesis-side tier.

    Every tier gets: usage tallying, a finite per-request timeout with a
    bounded retry count (a hung route aborts and retries on a new route
    instead of hanging the caller -- measured 478s/528s eval questions and a
    2h09m eval against ~20 min before this), and on OpenRouter, throughput
    routing plus the tier's reasoning bound (see prefer_fast_provider).
    `provider`/`reasoning` are OpenRouter extensions, so another gateway gets
    a plain request.

    This is the fourth place the same fix was needed; it lives here, in the
    layer both answer_api and theme_builder import, so it is not copied again.
    """
    client = instrument(AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url,
                                    timeout=timeout, max_retries=max_retries))
    if "openrouter" in base_url:
        client = prefer_fast_provider(client, reasoning_effort)
    return client


def usable_content(resp) -> str | None:
    """The text of a chat completion, or None when the model returned nothing.

    Two ways a HTTP 200 carries no answer: an empty `choices` list, and
    `content=None` because a reasoning model spent its whole max_tokens budget on
    reasoning tokens and emitted no final message. Both must stay distinguishable
    from a real reply — coercing them to "" has caused four separate defects here.
    """
    if not resp.choices:
        return None
    content = resp.choices[0].message.content
    return content if content and content.strip() else None
