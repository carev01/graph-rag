from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class UsageTally:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    by_call: dict = field(default_factory=dict)

    def add(self, kind: str, *, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1
        b = self.by_call.setdefault(kind, {"prompt": 0, "completion": 0, "calls": 0})
        b["prompt"] += prompt
        b["completion"] += completion
        b["calls"] += 1

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
    _TALLY.add("llm", prompt=prompt or 0, completion=completion or 0)


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
    async_openai.chat.completions.create = cc_create  # type: ignore[assignment]

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
