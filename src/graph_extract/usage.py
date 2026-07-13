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
        b["prompt"] += prompt; b["completion"] += completion; b["calls"] += 1

_TALLY = UsageTally()
def get_tally() -> UsageTally: return _TALLY
def reset_tally() -> None:
    global _TALLY
    _TALLY = UsageTally()

def instrument(async_openai):
    """Wrap an openai.AsyncOpenAI so chat.completions.create tallies .usage.
    Returns the same object with the create method patched."""
    orig = async_openai.chat.completions.create
    async def create(*args, **kwargs):
        resp = await orig(*args, **kwargs)
        u = getattr(resp, "usage", None)
        if u is not None:
            _TALLY.add("llm", prompt=getattr(u, "prompt_tokens", 0) or 0,
                       completion=getattr(u, "completion_tokens", 0) or 0)
        return resp
    async_openai.chat.completions.create = create  # type: ignore[assignment]
    return async_openai
