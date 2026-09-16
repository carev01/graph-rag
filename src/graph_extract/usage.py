from __future__ import annotations
import json
import logging
import threading
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

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

# A per-scope tally alongside the global one. The global `_TALLY` is what
# eval.py / probe.py read for a whole run; it cannot attribute spend to one
# article once articles run concurrently (a before/after delta on it includes
# every sibling's tokens). asyncio copies the context per task, so a scope set
# in one article's task is invisible to the others -- the same pattern as
# CURRENT_DEDUP_STATS in dedup_guard.py.
CURRENT_USAGE_TALLY: ContextVar[UsageTally | None] = ContextVar(
    "CURRENT_USAGE_TALLY", default=None)

# The graphiti prompt name (e.g. "extract_edges.edge", "dedupe_edges.resolve_edge")
# in scope for the LLM call `instrument()` is currently capturing. Lives here (not
# dedup_guard.py, which is the only place that ever sets it) so dedup_guard can
# import it without creating an import cycle -- dedup_guard already imports
# nothing from usage.py, and usage.py imports nothing from dedup_guard.
#
# A ContextVar, not a module global, for the same reason as CURRENT_USAGE_TALLY /
# CURRENT_DEDUP_STATS: ingest runs several articles concurrently, each issuing many
# in-flight LLM calls on one event loop, and asyncio only copies context per task --
# a global set-before/read-after would let one call's name leak onto a concurrent
# sibling's capture record. Call sites that never go through graphiti's LLM client
# (the global-search map step, the router classifier) never set this, so their
# records carry "" -- deliberately distinct from dedup_guard's "unknown" fallback,
# which means a graphiti call arrived with no prompt_name at all (a regression).
CURRENT_PROMPT_NAME: ContextVar[str] = ContextVar("CURRENT_PROMPT_NAME", default="")


def record(kind: str, *, prompt: int, completion: int, cached: int = 0) -> None:
    """Dual write: the global tally always, plus the open scope when there is one."""
    _TALLY.add(kind, prompt=prompt, completion=completion, cached=cached)
    scoped = CURRENT_USAGE_TALLY.get()
    if scoped is not None:
        scoped.add(kind, prompt=prompt, completion=completion, cached=cached)


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
    record("llm", prompt=prompt or 0, completion=completion or 0, cached=cached or 0)


# Serializes JSONL append writes across concurrent callers in this process.
# Ingest runs up to `max_coroutines` articles concurrently, each with many
# in-flight LLM calls sharing one event loop; without this, two coroutines'
# `open(...).write(...)` could interleave mid-line and corrupt the file. The
# lock is only ever held for one small append, so briefly blocking the loop
# is a non-issue in practice.
_capture_lock = threading.Lock()


def _jsonable(value: object) -> object:
    """Best-effort JSON-safe rendering. Never raises -- the caller (instrument's
    try/except) is the real safety net; this just maximizes what survives it.

    Handles the one non-JSON-native shape this codebase's LLM call sites pass:
    graphiti's Responses-API `text_format` kwarg is a pydantic BaseModel CLASS
    (not instance), used as the structured-output schema.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, type) and hasattr(value, "model_json_schema"):
        return {"pydantic_model": value.__name__, "json_schema": value.model_json_schema()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return str(value)


def _request_payload(kwargs: dict) -> dict:
    """The parts of a call's kwargs worth capturing: chat's `messages` or the
    Responses API's `input`, and whichever schema kwarg was passed
    (`response_format` for chat/generic-json, `text_format` for structured
    Responses-API parse calls). Never includes `api_key` -- that lives on the
    client object, not in per-call kwargs, so it is structurally excluded."""
    payload: dict = {}
    if "messages" in kwargs:
        payload["messages"] = _jsonable(kwargs.get("messages"))
    if "input" in kwargs:
        payload["input"] = _jsonable(kwargs.get("input"))
    fmt = kwargs.get("response_format", kwargs.get("text_format"))
    if fmt is not None:
        payload["response_format"] = _jsonable(fmt)
    return payload


def _completion_text(resp: object) -> str | None:
    choices = getattr(resp, "choices", None)
    if choices:
        content = getattr(choices[0].message, "content", None)
        if content is not None:
            return content
    parsed = getattr(resp, "output_parsed", None)
    if parsed is not None:
        dump = getattr(parsed, "model_dump_json", None)
        return dump() if callable(dump) else str(parsed)
    text = getattr(resp, "output_text", None)
    return text if text is not None else None


def _finish_reason(resp: object) -> str | None:
    choices = getattr(resp, "choices", None)
    if choices:
        return getattr(choices[0], "finish_reason", None)
    return getattr(resp, "status", None)  # Responses API's nearest equivalent


def capture_llm_call(path: str, *, call_id: str, tier: str, model: str, api: str,
                     kwargs: dict, resp: object) -> None:
    """Append one JSONL record of a full prompt/response pair.

    `prompt_name` is read from CURRENT_PROMPT_NAME, not passed in: only
    dedup_guard's `generate_response` wrapper ever sets it (to graphiti's
    `prompt_name` kwarg, e.g. "extract_edges.edge"), for the duration of the one
    call that reaches this layer beneath it. Call sites that never route through
    graphiti's LLM client (global-search map, router classifier) never set it, so
    their records get "" -- not "unknown", which is reserved for a graphiti call
    that arrived with no prompt_name at all (a regression, see dedup_guard.py).

    Not called directly from production code except via `instrument`'s
    try/except wrapper -- a serialisation failure here must never break the
    LLM call it observed.
    """
    u = getattr(resp, "usage", None)
    prompt = getattr(u, "prompt_tokens", None) if u is not None else None
    if prompt is None and u is not None:
        prompt = getattr(u, "input_tokens", None)
    completion = getattr(u, "completion_tokens", None) if u is not None else None
    if completion is None and u is not None:
        completion = getattr(u, "output_tokens", None)
    details = None
    if u is not None:
        details = (getattr(u, "prompt_tokens_details", None)
                  or getattr(u, "input_tokens_details", None))
    record = {
        "call_id": call_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tier": tier,
        "prompt_name": CURRENT_PROMPT_NAME.get(),
        "model": model or getattr(resp, "model", None),
        "api": api,
        **_request_payload(kwargs),
        "completion": _completion_text(resp),
        "finish_reason": _finish_reason(resp),
        "prompt_tokens": prompt or 0,
        "completion_tokens": completion or 0,
        "cached_tokens": (getattr(details, "cached_tokens", 0) if details is not None else 0) or 0,
    }
    line = json.dumps(record, default=str) + "\n"
    with _capture_lock:
        with open(path, "a", buffering=1, encoding="utf-8") as f:
            f.write(line)


def _safe_capture(path: str, **kw) -> None:
    """The whole point: capture must never break the call it observes."""
    try:
        capture_llm_call(path, **kw)
    except Exception:
        logger.warning("llm capture failed for path %s; continuing without it",
                       path, exc_info=True)


def instrument(async_openai, *, tier: str = "", capture_path: str = ""):
    """Wrap an openai AsyncOpenAI/AsyncAzureOpenAI so LLM calls tally usage.

    Patches chat.completions.create AND responses.create/parse (the Responses
    API path used by graphiti's structured OpenAIClient), so token usage is
    captured regardless of client mode. Returns the same object.

    `capture_path` additionally appends a full-fidelity JSONL record per call
    (see `capture_llm_call`) when non-empty -- empty (the default) costs
    nothing: no file is opened, nothing is serialized. `tier` is an opaque
    label written into each record ("cheap"/"strong"/"router-classify"/...);
    it plays no role in token tallying.
    """
    cc_orig = async_openai.chat.completions.create
    async def cc_create(*args, **kwargs):
        resp = await cc_orig(*args, **kwargs)
        _tally_usage(resp)
        if capture_path:
            _safe_capture(capture_path, call_id=str(uuid.uuid4()), tier=tier,
                          model=kwargs.get("model", ""), api="chat.completions",
                          kwargs=kwargs, resp=resp)
        return resp
    async_openai.chat.completions.create = cc_create

    responses = getattr(async_openai, "responses", None)
    for name in ("create", "parse"):
        fn = getattr(responses, name, None)
        if fn is None:
            continue
        def _wrap(orig, label: str):
            async def wrapped(*args, **kwargs):
                resp = await orig(*args, **kwargs)
                _tally_usage(resp)
                if capture_path:
                    _safe_capture(capture_path, call_id=str(uuid.uuid4()), tier=tier,
                                  model=kwargs.get("model", ""), api=f"responses.{label}",
                                  kwargs=kwargs, resp=resp)
                return resp
            return wrapped
        setattr(responses, name, _wrap(fn, name))
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
                       timeout: float = 180.0, max_retries: int = 3,
                       tier: str = "", capture_path: str = "") -> AsyncOpenAI:
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
                                    timeout=timeout, max_retries=max_retries),
                       tier=tier, capture_path=capture_path)
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
