"""The teacher transport and its answer store.

The distillation harness runs the production cheap-tier `OpenAIGenericClient`
(and every wrapper production puts around it -- `instrument`, OpenRouter
provider injection, `_bound_index_arrays`, `wrap_client`/`StaticTailLayout`,
etc.) against a fake `AsyncOpenAI`: `TeacherTransport`. By the time a call
reaches `TeacherTransport.chat.completions.create`, `kwargs["messages"]` and
`kwargs["response_format"]` are exactly the bytes production would have sent
to the cheap model -- which is also what the teacher (Claude Opus, via
subagents) is answering out of band. `AnswerStore` is the on-disk record of
that question/answer exchange:

- `root/answers.jsonl` -- append-only log of validated (key, answer, meta)
  triples. The in-memory index (`self._answers`) is rebuilt from it at
  construction, so a second harness process (or the same one, restarted)
  resumes exactly where the file left off.
- `root/pending/<key>.json` -- one file per request still waiting on an
  answer. `add_pending` is idempotent (same key -> same request, by
  construction of `key_for`), so two callers racing on the same unanswered
  prompt in the same round write the file once.
- `root/dropped.jsonl` -- append-only log of keys the validator gave up on
  (wrong schema twice, say). A dropped key must never silently re-enter the
  pending queue; replaying it raises `TeacherDropped` instead.

`CURRENT_PROMPT_NAME` (graph_extract.usage) is the default source for the
request's `prompt_name`: the dedup guard sets it around every graphiti LLM
call, so a pending request always carries the prompt type that produced it
even though `TeacherTransport` itself has no idea which graphiti prompt is
in flight -- it only ever sees the raw chat-completions kwargs.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from graph_extract.usage import CURRENT_PROMPT_NAME


class TeacherNeeded(Exception):
    """Raised when a prompt has no stored teacher answer yet.

    `request` is `{"prompt_name", "messages", "response_format", "model"}` --
    exactly what was written to `pending/<key>.json`.
    """

    def __init__(self, key: str, request: dict) -> None:
        super().__init__(f"no teacher answer for {key}")
        self.key = key
        self.request = request


class TeacherDropped(Exception):
    """Raised when a prompt's key was dropped by the validator (bad answer,
    twice). It never becomes pending again; the caller must treat the round
    as stuck on this prompt rather than re-asking the teacher."""

    def __init__(self, key: str, reason: str) -> None:
        super().__init__(f"{key} was dropped: {reason}")
        self.key = key
        self.reason = reason


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class AnswerStore:
    """The on-disk (key -> teacher answer) record, plus the pending queue.

    Not safe across processes without a shared filesystem; safe within one
    process against concurrent async callers (a `threading.Lock` serialises
    every append/remove, the same pattern as `usage._capture_lock`).
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.pending_dir = self.root / "pending"
        self.answers_path = self.root / "answers.jsonl"
        self.dropped_path = self.root / "dropped.jsonl"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._answers: dict[str, str] = {}
        self._dropped: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.answers_path.exists():
            for line in self.answers_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                self._answers[rec["key"]] = rec["answer"]
        if self.dropped_path.exists():
            for line in self.dropped_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                self._dropped[rec["key"]] = rec["reason"]

    def key_for(self, messages: list[dict], response_format: dict | None) -> str:
        canonical = _canonical_json({"messages": messages, "response_format": response_format})
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        return self._answers.get(key)

    def dropped_reason(self, key: str) -> str | None:
        return self._dropped.get(key)

    def _pending_path(self, key: str) -> Path:
        return self.pending_dir / f"{key}.json"

    def put(self, key: str, answer_json: str, meta: dict) -> None:
        record = {"key": key, "answer": answer_json, "meta": meta, "ts": _utcnow()}
        with self._lock:
            with open(self.answers_path, "a", buffering=1, encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
            self._answers[key] = answer_json
            self._pending_path(key).unlink(missing_ok=True)

    def add_pending(self, key: str, request: dict) -> None:
        path = self._pending_path(key)
        with self._lock:
            if path.exists():
                return
            path.write_text(json.dumps(request, sort_keys=True, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    def pending(self) -> list[tuple[str, dict]]:
        out = []
        for path in sorted(self.pending_dir.glob("*.json")):
            out.append((path.stem, json.loads(path.read_text(encoding="utf-8"))))
        return out

    def drop(self, key: str, reason: str) -> None:
        record = {"key": key, "reason": reason, "ts": _utcnow()}
        with self._lock:
            with open(self.dropped_path, "a", buffering=1, encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
            self._dropped[key] = reason
            self._pending_path(key).unlink(missing_ok=True)


def _fake_completion(answer_json: str, *, model: str) -> SimpleNamespace:
    """A chat.completions.create reply shaped like the real OpenAI SDK's.

    Attributes are exactly what `OpenAIGenericClient._generate_response` and
    `graph_extract.usage` (`_tally_usage`, `capture_llm_call`,
    `_completion_text`, `_finish_reason`) read off a response:
    `choices[0].message.content`, `choices[0].message.refusal` (unused by the
    generic client, read by the Responses-API client; provided anyway so a
    future mode swap does not break), `choices[0].finish_reason`, `usage.*`
    and `.model`.
    """
    message = SimpleNamespace(content=answer_json, refusal=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0,
                            prompt_tokens_details=None)
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


class TeacherTransport:
    """Fake AsyncOpenAI: `.chat.completions.create(**kwargs)` returns a stored
    teacher answer, or raises `TeacherNeeded`/`TeacherDropped`.

    Exposes `chat.completions.create` as a plain attribute (not a bound method
    looked up fresh each time) so every production wrapper that does
    `orig = client.chat.completions.create; client.chat.completions.create =
    new_fn` -- `instrument`, `_bound_index_arrays`, `wrap_client`, etc. -- can
    reassign it in place, exactly as it does on a real `AsyncOpenAI`.
    """

    MODEL = "claude-opus-teacher"

    def __init__(self, store: AnswerStore,
                prompt_name_getter: Callable[[], str] = CURRENT_PROMPT_NAME.get) -> None:
        self.store = store
        self.prompt_name_getter = prompt_name_getter
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages") or []
        response_format = kwargs.get("response_format")
        key = self.store.key_for(messages, response_format)

        reason = self.store.dropped_reason(key)
        if reason is not None:
            raise TeacherDropped(key, reason)

        answer = self.store.get(key)
        if answer is not None:
            return _fake_completion(answer, model=self.MODEL)

        request = {
            "prompt_name": self.prompt_name_getter(),
            "messages": messages,
            "response_format": response_format,
            "model": kwargs.get("model"),
        }
        self.store.add_pending(key, request)
        raise TeacherNeeded(key, request)

    async def close(self) -> None:
        """No-op: matches `AsyncOpenAI.close()`'s surface so `build_graphiti`'s
        close-everything-owned path (`graph_extract.graphiti_client.build_graphiti`)
        can call it unconditionally."""
