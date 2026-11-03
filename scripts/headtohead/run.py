"""Ask every question of both systems and record both answers.

Graph side: the deployed answer-api (`GET /answer?q=`, production defaults) over a kubectl
port-forward. vendorkb side: `vendorkb.answer.answer`. Spends money (GLM, Voyage): run
`--limit 2` first.

Run: uv run --extra dev python -m scripts.headtohead.run \
    --questions eval/headtohead/questions.yaml --answer-api http://127.0.0.1:18100 \
    --out eval/headtohead/runs/<date> [--limit 2]
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from vendorkb.answer import REFUSAL
from vendorkb.config import VendorkbSettings

log = logging.getLogger(__name__)

SIDES = ("graph", "vendorkb")
ANSWER_API_TIMEOUT = 600.0
NAMESPACE = "graph-rag"
DEPLOY = "graph-rag-answer"
ENV_NAMES = ("JUDGE_MODEL", "RERANK_MODEL", "SYNTHESIS_REASONING_EFFORT")
SECRET_WORDS = ("key", "password", "secret", "token", "dsn")
GRAPH_STATE_PATH = Path("eval/headtohead/graph_state.json")


def _refusal(answer: str | None) -> bool:
    return answer is not None and answer.strip().startswith(REFUSAL)


async def ask_answer_api(client: httpx.AsyncClient, question: str) -> dict:
    """One `GET /answer?q=` with no mode/scope. Never raises: a failure sets `error`."""
    t0 = time.monotonic()
    out: dict[str, Any] = {"answer": None, "citations": [], "raw": None, "error": None}
    try:
        resp = await client.get("/answer", params={"q": question}, timeout=ANSWER_API_TIMEOUT)
        if resp.status_code != 200:
            out["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
        else:
            env = resp.json()
            out["raw"] = env
            answer = env.get("answer") if isinstance(env, dict) else None
            if not isinstance(answer, str) or not answer.strip():
                out["error"] = "empty answer in envelope"
            else:
                out["answer"] = answer
                out["citations"] = env.get("citations") or []
                if "mode" in env:
                    out["mode"] = env["mode"]
    except (httpx.HTTPError, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    out["latency_s"] = time.monotonic() - t0
    return out


async def _graph_side(client: httpx.AsyncClient, question: str) -> dict:
    t0 = time.monotonic()
    res = await ask_answer_api(client, question)
    if res["error"]:
        res = await ask_answer_api(client, question)
    env = res["raw"] if isinstance(res["raw"], dict) else {}
    return {
        "answer": res["answer"], "citations": res["citations"], "usage": None,
        "error": res["error"], "latency_s": time.monotonic() - t0,
        "flags": {"mode": env.get("mode"), "retrieved": env.get("retrieved"),
                  "cited": env.get("cited")},
    }


def _add_usage(total: dict | None, usage: dict | None) -> dict | None:
    """Sum prompt/completion tokens and cost over the attempts of one call."""
    if not usage:
        return total
    out = dict(total or {"prompt_tokens": 0, "completion_tokens": 0, "cost": None})
    for k in ("prompt_tokens", "completion_tokens"):
        out[k] = (out.get(k) or 0) + (usage.get(k) or 0)
    if usage.get("cost") is not None:
        out["cost"] = (out.get("cost") or 0.0) + usage["cost"]
    return out


class ConnPool:
    """`size` connections, one per concurrent vendorkb call: search_docs opens a
    transaction with SET LOCAL, which must never interleave on a shared connection."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[Any] = asyncio.Queue()
        self._all: list[Any] = []

    @classmethod
    async def create(cls, factory: Callable[[], Awaitable[Any]], size: int) -> ConnPool:
        pool = cls()
        try:
            for _ in range(size):
                conn = await factory()
                pool._all.append(conn)
                pool._q.put_nowait(conn)
        except BaseException:
            await pool.close()
            raise
        return pool

    @contextlib.asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        conn = await self._q.get()
        try:
            yield conn
        finally:
            self._q.put_nowait(conn)

    async def close(self) -> None:
        for conn in self._all:
            await conn.close()
        self._all.clear()


async def _vendorkb_side(fn: Callable[[str], Awaitable[dict]], question: str) -> dict:
    t0 = time.monotonic()
    res: dict[str, Any] = {}
    error: str | None = None
    usage_total: dict[str, Any] | None = None
    for _attempt in (1, 2):
        try:
            res = await fn(question)
            error = res.get("error")
            if not error and not (isinstance(res.get("answer"), str) and res["answer"].strip()):
                error = "empty answer"
        except Exception as e:  # recorded, never swallowed: it lands in `error`
            res, error = {}, f"{type(e).__name__}: {e}"
        usage_total = _add_usage(usage_total, res.get("usage"))
        if not error:
            break
    scope = res.get("scope") or {}
    return {
        "answer": None if error else res["answer"],
        "citations": res.get("citations") or [], "usage": usage_total,
        "error": error, "latency_s": time.monotonic() - t0,
        "flags": {"uncited": res.get("uncited"), "unknown_markers": res.get("unknown_markers"),
                  "scope_relaxed": scope.get("relaxed"), "rerank_ok": res.get("rerank_ok"),
                  "retrieved": res.get("retrieved")},
    }


def _load_done(path: Path) -> dict[tuple[str, str], dict]:
    done: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return done
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            if i == len(lines) - 1:
                log.warning("skipping torn final line in %s", path)
                continue
            raise
        if row.get("answer") is not None or row.get("error"):
            done[(row["qid"], row["side"])] = row
    return done


def summarize(rows: list[dict]) -> dict:
    summary: dict[str, Any] = {s: {"ok": 0, "errors": 0, "refusals": 0} for s in SIDES}
    for r in rows:
        c = summary[r["side"]]
        if r.get("answer") is None:
            c["errors"] += 1
        elif _refusal(r["answer"]):
            c["refusals"] += 1
        else:
            c["ok"] += 1
    return summary


async def run(
    questions: list[dict], *, answer_api: httpx.AsyncClient,
    vendorkb_answer: Callable[[str], Awaitable[dict]], out_dir: Path, limit: int | None,
    concurrency: int = 4, graph_state_check: Callable[[], Awaitable[None]],
    meta_collector: Callable[[], Awaitable[dict]],
) -> dict:
    await graph_state_check()  # before ANY request: refuses (raises) if the graph changed
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = await meta_collector()
    meta_path = out_dir / "run_meta.json"
    if meta_path.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        meta_path = out_dir / f"run_meta.resume-{stamp}.json"
    meta_path.write_text(
        json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    qs = questions[:limit] if limit is not None else list(questions)
    path = out_dir / "answers.jsonl"
    done = _load_done(path)
    if path.exists() and path.stat().st_size:
        with path.open("rb+") as fb:  # a torn last line must not swallow the next row
            fb.seek(-1, 2)
            if fb.read(1) != b"\n":
                fb.write(b"\n")
    final: dict[tuple[str, str], dict] = {
        k: v for k, v in done.items() if k[0] in {q["id"] for q in qs}}
    sem = asyncio.Semaphore(concurrency)
    with path.open("a", encoding="utf-8") as fh:
        async def one(q: dict, side: str) -> None:
            if (q["id"], side) in done:
                return
            async with sem:
                if side == "graph":
                    res = await _graph_side(answer_api, q["question"])
                else:
                    res = await _vendorkb_side(vendorkb_answer, q["question"])
            row = {"qid": q["id"], "side": side} | res
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            final[(q["id"], side)] = row

        await asyncio.gather(*(one(q, s) for q in qs for s in SIDES))
    rows = [final[(q["id"], s)] for q in qs for s in SIDES]
    summary = summarize(rows)
    summary["rows"] = rows
    return summary


def smoke_assert(rows: list[dict]) -> None:
    """Prove the mechanisms engaged before the full (paid) run."""
    problems = []
    for side in SIDES:
        if not any(r["side"] == side and r["answer"] is not None and not _refusal(r["answer"])
                   and r["citations"] for r in rows):
            problems.append(f"{side}: no non-refusal answer with >= 1 citation")
    for r in rows:
        if r["side"] == "vendorkb" and r["answer"] is not None:
            if r["flags"].get("rerank_ok") is not True:
                problems.append(f"vendorkb {r['qid']}: rerank_ok is not true")
    if problems:
        sys.exit("SMOKE CHECK FAILED: " + "; ".join(problems))


def print_smoke(rows: list[dict], total_questions: int) -> None:
    cost = 0.0
    n = len({r["qid"] for r in rows})
    for r in rows:
        line = (f"{r['qid']} {r['side']}: answer_len={len(r['answer'] or '')} "
                f"citations={len(r['citations'])} latency={r['latency_s']:.1f}s")
        u = r.get("usage")
        if r["side"] == "vendorkb" and u:
            line += (f" tokens={u.get('prompt_tokens')}+{u.get('completion_tokens')}"
                     f" cost={u.get('cost')}")
            cost += u.get("cost") or 0.0
        if r.get("error"):
            line += f" ERROR={r['error']}"
        print(line)
    print(f"vendorkb cost so far: {cost:.4f}")
    if n:
        print(f"projection for {total_questions} questions: {cost / n * total_questions:.4f}")


def settings_dump(settings: VendorkbSettings) -> dict:
    return {k: v for k, v in settings.model_dump().items()
            if not any(w in k.lower() for w in SECRET_WORDS)}


async def _kubectl(*args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "-n", NAMESPACE, *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(err.decode(errors="replace").strip()[:200])
    return out.decode().strip()


async def kubectl_meta(settings: VendorkbSettings) -> dict:
    meta: dict[str, Any] = {}
    try:
        image = await _kubectl(
            "get", "deploy", DEPLOY, "-o",
            "jsonpath={.spec.template.spec.containers[0].image}")
        meta["answer_api_image"] = image
    except RuntimeError as e:
        print(f"WARNING: could not read answer-api image: {e}", file=sys.stderr)
        meta["answer_api_image"] = None
    for name in ENV_NAMES:  # one named var per call, never a bare env/printenv
        try:
            meta[name] = await _kubectl("exec", f"deploy/{DEPLOY}", "--", "printenv", name)
        except RuntimeError as e:
            print(f"WARNING: could not read {name}: {e}", file=sys.stderr)
            meta[name] = None
    meta["vendorkb_settings"] = settings_dump(settings)
    try:
        meta["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        meta["git_commit"] = None
    print(json.dumps(meta, indent=2, default=str))
    return meta


async def real_graph_state_check() -> None:
    from scripts.headtohead.export_corpus import read_max_episode_created_at
    frozen = json.loads(GRAPH_STATE_PATH.read_text(encoding="utf-8"))["max_episode_created_at"]
    live = await read_max_episode_created_at()
    if str(live) != str(frozen):
        raise RuntimeError(
            f"graph changed since export: max(Episodic.created_at) {frozen} -> {live}")


async def _amain(args: argparse.Namespace) -> int:
    from openai import AsyncOpenAI

    from scripts.headtohead.questions import load_questions
    from vendorkb.answer import answer
    from vendorkb.clients import Embedder, Reranker
    from vendorkb.db import connect
    from vendorkb.scope import scope_index_from_db

    settings = VendorkbSettings()
    questions = load_questions(Path(args.questions))
    pool: ConnPool | None = None
    client: Any = None
    try:
        client = AsyncOpenAI(base_url=settings.judge_base_url, api_key=settings.judge_api_key)
        pool = await ConnPool.create(
            lambda: connect(settings.vendorkb_dsn), args.concurrency)
        async with pool.acquire() as conn:
            scope_index = await scope_index_from_db(conn)
        async with (Embedder(settings) as embedder, Reranker(settings) as reranker,
                    httpx.AsyncClient(base_url=args.answer_api) as api):
            async def vk(q: str) -> dict:
                assert pool is not None
                async with pool.acquire() as conn:
                    return await answer(conn, q, settings=settings, embedder=embedder,
                                        reranker=reranker, scope_index=scope_index,
                                        client=client)

            async def meta() -> dict:
                return await kubectl_meta(settings)

            summary = await run(
                questions, answer_api=api, vendorkb_answer=vk, out_dir=Path(args.out),
                limit=args.limit, concurrency=args.concurrency,
                graph_state_check=real_graph_state_check, meta_collector=meta)
    finally:
        if client is not None:
            await client.close()
        if pool is not None:
            await pool.close()
    rows = summary.pop("rows")
    print(json.dumps(summary, indent=2))
    if args.limit is not None:
        print_smoke(rows, len(questions))
        smoke_assert(rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", default="eval/headtohead/questions.yaml")
    ap.add_argument("--answer-api", default="http://127.0.0.1:18100")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    return asyncio.run(_amain(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
