"""Ask every question of both systems and record both answers.

Graph side: the deployed answer-api (`GET /answer?q=`, production defaults) over a kubectl
port-forward. vendorkb side: `vendorkb.answer.answer`. Spends money (GLM, Voyage): run
`--limit 2` first. A `--limit` run writes into `<out>/smoke/` and, when its smoke check
passes, leaves `<out>/smoke/SMOKE_PASSED`; a full run (no `--limit`) refuses without it.

Before any request a run refuses (exit 2) when the graph changed since export, the vendorkb
index is incomplete, or the pod's models differ from vendorkb's. It aborts (exit 3) after
MAX_CONSECUTIVE_ERRORS consecutive errors on one side. `--retry-errors` re-asks the pairs
whose last row is an error.

Run: uv run --extra dev python -m scripts.headtohead.run \
    --questions eval/headtohead/questions.yaml --answer-api http://127.0.0.1:18100 \
    --out eval/headtohead/runs/<date> [--limit 2] [--retry-errors]
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

from answer_api.drift import _REFUSAL as _DRIFT_REFUSAL
from answer_api.global_search import _REFUSAL as _GLOBAL_REFUSAL
from answer_api.synthesize import _REFUSAL as _SYNTH_REFUSAL
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
DEFAULT_CORPUS = Path("eval/headtohead/corpus.json")
DEFAULT_INGEST_REPORT = Path("eval/headtohead/ingest_report.json")
# The frozen-graph markers compared before a run (graph_state.json, export_corpus).
FROZEN_KEYS = ("max_episode_created_at", "max_community_generated_at", "community_count",
               "stale_community_count")
# answer-api's diagnostics kept in a graph row's flags (None when absent). The raw envelope
# itself is never written: answers.jsonl would grow by the full fact lists.
GRAPH_DIAG_KEYS = ("degraded", "degraded_facets", "relevance_unchecked", "scope",
                   "applies_to", "item_grades", "checked_vendors")
RETRY_BACKOFF_S = 5.0  # ruling R16: wait before the second attempt
MAX_CONSECUTIVE_ERRORS = 5
SMOKE_DIR = "smoke"
SMOKE_MARKER = "SMOKE_PASSED"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
KEY_USAGE_SETTLE_S = 15.0  # OpenRouter books usage asynchronously
# printenv's exit status for an unset variable, as kubectl exec reports it
UNSET_MARK = "exit code 1"
# Every refusal wording answer-api produces (local/list/enumeration, global, DRIFT) and
# vendorkb's (identical to answer-api's local one). Read as data from answer_api.
REFUSALS = tuple(dict.fromkeys((REFUSAL, _SYNTH_REFUSAL, _GLOBAL_REFUSAL, _DRIFT_REFUSAL)))

_sleep = asyncio.sleep  # module attribute so tests record backoffs instead of waiting


class RunRefused(RuntimeError):
    """A pre-run check failed: no request was made."""


class CircuitBreakerOpen(RuntimeError):
    """Too many consecutive errors on one side: the run was aborted."""


def _refusal(answer: str | None) -> bool:
    return answer is not None and answer.strip().startswith(REFUSALS)


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


def graph_flags(env: dict) -> dict:
    flags = {"mode": env.get("mode"), "retrieved": env.get("retrieved"),
             "cited": env.get("cited")}
    return flags | {k: env.get(k) for k in GRAPH_DIAG_KEYS}


async def _graph_side(client: httpx.AsyncClient, question: str) -> dict:
    t0 = time.monotonic()
    res = await ask_answer_api(client, question)
    if res["error"]:
        await _sleep(RETRY_BACKOFF_S)
        res = await ask_answer_api(client, question)
    env = res["raw"] if isinstance(res["raw"], dict) else {}
    return {
        "answer": res["answer"], "citations": res["citations"], "usage": None,
        "error": res["error"], "latency_s": time.monotonic() - t0,
        "flags": graph_flags(env),
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
    for attempt in (1, 2):
        if attempt == 2:
            await _sleep(RETRY_BACKOFF_S)
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
                  "retrieved": res.get("retrieved"),
                  "scope": {"vendors": scope.get("vendors"), "products": scope.get("products")}},
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
    summary["vendorkb"] |= {"rerank_not_ok": 0, "uncited": 0}
    for r in rows:
        c = summary[r["side"]]
        if r.get("answer") is None:
            c["errors"] += 1
        elif _refusal(r["answer"]):
            c["refusals"] += 1
        else:
            c["ok"] += 1
        if r["side"] == "vendorkb":
            flags = r.get("flags") or {}
            c["rerank_not_ok"] += flags.get("rerank_ok") is False
            c["uncited"] += flags.get("uncited") is True
    return summary


def smoke_dir(out_dir: Path) -> Path:
    return out_dir / SMOKE_DIR


def write_smoke_marker(directory: Path, git_commit: str | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / SMOKE_MARKER).write_text(json.dumps(
        {"timestamp": datetime.now(UTC).isoformat(), "git_commit": git_commit}) + "\n",
        encoding="utf-8")


async def run(
    questions: list[dict], *, answer_api: httpx.AsyncClient,
    vendorkb_answer: Callable[[str], Awaitable[dict]], out_dir: Path, limit: int | None,
    concurrency: int = 4, graph_state_check: Callable[[], Awaitable[None]],
    index_check: Callable[[], Awaitable[None]], parity_check: Callable[[dict], None],
    meta_collector: Callable[[], Awaitable[dict]], retry_errors: bool = False,
) -> dict:
    """Every check runs before ANY request and raises (RunRefused) to refuse. A `limit`
    (smoke) run writes under `<out_dir>/smoke/`; a full run needs its SMOKE_PASSED."""
    if limit is not None:
        out_dir = smoke_dir(out_dir)
    elif not (smoke_dir(out_dir) / SMOKE_MARKER).exists():
        raise RunRefused(f"no {smoke_dir(out_dir) / SMOKE_MARKER}: run with --limit 2 first "
                         "and let its smoke check pass")
    await graph_state_check()
    await index_check()
    meta = await meta_collector()
    parity_check(meta)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    consecutive = dict.fromkeys(SIDES, 0)
    tripped: list[str] = []
    with path.open("a", encoding="utf-8") as fh:
        async def one(q: dict, side: str) -> None:
            prev = done.get((q["id"], side))
            if prev is not None and not (retry_errors and prev.get("answer") is None):
                return
            async with sem:
                if tripped:  # the breaker opened while this pair waited: never started
                    return
                if side == "graph":
                    res = await _graph_side(answer_api, q["question"])
                else:
                    res = await _vendorkb_side(vendorkb_answer, q["question"])
            row = {"qid": q["id"], "side": side} | res
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            final[(q["id"], side)] = row
            consecutive[side] = consecutive[side] + 1 if res["error"] else 0
            if consecutive[side] >= MAX_CONSECUTIVE_ERRORS and not tripped:
                tripped.append(f"{side}: {consecutive[side]} consecutive errors, last "
                               f"{res['error']!r}; aborted (in-flight requests finished, "
                               f"rows kept in {path})")

        await asyncio.gather(*(one(q, s) for q in qs for s in SIDES))
    if tripped:
        raise CircuitBreakerOpen(tripped[0])
    rows = [final[(q["id"], s)] for q in qs for s in SIDES]
    summary = summarize(rows)
    summary["rows"] = rows
    return summary


def smoke_assert(rows: list[dict]) -> None:
    """Prove the mechanisms engaged before the full (paid) run. Every row must be
    error-free."""
    problems = [f"{r['side']} {r['qid']}: errored ({r.get('error')})" for r in rows
                if r.get("answer") is None or r.get("error")]
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


async def openrouter_key_usage(client: httpx.AsyncClient, api_key: str) -> float | None:
    """The key's cumulative OpenRouter usage in USD, or None when it cannot be read.
    The key goes only into the Authorization header; it is never printed."""
    if not api_key:
        return None
    try:
        resp = await client.get(OPENROUTER_KEY_URL, timeout=30.0,
                                headers={"Authorization": f"Bearer {api_key}"})
        if resp.status_code != 200:
            return None
        usage = resp.json()["data"]["usage"]
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return None
    if isinstance(usage, bool) or not isinstance(usage, (int, float)):
        return None
    return float(usage)


def cost_lines(before: float | None, after: float | None, *, vendorkb_cost: float,
               n_questions: int, total_questions: int) -> list[str]:
    """Ruling R18: the graph side's cost is not in its rows; it is measured as the
    OpenRouter key-usage delta over the smoke run."""
    lines = []
    if before is None or after is None:
        lines.append("graph side cost NOT measured (OpenRouter key usage could not be read)")
    else:
        delta = after - before
        lines.append(
            f"OpenRouter key usage delta: ${delta:.4f} (covers graph-side GLM calls only if "
            "the answer-api pod uses the same key; vendorkb GLM measured separately: "
            f"${vendorkb_cost:.4f})")
        graph = delta - vendorkb_cost  # vendorkb's answer client uses this same key
        lines.append(f"graph side if the pod shares the key (delta - vendorkb): ${graph:.4f}")
        if n_questions:
            lines.append(f"graph side projection for {total_questions} questions: "
                         f"${graph / n_questions * total_questions:.4f}")
    lines.append("Voyage rerank cost is not measurable here (neither side reports it); "
                 "it is not included in any figure above.")
    return lines


def vendorkb_cost(rows: list[dict]) -> float:
    return sum((r.get("usage") or {}).get("cost") or 0.0 for r in rows
               if r["side"] == "vendorkb")


def check_index_counts(*, documents: int, corpus_n: int, failed_n: int) -> None:
    expected = corpus_n - failed_n
    if documents != expected:
        raise RunRefused(
            f"vendorkb index incomplete: count(documents)={documents}, expected {expected} "
            f"(corpus {corpus_n} - ingest failures {failed_n})")


def expected_documents(corpus: Path, ingest_report: Path) -> tuple[int, int]:
    """(corpus size, ingest failures); failures are "id: reason" entries."""
    for p in (corpus, ingest_report):
        if not p.exists():
            raise RunRefused(f"missing {p}: cannot check the vendorkb index is complete")
    entries = json.loads(corpus.read_text(encoding="utf-8"))
    failed = json.loads(ingest_report.read_text(encoding="utf-8"))["failed"]
    return len(entries), len(failed)


def check_parity(meta: dict, settings: VendorkbSettings) -> None:
    """The pod's answer model, its reasoning effort and its reranker must equal vendorkb's.
    An unset SYNTHESIS_REASONING_EFFORT is answer-api's default, "low"; any other value
    that could not be read refuses."""
    effort = meta.get("SYNTHESIS_REASONING_EFFORT")
    if effort is None and "SYNTHESIS_REASONING_EFFORT" in (meta.get("unset_env") or []):
        effort = "low"
    pairs = (("JUDGE_MODEL", meta.get("JUDGE_MODEL"), settings.judge_model),
             ("SYNTHESIS_REASONING_EFFORT", effort, settings.answer_reasoning_effort),
             ("RERANK_MODEL", meta.get("RERANK_MODEL"), settings.rerank_model))
    bad = [f"{name}: pod={pod!r} vendorkb={own!r}" for name, pod, own in pairs
           if pod is None or pod != own]
    if bad:
        raise RunRefused("model parity check failed: " + "; ".join(bad))


def compare_graph_state(frozen: dict, live: dict) -> None:
    diffs = [f"{k} {frozen.get(k)} -> {live.get(k)}" for k in FROZEN_KEYS
             if str(frozen.get(k)) != str(live.get(k))]
    if diffs:
        raise RunRefused("graph changed since export: " + "; ".join(diffs))


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
    unset: list[str] = []
    for name in ENV_NAMES:  # one named var per call, never a bare env/printenv
        try:
            meta[name] = await _kubectl("exec", f"deploy/{DEPLOY}", "--", "printenv", name)
        except RuntimeError as e:
            meta[name] = None
            if UNSET_MARK in str(e):
                unset.append(name)
            else:
                print(f"WARNING: could not read {name}: {e}", file=sys.stderr)
    meta["unset_env"] = unset
    meta["vendorkb_settings"] = settings_dump(settings)
    meta["git_commit"] = git_commit()
    print(json.dumps(meta, indent=2, default=str))
    return meta


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


async def real_graph_state_check() -> None:
    from scripts.headtohead.export_corpus import read_graph_state
    frozen = json.loads(GRAPH_STATE_PATH.read_text(encoding="utf-8"))
    compare_graph_state(frozen, await read_graph_state(FROZEN_KEYS))


async def _run_all(args: argparse.Namespace) -> int:
    from openai import AsyncOpenAI

    from scripts.headtohead.questions import load_questions
    from vendorkb.answer import answer
    from vendorkb.clients import Embedder, Reranker
    from vendorkb.db import connect
    from vendorkb.scope import scope_index_from_db

    settings = VendorkbSettings()
    questions = load_questions(Path(args.questions))
    smoke = args.limit is not None
    out = Path(args.out)
    pool: ConnPool | None = None
    client: Any = None
    usage_before: float | None = None
    usage_after: float | None = None
    try:
        client = AsyncOpenAI(base_url=settings.judge_base_url, api_key=settings.judge_api_key)
        pool = await ConnPool.create(
            lambda: connect(settings.vendorkb_dsn), args.concurrency)
        async with pool.acquire() as conn:
            scope_index = await scope_index_from_db(conn)

        async def index_check() -> None:
            corpus_n, failed_n = expected_documents(Path(args.corpus), Path(args.ingest_report))
            assert pool is not None
            async with pool.acquire() as conn:
                cur = await conn.execute("SELECT count(*) FROM documents")
                row = await cur.fetchone()
            check_index_counts(documents=row[0] if row else 0, corpus_n=corpus_n,
                               failed_n=failed_n)

        async with (Embedder(settings) as embedder, Reranker(settings) as reranker,
                    httpx.AsyncClient(base_url=args.answer_api) as api,
                    httpx.AsyncClient() as plain):
            async def vk(q: str) -> dict:
                assert pool is not None
                async with pool.acquire() as conn:
                    return await answer(conn, q, settings=settings, embedder=embedder,
                                        reranker=reranker, scope_index=scope_index,
                                        client=client)

            async def meta() -> dict:
                return await kubectl_meta(settings)

            if smoke:
                usage_before = await openrouter_key_usage(plain, settings.judge_api_key)
            summary = await run(
                questions, answer_api=api, vendorkb_answer=vk, out_dir=out,
                limit=args.limit, concurrency=args.concurrency,
                graph_state_check=real_graph_state_check, index_check=index_check,
                parity_check=lambda m: check_parity(m, settings), meta_collector=meta,
                retry_errors=args.retry_errors)
            if smoke and usage_before is not None:
                await _sleep(KEY_USAGE_SETTLE_S)
                usage_after = await openrouter_key_usage(plain, settings.judge_api_key)
    finally:
        if client is not None:
            await client.close()
        if pool is not None:
            await pool.close()
    rows = summary.pop("rows")
    print(json.dumps(summary, indent=2))
    if smoke:
        print_smoke(rows, len(questions))
        for line in cost_lines(usage_before, usage_after, vendorkb_cost=vendorkb_cost(rows),
                               n_questions=len({r["qid"] for r in rows}),
                               total_questions=len(questions)):
            print(line)
        smoke_assert(rows)
        write_smoke_marker(smoke_dir(out), git_commit())
        print(f"smoke passed: {smoke_dir(out) / SMOKE_MARKER}")
    return 0


async def _amain(args: argparse.Namespace) -> int:
    try:
        return await _run_all(args)
    except RunRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    except CircuitBreakerOpen as e:
        print(f"ABORTED: {e}", file=sys.stderr)
        return 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", default="eval/headtohead/questions.yaml")
    ap.add_argument("--answer-api", default="http://127.0.0.1:18100")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-ask the pairs whose last row is an error")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--ingest-report", default=str(DEFAULT_INGEST_REPORT))
    return asyncio.run(_amain(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
