"""SyncCore.bootstrap survives a dropped stream (BACKLOG 49's ReadTimeout).

CLIENT-USAGE-GUIDE §6: on a dropped stream, resume with
`?bootstrap_after=<highest applied id>`, keeping the ORIGINAL watermark. A transport
error mid-stream IS a dropped stream -- it used to raise straight out of `bootstrap`
with no progress row, so every re-run replayed from the first id and a server that
stalls at the same point could never be passed (the 2026-09-25 Veeam vendor
bootstrap died after 47 min and 9,254 records).
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from graph_sync.catalog import Catalog
from graph_sync.sync_core import SyncCore

pytestmark = pytest.mark.asyncio
FIX = Path(__file__).parent.parent / "fixtures"
_LINES = [ln for ln in FIX.joinpath("aws_delta.ndjson").read_text().splitlines() if ln.strip()]
_START = _LINES[0]                                   # bootstrap_start control line
_RECORDS = [ln for ln in _LINES[1:] if '"control"' not in ln][:10]
_IDS = [json.loads(ln)["id"] for ln in _RECORDS]
_TERMINAL = next(ln for ln in _LINES if '"control":"cursor"' in ln)
_WATERMARK = json.loads(_START)["next_since"]


def _cursor(seq: int) -> str:
    """An opaque cursor in DocExtractor's real encoding (the watermark maths
    decodes it)."""
    import base64
    return base64.b64encode(json.dumps({"seq": seq, "v": 1}).encode()).decode()


_ORIGINAL, _NEWER = _cursor(40000), _cursor(50000)


def _catalog() -> Catalog:
    def load(n):
        return json.loads(FIX.joinpath(n).read_text())
    return Catalog.from_lists(load("vendors.json")["vendors"], load("products.json")["products"],
                              load("sources.json")["sources"])


class _Store:
    def __init__(self, progress: dict | None = None) -> None:
        self.progress: dict[str, dict] = {} if progress is None else progress
        self.upserts: list[tuple] = []
        self.enqueued: list[str] = []
        self.cursor: str | None = None

    async def get_bootstrap(self, shard):
        return self.progress.get(shard)

    async def upsert_bootstrap(self, shard, watermark, last_id, status):
        self.upserts.append((shard, watermark, last_id, status))
        self.progress[shard] = {"shard": shard, "watermark": watermark,
                                "last_id": last_id, "status": status}

    async def all_bootstrap_watermarks(self):
        return [p["watermark"] for p in self.progress.values() if p["watermark"]]

    async def get_cursor(self):
        return self.cursor

    async def set_cursor(self, cursor):
        self.cursor = cursor

    async def enqueue_semantic_job(self, article_id, op, content_hash, lane, source_id=None):
        self.enqueued.append(article_id)

    async def has_semantic_job(self, article_id):
        return True


class _Repo:
    def __init__(self) -> None:
        self.applied: list[str] = []

    async def get_content_hash(self, article_id):
        return None

    async def apply_structural(self, write):
        self.applied.append(write.article["id"])

    async def has_episodes(self, article_id):
        return False


def _body(lines: list[str], *, stall_after: int | None = None):
    """An NDJSON body; with `stall_after`, raise ReadTimeout after that many lines
    -- the server going silent mid-stream."""
    async def gen():
        for i, ln in enumerate(lines):
            if stall_after is not None and i == stall_after:
                raise httpx.ReadTimeout("server sent nothing for 300 s")
            yield (ln + "\n").encode()
    return gen()


def _core(responses: list, requests: list, store: _Store, repo: _Repo) -> SyncCore:
    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(dict(req.url.params))
        return httpx.Response(200, content=responses.pop(0)())
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://x")
    core = SyncCore(client, _catalog(), repo, store, object())
    core._sleep = _no_sleep                      # type: ignore[method-assign]

    async def _no_toc(sid):
        return None
    core.refresh_toc = _no_toc                   # type: ignore[method-assign]
    return core


async def _no_sleep(_s: float) -> None:
    return None


async def test_a_mid_stream_timeout_resumes_from_the_last_applied_id():
    store, repo, requests = _Store(), _Repo(), []
    responses = [
        lambda: _body([_START] + _RECORDS, stall_after=5),        # start + 4 records, then stall
        # The resumed stream announces its own, NEWER bootstrap_start: it must not
        # be adopted (CLIENT-USAGE-GUIDE §6 keeps the original watermark).
        lambda: _body([json.dumps({"control": "bootstrap_start", "next_since": _NEWER})]
                      + _RECORDS[4:] + [_TERMINAL]),
    ]
    core = _core(responses, requests, store, repo)

    res = await core.bootstrap(source_id="21632f3b-5a4c-4c93-9f00-6701d0e9f677")

    assert "bootstrap_after" not in requests[0]
    assert requests[1]["bootstrap_after"] == _IDS[3]
    assert repo.applied == _IDS, "every record applied exactly once, in order"
    assert res.applied == 10
    final = store.progress["21632f3b-5a4c-4c93-9f00-6701d0e9f677"]
    assert final["status"] == "complete" and final["watermark"] == _WATERMARK


async def test_repeated_stalls_without_progress_give_up_loudly():
    store, repo, requests = _Store(), _Repo(), []
    responses = [lambda: _body([_START] + _RECORDS, stall_after=3)] + [
        lambda: _body([_START] + _RECORDS, stall_after=1)] * 10   # no record ever again
    core = _core(responses, requests, store, repo)
    core.BOOTSTRAP_MAX_STALLS = 3

    with pytest.raises(httpx.ReadTimeout):
        await core.bootstrap(source_id="21632f3b-5a4c-4c93-9f00-6701d0e9f677")

    # 1 attempt with progress + 3 consecutive attempts without any.
    assert len(requests) == 4
    progress = store.progress["21632f3b-5a4c-4c93-9f00-6701d0e9f677"]
    assert progress["status"] == "in_progress" and progress["last_id"] == _IDS[1]


async def test_a_new_run_resumes_an_in_progress_shard_with_its_original_watermark():
    """Crash-resume across processes: the next `bootstrap --source-id` continues
    from the stored `last_id` instead of replaying the shard from the start."""
    shard = "21632f3b-5a4c-4c93-9f00-6701d0e9f677"
    store = _Store({shard: {"shard": shard, "watermark": _ORIGINAL, "last_id": _IDS[6],
                            "status": "in_progress"}})
    repo, requests = _Repo(), []
    responses = [lambda: _body([json.dumps({"control": "bootstrap_start", "next_since": _NEWER})]
                               + _RECORDS[7:] + [_TERMINAL])]
    core = _core(responses, requests, store, repo)

    await core.bootstrap(source_id=shard)

    assert requests[0]["bootstrap_after"] == _IDS[6]
    assert repo.applied == _IDS[7:]
    assert store.progress[shard] == {"shard": shard, "watermark": _ORIGINAL,
                                     "last_id": _IDS[-1], "status": "complete"}


async def test_a_complete_shard_is_replayed_from_the_start():
    """A re-run of a COMPLETE shard is a deliberate repair (the hash gate makes it
    cheap, and it is how BACKLOG 50's requeue runs): never resumed."""
    shard = "21632f3b-5a4c-4c93-9f00-6701d0e9f677"
    store = _Store({shard: {"shard": shard, "watermark": _ORIGINAL, "last_id": _IDS[-1],
                            "status": "complete"}})
    repo, requests = _Repo(), []
    core = _core([lambda: _body([_START] + _RECORDS + [_TERMINAL])], requests, store, repo)

    await core.bootstrap(source_id=shard)

    assert "bootstrap_after" not in requests[0]
    assert repo.applied == _IDS


async def test_progress_is_saved_while_the_stream_runs():
    """A process killed mid-stream (OOM, eviction, deploy) must not lose everything
    since the stream began: progress is written every BOOTSTRAP_PROGRESS_EVERY
    applied records, as `in_progress`, with the watermark already known."""
    store, repo, requests = _Store(), _Repo(), []

    def boom():
        async def gen():
            for ln in [_START] + _RECORDS[:7]:
                yield (ln + "\n").encode()
            raise RuntimeError("process killed")        # not a transport error
        return gen()
    core = _core([boom], requests, store, repo)
    core.BOOTSTRAP_PROGRESS_EVERY = 3

    with pytest.raises(RuntimeError):
        await core.bootstrap(source_id="21632f3b-5a4c-4c93-9f00-6701d0e9f677")

    assert store.progress["21632f3b-5a4c-4c93-9f00-6701d0e9f677"] == {
        "shard": "21632f3b-5a4c-4c93-9f00-6701d0e9f677", "watermark": _WATERMARK,
        "last_id": _IDS[5], "status": "in_progress"}


def _status_core(statuses: list[int], requests: list, store: _Store, repo: _Repo) -> SyncCore:
    """Each request answers with the next status; a 200 serves the whole stream."""
    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(dict(req.url.params))
        code = statuses.pop(0)
        if code != 200:
            return httpx.Response(code, text="upstream unhappy")
        return httpx.Response(200, content=_body([_START] + _RECORDS + [_TERMINAL]))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://x")
    core = SyncCore(client, _catalog(), repo, store, object())
    core._sleep = _no_sleep                      # type: ignore[method-assign]

    async def _no_toc(sid):
        return None
    core.refresh_toc = _no_toc                   # type: ignore[method-assign]
    return core


@pytest.mark.parametrize("code", [429, 502, 503, 504])
async def test_a_transient_http_status_is_retried(code):
    """raise_for_status() raises HTTPStatusError, which is NOT a TransportError: a
    502 from the ingress mid-bootstrap must be retried like a dropped stream."""
    store, repo, requests = _Store(), _Repo(), []
    core = _status_core([code, 200], requests, store, repo)

    res = await core.bootstrap(source_id="21632f3b-5a4c-4c93-9f00-6701d0e9f677")

    assert len(requests) == 2 and res.applied == 10


@pytest.mark.parametrize("code", [401, 403, 404])
async def test_a_client_error_fails_immediately(code):
    """A bad key or a wrong source id will not fix itself: no retry, no backoff."""
    store, repo, requests = _Store(), _Repo(), []
    core = _status_core([code, 200], requests, store, repo)

    with pytest.raises(httpx.HTTPStatusError):
        await core.bootstrap(source_id="21632f3b-5a4c-4c93-9f00-6701d0e9f677")

    assert len(requests) == 1


async def test_repeated_clean_eof_without_records_gives_up():
    """A body that ends cleanly with no records and no terminal line (e.g. a
    connection-close-terminated response cut early) never raises: the stall
    counter alone must stop it."""
    store, repo, requests = _Store(), _Repo(), []
    responses = [lambda: _body([_START])] * 10
    core = _core(responses, requests, store, repo)
    core.BOOTSTRAP_MAX_STALLS = 3

    with pytest.raises(RuntimeError, match="terminal cursor line"):
        await core.bootstrap(source_id="21632f3b-5a4c-4c93-9f00-6701d0e9f677")

    assert len(requests) == 3


async def test_a_resumed_source_shard_refreshes_that_sources_toc():
    """Articles applied before the crash are not 'touched' by the resuming run, so
    without this the TOC pass would skip their source entirely."""
    shard = "21632f3b-5a4c-4c93-9f00-6701d0e9f677"
    store = _Store({shard: {"shard": shard, "watermark": _ORIGINAL, "last_id": _IDS[-1],
                            "status": "in_progress"}})
    repo, requests = _Repo(), []
    core = _core([lambda: _body([_START, _TERMINAL])], requests, store, repo)
    refreshed: list[str] = []

    async def _toc(sid):
        refreshed.append(sid)
    core.refresh_toc = _toc                      # type: ignore[method-assign]

    await core.bootstrap(source_id=shard)

    assert refreshed == [shard]
