"""Unit tests for the semantic-job enqueue trigger in SyncCore._apply_record
(fakes only)."""
from __future__ import annotations

import pytest

from graph_sync.catalog import Catalog
from graph_sync.sync_core import BootstrapResult, IncrementalResult, SyncCore

pytestmark = pytest.mark.asyncio


class _FakeStore:
    def __init__(self) -> None:
        self.enqueue_calls: list[tuple[str, str, str | None, str, str | None]] = []

    async def enqueue_semantic_job(
        self, article_id: str, op: str, content_hash: str | None, lane: str,
        source_id: str | None = None,
    ) -> None:
        self.enqueue_calls.append((article_id, op, content_hash, lane, source_id))


class _FakeRepo:
    def __init__(self, *, existing_hash: str | None,
                 has_episodes: bool | dict[str, bool] = False) -> None:
        self._existing_hash = existing_hash
        # bool -> the same answer for every article_id; dict -> per-id (missing
        # ids default to False, i.e. "never extracted").
        self._has_episodes = has_episodes
        self.structural_calls: list = []
        self.incomplete_calls: list = []
        self.tombstone_calls: list = []
        self.has_episodes_calls: list[str] = []

    async def get_content_hash(self, article_id: str) -> str | None:
        return self._existing_hash

    async def has_episodes(self, article_id: str) -> bool:
        self.has_episodes_calls.append(article_id)
        if isinstance(self._has_episodes, dict):
            return self._has_episodes.get(article_id, False)
        return self._has_episodes

    async def apply_structural(self, write) -> None:
        self.structural_calls.append(write)

    async def apply_incomplete_article(self, article: dict) -> None:
        self.incomplete_calls.append(article)

    async def tombstone_article(self, tombstone) -> None:
        self.tombstone_calls.append(tombstone)


class _FakeEmptyCatalog:
    """Resolves nothing and no-ops on refresh (models an unknown source_id
    without requiring a real httpx client)."""

    def resolve(self, source_id: str):
        return None

    async def refresh(self) -> None:
        pass


def _catalog_with_source(source_id: str) -> Catalog:
    return Catalog.from_lists(
        vendors=[{"id": "v1", "name": "Vendor One", "website": None}],
        products=[{"id": "p1", "name": "Product One", "version": None, "vendor_id": "v1"}],
        sources=[{"id": source_id, "product_id": "p1", "name": "Source One",
                   "base_url": None, "source_type": None, "platform": None,
                   "last_extracted_at": None}],
    )


def _content_record(*, article_id: str, source_id: str, content_hash: str) -> dict:
    return {
        "change_type": "updated",
        "id": article_id,
        "topic_key": "topic-1",
        "source_id": source_id,
        "vendor": "Vendor One",
        "product": "Product One",
        "title": "A Title",
        "source_url": "https://example.com/a",
        "last_updated_at": None,
        "content_hash": content_hash,
        "estimated_tokens": 100,
        "parent_chapter": None,
        "top_level_chapter": None,
        "sort_order": 1,
        "run_id": "run-1",
        "seq": 1,
    }


def _tombstone_record(*, article_id: str, source_id: str) -> dict:
    return {
        "change_type": "removed",
        "id": article_id,
        "source_id": source_id,
        "removed_at": "2026-07-15T00:00:00Z",
        "run_id": "run-1",
        "seq": 1,
        "topic_key": "topic-1",
    }


async def test_content_change_enqueues_upsert():
    from graph_sync.models import parse_delta_line
    import json

    rec = parse_delta_line(json.dumps(
        _content_record(article_id="a1", source_id="s1", content_hash="hash-new")))
    store = _FakeStore()
    repo = _FakeRepo(existing_hash=None)  # new article -> no prior hash
    core = SyncCore(object(), _catalog_with_source("s1"), repo, store, object())

    res = BootstrapResult()
    await core._apply_record(rec, res)

    assert store.enqueue_calls == [("a1", "upsert", "hash-new", "bootstrap", "s1")]
    assert len(repo.structural_calls) == 1


async def test_incomplete_article_path_enqueues_upsert():
    from graph_sync.models import parse_delta_line
    import json

    rec = parse_delta_line(json.dumps(
        _content_record(article_id="a2", source_id="unknown-source", content_hash="hash-x")))
    store = _FakeStore()
    repo = _FakeRepo(existing_hash=None)
    core = SyncCore(object(), _FakeEmptyCatalog(), repo, store, object())

    res = BootstrapResult()
    await core._apply_record(rec, res)

    assert store.enqueue_calls == [("a2", "upsert", "hash-x", "bootstrap", "unknown-source")]
    assert len(repo.incomplete_calls) == 1


async def test_tombstone_enqueues_remove():
    from graph_sync.models import parse_delta_line
    import json

    rec = parse_delta_line(json.dumps(_tombstone_record(article_id="a3", source_id="s1")))
    store = _FakeStore()
    repo = _FakeRepo(existing_hash=None)
    core = SyncCore(object(), _catalog_with_source("s1"), repo, store, object())

    res = BootstrapResult()
    await core._apply_record(rec, res)

    assert store.enqueue_calls == [("a3", "remove", None, "bootstrap", "s1")]
    assert len(repo.tombstone_calls) == 1


async def test_bootstrap_lane_inferred():
    from graph_sync.models import parse_delta_line
    import json

    rec = parse_delta_line(json.dumps(
        _content_record(article_id="a5", source_id="s1", content_hash="hash-boot")))
    store = _FakeStore()
    repo = _FakeRepo(existing_hash=None)
    core = SyncCore(object(), _catalog_with_source("s1"), repo, store, object())

    res = BootstrapResult()
    await core._apply_record(rec, res)

    assert store.enqueue_calls == [("a5", "upsert", "hash-boot", "bootstrap", "s1")]


async def test_incremental_update_of_extracted_article_is_incremental_lane():
    """CLAUDE.md: the unbudgeted incremental lane is for updates to articles
    that already have episodes."""
    from graph_sync.models import parse_delta_line
    import json

    rec = parse_delta_line(json.dumps(
        _content_record(article_id="a6", source_id="s1", content_hash="hash-incr")))
    store = _FakeStore()
    repo = _FakeRepo(existing_hash="hash-old", has_episodes=True)
    core = SyncCore(object(), _catalog_with_source("s1"), repo, store, object())

    res = IncrementalResult()
    await core._apply_record(rec, res)

    assert store.enqueue_calls == [("a6", "upsert", "hash-incr", "incremental", "s1")]


async def test_incremental_new_article_is_bootstrap_lane():
    """A brand-new article on an incremental pull has no episodes yet -- it
    must go to the budgeted bootstrap lane, not the unbudgeted incremental
    one (the 5,646-job gap this change fixes)."""
    from graph_sync.models import parse_delta_line
    import json

    rec = parse_delta_line(json.dumps(
        _content_record(article_id="a7", source_id="s1", content_hash="hash-new")))
    store = _FakeStore()
    repo = _FakeRepo(existing_hash=None, has_episodes=False)  # new article
    core = SyncCore(object(), _catalog_with_source("s1"), repo, store, object())

    res = IncrementalResult()
    await core._apply_record(rec, res)

    assert store.enqueue_calls == [("a7", "upsert", "hash-new", "bootstrap", "s1")]


async def test_incremental_changed_article_never_extracted_is_bootstrap_lane():
    """A CHANGED (not new) article that was never semantically extracted --
    e.g. a vendor never bootstrapped -- still belongs in bootstrap, not
    incremental: `existing_hash` differs from the incoming one, but there are
    no episodes."""
    from graph_sync.models import parse_delta_line
    import json

    rec = parse_delta_line(json.dumps(
        _content_record(article_id="a8", source_id="s1", content_hash="hash-changed")))
    store = _FakeStore()
    repo = _FakeRepo(existing_hash="hash-stale", has_episodes=False)
    core = SyncCore(object(), _catalog_with_source("s1"), repo, store, object())

    res = IncrementalResult()
    await core._apply_record(rec, res)

    assert store.enqueue_calls == [("a8", "upsert", "hash-changed", "bootstrap", "s1")]


async def test_tombstone_of_extracted_article_is_incremental_lane():
    from graph_sync.models import parse_delta_line
    import json

    rec = parse_delta_line(json.dumps(_tombstone_record(article_id="a9", source_id="s1")))
    store = _FakeStore()
    repo = _FakeRepo(existing_hash=None, has_episodes=True)
    core = SyncCore(object(), _catalog_with_source("s1"), repo, store, object())

    res = IncrementalResult()
    await core._apply_record(rec, res)

    assert store.enqueue_calls == [("a9", "remove", None, "incremental", "s1")]


async def test_tombstone_of_unextracted_article_is_bootstrap_lane():
    """A remove for an article that was never extracted never touched
    Graphiti -- it belongs in bootstrap, same as any other never-extracted
    record."""
    from graph_sync.models import parse_delta_line
    import json

    rec = parse_delta_line(json.dumps(_tombstone_record(article_id="a10", source_id="s1")))
    store = _FakeStore()
    repo = _FakeRepo(existing_hash=None, has_episodes=False)
    core = SyncCore(object(), _catalog_with_source("s1"), repo, store, object())

    res = IncrementalResult()
    await core._apply_record(rec, res)

    assert store.enqueue_calls == [("a10", "remove", None, "bootstrap", "s1")]


async def test_unchanged_hash_does_not_enqueue():
    from graph_sync.models import parse_delta_line
    import json

    rec = parse_delta_line(json.dumps(
        _content_record(article_id="a4", source_id="s1", content_hash="same-hash")))
    store = _FakeStore()
    repo = _FakeRepo(existing_hash="same-hash")  # unchanged
    core = SyncCore(object(), _catalog_with_source("s1"), repo, store, object())

    res = BootstrapResult()
    await core._apply_record(rec, res)

    assert store.enqueue_calls == []
    assert len(repo.structural_calls) == 0
    assert len(repo.incomplete_calls) == 0


async def test_incremental_unchanged_hash_never_calls_has_episodes():
    """The lane is computed AFTER the hash gate for content records: an
    unchanged replay must never pay the `has_episodes` Neo4j round trip -- it
    is discarded by the hash gate before the lane is even needed."""
    from graph_sync.models import parse_delta_line
    import json

    rec = parse_delta_line(json.dumps(
        _content_record(article_id="a11", source_id="s1", content_hash="same-hash")))
    store = _FakeStore()
    repo = _FakeRepo(existing_hash="same-hash", has_episodes=True)  # unchanged
    core = SyncCore(object(), _catalog_with_source("s1"), repo, store, object())

    res = IncrementalResult()
    await core._apply_record(rec, res)

    assert store.enqueue_calls == []
    assert repo.has_episodes_calls == []
