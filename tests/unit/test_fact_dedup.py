"""Near-duplicate collapse keeps every source (spec 2026-10-15 §4.3, §6)."""
import random
import time

import pytest

import answer_api.fact_dedup as fd


def _r(uuid, fact, url, vendor="Veeam"):
    return {"fact": fact, "fact_uuid": uuid, "valid_at": None, "invalid_at": None,
            "sources": [{"url": url, "title": url, "vendor": vendor, "product": "VBR"}]}


class _Driver:
    def __init__(self, vectors):
        self.vectors, self.calls = vectors, 0

    async def execute_query(self, query, **params):
        self.calls += 1
        assert query is fd.EMBEDDINGS_QUERY
        return ([{"uuid": u, "embedding": self.vectors.get(u)} for u in params["uuids"]],
                None, None)


def test_normalize_fact_ignores_case_punctuation_and_spacing():
    assert fd.normalize_fact("Veeam  Agent, provides X.") == fd.normalize_fact("veeam agent provides x")


async def test_identical_text_collapses_and_merges_sources():
    rs = [_r("a", "Veeam Agent provides instant restore to Hyper-V VMs.", "u1"),
          _r("b", "veeam agent provides instant restore to hyper-v vms", "u2")]
    out = await fd.collapse_near_duplicates(_Driver({}), rs, cosine=0.92)
    assert [o["fact_uuid"] for o in out] == ["a"]
    assert [s["url"] for s in out[0]["sources"]] == ["u1", "u2"]


async def test_near_identical_embeddings_collapse_distinct_ones_stay():
    rs = [_r("a", "Instant Recovery restores an agent computer as a Hyper-V VM.", "u1"),
          _r("b", "Agent computers can be restored by Instant Recovery to a Hyper-V VM.", "u2"),
          _r("c", "Restore to Amazon EC2 is supported.", "u3")]
    vec = {"a": [1.0, 0.0, 0.0], "b": [0.99, 0.05, 0.0], "c": [0.0, 1.0, 0.0]}
    out = await fd.collapse_near_duplicates(_Driver(vec), rs, cosine=0.92)
    assert [o["fact_uuid"] for o in out] == ["a", "c"]
    assert [s["url"] for s in out[0]["sources"]] == ["u1", "u2"]


async def test_missing_embeddings_keep_the_fact():
    rs = [_r("a", "fact one", "u1"), _r("b", "fact two", "u2")]
    out = await fd.collapse_near_duplicates(_Driver({"a": None}), rs, cosine=0.92)
    assert [o["fact_uuid"] for o in out] == ["a", "b"]


async def test_duplicate_source_urls_are_not_repeated():
    rs = [_r("a", "same fact", "u1"), _r("b", "same fact", "u1")]
    out = await fd.collapse_near_duplicates(_Driver({}), rs, cosine=0.92)
    assert [s["url"] for s in out[0]["sources"]] == ["u1"]


async def test_an_embedding_fetch_failure_falls_back_to_text_collapse(caplog):
    class _Broken:
        async def execute_query(self, *a, **k):
            raise RuntimeError("neo4j down")
    rs = [_r("a", "same", "u1"), _r("b", "same", "u2"), _r("c", "other", "u3")]
    out = await fd.collapse_near_duplicates(_Broken(), rs, cosine=0.92)
    assert [o["fact_uuid"] for o in out] == ["a", "c"]
    assert "embedding" in caplog.text


async def test_facts_from_different_vendors_never_merge():
    # A vendorless fact ("d") merges only with another vendorless fact, never into a
    # named vendor's -- no vendorless bridge between two named vendors.
    rs = [_r("a", "Immutability is supported for Amazon S3.", "u1", vendor="Veeam"),
          _r("b", "Immutability is supported for Amazon S3.", "u2", vendor="Rubrik"),
          _r("c", "S3 immutability is supported.", "u3", vendor="Rubrik"),
          _r("d", "Immutability is supported for Amazon S3.", "u4", vendor="")]
    vec = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.99, 0.01], "d": [1.0, 0.0]}
    out = await fd.collapse_near_duplicates(_Driver(vec), rs, cosine=0.92)
    assert [o["fact_uuid"] for o in out] == ["a", "b", "d"]
    assert [s["url"] for s in out[0]["sources"]] == ["u1"]
    assert [s["url"] for s in out[1]["sources"]] == ["u2", "u3"]
    assert [s["url"] for s in out[2]["sources"]] == ["u4"]


async def test_a_vendorless_fact_never_bridges_two_vendors():
    def r(u, fact, vendor):
        src = {"url": f"https://d/{u}", "title": u}
        if vendor:
            src["vendor"] = vendor
        return {"fact": fact, "fact_uuid": u, "valid_at": None, "invalid_at": None,
                "sources": [src]}
    rs = [r("a", "same statement", None), r("b", "same statement", "Veeam"),
          r("c", "same statement", "Rubrik"), r("d", "same statement", None)]
    out = await fd.collapse_near_duplicates(_Driver({}), rs, cosine=0.96)
    assert [o["fact_uuid"] for o in out] == ["a", "b", "c"]
    assert [s["url"] for s in out[0]["sources"]] == ["https://d/a", "https://d/d"]


async def test_raw_cosine_threshold_is_not_neo4j_normalised():
    # raw cosine 0 is (1+0)/2 = 0.5 normalised: must stay apart at 0.4.
    rs = [_r("a", "one", "u1"), _r("b", "two", "u2")]
    out = await fd.collapse_near_duplicates(_Driver({"a": [1.0, 0.0], "b": [0.0, 1.0]}), rs,
                                            cosine=0.4)
    assert [o["fact_uuid"] for o in out] == ["a", "b"]


async def test_220_facts_with_1024_dim_vectors_collapse_fast():
    rng = random.Random(7)
    vec = {f"f{i}": [rng.gauss(0, 1) for _ in range(1024)] for i in range(220)}
    vec["f219"] = list(vec["f0"])
    rs = [_r(f"f{i}", f"fact number {i}", f"u{i}") for i in range(220)]
    start = time.perf_counter()
    out = await fd.collapse_near_duplicates(_Driver(vec), rs, cosine=0.92)
    assert time.perf_counter() - start < 0.5
    assert len(out) == 219 and [s["url"] for s in out[0]["sources"]] == ["u0", "u219"]


class _Recording(_Driver):
    def __init__(self, vectors):
        super().__init__(vectors)
        self.asked: list[list[str]] = []

    async def execute_query(self, query, **params):
        self.asked.append(list(params["uuids"]))
        return await super().execute_query(query, **params)


async def test_fetch_vectors_fetches_only_missing_uuids_and_reuses_the_cache():
    drv = _Recording({"a": [1.0, 0.0], "b": [0.0, 1.0], "c": None})
    cache: dict[str, list[float]] = {}
    got = await fd.fetch_vectors(drv, ["a", "c"], cache)
    assert got == {"a": [1.0, 0.0]}
    got = await fd.fetch_vectors(drv, ["a", "b", "c", "b"], cache)
    # "c" has no embedding: remembered as fetched, never asked for again.
    assert got == {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    assert drv.asked == [["a", "c"], ["b"]]
    assert await fd.fetch_vectors(drv, ["a", "b", "c"], cache) == got
    assert drv.calls == 2


async def test_a_failed_fetch_leaves_the_cache_untouched():
    class _Broken:
        async def execute_query(self, *a, **k):
            raise RuntimeError("neo4j down")
    cache = {"a": [1.0]}
    with pytest.raises(RuntimeError, match="neo4j down"):
        await fd.fetch_vectors(_Broken(), ["a", "b"], cache)
    assert cache == {"a": [1.0]}


async def test_collapse_with_a_filled_cache_issues_no_embeddings_query():
    rs = [_r("a", "Instant Recovery to a Hyper-V VM.", "u1"),
          _r("b", "Instant Recovery restores to a Hyper-V VM.", "u2"),
          _r("c", "Restore to Amazon EC2 is supported.", "u3")]
    drv = _Driver({})
    cache = {"a": [1.0, 0.0, 0.0], "b": [0.99, 0.05, 0.0], "c": [0.0, 1.0, 0.0]}
    out = await fd.collapse_near_duplicates(drv, rs, cosine=0.92, vectors=cache)
    assert drv.calls == 0
    assert [o["fact_uuid"] for o in out] == ["a", "c"]
    assert [s["url"] for s in out[0]["sources"]] == ["u1", "u2"]


async def test_collapse_with_a_cache_fetches_only_the_new_uuids_into_it():
    rs = [_r("a", "one", "u1"), _r("b", "two", "u2")]
    drv = _Recording({"a": [1.0, 0.0], "b": [1.0, 0.0]})
    cache = {"a": [1.0, 0.0]}
    out = await fd.collapse_near_duplicates(drv, rs, cosine=0.92, vectors=cache)
    assert drv.asked == [["b"]] and cache["b"] == [1.0, 0.0]
    assert [o["fact_uuid"] for o in out] == ["a"]


async def test_collapse_with_a_cache_falls_back_to_text_when_the_fetch_fails(caplog):
    class _Broken:
        async def execute_query(self, *a, **k):
            raise RuntimeError("neo4j down")
    rs = [_r("a", "same", "u1"), _r("b", "same", "u2"), _r("c", "other", "u3")]
    out = await fd.collapse_near_duplicates(_Broken(), rs, cosine=0.92, vectors={})
    assert [o["fact_uuid"] for o in out] == ["a", "c"]
    assert "embedding" in caplog.text
