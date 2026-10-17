"""Near-duplicate collapse keeps every source (spec 2026-10-15 §4.3, §6)."""
import logging
import random
import time

import numpy as np
import pytest
from neo4j.exceptions import AuthError, CypherSyntaxError
from neo4j.vector import Vector

import answer_api.fact_dedup as fd


def _lists(vectors):
    return {u: np.asarray(v).tolist() for u, v in vectors.items()}


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
    assert _lists(got) == {"a": [1.0, 0.0]}
    got = await fd.fetch_vectors(drv, ["a", "b", "c", "b"], cache)
    # "c" has no embedding: remembered as fetched, never asked for again.
    assert _lists(got) == {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    assert drv.asked == [["a", "c"], ["b"]]
    assert _lists(await fd.fetch_vectors(drv, ["a", "b", "c"], cache)) == _lists(got)
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
    assert drv.asked == [["b"]] and cache["b"].tolist() == [1.0, 0.0]
    assert [o["fact_uuid"] for o in out] == ["a"]


async def test_collapse_with_a_cache_falls_back_to_text_when_the_fetch_fails(caplog):
    class _Broken:
        async def execute_query(self, *a, **k):
            raise RuntimeError("neo4j down")
    rs = [_r("a", "same", "u1"), _r("b", "same", "u2"), _r("c", "other", "u3")]
    out = await fd.collapse_near_duplicates(_Broken(), rs, cosine=0.92, vectors={})
    assert [o["fact_uuid"] for o in out] == ["a", "c"]
    assert "embedding" in caplog.text


def _int8(v):
    """The INTEGER8 encoding EMBEDDINGS_QUERY computes in Cypher: per-vector max-abs
    scaled to +-127 and rounded."""
    a = np.asarray(v, dtype=np.float64)
    return Vector(np.round(a * 127.0 / np.abs(a).max()).astype(np.int8))


class _VectorDriver(_Driver):
    """Returns each embedding as a neo4j INTEGER8 Vector, as the compact query does."""
    async def execute_query(self, query, **params):
        self.calls += 1
        assert query is fd.EMBEDDINGS_QUERY
        return ([{"uuid": u, "embedding": None if self.vectors.get(u) is None
                  else _int8(self.vectors[u])} for u in params["uuids"]], None, None)


async def test_an_int8_vector_decodes_to_a_float_array():
    got = await fd.fetch_vectors(_VectorDriver({"a": [0.5, -0.25, 0.0], "b": None}),
                                 ["a", "b"], {})
    assert list(got) == ["a"]
    assert isinstance(got["a"], np.ndarray) and got["a"].dtype == np.float32
    assert got["a"].tolist() == [127.0, -64.0, 0.0]


async def test_collapse_on_int8_vectors_matches_the_float_result():
    rng = random.Random(11)
    vec = {f"f{i}": [rng.gauss(0, 1) for _ in range(256)] for i in range(60)}
    for i in range(0, 60, 7):                  # near-duplicates of every 7th fact
        vec[f"f{i + 1}"] = [x + rng.gauss(0, 0.05) for x in vec[f"f{i}"]]
    rs = [_r(u, f"fact {u}", f"https://d/{u}") for u in vec]
    floats = await fd.collapse_near_duplicates(_Driver(vec), rs, cosine=0.92)
    compact = await fd.collapse_near_duplicates(_VectorDriver(vec), rs, cosine=0.92)
    assert len(floats) < len(rs)               # the near-duplicates did merge
    assert [(o["fact_uuid"], [s["url"] for s in o["sources"]]) for o in compact] == [
        (o["fact_uuid"], [s["url"] for s in o["sources"]]) for o in floats]


class _NoVectorType:
    """A server without vector()/INTEGER8: the compact query is a syntax error."""
    def __init__(self, vectors):
        self.vectors, self.asked = vectors, []

    async def execute_query(self, query, **params):
        self.asked.append(query)
        if query is fd.EMBEDDINGS_QUERY:
            raise CypherSyntaxError("Unknown function 'vector'")
        assert query is fd.EMBEDDINGS_QUERY_FLOAT
        return ([{"uuid": u, "embedding": self.vectors.get(u)} for u in params["uuids"]],
                None, None)


async def test_the_float_fallback_switches_once_and_logs_once(monkeypatch, caplog):
    monkeypatch.setattr(fd, "_compact_unavailable", False)
    drv = _NoVectorType({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    with caplog.at_level(logging.WARNING, logger="answer_api.fact_dedup"):
        first = await fd.fetch_vectors(drv, ["a"], {})
        second = await fd.fetch_vectors(drv, ["b"], {})
    assert _lists(first) == {"a": [1.0, 0.0]} and _lists(second) == {"b": [0.0, 1.0]}
    assert drv.asked == [fd.EMBEDDINGS_QUERY, fd.EMBEDDINGS_QUERY_FLOAT,
                         fd.EMBEDDINGS_QUERY_FLOAT]
    warnings = [r for r in caplog.records if "compact embedding transfer unavailable"
                in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING


@pytest.mark.parametrize("exc", [RuntimeError("neo4j down"), AuthError("expired")])
async def test_other_errors_propagate_without_switching_to_floats(monkeypatch, exc):
    monkeypatch.setattr(fd, "_compact_unavailable", False)

    class _Failing:
        async def execute_query(self, query, **params):
            raise exc
    cache: dict = {}
    with pytest.raises(type(exc)):
        await fd.fetch_vectors(_Failing(), ["a"], cache)
    assert fd._compact_unavailable is False and cache == {}
