"""Near-duplicate collapse keeps every source (spec 2026-10-15 §4.3, §6)."""
import random
import time

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
    rs = [_r("a", "Immutability is supported for Amazon S3.", "u1", vendor="Veeam"),
          _r("b", "Immutability is supported for Amazon S3.", "u2", vendor="Rubrik"),
          _r("c", "S3 immutability is supported.", "u3", vendor="Rubrik"),
          _r("d", "Immutability is supported for Amazon S3.", "u4", vendor="")]
    vec = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.99, 0.01], "d": [1.0, 0.0]}
    out = await fd.collapse_near_duplicates(_Driver(vec), rs, cosine=0.92)
    assert [o["fact_uuid"] for o in out] == ["a", "b"]
    assert [s["url"] for s in out[0]["sources"]] == ["u1", "u4"]
    assert [s["url"] for s in out[1]["sources"]] == ["u2", "u3"]


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
