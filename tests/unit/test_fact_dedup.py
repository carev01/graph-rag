"""Near-duplicate collapse keeps every source (spec 2026-10-15 §4.3, §6)."""
import answer_api.fact_dedup as fd


def _r(uuid, fact, url):
    return {"fact": fact, "fact_uuid": uuid, "valid_at": None, "invalid_at": None,
            "sources": [{"url": url, "title": url, "vendor": "Veeam", "product": "VBR"}]}


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
