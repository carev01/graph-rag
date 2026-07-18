from answer_api.router import _render_timeline, _normalize


def test_render_timeline_builds_markdown_and_citations():
    raw = {"timeline": [
        {"fact": "Veeam added immutability", "fact_uuid": "f1", "valid_at": "2023-01-01",
         "invalid_at": None, "status": "current", "sources": [{"url": "https://x/1"}]},
        {"fact": "Old behavior", "fact_uuid": "f2", "valid_at": "2020-01-01",
         "invalid_at": "2023-01-01", "status": "superseded", "sources": [{"url": "https://x/2"}]},
    ]}
    answer, citations = _render_timeline(raw)
    assert "[1]" in answer and "[2]" in answer
    assert "Veeam added immutability" in answer
    assert [c["marker"] for c in citations] == [1, 2]
    assert citations[0]["fact_uuid"] == "f1"
    assert citations[1]["sources"][0]["url"] == "https://x/2"


def test_render_timeline_empty():
    answer, citations = _render_timeline({"timeline": []})
    assert citations == []
    assert "No recorded changes" in answer


def test_normalize_global_carries_communities():
    raw = {"query": "q", "answer": "A [1]",
           "citations": [{"marker": 1, "fact_uuid": "f1", "sources": []}],
           "communities_used": [{"community_id": "c1", "title": "T"}]}
    env = _normalize("global", "heuristic", None, raw, "q")
    assert env["mode"] == "global"
    assert env["routing"] == {"chosen": "global", "via": "heuristic", "fallback_from": None}
    assert env["communities_used"] == [{"community_id": "c1", "title": "T"}]
    assert "follow_ups" not in env and "timeline" not in env


def test_normalize_drift_degrade_surfaces_flag():
    raw = {"query": "q", "answer": "local A", "citations": [],
           "retrieved": 2, "cited": 0, "degraded": "no-primer-communities"}
    env = _normalize("drift", "llm", None, raw, "q")
    assert env["routing"]["degraded"] == "no-primer-communities"
    assert env["answer"] == "local A"
    assert "communities_used" not in env


def test_normalize_timeline_renders_and_carries_array():
    raw = {"timeline": [{"fact": "F", "fact_uuid": "f1", "valid_at": "2023-01-01",
                         "invalid_at": None, "status": "current", "sources": []}]}
    env = _normalize("timeline", "heuristic", None, raw, "q")
    assert env["mode"] == "timeline"
    assert "[1]" in env["answer"]
    assert env["citations"][0]["fact_uuid"] == "f1"
    assert env["timeline"] == raw["timeline"]


def test_normalize_records_fallback_from():
    raw = {"query": "q", "answer": "A", "citations": []}
    env = _normalize("drift", "llm", "local", raw, "q")
    assert env["routing"]["fallback_from"] == "local"
