from __future__ import annotations

import json
from pathlib import Path

from scripts.headtohead import judge_context as J


def _q(qid: str, qtype: str = "single_vendor", gold=None) -> dict:
    return {"id": qid, "type": qtype, "question": f"What about {qid}?",
            "gold_facts": ["fact one", "fact two"] if gold is None else gold,
            "gold_sources": [{"article_id": "art-1"}], "vendors": ["Veeam"],
            "written_from": ["art-1"], "decides": True, "origin": "new"}


def _graph_row(qid: str, cits=None, answer="Graph answer [1].") -> dict:
    if cits is None:
        cits = [{"marker": 1, "fact": "X supports Y", "fact_uuid": "f1",
                 "valid_at": None, "invalid_at": None,
                 "sources": [{"url": "http://u/1", "title": "Art 1", "article_id": "art-1",
                              "section": "S", "vendor": "Veeam", "product": "VBR"}]}]
    return {"qid": qid, "side": "graph", "answer": answer, "citations": cits,
            "latency_s": 1.0, "usage": None, "error": None, "flags": {"mode": "local"}}


def _vk_row(qid: str, cits=None, answer="Chunk answer [1].") -> dict:
    if cits is None:
        cits = [{"marker": 1, "chunk_id": "c1", "doc_id": "art-1", "url": "http://u/1",
                 "title": "Art 1", "section_path": ["Art 1", "S"]}]
    return {"qid": qid, "side": "vendorkb", "answer": answer, "citations": cits,
            "latency_s": 2.0, "usage": None, "error": None, "flags": {"rerank_ok": True}}


GRAPH_SOURCES = {"f1": [
    {"fact": "X supports Y", "text": "Episode text: X supports Y in v12.",
     "article_id": "art-1", "title": "Art 1"},
    {"fact": "X supports Y", "text": "Second episode about X.",
     "article_id": "art-2", "title": "Art 2"},
]}
CHUNKS = {"c1": "Art 1\nArt 1 > S\n\nChunk text: X supports Y."}


def _build(questions, answers, **kw):
    kw.setdefault("graph_sources", GRAPH_SOURCES)
    kw.setdefault("chunk_texts", CHUNKS)
    kw.setdefault("seed", 1)
    return J.build_tasks(questions, answers, **kw)


def _side_answer(task: dict, side: str) -> dict:
    return task["answers"][task["labels"][side]]


def test_graph_sources_mark_fact_as_paraphrase_and_include_episode_text() -> None:
    [task] = _build([_q("q1")], [_graph_row("q1"), _vk_row("q1")])
    g = _side_answer(task, "graph")
    assert g["text"] == "Graph answer [1]."
    [src] = g["sources"]
    assert src["n"] == 1
    assert src["paraphrase"] == "X supports Y"
    assert "system paraphrase (not evidence)" in src["paraphrase_note"]
    texts = [d["text"] for d in src["documents"]]
    assert texts == ["Episode text: X supports Y in v12.", "Second episode about X."]
    assert [d["title"] for d in src["documents"]] == ["Art 1", "Art 2"]
    v = _side_answer(task, "vendorkb")
    [vsrc] = v["sources"]
    assert vsrc["paraphrase"] is None
    assert vsrc["documents"][0]["text"] == CHUNKS["c1"]
    # the judge's view has the same keys on both sides and no system-specific fields
    view = J.judge_view(task)
    blob = json.dumps(view)
    for leak in ("labels", "fact_uuid", "chunk_id", "doc_id", "article_id", "mode", "graph",
                 "vendorkb", "valid_at", "rerank", "http://"):
        assert leak not in blob, leak
    a, b = view["answers"]["A"], view["answers"]["B"]
    assert set(a) == set(b)
    assert set(a["sources"][0]) == set(b["sources"][0])


def test_labels_are_shuffled_and_recorded() -> None:
    qs = [_q(f"q{i:02d}") for i in range(20)]
    rows = [r for q in qs for r in (_graph_row(q["id"]), _vk_row(q["id"]))]
    tasks = _build(qs, rows, seed=7)
    assert [t["key"] for t in tasks] == sorted(q["id"] for q in qs)
    orders = {t["labels"]["graph"] for t in tasks}
    assert orders == {"A", "B"}
    for t in tasks:
        assert {t["labels"]["graph"], t["labels"]["vendorkb"]} == {"A", "B"}
        assert _side_answer(t, "graph")["text"].startswith("Graph answer")
        assert _side_answer(t, "vendorkb")["text"].startswith("Chunk answer")
    again = _build(qs, rows, seed=7)
    assert [t["labels"] for t in again] == [t["labels"] for t in tasks]
    other = _build(qs, rows, seed=8)
    assert [t["labels"] for t in other] != [t["labels"] for t in tasks]


def test_source_text_is_capped_and_truncation_flagged() -> None:
    big = {f"c{i}": str(i) + "x" * 399 for i in range(1, 4)}
    cits = [{"marker": i, "chunk_id": f"c{i}", "doc_id": "d", "url": "u", "title": "T",
             "section_path": ["T"]} for i in range(1, 4)]
    [task] = _build([_q("q1")], [_graph_row("q1"), _vk_row("q1", cits=cits)],
                    chunk_texts=big, max_source_chars=1000)
    v_label = task["labels"]["vendorkb"]
    g_label = task["labels"]["graph"]
    assert task["truncated"][v_label] is True
    assert task["truncated"][g_label] is False
    docs = [d for s in task["answers"][v_label]["sources"] for d in s["documents"]]
    kept = sum(len(d["text"]) for d in docs if not d.get("truncated"))
    assert kept == 800
    assert docs[2]["truncated"] is True
    assert docs[2]["text"].startswith("3" + "x" * 199)
    assert len(docs[2]["text"].replace(J.TRUNCATION_NOTE, "")) == 200


def test_citation_without_fact_uuid_is_marked_unresolvable() -> None:
    cits = [{"marker": 1, "fact": "no uuid here", "sources": []},
            {"marker": 2, "fact": "X supports Y", "fact_uuid": "f1", "sources": []}]
    [task] = _build([_q("q1")], [_graph_row("q1", cits=cits), _vk_row("q1")])
    g_label = task["labels"]["graph"]
    srcs = task["answers"][g_label]["sources"]
    assert srcs[0]["unresolvable"] is True
    assert srcs[0]["documents"] == []
    assert srcs[1]["unresolvable"] is False
    assert task["unresolvable"][g_label] == 1


def test_errored_side_is_not_sent_to_judge(tmp_path: Path) -> None:
    err = _vk_row("q1", answer=None, cits=[])
    err["error"] = "boom"
    both = [_graph_row("q2", answer=None, cits=[]) | {"error": "x"},
            _vk_row("q2", answer=None, cits=[]) | {"error": "y"}]
    tasks = _build([_q("q1"), _q("q2")], [_graph_row("q1"), err, *both])
    t1 = tasks[0]
    v_label = t1["labels"]["vendorkb"]
    assert t1["errored"][v_label] is True
    assert t1["answers"][v_label] == {"errored": True, "text": None, "sources": []}
    J.write_pass(tmp_path, tasks)
    lines = (tmp_path / "batches" / "batch-001.jsonl").read_text().splitlines()
    keys = [json.loads(x)["key"] for x in lines]
    assert keys == ["q1"]  # both-errored q2 needs no judge
    assert len((tmp_path / "tasks.jsonl").read_text().splitlines()) == 2


def test_last_answer_row_wins_and_missing_side_raises() -> None:
    old = _graph_row("q1", answer=None, cits=[]) | {"error": "first try"}
    [task] = _build([_q("q1")], [old, _vk_row("q1"), _graph_row("q1")])
    assert _side_answer(task, "graph")["text"] == "Graph answer [1]."
    try:
        _build([_q("q1")], [_graph_row("q1")])
    except ValueError as e:
        assert "q1" in str(e)
    else:
        raise AssertionError("missing vendorkb row must raise")


def test_batches_hold_four_tasks(tmp_path: Path) -> None:
    qs = [_q(f"q{i}") for i in range(9)]
    rows = [r for q in qs for r in (_graph_row(q["id"]), _vk_row(q["id"]))]
    J.write_pass(tmp_path, _build(qs, rows))
    files = sorted((tmp_path / "batches").glob("*.jsonl"))
    assert [f.name for f in files] == ["batch-001.jsonl", "batch-002.jsonl", "batch-003.jsonl"]
    assert [len(f.read_text().splitlines()) for f in files] == [4, 4, 1]


def test_episode_repeated_across_articles_or_facts_is_shown_once() -> None:
    rows = {"f1": [{"fact": "F", "text": "same episode", "article_id": "a", "title": "A"},
                   {"fact": "F", "text": "same episode", "article_id": "b", "title": "B"}],
            "f2": [{"fact": "G", "text": "same episode", "article_id": "a", "title": "A"}]}
    cits = [{"marker": 1, "fact": "F", "fact_uuid": "f1", "sources": []},
            {"marker": 2, "fact": "G", "fact_uuid": "f2", "sources": []}]
    [task] = _build([_q("q1")], [_graph_row("q1", cits=cits), _vk_row("q1")],
                    graph_sources=rows)
    s1, s2 = _side_answer(task, "graph")["sources"]
    assert [d["text"] for d in s1["documents"]] == ["same episode"]
    assert s2["documents"][0]["text"] == ""
    assert s2["documents"][0]["same_text_as_source"] == 1
    assert s2["unresolvable"] is False
