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
    # fair share: 1000 // 3 = 333 each, the 1 left over goes round-robin to the first
    shown = [len(d["text"].replace(J.TRUNCATION_NOTE, "")) for d in docs]
    assert sorted(shown) == [333, 333, 334]
    assert all(d["truncated"] for d in docs)
    assert docs[2]["text"].startswith("3" + "x" * 199)
    assert task["zero_evidence"] == {"A": 0, "B": 0}


def test_fair_share_redistributes_unused_budget() -> None:
    texts = {"c1": "a" * 100, "c2": "b" * 5000, "c3": "c" * 5000}
    cits = [{"marker": i, "chunk_id": f"c{i}", "title": "T"} for i in range(1, 4)]
    [task] = _build([_q("q1")], [_graph_row("q1"), _vk_row("q1", cits=cits)],
                    chunk_texts=texts, max_source_chars=3000)
    srcs = _side_answer(task, "vendorkb")["sources"]
    shown = [len(s["documents"][0]["text"].replace(J.TRUNCATION_NOTE, "")) for s in srcs]
    assert shown == [100, 1450, 1450]


def test_long_fact_does_not_starve_later_citations() -> None:
    episodes = {"f1": [{"fact": "F", "text": f"episode {i} " + "e" * 20000,
                        "article_id": f"a{i}", "title": f"T{i}"} for i in range(30)]}
    for k in range(2, 5):
        episodes[f"f{k}"] = [{"fact": "G", "text": f"other {k} " + "o" * 300,
                              "article_id": f"o{k}", "title": "O"}]
    cits = [{"marker": 1, "fact": "F", "fact_uuid": "f1", "sources": []}] + [
        {"marker": k, "fact": "G", "fact_uuid": f"f{k}", "sources": []} for k in range(2, 5)]
    [task] = _build([_q("q1")], [_graph_row("q1", cits=cits), _vk_row("q1")],
                    graph_sources=episodes)
    g_label = task["labels"]["graph"]
    srcs = task["answers"][g_label]["sources"]
    assert len(srcs) == 4
    for s in srcs:
        assert sum(len(d["text"].replace(J.TRUNCATION_NOTE, "")) for d in s["documents"]) > 0
    for s in srcs[1:]:  # small sources are shown whole
        assert s["documents"][0]["truncated"] is False
    assert task["truncated"][g_label] is True
    assert task["zero_evidence"][g_label] == 0
    total = sum(len(d["text"].replace(J.TRUNCATION_NOTE, ""))
                for s in srcs for d in s["documents"])
    assert total <= J.MAX_SOURCE_CHARS


def test_episodes_of_the_cited_article_come_first_and_are_capped_at_five() -> None:
    rows = {"f1": [{"fact": "F", "text": f"ep {i}", "article_id": f"a{i}", "title": f"T{i}"}
                   for i in range(8)]}
    cits = [{"marker": 1, "fact": "F", "fact_uuid": "f1",
             "sources": [{"article_id": "a6", "title": "T6"}]}]
    [task] = _build([_q("q1")], [_graph_row("q1", cits=cits), _vk_row("q1")],
                    graph_sources=rows)
    [src] = _side_answer(task, "graph")["sources"]
    assert [d["text"] for d in src["documents"]] == ["ep 6", "ep 0", "ep 1", "ep 2", "ep 3"]
    assert src["documents_omitted"] == 3
    assert J.MAX_EPISODES_PER_FACT == 5
    assert "ORDER BY" in J.EPISODES_FOR_FACTS


def test_existing_pass_is_not_overwritten(tmp_path: Path) -> None:
    pdir = tmp_path / "pass-1"
    assert J.check_pass_dir(pdir, force=False) is None
    pdir.mkdir()
    (pdir / "tasks.jsonl").write_text("{}\n")
    assert "exists" in (J.check_pass_dir(pdir, force=False) or "")
    assert J.check_pass_dir(pdir, force=True) is None
    (pdir / "judgements").mkdir()
    assert J.check_pass_dir(pdir, force=False) is not None
    assert J.check_pass_dir(pdir, force=True) is None  # empty judgements/
    (pdir / "judgements" / "batch-001.jsonl").write_text("{}\n")
    err = J.check_pass_dir(pdir, force=True)
    assert err is not None and "judgements" in err


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


def test_batches_hold_two_tasks(tmp_path: Path) -> None:
    qs = [_q(f"q{i}") for i in range(9)]
    rows = [r for q in qs for r in (_graph_row(q["id"]), _vk_row(q["id"]))]
    J.write_pass(tmp_path, _build(qs, rows))
    files = sorted((tmp_path / "batches").glob("*.jsonl"))
    assert J.BATCH_SIZE == 2
    assert [f.name for f in files] == [f"batch-{i:03d}.jsonl" for i in range(1, 6)]
    assert [len(f.read_text().splitlines()) for f in files] == [2, 2, 2, 2, 1]


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


def test_pointer_to_a_truncated_copy_gets_its_own_text() -> None:
    rows = {"f1": [{"fact": "F", "text": "s" * 3000, "article_id": "a", "title": "A"}],
            "f2": [{"fact": "G", "text": "s" * 3000, "article_id": "a", "title": "A"}]}
    cits = [{"marker": 1, "fact": "F", "fact_uuid": "f1", "sources": []},
            {"marker": 2, "fact": "G", "fact_uuid": "f2", "sources": []}]
    [task] = _build([_q("q1")], [_graph_row("q1", cits=cits), _vk_row("q1")],
                    graph_sources=rows, max_source_chars=2000)
    s1, s2 = _side_answer(task, "graph")["sources"]
    assert s1["documents"][0]["truncated"] is True
    assert s2["documents"][0]["same_text_as_source"] is None
    assert len(s2["documents"][0]["text"].replace(J.TRUNCATION_NOTE, "")) == 1000


def test_vendorkb_citation_text_is_preferred_over_the_db_lookup() -> None:
    cits = [{"marker": 1, "chunk_id": "c1", "title": "Art 1", "text": "Text at answer time."},
            {"marker": 2, "chunk_id": "c1", "title": "Art 1"}]
    tasks = _build([_q("q1")], [_graph_row("q1"), _vk_row("q1", cits=cits)])
    srcs = _side_answer(tasks[0], "vendorkb")["sources"]
    assert srcs[0]["documents"][0]["text"] == "Text at answer time."
    assert srcs[1]["documents"][0]["text"] == CHUNKS["c1"]  # fallback: DB by chunk_id


def test_db_lookup_only_for_citations_without_text() -> None:
    rows = [_vk_row("q1", cits=[{"marker": 1, "chunk_id": "c1", "text": "t"},
                                {"marker": 2, "chunk_id": "c2"}])]
    assert J.chunk_ids_needing_lookup(rows) == {"c2"}
