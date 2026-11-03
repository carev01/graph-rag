from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.headtohead import judge_context as J
from scripts.headtohead import score as S


def _rec(gold=(True, True), claims=(True,), citations_ok=True, abstained=False) -> dict:
    return {"gold": list(gold), "claims": [{"text": f"c{i}", "supported": s}
                                           for i, s in enumerate(claims)],
            "citations_ok": citations_ok, "abstained": abstained}


def test_perfect_requires_all_gold_supported_claims_and_citations() -> None:
    assert S.perfect(_rec(), "single_vendor") is True
    assert S.perfect(_rec(gold=(True, False)), "single_vendor") is False
    assert S.perfect(_rec(claims=(True, False)), "single_vendor") is False
    assert S.perfect(_rec(citations_ok=False), "multi_article") is False
    assert S.perfect(_rec(abstained=True), "cross_vendor") is False


def test_unanswerable_perfect_only_when_abstaining_without_claims() -> None:
    assert S.perfect(_rec(gold=(), claims=(), abstained=True), "unanswerable") is True
    assert S.perfect(_rec(gold=(), claims=(True,), abstained=True), "unanswerable") is False
    assert S.perfect(_rec(gold=(), claims=(), abstained=False), "unanswerable") is False


def test_errored_answer_is_never_perfect() -> None:
    assert S.perfect({"errored": True}, "single_vendor") is False
    assert S.perfect({"errored": True}, "unanswerable") is False


def _task(key="q1", n_gold=2, errored=None) -> dict:
    return {"key": key, "qid": key, "type": "single_vendor", "question": "?",
            "gold_facts": [f"g{i}" for i in range(n_gold)],
            "labels": {"graph": "A", "vendorkb": "B"},
            "errored": errored or {"A": False, "B": False}}


def test_wrong_gold_count_is_rejected_with_key() -> None:
    task = _task("sv-007")
    ok = {"key": "sv-007", "A": _rec(), "B": _rec()}
    assert S.validate_judgement(ok, task) is None
    bad = {"key": "sv-007", "A": _rec(gold=(True,)), "B": _rec()}
    err = S.validate_judgement(bad, task)
    assert err is not None and "sv-007" in err and "gold" in err
    missing = {"key": "sv-007", "A": _rec()}
    err = S.validate_judgement(missing, task)
    assert err is not None and "sv-007" in err and "B" in err
    # an errored label needs no record
    t2 = _task("sv-008", errored={"A": False, "B": True})
    assert S.validate_judgement({"key": "sv-008", "A": _rec()}, t2) is None
    bad_type = {"key": "sv-007", "A": _rec(), "B": _rec() | {"citations_ok": "yes"}}
    assert "sv-007" in (S.validate_judgement(bad_type, task) or "")


def test_majority_of_three_passes() -> None:
    assert S.combine_passes([True, True]) is True
    assert S.combine_passes([False, False, None]) is False
    assert S.combine_passes([True, False]) is None
    assert S.combine_passes([True, False, True]) is True
    assert S.combine_passes([True, False, False]) is False
    assert S.combine_passes([True, None]) is None
    assert S.combine_passes([None, None, None]) is None
    assert S.combine_passes([{"perfect": True}, {"perfect": True}]) is True
    assert S.needs_third_pass(True, False) is True
    assert S.needs_third_pass(False, False) is False


def test_sign_test_values() -> None:
    assert round(S.sign_test_p(15, 5), 4) == 0.0414
    assert round(S.sign_test_p(12, 8), 4) == 0.5034
    assert S.sign_test_p(0, 0) == 1.0
    assert S.sign_test_p(10, 10) == 1.0
    assert S.sign_test_p(5, 15) == S.sign_test_p(15, 5)


def _pq(qid, g, v, decides=True, qtype="single_vendor") -> dict:
    return {"qid": qid, "type": qtype, "decides": decides, "graph": g, "vendorkb": v}


def test_tie_decides_vendorkb() -> None:
    # 12 graph wins vs 8: p ~ 0.50 -> tie -> vendorkb
    pq = ([_pq(f"g{i}", True, False) for i in range(12)]
          + [_pq(f"v{i}", False, True) for i in range(8)]
          + [_pq(f"b{i}", True, True) for i in range(30)]
          + [_pq(f"n{i}", False, False) for i in range(30)])
    d = S.decide(pq)
    assert (d["wins_graph"], d["wins_vendorkb"]) == (12, 8)
    assert d["winner"] == "tie" and d["decision"] == "vendorkb"
    # a significant graph margin decides graph
    pq2 = [_pq(f"g{i}", True, False) for i in range(15)] + [
        _pq(f"v{i}", False, True) for i in range(5)]
    d2 = S.decide(pq2)
    assert d2["winner"] == "graph" and d2["decision"] == "graph"
    assert d2["p"] < 0.05
    d3 = S.decide([_pq(f"v{i}", False, True) for i in range(15)]
                  + [_pq(f"g{i}", True, False) for i in range(5)])
    assert d3["winner"] == "vendorkb" and d3["decision"] == "vendorkb"


def test_reported_questions_do_not_count_toward_decision() -> None:
    pq = ([_pq(f"r{i}", True, False, decides=False) for i in range(30)]
          + [_pq(f"v{i}", False, True) for i in range(3)])
    d = S.decide(pq)
    assert (d["wins_graph"], d["wins_vendorkb"]) == (0, 3)
    assert d["winner"] == "tie" and d["decision"] == "vendorkb"


# --- end to end over files -------------------------------------------------------


def _question(qid, qtype="single_vendor", decides=True) -> dict:
    return {"id": qid, "type": qtype, "question": f"{qid}?",
            "gold_facts": [] if qtype == "unanswerable" else ["g1", "g2"],
            "gold_sources": [] if qtype == "unanswerable" else [{"article_id": "art-1"}],
            "vendors": ["Veeam"], "written_from": [], "decides": decides,
            "origin": "new" if decides else "golden_questions"}


def _answers(qid) -> list[dict]:
    return [
        {"qid": qid, "side": "graph", "answer": "G [1]", "latency_s": 3.0, "usage": None,
         "error": None, "flags": {},
         "citations": [{"marker": 1, "fact": "f", "fact_uuid": "f1",
                        "sources": [{"article_id": "art-1", "title": "T"}]}]},
        {"qid": qid, "side": "vendorkb", "answer": "V [1]", "latency_s": 1.0,
         "usage": {"cost": 0.01}, "error": None, "flags": {},
         "citations": [{"marker": 1, "chunk_id": "c1", "doc_id": "art-9", "title": "T",
                        "url": "u", "section_path": ["T"]}]},
    ]


def _setup(tmp_path: Path, qs: list[dict]) -> None:
    rows = [r for q in qs for r in _answers(q["id"])]
    (tmp_path / "answers.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    for n, seed in ((1, 1), (2, 2)):
        tasks = J.build_tasks(qs, rows, graph_sources={"f1": [
            {"fact": "f", "text": "t", "article_id": "art-1", "title": "T"}]},
            chunk_texts={"c1": "chunk"}, seed=seed)
        J.write_pass(tmp_path / f"pass-{n}", tasks)


def _judge(tmp_path: Path, n: int, verdicts: dict[str, dict[str, bool]],
           skip: tuple[str, ...] = ()) -> None:
    tasks = [json.loads(x) for x in (tmp_path / f"pass-{n}" / "tasks.jsonl")
             .read_text().splitlines()]
    out = []
    for t in tasks:
        if t["key"] in skip:
            continue
        j: dict = {"key": t["key"]}
        for side, label in t["labels"].items():
            ok = verdicts[t["key"]][side]
            ngold = len(t["gold_facts"])
            j[label] = _rec(gold=[ok] * ngold, claims=() if not ngold else (True,),
                            abstained=not ngold)
        out.append(j)
    d = tmp_path / f"pass-{n}" / "judgements"
    d.mkdir(parents=True, exist_ok=True)
    (d / "batch-001.jsonl").write_text("".join(json.dumps(j) + "\n" for j in out))


def test_score_run_maps_labels_back_and_reports(tmp_path: Path) -> None:
    qs = [_question("q1"), _question("q2"), _question("u1", "unanswerable"),
          _question("old1", decides=False)]
    _setup(tmp_path, qs)
    v = {"q1": {"graph": True, "vendorkb": False}, "q2": {"graph": False, "vendorkb": True},
         "u1": {"graph": True, "vendorkb": True}, "old1": {"graph": True, "vendorkb": False}}
    _judge(tmp_path, 1, v)
    _judge(tmp_path, 2, v)
    result, problems = S.score_run(tmp_path, qs)
    assert problems == []
    assert result["decision"]["wins_graph"] == 1
    assert result["decision"]["wins_vendorkb"] == 1
    assert result["decision"]["decision"] == "vendorkb"
    assert result["needs_third_pass"] == []
    tables = result["tables"]
    assert tables["deciding"]["all"]["graph"]["hit_rate"] == 1.0
    assert tables["deciding"]["all"]["vendorkb"]["hit_rate"] == 0.0
    assert tables["reported"]["all"]["graph"]["n"] == 1
    assert tables["deciding"]["all"]["vendorkb"]["latency_mean_s"] == 1.0
    assert result["zero_evidence_citations_pass1"] == {"graph": 0, "vendorkb": 0}
    assert "all support" in result["metric_notes"]["citation_precision"]


def test_missing_judgement_is_reported_by_key(tmp_path: Path) -> None:
    qs = [_question("q1"), _question("q2")]
    _setup(tmp_path, qs)
    v = {"q1": {"graph": True, "vendorkb": False}, "q2": {"graph": True, "vendorkb": True}}
    _judge(tmp_path, 1, v, skip=("q2",))
    _judge(tmp_path, 2, v)
    _result, problems = S.score_run(tmp_path, qs)
    assert any("q2" in p and "pass 1" in p for p in problems)
    with pytest.raises(SystemExit) as e:
        S.main(["--run", str(tmp_path), "--questions", str(_qfile(tmp_path, qs))])
    assert e.value.code == 1


def test_disagreeing_passes_need_a_third(tmp_path: Path) -> None:
    qs = [_question("q1")]
    _setup(tmp_path, qs)
    _judge(tmp_path, 1, {"q1": {"graph": True, "vendorkb": False}})
    _judge(tmp_path, 2, {"q1": {"graph": False, "vendorkb": False}})
    result, problems = S.score_run(tmp_path, qs)
    assert problems == []
    assert result["needs_third_pass"] == ["q1"]
    assert result["complete"] is False


def _qfile(tmp_path: Path, qs: list[dict]) -> Path:
    p = tmp_path / "questions.yaml"
    p.write_text(yaml.safe_dump(qs))
    return p
