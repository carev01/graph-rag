"""dpo_labels: subagent task files, intake into an AnswerStore (one re-ask, then drop),
calibration and task selection (spec 2026-10-24 section 3)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.distill import dpo_labels as dl
from scripts.distill.teacher import AnswerStore

CUR = "Veeam Backup supports Amazon S3 as a backup repository. Restores go to the original VM."
Q = "supports Amazon S3 as a backup repository"


def _edge(i: int) -> dict:
    return {"source_entity_name": f"S{i}", "target_entity_name": f"T{i}",
            "relation_type": "R", "fact": f"fact {i}", "valid_at": None,
            "invalid_at": None, "episode_indices": [0]}


def _rec(call_id: str = "c1", n: int = 3) -> dict:
    user = ("<PREVIOUS_MESSAGES>\n[]\n</PREVIOUS_MESSAGES>\n<CURRENT_MESSAGE>\n" + CUR
            + "\n</CURRENT_MESSAGE>\n<ENTITIES>\n[]\n</ENTITIES>\n")
    return {"id": call_id, "group": "g|A", "context": "DOCUMENT CONTEXT: x",
            "messages": [{"role": "system", "content": "SYSTEM RULES"},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_schema"},
            "payload": {"edges": [_edge(i) for i in range(n)]}}


def _labels_answer(order: list[int], verdict_of) -> dict:
    return {"labels": [{"n": n, "verdict": verdict_of(i),
                        "category": "over_inference" if verdict_of(i) == "unsupported" else None,
                        "quote": Q if verdict_of(i) == "stated" else None, "why": "w"}
                       for n, i in enumerate(order, 1)]}


def test_task_keys_differ_per_kind() -> None:
    keys = {dl.task_key(k, "c1") for k in dl.KINDS}
    assert len(keys) == 3 and dl.task_key("label", "c1") == dl.task_key("label", "c1")


def test_label_request_shows_shuffled_facts_and_the_chunk() -> None:
    req = dl.label_request(_rec(n=5))
    assert sorted(req["order"]) == [0, 1, 2, 3, 4] and req["kind"] == "label"
    shown = [ln.split(": ", 1)[1] for ln in req["prompt"].split("FACTS:\n")[1].splitlines()]
    assert shown == [f"fact {i}" for i in req["order"]]
    assert CUR in req["prompt"] and "quote" in req["prompt"]
    assert dl.label_request(_rec(n=5)) == req


def test_normalise_label_reindexes_by_edge_and_rejects_bad_answers() -> None:
    req = dl.label_request(_rec(n=3))
    ans = _labels_answer(req["order"], lambda i: "unsupported" if i == 1 else "stated")
    out, why = dl.normalise("label", req, ans)
    assert why == "" and [x["verdict"] for x in out] == ["stated", "unsupported", "stated"]
    assert out[0]["quote"] == Q and out[1]["category"] == "over_inference"
    short = {"labels": ans["labels"][:2]}
    assert dl.normalise("label", req, short)[0] is None
    rep = {"labels": [ans["labels"][0]] * 3}
    assert "repeated" in dl.normalise("label", req, rep)[1]
    nocat = json.loads(json.dumps(ans))
    for lab in nocat["labels"]:
        lab["category"] = None
    assert dl.normalise("label", req, nocat)[0] is None
    noquote = json.loads(json.dumps(ans))
    for lab in noquote["labels"]:
        lab["quote"] = None
    assert dl.normalise("label", req, noquote)[0] is None


def test_normalise_recheck_maps_subset_numbers_to_edges() -> None:
    req = dl.recheck_request(_rec(n=5), [1, 4])
    ans = {"labels": [{"n": n, "verdict": "unsupported" if i == 4 else "supported",
                       "category": "wrong_detail" if i == 4 else None, "why": "w"}
                      for n, i in enumerate(req["edges"], 1)]}
    out, why = dl.normalise("recheck", req, ans)
    assert why == "" and out == {"1": {"verdict": "supported", "category": None, "why": "w"},
                                 "4": {"verdict": "unsupported", "category": "wrong_detail",
                                       "why": "w"}}
    assert "SYSTEM RULES" not in req["prompt"] and "fact 1" in req["prompt"]


def test_normalise_missed_accepts_empty_and_rejects_quoteless() -> None:
    req = dl.missed_request(_rec())
    assert "SYSTEM RULES" in req["prompt"] and '"fact 0"' in req["prompt"]
    assert dl.normalise("missed", req, {"edges": []}) == ([], "")
    good = {**_edge(9), "quote": Q}
    assert dl.normalise("missed", req, {"edges": [good]}) == ([good], "")
    assert dl.normalise("missed", req, {"edges": [_edge(9)]})[0] is None
    assert dl.normalise("missed", req, {"edges": ["x"]})[0] is None
    assert dl.normalise("missed", req, {"nope": []})[0] is None


def _answers_file(batch: Path, rows: list[dict]) -> None:
    batch.with_suffix(".answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_batches_intake_reask_once_then_drop(tmp_path: Path) -> None:
    store = AnswerStore(tmp_path / "store")
    reqs = {r["id"]: dl.label_request(r) for r in (_rec("c1"), _rec("c2"), _rec("c3"))}
    for cid, req in reqs.items():
        store.add_pending(dl.task_key("label", cid), req)
    batches = dl.write_batches(store, tmp_path / "round-1", size=2)
    assert [p.name for p in batches] == ["batch-001.jsonl", "batch-002.jsonl"]
    lines = [json.loads(x) for x in batches[0].read_text().splitlines()]
    assert [ln["ordinal"] for ln in lines] == [1, 2] and all(len(ln["check"]) == 12 for ln in lines)
    by_key = {ln["key"]: ln for ln in lines}
    k1, k2 = list(by_key)
    cid = {dl.task_key("label", c): c for c in reqs}
    good = _labels_answer(reqs[cid[k1]]["order"], lambda i: "stated")
    _answers_file(batches[0], [
        {"key": k1, "ordinal": 1, "check": by_key[k1]["check"], "answer": good},
        {"key": k2, "ordinal": 2, "check": by_key[k2]["check"], "answer": {"labels": []}},
        {"key": "f" * 64, "ordinal": 9, "check": "x", "answer": good},
    ])
    counts = dl.intake(store, tmp_path / "round-1")
    assert counts == {"stored": 1, "reasked": 1, "unknown_key": 1}
    assert [x["verdict"] for x in json.loads(store.get(k1))] == ["stated"] * 3
    pending = dict(store.pending())
    assert "rejection" in pending[k2] and pending[k2]["previous_answer"] == {"labels": []}
    k3 = json.loads(batches[1].read_text())["key"]
    assert k3 in pending                                   # never answered: still pending
    batches2 = dl.write_batches(store, tmp_path / "round-2", size=10)
    lines2 = {json.loads(x)["key"]: json.loads(x) for x in batches2[0].read_text().splitlines()}
    assert "rejection" in lines2[k2]
    _answers_file(batches2[0], [
        {"key": k2, "ordinal": lines2[k2]["ordinal"], "check": lines2[k2]["check"],
         "answer": {"labels": []}}])
    assert dl.intake(store, tmp_path / "round-2") == {"dropped": 1}
    assert store.dropped_reason(k2) and k2 not in dict(store.pending())


def test_intake_rejects_an_answer_under_the_wrong_ordinal(tmp_path: Path) -> None:
    store = AnswerStore(tmp_path / "store")
    req = dl.label_request(_rec("c1"))
    key = dl.task_key("label", "c1")
    store.add_pending(key, req)
    batch = dl.write_batches(store, tmp_path / "r", size=10)[0]
    line = json.loads(batch.read_text())
    _answers_file(batch, [{"key": key, "ordinal": 2, "check": line["check"],
                           "answer": _labels_answer(req["order"], lambda i: "stated")}])
    assert dl.intake(store, tmp_path / "r") == {"reasked": 1}
    assert store.get(key) is None
    assert "wrong prompt" in dict(store.pending())[key]["rejection"]


def test_write_batches_refuses_a_used_directory(tmp_path: Path) -> None:
    store = AnswerStore(tmp_path / "store")
    store.add_pending(dl.task_key("label", "c1"), dl.label_request(_rec("c1")))
    dl.write_batches(store, tmp_path / "r", size=10)
    with pytest.raises(FileExistsError):
        dl.write_batches(store, tmp_path / "r", size=10)


def test_deepseek_verdicts_pairs_both_passes(tmp_path: Path) -> None:
    for n, verdicts in ((1, ["unsupported", "stated"]), (2, ["unsupported", "unsupported"])):
        (tmp_path / f"judged-{n}.jsonl").write_text(json.dumps(
            {"id": "c1", "labels": [{"verdict": v} for v in verdicts]}) + "\n")
    assert dl.deepseek_verdicts(tmp_path) == {"c1": [("unsupported", "unsupported"),
                                                    ("stated", "unsupported")]}


def test_calibration_ids_half_flagged_half_not_seeded() -> None:
    recs = [_rec(f"c{i}") for i in range(300)]
    ds = {f"c{i}": [("unsupported", "unsupported")] if i % 5 == 0 else [("stated", "stated")]
          for i in range(300)}
    ids = dl.calibration_ids(recs, ds, 100, seed="s")
    flagged = [i for i in ids if ds[i][0] == ("unsupported", "unsupported")]
    assert len(ids) == 100 and len(flagged) == 50 and len(set(ids)) == 100
    assert ids == dl.calibration_ids(recs, ds, 100, seed="s")
    few = {k: [("stated", "stated")] for k in ds} | {"c0": [("unsupported", "unsupported")]}
    ids2 = dl.calibration_ids(recs, few, 100, seed="s")
    assert "c0" in ids2 and len(ids2) == 100


def _lab(v: str) -> dict:
    return {"verdict": v, "category": "other" if v == "unsupported" else None,
            "quote": Q if v == "stated" else None, "why": "w"}


def test_calibrate_recall_bar_is_inclusive_at_090() -> None:
    opus = {f"c{i}": [_lab("unsupported")] for i in range(10)}
    nine = {f"c{i}": [_lab("unsupported" if i < 9 else "stated")] for i in range(10)}
    res = dl.calibrate(list(opus), nine, opus)
    assert res["recall"] == pytest.approx(0.9) and res["passed"] and res["labeller"] == "sonnet"
    assert res["matrix"]["unsupported|unsupported"] == 9
    eight = {f"c{i}": [_lab("unsupported" if i < 8 else "stated")] for i in range(10)}
    assert dl.calibrate(list(opus), eight, opus)["labeller"] == "opus"
    with pytest.raises(ValueError):
        dl.calibrate(["c0"], {"c0": [_lab("stated")]}, {"c0": [_lab("stated")]})


def test_recheck_targets_and_missed_candidates() -> None:
    recs = [_rec(f"c{i}") for i in range(6)]
    labels = {f"c{i}": [_lab("stated"), _lab("stated"),
                        _lab("unsupported" if i < 2 else "stated")] for i in range(6)}
    assert dl.recheck_targets(recs, labels) == {"c0": [2], "c1": [2]}
    rechecks = {"c0": {2: {"verdict": "unsupported"}}, "c1": {2: {"verdict": "supported"}}}
    picked = dl.missed_candidates(recs, labels, rechecks, n=3, seed="s")
    assert len(picked) == 3 and not {"c0", "c1"} & set(picked)
    assert picked == dl.missed_candidates(recs, labels, rechecks, n=3, seed="s")


def test_intake_counts_malformed_rows_without_crashing(tmp_path: Path) -> None:
    store = AnswerStore(tmp_path / "store")
    key = dl.task_key("label", "c1")
    store.add_pending(key, dl.label_request(_rec("c1")))
    batch = dl.write_batches(store, tmp_path / "r", size=10)[0]
    batch.with_suffix(".answers.jsonl").write_text('[1, 2]\n"str"\n{"key": ["x"]}\n')
    assert dl.intake(store, tmp_path / "r") == {"unparseable_line": 3}
    assert key in dict(store.pending())


def test_replaying_intake_skips_rows_from_an_earlier_round(tmp_path: Path) -> None:
    store = AnswerStore(tmp_path / "store")
    key = dl.task_key("label", "c1")
    store.add_pending(key, dl.label_request(_rec("c1")))
    batch = dl.write_batches(store, tmp_path / "r", size=10)[0]
    line = json.loads(batch.read_text())
    _answers_file(batch, [{"key": key, "ordinal": 1, "check": line["check"],
                           "answer": {"labels": []}}])
    assert dl.intake(store, tmp_path / "r") == {"reasked": 1}
    assert dl.intake(store, tmp_path / "r") == {"stale": 1}
    pending = dict(store.pending())
    assert key in pending and "rejection" in pending[key]
    assert store.dropped_reason(key) is None


def test_calibrate_missing_reply_fails_the_bar(tmp_path: Path) -> None:
    opus = {f"c{i}": [_lab("unsupported")] for i in range(10)}
    sonnet = {f"c{i}": [_lab("unsupported")] for i in range(9)}
    res = dl.calibrate(list(opus), sonnet, opus)
    assert res["missing"] == ["c9"] and res["edge_count_mismatch"] == []
    assert res["passed"] is False and res["labeller"] == "opus"


def test_calibrate_length_mismatch_fails_the_bar() -> None:
    opus = {"c0": [_lab("unsupported")], "c1": [_lab("unsupported")]}
    sonnet = {"c0": [_lab("unsupported"), _lab("stated")], "c1": [_lab("unsupported")]}
    res = dl.calibrate(["c0", "c1"], sonnet, opus)
    assert res["edge_count_mismatch"] == ["c0"] and res["missing"] == []
    assert res["passed"] is False and res["labeller"] == "opus"
    assert res["matrix"] == {"unsupported|unsupported": 1}
