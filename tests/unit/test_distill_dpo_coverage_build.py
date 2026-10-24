"""dpo_coverage build: replies to A/B/C by their labels, one pair per reply, per-split
balance, val_eval, and the two reports (spec 2026-10-24 sections 2 and 7)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.distill import dpo_coverage as dc
from scripts.distill import dpo_labels as dl
from scripts.distill import dpo_pairs as dp
from scripts.distill.teacher import AnswerStore

CTX = "DOCUMENT CONTEXT: This chunk comes from the Veeam Backup & Replication documentation."
CUR = "Veeam Backup supports Amazon S3 as a backup repository. Restores go to the original VM."
Q = "supports Amazon S3 as a backup repository"
NAMES = ["S0", "T0", "S1", "T1", "S2", "T2", "X", "Y"]


def _edge(src: str, tgt: str) -> dict:
    return {"source_entity_name": src, "target_entity_name": tgt, "relation_type": "R",
            "fact": f"fact {src}-{tgt}", "valid_at": None, "invalid_at": None,
            "episode_indices": [0]}


def _line(call_id: str, title: str) -> str:
    user = (f"<PREVIOUS_MESSAGES>\n[]\n</PREVIOUS_MESSAGES>\n<CURRENT_MESSAGE>\n[{title} › Steps]\n\n"
            f"{CUR}\n</CURRENT_MESSAGE>\n<ENTITIES>\n"
            + json.dumps([{"name": n, "entity_types": ["Entity"]} for n in NAMES])
            + "\n</ENTITIES>\n")
    return json.dumps({
        "call_id": call_id, "tier": "cheap", "prompt_name": dp.PROMPT,
        "model": dp.STUDENT_MODEL, "finish_reason": "stop",
        "messages": [{"role": "system", "content": f"sys\n{CTX}"}, {"role": "user", "content": user}],
        "response_format": {"type": "json_schema"},
        "completion": json.dumps({"edges": [_edge(f"S{i}", f"T{i}") for i in range(3)]})})


def _lab(v: str) -> dict:
    return {"verdict": v, "category": "over_inference" if v == "unsupported" else None,
            "quote": Q if v == "stated" else None, "why": "w"}


def _titles(split: str, k: int) -> list[str]:
    out, i = [], 0
    while len(out) < k:
        title = f"Article {i}"
        if dp.split_of(f"{CTX}|{title}", 0.5) == split:
            out.append(title)
        i += 1
    return out


def test_assemble_routes_each_reply_once() -> None:
    lines = [_line(cid, f"T{cid}") for cid in ("a", "c", "b", "u", "d")] + [_line("a", "Ta")]
    recs = dl.read_records_from_lines(lines)
    assert sorted(r["id"] for r in recs) == ["a", "b", "c", "d", "u"]     # "a" once
    labels = {"a": [_lab("stated"), _lab("stated"), _lab("unsupported")],
              "c": [_lab("stated")] * 3, "b": [_lab("stated")] * 3,
              "d": [_lab("stated"), _lab("stated"), _lab("unsupported")]}
    rechecks = {"a": {2: {"verdict": "unsupported"}}, "d": {2: {"verdict": "supported"}}}
    missed = {"c": [{**_edge("X", "Y"), "quote": Q}], "b": [{**_edge("S0", "T0"), "quote": Q}]}
    by_split, stats, c_why = dc.assemble(recs, labels, rechecks, missed, "sonnet", 0.0)
    train = by_split["train"]
    assert [p["meta"]["call_id"] for p in train["A"]] == ["a"]
    assert [p["meta"]["call_id"] for p in train["C"]] == ["c"]
    assert [p["meta"]["call_id"] for p in train["B"]] == ["b"]            # C failed -> B
    assert stats["unlabelled"] == 1 and stats["disputed_only"] == 1
    assert c_why == {"pair_already_present": 1}
    assert by_split["val"] == {"A": [], "B": [], "C": []}


def _store(root: Path, kind: str, answers: dict) -> None:
    st = AnswerStore(root)
    for cid, ans in answers.items():
        st.put(dl.task_key(kind, cid), json.dumps(ans), {"kind": kind, "call_id": cid})


def test_main_writes_balanced_splits_and_reports(tmp_path: Path) -> None:
    rows, labels, rechecks, ds_rows = [], {}, {}, []
    for split in ("train", "val"):
        for j, title in enumerate(_titles(split, 3)):
            cid = f"{split}{j}"
            rows.append(_line(cid, title))
            if j == 0:      # A: edge 2 confirmed unsupported
                labels[cid] = [_lab("stated"), _lab("stated"), _lab("unsupported")]
                rechecks[cid] = {"2": {"verdict": "unsupported", "category": "other", "why": "w"}}
            else:           # B candidates
                labels[cid] = [_lab("stated")] * 3
            ds_rows.append({"id": cid, "labels": [{"verdict": "unsupported"}] * 3})
    cap = tmp_path / "capture.jsonl"
    cap.write_text("\n".join(rows) + "\n")
    _store(tmp_path / "label", "label", labels)
    _store(tmp_path / "recheck", "recheck", rechecks)
    AnswerStore(tmp_path / "missed")
    ds = tmp_path / "ds"
    ds.mkdir()
    for n in (1, 2):
        (ds / f"judged-{n}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ds_rows))
    out = tmp_path / "pairs"
    rc = dc.main(["--capture", str(cap), "--labels", str(tmp_path / "label"), "--labeller",
                  "sonnet", "--recheck", str(tmp_path / "recheck"), "--missed",
                  str(tmp_path / "missed"), "--deepseek", str(ds), "--out", str(out),
                  "--val-fraction", "0.5", "--eval-size", "2"])
    assert rc == 0
    for split in ("train", "val"):
        pairs = [json.loads(x) for x in (out / f"{split}.jsonl").read_text().splitlines()]
        assert abs(dc.mean_delta(pairs)) <= dc.BAND
        assert sum(p["meta"]["type"] == "A" for p in pairs) == 1
    assert len((out / "val_eval.jsonl").read_text().splitlines()) == 2
    summary = json.loads((out / "summary.json").read_text())
    assert summary["pairs"]["train"]["A"] == 1 and summary["mean_delta"]["val"] == 0.0
    report = json.loads((out / "labels_report.json").read_text())
    assert report["deepseek"]["both_unsupported_edges"] == 18
    assert report["deepseek"]["of_which_confirmed"] == 2


def test_same_call_id_in_two_captures_gives_one_pair(tmp_path: Path) -> None:
    old, new = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    old.write_text(_line("x", "Article A") + "\n")
    new.write_text(_line("x", "Article A") + "\n" + _line("y", "Article B") + "\n")
    recs = dl.read_records([old, new])
    assert sorted(r["id"] for r in recs) == ["x", "y"]
    labels = {"x": [_lab("stated")] * 3, "y": [_lab("stated")] * 3}
    by_split, stats, _ = dc.assemble(recs, labels, {}, {}, "sonnet", 0.0)
    assert sorted(p["meta"]["call_id"] for p in by_split["train"]["B"]) == ["x", "y"]
    assert stats["clean"] == 2


def test_main_writes_nothing_when_val_has_no_a_pairs(tmp_path: Path) -> None:
    rows, labels, rechecks = [], {}, {}
    for split in ("train", "val"):
        for j, title in enumerate(_titles(split, 2)):
            cid = f"{split}{j}"
            rows.append(_line(cid, title))
            if split == "train" and j == 0:       # train has an A pair, val has none
                labels[cid] = [_lab("stated"), _lab("stated"), _lab("unsupported")]
                rechecks[cid] = {"2": {"verdict": "unsupported", "category": "other", "why": "w"}}
            else:
                labels[cid] = [_lab("stated")] * 3
    cap = tmp_path / "capture.jsonl"
    cap.write_text("\n".join(rows) + "\n")
    _store(tmp_path / "label", "label", labels)
    _store(tmp_path / "recheck", "recheck", rechecks)
    AnswerStore(tmp_path / "missed")
    out = tmp_path / "pairs"
    with pytest.raises(SystemExit, match="val"):
        dc.main(["--capture", str(cap), "--labels", str(tmp_path / "label"), "--labeller",
                 "sonnet", "--recheck", str(tmp_path / "recheck"), "--missed",
                 str(tmp_path / "missed"), "--out", str(out), "--val-fraction", "0.5"])
    assert not out.exists() or not list(out.iterdir())
