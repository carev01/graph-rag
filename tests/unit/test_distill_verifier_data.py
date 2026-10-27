"""verifier_data: dpo4 captures + Opus labels -> verifier chat records (spec
2026-10-27-fact-verifier-design.md section 3)."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.distill import verifier_data as vd
from scripts.distill.dpo_pairs import split_of

CUR = "Veeam Backup supports Amazon S3 as a backup repository. Restores go to the original VM."
Q = "supports Amazon S3 as a backup repository"


def _user(cur: str = CUR) -> str:
    return ("<PREVIOUS_MESSAGES>\n[\"earlier chunk\"]\n</PREVIOUS_MESSAGES>\n<CURRENT_MESSAGE>\n"
            + cur + "\n</CURRENT_MESSAGE>\n<ENTITIES>\n[]\n</ENTITIES>\n")


def _edge(i: int) -> dict:
    return {"source_entity_name": f"S{i}", "target_entity_name": f"T{i}",
            "relation_type": "R", "fact": f"fact {i}"}


def _rec(n: int = 4, call_id: str = "c1", group: str = "ctx|Article") -> dict:
    return {"id": call_id, "group": group, "context": "DOCUMENT CONTEXT: This chunk comes from X.",
            "messages": [{"role": "system", "content": "sys"},
                         {"role": "user", "content": _user()}],
            "payload": {"edges": [_edge(i) for i in range(n)]}}


def _lab(verdict: str, quote: str | None = Q, category: str | None = "over_inference") -> dict:
    return {"verdict": verdict, "category": category if verdict == "unsupported" else None,
            "quote": quote if verdict == "stated" else None, "why": "w"}


def test_instructions_keep_the_definitions_and_change_only_the_output() -> None:
    assert "taxonomy_as_fact" in vd.VERIFIER_INSTRUCTIONS
    assert '"why"' not in vd.VERIFIER_INSTRUCTIONS
    assert vd.VERIFIER_INSTRUCTIONS.rstrip().endswith("in order.")


def test_build_record_maps_final_labels_and_excludes_uncertain_facts() -> None:
    labels = [_lab("stated"), _lab("unsupported", category="wrong_detail"),
              _lab("unsupported"), _lab("previous_only"), _lab("stated", quote="not in chunk at all")]
    recheck = {1: {"verdict": "unsupported", "category": "subject_substitution"},
               2: {"verdict": "supported", "category": None}}
    rec, why = vd.build_record(_rec(5), labels, recheck)
    assert rec is not None
    assert rec["meta"]["edge_index"] == [0, 1, 3]          # 2 disputed, 4 unverified: left out
    assert rec["meta"]["final"] == ["stated", "confirmed", "previous_only"]
    assert rec["meta"]["categories"] == [None, "subject_substitution", None]  # recheck's category
    assert rec["meta"]["facts"] == ["S0 -> T0: fact 0", "S1 -> T1: fact 1", "S3 -> T3: fact 3"]
    assert rec["meta"]["current"] == CUR
    target = json.loads(rec["messages"][2]["content"])
    assert target == {"labels": [{"n": 1, "verdict": "stated", "quote": Q},
                                 {"n": 2, "verdict": "unsupported",
                                  "category": "subject_substitution"},
                                 {"n": 3, "verdict": "previous_only"}]}
    user = rec["messages"][1]["content"]
    assert "1. S0 -> T0: fact 0\n2. S1 -> T1: fact 1\n3. S3 -> T3: fact 3" in user
    assert "CURRENT MESSAGE:\n" + CUR in user and "DOCUMENT CONTEXT: This chunk comes from X." in user
    assert rec["messages"][0] == {"role": "system", "content": vd.VERIFIER_INSTRUCTIONS}
    assert why == {"stated": 1, "confirmed": 1, "previous_only": 1, "disputed": 1,
                   "stated_unverified": 1}


def test_build_record_falls_back_to_the_first_label_category() -> None:
    rec, _ = vd.build_record(_rec(1), [_lab("unsupported", category="taxonomy_as_fact")],
                             {0: {"verdict": "unsupported", "category": None}})
    assert rec is not None and rec["meta"]["categories"] == ["taxonomy_as_fact"]


def test_build_record_skips_count_mismatch_and_empty() -> None:
    rec, why = vd.build_record(_rec(3), [_lab("stated")] * 2, {})
    assert rec is None and why == {"label_count_mismatch": 1}
    rec, why = vd.build_record(_rec(1), [_lab("unsupported")], {0: {"verdict": "supported"}})
    assert rec is None and why["no_facts_left"] == 1


def test_split_name_keeps_dpo4_test_and_draws_dev_from_train_only() -> None:
    groups = [f"ctx|Article {i}" for i in range(400)]
    names = {g: vd.split_name(g, 0.2, 0.05, "s") for g in groups}
    for g, name in names.items():
        assert (name == "test") == (split_of(g, 0.2) == "val")
    assert 0 < sum(n == "dev" for n in names.values()) < 0.15 * len(groups)
    assert names == {g: vd.split_name(g, 0.2, 0.05, "s") for g in groups}   # seeded


def _write_store(path: Path, kind: str, answers: dict[str, object]) -> None:
    from scripts.distill.dpo_labels import task_key
    from scripts.distill.teacher import AnswerStore
    store = AnswerStore(path)
    for cid, ans in answers.items():
        store.put(task_key(kind, cid), json.dumps(ans), {"kind": kind, "call_id": cid})


def test_main_writes_three_splits_and_counts_unlabelled(tmp_path: Path, monkeypatch) -> None:
    recs = [_rec(2, f"c{i}", f"ctx|Article {i}") for i in range(60)]
    monkeypatch.setattr(vd, "read_records", lambda paths: recs)
    labels = {r["id"]: [_lab("stated"), _lab("unsupported")] for r in recs[:-1]}   # last unlabelled
    _write_store(tmp_path / "lab", "label", labels)
    _write_store(tmp_path / "re", "recheck",
                 {cid: {"1": {"verdict": "unsupported", "category": "other", "why": "w"}}
                  for cid in labels})
    out = tmp_path / "out"
    rc = vd.main(["--capture", str(tmp_path / "cap.jsonl"), "--labels", str(tmp_path / "lab"),
                  "--recheck", str(tmp_path / "re"), "--out", str(out)])
    assert rc == 0
    rows = {s: [json.loads(x) for x in (out / f"{s}.jsonl").read_text().splitlines()]
            for s in ("train", "val", "test")}
    assert sum(len(v) for v in rows.values()) == 59
    summary = json.loads((out / "summary.json").read_text())
    assert summary["skipped"]["unlabelled"] == 1
    assert summary["records"] == {"train": len(rows["train"]), "dev": len(rows["val"]),
                                  "test": len(rows["test"])}
    assert all(r["prompt_name"] == "verify_edges" for v in rows.values() for r in v)
