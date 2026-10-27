"""verifier_eval: parse predictions, apply drop policies P1/P2, metrics against Opus's
labels, the pre-registered bar, and the lexical baseline (spec 2026-10-27 sections 5-6)."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.distill import verifier_eval as ve

CUR = "Veeam Backup supports Amazon S3 as a backup\nrepository. Restores go to the original VM."
Q = "supports Amazon S3 as a backup repository"


def _rec(cid: str, final: list[str], cats: list | None = None) -> dict:
    return {"meta": {"call_id": cid, "final": final,
                     "categories": cats or [None] * len(final),
                     "facts": [f"S{i} -> T{i}: fact {i}" for i in range(len(final))],
                     "current": CUR}}


def _reply(*labels: dict) -> str:
    return json.dumps({"labels": [{"n": n, **lab} for n, lab in enumerate(labels, 1)]})


ST = {"verdict": "stated", "quote": Q}
UN = {"verdict": "unsupported", "category": "over_inference"}
PO = {"verdict": "previous_only"}


def test_parse_reply_accepts_a_good_reply_in_any_entry_order() -> None:
    text = json.dumps({"labels": [{"n": 2, **UN}, {"n": 1, **ST}]})
    got = ve.parse_reply(text, 2)
    assert got is not None and [g["verdict"] for g in got] == ["stated", "unsupported"]


def test_parse_reply_rejects_fences_prose_and_empty() -> None:
    good = _reply(ST)
    for bad in ("```json\n" + good + "\n```", good + " Done.", "", "   ", "null"):
        assert ve.parse_reply(bad, 1) is None, bad


def test_parse_reply_rejects_bad_numbering_and_verdicts() -> None:
    assert ve.parse_reply(json.dumps({"labels": [{"n": 1, **ST}, {"n": 1, **ST}]}), 2) is None
    assert ve.parse_reply(json.dumps({"labels": [{"n": 1, **ST}, {"n": 3, **ST}]}), 2) is None
    assert ve.parse_reply(_reply(ST), 2) is None                          # wrong count
    assert ve.parse_reply(_reply({"verdict": "maybe"}), 1) is None
    assert ve.parse_reply(json.dumps({"labels": [{"n": "1", **ST}]}), 1) is None


def test_unsupported_without_category_still_drops() -> None:
    parsed = ve.parse_reply(_reply({"verdict": "unsupported"}), 1)
    assert ve.drops(parsed, CUR, "P2", 1) == [True]


def test_p1_quote_whitespace_and_paraphrase() -> None:
    ok = ve.parse_reply(_reply(ST), 1)
    para = ve.parse_reply(_reply({"verdict": "stated", "quote": "S3 can be a repository"}), 1)
    assert ve.drops(ok, CUR, "P1", 1) == [False]       # nbsp + newline normalise to spaces
    assert ve.drops(para, CUR, "P1", 1) == [True]
    assert ve.drops(para, CUR, "P2", 1) == [False]
    assert ve.drops(None, CUR, "P1", 3) == [False, False, False]   # fail open


def test_score_metrics_and_bar() -> None:
    recs = [_rec("a", ["stated", "confirmed", "previous_only"],
                 [None, "wrong_detail", None]),
            _rec("b", ["stated", "confirmed"], [None, "over_inference"])]
    replies = {"a": _reply(ST, UN, PO), "b": _reply(ST, ST)}
    m = ve.score(recs, replies, "P1", resamples=200)
    assert m["catch"] == 0.5 and m["false_drop"] == 0.0
    assert m["counts"]["confirmed"] == 2 and m["counts"]["stated"] == 2
    assert m["previous_only_dropped"] == 0.0 and m["parse_fail"] == 0.0
    assert m["catch_by_category"] == {"over_inference": 0.0, "wrong_detail": 1.0}
    lo, hi = m["catch_ci"]
    assert 0.0 <= lo <= m["catch"] <= hi <= 1.0
    ok, why = ve.passes(m)
    assert ok and why == []
    ok, why = ve.passes({**m, "false_drop": 0.03, "parse_fail": 0.02, "catch": 0.4})
    assert not ok and len(why) == 3


def test_score_missing_and_duplicate_predictions(tmp_path: Path) -> None:
    p = tmp_path / "pred.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in (
        {"call_id": "a", "reply": _reply(UN), "step": 50},
        {"call_id": "a", "reply": _reply(ST), "step": 50})) + "\n")
    replies, dupes = ve.read_predictions(p)
    assert replies == {"a": _reply(UN)} and dupes == 1          # first wins
    recs = [_rec("a", ["confirmed"]), _rec("b", ["confirmed"])]
    m = ve.score(recs, replies, "P1", resamples=50)
    assert m["parse_fail"] == 0.5 and m["counts"]["missing"] == 1 and m["catch"] == 0.5


def test_score_is_seeded() -> None:
    recs = [_rec(str(i), ["stated", "confirmed"]) for i in range(30)]
    replies = {str(i): _reply(ST, UN if i % 2 else ST) for i in range(30)}
    assert ve.score(recs, replies, "P1", resamples=300) == ve.score(recs, replies, "P1", resamples=300)


def test_lexical_baseline_and_threshold() -> None:
    rec = {"meta": {"call_id": "x", "final": ["stated", "confirmed"], "categories": [None, None],
                    "facts": ["Veeam -> S3: Veeam supports Amazon S3 as a repository",
                              "Veeam -> Tape: Veeam encrypts tape with AES-512 hardware"],
                    "current": "Veeam Backup supports Amazon S3 as a backup repository."}}
    assert ve.lexical_drops(rec, 0.5) == [False, True]
    assert 0.0 <= ve.tune_threshold([rec]) <= 1.0


def test_main_writes_report_and_exit_code(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    recs = [{"messages": [], **_rec(str(i), ["stated", "confirmed"])} for i in range(20)]
    for name in ("train.jsonl", "test.jsonl"):
        (data / name).write_text("".join(json.dumps(r) + "\n" for r in recs))
    pred = tmp_path / "pred.jsonl"
    pred.write_text("".join(json.dumps({"call_id": str(i), "reply": _reply(ST, UN), "step": 25}) + "\n"
                            for i in range(20)))
    out = tmp_path / "report.json"
    rc = ve.main(["--data", str(data), "--predictions", str(pred), "--out", str(out),
                  "--resamples", "100"])
    report = json.loads(out.read_text())
    assert rc == 0 and report["passed"] and report["policy"] == "P1"
    assert set(report["policies"]) == {"P1", "P2"} and "baseline" in report
