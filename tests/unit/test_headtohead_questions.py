from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts.headtohead import questions as Q

CORPUS = {f"a{i}" for i in range(300)}
CHANGED = {"c1", "c2"}
ALL_IDS = CORPUS | CHANGED
VENDORS = {"Veeam", "Microsoft", "AWS", "Cohesity"}


def _make(qid: str, qtype: str, origin: str = "new") -> dict:
    q: dict = {
        "id": qid,
        "type": qtype,
        "question": f"How does {qid} work?",
        "gold_facts": ["fact one", "fact two"],
        "gold_sources": [{"article_id": "a1", "section": "Limits"}],
        "vendors": ["Veeam"],
        "written_from": ["a1"],
        "decides": origin == "new",
        "origin": origin,
    }
    if qtype == "multi_article":
        q["gold_sources"].append({"article_id": "a2"})
        q["written_from"] = ["a1", "a2"]
    if qtype == "cross_vendor":
        q["vendors"] = ["Veeam", "Microsoft"]
        q["gold_facts"] = ["Veeam: documented: x", "Microsoft: no evidence in the corpus"]
    if qtype == "unanswerable":
        q["gold_facts"] = []
        q["gold_sources"] = []
        q["written_from"] = []
        q["search_check"] = "searched 'foo bar': nothing relevant"
    return q


def good() -> list[dict]:
    out = []
    for t, n in Q.TARGET_COUNTS.items():
        out += [_make(f"{t}-{i}", t) for i in range(n)]
    out.append(_make("g1", "single_vendor", "golden_questions"))
    out.append(_make("r1", "single_vendor", "router_golden_tier1"))
    return out


def check(qs: list[dict]) -> list[str]:
    return Q.validate(qs, ALL_IDS - CHANGED, CHANGED, corpus_vendors=VENDORS)


def by_id(qs: list[dict], qid: str) -> dict:
    return next(q for q in qs if q["id"] == qid)


def test_passing_fixture_is_valid() -> None:
    assert len(good()) == 102
    assert check(good()) == []


def test_duplicate_id_is_rejected() -> None:
    qs = good()
    qs[1]["id"] = qs[0]["id"]
    assert any("duplicate" in p for p in check(qs))


def test_unknown_type_is_rejected() -> None:
    qs = good()
    by_id(qs, "g1")["type"] = "nope"
    assert any("g1" in p and "type" in p for p in check(qs))


def test_unknown_origin_is_rejected() -> None:
    qs = good()
    by_id(qs, "g1")["origin"] = "elsewhere"
    assert any("g1" in p and "origin" in p for p in check(qs))


def test_missing_key_is_reported() -> None:
    qs = good()
    del by_id(qs, "g1")["vendors"]
    assert any("g1" in p and "vendors" in p for p in check(qs))


def test_more_than_eight_gold_facts_is_rejected() -> None:
    qs = good()
    by_id(qs, "g1")["gold_facts"] = [f"f{i}" for i in range(9)]
    assert any("g1" in p and "gold_facts" in p for p in check(qs))


def test_zero_gold_facts_is_rejected_for_answerable() -> None:
    qs = good()
    by_id(qs, "g1")["gold_facts"] = []
    assert any("g1" in p and "gold_facts" in p for p in check(qs))


def test_unanswerable_needs_empty_gold_and_search_check() -> None:
    qs = good()
    u = by_id(qs, "unanswerable-0")
    u["gold_facts"] = ["x"]
    assert any("unanswerable-0" in p for p in check(qs))
    qs = good()
    by_id(qs, "unanswerable-0")["search_check"] = ""
    assert any("search_check" in p for p in check(qs))
    qs = good()
    del by_id(qs, "unanswerable-0")["search_check"]
    assert any("search_check" in p for p in check(qs))


def test_changed_article_cannot_be_a_gold_source() -> None:
    qs = good()
    by_id(qs, "g1")["gold_sources"] = [{"article_id": "c1"}]
    assert any("g1" in p and "c1" in p for p in check(qs))


def test_changed_article_cannot_be_written_from() -> None:
    qs = good()
    by_id(qs, "g1")["written_from"] = ["c2"]
    assert any("g1" in p and "c2" in p for p in check(qs))


def test_article_outside_corpus_is_rejected() -> None:
    qs = good()
    by_id(qs, "g1")["gold_sources"] = [{"article_id": "zzz"}]
    assert any("g1" in p and "zzz" in p for p in check(qs))


def test_existing_questions_cannot_decide() -> None:
    qs = good()
    by_id(qs, "g1")["decides"] = True
    assert any("g1" in p and "decides" in p for p in check(qs))


def test_new_questions_must_decide() -> None:
    qs = good()
    by_id(qs, "single_vendor-0")["decides"] = False
    assert any("single_vendor-0" in p and "decides" in p for p in check(qs))


def test_wrong_type_counts_are_rejected() -> None:
    qs = [q for q in good() if q["id"] != "single_vendor-0"]
    assert any("single_vendor" in p and "49" in p for p in check(qs))


def test_existing_questions_do_not_count_toward_targets() -> None:
    qs = good() + [_make(f"extra{i}", "cross_vendor", "golden_questions") for i in range(5)]
    assert check(qs) == []


def test_cross_vendor_needs_two_vendors() -> None:
    qs = good()
    by_id(qs, "cross_vendor-0")["vendors"] = ["Veeam"]
    assert any("cross_vendor-0" in p and "2 vendors" in p for p in check(qs))


def test_cross_vendor_gold_must_cover_each_vendor() -> None:
    qs = good()
    by_id(qs, "cross_vendor-0")["gold_facts"] = ["Veeam: documented: x"]
    problems = check(qs)
    assert any("cross_vendor-0" in p and "Microsoft" in p for p in problems)


def test_vendor_name_match_is_whole_word_and_case_insensitive() -> None:
    qs = good()
    q = by_id(qs, "cross_vendor-0")
    q["vendors"] = ["Veeam", "AWS"]
    q["gold_facts"] = ["veeam: ok", "AWSome: no"]
    assert any("AWS" in p for p in check(qs))
    q["vendors"] = ["Veeam", "N-able (Cove)"]
    q["gold_facts"] = ["VEEAM: ok", "n-able (cove): no evidence"]
    assert Q.validate(qs, ALL_IDS - CHANGED, CHANGED,
                      corpus_vendors=VENDORS | {"N-able (Cove)"}) == []


def test_every_problem_is_reported() -> None:
    qs = good()
    by_id(qs, "g1")["type"] = "nope"
    by_id(qs, "r1")["decides"] = True
    assert len(check(qs)) >= 2


def test_load_questions_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    qs = good()
    qfile = tmp_path / "q.yaml"
    qfile.write_text(yaml.safe_dump(qs), encoding="utf-8")
    assert Q.load_questions(qfile) == qs
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([{"article_id": i, "vendor": v} for i, v in
                                  zip(sorted(CORPUS), sorted(VENDORS) * 100, strict=False)]))
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"changed_since_graph": ["c1: hash differs", "c2"]}))
    args = ["--check", str(qfile), "--corpus", str(corpus), "--ingest-report", str(report)]
    assert Q.main(args) == 0

    bad = copy.deepcopy(qs)
    bad[0]["gold_sources"] = [{"article_id": "c1"}]
    qfile.write_text(yaml.safe_dump(bad), encoding="utf-8")
    assert Q.main(args) == 1
    assert "c1" in capsys.readouterr().out


def test_cli_missing_ingest_report_exits_2(tmp_path: Path) -> None:
    qfile = tmp_path / "q.yaml"
    qfile.write_text("[]")
    corpus = tmp_path / "corpus.json"
    corpus.write_text("[]")
    args = ["--check", str(qfile), "--corpus", str(corpus),
            "--ingest-report", str(tmp_path / "missing.json")]
    assert Q.main(args) == 2



def test_answerable_needs_gold_sources_and_written_from() -> None:
    qs = good()
    by_id(qs, "single_vendor-0")["gold_sources"] = []
    by_id(qs, "multi_article-0")["written_from"] = []
    problems = check(qs)
    assert any("single_vendor-0" in p and "gold_sources" in p for p in problems)
    assert any("multi_article-0" in p and "written_from" in p for p in problems)


def test_integer_article_ids_are_coerced_like_written_from() -> None:
    corpus = {"101", "102"}
    qs = good()
    for q in qs:
        if q["type"] != "unanswerable":
            q["gold_sources"] = [{"article_id": 101}]
            q["written_from"] = [101]
    assert Q.validate(qs, corpus, set(), corpus_vendors=VENDORS) == []
    by_id(qs, "g1")["gold_sources"] = [{"article_id": 999}]
    by_id(qs, "r1")["gold_sources"] = [{"article_id": True}]
    problems = Q.validate(qs, corpus, set(), corpus_vendors=VENDORS)
    assert any("g1" in p and "999" in p for p in problems)
    assert any("r1" in p and "article_id" in p for p in problems)


def test_non_mapping_items_are_problems_not_tracebacks() -> None:
    qs = [*good(), "just a string", ["a", "list"]]
    problems = check(qs)
    assert sum("not a mapping" in p for p in problems) == 2


def test_vendors_must_be_corpus_vendors() -> None:
    qs = good()
    by_id(qs, "single_vendor-0")["vendors"] = ["Rubrik"]
    by_id(qs, "g1")["vendors"] = "Veeam"
    problems = check(qs)
    assert any("single_vendor-0" in p and "Rubrik" in p and "corpus" in p for p in problems)
    assert any("g1" in p and "vendors must be a list" in p for p in problems)


def _cli_files(tmp_path: Path, questions_text: str, report: object) -> list[str]:
    qfile = tmp_path / "q.yaml"
    qfile.write_text(questions_text, encoding="utf-8")
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([{"article_id": "a1", "vendor": "Veeam"}]))
    rep = tmp_path / "report.json"
    rep.write_text(json.dumps(report))
    return ["--check", str(qfile), "--corpus", str(corpus), "--ingest-report", str(rep)]


def test_cli_non_list_yaml_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _cli_files(tmp_path, "id: q1\ntype: single_vendor\n", {"changed_since_graph": []})
    assert Q.main(args) == 2
    assert "list" in capsys.readouterr().err


def test_cli_unparseable_yaml_exits_2(tmp_path: Path) -> None:
    args = _cli_files(tmp_path, "- id: [unclosed\n", {"changed_since_graph": []})
    assert Q.main(args) == 2


def test_cli_report_without_changed_key_exits_2(tmp_path: Path,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    args = _cli_files(tmp_path, "[]", {"failed": []})
    assert Q.main(args) == 2
    assert "changed_since_graph" in capsys.readouterr().err
