import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "judge", Path(__file__).parents[2] / "scripts" / "chunk_ab_judge.py")
j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(j)

ARMS = ["today", "pack1200", "pack1200_coverage"]


def test_labels_are_a_deterministic_shuffle():
    a = j.labels_for("art-1", ARMS)
    assert sorted(a.values()) == ["A", "B", "C"] and a == j.labels_for("art-1", ARMS)


def test_prompt_numbers_every_fact_under_its_label():
    p = j.build_prompt("SOURCE TEXT", {"A": ["f1", "f2"], "B": ["g1"]})
    assert "SOURCE TEXT" in p and "A1: f1" in p and "A2: f2" in p and "B1: g1" in p


def test_validate_accepts_a_complete_judgement():
    obj = {"ideas": [{"id": 1, "text": "x"}, {"id": 2, "text": "y"}],
           "labels": {"A": [{"idea": 1, "supported": True}, {"idea": 2, "supported": False}],
                      "B": [{"idea": 1, "supported": True}]}}
    assert j.validate(obj, {"A": 2, "B": 1}) is None


def test_validate_rejects_wrong_counts_unknown_ideas_and_bad_types():
    base = {"ideas": [{"id": 1, "text": "x"}],
            "labels": {"A": [{"idea": 1, "supported": True}], "B": []}}
    assert j.validate(base, {"A": 1, "B": 1}) is not None          # B missing a label
    bad_id = {"ideas": [{"id": 1, "text": "x"}],
              "labels": {"A": [{"idea": 9, "supported": True}]}}
    assert j.validate(bad_id, {"A": 1}) is not None                  # unknown idea
    bad_type = {"ideas": [{"id": 1, "text": "x"}],
                "labels": {"A": [{"idea": 1, "supported": "yes"}]}}
    assert j.validate(bad_type, {"A": 1}) is not None                # non-bool
    assert j.validate("not a dict", {"A": 1}) is not None


def test_score_counts_distinct_supported_ideas():
    labels = [{"idea": 1, "supported": True}, {"idea": 1, "supported": True},
              {"idea": 2, "supported": True}, {"idea": 3, "supported": False}]
    assert j.score(labels) == {"facts": 4, "supported": 3, "distinct": 2,
                               "redundant": 1, "unsupported": 1}


def test_decide_applies_the_spec_rule():
    rows = [{"today": {"distinct": 10, "unsupported": 1}, "pack1200_coverage":
             {"distinct": 10, "unsupported": 1}, "cost_ratio": 0.6}] * 3
    assert j.decide(rows)["adopt"] is True
    rows2 = [{"today": {"distinct": 10, "unsupported": 1}, "pack1200_coverage":
              {"distinct": 8, "unsupported": 1}, "cost_ratio": 0.6}] * 3
    assert j.decide(rows2)["adopt"] is False


def test_decide_rejects_when_coverage_has_more_unsupported():
    rows = [{"today": {"distinct": 10, "unsupported": 1}, "pack1200_coverage":
             {"distinct": 10, "unsupported": 3}, "cost_ratio": 0.6}] * 3
    assert j.decide(rows)["adopt"] is False


def test_judgeable_partitions_articles():
    rows = {
        "today": {
            "art-1": {"facts": 5}, "art-2": {"facts": 200}, "art-3": {"facts": 10}, "art-4": {"facts": 8}
        },
        "pack1200": {
            "art-1": {"facts": 5}, "art-2": {"facts": 200}, "art-4": {"facts": 8}
        },
        "pack1200_coverage": {
            "art-1": {"facts": 5}, "art-2": {"facts": 200}, "art-3": {"facts": 10}, "art-4": {"facts": 8}
        }
    }
    to_judge, excluded, missing = j.judgeable(rows, ARMS, 150)
    assert to_judge == ["art-1", "art-4"]
    assert excluded == ["art-2"]
    assert missing == ["art-3"]
